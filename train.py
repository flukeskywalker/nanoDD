"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""
import inspect
import os
import time
import math
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from d3pm import D3PM

# default config values designed to train a 6 layer D3PM on text8 for 400B tokens
# -----------------------------------------------------------------------------
log_to_stdout = True
# neptune logging
log_to_neptune = False
neptune_project = ""

# I/O
out_dir = Path("./")
eval_interval = 25_000
log_interval = 10
eval_iters = 1000
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = "scratch" # "scratch" or "resume"
trn_limit = None

# data
dataset = "text8"
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 256 # if gradient_accumulation_steps > 1, this is the micro-batch size
seq_len = 256

# model
model_cls = D3PM
vocab_size = None # None means read from meta.pkl
n_embed = 768
n_heads = n_embed // 64
n_blocks = 6
n_cond = 128
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
T = 1000
lambda_ce = 0.01

# adamw optimizer
learning_rate = 1e-3 # max learning rate
max_iters = 500_000 # total number of training iterations
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.98
grad_clip = 0.1 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 5000 # how many steps to warm up for
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
min_lr = 1e-4 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
backend = "nccl" # "nccl", "gloo", etc.

# system
if torch.cuda.is_available():
    assert torch.cuda.is_bf16_supported(), "bf16 not supported!"
device = "cuda" if torch.cuda.is_available() else "cpu" # examples: "cpu", "cuda", "cuda:0", "cuda:1" etc.
compile = True # if torch.cuda.is_available() else False # use PyTorch 2.0 to compile the model to be faster

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps == 1 or gradient_accumulation_steps % ddp_world_size == 0
    # gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * seq_len
print(f"tokens per iteration will be: {tokens_per_iter:,}")

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu" # for later use in torch.autocast

# poor man's data loader
data_dir = Path(f"./{dataset}")
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == "train":
        data = np.load(data_dir / "train.npy", mmap_mode="r")
        ix = torch.randint(trn_limit or data.shape[0] - seq_len, (batch_size,))
    else:
        data = np.load(data_dir / "val.npy", mmap_mode="r")
        ix = torch.randint(data.shape[0] - seq_len, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i: i + seq_len].astype(np.int64)) for i in ix])
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
    return x

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def configure_optimizer(model, weight_decay: float, lr: float, betas, device_type):
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")
    return optimizer

# helps estimate an arbitrarily accurate val loss
@torch.no_grad()
def estimate_val_loss():
    print("\nValidating...")
    model.eval()
    loss, metrics = 0.0, defaultdict(float)
    for k in range(eval_iters):
        X = get_batch("val")
        _, batch_loss, batch_metrics = model(X)
        loss += (batch_loss.item() / eval_iters)
        for k, v in batch_metrics.items():
            metrics[k] += (v.item() / eval_iters)

    model.train()
    return loss, metrics

# init these up here, can override if init_from="resume" (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
if vocab_size is None:
    meta_path = data_dir / "meta.pkl"
    assert meta_path.exists(), f"{meta_path} does not exist!"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {vocab_size} (inside {meta_path})")
    # decoding = meta["decoding"]

# model init
model_args = dict(vocab_size=vocab_size,n_embed=n_embed, n_heads=n_heads, n_blocks=n_blocks, n_cond=n_cond,
                  dropout=dropout, T=T, lambda_ce=lambda_ce)

if init_from == "scratch":
    print("Initializing a new model from scratch")
    model = D3PM(**model_args)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    checkpoint = torch.load(out_dir / "ckpt.pt", map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    model = D3PM(**checkpoint_model_args)
    model.load_state_dict(checkpoint["model"])
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
else:
    raise ValueError(init_from)

model.to(device)
model.train()

# optimizer
optimizer = configure_optimizer(model, weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None # free up memory

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# compile the model
if compile:
    print("compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0

raw_model = model._orig_mod if compile else model
raw_model = raw_model.module if ddp else raw_model

# set up logging and saving
if log_to_neptune and master_process:
    import neptune
    run = neptune.init_run(project=neptune_project, source_files="*.py")
    run["model_cls"] = raw_model.__class__.__name__
    run["model_args"] = model_args
    out_dir = out_dir / f"{run['sys/id'].fetch()}"
else:
    out_dir = out_dir / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

if master_process:
    os.makedirs(out_dir, exist_ok=True)

# training loop
X = get_batch("train") # fetch the very first batch
t0 = time.time()

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:

        val_loss, metrics = estimate_val_loss()
        metrics_repr = " | ".join([f"{k} " + f"{v:.5f}" for k, v in metrics.items()])
        print(f"Val @ {iter_num} updates: loss {val_loss:.4f}, {metrics_repr}\n")
        if log_to_neptune:
            run["metrics/val/loss"].log(val_loss, step=iter_num)
            for k, v in metrics.items():
                run[f"metrics/val/{k}"].log(v, step=iter_num)

        if val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = val_loss
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_cls": raw_model.__class__.__name__,
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        logits, loss, metrics = model(X)
        loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X = get_batch("train")
        loss.backward()

    # clip the gradient
    if grad_clip != 0.0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    optimizer.step()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if (iter_num + 1) % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if log_to_stdout:
            metrics_repr = " | ".join([f"{k} {v.item() * gradient_accumulation_steps:.5f}"
                                       for k, v in metrics.items() if torch.is_tensor(v)])
            print(f"iter {iter_num + 1}: loss {lossf:.5f}, {metrics_repr}, time {dt*1000:.2f}ms")
        if log_to_neptune:
            run["lr"].log(lr, step=iter_num + 1)
            run["metrics/train/loss"].log(lossf, step=iter_num + 1)
            for k, v in metrics.items():
                if torch.is_tensor(v):
                    run[f"metrics/train/{k}"].log(v.item(), step=iter_num + 1)
            param_norm = sum([p.data.norm().item() ** 2 for p in model.parameters() if p.requires_grad]) ** 0.5
            run["metrics/train/grad_norm"].log(grad_norm.item(), step=iter_num + 1)
            run["metrics/train/param_norm"].log(param_norm, step=iter_num + 1)


    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
