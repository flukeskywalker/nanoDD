"""
Heavily modified training script derived from nanoGPT.

This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py d3pm_text8

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=8 train.py d3pm_text8_4gpu

To run with DDP on 8 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py d3pm_text8_4gpu
- Run on the worker node:
$ torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py d3pm_text8_4gpu
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import argparse
import collections
import inspect
import math
import os
import time
from collections import defaultdict
from inspect import getmembers, isfunction
from pathlib import Path

import numpy as np
import torch
from ema_pytorch import EMA
from torch.distributed import init_process_group, destroy_process_group, reduce, ReduceOp
from torch.nn.parallel import DistributedDataParallel as DDP

import configs

# -----------------------------------------------------------------------------
# load model class, model and training config

parser = argparse.ArgumentParser(description="nanoDD training script")
parser.add_argument(
    "config",
    type=str,
    choices=[k for (k, v) in getmembers(configs, isfunction)],
    help="config function name in config.py",
)
parser.add_argument('--no-compile', action='store_false', dest='compile', help='Disable torch.compile')
args = parser.parse_args()

print(f"importing model and cfg: {args.config}")
model_cls, model_args, training_args = getattr(configs, args.config)()

# -----------------------------------------------------------------------------
# default training config with overrides from model-specific values at the end
# model-specific values for these go in configs.py, which over-ride values below

log_to_stdout = True
log_to_neptune = False
neptune_project = None
training_seed = 73311337

# I/O & eval
out_dir = Path("./checkpoints")
resume_dir = None  # if not None, resume from ckpt.pt in this dir
eval_interval = 25_000
log_interval = 10
eval_iters = 125  # per GPU evaluation iters
eval_key = "l_T"  # l_T for D3PM is the approx. T-step loss
always_save_checkpoint = False  # if True, always save a checkpoint after each eval

# data
dataset = "text8"
data_root_dir = "."
gradient_accumulation_steps = 1  # used to simulate larger batch sizes
batch_size = 256  # note: this is the micro-batch size PER GPU
seq_len = 256

# adamw optimizer
learning_rate = 1e-3  # max learning rate
max_iters = 500_000  # total number of training iterations
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.98
grad_clip = 0.1  # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 5000  # how many steps to warm up for
min_lr = 1e-4  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
backend = "nccl"  # "nccl", "gloo", etc.

# system
if torch.cuda.is_available():
    assert torch.cuda.is_bf16_supported(), "bf16 not supported!"
device = "cuda" if torch.cuda.is_available() else "cpu"  # examples: "cpu", "cuda", "cuda:0", "cuda:1" etc.
compile = True if torch.cuda.is_available() and args.compile else False

# update training args with model-specific values
for k in training_args.keys():
    assert k in globals().keys(), f"training arg {k} was not recognized"
globals().update(training_args)

# -----------------------------------------------------------------------------
# useful functions for training


# poor man's data loader with support for loading simple np arrays
data_dir = Path(data_root_dir) / dataset
if dataset == "text8":
    # text8 is stored as a simple np.array
    def mmap_data(split: str) -> np.array:
        return np.load(data_dir / f"{split}.npy", mmap_mode="r")
else:
    def mmap_data(split: str) -> np.array:
        return np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r")

def get_batch(split):
    data = mmap_data(split)
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
    lr_decay_iters = max_iters

    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters

    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr

    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# optimizer with weight decay for 1D vectors set to 0.0
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
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")

    return optimizer


# distributed mean computation for val
def dist_mean(x: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(x, collections.abc.Mapping):
        return {k: dist_mean(v) for k, v in x.items() if isinstance(v, torch.Tensor)}
    if ddp:
        reduce(x, 0, op=ReduceOp.SUM)
    return x / ddp_world_size


# val metrics estimator
@torch.no_grad()
def estimate_val_loss():
    print("\nvalidating...")
    loss, metrics = 0.0, defaultdict(float)
    for k in range(eval_iters):
        X = get_batch("val")
        _, batch_loss, batch_metrics = ema(X)
        batch_loss, batch_metrics = dist_mean(batch_loss), dist_mean(batch_metrics)
        loss += batch_loss.item() / eval_iters
        for k, v in batch_metrics.items():
            metrics[k] += v.item() / eval_iters

    return loss, metrics


# -----------------------------------------------------------------------------
# start with various inits, derived values, I/O setup

ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

torch.manual_seed(training_seed + seed_offset)
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * seq_len
print(f"tokens per iteration will be: {tokens_per_iter:,}")

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
torch.backends.cudnn.benchmark = True
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast

# set up logging and saving
if log_to_neptune and master_process:
    import neptune

    run = neptune.init_run(project=neptune_project, source_files="*.py")
    run["model_cls"] = model_cls.__name__
    run["model_args"] = model_args
    run["training_args"] = training_args
    run["total_batch_size"] = batch_size * ddp_world_size * gradient_accumulation_steps
    run["DET_EXP_ID"] = os.getenv("DET_EXPERIMENT_ID")
    out_dir = out_dir / run['sys/id'].fetch()

else:
    from datetime import datetime
    out_dir = out_dir / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# -----------------------------------------------------------------------------
# init model and optimizer

if resume_dir is None:
    print("initializing a new model from scratch")
    model = model_cls(**model_args)
    model.to(device)
    ema = EMA(
        model, beta=0.9999, update_after_step=100, update_every=1, inv_gamma=1.0, power=1.0, include_online_model=False
    )
    iter_num = 0
    best_val_loss = 1e9
else:
    print(f"Resuming training from {resume_dir}")
    checkpoint = torch.load(resume_dir / "ckpt.pt", map_location=device)
    model_args = checkpoint["model_args"]
    model = model_cls(**model_args)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    ema = EMA(
        model, beta=0.9999, update_after_step=100, update_every=1, inv_gamma=1.0, power=1.0, include_online_model=False
    )
    ema.load_state_dict(checkpoint["ema"])

    training_args = checkpoint["training_args"]
    for k in training_args.keys():
        assert k in globals().keys(), f"training arg {k} was not recognized"
    globals().update(training_args)

    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

model.train()

# optimizer
optimizer = configure_optimizer(model, weight_decay, learning_rate, (beta1, beta2), device_type)
if resume_dir is not None:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# compile the model
if compile:
    print("torch.compile(model): enabled")
    model = torch.compile(
        model,
        mode=None,
        dynamic=False,
        fullgraph=False,
        backend="inductor",
        options={
            "max_autotune_gemm": True,
            "max_autotune_pointwise": False,
            "triton.cudagraphs": False,
            "triton.cudagraph_trees": False,
            "permute_fusion": True,
            "shape_padding": True,
        },
    )  # requires PyTorch 2.0

raw_model = model._orig_mod if compile else model
raw_model = raw_model.module if ddp else raw_model


# -----------------------------------------------------------------------------
# training loop

X = get_batch("train")  # fetch the very first batch
t0 = time.time()
print("starting training")

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        logits, loss, metrics = model(X)
        loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
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
    if iter_num == 0 or (iter_num + 1) % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if log_to_stdout:
            metrics_repr = " | ".join([f"{k} {v:.5f}" for k, v in metrics.items()])
            print(f"iter {iter_num + 1}: loss {lossf:.5f}, {metrics_repr}, time {dt * 1000:.2f}ms")
        if log_to_neptune:
            run["lr"].log(lr, step=iter_num + 1)
            run["metrics/train/loss"].log(lossf, step=iter_num + 1)
            for k, v in metrics.items():
                run[f"metrics/train/{k}"].log(v, step=iter_num + 1)
            param_norm = sum([p.data.norm().item() ** 2 for p in model.parameters() if p.requires_grad]) ** 0.5
            run["metrics/train/grad_norm"].log(grad_norm.item(), step=iter_num + 1)
            run["metrics/train/param_norm"].log(param_norm, step=iter_num + 1)
            run["metrics/train/step_time"].log(dt, step=iter_num + 1)

    ema.update()
    iter_num += 1  # inc true num of "completed" iterations

    # evaluate the loss on val set and write checkpoints
    if iter_num % eval_interval == 0:
        val_loss, metrics = estimate_val_loss()

    if iter_num % eval_interval == 0 and master_process:
        # print val metrics to stdout
        metrics_repr = " | ".join([f"{k} " + f"{v:.5f}" for k, v in metrics.items()])
        print(f"val @ {iter_num} updates: loss {val_loss:.4f}, {metrics_repr}\n")

        if log_to_neptune:
            run["metrics/val/loss"].log(val_loss, step=iter_num)
            for k, v in metrics.items():
                run[f"metrics/val/{k}"].log(v, step=iter_num)

        # save checkpoint
        if metrics[eval_key] < best_val_loss or always_save_checkpoint:
            best_val_loss = metrics[eval_key]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_cls": raw_model.__class__.__name__,
                    "model_args": model_args,
                    "training_args": training_args,
                    "total_batch_size": batch_size * ddp_world_size * gradient_accumulation_steps,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, out_dir / "ckpt.pt")
                print("checkpoint created")

    # termination conditions
    if iter_num >= max_iters:
        print(f"training complete at {iter_num} iterations.")
        break

if ddp:
    destroy_process_group()
