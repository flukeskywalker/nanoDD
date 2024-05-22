import math
from pathlib import Path

from ema_pytorch import EMA

from d3pm import D3PM
import torch
import numpy as np

seed = 1
split = "test"  # train, val or test
checkpoint_dir = Path("./checkpoints/d3pm_text8_D6") # specify dir containing ckpt.pt to load here
data_dir = Path("text8") # dir containing {split}.npy to evaluate on
load_ema = True # load EMA weights if True
seq_len = 256
batch_size = 512
n_batches_kl = 10000 # num of batches to evaluate the n-step or continuous time KL randomly
n_batches_recon = 100 # num of batches to evaluate the final cross-entropy or "reconstruction loss"
log_interval = 50 # print current estimate periodically
device = "cuda"

torch.manual_seed(seed)
assert split in ["train", "val", "test"], f"split must be train, val or test, but was {split}"
data = np.load(data_dir / f"{split}.npy")
print(f"Loaded {split}.npy")

ckpt = torch.load(checkpoint_dir / "ckpt.pt", weights_only=True)
model_cls = dict(d3pm=D3PM)[ckpt["model_cls"].lower()]
model_args = ckpt["model_args"]
model = model_cls(**model_args)

if load_ema:
    ema = EMA(model, include_online_model=False)
    ema.load_state_dict(ckpt["ema"])
    eval_model = ema
else:
    model.load_state_dict(ckpt["model"])
    eval_model = model

eval_model.to(device)
eval_model.eval()
print(f"Loaded model with load_ema={load_ema}")

def get_batch() -> torch.Tensor:
    """Returns a random batch of data"""
    ix = torch.randint(data.shape[0] - seq_len, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i: i + seq_len].astype(np.int64)) for i in ix])
    if device == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
    return x

def estimate_kl() -> float:
    """Returns the batch mean of n-step or inf-step loss on a random batch of data & time samples"""
    t = torch.randint(1, model.T, (x.size(0),), device=device)
    _, _, metrics = eval_model(x, t)
    return metrics["kl"].item()

def estimate_recon() -> float:
    """Returns the batch mean of reconstruction or cross-entropy loss on a random data batch at t=0 (for D3PM)"""
    t = torch.zeros(x.size(0), device=device).long()
    _, _, metrics = eval_model(x, t)
    return metrics["ce"].item()

def mean_bits(loss_list: list[float]) -> float:
    return np.array(loss_list).mean() / math.log(2)

def std_bits(loss_list: list[float]) -> float:
    return np.array(loss_list).std(ddof=1) / math.log(2)

def estimate_total_loss() -> dict:
    return {
        "kl_mean": mean_bits(kl_list),
        "recon_mean": mean_bits(recon_list),
        "kl_std": std_bits(kl_list),
        "recon_std": std_bits(recon_list),
        "bits_per_token_mean": (mean_bits(kl_list) * (model.T - 1)) + mean_bits(recon_list),
    }


kl_list = []
recon_list = []
with torch.no_grad():
    print(f"Estimating KL with {n_batches_kl} batches of size {batch_size}")
    for i in range(n_batches_kl):
        x = get_batch()
        kl = estimate_kl()
        kl_list.append(kl)
        if (i + 1) % log_interval == 0:
            print(f"batch {i + 1:5d}/{n_batches_kl}, kl_mean {mean_bits(kl_list) * (model.T - 1)} bits per token")

    print(f"Estimating recon loss with {n_batches_recon} batches of size {batch_size}")
    for i in range(n_batches_recon):
        x = get_batch()
        recon = estimate_recon()
        recon_list.append(recon)
        if (i + 1) % log_interval == 0:
            print(f"batch {i + 1:5d}/{n_batches_recon}, recon_mean {mean_bits(recon_list)} bits per token")

evals = estimate_total_loss()
print("Final estimated evals:\n", evals)
