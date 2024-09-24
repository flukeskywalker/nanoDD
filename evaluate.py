import math
from pathlib import Path

import fire
import numpy as np
import torch
from torch import Tensor

from d3pm_absorbing import D3PMAbsorbing


def get_batch(data: np.array, batch_size: int, seq_len: int, device: str) -> torch.Tensor:
    """Returns a random batch of data"""
    ix = torch.randint(data.shape[0] - seq_len, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i: i + seq_len].astype(np.int64)) for i in ix])
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
    return x


def estimate_kl(model: torch.nn.Module, x: Tensor) -> float:
    """Returns the batch mean of n-step or inf-step loss on a random batch of data & time samples"""
    t = torch.randint(1, model.T, (x.size(0),), device=x.device)
    _, _, metrics = model(x, t)
    return metrics["kl"].item()


def estimate_recon(model: torch.nn.Module, x: Tensor) -> float:
    """Returns the batch mean of reconstruction or cross-entropy loss on a random data batch at t=0 (for D3PM)"""
    t = torch.zeros(x.size(0), device=x.device).long()
    _, _, metrics = model(x, t)
    return metrics["ce"].item()


def mean_bits(loss_list: list[float]) -> float:
    return np.array(loss_list).mean() / math.log(2)


def std_bits(loss_list: list[float]) -> float:
    return np.array(loss_list).std(ddof=1) / math.log(2)


def estimate_total_loss(model: torch.nn.Module, kl_list: list[float], recon_list: list[float]) -> dict:
    return {
        "kl_mean": mean_bits(kl_list), "recon_mean": mean_bits(recon_list), "kl_std": std_bits(kl_list),
        "recon_std": std_bits(recon_list),
        "bits_per_token_mean": (mean_bits(kl_list) * (model.T - 1)) + mean_bits(recon_list),
    }


@torch.no_grad()
def main(
    checkpoint_path,  # path to ckpt.pt
    seed=1,
    split="val",  # train, val or test
    data_dir="text8",  # dir containing {split}.npy to evaluate on
    load_ema=True,  # load EMA weights if True
    seq_len=256,
    batch_size=512,
    n_batches_kl=10000,  # num of batches to evaluate the n-step or continuous time KL randomly
    n_batches_recon=100,  # num of batches to evaluate the final cross-entropy or "reconstruction loss"
    log_interval=50,  # print current estimate periodically
    device="cuda",
):

    torch.manual_seed(seed)
    assert split in ["train", "val", "test"], f"split must be train, val or test, but was {split}"
    data = np.load(Path(data_dir) / f"{split}.npy")
    print(f"Loaded {split}.npy")

    ckpt = torch.load(checkpoint_path, weights_only=True)
    model_cls = dict(D3PMAbsorbing=D3PMAbsorbing)[ckpt["model_cls"]]
    model_args = ckpt["model_args"]
    model = model_cls(**model_args)

    if load_ema:
        state_dict = {k[len("ema_model."):]: v for k, v in ckpt["ema"].items() if k.startswith("ema_model.")}
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(ckpt["model"])

    model.to(device)
    model.eval()
    print(f"Loaded model with load_ema={load_ema}")

    kl_list = []
    recon_list = []
    with torch.no_grad():

        print(f"Estimating KL with {n_batches_kl} batches of size {batch_size}")
        for i in range(n_batches_kl):
            x = get_batch(data, batch_size, seq_len, device)
            kl = estimate_kl(model, x)
            kl_list.append(kl)

            if (i + 1) % log_interval == 0:
                print(f"batch {i + 1:5d}/{n_batches_kl}, kl_mean {mean_bits(kl_list) * (model.T - 1)} bits per token")

        print(f"Estimating recon loss with {n_batches_recon} batches of size {batch_size}")
        for i in range(n_batches_recon):
            x = get_batch(data, batch_size, seq_len, device)
            recon = estimate_recon(model, x)
            recon_list.append(recon)

            if (i + 1) % log_interval == 0:
                print(f"batch {i + 1:5d}/{n_batches_recon}, recon_mean {mean_bits(recon_list)} bits per token")

    evals = estimate_total_loss(model, kl_list, recon_list)
    print("Final estimated evals:\n", evals)


if __name__ == "__main__":
    fire.Fire(main)
