import shutil
from typing import Optional

import fire
import torch
import torch.distributions as D
import torch.nn.functional as F
from torch import Tensor

from d3pm_absorbing import D3PMAbsorbing


class Text8SampleDecoder:
    TEXT8_CHARS = list("_abcdefghijklmnopqrstuvwxyz█")

    @staticmethod
    def char_ids_to_str(char_ids: Tensor) -> str:
        """Decode a 1D sequence of character IDs to a string."""
        return "".join([Text8SampleDecoder.TEXT8_CHARS[i] for i in char_ids.squeeze(-1)])

    @classmethod
    def batch_to_str(cls, batch: Tensor) -> list[str]:
        """Decode a batch of character IDs to a list of strings."""
        return [cls.char_ids_to_str(row_char_ids) for row_char_ids in batch]


class OwtSampleDecoder:
    import tiktoken

    enc_base = tiktoken.get_encoding("gpt2")
    enc = tiktoken.Encoding(
        "gpt2mask",
        pat_str=enc_base._pat_str,
        mergeable_ranks=enc_base._mergeable_ranks,
        special_tokens=enc_base._special_tokens | {"█": 50257},
    )

    @classmethod
    def batch_to_str(cls, batch: Tensor) -> list[str]:
        return cls.enc.decode_batch(batch.tolist())


def decode_batch(batch: Tensor, dataset: str):
    if dataset.lower() == "text8":
        return Text8SampleDecoder.batch_to_str(batch)
    elif dataset.lower() == "openwebtext":
        return OwtSampleDecoder.batch_to_str(batch)
    else:
        raise NotImplementedError(f"No decoder for {dataset}!")


def visualize_batch(batch: Tensor, dataset: str, truncate: bool = True, return_to_top: bool = False):

    decoded = decode_batch(batch, dataset)
    term_width = shutil.get_terminal_size().columns
    truncate = truncate and term_width < len(decoded[0])

    for i in range(len(decoded)):
        if truncate:
            # for each sample, print only what fits on one terminal line
            print(repr(decoded[i])[1: term_width - 5] + "...", flush=True)
        else:
            print(" - " * (term_width // 3))
            print(decoded[i], flush=True)

    if return_to_top:
        # Use ANSI escape code for moving up lines so next output will over-write previous lines
        print("\033[F" * len(decoded), end="", flush=True)
        import sys
        sys.stdout.flush()


@torch.no_grad()
def sample_d3pm_absorbing(
    model: D3PMAbsorbing, batch_size: int, seq_len: int, visualize: bool, dataset: Optional[str] = None
) -> Tensor:

    if visualize is True:
        assert dataset is not None, "dataset can not be None if visualize is True!"
    device = next(model.parameters()).device

    # sample from stationary distribution (our convention says last class is the mask class)
    x_T = torch.ones(batch_size, seq_len, device=device).long() * (model.K - 1)

    # init
    trajectory = [x_T]
    time = torch.ones(batch_size).long().to(device)
    x_t = x_T

    # iteratively sample from the log posterior
    for _t in range(model.T, 0, -1):
        if visualize:
            visualize_batch(x_t, dataset=dataset, return_to_top=_t != 1)

        t = time * _t
        log_predicted_x_0 = model.net(x_t, t.float())
        p_x_0 = F.softmax(log_predicted_x_0, dim=-1)

        if _t > 1:
            log_p_x_tminus1, _ = model.compute_unnormalized_log_posterior(p_x_0, t - 1, x_t)
            x_tminus1 = D.Categorical(logits=log_p_x_tminus1).sample()
            x_t = x_tminus1
            trajectory.append(x_tminus1)

    # now _t is 1, and we have the final predicted p(x_0)
    trajectory.append(D.Categorical(logits=log_predicted_x_0).sample())
    if visualize:
        print("\nFinal sample(s):")
        visualize_batch(trajectory[-1], dataset=dataset, truncate=False)

    # return the full trajectory of T steps, plus the final sample from p(x_0)
    return torch.stack(trajectory, dim=0)


def main(
    ckpt_path: str,  # full path to ckpt.pt
    load_ema: bool = True,
    batch_size: int = 1,
    dataset: str = "text8",
    seed: int = 123,
):

    torch.manual_seed(seed)

    assert dataset in ["text8", "openwebtext"]
    if dataset == "text8":
        seq_len = 256
    else:
        seq_len = 1024

    ckpt = torch.load(ckpt_path, weights_only=True)
    model_cls = dict(D3PMAbsorbing=D3PMAbsorbing)[ckpt["model_cls"]]
    model_args = ckpt["model_args"]

    model = model_cls(**model_args)
    if load_ema:
        state_dict = {k[len("ema_model."):]: v for k, v in ckpt["ema"].items() if k.startswith("ema_model.")}
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(ckpt["model"])
    model.to("cuda").eval()

    print(f"Sampling with {model.T} steps ...")
    sample_d3pm_absorbing(model, batch_size, seq_len, visualize=True, dataset=dataset)


if __name__ == "__main__":
    fire.Fire(main)
