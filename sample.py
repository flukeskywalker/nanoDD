import shutil

import fire
import torch
import torch.distributions as D
import torch.nn.functional as F
from torch import Tensor

from d3pm_absorbing import D3PMAbsorbing


def visualize_batch(batch: list[str], truncate: bool = True, return_to_top: bool = False):
    term_width = shutil.get_terminal_size().columns
    truncate = truncate and term_width < len(batch[0])

    for i in range(len(batch)):
        if truncate:
            print(batch[i][: term_width - 3] + "...", flush=True)
        else:
            print(batch[i], flush=True)

    if return_to_top:
        print("\033[F" * len(batch), end="", flush=True)


class Text8SampleDecoder:
    TEXT8_CHARS = list("_abcdefghijklmnopqrstuvwxyz█")

    @staticmethod
    def char_ids_to_str(char_ids) -> str:
        """Decode a 1D sequence of character IDs to a string."""
        return "".join([Text8SampleDecoder.TEXT8_CHARS[i] for i in char_ids.squeeze(-1)])

    @classmethod
    def batch_to_str(cls, text_batch) -> list[str]:
        """Decode a batch of character IDs to a list of strings."""
        return [cls.char_ids_to_str(row_char_ids) for row_char_ids in text_batch]


@torch.no_grad()
def sample_d3pm_absorbing(model: D3PMAbsorbing, batch_size: int, seq_len: int, visualize: bool) -> Tensor:
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
            visualize_batch(Text8SampleDecoder.batch_to_str(x_t), return_to_top=_t != 1)

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
        visualize_batch(Text8SampleDecoder.batch_to_str(trajectory[-1]), truncate=False)

    # return the full trajectory of T steps, plus the final sample from p(x_0)
    return torch.stack(trajectory, dim=0)


def main(
    ckpt_path: str,  # full path to ckpt.pt
    batch_size: int = 1,
    seq_len: int = 256,
    seed: int = 123,
):

    torch.manual_seed(seed)

    ckpt = torch.load(ckpt_path, weights_only=True)
    model_cls = dict(D3PMAbsorbing=D3PMAbsorbing)[ckpt["model_cls"]]
    model_args = ckpt["model_args"]

    model = model_cls(**model_args)
    model.load_state_dict(ckpt["model"])
    model.to("cuda").eval()

    print(f"Sampling with {model.T} steps ...")
    sample_d3pm_absorbing(model, batch_size, seq_len, visualize=True)


if __name__ == "__main__":
    fire.Fire(main)
