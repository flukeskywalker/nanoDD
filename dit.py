# A heavily modified version of https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/tree/main/model (MIT)
import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch import nn
from torch.amp import autocast


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    @autocast("cuda", enabled=False)
    def forward(self, x: Tensor, seq_dim: int = 1) -> tuple[Tensor, Tensor]:
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)

            # This makes the transformation on v an identity.
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)

        return self.cos_cached, self.sin_cached


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@autocast("cuda", enabled=False)
def apply_rotary_pos_emb(qkv: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return (qkv * cos) + (rotate_half(qkv) * sin)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, n_embed, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, n_embed, bias=True),
            nn.SiLU(),
            nn.Linear(n_embed, n_embed, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: float = 10000) -> Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D (N,) or 2-D (N, L) Tensor with time indices. These may be fractional.
        :param dim: the dimension of the output D.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings if input is (N,) and (N, L, D) if input is (N, L)
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
        if t.dim() == 2:  # different time step for each batch item and variable
            args = t[:, :, None] * freqs[None, None, :]
        else:  # t.dim() == 1, different time step for each batch item
            args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, n_cond, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(n_cond, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: Tensor, rotary_cos_sin: Tensor, c: Tensor) -> Tensor:

        modulation = self.adaLN_modulation(c)
        # if c.dim() == 2, add a dummy dim since all variables are modulated the same way, otherwise c.dim() == 3
        modulation = modulation[:, None] if c.dim() == 2 else modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)

        # attention operation
        x_skip = x
        x = modulate(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads)
        cos, sin = rotary_cos_sin
        qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

        q, k, v = rearrange(
            qkv, "b s three h d -> three b h s d", three=3, h=self.n_heads, d=self.dim // self.n_heads
        ).unbind(0)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0, is_causal=False)
        x = rearrange(x, "b h s d -> b s (h d)", h=self.n_heads, d=self.dim // self.n_heads)

        x = self.attn_out(x)
        x = gate_msa * F.dropout(x, p=self.dropout, training=self.training) + x_skip

        # mlp operation
        x_skip = x
        x = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = gate_mlp * F.dropout(x, p=self.dropout, training=self.training) + x_skip
        return x


class DiT(nn.Module):
    def __init__(self, vocab_size, n_embed=768, n_heads=12, n_blocks=24, n_cond=128, dropout=0.1):
        super().__init__()

        self.vocab_embed = nn.Embedding(vocab_size, n_embed)
        init_std = 1 / math.sqrt(n_embed)
        torch.nn.init.trunc_normal_(self.vocab_embed.weight, 0, init_std, a=-3 * init_std, b=3 * init_std)

        self.c_embed = TimestepEmbedder(n_cond)
        self.rotary_emb = Rotary(n_embed // n_heads)
        self.blocks = nn.ModuleList([DDiTBlock(n_embed, n_heads, n_cond, dropout=dropout) for _ in range(n_blocks)])

        # build output modulation; we will reuse input embedding weights for projection
        self.norm_final = LayerNorm(n_embed)
        self.adaLN_modulation = nn.Linear(n_cond, 2 * n_embed, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

        # report number of parameters
        print(f"Number of parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6:.2f}M")

    @autocast("cuda", dtype=torch.bfloat16)
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.vocab_embed(x)
        c = F.silu(self.c_embed(t))

        rotary_cos_sin = self.rotary_emb(x)

        for i in range(len(self.blocks)):
            x = self.blocks[i](x, rotary_cos_sin, c)

        modulation = self.adaLN_modulation(c)
        modulation = modulation[:, None] if c.dim() == 2 else modulation
        shift, scale = modulation.chunk(2, dim=-1)

        x = modulate(self.norm_final(x), shift, scale)
        x = F.linear(x, self.vocab_embed.weight)

        return x.float()


if __name__ == "__main__":
    batch_size, seq_len, vocab_size = 3, 7, 27
    dit = DiT(vocab_size=vocab_size, n_embed=768, n_heads=12, n_blocks=1, n_cond=128, dropout=0.1)
    x = torch.randn(batch_size, seq_len, vocab_size)
    t = torch.rand(batch_size)
    print(dit(x, t).shape)
