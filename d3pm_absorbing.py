import math
from typing import Optional

import torch
import torch.distributions as D
import torch.nn.functional as F
from torch import nn, Tensor

from dit import DiT


def onehot(x: Tensor, K: int):
    return F.one_hot(x, K).float() if x.ndim == 2 else x.clone()


class D3PMAbsorbing(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        n_heads: int,
        n_blocks: int,
        n_cond: int,
        dropout: float,
        T: int,
        lambda_ce: float,
    ) -> None:
        super().__init__()

        self.T = T
        self.lambda_ce = lambda_ce
        self.K = vocab_size + 1
        self.net = DiT(self.K, n_embed, n_heads, n_blocks, n_cond, dropout)
        self.eps = 1e-20

        # betas and alpha_bars are 1-indexed, not zero-indexed, to keep indexing simpler
        # beta(0) = undef, beta(1) = 1/T, beta(2) = 1/(T-1), beta(T) = 1
        betas = torch.reciprocal(T - torch.arange(T + 1) + 1)
        betas[0] = 0.0
        alpha_bars = torch.cumprod(1.0 - betas, dim=0)
        alpha_bars[-1] = 0.0
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alpha_bars)

    def mul_Qbar(self, x: Tensor, t: Tensor) -> Tensor:
        y = onehot(x, self.K)
        alpha_bar_t = self.alpha_bars[t]
        y.mul_(alpha_bar_t[:, None, None])
        y[:, :, -1] += (1 - alpha_bar_t)[:, None]
        return y

    def mul_QT(self, x: Tensor, t: Tensor) -> Tensor:
        y = onehot(x, self.K)
        beta_t = self.betas[t][:, None, None]
        z = beta_t * y[:, :, -1:]
        y.mul_(1 - beta_t).add_(z)
        return y

    def compute_unnormalized_log_posterior(self, x_0, t, x_tplus1=None) -> tuple[Tensor, Tensor]:
        """D3PM's key method: it computes log of unnormalized posterior probs q(x_t | x_t+1, x_0).
         This method is called twice, once when x_0 is the clean data and once when x_0 is a predictive distribution
         produced by self.net(). The two outputs are then used to compute the loss.

        Args:
            x_0: class indices (BL) or predicted probabilities (BLK) at t = 0
            t: transition times LongTensor with values in [0, T-1] (B)
            x_tplus1 (optional): Sample from q(x_t+1 | x_0) if already computed in previous call to this function (BL)

        Returns:
            tuple: (unnormalized posterior log probs (BLK), x_tplus1 sample (BL))
        """

        # compute q(x_t+1 | x_0) = q_0 @ Qbar_t+1 (note that t goes from 0 to T-1)
        q_x_tplus1_given_x_0 = self.mul_Qbar(x_0, t + 1)

        if x_tplus1 is None:
            # sample x_t+1 from q(x_t+1 | x_0)
            x_tplus1 = D.Categorical(probs=q_x_tplus1_given_x_0).sample()

        # multiply x_t+1 with transpose(Q_t+1) to get q(x_t+1 | x_t) **as a function of x_t+1**
        q_x_tplus1_given_x_t = self.mul_QT(x_tplus1, t + 1)

        # compute q(x_t | q_0) = q_0 @ Qbar_t
        q_x_t_given_x_0 = self.mul_Qbar(x_0, t)

        # q(x_t | x_0) is already computed, so compute unnormalized posterior log probs
        # log[q(x_t | x_t+1, x_0)] = log[q(x_t+1 | x_t)] + log[q(x_t | x_0)]
        log_posterior = torch.log(q_x_tplus1_given_x_t + self.eps) + torch.log(q_x_t_given_x_0 + self.eps)

        # if t = 0, simply set posterior to x_0
        if x_0.ndim == 2:
            x_0 = F.one_hot(x_0, self.K).float()
        log_posterior = torch.where(t[:, None, None] == 0, torch.log(x_0 + self.eps), log_posterior)

        return log_posterior, x_tplus1

    def forward(self, data: Tensor, t: Optional[Tensor] = None) -> tuple[Tensor, Tensor, dict]:
        """Returns the output params, training loss, and dict with useful items to log"""

        # time indexing notes:
        # in paper: for t in {2, ... T},   compute E_{x_t}   kl[ q(x_t-1 | ... || p(x_t-1 | ... ) ] + recon loss
        #  in code: for t in {1, ... T-1}, compute E_{x_t+1} kl[ q(x_t | ...)  || p(x_t | ...) ] + recon loss
        # t == 0 means recon loss only

        t = torch.randint(0, self.T, (data.size(0),), device=data.device) if t is None else t
        # x_0 = F.one_hot(data, self.K).float()  # x_0.shape == BTK, t.shape = B

        # 1. Compute the log posterior: first sample from q(x_{t+1} | x_0), then compute q(x_t | x_{t+1}, x_0)
        log_q, x_tplus1 = self.compute_unnormalized_log_posterior(data, t)

        # 2. Predict x_0 and use it to compute p(x_t | x_{t+1})
        log_predicted_x_0 = self.net(x_tplus1, (t + 1).float())
        log_predicted_x_0[:, :, -1] = -float("inf")
        p_x_0 = F.softmax(log_predicted_x_0, dim=-1)
        log_p, _ = self.compute_unnormalized_log_posterior(p_x_0, t, x_tplus1)

        # 3. Compute KL(q || p)
        l_kl = F.softmax(log_q, dim=-1) * (F.log_softmax(log_q, dim=-1) - F.log_softmax(log_p, dim=-1))
        l_kl = F.relu(l_kl.sum(dim=-1))  # stability trick from official impl

        # 4. Compute CE for use as auxiliary loss and l_0
        l_ce = F.cross_entropy(log_predicted_x_0.view(-1, self.K), data.flatten(), reduction="none").view_as(data)

        loss = l_kl + self.lambda_ce * l_ce
        loss = torch.where(t[:, None] == 0, l_ce, loss)

        # 5. Compute an estimate of the T-step loss
        l_0 = l_ce[t == 0]
        l_kl = l_kl[t > 0]  # this is l_{T-1}
        if l_0.numel() > 0:
            l_T = l_kl.mean() * (self.T - 1) + l_0.mean()
        else:
            l_T = l_kl.mean() * self.T

        return log_predicted_x_0, loss.mean(), dict(kl=l_kl.mean(), ce=l_ce.mean(), l_T=l_T, bpt=l_T / math.log(2))


if __name__ == "__main__":
    B, L, V = 2, 3, 4
    x = torch.randint(0, V, (B, L))
    T = 5
    d3pm = D3PMAbsorbing(
        vocab_size=V,
        n_embed=8,
        n_cond=4,
        n_heads=1,
        n_blocks=1,
        dropout=0.0,
        T=T,
        lambda_ce=0.01,
    )
    print(d3pm(x)[0].shape)
