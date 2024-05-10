import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributions as D

from dit import DiT


class D3PM(nn.Module):
    def __init__(self, vocab_size, n_embed, n_heads, n_blocks, n_cond, dropout, T: int, lambda_ce: float) -> None:
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

        # these are used to compute Q_t and Qbar_t
        Q_full_mask = torch.zeros(self.K, self.K)
        Q_full_mask[:, -1] = 1.0
        self.register_buffer("Q_full_mask", Q_full_mask)
        self.register_buffer("eye_K", torch.eye(self.K))

    def Q_t(self, t: Tensor) -> Tensor:
        # t is shaped (B,) output is shaped (B, K, K)
        beta_t = self.betas[t][:, None, None]
        return (1 - beta_t) * self.eye_K + beta_t * self.Q_full_mask

    def Qbar_t(self, t: Tensor) -> Tensor:
        # t is shaped (B,) output is shaped (B, K, K)
        alpha_bar_t = self.alpha_bars[t][:, None, None]
        return alpha_bar_t * self.eye_K + (1 - alpha_bar_t) * self.Q_full_mask

    def compute_unnormalized_log_posterior(self, q_0, t, x_tplus1=None):
        # TODO: efficient multiplication so q_0 and x_tplus1 can be indices instead of one-hot

        # compute q(x_t | q_0) = q_0 @ Qbar_t, q_0 can be one_hot(x_0) or a predictive distribution p_x_0
        q_x_t_given_x_0 = q_0 @ self.Qbar_t(t)

        # compute q(x_t+1 | x_0) = q(x_t | x_0) @ Q_t+1 (note that Q_t(T) should be defined since t goes from 0 to T-1)
        Q_tplus1 = self.Q_t(t + 1)
        q_x_tplus1_given_x_0 = q_x_t_given_x_0 @ Q_tplus1

        if x_tplus1 is None:
            # sample x_t+1 from q(x_t+1 | x_0)
            x_tplus1 = D.Categorical(probs=q_x_tplus1_given_x_0).sample()  # TODO: BT, convert to one_hot?
            x_tplus1 = F.one_hot(x_tplus1, self.K).float()

        # multiply x_t+1 with transpose(Q_t+1) to get q(x_t+1 | x_t)
        q_x_tplus1_given_x_t = x_tplus1 @ Q_tplus1.transpose(-1, -2)

        # q(x_t | x_0) is already computed, so compute unnormalized posterior log probs
        # log[q(x_t | x_t+1, x_0)] = log[q(x_t+1 | x_t)] + log[q(x_t | x_0)]
        log_posterior = torch.log(q_x_tplus1_given_x_t + self.eps) + torch.log(q_x_t_given_x_0 + self.eps)
        log_posterior = torch.where(t[:, None, None] == 0, torch.log(q_0 + self.eps), log_posterior)

        return log_posterior, x_tplus1

    def forward(self, data: Tensor) -> tuple[Tensor, Tensor, dict]:
        """Return the output params, training loss, and dict with useful items"""
        # in paper: for t in {2, ... T},   compute E_{x_t}   kl[ q(x_t-1 | ... || p(x_t-1 | ... ) ] + recon loss
        #  in code: for t in {1, ... T-1}, compute E_{x_t+1} kl[ q(x_t | ...)  || p(x_t | ...) ] + recon loss
        # t == 0 means recon loss only

        t = torch.randint(0, self.T, (data.size(0),), device=data.device)
        x_0 = F.one_hot(data, self.K).float()  # x_0.shape == BTK, t.shape = B

        # 1. Compute the log posterior: first sample from q(x_{t+1} | x_0), then compute q(x_t | x_{t+1}, x_0)
        log_q, x_tplus1 = self.compute_unnormalized_log_posterior(x_0, t)

        # 2. Compute p(x_t | x_{t+1}), note that x_tplus1 is one_hot(x_tplus1)
        log_predicted_x_0 = self.net(x_tplus1, (t + 1).float())
        p_x_0 = F.softmax(log_predicted_x_0, dim=-1)
        log_p, _ = self.compute_unnormalized_log_posterior(p_x_0, t, x_tplus1)

        # 3. Compute KL(q || p)
        l_vb = F.softmax(log_q, dim=-1) * (F.log_softmax(log_q, dim=-1) - F.log_softmax(log_p, dim=-1))
        l_vb = F.relu(l_vb.sum(dim=-1))  # stability trick from official impl

        # 4. Compute CE(q) if t == 0
        l_ce = F.cross_entropy(log_predicted_x_0.view(-1, self.K), data.flatten(), reduction="none").view_as(data)

        loss = l_vb + self.lambda_ce * l_ce
        loss = torch.where(t[:, None] == 0, l_ce, loss)

        # 5. Compute an estimate of the T-step loss
        l_0 = l_ce[t == 0]
        l_T = l_vb.mean() * self.T + (l_0.mean() if l_0.numel() > 0 else 0.0)

        return log_predicted_x_0, loss.mean(), dict(vb=l_vb.mean(), ce=l_ce.mean(), l_T=l_T, bpt=l_T / math.log(2))

    @torch.no_grad()
    def sample(self, batch_size: int, seq_len: int) -> Tensor:
        device = next(self.parameters()).device

        # sample from stationary distribution
        x_T = torch.ones(batch_size, seq_len, device=device).long() * (self.K - 1)
        trajectory = [x_T]
        time = torch.ones(batch_size).long().to(device)
        x_t = x_T

        # iteratively sample from the log posterior
        for _t in range(self.T, 0, -1):
            t = time * _t
            x_t = F.one_hot(x_t, self.K).float()
            log_predicted_x_0 = self.net(x_t, t.float() / self.T)
            p_x_0 = F.softmax(log_predicted_x_0, dim=-1)
            if _t > 1:
                log_p_x_tminus1, _ = self.compute_unnormalized_log_posterior(p_x_0, t - 1, x_t)
                x_tminus1 = D.Categorical(logits=log_p_x_tminus1).sample()
                x_t = x_tminus1
                trajectory.append(x_tminus1)

        # now _t is 1, and we have the final predicted p(x_0)
        trajectory.append(D.Categorical(logits=log_predicted_x_0).sample())
        # return the full trajectory of T steps, plus the final sample from p(x_0)
        return torch.stack(trajectory, dim=0)


if __name__ == "__main__":
    B, L, V = 2, 3, 4
    x = torch.randint(0, V, (B, L))
    T = 5
    d3pm = D3PM(vocab_size=V, n_embed=8, n_cond=4, n_heads=1, n_blocks=1, dropout=0.0, T=T, lambda_ce=0.01)
    print(d3pm(x))
