import torch, torch.nn as nn
import torch.nn.functional as F
from collections.abc import Sequence
import numpy as np

def pinball_loss(y_true, q_pred, taus, horizon_weights=None, reduction="mean"):
    y = y_true.unsqueeze(-1)
    tau = taus.view(1,1,1,-1).to(q_pred.device)
    diff = y - q_pred
    loss = torch.maximum(tau*diff, (tau-1)*diff)
    if horizon_weights is not None:
        w = horizon_weights.view(1,-1,1,1).to(q_pred.device)
        loss = loss * w
    return loss.mean() if reduction=="mean" else loss.sum()

def soft_coverage_penalty(y_true, q_pred, taus, T=0.5):
    y = y_true.unsqueeze(-1)
    cov_soft = torch.sigmoid((q_pred - y) / T)
    cov_hat = cov_soft.mean(dim=(0,1,2))
    return ((cov_hat - taus.to(q_pred.device))**2).mean()



def compute_PICP(batch_true_y, batch_pred_y, cp_range=(2.5, 97.5), return_CI=False):
    """
    Another coverage metric.
    """
    batch_true_y = batch_true_y.squeeze()
    batch_pred_y = batch_pred_y.squeeze()
    batch_true_y = batch_true_y.reshape(-1,1)
    batch_pred_y = batch_pred_y.reshape(-1,batch_pred_y.shape[-1])
    low, high = cp_range
    CI_y_pred = np.percentile(batch_pred_y, q=[low, high], axis=1)
    y_in_range = (batch_true_y >= CI_y_pred[0]) & (batch_true_y <= CI_y_pred[1])
    coverage = y_in_range.mean()
    if return_CI:
        return coverage, CI_y_pred, low, high
    else:
        return coverage, low, high
    
def compute_PICP_torch(batch_true_y, batch_pred_y, cp_range=(2.5, 97.5), return_CI=False):
    """
    Another coverage metric.
    """
    batch_true_y = batch_true_y.squeeze()
    batch_pred_y = batch_pred_y.squeeze()
    batch_true_y = batch_true_y.reshape(-1,1)
    batch_pred_y = batch_pred_y.reshape(-1,batch_pred_y.shape[-1])
    low, high = cp_range
    quan = torch.tensor([low/100, high/100]).to(batch_pred_y.device)
    CI_y_pred = torch.quantile(batch_pred_y, q=quan, axis=1)
    y_in_range = (batch_true_y >= CI_y_pred[0]) & (batch_true_y <= CI_y_pred[1])
    coverage = y_in_range.float().mean()
    if return_CI:
        return coverage, CI_y_pred, low, high
    else:
        return coverage, low, high






# ---------- QUANTILE CALIBRATOR qcal ----------

class QuantileCalibrator(nn.Module):
    """
    External adapter that turns frozen point forecasts into calibrated quantiles.

    Inputs:
      x:      [B,L,d]   (optional; used for covariate conditioning)
      y_hat:  [B,H,m]   (frozen forecaster output)

    Output:
      q_pred: [B,H,m,J] (taus ascending)
    """
    def __init__(
        self,
        L: int,
        d: int,
        H: int,
        m: int,
        taus: Sequence[float],
        hidden: int = 64,
        eps_add: float = 0.1,
        use_covariates: bool = True,
        anchor_tau: float = 0.5
    ):
        super().__init__()
        self.H, self.m = H, m
        self.taus = torch.tensor(sorted(taus), dtype=torch.float32)
        self.J = len(taus)
        self.eps_add = eps_add
        self.use_covariates = use_covariates

        # Anchor index (closest to anchor_tau)
        self.anchor_idx = int(torch.argmin(torch.abs(self.taus - anchor_tau)))

        # >>> FIX 1: trunk input now matches what we feed in forward <<<
        # We will feed [y_hat[:,h,:] (size m)] and optionally [x pooled over time (size d)]
        in_dim = m + (d if use_covariates else 0)

        self.pool_x = nn.AdaptiveAvgPool1d(1) if use_covariates else nn.Identity()

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU()
        )
        self.h_emb = nn.Embedding(H, hidden)

        # Heads
        self.head_anchor = nn.Linear(hidden, m)
        self.head_up = nn.Linear(hidden, m * max(0, self.J - self.anchor_idx - 1))
        self.head_down = nn.Linear(hidden, m * max(0, self.anchor_idx))

        for lin in [self.head_anchor, self.head_up, self.head_down]:
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """
        x:     [B,L,d] (if use_covariates=False, pass a dummy tensor)
        y_hat: [B,H,m]
        returns q_pred: [B,H,m,J]
        """
        B, H, m, J = y_hat.size(0), self.H, self.m, self.J
        device = y_hat.device

        if self.use_covariates:
            # [B,d] from [B,L,d]
            x_pool = self.pool_x(x.transpose(1, 2)).squeeze(-1)
        else:
            x_pool = torch.zeros(B, 0, device=device)

        q_pred = []
        for h in range(H):
            # >>> FIX 2: use full m-dim horizon vector instead of a scalar mean <<<
            y_feat = y_hat[:, h, :]                     # [B,m]
            feats = [y_feat]
            if x_pool.numel() > 0:
                feats.append(x_pool)                    # [B,d]
            feats = torch.cat(feats, dim=-1)            # [B, m + (d?)]

            z = self.trunk(feats) + self.h_emb.weight[h:h+1]   # [B,hidden]

            # Anchor around y_hat_h
            anchor_off = torch.tanh(self.head_anchor(z)) * self.eps_add   # [B,m]
            anchor = y_hat[:, h, :] + anchor_off                          # [B,m]

            # Build monotone quantiles via positive increments
            up_count = max(0, J - self.anchor_idx - 1)
            down_count = max(0, self.anchor_idx)

            q_h = anchor.unsqueeze(-1).repeat(1, 1, J)                    # [B,m,J]

            if up_count > 0:
                up_raw = self.head_up(z).view(B, m, up_count)             # [B,m,up]
                up_inc = F.softplus(up_raw)
                for j in range(self.anchor_idx + 1, J):
                    step = j - (self.anchor_idx + 1)
                    q_h[:, :, j] = q_h[:, :, j - 1] + up_inc[:, :, step]

            if down_count > 0:
                down_raw = self.head_down(z).view(B, m, down_count)       # [B,m,down]
                down_inc = F.softplus(down_raw)
                for j in range(self.anchor_idx - 1, -1, -1):
                    step = (self.anchor_idx - 1) - j
                    q_h[:, :, j] = q_h[:, :, j + 1] - down_inc[:, :, step]

            q_pred.append(q_h.unsqueeze(1))                                # [B,1,m,J]

        q_pred = torch.cat(q_pred, dim=1)                                  # [B,H,m,J]
        return q_pred