import torch, torch.nn as nn
import torch.nn.functional as F


def large_tensor_quantile(R, q, per_dim=False, chunk_size=1000):
    """
    When dealing with quantile calculations of large tensors, memory overflow is avoided through block partitioning
    Args:
        R: input tensor
        q: quantiles（0-1）
        per_dim: Whether to calculate by dimensions
        chunk_size: The size of each chunk
    """
    # If the tensor is not large, calculate directly
    if R.numel() <= chunk_size:
        return torch.quantile(R, q, dim=0) if per_dim else torch.quantile(R, q)
    
    # multi-resolution blocked processing
    chunks = torch.split(R, chunk_size, dim=0)  # Block by the first dimension
    quantiles = []
    
    for chunk in chunks:
        # Calculate the quantile of each block
        q_chunk = torch.quantile(chunk, q, dim=0) if per_dim else torch.quantile(chunk, q)
        quantiles.append(q_chunk)
    
    # Merge the quantile results of all blocks (recalculate the quantiles to obtain the overall result)
    quantiles = torch.stack(quantiles)
    return torch.quantile(quantiles, q, dim=0) if per_dim else torch.quantile(quantiles, q)


def global_quantile_kth(R, q):
    flat = R.reshape(-1).cpu()
    n = flat.numel()
    # PyTorch's definition aligns with numpy: p in [0,1], linear interpolation.
    # kthvalue gives the 'nearest-rank' (no interpolation). If you need the exact
    # 'linear' interpolation behavior, do the two bracketing ks and interpolate.
    k = int((n - 1) * q) + 1  # 1-based
    kth = flat.kthvalue(k).values
    # Optional interpolation to mimic torch.quantile(mode='linear'):
    if q * (n - 1) % 1 != 0:
        k0 = int((n - 1) * q) + 1
        k1 = min(k0 + 1, n)
        v0 = flat.kthvalue(k0).values
        v1 = flat.kthvalue(k1).values
        t = (q * (n - 1)) % 1
        return (1 - t) * v0 + t * v1
    return kth



@torch.no_grad()
def compute_kappa_from_R(R, alpha:float, per_dim=False):
    q=1.0-alpha
    device = R.device
    if len(R)>10e5:
        kappa = global_quantile_kth(R, q)
    else:
        kappa = torch.quantile(R, q, dim=0) if per_dim else torch.quantile(R, q)
   
    return kappa.to(device)


def compute_kappa_from_R2(R, alpha:float, per_dim=False):
    q=1.0-alpha
    if len(R)<10e5:
        kappa = torch.quantile(R, q, dim=0) if per_dim else torch.quantile(R, q)
    else:
        kappa = large_tensor_quantile(R, q, per_dim)
    return kappa

@torch.no_grad()
def conformal_intervals(w, yhat, kappa, alpha, per_dim=False):
    if per_dim:
        kappa_b = kappa.to(yhat.device).unsqueeze(0).expand_as(yhat)
    else:
        kappa_b = kappa.to(yhat.device)
    rad=kappa_b*w
    return yhat - rad, yhat + rad

class ConformalScaleNet(nn.Module):
    """
    External adapter w_phi(X, y_hat) -> positive scale [B,H,m].
    Fix: when m>1, feed the full horizon vector y_hat[:,h,:] (size m)
         instead of a scalar summary, so in_dim matches forward().
    """
    def __init__(self, L:int, d:int, H:int, m:int,
                 hidden:int=512, use_covariates:bool=True):
        super().__init__()
        self.H, self.m = H, m
        self.use_covariates = use_covariates

        # We will feed [y_hat[:,h,:] (size m)] + optional [x_pool (size d)]
        in_dim = m + (d if use_covariates else 0)

        self.pool_x = nn.AdaptiveAvgPool1d(1) if use_covariates else nn.Identity()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU()
        )
        self.h_emb = nn.Embedding(H, hidden)
        self.head = nn.Linear(hidden, m)  # output per-target scales

        # conservative init
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """
        x:     [B,L,d]
        y_hat: [B,H,m]
        returns w: [B,H,m] > 0
        """
        B, H, m = y_hat.shape
        device = y_hat.device

        if self.use_covariates:
            # pool over time: [B,L,d] -> [B,d,1] -> [B,d]
            x_pool = self.pool_x(x.transpose(1, 2)).squeeze(-1)
        else:
            x_pool = torch.zeros(B, 0, device=device)

        w_list = []
        for h in range(H):
            # >>> FIX: use full m-dim vector for the h-th horizon <<<
            y_feat = y_hat[:, h, :]  # [B, m]

            feats = torch.cat([y_feat, x_pool], dim=-1) if x_pool.numel() > 0 else y_feat
            z = self.trunk(feats) + self.h_emb.weight[h:h+1]   # [B, hidden]

            s = self.head(z)                                   # [B, m]
            w_h = F.softplus(s) + 1e-3                         # positivity + floor
            w_list.append(w_h.unsqueeze(1))                    # [B,1,m]

        w = torch.cat(w_list, dim=1)                            # [B, H, m]
        return w