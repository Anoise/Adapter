import torch
import torch.nn.functional as F

from torch import Tensor


def gumbel_sigmoid(logits: Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5) -> Tensor:
    """
    Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
    The discretization converts the values greater than `threshold` to 1 and the rest to 0.
    The code is adapted from the official PyTorch implementation of gumbel_softmax:
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized,
            but will be differentiated as if it is the soft sample in autograd
     threshold: threshold for the discretization,
                values greater than this will be set to 1 and the rest to 0

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
      If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
      be probability distributions.

    """
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret



# ------------------------------
# Regularizers (selectors)
# ------------------------------
def l1_sparsity(M: torch.Tensor) -> torch.Tensor:
    # Encourage fewer selected inputs
    return M.abs().mean()

def binary_entropy(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Push probabilities toward {0,1}
    return -(M.clamp(eps, 1 - eps) * torch.log(M.clamp(eps, 1 - eps)) +
             (1 - M.clamp(eps, 1 - eps)) * torch.log((1 - M).clamp(eps, 1 - eps))).mean()

def tv_temporal(M: torch.Tensor) -> torch.Tensor:
    # Total variation along time axis to encourage contiguous selections
    # M: [B, L, d]
    if M.size(1) < 2: return torch.tensor(0.0, device=M.device)
    return (M[:, 1:, :] - M[:, :-1, :]).abs().mean()

def budget_violation(M: torch.Tensor, kappa: float = 0.5) -> torch.Tensor:
    """
    Enforce average selection fraction <= kappa (0..1)
    returns max(mean(M) - kappa, 0)
    """
    bar_m = M.mean()
    return F.relu(bar_m - kappa)
