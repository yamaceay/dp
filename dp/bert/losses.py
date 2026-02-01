from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.nn import functional as F


def compute_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: Optional[float] = None,
    ignore_pt: Optional[float] = None,
    reduction: str = "mean",
    return_mask: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"invalid reduction: {reduction}")
    log_probs = F.log_softmax(logits, dim=-1)
    targets = labels.view(-1, 1)
    log_pt = log_probs.gather(1, targets).squeeze(1)
    pt = log_pt.exp()
    loss = -((1.0 - pt) ** gamma) * log_pt
    if alpha is not None:
        loss = loss * float(alpha)
    mask = None
    if ignore_pt is not None:
        mask = (pt <= float(ignore_pt)).to(loss.dtype)
        loss = loss * mask
        if reduction == "mean":
            denom = mask.sum()
            loss = (loss.sum() / denom) if denom > 0 else loss.mean() * 0.0
        elif reduction == "sum":
            loss = loss.sum()
    else:
        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
    if return_mask:
        return loss, mask
    return loss
