"""Multi-task loss combining side CE + bbox Smooth L1 + direction CE."""

from __future__ import annotations

import torch
from torch import nn


class MultiTaskLoss(nn.Module):
    """Weighted sum: total = w_side * CE + w_reg * SmoothL1 + w_dir * CE.

    Smooth L1 (Huber) is used on the regression targets instead of MSE because it
    handles occasional large errors more gracefully.
    """

    def __init__(self, w_side: float = 1.0, w_reg: float = 5.0, w_dir: float = 1.0):
        super().__init__()
        self.w_side = w_side
        self.w_reg = w_reg
        self.w_dir = w_dir
        self.ce = nn.CrossEntropyLoss()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(
        self,
        side_logits: torch.Tensor,
        reg_preds: torch.Tensor,
        dir_logits: torch.Tensor,
        side_t: torch.Tensor,
        reg_t: torch.Tensor,
        dir_t: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ls = self.ce(side_logits, side_t)
        lr = self.smooth_l1(reg_preds, reg_t)
        ld = self.ce(dir_logits, dir_t)
        total = self.w_side * ls + self.w_reg * lr + self.w_dir * ld
        return total, {
            "side": float(ls.item()),
            "reg": float(lr.item()),
            "dir": float(ld.item()),
            "total": float(total.item()),
        }
