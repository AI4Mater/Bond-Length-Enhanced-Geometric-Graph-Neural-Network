"""Loss functions for E4 training."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from .config import TrainConfig


def get_loss_func(config: TrainConfig) -> Callable:
    if config.dataset_type == "classification":
        return nn.BCEWithLogitsLoss(reduction="none")
    if config.dataset_type == "multiclass":
        return nn.CrossEntropyLoss(reduction="none")
    return nn.MSELoss(reduction="none")


def masked_loss(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape != loss.shape:
        mask = mask.expand_as(loss)
    valid_count = mask.sum()
    if valid_count == 0:
        return torch.tensor(0.0, device=loss.device)
    return (loss * mask).sum() / valid_count
