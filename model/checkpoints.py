"""Checkpoint save/load helpers."""

from __future__ import annotations

import os
from dataclasses import asdict

import torch
from torch import nn

from .config import TrainConfig


def save_checkpoint(path: str, model: nn.Module, config: TrainConfig, scaler=None, scores=None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "scaler": scaler,
            "scores": scores or {},
        },
        path,
    )


def load_checkpoint(path: str, device: str = "cpu"):
    return torch.load(path, map_location=device, weights_only=False)
