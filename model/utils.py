"""General utilities shared by E4 training modules."""

from __future__ import annotations

import random
from dataclasses import asdict

import numpy as np
import torch

from .config import TrainConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_config(config: TrainConfig, **updates) -> TrainConfig:
    data = asdict(config)
    data.update(updates)
    return TrainConfig(**data)
