"""Configuration objects for E4 training and hyperparameter search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class TrainConfig:
    processed_data_dir: str
    save_dir: str
    dataset_type: str = "regression"
    num_tasks: int = 1
    metric: str = "rmse"
    extra_metrics: List[str] = field(default_factory=list)
    minimize_score: Optional[bool] = None
    epochs: int = 30
    batch_size: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: Optional[float] = None
    seed: int = 0
    device: Optional[str] = None
    num_workers: int = 0
    pin_memory: bool = False
    lazy_shards: bool = True
    shard_cache_size: int = 2

    # Model args.
    use_ffkn: bool = False
    use_bond_attention: bool = True
    hidden_size: int = 300
    depth: int = 3
    dropout: float = 0.0
    activation: str = "ReLU"
    ffn_hidden_size: Optional[int] = None
    ffn_num_layers: int = 2
    num_heads: int = 8
    loss_function: Optional[str] = None
    multiclass_num_classes: int = 3
    spectra_activation: Optional[str] = None
    cuda: bool = False
    bias: bool = False
    train_data_size: int = 0

    def __post_init__(self) -> None:
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size
        if self.loss_function is None:
            if self.dataset_type == "classification":
                self.loss_function = "binary_cross_entropy"
            elif self.dataset_type == "multiclass":
                self.loss_function = "cross_entropy"
            else:
                self.loss_function = "mse"
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cuda = str(self.device).startswith("cuda")
        if self.minimize_score is None:
            self.minimize_score = self.metric not in {
                "auc",
                "prc-auc",
                "accuracy",
                "r2",
                "f1",
                "mcc",
            }
