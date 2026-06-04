"""Prediction and metric utilities."""

from __future__ import annotations

import math
import os
import pickle
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from torch import nn
from torch_geometric.loader import DataLoader

from .config import TrainConfig


def load_scaler(processed_data_dir: str):
    path = os.path.join(processed_data_dir, "scaler.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def inverse_scale(values: np.ndarray, scaler) -> np.ndarray:
    if scaler is None:
        return values
    if hasattr(scaler, "inverse_transform"):
        return scaler.inverse_transform(values)
    return values


@torch.no_grad()
def predict(
    model: nn.Module,
    data_loader: DataLoader,
    config: TrainConfig,
    scaler=None,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    device = torch.device(config.device)

    for batch in data_loader:
        batch = batch.to(device)
        batch_preds = model(batch)
        preds.append(batch_preds.detach().cpu().numpy())
        targets.append(batch.y.detach().cpu().numpy().reshape(batch_preds.shape))

    if not preds:
        return np.empty((0, config.num_tasks)), np.empty((0, config.num_tasks))

    preds_array = np.concatenate(preds, axis=0)
    targets_array = np.concatenate(targets, axis=0)
    if config.dataset_type == "classification":
        preds_array = 1 / (1 + np.exp(-preds_array))
    if config.dataset_type == "regression":
        preds_array = inverse_scale(preds_array, scaler)
        targets_array = inverse_scale(targets_array, scaler)
    return preds_array, targets_array


def metric_value(metric: str, targets: Sequence[float], preds: Sequence[float]) -> float:
    if metric == "rmse":
        return math.sqrt(mean_squared_error(targets, preds))
    if metric == "mse":
        return mean_squared_error(targets, preds)
    if metric == "mae":
        return mean_absolute_error(targets, preds)
    if metric == "r2":
        return r2_score(targets, preds)
    if metric == "auc":
        return roc_auc_score(targets, preds)
    if metric == "accuracy":
        return accuracy_score(targets, [1 if pred > 0.5 else 0 for pred in preds])
    if metric == "f1":
        return f1_score(targets, [1 if pred > 0.5 else 0 for pred in preds])
    if metric == "mcc":
        return matthews_corrcoef(targets, [1 if pred > 0.5 else 0 for pred in preds])
    raise ValueError(f"Unsupported metric: {metric}")


def evaluate_predictions(
    preds: np.ndarray,
    targets: np.ndarray,
    metrics: Sequence[str],
    dataset_type: str,
) -> Dict[str, List[float]]:
    results = {metric: [] for metric in metrics}
    num_tasks = preds.shape[1] if preds.ndim == 2 else 1

    for task_index in range(num_tasks):
        task_preds = preds[:, task_index]
        task_targets = targets[:, task_index]
        valid = ~np.isnan(task_targets)
        task_preds = task_preds[valid]
        task_targets = task_targets[valid]

        for metric in metrics:
            if len(task_targets) == 0:
                results[metric].append(float("nan"))
                continue
            if dataset_type == "classification" and len(set(task_targets.tolist())) < 2 and metric == "auc":
                results[metric].append(float("nan"))
                continue
            results[metric].append(metric_value(metric, task_targets, task_preds))

    return results


def mean_metric(scores: Sequence[float]) -> float:
    values = [score for score in scores if not np.isnan(score)]
    return float(np.mean(values)) if values else float("nan")
