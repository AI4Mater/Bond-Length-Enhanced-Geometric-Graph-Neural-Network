"""High-level model training orchestration."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from .checkpoints import load_checkpoint, save_checkpoint
from .config import TrainConfig
from .datasets import make_loader
from .evaluation import evaluate_predictions, load_scaler, mean_metric, predict
from .losses import get_loss_func, masked_loss
from .models import build_model
from .utils import set_seed


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_func: Callable,
    config: TrainConfig,
) -> float:
    model.train()
    losses = []
    device = torch.device(config.device)

    for batch in data_loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        preds = model(batch)
        targets = batch.y.to(device).reshape_as(preds)

        if hasattr(batch, "y_mask") and batch.y_mask is not None:
            mask = batch.y_mask.to(device).reshape_as(preds)
            targets = torch.where(mask, targets, torch.zeros_like(targets))
        else:
            mask = ~torch.isnan(targets)
            targets = torch.where(mask, targets, torch.zeros_like(targets))

        loss = masked_loss(loss_func(preds, targets), mask)
        loss.backward()

        if config.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else float("nan")


def make_split_loaders(config: TrainConfig) -> Dict[str, DataLoader]:
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "lazy_shards": config.lazy_shards,
        "shard_cache_size": config.shard_cache_size,
    }
    return {
        "train": make_loader(config.processed_data_dir, "train", shuffle=True, **loader_kwargs),
        "val": make_loader(config.processed_data_dir, "val", shuffle=False, **loader_kwargs),
        "test": make_loader(config.processed_data_dir, "test", shuffle=False, **loader_kwargs),
    }


def train_model(
    config: TrainConfig,
    model_factory: Optional[Callable[[TrainConfig], nn.Module]] = None,
) -> Dict[str, Any]:
    set_seed(config.seed)
    os.makedirs(config.save_dir, exist_ok=True)

    model_factory = model_factory or build_model
    model = model_factory(config).to(torch.device(config.device))
    scaler = load_scaler(config.processed_data_dir)
    loaders = make_split_loaders(config)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_func = get_loss_func(config)
    metrics = [config.metric] + [m for m in config.extra_metrics if m != config.metric]

    best_score = float("inf") if config.minimize_score else -float("inf")
    best_epoch = -1
    history = []
    best_path = os.path.join(config.save_dir, "model.pt")

    for epoch in range(config.epochs):
        train_loss = train_epoch(model, loaders["train"], optimizer, loss_func, config)
        val_preds, val_targets = predict(model, loaders["val"], config, scaler=scaler)
        val_scores = evaluate_predictions(val_preds, val_targets, metrics, config.dataset_type)
        score = mean_metric(val_scores[config.metric])
        print(
            f"Epoch {epoch + 1}/{config.epochs} "
            f"train_loss={train_loss:.6f} "
            f"val_{config.metric}={score:.6f}",
            flush=True,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_scores": val_scores,
            "val_mean_score": score,
        }
        history.append(epoch_record)

        improved = score < best_score if config.minimize_score else score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            save_checkpoint(best_path, model, config, scaler=scaler, scores=epoch_record)

        with open(os.path.join(config.save_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    checkpoint = load_checkpoint(best_path, device=config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_preds, test_targets = predict(model, loaders["test"], config, scaler=scaler)
    test_scores = evaluate_predictions(test_preds, test_targets, metrics, config.dataset_type)

    result = {
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "test_scores": test_scores,
        "config": asdict(config),
        "checkpoint_path": best_path,
    }
    with open(os.path.join(config.save_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result
