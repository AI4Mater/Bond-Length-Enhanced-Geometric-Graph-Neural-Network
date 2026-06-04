"""High-level hyperparameter optimization orchestration."""

from __future__ import annotations

import os
from functools import partial
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .config import TrainConfig
from .hyperopt_search import DEFAULT_SEARCH_SPACE, build_hyperopt_search_space
from .hyperopt_storage import (
    load_completed_trials,
    load_trials_object,
    save_best_trial,
    save_trials,
    save_trials_object,
    trial_path,
)
from .training import train_model
from .utils import clone_config


@dataclass
class HyperoptConfig:
    base_train_config: TrainConfig
    save_dir: str
    num_trials: int = 20
    search_space: Dict[str, List[Any]] = field(default_factory=lambda: dict(DEFAULT_SEARCH_SPACE))
    seed: int = 0
    metric: Optional[str] = None
    minimize_score: Optional[bool] = None
    resume: bool = True
    startup_trials: int = 5


def run_hyperparameter_optimization(
    config: HyperoptConfig,
    model_factory: Optional[Callable[[TrainConfig], Any]] = None,
) -> Dict[str, Any]:
    os.makedirs(config.save_dir, exist_ok=True)

    trial_records = load_completed_trials(config.save_dir) if config.resume else []
    from hyperopt import STATUS_OK, Trials, fmin, tpe

    hyperopt_trials = load_trials_object(config.save_dir) if config.resume else Trials()

    metric = config.metric or config.base_train_config.metric
    minimize_score = (
        config.minimize_score
        if config.minimize_score is not None
        else config.base_train_config.minimize_score
    )
    search_space = build_hyperopt_search_space(config.search_space)

    def objective(sampled: Dict[str, Any]) -> Dict[str, Any]:
        trial_index = len(trial_records)
        save_dir = trial_path(config.save_dir, trial_index)
        train_config = clone_config(
            config.base_train_config,
            save_dir=save_dir,
            metric=metric,
            minimize_score=minimize_score,
            seed=config.base_train_config.seed + trial_index,
            **sampled,
        )

        try:
            result = train_model(train_config, model_factory=model_factory)
            score = result["best_val_score"]
            status = "ok"
            error = None
        except Exception as exc:
            score = float("inf") if minimize_score else -float("inf")
            status = "failed"
            error = repr(exc)
            result = {}

        loss = score if minimize_score else -score
        if np.isnan(loss):
            loss = float("inf")

        trial_records.append({
            "trial_index": trial_index,
            "status": status,
            "score": score,
            "loss": loss,
            "metric": metric,
            "hyperparameters": sampled,
            "save_dir": save_dir,
            "error": error,
            "result": result,
        })
        save_trials(config.save_dir, trial_records)
        return {"loss": loss, "status": STATUS_OK}

    if len(hyperopt_trials.trials) < config.num_trials:
        algorithm = partial(tpe.suggest, n_startup_jobs=config.startup_trials)
        fmin(
            fn=objective,
            space=search_space,
            algo=algorithm,
            max_evals=config.num_trials,
            trials=hyperopt_trials,
            rstate=np.random.default_rng(config.seed),
            show_progressbar=True,
        )
        save_trials_object(config.save_dir, hyperopt_trials)

    valid_trials = [
        trial
        for trial in trial_records
        if trial["status"] == "ok" and not np.isnan(trial["score"])
    ]
    if not valid_trials:
        raise RuntimeError("No successful hyperparameter trials.")

    best_trial = min(valid_trials, key=lambda trial: trial["score"]) if minimize_score else max(
        valid_trials, key=lambda trial: trial["score"]
    )

    save_best_trial(config.save_dir, best_trial)

    return {
        "best_trial": best_trial,
        "num_trials": len(trial_records),
        "config": {
            "save_dir": config.save_dir,
            "num_trials": config.num_trials,
            "search_space": config.search_space,
            "seed": config.seed,
            "startup_trials": config.startup_trials,
            "base_train_config": asdict(config.base_train_config),
        },
    }
