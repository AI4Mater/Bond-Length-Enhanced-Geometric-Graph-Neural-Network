"""Trial persistence helpers for hyperparameter optimization."""

from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List


def trial_path(save_dir: str, trial_index: int) -> str:
    return os.path.join(save_dir, f"trial_{trial_index:04d}")


def load_completed_trials(save_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(save_dir, "trials.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_trials(save_dir: str, trials: List[Dict[str, Any]]) -> None:
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "trials.json"), "w", encoding="utf-8") as f:
        json.dump(trials, f, indent=2, ensure_ascii=False)


def save_best_trial(save_dir: str, best_trial: Dict[str, Any]) -> None:
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "best_config.json"), "w", encoding="utf-8") as f:
        json.dump(best_trial, f, indent=2, ensure_ascii=False)


def trials_object_path(save_dir: str) -> str:
    return os.path.join(save_dir, "hyperopt_trials.pkl")


def load_trials_object(save_dir: str):
    from hyperopt import Trials

    path = trials_object_path(save_dir)
    if not os.path.exists(path):
        return Trials()
    with open(path, "rb") as f:
        return pickle.load(f)


def save_trials_object(save_dir: str, trials) -> None:
    os.makedirs(save_dir, exist_ok=True)
    path = trials_object_path(save_dir)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(trials, f)
    os.replace(tmp_path, path)
