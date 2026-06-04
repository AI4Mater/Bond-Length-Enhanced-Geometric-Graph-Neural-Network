"""Hyperopt search-space utilities."""

from __future__ import annotations

from typing import Any, Dict, List


DEFAULT_SEARCH_SPACE = {
    "hidden_size": [128, 256, 300, 512],
    "ffn_hidden_size": [128, 256, 300, 512],
    "depth": [2, 3, 4],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3],
    "batch_size": [32, 50, 64, 128],
    "num_heads": [4, 8],
}


def build_hyperopt_search_space(search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Convert a discrete candidate dictionary into a Hyperopt search space."""

    from hyperopt import hp

    return {key: hp.choice(key, values) for key, values in search_space.items()}
