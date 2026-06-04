"""Data loading utilities for saved PyG graph splits."""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader


def load_graph_file(path: str) -> List[Any]:
    return torch.load(path, weights_only=False)


def get_split_files(processed_data_dir: str, split_name: str) -> List[str]:
    single = os.path.join(processed_data_dir, f"{split_name}.pt")
    if os.path.exists(single):
        return [single]
    files = sorted(
        os.path.join(processed_data_dir, name)
        for name in os.listdir(processed_data_dir)
        if name.startswith(f"{split_name}_") and name.endswith(".pt")
    )
    if not files:
        raise FileNotFoundError(f"No {split_name} graph files found in {processed_data_dir}")
    return files


def load_graphs(processed_data_dir: str, split_name: str) -> List[Any]:
    graphs = []
    for path in get_split_files(processed_data_dir, split_name):
        graphs.extend(load_graph_file(path))
    return graphs


class GraphShardDataset(Dataset):
    """Lazy dataset for split files saved as multiple graph shards."""

    def __init__(self, files: Sequence[str], shard_cache_size: int = 2) -> None:
        self.files = list(files)
        self.shard_cache_size = max(1, shard_cache_size)
        self.shard_lengths = []
        self.cumulative_lengths = []
        total = 0
        for path in self.files:
            shard = load_graph_file(path)
            total += len(shard)
            self.shard_lengths.append(len(shard))
            self.cumulative_lengths.append(total)
        self._cache = OrderedDict()

    def __len__(self) -> int:
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def _locate(self, index: int) -> Tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = int(np.searchsorted(self.cumulative_lengths, index, side="right"))
        previous_total = 0 if shard_index == 0 else self.cumulative_lengths[shard_index - 1]
        return shard_index, index - previous_total

    def _load_shard(self, shard_index: int):
        if shard_index in self._cache:
            self._cache.move_to_end(shard_index)
            return self._cache[shard_index]
        shard = load_graph_file(self.files[shard_index])
        self._cache[shard_index] = shard
        if len(self._cache) > self.shard_cache_size:
            self._cache.popitem(last=False)
        return shard

    def __getitem__(self, index: int):
        shard_index, local_index = self._locate(index)
        return self._load_shard(shard_index)[local_index]


def make_loader(
    processed_data_dir: str,
    split_name: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    lazy_shards: bool = True,
    shard_cache_size: int = 2,
) -> DataLoader:
    files = get_split_files(processed_data_dir, split_name)
    if lazy_shards and len(files) > 1:
        dataset = GraphShardDataset(files, shard_cache_size=shard_cache_size)
    else:
        dataset = []
        for path in files:
            dataset.extend(load_graph_file(path))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
