"""CSV reading, splitting, and preprocessing orchestration utilities."""

from __future__ import annotations

import csv
import math
import os
import pickle
from dataclasses import dataclass
from random import Random
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

from .cache import GRAPH_FEATURE_VERSION, write_manifest
from .graph_conversion import dataset_to_pyg


RDLogger.DisableLog("rdApp.*")


class StandardScaler:
    """Simple target scaler compatible with regression preprocessing."""

    def __init__(
        self,
        means: Optional[Union[np.ndarray, Sequence[float]]] = None,
        stds: Optional[Union[np.ndarray, Sequence[float]]] = None,
        replace_nan_token: float = 0.0,
    ) -> None:
        self.means = None if means is None else np.array(means, dtype=float)
        self.stds = None if stds is None else np.array(stds, dtype=float)
        self.replace_nan_token = replace_nan_token

    def fit(self, values: Sequence[Sequence[Optional[float]]]) -> "StandardScaler":
        array = np.array(values, dtype=float)
        self.means = np.nanmean(array, axis=0)
        self.stds = np.nanstd(array, axis=0)
        self.means = np.where(np.isnan(self.means), self.replace_nan_token, self.means)
        self.stds = np.where(
            np.isnan(self.stds) | (self.stds == 0), 1.0, self.stds
        )
        return self

    def transform(self, values: Sequence[Sequence[Optional[float]]]) -> np.ndarray:
        if self.means is None or self.stds is None:
            raise ValueError("StandardScaler must be fit before transform().")
        array = np.array(values, dtype=float)
        scaled = (array - self.means) / self.stds
        return np.where(np.isnan(scaled), self.replace_nan_token, scaled)

    def inverse_transform(self, values: Sequence[Sequence[float]]) -> np.ndarray:
        if self.means is None or self.stds is None:
            raise ValueError("StandardScaler must be fit before inverse_transform().")
        return np.array(values, dtype=float) * self.stds + self.means


@dataclass
class MoleculeDatapoint:
    smiles: List[str]
    targets: Optional[List[Optional[float]]] = None
    row: Optional[dict] = None
    index: Optional[int] = None

    @property
    def mol(self) -> List[Optional[Chem.Mol]]:
        return [Chem.MolFromSmiles(smiles) for smiles in self.smiles]


class MoleculeDataset:
    """Lightweight list-like dataset for SMILES and targets."""

    def __init__(self, data: Sequence[MoleculeDatapoint]) -> None:
        self._data = list(data)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return MoleculeDataset(self._data[item])
        if isinstance(item, (list, tuple, np.ndarray)):
            return MoleculeDataset([self._data[int(i)] for i in item])
        return self._data[item]

    def smiles(self, flatten: bool = False) -> Union[List[str], List[List[str]]]:
        values = [datapoint.smiles for datapoint in self._data]
        if flatten:
            return [smiles for row in values for smiles in row]
        return values

    def targets(self) -> List[Optional[List[Optional[float]]]]:
        return [datapoint.targets for datapoint in self._data]

    def normalize_targets(self, scaler: Optional[StandardScaler] = None) -> StandardScaler:
        original_targets = self.targets()
        if scaler is None:
            scaler = StandardScaler().fit(original_targets)
        scaled_targets = scaler.transform(original_targets)
        for datapoint, raw, scaled in zip(self._data, original_targets, scaled_targets):
            datapoint.targets = [
                None if target is None else float(scaled_value)
                for target, scaled_value in zip(raw, scaled)
            ]
        return scaler


def get_header(path: str) -> List[str]:
    with open(path, newline="") as f:
        return next(csv.reader(f))


def preprocess_smiles_columns(
    path: str,
    smiles_columns: Optional[Union[str, List[str]]] = None,
    number_of_molecules: int = 1,
) -> List[str]:
    if smiles_columns is None:
        return get_header(path)[:number_of_molecules]

    if isinstance(smiles_columns, str):
        smiles_columns = [smiles_columns]

    columns = get_header(path)
    missing = [column for column in smiles_columns if column not in columns]
    if missing:
        raise ValueError(f"SMILES columns not found in {path}: {missing}")
    if len(smiles_columns) != number_of_molecules:
        raise ValueError("Length of smiles_columns must match number_of_molecules.")
    return smiles_columns


def get_task_names(
    path: str,
    target_columns: Optional[List[str]] = None,
    smiles_columns: Optional[Union[str, List[str]]] = None,
    ignore_columns: Optional[List[str]] = None,
) -> List[str]:
    if target_columns is not None:
        return target_columns

    smiles_columns = preprocess_smiles_columns(path, smiles_columns)
    ignore = set(smiles_columns + ([] if ignore_columns is None else ignore_columns))
    return [column for column in get_header(path) if column not in ignore]


def _parse_target(value: str) -> Optional[float]:
    if value is None or value == "" or value.lower() == "nan":
        return None
    return float(value)


def filter_invalid_smiles(data: MoleculeDataset) -> MoleculeDataset:
    valid = []
    for datapoint in tqdm(data, desc="Filtering invalid SMILES"):
        mols = datapoint.mol
        if all(smiles != "" for smiles in datapoint.smiles) and all(
            mol is not None and mol.GetNumHeavyAtoms() > 0 for mol in mols
        ):
            valid.append(datapoint)
    return MoleculeDataset(valid)


def get_data(
    path: str,
    smiles_columns: Optional[Union[str, List[str]]] = None,
    target_columns: Optional[List[str]] = None,
    ignore_columns: Optional[List[str]] = None,
    skip_invalid_smiles: bool = True,
    max_data_size: Optional[int] = None,
    store_row: bool = False,
    skip_none_targets: bool = False,
) -> MoleculeDataset:
    """Read a CSV file into a MoleculeDataset."""

    smiles_columns = preprocess_smiles_columns(path, smiles_columns)
    target_columns = get_task_names(
        path=path,
        target_columns=target_columns,
        smiles_columns=smiles_columns,
        ignore_columns=ignore_columns,
    )

    data = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing_smiles = [column for column in smiles_columns if column not in reader.fieldnames]
        missing_targets = [column for column in target_columns if column not in reader.fieldnames]
        if missing_smiles or missing_targets:
            raise ValueError(
                f"Missing columns in {path}. SMILES: {missing_smiles}; targets: {missing_targets}"
            )

        for index, row in enumerate(tqdm(reader, desc="Reading CSV")):
            smiles = [row[column] for column in smiles_columns]
            targets = [_parse_target(row[column]) for column in target_columns]
            if skip_none_targets and all(target is None for target in targets):
                continue
            data.append(
                MoleculeDatapoint(
                    smiles=smiles,
                    targets=targets,
                    row=row if store_row else None,
                    index=index,
                )
            )
            if max_data_size is not None and len(data) >= max_data_size:
                break

    dataset = MoleculeDataset(data)
    if skip_invalid_smiles:
        dataset = filter_invalid_smiles(dataset)
    return dataset


def scaffold_from_smiles(smiles: str) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=True)


def split_data(
    data: MoleculeDataset,
    split_type: str = "random",
    sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    key_molecule_index: int = 0,
) -> Tuple[MoleculeDataset, MoleculeDataset, MoleculeDataset]:
    """Split data into train, validation, and test datasets."""

    if len(sizes) != 3 or not math.isclose(sum(sizes), 1.0):
        raise ValueError(f"Split sizes must contain 3 values summing to 1. Got {sizes}.")
    if any(size < 0 for size in sizes):
        raise ValueError(f"Split sizes must be non-negative. Got {sizes}.")

    random = Random(seed)
    indices = list(range(len(data)))

    if split_type == "random":
        random.shuffle(indices)
        train_end = int(sizes[0] * len(indices))
        val_end = train_end + int(sizes[1] * len(indices))
        return data[indices[:train_end]], data[indices[train_end:val_end]], data[indices[val_end:]]

    if split_type in {"scaffold", "scaffold_balanced"}:
        scaffold_to_indices = {}
        for index, datapoint in enumerate(data):
            scaffold = scaffold_from_smiles(datapoint.smiles[key_molecule_index])
            scaffold_to_indices.setdefault(scaffold, []).append(index)

        scaffold_sets = list(scaffold_to_indices.values())
        random.shuffle(scaffold_sets)
        scaffold_sets.sort(key=len, reverse=True)

        train_cutoff = sizes[0] * len(data)
        val_cutoff = (sizes[0] + sizes[1]) * len(data)
        train_indices, val_indices, test_indices = [], [], []

        for scaffold_set in scaffold_sets:
            if len(train_indices) + len(scaffold_set) <= train_cutoff:
                train_indices.extend(scaffold_set)
            elif len(train_indices) + len(val_indices) + len(scaffold_set) <= val_cutoff:
                val_indices.extend(scaffold_set)
            else:
                test_indices.extend(scaffold_set)

        return data[train_indices], data[val_indices], data[test_indices]

    raise ValueError(f"Unsupported split_type: {split_type}")


def preprocess_csv_to_pyg(
    data_path: str,
    save_dir: str,
    smiles_columns: Optional[Union[str, List[str]]] = None,
    target_columns: Optional[List[str]] = None,
    ignore_columns: Optional[List[str]] = None,
    split_type: str = "random",
    split_sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    dataset_type: str = "regression",
    smiles_index: int = 0,
    graph_cache_dir: Optional[str] = None,
    shard_size: Optional[int] = None,
    num_workers: int = 0,
    pos_mode: str = "mmff",
    num_confs: int = 10,
    overwrite_cache: bool = False,
    resume: bool = True,
    return_graphs: Optional[bool] = None,
) -> Tuple[list, list, list]:
    """End-to-end CSV -> split -> PyG .pt preprocessing."""

    os.makedirs(save_dir, exist_ok=True)
    if graph_cache_dir is not None:
        os.makedirs(graph_cache_dir, exist_ok=True)
    if return_graphs is None:
        return_graphs = shard_size is None

    data = get_data(
        path=data_path,
        smiles_columns=smiles_columns,
        target_columns=target_columns,
        ignore_columns=ignore_columns,
    )
    train_data, val_data, test_data = split_data(
        data=data,
        split_type=split_type,
        sizes=split_sizes,
        seed=seed,
        key_molecule_index=smiles_index,
    )

    raw_test_targets = test_data.targets()

    if dataset_type == "regression":
        scaler = train_data.normalize_targets()
        val_data.normalize_targets(scaler)
        test_data.normalize_targets(scaler)
        with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

    train_graphs, train_shards = convert_and_save_split(
        data=train_data,
        split_name="train",
        save_dir=save_dir,
        smiles_index=smiles_index,
        graph_cache_dir=graph_cache_dir,
        shard_size=shard_size,
        num_workers=num_workers,
        pos_mode=pos_mode,
        num_confs=num_confs,
        overwrite_cache=overwrite_cache,
        resume=resume,
        return_graphs=return_graphs,
    )
    val_graphs, val_shards = convert_and_save_split(
        data=val_data,
        split_name="val",
        save_dir=save_dir,
        smiles_index=smiles_index,
        graph_cache_dir=graph_cache_dir,
        shard_size=shard_size,
        num_workers=num_workers,
        pos_mode=pos_mode,
        num_confs=num_confs,
        overwrite_cache=overwrite_cache,
        resume=resume,
        return_graphs=return_graphs,
    )
    test_graphs, test_shards = convert_and_save_split(
        data=test_data,
        split_name="test",
        save_dir=save_dir,
        smiles_index=smiles_index,
        graph_cache_dir=graph_cache_dir,
        shard_size=shard_size,
        num_workers=num_workers,
        pos_mode=pos_mode,
        num_confs=num_confs,
        overwrite_cache=overwrite_cache,
        resume=resume,
        return_graphs=return_graphs,
    )

    with open(os.path.join(save_dir, "test_targets.pkl"), "wb") as f:
        pickle.dump(raw_test_targets, f)

    write_manifest(
        save_dir=save_dir,
        manifest={
            "data_path": data_path,
            "smiles_columns": smiles_columns,
            "target_columns": target_columns,
            "ignore_columns": ignore_columns,
            "split_type": split_type,
            "split_sizes": list(split_sizes),
            "seed": seed,
            "dataset_type": dataset_type,
            "smiles_index": smiles_index,
            "feature_version": GRAPH_FEATURE_VERSION,
            "graph_cache_dir": graph_cache_dir,
            "shard_size": shard_size,
            "num_workers": num_workers,
            "pos_mode": pos_mode,
            "num_confs": num_confs,
            "return_graphs": return_graphs,
            "sizes": {
                "train": len(train_data),
                "val": len(val_data),
                "test": len(test_data),
            },
            "shards": {
                "train": train_shards,
                "val": val_shards,
                "test": test_shards,
            },
        },
    )

    return train_graphs, val_graphs, test_graphs


def iter_dataset_shards(data: MoleculeDataset, shard_size: int):
    if shard_size <= 0:
        raise ValueError("shard_size must be positive.")
    for start in range(0, len(data), shard_size):
        yield start // shard_size, data[start : start + shard_size]


def convert_and_save_split(
    data: MoleculeDataset,
    split_name: str,
    save_dir: str,
    smiles_index: int = 0,
    graph_cache_dir: Optional[str] = None,
    shard_size: Optional[int] = None,
    num_workers: int = 0,
    pos_mode: str = "mmff",
    num_confs: int = 10,
    overwrite_cache: bool = False,
    resume: bool = True,
    return_graphs: bool = True,
) -> Tuple[list, List[str]]:
    """Convert a split to graphs and save either a single file or shard files."""

    if shard_size is None:
        graphs = dataset_to_pyg(
            data,
            smiles_index=smiles_index,
            graph_cache_dir=graph_cache_dir,
            pos_mode=pos_mode,
            num_confs=num_confs,
            num_workers=num_workers,
            overwrite_cache=overwrite_cache,
        )
        file_name = f"{split_name}.pt"
        torch.save(graphs, os.path.join(save_dir, file_name))
        return graphs if return_graphs else [], [file_name]

    all_graphs = []
    shard_files = []
    for shard_index, shard_data in iter_dataset_shards(data, shard_size):
        file_name = f"{split_name}_{shard_index:05d}.pt"
        file_path = os.path.join(save_dir, file_name)
        shard_files.append(file_name)

        if resume and os.path.exists(file_path):
            if return_graphs:
                shard_graphs = torch.load(file_path, weights_only=False)
            else:
                shard_graphs = []
        else:
            shard_graphs = dataset_to_pyg(
                shard_data,
                smiles_index=smiles_index,
                graph_cache_dir=graph_cache_dir,
                pos_mode=pos_mode,
                num_confs=num_confs,
                num_workers=num_workers,
                overwrite_cache=overwrite_cache,
            )
            tmp_path = f"{file_path}.tmp"
            torch.save(shard_graphs, tmp_path)
            os.replace(tmp_path, file_path)

        if return_graphs:
            all_graphs.extend(shard_graphs)

    return all_graphs, shard_files


def load_saved_split(save_dir: str, split_name: str):
    """Load either a single split file or all shard files for a split."""

    single_path = os.path.join(save_dir, f"{split_name}.pt")
    if os.path.exists(single_path):
        return torch.load(single_path, weights_only=False)

    graphs = []
    shard_names = sorted(
        name
        for name in os.listdir(save_dir)
        if name.startswith(f"{split_name}_") and name.endswith(".pt")
    )
    for shard_name in shard_names:
        graphs.extend(torch.load(os.path.join(save_dir, shard_name), weights_only=False))
    return graphs
