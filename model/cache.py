"""Cache helpers for graph preprocessing."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional

import torch
from rdkit import Chem


GRAPH_FEATURE_VERSION = "epic_geo_attn_no_fp_x_v1"


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def graph_cache_key(
    smiles: str,
    pos_mode: str = "mmff",
    num_confs: int = 10,
    feature_version: str = GRAPH_FEATURE_VERSION,
) -> str:
    canonical = canonicalize_smiles(smiles)
    raw_key = f"{feature_version}|{pos_mode}|{num_confs}|{canonical}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def graph_cache_path(
    graph_cache_dir: str,
    smiles: str,
    pos_mode: str = "mmff",
    num_confs: int = 10,
    feature_version: str = GRAPH_FEATURE_VERSION,
) -> str:
    key = graph_cache_key(
        smiles=smiles,
        pos_mode=pos_mode,
        num_confs=num_confs,
        feature_version=feature_version,
    )
    return os.path.join(graph_cache_dir, f"{key}.pt")


def load_graph_cache(path: str):
    if not os.path.exists(path):
        return None
    return torch.load(path, weights_only=False)


def save_graph_cache(data, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(data, tmp_path)
    os.replace(tmp_path, path)


def write_manifest(save_dir: str, manifest: Dict[str, Any]) -> None:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "manifest.json")
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(tmp_path, path)


def read_manifest(save_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(save_dir, "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
