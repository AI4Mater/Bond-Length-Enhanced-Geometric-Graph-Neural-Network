"""SMILES to PyG graph conversion utilities.

The atom and bond features match the original EPIC_Geo_Attn implementation.
Molecular fingerprint features such as `fp_x` are intentionally omitted.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from tqdm import tqdm

from .cache import (
    GRAPH_FEATURE_VERSION,
    graph_cache_path,
    load_graph_cache,
    save_graph_cache,
)


RDLogger.DisableLog("rdApp.*")


ATOM_FEATURES = {
    "atomic_num": list(range(100)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-1, -2, 1, 2, 0],
    "chiral_tag": [0, 1, 2, 3],
    "num_Hs": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14


def onek_encoding_unk(value, choices: Sequence) -> List[int]:
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def atom_features(atom: Optional[Chem.rdchem.Atom]) -> List[Union[bool, int, float]]:
    """Build the original project atom feature vector."""

    if atom is None:
        return [0] * ATOM_FDIM

    return (
        onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES["atomic_num"])
        + onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES["degree"])
        + onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
        + onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES["chiral_tag"])
        + onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES["num_Hs"])
        + onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES["hybridization"])
        + [1 if atom.GetIsAromatic() else 0]
        + [atom.GetMass() * 0.01]
    )


def bond_features(bond: Optional[Chem.rdchem.Bond]) -> List[Union[bool, int, float]]:
    """Build the original project bond feature vector."""

    if bond is None:
        return [1] + [0] * (BOND_FDIM - 1)

    bond_type = bond.GetBondType()
    features = [
        0,
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated() if bond_type is not None else 0,
        bond.IsInRing() if bond_type is not None else 0,
    ]
    features += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return features


def _get_atom_positions(mol: Chem.Mol, conf: Chem.Conformer) -> List[List[float]]:
    positions = []
    for index, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            return [[0.0, 0.0, 0.0]] * mol.GetNumAtoms()
        pos = conf.GetAtomPosition(index)
        positions.append([pos.x, pos.y, pos.z])
    return positions


def get_2d_positions(mol: Chem.Mol) -> List[List[float]]:
    AllChem.Compute2DCoords(mol)
    return _get_atom_positions(mol, mol.GetConformer())


def get_mmff_positions(mol: Chem.Mol, num_confs: int = 10) -> List[List[float]]:
    try:
        mol_with_h = Chem.AddHs(mol)
        conf_ids = list(AllChem.EmbedMultipleConfs(mol_with_h, numConfs=num_confs))
        if not conf_ids:
            return get_2d_positions(mol)

        results = AllChem.MMFFOptimizeMoleculeConfs(mol_with_h)
        if not results:
            return get_2d_positions(mol)

        best_result_index = int(np.argmin([result[1] for result in results]))
        best_conf_id = conf_ids[best_result_index]
        mol_no_h = Chem.RemoveHs(mol_with_h)
        return _get_atom_positions(mol_no_h, mol_no_h.GetConformer(id=best_conf_id))
    except Exception:
        return get_2d_positions(mol)


def get_zero_positions(mol: Chem.Mol) -> List[List[float]]:
    return [[0.0, 0.0, 0.0]] * mol.GetNumAtoms()


def get_pos(mol: Chem.Mol, pos_mode: str = "mmff", num_confs: int = 10) -> List[List[float]]:
    if pos_mode == "none":
        return get_zero_positions(mol)
    if pos_mode == "2d":
        return get_2d_positions(mol)
    if pos_mode != "mmff":
        raise ValueError(f"Unsupported pos_mode: {pos_mode}")
    if mol.GetNumAtoms() <= 400:
        return get_mmff_positions(mol, num_confs=num_confs)
    return get_2d_positions(mol)


def attach_targets(data: Data, targets: Optional[Sequence[Optional[float]]] = None) -> Data:
    """Attach task targets and masks to a graph Data object."""

    data = data.clone()
    if targets is None:
        data.y = torch.empty((1, 0), dtype=torch.float)
        data.y_mask = torch.empty((1, 0), dtype=torch.bool)
        return data

    y = []
    mask = []
    for target in targets:
        if target is None:
            y.append(float("nan"))
            mask.append(False)
        else:
            y.append(float(target))
            mask.append(True)
    data.y = torch.tensor(y, dtype=torch.float).unsqueeze(0)
    data.y_mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
    return data


def smiles_to_base_pyg_data(
    smiles: str,
    pos_mode: str = "mmff",
    num_confs: int = 10,
) -> Data:
    """Convert one SMILES string into a PyG Data object without targets or fp_x."""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
    pos = torch.tensor(get_pos(mol, pos_mode=pos_mode, num_confs=num_confs), dtype=torch.float)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        features = bond_features(bond)
        edge_index.extend([[begin, end], [end, begin]])
        edge_attr.extend([features, features])

    if edge_index:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.empty((0, BOND_FDIM), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, pos=pos)
    data.smiles = smiles
    data.num_bonds = torch.tensor([edge_attr_tensor.size(0)], dtype=torch.long)
    data.feature_version = GRAPH_FEATURE_VERSION
    data.pos_mode = pos_mode

    return data


def smiles_to_pyg_data(
    smiles: str,
    targets: Optional[Sequence[Optional[float]]] = None,
    pos_mode: str = "mmff",
    num_confs: int = 10,
) -> Data:
    """Convert one SMILES string into a PyG Data object without fp_x."""

    return attach_targets(
        smiles_to_base_pyg_data(smiles=smiles, pos_mode=pos_mode, num_confs=num_confs),
        targets=targets,
    )


def smiles_to_pyg_data_cached(
    smiles: str,
    targets: Optional[Sequence[Optional[float]]] = None,
    graph_cache_dir: Optional[str] = None,
    pos_mode: str = "mmff",
    num_confs: int = 10,
    overwrite_cache: bool = False,
) -> Data:
    """Convert a SMILES string to PyG Data, reusing a target-free graph cache."""

    base_graph = None
    cache_path = None
    if graph_cache_dir is not None:
        cache_path = graph_cache_path(
            graph_cache_dir=graph_cache_dir,
            smiles=smiles,
            pos_mode=pos_mode,
            num_confs=num_confs,
        )
        if not overwrite_cache:
            base_graph = load_graph_cache(cache_path)

    if base_graph is None:
        base_graph = smiles_to_base_pyg_data(
            smiles=smiles,
            pos_mode=pos_mode,
            num_confs=num_confs,
        )
        if cache_path is not None:
            save_graph_cache(base_graph, cache_path)

    graph = attach_targets(base_graph, targets=targets)
    graph.smiles = smiles
    return graph


def _convert_one_cached(args) -> Data:
    return smiles_to_pyg_data_cached(*args)


def dataset_to_pyg(
    data,
    smiles_index: int = 0,
    graph_cache_dir: Optional[str] = None,
    pos_mode: str = "mmff",
    num_confs: int = 10,
    num_workers: int = 0,
    overwrite_cache: bool = False,
) -> List[Data]:
    """Convert a MoleculeDataset-like object into PyG Data objects."""

    jobs = [
        (
            datapoint.smiles[smiles_index],
            datapoint.targets,
            graph_cache_dir,
            pos_mode,
            num_confs,
            overwrite_cache,
        )
        for datapoint in data
    ]

    if num_workers and num_workers > 1:
        from multiprocessing import get_context

        with get_context("spawn").Pool(num_workers) as pool:
            return list(
                tqdm(
                    pool.imap(_convert_one_cached, jobs),
                    total=len(jobs),
                    desc="Converting SMILES to PyG graphs",
                )
            )

    pyg_data = []
    for job in tqdm(jobs, desc="Converting SMILES to PyG graphs"):
        pyg_data.append(_convert_one_cached(job))
    return pyg_data
