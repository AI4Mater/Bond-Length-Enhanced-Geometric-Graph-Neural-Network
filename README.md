# Bond-Length-Enhanced Geometric Graph Neural Network

This repository contains code and datasets for molecular property prediction with a bond-length-enhanced geometric graph neural network. The workflow reads molecular CSV files, converts SMILES strings into PyTorch Geometric graph data with 3D geometric information, and trains graph neural network models for regression or classification tasks.

## Project Structure

```text
.
├── data/                  # Molecular benchmark datasets in CSV format
├── model/                 # Data preprocessing, model, training, and evaluation code
├── run_esol_train.sh      # Example ESOL training script
├── .gitignore
└── README.md
```

## Datasets

The `data/` folder currently includes:

- `bace.csv`
- `bbbp.csv`
- `esol.csv`
- `hiv.csv`
- `qm7.csv`
- `qm8.csv`
- `qm9.csv`
- `tox21.csv`

Each dataset should contain a SMILES column and one or more target columns. For example, the ESOL script expects:

- SMILES column: `smiles`
- Target column: `logSolubility`

## Main Components

- `model/data_processing.py`: reads CSV files, filters invalid SMILES, splits datasets, normalizes regression targets, and saves processed PyG graph data.
- `model/graph_conversion.py`: converts molecules into graph objects with atom, bond, and geometric features.
- `model/models.py`: defines the bond-length-enhanced graph neural network model and optional FFKN/KAN layers.
- `model/training.py`: handles training, validation, checkpointing, and test evaluation.
- `model/evaluation.py`: computes regression and classification metrics.
- `model/hyperparameter_optimization.py`: runs hyperparameter search with Hyperopt.

## Environment

The code depends on common molecular graph learning packages, including:

- Python
- PyTorch
- PyTorch Geometric
- RDKit
- NumPy
- tqdm
- Hyperopt, if running hyperparameter optimization

Install the versions compatible with your CUDA/PyTorch environment. RDKit is commonly installed with Conda:

```bash
conda install -c conda-forge rdkit
```

## Running ESOL Training

An example script is provided:

```bash
bash run_esol_train.sh
```

Before running it on a new machine, check and update the absolute paths in `run_esol_train.sh`, especially:

- `PROJECT_ROOT`
- `PYTHON`
- `data_path`
- `processed_dir`
- `save_dir`

The script preprocesses ESOL if processed graph files are not found, then trains the model and saves outputs such as:

- `model.pt`
- `history.json`
- `results.json`

## Minimal Python Usage

```python
from model.data_processing import preprocess_csv_to_pyg
from model.config import TrainConfig
from model.training import train_model

processed_dir = "runs/esol/processed"
save_dir = "runs/esol/train_seed0"

preprocess_csv_to_pyg(
    data_path="data/esol.csv",
    save_dir=processed_dir,
    smiles_columns="smiles",
    target_columns=["logSolubility"],
    split_type="random",
    split_sizes=(0.8, 0.1, 0.1),
    seed=0,
    dataset_type="regression",
    pos_mode="mmff",
    num_confs=10,
    num_workers=0,
    resume=True,
)

config = TrainConfig(
    processed_data_dir=processed_dir,
    save_dir=save_dir,
    dataset_type="regression",
    num_tasks=1,
    metric="rmse",
    epochs=30,
    batch_size=50,
    learning_rate=1e-3,
    hidden_size=300,
    depth=3,
    num_heads=8,
)

result = train_model(config)
print(result)
```

## Notes

- Generated files such as `__pycache__/` and `.DS_Store` are ignored by Git.
- Large raw datasets and generated training artifacts should be kept out of the repository unless they are intentionally versioned.
- For reproducibility, record the Python, PyTorch, PyTorch Geometric, and RDKit versions used for experiments.

