#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PROJECT_ROOT="/Users/happy_jiao/Documents/研究生/6-模型"
PYTHON="/opt/anaconda3/envs/ame/bin/python"

cd "$PROJECT_ROOT"

"$PYTHON" - <<'PY'
import os

from E4.data_processing import preprocess_csv_to_pyg
from E4.config import TrainConfig
from E4.training import train_model

data_path = "/Users/happy_jiao/Documents/研究生/6-模型/2-allModel/EPIC_Geo_Attn/1-data/esol.csv"
processed_dir = "/Users/happy_jiao/Documents/研究生/6-模型/E4/runs/esol/processed"
save_dir = "/Users/happy_jiao/Documents/研究生/6-模型/E4/runs/esol/train_seed0"

required_processed_files = ["train.pt", "val.pt", "test.pt", "scaler.pkl"]
has_processed_data = all(
    os.path.exists(os.path.join(processed_dir, file_name))
    for file_name in required_processed_files
)

if has_processed_data:
    print(f"Found processed data in {processed_dir}. Skip preprocessing.", flush=True)
else:
    print("Processed data not found. Start preprocessing.", flush=True)
    preprocess_csv_to_pyg(
        data_path=data_path,
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
    print("Preprocessing done.", flush=True)

config = TrainConfig(
    processed_data_dir=processed_dir,
    save_dir=save_dir,
    dataset_type="regression",
    num_tasks=1,
    metric="rmse",
    epochs=30,
    batch_size=50,
    learning_rate=1e-3,
    weight_decay=0.0,
    seed=0,
    device="cpu",
    num_workers=0,
    hidden_size=300,
    depth=3,
    dropout=0.0,
    ffn_hidden_size=300,
    ffn_num_layers=2,
    num_heads=8,
)

print(f"Start training. processed_dir={processed_dir}", flush=True)
print(f"Save training outputs to {save_dir}", flush=True)
print(
    "Hyperparameters: "
    f"epochs={config.epochs}, batch_size={config.batch_size}, "
    f"learning_rate={config.learning_rate}, hidden_size={config.hidden_size}, "
    f"depth={config.depth}, num_heads={config.num_heads}, device={config.device}",
    flush=True,
)

result = train_model(config)
print(result)
PY
