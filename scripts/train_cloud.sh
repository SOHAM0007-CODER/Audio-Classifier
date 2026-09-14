#!/usr/bin/env bash
# scripts/train_cloud.sh
#
# Cloud GPU training for the Multi-Task AST classifier on any Linux GPU machine (RunPod / Lambda / Vast.ai).
# Steps: environment check + install, data preparation + verification, training, test evaluation + inference demo.
#
# Usage (from anywhere; runs in the repository root):
#   bash scripts/train_cloud.sh                                        # all steps
#   GTZAN_DIR=/workspace/genres_original bash scripts/train_cloud.sh   # use an existing GTZAN copy
#   bash scripts/train_cloud.sh --batch_size 8 --grad_accum_steps 2    # extra flags go to src.train
#
# Environment variables:
#   CONFIG      training config (default: configs/cloud_train.yaml)
#   GTZAN_DIR   GTZAN genres_original directory (default: download from Kaggle with kagglehub)
#   SKIP_TESTS  set to 1 to skip the pre-flight unit tests
# Re-running the script resumes training from <output_dir>/latest_checkpoint.pt.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/cloud_train.yaml}"

echo "=== 1/4 Environment check and installation ==="
nvidia-smi
python -m pip install -q -r requirements.txt kagglehub
python -c "import torch, transformers; assert torch.cuda.is_available(), 'No CUDA GPU available'; print('torch', torch.__version__, '| transformers', transformers.__version__, '|', torch.cuda.get_device_name(0))"
if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
  python -m pytest -q tests/test_model.py tests/test_train.py
fi

echo "=== 2/4 Data preparation and verification ==="
if [[ ! -f data/manifests/train.csv ]]; then
  if [[ -z "${GTZAN_DIR:-}" ]]; then
    GTZAN_DIR="$(python -c "
from pathlib import Path
import kagglehub
path = Path(kagglehub.dataset_download('andradaolteanu/gtzan-dataset-music-genre-classification'))
print(sorted(path.rglob('genres_original'))[0])
" | tail -n 1)"
  fi
  echo "GTZAN: $GTZAN_DIR"
  python scripts/prepare_manifests.py --gtzan-dir "$GTZAN_DIR"
fi
python -m pytest -q tests/test_dataset.py

echo "=== 3/4 Training ==="
NUM_WORKERS="$(python -c "import os; print(min(4, os.cpu_count() or 1))")"
python -m src.train --config "$CONFIG" --resume auto --num_workers "$NUM_WORKERS" "$@"

echo "=== 4/4 Test evaluation and inference demo ==="
OUTPUT_DIR="$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$CONFIG")"
CHECKPOINT="$OUTPUT_DIR/best_model.pt"
python -m src.evaluate --checkpoint "$CHECKPOINT" --manifest data/manifests/test.csv --fp16 \
  --num_workers "$NUM_WORKERS" --output "$OUTPUT_DIR/test_metrics.json"

for DOMAIN in music env; do
  DOMAIN_ID="$([[ "$DOMAIN" == music ]] && echo 0 || echo 1)"
  CLIP="$(python -c "import pandas as pd; df = pd.read_csv('data/manifests/test.csv'); print(df[df['domain'] == $DOMAIN_ID]['filepath'].iloc[0])")"
  echo "--- $CLIP ---"
  python -m src.predict --audio "$CLIP" --checkpoint "$CHECKPOINT" --domain "$DOMAIN" --top_k 5
done
