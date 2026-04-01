#!/usr/bin/env bash
# Trains the autoencoder with tissue mask, then rebuilds the ae16 encoded dataset.
# Run from the ML_PIPELINE_G17_AITFMD directory:
#   bash scripts/run_ae_overnight.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

CONFIG="configs/preprocessing/autoencoder.yaml"

echo "================================================"
echo "[1/2] Trener autoencoder med tissue mask..."
echo "      Config: $CONFIG"
echo "================================================"
python scripts/preprocessing/run/train_autoencoder.py --config "$CONFIG"

echo ""
echo "================================================"
echo "[2/2] Bygger ae16-dataset med tissue mask..."
echo "      Config: $CONFIG"
echo "================================================"
python scripts/preprocessing/build/build_ae_dataset.py --config "$CONFIG"

echo ""
echo "================================================"
echo "AE pipeline ferdig."
echo "================================================"
