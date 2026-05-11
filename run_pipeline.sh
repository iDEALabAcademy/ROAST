#!/usr/bin/env bash
# ============================================================================
#  ROAST end-to-end pipeline
#  Usage:  ./run_pipeline.sh <dataset>     e.g. ./run_pipeline.sh mnist
# ============================================================================
set -e

DATASET="${1:-mnist}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "============================================================"
echo " ROAST pipeline for dataset: ${DATASET}"
echo "============================================================"

echo
echo "[1/3] Reverse-training attack -> worst_model.pth"
python reverse_training_attack.py --dataset "$DATASET"

echo
echo "[2/3] Hamming-distance closest-offset analysis"
python find_closest_offsets.py --dataset "$DATASET"

echo
echo "[3a/3] Progressive attack (cumulative)"
python progressive_attack.py --dataset "$DATASET" --mode cumulative

echo
echo "[3b/3] Progressive attack (independent)"
python progressive_attack.py --dataset "$DATASET" --mode independent

echo
echo "Done. See attack_outputs/${DATASET}/ for results."
