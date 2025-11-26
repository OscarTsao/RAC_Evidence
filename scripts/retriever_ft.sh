#!/usr/bin/env bash
set -euo pipefail

EXP=${1:-demo}
EPOCHS=${2:-2}
LR=${3:-2e-5}

PYTHONPATH=src python -m Project.retrieval.finetune_bge_m3 --exp "$EXP" --epochs "$EPOCHS" --lr "$LR"
