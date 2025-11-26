#!/usr/bin/env bash
set -euo pipefail

SWEEP_JSON=outputs/runs/demo/retrieval_sweep/sweep_metrics.json
CFG=configs/retrieval/k_selection.yaml
RUNTIME_OUT=configs/retrieval/runtime.yaml
EXP=
PREFER_MIN_K=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sweep_json) SWEEP_JSON=$2; shift 2;;
    --cfg) CFG=$2; shift 2;;
    --runtime_out) RUNTIME_OUT=$2; shift 2;;
    --exp) EXP=$2; shift 2;;
    --prefer_min_k) PREFER_MIN_K=true; shift 1;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

CMD=(PYTHONPATH=src python -m Project.retrieval.dynamic_k --sweep_json "$SWEEP_JSON" --cfg "$CFG" --runtime_out "$RUNTIME_OUT")
[[ -n "$EXP" ]] && CMD+=(--exp "$EXP")
[[ "$PREFER_MIN_K" == "true" ]] && CMD+=(--prefer_min_k)

eval "${CMD[@]}"
