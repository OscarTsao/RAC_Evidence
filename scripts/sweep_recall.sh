#!/usr/bin/env bash
set -euo pipefail

CFG=configs/retrieval/k_selection.yaml
BI_CFG=configs/bi.yaml
EXP=
DEV_MANIFEST=

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cfg) CFG=$2; shift 2;;
    --bi_cfg) BI_CFG=$2; shift 2;;
    --exp) EXP=$2; shift 2;;
    --dev_manifest) DEV_MANIFEST=$2; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

CMD=(PYTHONPATH=src python -m Project.retrieval.sweep_k --cfg "$CFG" --bi_cfg "$BI_CFG")
[[ -n "$EXP" ]] && CMD+=(--exp "$EXP")
[[ -n "$DEV_MANIFEST" ]] && CMD+=(--dev_manifest "$DEV_MANIFEST")

eval "${CMD[@]}"
