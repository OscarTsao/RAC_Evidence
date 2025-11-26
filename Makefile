PYTHON ?= python
EXP ?= demo
MAMBA ?= mamba
DEV_MANIFEST ?=
BI_CFG ?= configs/bi.yaml
CE_SC_CFG ?= configs/ce_sc.yaml
CE_PC_CFG ?= configs/ce_pc.yaml
RERANKER_CFG ?= configs/evidence_reranker.yaml
RUNTIME_CFG ?= configs/retrieval/runtime.yaml
export PYTHONPATH := src
export LD_LIBRARY_PATH := $(CONDA_PREFIX)/lib:$(LD_LIBRARY_PATH)

.PHONY: help
help:
	@echo "==================================================================="
	@echo "  RAC Evidence Pipeline - Available Commands"
	@echo "==================================================================="
	@echo ""
	@echo "Setup:"
	@echo "  make setup              Install dependencies"
	@echo "  make prepare            Prepare dataset"
	@echo ""
	@echo "Retrieval:"
	@echo "  make index              Build FAISS index"
	@echo "  make retrieve           Retrieve top-K candidates"
	@echo "  make sweep_recall       Sweep K values for recall"
	@echo "  make select_k           Select optimal K per criterion"
	@echo ""
	@echo "Evidence Training:"
	@echo "  make train_evidence_5fold    Train 5-fold Evidence CE"
	@echo "  make sc_5fold                STRICT 5-fold pipeline (recommended)"
	@echo "  make sc_5fold_dry            Dry run for testing"
	@echo "  make acceptance_checks       Run quality gates"
	@echo ""
	@echo "Real Dev Shortcuts (EXP=real_dev):"
	@echo "  make real_dev_retrieve       Retrieve for real_dev"
	@echo "  make real_dev_sweep          Sweep K for real_dev"
	@echo "  make real_dev_train_5fold    Train 5-fold for real_dev"
	@echo "  make real_dev_sc_5fold       STRICT pipeline for real_dev"
	@echo "  make real_dev_acceptance     Check quality gates for real_dev"
	@echo "  make real_dev_full_pipeline  Run complete pipeline for real_dev"
	@echo ""
	@echo "Parameters (can override):"
	@echo "  EXP=<name>                   Experiment name (default: demo)"
	@echo "  BI_CFG=<path>                Bi-encoder config (default: configs/bi.yaml)"
	@echo "  RERANKER_CFG=<path>          Reranker config (default: configs/evidence_reranker.yaml)"
	@echo ""
	@echo "==================================================================="

setup:
	$(MAMBA) install -y python=3.10 pip pytorch==2.3 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
	pip install -f https://data.pyg.org/whl/torch-2.3.0+cu121.html torch_scatter torch_sparse torch_geometric
	pip install -e '.[dev,retrieval]'

prepare:
	$(PYTHON) scripts/prepare_data.py --cfg configs/dataset.yaml

index:
	$(PYTHON) scripts/build_index.py --cfg $(BI_CFG)

retrieve:
	$(PYTHON) scripts/retrieve_topk.py --cfg $(BI_CFG)

sweep_recall:
	@if [ -n "$(DEV_MANIFEST)" ]; then DEV_ARG="--dev_manifest $(DEV_MANIFEST)"; else DEV_ARG=""; fi; \
	PYTHONPATH=src $(PYTHON) -m Project.retrieval.sweep_k --cfg configs/retrieval/k_selection.yaml --bi_cfg $(BI_CFG) --exp $(EXP) $$DEV_ARG

select_k:
	PYTHONPATH=src $(PYTHON) -m Project.retrieval.dynamic_k --sweep_json outputs/runs/$(EXP)/retrieval_sweep/sweep_metrics.json --cfg configs/retrieval/k_selection.yaml --runtime_out $(RUNTIME_CFG) --exp $(EXP)

retriever_ft:
	PYTHONPATH=src $(PYTHON) -m Project.retrieval.finetune_bge_m3 --exp $(EXP)

train_ce_sc:
	$(PYTHON) scripts/train_ce_sc.py --cfg $(CE_SC_CFG)

train_ce_pc:
	$(PYTHON) scripts/train_ce_pc.py --cfg $(CE_PC_CFG)

train_evidence:
	PYTHONPATH=src $(PYTHON) scripts/train_bge_reranker.py --cfg $(RERANKER_CFG) --runtime_cfg $(RUNTIME_CFG) --exp $(EXP)

train_evidence_5fold:
	@echo "Training 5-fold Evidence CE with STRICT parameters:"
	@echo "  - k_train=100, k_infer=20 (from config)"
	@echo "  - Negatives: 80% hard / 10% random / 10% cross-post"
	@echo "  - Cross-post cap: ≤20% (STRICT)"
	PYTHONPATH=src $(PYTHON) scripts/train_evidence_5fold.py --cfg $(RERANKER_CFG) --exp $(EXP)

acceptance_checks:
	PYTHONPATH=src $(PYTHON) scripts/run_acceptance_checks.py --exp $(EXP)

sc_5fold:
	@echo "==================================================================="
	@echo "  STRICT 5-Fold Evidence CE Pipeline (Unified Prompt Compliant)"
	@echo "==================================================================="
	@echo "Parameters:"
	@echo "  • K_train=100, K_infer=20 (STRICT)"
	@echo "  • Negatives: 80% hard / 10% random / 10% cross-post (cap ≤20%)"
	@echo "  • Same-post retrieval enforced"
	@echo "  • Per-fold temperature calibration"
	@echo "  • Quality gates with ACCEPTED/FAILED verdict"
	@echo "==================================================================="
	bash scripts/sc_5fold.sh --exp $(EXP)

sc_5fold_dry:
	@echo "Running STRICT 5-Fold pipeline in DRY RUN mode..."
	bash scripts/sc_5fold.sh --exp $(EXP) --dry_run

build_graph:
	$(PYTHON) scripts/build_graph.py --cfg configs/graph.yaml

train_gnn:
	$(PYTHON) scripts/train_graph.py --cfg configs/graph.yaml

evaluate:
	$(PYTHON) scripts/evaluate.py --cfg configs/evaluate.yaml

test:
	pytest -q

# Real dev experiment shortcuts (EXP=real_dev)
real_dev_index:
	$(MAKE) index EXP=real_dev BI_CFG=configs/bi_real.yaml

real_dev_retrieve:
	$(MAKE) retrieve EXP=real_dev BI_CFG=configs/bi_real.yaml

real_dev_sweep:
	$(MAKE) sweep_recall EXP=real_dev BI_CFG=configs/bi_real.yaml

real_dev_select_k:
	$(MAKE) select_k EXP=real_dev

real_dev_train_5fold:
	$(MAKE) train_evidence_5fold EXP=real_dev RERANKER_CFG=configs/evidence_reranker.yaml

real_dev_sc_5fold:
	$(MAKE) sc_5fold EXP=real_dev

real_dev_sc_5fold_dry:
	$(MAKE) sc_5fold_dry EXP=real_dev

real_dev_acceptance:
	$(MAKE) acceptance_checks EXP=real_dev

real_dev_full_pipeline:
	@echo "==================================================================="
	@echo "  Running FULL Evidence Pipeline for real_dev"
	@echo "==================================================================="
	@echo "Steps: retrieve → train_5fold → acceptance"
	@echo "==================================================================="
	$(MAKE) real_dev_retrieve
	$(MAKE) real_dev_train_5fold
	$(MAKE) real_dev_acceptance

all: prepare index retrieve train_ce_sc train_ce_pc calibrate infer build_graph train_gnn evaluate
