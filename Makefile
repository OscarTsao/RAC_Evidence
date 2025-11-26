PYTHON ?= python
EXP ?= demo
MAMBA ?= mamba
DEV_MANIFEST ?=
DATA_CFG ?= configs/dataset.yaml
BI_CFG ?= configs/bi.yaml
CE_SC_CFG ?= configs/ce_sc.yaml
CE_PC_CFG ?= configs/ce_pc.yaml
CALIBRATE_CFG ?= configs/calibrate.yaml
GRAPH_CFG ?= configs/graph.yaml
EVAL_CFG ?= configs/evaluate.yaml
RERANKER_CFG ?= configs/evidence_reranker.yaml
RUNTIME_CFG ?= configs/retrieval/runtime.yaml
export PYTHONPATH := src
export LD_LIBRARY_PATH := $(CONDA_PREFIX)/lib:$(LD_LIBRARY_PATH)

.PHONY: help setup prepare index retrieve sweep_recall select_k retriever_ft \
	train_ce_sc train_ce_pc calibrate infer infer_sc infer_pc hpo_ce_sc hpo_ce_pc \
	train_evidence train_evidence_5fold acceptance_checks sc_5fold sc_5fold_dry \
	build_graph train_gnn evaluate test real_dev_index real_dev_retrieve \
	real_dev_sweep real_dev_select_k real_dev_train_5fold real_dev_sc_5fold \
	real_dev_sc_5fold_dry real_dev_acceptance real_dev_full_pipeline all

help:
	@echo "==================================================================="
	@echo "  clin-gnn-rac - Available Commands"
	@echo "==================================================================="
	@echo ""
	@echo "Setup & data:"
	@echo "  make setup              Install dependencies"
	@echo "  make prepare            Prepare dataset splits ($(DATA_CFG))"
	@echo ""
	@echo "Main pipeline (EXP=$(EXP)):"
	@echo "  make index              Train retriever + build indexes"
	@echo "  make retrieve           Retrieve top-K candidates"
	@echo "  make train_ce_sc        Train sentence-criterion cross-encoder"
	@echo "  make train_ce_pc        Train post-criterion cross-encoder"
	@echo "  make infer              Run CE inference (SC + PC)"
	@echo "  make calibrate          Fit temperature scaling from inference outputs"
	@echo "  make build_graph        Build heterogeneous graphs"
	@echo "  make train_gnn          Train heterogeneous GNN"
	@echo "  make evaluate           Run quality gates"
	@echo "  make test               Run pytest suite"
	@echo ""
	@echo "Retrieval tuning:"
	@echo "  make sweep_recall       Sweep K values for recall"
	@echo "  make select_k           Select K and update runtime config"
	@echo ""
	@echo "Hyperparameter search:"
	@echo "  make hpo_ce_sc          Optuna HPO for CE-SC (GPU-parallel)"
	@echo "  make hpo_ce_pc          Optuna HPO for CE-PC (GPU-parallel)"
	@echo ""
	@echo "STRICT 5-fold evidence pipeline:"
	@echo "  make train_evidence_5fold    Train 5-fold Evidence CE"
	@echo "  make sc_5fold                Run STRICT pipeline"
	@echo "  make sc_5fold_dry            Dry run for STRICT pipeline"
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
	@echo "==================================================================="

setup:
	$(MAMBA) install -y python=3.10 pip pytorch==2.3 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
	pip install -f https://data.pyg.org/whl/torch-2.3.0+cu121.html torch_scatter torch_sparse torch_geometric
	pip install -e '.[dev,retrieval]'

prepare:
	$(PYTHON) scripts/prepare_data.py --cfg $(DATA_CFG)

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

calibrate:
	$(PYTHON) scripts/calibrate.py --cfg $(CALIBRATE_CFG)

infer: infer_sc infer_pc

infer_sc:
	$(PYTHON) scripts/infer_ce_sc.py --cfg $(CE_SC_CFG)

infer_pc:
	$(PYTHON) scripts/infer_ce_pc.py --cfg $(CE_PC_CFG)

hpo_ce_sc:
	$(PYTHON) scripts/hpo_ce.py --cfg $(CE_SC_CFG) --task sc

hpo_ce_pc:
	$(PYTHON) scripts/hpo_ce.py --cfg $(CE_PC_CFG) --task pc

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
	$(PYTHON) scripts/build_graph.py --cfg $(GRAPH_CFG)

train_gnn:
	$(PYTHON) scripts/train_graph.py --cfg $(GRAPH_CFG)

evaluate:
	$(PYTHON) scripts/evaluate.py --cfg $(EVAL_CFG)

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

all: prepare index retrieve train_ce_sc train_ce_pc infer calibrate build_graph train_gnn evaluate
