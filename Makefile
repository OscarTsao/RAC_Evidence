PYTHON ?= python

prepare:
	$(PYTHON) scripts/prepare_data.py --cfg configs/dataset.yaml

index:
	$(PYTHON) scripts/build_index.py --cfg configs/bi.yaml

retrieve:
	$(PYTHON) scripts/retrieve_topk.py --cfg configs/bi.yaml

train_ce_sc:
	$(PYTHON) scripts/train_ce_sc.py --cfg configs/ce_sc.yaml

train_ce_pc:
	$(PYTHON) scripts/train_ce_pc.py --cfg configs/ce_pc.yaml

calibrate:
	$(PYTHON) scripts/calibrate.py --cfg configs/calibrate.yaml

infer:
	$(PYTHON) scripts/infer_ce_sc.py --cfg configs/ce_sc.yaml
	$(PYTHON) scripts/infer_ce_pc.py --cfg configs/ce_pc.yaml

build_graph:
	$(PYTHON) scripts/build_graph.py --cfg configs/graph.yaml

train_gnn:
	$(PYTHON) scripts/train_graph.py --cfg configs/graph.yaml

evaluate:
	$(PYTHON) scripts/evaluate.py --cfg configs/evaluate.yaml

test:
	pytest -q

all: prepare index retrieve train_ce_sc train_ce_pc calibrate infer build_graph train_gnn evaluate
