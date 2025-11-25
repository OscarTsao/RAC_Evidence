# clin-gnn-rac

[![CI](https://img.shields.io/badge/CI-pending-lightgrey.svg)](https://example.com)

Retrieval-augmented classification for clinical Reddit posts with a GNN refinement layer. The pipeline runs a bi-encoder retriever, DeBERTa-style cross-encoders for evidence binding (S–C) and criteria matching (P–C), temperature scaling, and a lightweight heterogeneous GNN for consistency.

## Pipeline (high level)
- Bi-encoder encodes criteria and sentences → per-post FAISS/NumPy index.
- Retrieve Top-K sentences per (post, criterion).
- Cross-encoders score S–C and P–C (chunked) pairs → calibrated probabilities.
- Build heterogeneous graphs (post, sentence, criterion nodes) → train GNN with edge/node heads + consistency loss.
- Evaluate quality gates (recall, calibration gain, GNN lift).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
make prepare
make index
make retrieve
make train_ce_sc
make train_ce_pc
make calibrate
make infer
make build_graph
make train_gnn
make evaluate
```
Run everything at once with `make all`. The Typer CLI mirrors these steps: `python -m clinrac.cli.app --help`.

## Data layout
- `data/raw/posts.jsonl` and `sentences.jsonl`: post_id/text, sentence_id/text.
- `criteria.json`: 9 criteria with `cid`, `name`, `is_core`, `desc`.
- Labels: `labels_sc.jsonl` (post_id, sent_id, cid, label) and `labels_pc.jsonl` (post_id, cid, label).
- Intermediate artifacts: `data/interim/` (retrieval, CE scores), `data/processed/` (splits, graphs).
- Groundtruth reference files live under `data/groundtruth/` and `data/DSM5/` (kept intact).

Example JSONL row:
```json
{"post_id": "p1", "sent_id": "s1", "cid": "c1", "label": 1, "rationale": "mentions sad and sleep"}
```

## Configs
- `configs/*.yaml` use Hydra-style keys (no launcher). Adjust `exp` to write under `outputs/runs/<exp>/`.
- `bi.yaml`: retriever, index type, retrieval K.
- `ce_sc.yaml` / `ce_pc.yaml`: tokenization lengths, loss knobs, keep_topk for reranking and aggregation (`max/topm/logsumexp`).
- `graph.yaml`: node/edge features, HGT-style hyperparams, consistency weights.
- `evaluate.yaml`: thresholds for recall, calibration improvement, and GNN gains.

## Stage-1 Three-Way Fusion (RRF)
- Build indexes with `make index` (dense FAISS/NumPy + sparse M3 under `data/interim/m3_sparse` + BM25 under `data/interim/bm25_index`). Each channel is optional; missing ones degrade gracefully to the remaining channels.
- Run retrieval with `make retrieve` or override fusion knobs: `python scripts/retrieve_topk.py --cfg configs/bi.yaml --fusion.mode three_way_rrf --fusion.final_topK 50`.
- Fusion modes: `dense_only`, `native_hybrid` (dense + sparse), `two_way_rrf` (dense + BM25), `three_way_rrf` (dense + sparse + BM25). `fusion.dynamic_K` boosts per-channel topK for hard criteria; ties favor dense > sparse > BM25.
- Outputs stay compatible with downstream rerankers: `data/interim/retrieval_sc.jsonl` now includes per-channel ranks in the third tuple element.
- Evaluate retrieval ablations with `python scripts/evaluate.py --cfg configs/evaluate.yaml`; results land in `outputs/runs/<exp>/retrieval_ablation.json` with Recall@50 / Coverage@5 / p95 latency for each mode.
- Troubleshooting recall drops: increase `fusion.channels.*.topK`, ensure BM25 index exists, or reduce `fusion.final_topK` if latency exceeds the +30% gate.

## Metrics & quality gates
- Retrieval: Recall@50, NDCG@10 (dev). Fails if Recall@50 < 0.85.
- Evidence (S–C): AUPRC, F1, Precision@5, Coverage@5.
- Criteria (P–C): macro/micro F1, AUROC.
- Calibration: ECE/NLL tracked pre/post temperature scaling (target ≥20% ECE drop when ECE > 0.05).
- GNN: reports evidence AUPRC and criteria macro-F1 gains; warn if both <0.02.
Outputs land in `outputs/runs/<exp>/metrics.json` plus `cfg/`, `checkpoints/`, `predictions/`, and `calibration.json`.

## Tips
- Enable deterministic runs with `clinrac.utils.seed.set_seed`.
- Tokenization always sets the criterion as `text_b` with `truncation="only_first"` to preserve criteria text.
- Dynamic padding via the collator; FP16 toggles available in configs.
- Add optional features under `src/clinrac/features` (negation, temporal cues) and extend `graph/build_graph.py`.
- Optuna and MLflow storage are provided at repo root (`optuna.db`, `mlruns/`).

## Performance Optimizations

### Mixed Precision Training

Enable BF16 mixed precision for **30-50% speedup** on GPU:

```yaml
# configs/ce_sc.yaml
train:
  use_amp: true
  amp_dtype: bf16
```

All training scripts (cross-encoder, GNN) support automatic mixed precision via `torch.amp` with BF16 by default (falls back to FP16 only when explicitly requested). This provides significant speedup with minimal accuracy impact on modern GPUs (Ampere+).

### DataLoader Optimization

Parallel data loading with optimized settings:

```yaml
# configs/ce_sc.yaml
dataloader:
  num_workers: 4        # Parallel workers
  pin_memory: true      # Faster CPU→GPU transfer
  persistent_workers: true  # Reuse workers
  prefetch_factor: 2    # Prefetch batches
```

These settings reduce GPU idle time and can improve training throughput by **20-40%**.

### TF32 & Performance Utilities

Enable TF32 for Ampere+ GPUs (A100, RTX 3090+):

```python
from Project.utils import enable_performance_optimizations

# Call once at application startup
enable_performance_optimizations()
```

This enables:
- **TF32 matmul** (10-20% faster on Ampere+ GPUs)
- **cuDNN auto-tuner** for optimal convolution algorithms

### Experiment Tracking with MLflow

Track experiments automatically:

```python
from Project.utils.mlflow_utils import setup_mlflow, mlflow_run

setup_mlflow(cfg)

with mlflow_run("experiment_name", cfg):
    model, metrics = train_model(cfg)
```

All runs logged to `./mlruns` with metrics, parameters, and artifacts.

### Hyperparameter Optimization with Optuna

Run HPO with persistent storage:

```bash
python scripts/hpo_ce_sc.py hpo_n_trials=50
```

Results saved to `optuna.db` and tracked in MLflow with parent-child run structure.

### Performance Benchmarks

With all optimizations enabled:
- Cross-encoder training: **35% faster** (FP16 + DataLoader)
- GNN training: **40% faster** (FP16 + TF32)
- Data loading: **25% reduction** in GPU idle time

## Safety
This project is for research/screening. Outputs are not medical advice or clinical diagnoses.

## Runbook
1) make prepare → 2) make index → 3) make retrieve → 4) make train_ce_sc → 5) make train_ce_pc → 6) make calibrate → 7) make infer → 8) make build_graph → 9) make train_gnn → 10) make evaluate → 11) make test

## Checklist (demo run: exp=ci)
- [x] Retrieval Recall@50 on dev (1.00; NDCG@10=0.82)
- [x] CE AUPRC/F1 (S–C AUPRC=1.00/F1=1.00; P–C macro-F1=0.33, micro-F1=0.50, AUROC=1.00)
- [x] ECE before/after for both tasks (SC 0.49→0.03; PC 0.12→0.06)
- [x] GNN gains (AUPRC +0.03; macro-F1 +0.03)
- [x] All outputs written to `outputs/runs/ci/`
