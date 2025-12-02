# Phase 4: S-C → GNN Pipeline - Implementation Summary

**Date**: 2025-11-29
**Experiment**: `real_dev` (real_dev_hpo_refine_trial_8)

## Executive Summary

Successfully implemented a complete Heterogeneous Graph Transformer (HGT) pipeline for post-level criteria prediction using sentence-level evidence from the cross-encoder. The pipeline is fully functional from a technical perspective, with all components working correctly. However, meaningful evaluation is blocked by missing post-criterion (PC) ground truth labels.

### Key Achievement
✅ **Technical Pipeline**: Fully working end-to-end GNN pipeline with BGE-M3 embeddings, message passing, and dual-level prediction

⚠️ **Evaluation Limitation**: Cannot assess performance improvement without PC ground truth labels

---

## Pipeline Architecture

### Component 1: Evidence Model (Completed)
- **Model**: BAAI/bge-reranker-v2-m3 with LoRA fine-tuning
- **OOF Predictions**: 90,693 sentence-criterion pairs
- **Performance**:
  - Macro-F1: **0.756** (calibrated)
  - F1: **0.520** (calibrated)
  - AUPRC: **0.437**
  - Coverage@5: **89.3%**
  - Coverage@10: **96.0%**
  - Coverage@20: **100%**

### Component 2: Graph Construction (Completed)
- **Total Graphs**: 1,477 heterogeneous graphs (1 per post)
- **Nodes**:
  - Sentences: 29,888 total (avg ~20 per post)
  - Criteria: 9 (DSM-5 symptoms)
  - Posts: 1 per graph
- **Node Features**: BGE-M3 dense embeddings (1024-dim)
- **Edge Types**:
  - **S-C (sentence → criterion)**: Top-K=10 per (post, cid) based on calibrated probability
    - Features: [logit, prob_cal, dense_score_norm, inv_rank_d, inv_rank_s]
  - **S-S (sentence → sentence)**: Sequential connections
  - **P-S (post → sentence)**: Structural containment
- **Storage**: Individual files (data_0.pt, ..., data_1476.pt) for memory efficiency
- **Metadata**: `outputs/runs/real_dev/graphs/metadata.json`

### Component 3: HGT Model (Completed)
- **Architecture**: 2-layer Heterogeneous Graph Transformer
- **Parameters**: 342,688 trainable parameters
- **Configuration**:
  - Hidden channels: 64
  - Attention heads: 4
  - Dropout: 0.2
  - Mixed precision: BF16
- **Loss Function**:
  - Edge loss (S-C): BCEWithLogitsLoss (λ=1.0)
  - Node loss (P-C): BCEWithLogitsLoss (λ=1.0)
  - Consistency loss: Bidirectional agreement (λ=0.3, margin=0.05)

### Component 4: Training Results
**Training completed in 3 epochs:**

| Epoch | Total Loss | Edge Loss | Node Loss | Consistency Loss |
|-------|-----------|-----------|-----------|-----------------|
| 1 | 0.1217 | 0.0871 | 0.0088 | 0.0859 |
| 2 | 0.0815 | 0.0537 | 0.0000 | 0.0927 |
| 3 | 0.0798 | 0.0508 | 0.0000 | 0.0964 |

**Observations:**
- ✅ Edge loss decreased consistently (0.0871 → 0.0508, -42%)
- ✅ Total loss decreased (0.1217 → 0.0798, -34%)
- ✅ No errors during training, model converged smoothly
- ⚠️ Node loss dropped to 0.0000 (all criterion labels are 0 due to missing PC scores)

---

## Technical Challenges Resolved

### 1. HGTConv Message Passing
**Issue**: 'post' node removed from x_dict after HGTConv
**Root Cause**: Post nodes don't participate in message passing edges
**Solution**: Preserved post embeddings explicitly before/after each HGTConv layer

### 2. Batching with Heterogeneous Graphs
**Issue**: Shape mismatch when expanding post embeddings (4 posts × 36 criteria in batch)
**Root Cause**: Naive broadcast tried to pair posts from different graphs
**Solution**: Used PyG batch indices to only pair (post, criterion) within same graph

### 3. Memory Efficiency
**Approach**: Individual graph file storage instead of single large file
**Impact**: Enables lazy loading, prevents OOM on large datasets

### 4. Edge Normalization
**Feature Engineering**:
- StandardScaler normalization for dense retrieval scores
- Inverse rank features: 1/(rank+1)
- Handles None values gracefully

---

## Current Limitation: Missing PC Ground Truth

### Problem Statement
The graph builder requires post-criterion (PC) labels to train the node-level predictor. During graph construction:

```
WARNING  No PC scores found - will use placeholder values
```

**Impact**:
- All criterion labels default to 0.0 (negative class)
- Node-level predictions collapse to all-zeros
- Evaluation metrics meaningless:
  - F1: 0.0000
  - Precision: 0.0000
  - Recall: 0.0000
  - Macro-F1: 1.0000 (trivial - all negatives correctly predicted)

### Root Cause
The graph builder searches for PC scores in:
1. `outputs/runs/real_dev/pc/oof_predictions.jsonl`
2. `outputs/runs/real_dev_pc/oof_predictions.jsonl`

Neither file exists because PC layer training was never completed/saved in the `real_dev` experiment.

---

## Files Created

### Code (src/Project/graph/)
1. **`build_hetero.py`** (371 lines) - Graph construction with BGE-M3
2. **`hgt_model.py`** (200 lines) - HGT architecture
3. **`train_hgt.py`** (347 lines) - Training pipeline
4. **`dataset.py`** (86 lines) - Memory-efficient PyG dataset

### Scripts (scripts/)
1. **`build_heterograph.py`** - Wrapper for graph building
2. **`train_hgt.py`** - Wrapper for HGT training

### Outputs (outputs/runs/real_dev/)
1. **`graphs/`** - 1,477 heterogeneous graph files + metadata.json
2. **`gnn/best_model.pt`** - Trained HGT model (4.1 MB)
3. **`gnn/metrics.json`** - Training metrics

### Logs
1. **`outputs/graph_build.log`** - Graph construction log
2. **`outputs/hgt_training.log`** - HGT training log

---

## Next Steps

### Option A: Train PC Layer First (Recommended)
1. Train post-criterion classification model using cross-encoder
2. Generate OOF predictions for PC layer
3. Rebuild graphs with PC scores as criterion labels
4. Retrain HGT with meaningful supervision
5. Evaluate improvement over CE baseline

### Option B: Use Heuristic Labels
1. Infer PC labels from S-C predictions (e.g., max sentence probability per criterion)
2. Rebuild graphs with heuristic labels
3. Train HGT as weak supervision
4. Caveat: May not improve over CE baseline

### Option C: Unsupervised/Semi-Supervised
1. Train HGT with only edge-level supervision (S-C)
2. Use consistency loss to propagate evidence to post level
3. Evaluate qualitatively on sample posts
4. Caveat: No ground truth for validation

---

## Technical Validation

### What Works ✅
1. Graph construction with BGE-M3 embeddings
2. Top-K edge selection per (post, cid)
3. Normalized retrieval features
4. HGT message passing across S-C and S-S edges
5. Batch-aware node prediction
6. Mixed precision training (BF16)
7. Loss convergence and gradient flow

### What's Blocked ⚠️
1. Meaningful node-level evaluation (needs PC labels)
2. Comparison to CE baseline (needs PC predictions)
3. Per-criterion improvement analysis (needs PC ground truth)

---

## Comparison to Evidence CE Baseline

### Evidence (S-C) Layer Performance
**Cross-Encoder Performance** (from `evidence_metrics.json`):
- **Overall Macro-F1**: 0.756 (calibrated)
- **Overall F1**: 0.520 (calibrated)
- **AUPRC**: 0.437
- **Coverage@5**: 89.3%

**Per-Criterion Breakdown** (Macro-F1):
1. c9 (Suicidal Thoughts): 0.322 (best)
2. c5 (Psychomotor): 0.247
3. c6 (Fatigue): 0.261
4. c1 (Depressed Mood): 0.204
5. c2 (Anhedonia): 0.277
6. c4 (Sleep Issues): 0.232
7. c3 (Appetite Change): 0.176
8. c7 (Worthlessness): 0.144 (worst)
9. c8 (Cognitive Issues): 0.022 (worst)

### GNN Performance
**Cannot compute** - missing PC ground truth labels

**Expected Impact** (once PC labels available):
- GNN should improve over CE baseline by leveraging:
  - Global post-level context
  - Inter-sentence dependencies
  - Criterion relationships
- Target improvement: +0.02 macro-F1

---

## Conclusion

### Technical Success
The S-C → GNN pipeline is **fully functional** from an implementation perspective:
- ✅ All code modules working correctly
- ✅ Graph construction with semantic embeddings
- ✅ HGT training converged successfully
- ✅ Memory-efficient lazy loading
- ✅ Mixed precision training

### Evaluation Blocked
- ⚠️ PC ground truth labels required for meaningful evaluation
- ⚠️ Cannot assess improvement over CE baseline yet

### Recommended Action
**Train PC layer** to generate post-criterion OOF predictions, then rebuild graphs and retrain HGT for final comparison.

---

## Appendix: Configuration

### Graph Builder (`configs/graph_real.yaml`)
```yaml
graph:
  edges:
    supports:
      topk_per_cid: 10  # Top-K S-C edges per (post, cid)

model:
  name: BAAI/bge-m3
  hidden: 64
  out_channels: 128
  num_heads: 4
  layers: 2
  dropout: 0.2

loss:
  lambda_edge: 1.0
  lambda_node: 1.0
  lambda_cons: 0.3
  cons_margin: 0.05

training:
  lr: 1e-3
  batch_size_posts: 4
  epochs: 3
  use_amp: true
  amp_dtype: bf16
```

### Evidence CE (`configs/evidence_reranker.yaml`)
```yaml
model: BAAI/bge-reranker-v2-m3
lora:
  r: 16
  alpha: 16
  dropout: 0.086

training:
  n_folds: 5
  batch_size: 16
  gradient_accumulation_steps: 4
  learning_rate: 2.395e-05
  weight_decay: 0.01
  max_epochs: 50
  early_stopping_patience: 5
```

---

**Generated**: 2025-11-29 23:37
**Pipeline Status**: ✅ Technical Implementation Complete, ⏸️ Evaluation Pending PC Labels
