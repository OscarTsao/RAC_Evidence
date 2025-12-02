# Phase 4: PC Layer Critical Issues - Diagnosis & Solution

**Date**: 2025-12-01
**Status**: 🔴 **CRITICAL ISSUES FOUND**

---

## Executive Summary

The Phase 4 pipeline completed technically, but discovered **critical failures** in the PC (Post-Criterion) layer:

1. ✅ PC training ran for 4 hours (Macro-F1: 0.0875)
2. ❌ **Trained model outputs identical predictions for ALL inputs** (prob=0.5242)
3. ❌ **Training used WRONG labels** with only 89% agreement vs ground truth
4. ❌ **1,468 label errors** (11% of data) led to degenerate model

---

## Critical Findings

### 1. Degenerate Model Output

**Symptom**: All predictions have the same value:
- Logit: 0.0970
- Probability: 0.5242
- AUC: 0.5000 (random guessing)

**Evidence**:
```python
# Tested with completely different inputs:
"I feel very sad and depressed [SEP] Depressed mood" → prob: 0.5242
"I sleep well and feel energetic [SEP] Depressed mood" → prob: 0.5242
```

**Impact**:
- Model learned nothing discriminative
- Cannot be used for meaningful PC predictions
- GNN trained with useless supervision signal

### 2. Wrong Training Labels

**Training Data**: `data/processed/labels_pc.jsonl`
- 13,293 samples
- 12,000 negatives, 1,293 positives

**Ground Truth**: `data/groundtruth/criteria_matching_groundtruth.csv`
- 13,356 samples (A.1-A.9 only)
- 12,063 negatives, 1,293 positives

**Comparison**:
- **Agreement**: 11,825 / 13,293 = 88.96%
- **Mismatches**: 1,468 samples (11.04% error rate)
- Same number of positives, but different label assignments

**Conclusion**: `labels_pc.jsonl` is NOISY/INCORRECT - possibly auto-generated from sentence-level aggregation rather than expert annotations.

### 3. Wasted Compute

- **4 hours 5 minutes** training on wrong labels
- **5 fold models** all degenerate
- **19 minutes** generating useless predictions
- **2 minutes** graph reconstruction with bad labels
- **1 minute** HGT training with bad supervision

---

## Root Cause Analysis

### Why Did Training Report 8.9% Macro-F1?

The reported Macro-F1 (0.0875-0.0898) during training matches the label noise level (11%). This suggests:
1. Model learned to fit the noisy labels during training
2. CV folds showed consistent but low performance
3. No validation against TRUE ground truth
4. Model converged to a degenerate solution

### Why Are All Predictions Identical?

Possible explanations:
1. **Model collapse during training**: Loss function may have driven all weights toward producing constant output
2. **Extreme class imbalance**: 90% negatives led to "always predict negative" strategy
3. **Learning rate too high**: Caused weights to diverge to degenerate state
4. **No regularization**: Model found trivial solution

The exact mechanism requires deeper investigation of training dynamics.

---

## Impact Assessment

### What Works ✅
- Evidence (S-C) layer: Macro-F1 0.756 (verified with ground truth)
- Graph construction infrastructure
- HGT training infrastructure
- Evaluation scripts and metrics

### What's Broken ❌
- PC layer models (all 5 folds)
- PC predictions (all identical)
- Graph PC labels (derived from broken predictions)
- HGT supervision signal (PC labels useless)
- Pipeline evaluation (metrics meaningless)

---

## Recommended Solution

### Option 1: Retrain PC Layer with Correct Labels (RECOMMENDED)

**Steps**:
1. Convert `criteria_matching_groundtruth.csv` to JSONL format compatible with training
   - Map A.1→c1, A.2→c2, ..., A.9→c9
   - Replace `data/processed/labels_pc.jsonl` with correct labels
2. Retrain PC layer (5-fold CV) - 4 hours
3. Generate new PC predictions with correct models
4. Rebuild graphs with corrected PC labels
5. Retrain HGT with meaningful supervision
6. Re-evaluate full pipeline

**Expected Outcomes**:
- PC Macro-F1: 0.20-0.40 (realistic for this task)
- Meaningful PC predictions with variance
- Useful supervision signal for HGT
- Valid pipeline evaluation

**Time**: ~6 hours total (4h training + 2h pipeline)

### Option 2: Skip PC Layer Entirely

**Steps**:
1. Use graphs with only S-C, S-S, P-S edges (no P-C edges)
2. Train HGT with only edge-level supervision
3. Evaluate S-C predictions only
4. Compare GNN S-C performance to Evidence CE baseline

**Pros**: Fast, avoids broken PC layer
**Cons**: Can't test full GNN architecture with node supervision

### Option 3: Use Ground Truth as "Oracle" PC

**Steps**:
1. Directly use ground truth PC labels in graphs (no PC model)
2. Train HGT with oracle PC supervision
3. Establishes upper bound on GNN performance

**Pros**: Shows best-case GNN performance
**Cons**: Not realistic deployment scenario

---

## Next Actions

**IMMEDIATE**:
1. ✅ Diagnose and document issues (this report)
2. ⏸️ PAUSE further pipeline work
3. ❓ **USER DECISION REQUIRED**: Choose Option 1, 2, or 3

**IF Option 1 Selected**:
1. Create script to convert ground truth to training format
2. Replace labels_pc.jsonl with correct labels
3. Delete broken PC models
4. Retrain PC layer (4h)
5. Continue pipeline with corrected models

---

## Files to Fix/Replace

### Delete (Broken Models)
```
outputs/runs/real_dev/pc/fold{0-4}/  # All 5 fold models
outputs/runs/real_dev/pc/oof_predictions.jsonl  # Broken predictions
outputs/runs/real_dev/graphs/  # Graphs with bad PC labels
outputs/runs/real_dev/gnn/best_model.pt  # GNN trained with bad supervision
```

### Replace (Wrong Labels)
```
data/processed/labels_pc.jsonl  # Replace with ground truth
```

### Keep (Working Components)
```
data/groundtruth/criteria_matching_groundtruth.csv  # TRUE labels
outputs/runs/real_dev/sc/  # Evidence layer models (working)
configs/  # Configurations
scripts/  # Scripts (working)
```

---

## Lessons Learned

1. **Always validate labels against ground truth** before expensive training
2. **Check model output diversity** before proceeding to next stage
3. **Compare auto-generated labels** to expert annotations
4. **Monitor AUC during training** - AUC=0.5 is a red flag
5. **Test inference on diverse inputs** before batch prediction

---

## Conclusion

The Phase 4 pipeline infrastructure is solid, but the PC layer training failed due to using incorrect labels. The issue is fixable by retraining with the correct ground truth labels from `criteria_matching_groundtruth.csv`.

**Status**: ⏸️ **PAUSED - Awaiting User Decision**
**Recommended**: Retrain PC layer with correct labels (Option 1)
**Alternative**: Skip PC layer entirely (Option 2)
