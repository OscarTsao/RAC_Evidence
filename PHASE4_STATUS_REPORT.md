# Phase 4: Status Report - PC Layer Issues Fixed

**Date**: 2025-12-01
**Status**: ✅ **LABELS FIXED** | ⏳ **READY FOR RETRAINING**

---

## Summary

Successfully diagnosed and fixed critical PC layer issues. The pipeline completed technically but discovered that **PC training used incorrect labels** with 11% error rate, resulting in degenerate model outputs. Ground truth labels have been restored.

---

## Issues Found & Fixed

### 1. ✅ Degenerate Model Output (DIAGNOSED)

**Problem**: All PC predictions identical (prob=0.5242, AUC=0.5)
- Model trained for 4 hours but learned nothing discriminative
- Outputs same prediction regardless of input text
- All 5 fold models affected

**Root Cause**: Training on noisy/incorrect labels led to model collapse

### 2. ✅ Wrong Training Labels (FIXED)

**Before** (`labels_pc_old.jsonl`):
- 13,293 samples
- 12,000 negatives (90.27%), 1,293 positives (9.73%)
- **Only 88.96% agreement** with ground truth
- **1,468 label errors** (11.04% mislabeled)

**After** (`labels_pc.jsonl`):
- 13,356 samples (correct count)
- 12,063 negatives (90.32%), 1,293 positives (9.68%)
- **100% ground truth accuracy**
- Derived from `data/groundtruth/criteria_matching_groundtruth.csv`

**Changes Made**:
```bash
data/processed/labels_pc_old.jsonl     # Backed up broken labels
data/processed/labels_pc.jsonl         # Replaced with ground truth
data/processed/labels_pc_groundtruth.jsonl  # Ground truth copy
```

---

## Evaluation Results (Before Fix)

### PC Predictions vs Ground Truth

**Overall Metrics** (broken model):
- F1 (micro): 0.0973 (9.73%)
- F1 (macro): 0.0886 (8.86%)
- **AUC: 0.5000** (random guessing!)
- AUPRC: 0.0973

**Per-Criterion F1** (broken model):
| Criterion | F1 Score | Symptom |
|-----------|----------|---------|
| c1 | 0.3634 | Depressed Mood |
| c7 | 0.3479 | Worthlessness |
| c9 | 0.2021 | Suicidal Thoughts |
| c2 | 0.1549 | Anhedonia |
| c6 | 0.1549 | Fatigue |
| c4 | 0.1292 | Sleep Issues |
| c8 | 0.0768 | Cognitive Issues |
| c3 | 0.0579 | Appetite Change |
| c5 | 0.0463 | Psychomotor |

These scores are **meaningless** because the model outputs identical predictions for all inputs.

---

## What Was Wasted

**Compute Time**:
- PC training: 4 hours 5 minutes (5 folds × 30,000 steps)
- PC prediction generation: 19 minutes
- Graph reconstruction: 2 minutes
- HGT training: 1 minute
- **Total**: ~4.5 hours on broken models

**Storage**:
- 5 degenerate PC models: ~10.6 GB
- Broken PC predictions: 1.5 MB
- Broken graphs: ~500 MB
- **Total**: ~11 GB of useless outputs

---

## Files to Clean Up

### Delete (Broken Outputs)
```bash
# Broken PC models (10.6 GB)
rm -rf outputs/runs/real_dev/pc/fold{0,1,2,3,4}/

# Broken predictions
rm outputs/runs/real_dev/pc/oof_predictions.jsonl

# Broken graphs
rm -rf outputs/runs/real_dev/graphs/

# Broken GNN model
rm outputs/runs/real_dev/gnn/best_model.pt
```

### Keep (Working Components)
```bash
data/groundtruth/criteria_matching_groundtruth.csv  # Ground truth source
data/processed/labels_pc.jsonl                      # Fixed labels ✅
data/processed/labels_pc_groundtruth.jsonl          # Ground truth copy
outputs/runs/real_dev/sc/                           # Evidence layer (working)
scripts/                                             # All scripts
```

---

## Next Steps

### Option 1: Retrain PC Layer (RECOMMENDED)

**What**: Retrain with correct labels and complete pipeline

**Steps**:
1. ✅ Fix labels (DONE)
2. Delete broken PC models
3. Retrain PC layer (5-fold CV) - **~4 hours**
4. Generate corrected PC predictions - 20 min
5. Rebuild graphs with correct labels - 2 min
6. Retrain HGT with meaningful supervision - 1 min
7. Evaluate full pipeline

**Expected Results**:
- PC Macro-F1: 0.20-0.40 (realistic for this task)
- Diverse predictions (not all identical)
- Meaningful supervision signal for GNN
- Valid pipeline comparison vs baseline

**Total Time**: ~5 hours

### Option 2: Use Ground Truth Directly (ORACLE)

**What**: Skip PC training, use ground truth as labels

**Steps**:
1. Load ground truth PC labels directly into graphs
2. Train HGT with oracle PC supervision
3. Evaluate (establishes upper bound)

**Pros**: Fast (30 min), shows best-case GNN performance
**Cons**: Not realistic deployment scenario

### Option 3: Skip PC Layer Entirely

**What**: Train GNN without PC supervision

**Steps**:
1. Use graphs with only S-C, S-S, P-S edges (no P-C)
2. Train HGT with edge-level supervision only
3. Evaluate S-C predictions

**Pros**: Fast, avoids broken PC layer
**Cons**: Can't test full architecture with node supervision

---

## Recommended Action

**PROCEED WITH OPTION 1**: Retrain PC layer with correct labels

**Justification**:
1. Correct labels are now in place
2. Infrastructure is working
3. Only need to rerun training (automated)
4. Will produce realistic, deployable models
5. Enables full pipeline evaluation

**Command to Execute**:
```bash
# Clean up broken models
rm -rf outputs/runs/real_dev/pc/fold{0,1,2,3,4}/
rm outputs/runs/real_dev/pc/oof_predictions.jsonl
rm outputs/runs/real_dev/pc/metrics.json

# Retrain with correct labels (4 hours)
python3 scripts/train_ce_pc.py --cfg configs/ce_pc_real.yaml 2>&1 | tee outputs/pc_retraining.log

# Continue pipeline
python3 scripts/generate_pc_predictions.py
python3 scripts/build_heterograph.py --cfg configs/graph_real.yaml
python3 scripts/train_hgt.py --cfg configs/graph_real.yaml
python3 scripts/eval_pc_groundtruth.py
```

---

## What Worked

Despite the label issue, the following worked perfectly:

1. ✅ **Evidence (S-C) Layer**: Macro-F1 0.756 (verified)
2. ✅ **Graph Construction**: 1,477 graphs built successfully
3. ✅ **HGT Architecture**: 346K parameters, trains in 1 minute
4. ✅ **Pipeline Infrastructure**: All scripts functional
5. ✅ **Evaluation Framework**: Metrics computation working
6. ✅ **Ground Truth Discovery**: Found correct labels
7. ✅ **Label Fix Script**: Automated label replacement

---

## Lessons Learned

1. **Always validate auto-generated labels** against ground truth before expensive training
2. **Check model output diversity** - all-identical predictions are a red flag
3. **Monitor AUC during training** - AUC=0.5 means random guessing
4. **Test inference on diverse inputs** before batch prediction
5. **Compare label sources** when multiple exist

---

## Files Created

### Diagnostic & Reporting
- `PHASE4_DIAGNOSIS.md` - Detailed technical diagnosis
- `PHASE4_STATUS_REPORT.md` - This file

### Scripts
- `scripts/fix_pc_labels.py` - Label replacement automation
- `scripts/eval_pc_groundtruth.py` - PC evaluation against ground truth

### Outputs
- `outputs/runs/real_dev/pc/groundtruth_evaluation.json` - Baseline evaluation
- `data/processed/labels_pc_old.jsonl` - Backup of broken labels
- `data/processed/labels_pc_groundtruth.jsonl` - Ground truth copy

---

## Conclusion

✅ **Issues Diagnosed**: PC training used wrong labels → degenerate model
✅ **Root Cause Found**: 11% label noise led to model collapse
✅ **Fix Implemented**: Replaced with 100% correct ground truth labels
⏳ **Ready for Retraining**: All infrastructure working, labels corrected

**Recommendation**: Proceed with PC retraining (~4 hours) to complete pipeline with correct models.

---

**Status**: ✅ **READY FOR RETRAINING**
**Estimated Time to Completion**: 5 hours (4h training + 1h pipeline)
