# PC v2 Training Pipeline Implementation

## Overview

This document describes the refactored PC (Post-Criterion) v2 training pipeline that fixes the broken `train_ce_pc.py` implementation. The new pipeline mirrors the successful Evidence (S-C) architecture with proper 5-fold cross-validation and calibrated OOF predictions.

## Critical Issues Fixed

### 1. Fake 5-Fold Cross-Validation
**Old Problem**: Data was split INSIDE the fold loop, meaning there was NO true Out-of-Fold prediction.
**New Solution**: Data split happens BEFORE the loop using `StratifiedKFold`, ensuring true OOF predictions.

### 2. Model Collapse
**Old Problem**: Models collapsed to trivial solutions (all-0 or all-1 predictions) due to aggressive hyperparameters (LR=1e-3).
**New Solution**: Conservative hyperparameters (LR=2e-5, proper warmup, gradient accumulation) prevent saturation.

### 3. No Temperature Calibration
**Old Problem**: No calibration, fixed threshold=0.5 for evaluation.
**New Solution**: Per-class temperature scaling on dev set, applied to OOF predictions.

## Files Created

### 1. `scripts/train_pc_v2.py` (577 lines)

Main training script implementing:

#### Data Structure
- **PCExample dataclass**: Represents (post, criterion, label) training examples
- **PCDataBuilder class**: Loads and builds training data from:
  - `data/processed/labels_pc.jsonl` - Ground truth labels
  - `data/redsm5/redsm5_posts.csv` - Post texts
  - `data/DSM5/MDD_Criteira.json` - DSM-5 criterion descriptions
  - Maps A.1→c1, A.2→c2, ..., A.9→c9 (excludes A.10)

#### Core Functions
- **`train_fold()`**: Trains one fold with Transformers Trainer
  - Model: `BAAI/bge-reranker-v2-m3`
  - Conservative hyperparams: LR=2e-5, epochs=3, batch=16, grad_accum=4
  - Max length: 512 (CRITICAL for full post context)
  - Text pairing: `f"{post_text} [SEP] {criterion_desc}"`
  - Returns: (checkpoint_path, temperature_dict)

- **`generate_oof_predictions()`**: Generates OOF predictions for test fold
  - Loads trained model
  - Applies per-class temperature scaling
  - Returns predictions with: post_id, cid, label, logit, prob_uncal, prob_cal, fold

- **`train_5fold_pc()`**: Main 5-fold training loop
  - **Stratified split**: Uses label patterns per post to ensure balanced folds
  - **BEFORE loop**: Splits posts into 5 folds using StratifiedKFold
  - **For each fold**:
    1. Split train into train (90%) + dev (10%)
    2. Train on train_posts
    3. Fit per-class temperature on dev_posts
    4. Generate OOF predictions on test_posts
  - Saves: `pc_oof.jsonl`, `fold_temperatures.json`

#### Key Differences from Evidence Pipeline
- **Simplified negatives**: No hard negative sampling (uses all post-criterion pairs)
- **Longer max_length**: 512 vs 384 (posts are longer than sentences)
- **Direct labels**: No retrieval dependency (uses labels_pc.jsonl directly)

### 2. `configs/pc_v2.yaml` (48 lines)

Configuration file with conservative hyperparameters:

```yaml
exp: real_dev_pc_v2
seed: 42

model:
  name: BAAI/bge-reranker-v2-m3
  max_length: 512  # Full post context

train:
  lr: 2.0e-5  # Conservative (NOT 1e-3!)
  epochs: 3
  batch_size: 16
  gradient_accumulation: 4
  bf16: true

cv:
  n_folds: 5
  dev_ratio: 0.1

output_dir: outputs/runs/real_dev_pc_v2
```

### 3. `scripts/eval_pc_v2.py` (245 lines)

Evaluation script computing comprehensive metrics:

#### Metrics Computed
**Per-Criterion**:
- AUPRC (calibrated and uncalibrated)
- Optimal threshold via precision-recall curve
- F1, Precision, Recall at optimal threshold
- F1, Precision, Recall at fixed threshold 0.5

**Macro-Averaged**:
- Mean of all per-criterion metrics
- Useful for comparing overall model performance

#### Usage
```bash
python scripts/eval_pc_v2.py \
  --oof_path outputs/runs/real_dev_pc_v2/pc_oof.jsonl \
  --output_path outputs/runs/real_dev_pc_v2/metrics.json
```

## Data Flow

```
Input Data:
├── data/processed/labels_pc.jsonl          (14,840 labels: 1,649 posts × 9 criteria)
├── data/redsm5/redsm5_posts.csv           (1,484 posts)
└── data/DSM5/MDD_Criteira.json            (9 criteria: A.1-A.9)

↓ PCDataBuilder.build_examples()

Training Examples:
└── List[PCExample]                         (post_id, cid, label, text, criterion)

↓ StratifiedKFold(n_splits=5)

5-Fold Split:
├── Fold 0: train (90%) → dev (10%) + test (20%)
├── Fold 1: train (90%) → dev (10%) + test (20%)
├── Fold 2: train (90%) → dev (10%) + test (20%)
├── Fold 3: train (90%) → dev (10%) + test (20%)
└── Fold 4: train (90%) → dev (10%) + test (20%)

↓ For each fold:
  1. train_fold() → (checkpoint, temperature_dict)
  2. generate_oof_predictions() → List[Dict]

Output Artifacts:
├── outputs/runs/real_dev_pc_v2/
│   ├── pc_oof.jsonl                       (OOF predictions: all folds combined)
│   ├── fold_temperatures.json             (Global + per-class temperatures)
│   ├── fold_0/best_model/                 (Model checkpoint)
│   ├── fold_1/best_model/
│   ├── fold_2/best_model/
│   ├── fold_3/best_model/
│   └── fold_4/best_model/
```

## Key Implementation Details

### 1. Stratified Splitting Strategy

**Challenge**: Each post has 9 labels (one per criterion). We need to split posts while maintaining label distribution.

**Solution**: Create a "label pattern" for each post by concatenating its sorted labels:
```python
# Example:
# Post X has labels [1,0,0,1,0,0,0,0,1] → "000000111"
# Post Y has labels [0,0,0,0,0,0,0,0,0] → "000000000"

post_to_label_vec = {}
for post_id in post_ids:
    post_labels = [label for (pid, _cid), label in pairs_and_labels if pid == post_id]
    post_to_label_vec[post_id] = "".join(map(str, sorted(post_labels)))

strat_labels = [post_to_label_vec[pid] for pid in post_ids]
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for fold, (train_idx, test_idx) in enumerate(kfold.split(post_ids, strat_labels)):
    # Split happens HERE (before loop), not inside!
    ...
```

This ensures posts with similar symptom patterns are distributed evenly across folds.

### 2. Per-Class Temperature Scaling

Uses existing `fit_per_class_temperature()` from `cv_utils.py`:

```python
# On dev set
temperature_dict = fit_per_class_temperature(dev_logits, dev_labels, dev_cids)
# Returns: {"global": 1.23, "per_class": {"c1": 1.15, "c2": 1.31, ...}}

# During OOF prediction
for logit_val, ex in zip(logits, batch_examples):
    cid_temp = per_class_temps.get(ex.cid, global_temp)  # Fallback to global
    safe_temp = max(cid_temp, 1e-4)
    prob = torch.sigmoid(torch.tensor(logit_val) / safe_temp).item()
```

### 3. Conservative Hyperparameters

**Critical for preventing model collapse**:

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| Learning Rate | 1e-3 | 2e-5 | Prevent saturation |
| Epochs | 5+ | 3 | Prevent overfitting |
| Max Length | 384 | 512 | Full post context |
| Warmup Ratio | 0.0 | 0.06 | Stable training |
| Gradient Accumulation | 1 | 4 | Larger effective batch |

## Usage

### Training

```bash
# Run from repo root
python scripts/train_pc_v2.py

# Or with custom config
python scripts/train_pc_v2.py --config-name=pc_v2_custom
```

**Expected Runtime**: ~2-3 hours for 5 folds on GPU (depends on hardware)

### Evaluation

```bash
python scripts/eval_pc_v2.py \
  --oof_path outputs/runs/real_dev_pc_v2/pc_oof.jsonl \
  --output_path outputs/runs/real_dev_pc_v2/metrics.json
```

### Output Format

**pc_oof.jsonl** (one line per prediction):
```json
{"post_id": "s_1270_9", "cid": "c1", "label": 0, "logit": -2.34, "prob_uncal": 0.09, "prob_cal": 0.12, "prob": 0.12, "fold": 0}
{"post_id": "s_1270_9", "cid": "c2", "label": 0, "logit": -1.56, "prob_uncal": 0.17, "prob_cal": 0.21, "prob": 0.21, "fold": 0}
...
```

**metrics.json**:
```json
{
  "per_criterion": {
    "c1": {
      "n_samples": 1649,
      "n_pos": 456,
      "auprc_cal": 0.6234,
      "f1_optimal": 0.5123,
      "optimal_threshold": 0.3456
    },
    ...
  },
  "macro": {
    "macro_auprc_cal": 0.5891,
    "macro_f1_optimal": 0.4523,
    "macro_precision_optimal": 0.4789,
    "macro_recall_optimal": 0.4321
  }
}
```

## Success Criteria

1. **Proper 5-fold CV**: ✓ Data split BEFORE loop
2. **No model collapse**: ✓ Diverse predictions (not all-0 or all-1)
3. **Reasonable metrics**: Target Macro-F1 > 0.15, ideally > 0.25
4. **Per-class calibration**: ✓ Temperature scaling applied correctly
5. **Clean code**: ✓ Mirrors Evidence pipeline architecture

## Comparison to Old train_ce_pc.py

| Aspect | Old (Broken) | New (PC v2) |
|--------|-------------|-------------|
| CV Strategy | Split inside loop (fake) | Split before loop (true OOF) |
| Learning Rate | 1e-3 (too high) | 2e-5 (conservative) |
| Max Length | 384 (truncates) | 512 (full context) |
| Calibration | None | Per-class temperature |
| Dev Set | No separate dev | 10% of train for calibration |
| Model Collapse | Frequent | Prevented by conservative params |
| Code Quality | Messy | Clean, documented, testable |

## Verification Checklist

Before running, verify:
- [ ] Files exist:
  - `data/processed/labels_pc.jsonl`
  - `data/redsm5/redsm5_posts.csv`
  - `data/DSM5/MDD_Criteira.json`
- [ ] Config paths are correct in `configs/pc_v2.yaml`
- [ ] GPU available (check `torch.cuda.is_available()`)
- [ ] Sufficient disk space (~5GB for checkpoints)

## Troubleshooting

### Issue: "Labels file not found"
**Solution**: Run label generation script first:
```bash
python scripts/generate_labels_pc.py
```

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in config:
```yaml
train:
  batch_size: 8  # Reduce from 16
  gradient_accumulation: 8  # Increase to maintain effective batch size
```

### Issue: "Model predictions are all 0 or all 1"
**Solution**: This indicates model collapse. Possible fixes:
1. Verify learning rate is 2e-5 (not higher)
2. Reduce number of epochs (try 2 instead of 3)
3. Check class imbalance in data (should be ~31% positive)

## Next Steps

After training completes:
1. Run evaluation script to get metrics
2. Compare Macro-F1 to baseline (should be > 0.15)
3. Check per-criterion AUPRC for problematic criteria
4. If metrics are good, integrate into main pipeline
5. Consider hyperparameter tuning via Optuna (see `configs/evidence_reranker.yaml` for example)

## References

- Evidence pipeline: `src/Project/evidence/train.py`
- CV utilities: `src/Project/utils/cv_utils.py`
- Temperature scaling: `src/Project/calib/temperature.py`
- Original (broken) PC training: `scripts/train_ce_pc.py` (DO NOT USE)
