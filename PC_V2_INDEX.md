# PC v2 Training Pipeline - Documentation Index

This index helps you navigate the PC v2 implementation documentation.

## Quick Links

### For Users (Start Here)
1. **PC_V2_QUICKSTART.md** - Quick start guide with commands and examples
   - How to run training
   - How to evaluate results
   - Common issues and solutions

### For Developers
2. **PC_V2_IMPLEMENTATION.md** - Comprehensive technical documentation
   - Detailed architecture
   - Design decisions
   - Code walkthroughs
   - Troubleshooting guide

### For Managers/Reviewers
3. **PC_V2_SUMMARY.md** - Executive summary and implementation report
   - What was fixed and why
   - Verification results
   - Success criteria checklist

## File Structure

### Code Files
```
scripts/
├── train_pc_v2.py           # Main training script (577 lines)
├── eval_pc_v2.py            # Evaluation script (245 lines)
└── verify_pc_v2_data.py     # Data verification (177 lines)

configs/
└── pc_v2.yaml               # Configuration file (48 lines)
```

### Documentation Files
```
├── PC_V2_QUICKSTART.md      # Quick start guide (220 lines)
├── PC_V2_IMPLEMENTATION.md  # Technical documentation (403 lines)
├── PC_V2_SUMMARY.md         # Implementation report (320 lines)
└── PC_V2_INDEX.md           # This file
```

## Common Tasks

### I want to run training
→ Read: **PC_V2_QUICKSTART.md** (sections 1-2)
→ Run: `python scripts/train_pc_v2.py`

### I want to understand the architecture
→ Read: **PC_V2_IMPLEMENTATION.md** (sections "Data Flow" and "Architecture Design")

### I want to evaluate results
→ Read: **PC_V2_QUICKSTART.md** (section "Evaluation")
→ Run: `python scripts/eval_pc_v2.py --oof_path ... --output_path ...`

### I want to customize hyperparameters
→ Read: **PC_V2_IMPLEMENTATION.md** (section "Key Design Decisions")
→ Edit: `configs/pc_v2.yaml` or create a custom config

### I encountered an error
→ Read: **PC_V2_QUICKSTART.md** (section "Troubleshooting")
→ Or: **PC_V2_IMPLEMENTATION.md** (section "Troubleshooting")

### I want to verify data before training
→ Run: `python scripts/verify_pc_v2_data.py`

### I want to understand what was fixed
→ Read: **PC_V2_SUMMARY.md** (section "Critical Issues Fixed")

### I want to compare to old pipeline
→ Read: **PC_V2_SUMMARY.md** (section "Comparison to Old Pipeline")
→ Or: **PC_V2_IMPLEMENTATION.md** (section "Comparison to Old train_ce_pc.py")

## Document Details

### PC_V2_QUICKSTART.md
**Best for**: First-time users, quick reference
**Length**: 220 lines
**Sections**:
- Prerequisites
- Training (default and custom configs)
- Evaluation
- Expected Results
- Troubleshooting (GPU memory, model collapse)
- File Structure
- Next Steps

### PC_V2_IMPLEMENTATION.md
**Best for**: Developers, in-depth understanding
**Length**: 403 lines
**Sections**:
- Overview
- Critical Issues Fixed
- Files Created
- Data Flow
- Key Implementation Details
- Usage
- Success Criteria
- Comparison to Old Pipeline
- Verification Checklist
- Troubleshooting
- Next Steps

### PC_V2_SUMMARY.md
**Best for**: Managers, reviewers, executive overview
**Length**: 320 lines
**Sections**:
- Executive Summary
- Files Created
- Critical Issues Fixed
- Architecture Design
- Verification Results
- Usage
- Key Design Decisions
- Comparison to Old Pipeline
- Documentation
- Success Criteria
- Next Steps
- Git Status
- Execution Report

## Key Concepts

### 5-Fold Cross-Validation
- Old: Fake CV (split inside loop)
- New: True CV (split before loop)
- See: PC_V2_IMPLEMENTATION.md → "Critical Issues Fixed" → #1

### Model Collapse
- Problem: All predictions are 0 or 1
- Cause: Aggressive hyperparameters (LR=1e-3)
- Solution: Conservative params (LR=2e-5)
- See: PC_V2_IMPLEMENTATION.md → "Critical Issues Fixed" → #2

### Temperature Calibration
- Purpose: Better-calibrated probabilities
- Method: Per-class temperature on dev set
- See: PC_V2_IMPLEMENTATION.md → "Temperature Scaling"

### Stratified Splitting
- Goal: Balanced symptom patterns across folds
- Method: Label pattern hashing
- See: PC_V2_IMPLEMENTATION.md → "Stratified Splitting Strategy"

## Data Files

Input data required:
```
data/processed/labels_pc.jsonl    # 13,356 labels (1,484 posts × 9 criteria)
data/redsm5/redsm5_posts.csv      # 1,484 posts
data/DSM5/MDD_Criteira.json       # 9 criteria (A.1-A.9)
```

Output data generated:
```
outputs/runs/real_dev_pc_v2/
├── pc_oof.jsonl                  # 13,356 OOF predictions
├── fold_temperatures.json        # Temperature scaling results
├── metrics.json                  # Evaluation metrics (after eval_pc_v2.py)
└── fold_{0-4}/best_model/        # PyTorch checkpoints (~5GB total)
```

## Timeline

1. **Data verification**: ~1 minute
   ```bash
   python scripts/verify_pc_v2_data.py
   ```

2. **Training**: ~2-3 hours on GPU (A100)
   ```bash
   python scripts/train_pc_v2.py
   ```

3. **Evaluation**: ~1 minute
   ```bash
   python scripts/eval_pc_v2.py
   ```

Total: ~2-3 hours

## Expected Metrics

Target performance (proper CV):
- **Macro-F1 (optimal)**: >0.15 (acceptable), >0.25 (good)
- **Macro-AUPRC**: >0.40
- **No model collapse**: Diverse predictions (not all-0 or all-1)

Class distribution:
- Positive: ~1,293 (9.7%)
- Negative: ~12,063 (90.3%)

## Git Integration

To commit the changes:
```bash
git add scripts/train_pc_v2.py \
        configs/pc_v2.yaml \
        scripts/eval_pc_v2.py \
        scripts/verify_pc_v2_data.py \
        PC_V2_*.md

git commit -m "Implement PC v2 training pipeline with proper 5-fold CV

- Fix fake CV → true OOF predictions (split before loop)
- Fix model collapse → conservative hyperparams (LR=2e-5)
- Add per-class temperature calibration
- Mirror Evidence pipeline architecture
- Add comprehensive documentation

Closes #XX (replace with issue number)"
```

## Support

For issues or questions:
1. Check **PC_V2_QUICKSTART.md** → Troubleshooting
2. Check **PC_V2_IMPLEMENTATION.md** → Troubleshooting
3. Review git history for recent changes
4. Run data verification: `python scripts/verify_pc_v2_data.py`

## References

- Evidence pipeline (template): `src/Project/evidence/train.py`
- CV utilities: `src/Project/utils/cv_utils.py`
- Temperature scaling: `src/Project/calib/temperature.py`
- Old broken pipeline (DO NOT USE): `scripts/train_ce_pc.py`

---

**Last Updated**: 2025-12-02
**Status**: Complete and Verified ✓
