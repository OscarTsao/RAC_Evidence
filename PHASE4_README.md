# Phase 4: Production Readiness & PC Layer Initiation

**Status:** ⚠️ BLOCKED by CUDA environment issue
**Date:** 2025-11-28
**Trial:** real_dev_hpo_refine_trial_8

---

## Executive Summary

### ✅ What's Working

1. **OOF Predictions Generated** (28MB, 90,693 samples)
   - Contains both uncalibrated and calibrated probabilities
   - Ready for use in PC layer training

2. **Strong Ranking Performance**
   - Precision@5: **80.0%** ✓ (target: ≥80%)
   - Coverage@5: **89.3%** ✓ (excellent)
   - Coverage@20: **100%** ✓ (perfect)
   - AUROC: **89.6%**

3. **Optimal Thresholds Computed**
   - Per-class thresholds optimized for F1 score
   - F1 (optimal): **51.95%**
   - Macro-F1: **75.60%**

### ❌ Critical Issues

1. **CUDA Environment Broken**
   - PyTorch cannot import: `libnvJitLink.so.12` missing
   - Root cause: CPU-only PyTorch + mixed conda/pip installation
   - **Impact:** Cannot run calibration, graph building, or PC training

2. **Poor Calibration (Not Critical)**
   - ECE before: 51.3%
   - ECE after: 50.6% (target: <10%)
   - Temperature scaling ineffective (only 1.4% improvement)
   - Root cause: Model outputs poorly-separated logits (median ≈ 0.03)
   - **Impact:** Probability estimates unreliable, but ranking still works

---

## Quick Start

### Option 1: Fix Everything (Recommended)

```bash
# Step 1: Fix CUDA environment
./fix_cuda_environment.sh

# Step 2: Run comprehensive workflow
./phase4_workflow_fixed.sh
```

### Option 2: Generate Status Report Only (No PyTorch Required)

```bash
# Works even with broken CUDA
python phase4_status_report.py --exp real_dev_hpo_refine_trial_8
```

### Option 3: Manual CUDA Fix

```bash
# Remove conflicting packages
conda remove -y pytorch torchvision torchaudio cpuonly pytorch-mutex --force
pip uninstall -y torch torchvision torchaudio

# Install PyTorch with CUDA 12.1
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Reinstall graph libraries
pip install torch-geometric torch-scatter torch-sparse

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Files Created

| File | Purpose | Requires PyTorch |
|------|---------|------------------|
| `fix_cuda_environment.sh` | Auto-fix CUDA environment | No |
| `phase4_status_report.py` | Comprehensive analysis | No |
| `phase4_workflow_fixed.sh` | Interactive workflow | Yes (for training) |
| `PHASE4_README.md` | This file | No |

---

## Detailed Analysis

### 1. CUDA Environment Issue

**Diagnosis:**
```
✗ cpuonly                   2.0         (blocking CUDA)
✗ pytorch                   2.3.0  cpu  (CPU-only build)
✗ torch (pip)               2.3.1       (conflicts with conda)
✓ pytorch-cuda              12.1        (installed but unused)
✓ NVIDIA Driver             CUDA 12.9   (RTX 3090 available)
```

**Why This Happened:**
- Initial conda environment installed CPU-only PyTorch
- Later pip installation tried to add CUDA support
- The `cpuonly` package blocked CUDA backend
- Missing CUDA 12.x runtime libraries

**Solution:**
- Remove ALL PyTorch packages (conda + pip)
- Fresh install with explicit CUDA support
- Use `fix_cuda_environment.sh` for automated fix

### 2. Calibration Analysis

**The Problem:**

Temperature scaling failed to improve ECE because the underlying model has poor discrimination:

```
Logit Statistics:
  Median:           0.0326  (close to zero)
  Std Dev:          0.1508  (low variance)
  Positive median:  0.6682
  Negative median:  0.0317
  Separation:       0.6364  (weak)
```

**Why Temperature Scaling Didn't Work:**

1. Model outputs logits near zero for most samples
2. Temperature T=2.27 divides logits: `prob = sigmoid(logit / T)`
3. Small logits / 2.27 → even smaller logits → probabilities closer to 0.5
4. Result: ECE improvement of only 1.4%

**Visual Example:**
```
Original:  logit=0.03 → prob=0.508
After T=2.27: logit/T=0.013 → prob=0.503 (minimal change)

Should be:  logit=2.0 → prob=0.88
After T=2.27: logit/T=0.88 → prob=0.71 (proper calibration)
```

**Is This a Problem?**

- ✓ **For ranking:** No - the model still ranks evidence correctly (Precision@5=80%)
- ✗ **For probabilities:** Yes - cannot trust probability estimates
- ⚠️ **For PC layer:** Maybe - PC layer might compensate for poor calibration

**Recommendation:**

1. **Accept current performance** and proceed to PC layer
2. **Use optimal thresholds** instead of probability thresholds
3. **Retrain S-C model** only if probability calibration is critical

### 3. Per-Criterion Breakdown

| Criterion | Samples | Positives | Pos% | Median Logit | Notes |
|-----------|---------|-----------|------|--------------|-------|
| c1 | 10,077 | 333 | 3.30% | 0.0444 | Depressed mood |
| c2 | 10,077 | 105 | 1.04% | 0.0133 | Anhedonia |
| c3 | 10,077 | 47 | 0.47% | 0.0304 | Appetite change |
| c4 | 10,077 | 134 | 1.33% | 0.0256 | Sleep issues |
| c5 | 10,077 | 140 | 1.39% | 0.0212 | Psychomotor |
| c6 | 10,077 | 322 | 3.20% | 0.0293 | Fatigue |
| c7 | 10,077 | 60 | 0.60% | 0.0427 | Worthlessness |
| c8 | 10,077 | 33 | 0.33% | 0.0863 | Cognitive issues |
| c9 | 10,077 | 189 | 1.88% | 0.0102 | Suicidal thoughts |

**Observations:**
- Highly imbalanced (0.33% - 3.30% positive)
- c8 has highest median logit (0.0863) despite lowest positive rate
- c9 has lowest median logit (0.0102) despite moderate positive rate
- Temperature scaling per-class varies: c9 (T=2.21) to c8 (T=2.29)

---

## Decision Matrix

### What Should You Do?

| Goal | Recommendation | Effort | Impact |
|------|----------------|--------|--------|
| **Start PC training ASAP** | Fix CUDA → Run workflow | Low | High |
| **Improve calibration** | Retrain S-C model | High | Uncertain |
| **Get probabilities working** | Use optimal thresholds | Low | Medium |
| **Debug issues** | Run status report | None | Info only |

### Recommended Path

```
1. Fix CUDA environment (5 minutes)
   └─> Run: ./fix_cuda_environment.sh

2. Run workflow (interactive)
   └─> Run: ./phase4_workflow_fixed.sh
   └─> Choose: Skip re-calibration
   └─> Choose: Train CE-PC layer

3. Evaluate PC layer
   └─> Check if PC compensates for S-C calibration issues
```

### Alternative Path (If CUDA Fix Fails)

```
1. Use CPU-only PyTorch
   └─> pip install torch --index-url https://download.pytorch.org/whl/cpu
   └─> WARNING: Very slow training

2. Or: Use existing predictions
   └─> OOF predictions already have prob_cal
   └─> Train PC layer on different machine with CUDA
```

---

## Expected Outcomes

### After CUDA Fix + PC Training

**Best Case:**
- PC layer learns to weight S-C predictions effectively
- Final ECE improves to <10% at PC level
- F1 score improves beyond 51.95%

**Realistic Case:**
- PC layer provides modest improvement (5-10% F1 gain)
- ECE still poor but system makes correct predictions
- Use optimal thresholds for final decisions

**Worst Case:**
- PC layer cannot compensate for S-C issues
- Need to retrain S-C layer with different hyperparameters
- Consider: focal loss, higher pos_weight_scale, more epochs

---

## Troubleshooting

### CUDA Fix Doesn't Work

```bash
# Check CUDA toolkit
nvidia-smi
ldconfig -p | grep cuda

# Check library path
echo $LD_LIBRARY_PATH

# Add CUDA to library path (if needed)
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
```

### PyTorch Still Fails After Fix

```bash
# Nuclear option: Create fresh environment
conda create -n rac_evidence_clean python=3.10
conda activate rac_evidence_clean
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -e .
```

### PC Training Fails

```bash
# Check logs
tail -f outputs/runs/real_dev_pc/train.log

# Reduce batch size if OOM
# Edit configs/ce_pc_real.yaml: batch_size: 16 → 8
```

---

## Next Steps After PC Training

1. **Evaluate End-to-End Pipeline**
   ```bash
   python scripts/evaluate.py --cfg configs/evidence_reranker.yaml
   ```

2. **Compare S-C vs PC Performance**
   - S-C: F1=51.95%, ECE=50.6%
   - PC: TBD after training

3. **Production Deployment Considerations**
   - If ECE remains poor: use optimal thresholds, not probabilities
   - If ranking is good: deploy for top-K recommendation
   - If both fail: retrain S-C layer

---

## Contact & Support

For issues with this workflow:
1. Check `outputs/runs/real_dev_hpo_refine_trial_8/phase4_status_report.txt`
2. Review error logs in `outputs/runs/*/`
3. Verify CUDA setup: `python -c "import torch; print(torch.cuda.is_available())"`

**Generated by:** Claude Code
**Last Updated:** 2025-11-28
