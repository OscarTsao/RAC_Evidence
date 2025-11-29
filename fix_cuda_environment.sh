#!/bin/bash
# CUDA Environment Fix Script for RAC_Evidence
# This script fixes the PyTorch CPU/CUDA conflict

set -e

echo "=== PyTorch CUDA Environment Fix ==="
echo "Current environment: $CONDA_DEFAULT_ENV"
echo ""

echo "Step 1: Backing up current environment..."
conda env export > environment_backup_$(date +%Y%m%d_%H%M%S).yml
echo "✓ Backup saved"

echo ""
echo "Step 2: Removing conflicting PyTorch packages..."
# Remove all PyTorch-related packages
pip uninstall -y torch torchvision torchaudio torch-geometric torch-scatter torch-sparse 2>/dev/null || true
conda remove -y pytorch torchvision torchaudio cpuonly pytorch-mutex --force 2>/dev/null || true

echo ""
echo "Step 3: Installing PyTorch with CUDA 12.1 support..."
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

echo ""
echo "Step 4: Reinstalling torch-geometric and dependencies..."
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

echo ""
echo "Step 5: Verifying installation..."
python -c "
import torch
print('✓ PyTorch version:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ CUDA version:', torch.version.cuda)
    print('✓ GPU count:', torch.cuda.device_count())
    print('✓ GPU name:', torch.cuda.get_device_name(0))
else:
    print('✗ CUDA NOT available - installation may have failed')
    exit(1)
"

echo ""
echo "=== Fix Complete! ==="
echo "PyTorch is now properly configured with CUDA 12.1 support."
echo ""
echo "You can now run:"
echo "  python scripts/calibrate.py --exp real_dev_hpo_refine_trial_8 --task sc"
echo "  python scripts/train_ce_pc.py --cfg configs/ce_pc_real.yaml"
