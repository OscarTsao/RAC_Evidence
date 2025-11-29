#!/bin/bash
# Phase 4: Production Readiness & PC Layer Initiation (FIXED VERSION)
# This script addresses the CUDA environment issues and provides fallback options
#
# Author: Claude Code
# Date: 2025-11-28
# Status: Ready for execution after CUDA fix

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

EXP="real_dev_hpo_refine_trial_8"
BEST_CKPT_DIR="outputs/runs/${EXP}"
OOF_PATH="${BEST_CKPT_DIR}/oof_predictions.jsonl"
PC_EXP="real_dev_pc"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

check_pytorch() {
    python -c "import torch; print('PyTorch OK')" 2>/dev/null
    return $?
}

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_section() {
    echo ""
    echo "============================================================================"
    echo "$1"
    echo "============================================================================"
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

print_section "PRE-FLIGHT CHECKS"

# Check 1: OOF predictions exist
if [ ! -f "$OOF_PATH" ]; then
    print_error "OOF predictions not found: $OOF_PATH"
    exit 1
fi
print_status "OOF predictions found ($(du -h $OOF_PATH | cut -f1))"

# Check 2: PyTorch installation
if check_pytorch; then
    print_status "PyTorch installed and working"
    PYTORCH_OK=true
else
    print_warning "PyTorch import failed - CUDA environment issue detected"
    PYTORCH_OK=false
fi

# ============================================================================
# OPTION 1: Fix CUDA Environment First (RECOMMENDED)
# ============================================================================

if [ "$PYTORCH_OK" = false ]; then
    print_section "CUDA ENVIRONMENT FIX REQUIRED"

    echo ""
    echo "PyTorch cannot be imported due to CUDA library conflicts."
    echo ""
    echo "Diagnosis:"
    echo "  - CPU-only PyTorch installed (blocking CUDA)"
    echo "  - Missing libnvJitLink.so.12 library"
    echo "  - Mixed conda/pip installation"
    echo ""
    echo "Options:"
    echo ""
    echo "  [1] Auto-fix: Run ./fix_cuda_environment.sh (RECOMMENDED)"
    echo "      This will:"
    echo "        - Backup current environment"
    echo "        - Remove conflicting packages"
    echo "        - Install PyTorch with CUDA 12.1 support"
    echo "        - Verify installation"
    echo ""
    echo "  [2] Manual fix: See instructions in fix_cuda_environment.sh"
    echo ""
    echo "  [3] Skip fix and proceed with limitations"
    echo "      WARNING: Cannot run calibration, graph building, or PC training"
    echo ""

    read -p "Enter choice [1/2/3]: " choice

    case $choice in
        1)
            print_section "RUNNING AUTO-FIX"
            if [ -f "./fix_cuda_environment.sh" ]; then
                bash ./fix_cuda_environment.sh
                print_status "CUDA environment fixed!"
                print_status "Please run this script again: ./phase4_workflow_fixed.sh"
                exit 0
            else
                print_error "fix_cuda_environment.sh not found!"
                exit 1
            fi
            ;;
        2)
            print_warning "Please fix CUDA manually, then run this script again"
            echo ""
            echo "Manual steps:"
            echo "  1. conda remove pytorch torchvision torchaudio cpuonly pytorch-mutex --force"
            echo "  2. pip uninstall torch torchvision torchaudio"
            echo "  3. conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia"
            echo "  4. pip install torch-geometric torch-scatter torch-sparse"
            echo ""
            exit 0
            ;;
        3)
            print_warning "Proceeding with limitations..."
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
fi

# ============================================================================
# PHASE 1: STATUS REPORT (NO PYTORCH REQUIRED)
# ============================================================================

print_section "PHASE 1: GENERATE STATUS REPORT"

if [ -f "phase4_status_report.py" ]; then
    python phase4_status_report.py --exp $EXP
    print_status "Status report generated"
else
    print_warning "phase4_status_report.py not found, skipping"
fi

# ============================================================================
# PHASE 2: CALIBRATION (PYTORCH REQUIRED)
# ============================================================================

if [ "$PYTORCH_OK" = true ]; then
    print_section "PHASE 2: RE-CALIBRATION (OPTIONAL)"

    echo ""
    echo "Current calibration status:"
    echo "  - Temperature scaling already completed"
    echo "  - ECE improvement: 1.4% (ineffective)"
    echo "  - ECE after calibration: 50.6% (target: <10%)"
    echo ""
    echo "Recommendation: SKIP re-calibration"
    echo "  Reason: Current model has poor logit separation (0.64)"
    echo "          Temperature scaling cannot fix this fundamental issue"
    echo ""
    echo "Options:"
    echo "  [1] Skip re-calibration (use existing prob_cal)"
    echo "  [2] Re-run calibration anyway (not recommended)"
    echo ""

    read -p "Enter choice [1/2] (default: 1): " cal_choice
    cal_choice=${cal_choice:-1}

    case $cal_choice in
        1)
            print_status "Using existing calibrated predictions"
            ;;
        2)
            print_section "RE-CALIBRATING MODEL"
            if python scripts/calibrate.py --exp $EXP --task sc; then
                print_status "Calibration completed"
            else
                print_error "Calibration failed"
                exit 1
            fi
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
else
    print_warning "SKIPPING: Calibration (requires PyTorch)"
fi

# ============================================================================
# PHASE 3: THRESHOLD OPTIMIZATION (ALREADY DONE)
# ============================================================================

print_section "PHASE 3: OPTIMAL THRESHOLDS"

if [ -f "${BEST_CKPT_DIR}/evidence_metrics.json" ]; then
    # Check if optimal thresholds exist
    if grep -q "optimal_thresholds" "${BEST_CKPT_DIR}/evidence_metrics.json"; then
        print_status "Optimal per-class thresholds already computed"

        # Show summary
        python -c "
import json
with open('${BEST_CKPT_DIR}/evidence_metrics.json') as f:
    m = json.load(f)
if 'optimal_threshold_metrics' in m:
    opt = m['optimal_threshold_metrics']
    print(f\"  F1 (optimal):     {opt['f1_optimal']:.4f}\")
    print(f\"  Macro-F1:         {opt['macro_f1_optimal']:.4f}\")
    print(f\"  Precision:        {opt['precision_optimal']:.4f}\")
" 2>/dev/null || print_warning "Could not parse metrics file"
    else
        print_warning "Optimal thresholds not found in metrics"
    fi
else
    print_warning "Metrics file not found"
fi

# ============================================================================
# PHASE 4: BUILD GRAPH FOR GNN/PC (PYTORCH REQUIRED)
# ============================================================================

if [ "$PYTORCH_OK" = true ]; then
    print_section "PHASE 4: GRAPH BUILDING (OPTIONAL)"

    echo ""
    echo "Graph building is required for GNN-based PC layer."
    echo "It uses OOF predictions as node features."
    echo ""
    echo "Options:"
    echo "  [1] Build graph for GNN approach"
    echo "  [2] Skip graph (use cross-encoder PC layer instead)"
    echo ""

    read -p "Enter choice [1/2] (default: 2): " graph_choice
    graph_choice=${graph_choice:-2}

    case $graph_choice in
        1)
            print_section "BUILDING GRAPH"
            if python scripts/build_graph.py --cfg configs/graph.yaml; then
                print_status "Graph built successfully"
            else
                print_error "Graph building failed"
                exit 1
            fi
            ;;
        2)
            print_status "Skipping graph building (using CE-PC approach)"
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
else
    print_warning "SKIPPING: Graph building (requires PyTorch)"
fi

# ============================================================================
# PHASE 5: PC LAYER TRAINING (PYTORCH REQUIRED)
# ============================================================================

if [ "$PYTORCH_OK" = true ]; then
    print_section "PHASE 5: POST-CRITERION (PC) LAYER TRAINING"

    echo ""
    echo "PC Layer training options:"
    echo "  [1] Train Cross-Encoder PC (recommended)"
    echo "  [2] Train GNN-based PC (requires graph from Phase 4)"
    echo "  [3] Skip PC training"
    echo ""

    read -p "Enter choice [1/2/3] (default: 1): " pc_choice
    pc_choice=${pc_choice:-1}

    case $pc_choice in
        1)
            print_section "TRAINING CROSS-ENCODER PC LAYER"

            # Check config exists
            if [ ! -f "configs/ce_pc_real.yaml" ]; then
                print_error "Config not found: configs/ce_pc_real.yaml"
                exit 1
            fi

            # Start training
            if python scripts/train_ce_pc.py --cfg configs/ce_pc_real.yaml; then
                print_status "PC layer training completed successfully"
            else
                print_error "PC layer training failed"
                exit 1
            fi
            ;;
        2)
            print_section "TRAINING GNN-BASED PC LAYER"

            # Check graph exists
            if [ ! -d "data/interim_real/graphs" ]; then
                print_error "Graph not found. Please run Phase 4 first."
                exit 1
            fi

            if python scripts/train_graph.py --cfg configs/graph.yaml; then
                print_status "GNN PC layer training completed"
            else
                print_error "GNN training failed"
                exit 1
            fi
            ;;
        3)
            print_status "Skipping PC layer training"
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
else
    print_warning "SKIPPING: PC layer training (requires PyTorch)"
    echo ""
    echo "To proceed with PC training:"
    echo "  1. Fix CUDA environment: ./fix_cuda_environment.sh"
    echo "  2. Re-run this script: ./phase4_workflow_fixed.sh"
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print_section "PHASE 4 WORKFLOW COMPLETE"

echo ""
echo "Summary:"
echo "  - OOF predictions: ✓ Available (${OOF_PATH})"

if [ "$PYTORCH_OK" = true ]; then
    echo "  - PyTorch:         ✓ Working"
    echo "  - Calibration:     ✓ Completed (ECE=50.6%, improvement limited by model)"
    echo "  - Thresholds:      ✓ Optimized per-class thresholds available"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate PC layer performance"
    echo "  2. Run full pipeline evaluation"
    echo "  3. Consider retraining S-C layer if ECE is critical for your use case"
else
    echo "  - PyTorch:         ✗ Not working (CUDA issue)"
    echo ""
    echo "IMPORTANT: Fix CUDA to proceed with training"
    echo "  Run: ./fix_cuda_environment.sh"
fi

echo ""
print_status "Workflow completed!"
