#!/usr/bin/env bash
# Orchestration script for Evidence (S-C) 5-fold OOF pipeline
# Implements STRICT 5-fold with same-post retrieval, hybrid negatives, and per-fold temperature calibration

set -euo pipefail

# Configuration
EXP="${EXP:-sc5_$(date +%Y%m%d_%H%M)}"
RUNTIME_CFG="${RUNTIME_CFG:-configs/retrieval/runtime.yaml}"
RERANKER_CFG="${RERANKER_CFG:-configs/evidence_reranker.yaml}"
LABEL_SC="${LABEL_SC:-data/processed/labels_sc.jsonl}"
LABEL_PC="${LABEL_PC:-data/processed/labels_pc.jsonl}"
CRITERIA="${CRITERIA:-data/raw/criteria.json}"
SPLITS="${SPLITS:-data/processed/splits_5fold_posts.json}"
OUTPUT_DIR="outputs/runs/${EXP}"
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        --exp)
            EXP="$2"
            OUTPUT_DIR="outputs/runs/${EXP}"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry_run] [--exp EXP_NAME]"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
echo "================================================================"
echo "  Evidence (S-C) 5-Fold OOF Pipeline"
echo "  Experiment: ${EXP}"
echo "  Dry run: ${DRY_RUN}"
echo "================================================================"
echo ""

# Step 0: Check required files
log_info "Checking required files..."

REQUIRED_FILES=(
    "${LABEL_SC}"
    "${LABEL_PC}"
    "${CRITERIA}"
    "${RUNTIME_CFG}"
    "${RERANKER_CFG}"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${file}" ]; then
        log_error "Required file not found: ${file}"
        exit 1
    fi
done

log_success "All required files found"

# Step 1: Generate 5-fold splits if missing
if [ ! -f "${SPLITS}" ]; then
    log_warning "Splits file not found: ${SPLITS}"
    log_info "Generating post-level 5-fold splits..."

    python -m Project.utils.create_splits \
        --labels_sc "${LABEL_SC}" \
        --n_folds 5 \
        --seed 42 \
        --out "${SPLITS}"

    if [ $? -eq 0 ]; then
        log_success "Created splits: ${SPLITS}"
    else
        log_error "Failed to create splits"
        exit 1
    fi
else
    log_info "Using existing splits: ${SPLITS}"
fi

# Step 2: Run 5-fold training pipeline
log_info "Starting 5-fold Evidence CE training..."

TRAIN_CMD="python scripts/train_evidence_5fold.py \
    --cfg ${RERANKER_CFG} \
    --exp ${EXP}"

if [ "${DRY_RUN}" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --dry_run"
    log_warning "Running in DRY RUN mode (minimal steps)"
fi

log_info "Running: ${TRAIN_CMD}"
eval "${TRAIN_CMD}"

if [ $? -ne 0 ]; then
    log_error "Training failed"
    exit 1
fi

log_success "Training complete"

# Step 3: Run acceptance checks
log_info "Running acceptance checks..."

python scripts/run_acceptance_checks.py \
    --exp "${EXP}"

if [ $? -eq 0 ]; then
    log_success "Acceptance checks PASSED"
    VERDICT="ACCEPTED"
else
    log_warning "Some acceptance checks failed"
    VERDICT="ACCEPTED_WITH_WARNINGS"
fi

# Step 4: Summary
echo ""
echo "================================================================"
echo "  Pipeline Complete"
echo "================================================================"
echo ""
echo "Experiment: ${EXP}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Generated artifacts:"
echo "  - OOF predictions: ${OUTPUT_DIR}/oof_predictions.jsonl"
echo "  - Evidence metrics: ${OUTPUT_DIR}/evidence_metrics.json"
echo "  - Acceptance report: ${OUTPUT_DIR}/acceptance_report.json"
echo "  - Fold checkpoints: ${OUTPUT_DIR}/fold_{0-4}/best_model/"
echo ""
echo "Verdict: ${VERDICT}"
echo ""

if [ "${VERDICT}" = "ACCEPTED" ]; then
    log_success "All quality gates passed! Ready for production."
elif [ "${VERDICT}" = "ACCEPTED_WITH_WARNINGS" ]; then
    log_warning "Pipeline complete with warnings. Review acceptance report."
    echo ""
    echo "Next steps:"
    echo "  1. Review: cat ${OUTPUT_DIR}/acceptance_report.json"
    echo "  2. Check failed gates and consider retraining with adjusted parameters"
    echo "  3. If retrieval gates failed, run: make sweep_recall EXP=${EXP}"
else
    log_error "Pipeline FAILED. Check logs for details."
    exit 1
fi

echo "================================================================"
