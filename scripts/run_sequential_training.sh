#!/usr/bin/env bash
set -eo pipefail

# Move to vishwas root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISHWAS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VISHWAS_ROOT}"

mkdir -p logs
mkdir -p reports
mkdir -p checkpoints

PYTHON_EXEC=".venv/bin/python3"

echo "========================================================================"
echo "Starting Vishwas Sequential 5-Fold Training Pipeline"
echo "Timestamp: $(date)"
echo "Host: $(hostname)"
echo "Python: $("${PYTHON_EXEC}" --version)"
echo "========================================================================"

# RUN 1: Two-Image Tabular Fusion
echo ""
echo "------------------------------------------------------------------------"
echo ">>> STEP 1: Launching Run 1 (two_image_tabular_fusion, d_cw_128)"
echo "------------------------------------------------------------------------"
"${PYTHON_EXEC}" -m src.train.train \
    --model two_image_tabular_fusion \
    --fold all \
    --epochs 12 \
    --img-size 128 \
    --mitigation class_weighted \
    --run-name d_cw_128

echo ""
echo "------------------------------------------------------------------------"
echo ">>> Run 1 (d_cw_128) Completed at: $(date)"
echo "------------------------------------------------------------------------"

# RUN 2: Dermoscopic Only
echo ""
echo "------------------------------------------------------------------------"
echo ">>> STEP 2: Launching Run 2 (dermoscopic_only, a_cw_128)"
echo "------------------------------------------------------------------------"
"${PYTHON_EXEC}" -m src.train.train \
    --model dermoscopic_only \
    --fold all \
    --epochs 12 \
    --img-size 128 \
    --mitigation class_weighted \
    --run-name a_cw_128

echo ""
echo "------------------------------------------------------------------------"
echo ">>> Run 2 (a_cw_128) Completed at: $(date)"
echo "------------------------------------------------------------------------"

# STEP 3: Fairness Audits and Cross-Run Comparison
echo ""
echo "------------------------------------------------------------------------"
echo ">>> STEP 3: Running Fairness Audits & Cross-Model Comparison"
echo "------------------------------------------------------------------------"
"${PYTHON_EXEC}" -m src.fairness.evaluate --run-dir reports/d_cw_128
"${PYTHON_EXEC}" -m src.fairness.evaluate --run-dir reports/a_cw_128
"${PYTHON_EXEC}" -m src.fairness.compare_runs --runs d_cw_128 a_cw_128

# STEP 4: Run Test Suite
echo ""
echo "------------------------------------------------------------------------"
echo ">>> STEP 4: Executing Pytest Suite"
echo "------------------------------------------------------------------------"
"${PYTHON_EXEC}" -m pytest

echo ""
echo "========================================================================"
echo "All Runs and Evaluations Completed Successfully!"
echo "Finished at: $(date)"
echo "Comparison Report available at: reports/model_fairness_comparison.md"
echo "========================================================================"
