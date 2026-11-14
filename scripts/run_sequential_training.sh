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

# RUN 3: Two-Image Tabular Fusion with Stage-1 SLICE-3D Initialized Clinical Encoder
echo ""
echo "------------------------------------------------------------------------"
echo ">>> STEP 3: Launching Run 3 (two_image_tabular_fusion, e_cw_128_s3d)"
echo "------------------------------------------------------------------------"
"${PYTHON_EXEC}" -m src.train.train \
    --model two_image_tabular_fusion \
    --fold all \
    --epochs 12 \
    --img-size 128 \
    --mitigation class_weighted \
    --init-encoder checkpoints/slice3d_pretrain_effb0.pt \
    --init-branches clinical \
    --run-name e_cw_128_s3d

echo ""
echo "------------------------------------------------------------------------"
echo ">>> Run 3 (e_cw_128_s3d) Completed at: $(date)"
echo "------------------------------------------------------------------------"

# RUN 4: Two-Image Tabular Fusion with Stage-1 SLICE-3D Initialized Dual Encoders
echo ""
echo "------------------------------------------------------------------------"
echo ">>> STEP 4: Launching Run 4 (two_image_tabular_fusion, f_cw_128_s3d_both)"
echo "------------------------------------------------------------------------"
"${PYTHON_EXEC}" -m src.train.train \
    --model two_image_tabular_fusion \
    --fold all \
    --epochs 12 \
    --img-size 128 \
    --mitigation class_weighted \
    --init-encoder checkpoints/slice3d_pretrain_effb0.pt \
    --init-branches both \
    --run-name f_cw_128_s3d_both

echo ""
echo "------------------------------------------------------------------------"
echo ">>> Run 4 (f_cw_128_s3d_both) Completed at: $(date)"
echo "------------------------------------------------------------------------"

# STEP 5: Fairness Audits and Cross-Run Comparison
echo ""
echo "------------------------------------------------------------------------"
echo ">>> STEP 5: Running Fairness Audits & Cross-Model Comparison"
echo "------------------------------------------------------------------------"
"${PYTHON_EXEC}" -m src.fairness.evaluate --run-dir reports/d_cw_128
"${PYTHON_EXEC}" -m src.fairness.evaluate --run-dir reports/a_cw_128
if [ -d "reports/e_cw_128_s3d" ] && [ -f "reports/e_cw_128_s3d/predictions.csv" ]; then
    "${PYTHON_EXEC}" -m src.fairness.evaluate --run-dir reports/e_cw_128_s3d
fi
if [ -d "reports/f_cw_128_s3d_both" ] && [ -f "reports/f_cw_128_s3d_both/predictions.csv" ]; then
    "${PYTHON_EXEC}" -m src.fairness.evaluate --run-dir reports/f_cw_128_s3d_both
fi
"${PYTHON_EXEC}" -m src.fairness.compare_runs

# STEP 6: Run Test Suite
echo ""
echo "------------------------------------------------------------------------"
echo ">>> STEP 6: Executing Pytest Suite"
echo "------------------------------------------------------------------------"
"${PYTHON_EXEC}" -m pytest

echo ""
echo "========================================================================"
echo "All Runs and Evaluations Completed Successfully!"
echo "Finished at: $(date)"
echo "Comparison Report available at: reports/model_fairness_comparison.md"
echo "========================================================================"
