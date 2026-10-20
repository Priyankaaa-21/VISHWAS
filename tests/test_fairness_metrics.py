"""
Unit tests for Vishwas fairness evaluation metrics and audit modes.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure repo and vishwas root in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VISHWAS_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(VISHWAS_ROOT) not in sys.path:
    sys.path.insert(0, str(VISHWAS_ROOT))

try:
    from vishwas.src.fairness.evaluate import compute_ece, bootstrap_ci, evaluate_run
except ImportError:
    from src.fairness.evaluate import compute_ece, bootstrap_ci, evaluate_run


def test_compute_ece_perfect_calibration():
    """ECE should be near 0 when predicted probabilities match empirical outcomes."""
    # 100 samples with 0.9 confidence and exactly 90% accuracy
    probs = np.zeros((100, 2))
    probs[:, 0] = 0.9
    probs[:, 1] = 0.1
    preds = np.zeros(100, dtype=int)
    targets = np.zeros(100, dtype=int)
    targets[90:] = 1  # 90 correct out of 100

    ece, _, _, _ = compute_ece(probs, preds, targets, n_bins=10)
    assert pytest.approx(ece, abs=1e-3) == 0.0


def test_bootstrap_ci_bounds():
    """Bootstrap 95% confidence intervals should bound empirical mean."""
    targets = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1] * 5)
    preds = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 1] * 5)

    acc_ci, f1_ci = bootstrap_ci(targets, preds, n_resamples=200, seed=42)

    assert 0.0 <= acc_ci[0] <= acc_ci[1] <= 1.0
    assert 0.0 <= f1_ci[0] <= f1_ci[1] <= 1.0


def test_evaluate_run_mode_b_without_skin_tones(tmp_path):
    """Verify Mode (b) evaluation runs on synthetic predictions lacking skin-tone groups."""
    run_dir = tmp_path / "test_run_mode_b"
    run_dir.mkdir()

    # Create synthetic predictions (DermaMNIST style)
    n = 30
    df = pd.DataFrame({
        "image_id": [f"img_{i}" for i in range(n)],
        "y_true": [i % 3 for i in range(n)],
        "y_pred": [i % 3 for i in range(n)],
        "y_prob": [0.85] * n,
    })
    df.to_csv(run_dir / "predictions.csv", index=False)

    probs = np.full((n, 3), 0.05)
    for i in range(n):
        probs[i, i % 3] = 0.90
    np.save(run_dir / "probabilities.npy", probs)

    results = evaluate_run(run_dir)
    assert results["mode"] == "mode_b"
    assert not results["has_skin_tone"]

    report_text = (run_dir / "fairness_audit_report.md").read_text()
    assert "Mode (b)" in report_text
    assert "WHY" in report_text


def test_evaluate_run_mode_a_with_skin_tones(tmp_path):
    """Verify Mode (a) evaluation runs on synthetic predictions with skin-tone groups."""
    run_dir = tmp_path / "test_run_mode_a"
    run_dir.mkdir()

    # Create synthetic predictions (Fitzpatrick style)
    groups = ["light", "medium", "dark"]
    n = 30
    df = pd.DataFrame({
        "image_id": [f"fitz_{i}" for i in range(n)],
        "y_true": [i % 3 for i in range(n)],
        "y_pred": [i % 3 for i in range(n)],
        "y_prob": [0.88] * n,
        "skin_tone_group": [groups[i % 3] for i in range(n)],
        "fitzpatrick": [(i % 6) + 1 for i in range(n)],
    })
    df.to_csv(run_dir / "predictions.csv", index=False)

    probs = np.full((n, 3), 0.05)
    for i in range(n):
        probs[i, i % 3] = 0.90
    np.save(run_dir / "probabilities.npy", probs)

    results = evaluate_run(run_dir)
    assert results["mode"] == "mode_a"
    assert results["has_skin_tone"]
    assert "subgroups" in results
    assert "disparity_gap" in results

    report_text = (run_dir / "fairness_audit_report.md").read_text()
    assert "Mode (a)" in report_text
    assert "Worst-Group F1 Score" in report_text
