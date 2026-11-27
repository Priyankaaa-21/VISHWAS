"""
Unit tests for VISHWAS Phase B: Calibration and Conformal Prediction Sets
tests/test_calibration_conformal.py
"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.calibration.conformal import (
    CLASS_NAMES,
    apply_temperature_scaling,
    compute_ece_metrics,
    fit_temperature,
    get_conformal_prediction_set,
    load_calibration_params,
    run_calibration_and_conformal_analysis,
)


def test_calibration_file_exists_and_valid():
    """Verify results/calibration.json exists and contains all required calibration metrics."""
    params = load_calibration_params()
    assert params["target_coverage"] == 0.90
    assert params["alpha"] == 0.10
    assert "conformal_quantile_q" in params
    assert "score_threshold_p" in params
    assert params["score_threshold_p"] > 0.0

    # Fitted temperatures
    temps = params["fitted_temperatures"]
    assert "global" in temps and temps["global"] > 0
    assert "G1" in temps and temps["G1"] > 0
    assert "G2" in temps and temps["G2"] > 0
    assert "G3" in temps and temps["G3"] > 0

    # ECE improvement
    ece_data = params["ece"]
    assert ece_data["after"] <= ece_data["before"]
    assert len(ece_data["before_ci_95"]) == 2
    assert len(ece_data["after_ci_95"]) == 2


def test_held_out_coverage_meets_target():
    """Verify empirical coverage on held-out test split meets the 90% target (within tolerance)."""
    params = load_calibration_params()
    conf_eval = params["conformal_evaluation"]
    overall_cov = conf_eval["overall_coverage"]
    
    # Target is 90% (0.90); overall coverage should be >= 0.90
    assert overall_cov >= 0.90, f"Expected overall coverage >= 0.90, got {overall_cov}"

    # Verify per-tone group coverage and equity
    tg_data = conf_eval["per_tone_group"]
    for group in ["G1", "G2", "G3"]:
        assert group in tg_data
        cov = tg_data[group]["coverage"]
        assert cov >= 0.88, f"Group {group} coverage {cov} below tolerance of 0.88"
        assert tg_data[group]["mean_set_size"] > 1.0

    # Check fairness disparity between tone groups is < 5%
    coverages = [tg_data[g]["coverage"] for g in ["G1", "G2", "G3"]]
    disparity = max(coverages) - min(coverages)
    assert disparity < 0.05, f"Tone group coverage disparity {disparity:.4f} exceeds 5% equity threshold"


def test_conformal_sets_never_empty():
    """Verify conformal prediction sets are never empty under any probability distribution."""
    # 1. Uniform distribution
    uniform_probs = [1.0 / 11.0] * 11
    res = get_conformal_prediction_set(uniform_probs)
    assert len(res["prediction_set"]) >= 1

    # 2. Extreme peaked distribution
    peaked_probs = [0.0] * 11
    peaked_probs[7] = 1.0  # MEL
    res_peaked = get_conformal_prediction_set(peaked_probs)
    assert len(res_peaked["prediction_set"]) >= 1
    assert "MEL" in res_peaked["prediction_set"]

    # 3. Near-zero tiny probabilities
    tiny_probs = [1e-6] * 11
    res_tiny = get_conformal_prediction_set(tiny_probs)
    assert len(res_tiny["prediction_set"]) >= 1

    # 4. Dict input with arbitrary values
    dict_probs = {c: 0.01 for c in CLASS_NAMES}
    dict_probs["BCC"] = 0.89
    res_dict = get_conformal_prediction_set(dict_probs)
    assert len(res_dict["prediction_set"]) >= 1
    assert res_dict["prediction_set"][0] == "BCC"


def test_determinism_with_fixed_seed():
    """Verify conformal prediction results are perfectly deterministic across runs."""
    test_probs = [0.05, 0.40, 0.02, 0.15, 0.01, 0.02, 0.01, 0.25, 0.05, 0.02, 0.02]
    
    res1 = get_conformal_prediction_set(test_probs, tone_group="G2")
    res2 = get_conformal_prediction_set(test_probs, tone_group="G2")

    assert res1["prediction_set"] == res2["prediction_set"]
    assert res1["set_size"] == res2["set_size"]
    assert res1["temperature_applied"] == res2["temperature_applied"]
    assert res1["score_threshold"] == res2["score_threshold"]
    assert res1["is_uncertain"] == res2["is_uncertain"]


def test_conformal_uncertainty_flag():
    """Verify uncertainty flag behaves correctly for peaked vs ambiguous predictions."""
    # High certainty: one dominant class
    peaked_probs = [0.01] * 11
    peaked_probs[1] = 0.90  # BCC dominant
    res_certain = get_conformal_prediction_set(peaked_probs)
    assert res_certain["is_uncertain"] is False
    assert res_certain["set_size"] == 1

    # High uncertainty: flat/diffuse distribution
    diffuse_probs = [1.0 / 11.0] * 11
    res_uncertain = get_conformal_prediction_set(diffuse_probs)
    assert res_uncertain["is_uncertain"] is True
    assert res_uncertain["set_size"] >= 4
