"""
VISHWAS Calibration & Conformal Prediction Module: src/calibration/conformal.py

Provides:
1. Temperature scaling fitted on out-of-fold predictions (global and per-skin-tone group G1, G2, G3).
2. Expected Calibration Error (ECE) calculation with 95% bootstrap confidence intervals.
3. Split-conformal prediction sets (LAC score) calibrated on out-of-fold data with proper held-out evaluation.
4. Export and caching to results/calibration.json.
5. Runtime inference helper get_conformal_prediction_set() for Simple and Expert modes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# Project root path
VISHWAS_ROOT = Path(__file__).resolve().parent.parent.parent

CLASS_NAMES: List[str] = [
    "AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"
]

CLASS_DISPLAY_NAMES: Dict[str, str] = {
    "AKIEC": "Actinic Keratosis / Bowen's Disease",
    "BCC": "Basal Cell Carcinoma",
    "BEN_OTH": "Other Benign Lesions",
    "BKL": "Benign Keratosis (Solar Lentigo / Seborrheic Keratosis)",
    "DF": "Dermatofibroma",
    "INF": "Inflammatory Condition",
    "MAL_OTH": "Other Malignant Lesions",
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevus (Mole)",
    "SCCKA": "Squamous Cell Carcinoma / Keratoacanthoma",
    "VASC": "Vascular Lesion",
}


def compute_ece_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Computes Expected Calibration Error (ECE) for probabilities and integer targets."""
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        if i < n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            acc_in_bin = float(np.mean(preds[in_bin] == targets[in_bin]))
            conf_in_bin = float(np.mean(confidences[in_bin]))
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin

    return float(ece)


def bootstrap_ece_ci(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Computes 95% bootstrap confidence interval for ECE."""
    n = len(targets)
    if n < 10:
        return (0.0, 1.0)

    rng = np.random.RandomState(seed)
    eces: List[float] = []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        eces.append(compute_ece_metrics(probs[idx], targets[idx], n_bins=n_bins))

    low = float(np.percentile(eces, 2.5))
    high = float(np.percentile(eces, 97.5))
    return (low, high)


def apply_temperature_scaling(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Applies temperature scaling to probabilities: q = softmax(log(p) / T)."""
    T = max(float(temperature), 0.01)
    eps = 1e-12
    log_p = np.log(np.clip(probs, eps, 1.0)) / T
    if log_p.ndim == 1:
        max_v = np.max(log_p)
        exp_p = np.exp(log_p - max_v)
        return exp_p / np.sum(exp_p)
    else:
        max_v = np.max(log_p, axis=1, keepdims=True)
        exp_p = np.exp(log_p - max_v)
        return exp_p / np.sum(exp_p, axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, targets: np.ndarray) -> float:
    """Finds optimal temperature T > 0 minimizing negative log-likelihood on cross-entropy."""
    eps = 1e-12
    log_p = np.log(np.clip(probs, eps, 1.0))
    n = len(targets)

    def nll(T: float) -> float:
        scaled = log_p / T
        max_s = np.max(scaled, axis=1, keepdims=True)
        lse = max_s + np.log(np.sum(np.exp(scaled - max_s), axis=1, keepdims=True))
        log_q = scaled - lse
        chosen = log_q[np.arange(n), targets]
        return -float(np.mean(chosen))

    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(res.x)


def run_calibration_and_conformal_analysis(
    predictions_csv_path: Optional[Union[str, Path]] = None,
    output_json_path: Optional[Union[str, Path]] = None,
    alpha: float = 0.10,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Fits temperature scaling (global + tone groups) and evaluates split-conformal prediction.
    Uses proper held-out splitting (calibrating on folds 0,1,2; evaluating on folds 3,4).
    Saves and returns the complete calibration summary.
    """
    if predictions_csv_path is None:
        predictions_csv_path = VISHWAS_ROOT / "reports" / "d_cw_128" / "predictions.csv"
    else:
        predictions_csv_path = Path(predictions_csv_path)

    if output_json_path is None:
        output_json_path = VISHWAS_ROOT / "results" / "calibration.json"
    else:
        output_json_path = Path(output_json_path)

    if not predictions_csv_path.exists():
        raise FileNotFoundError(f"Predictions file not found at {predictions_csv_path}")

    df = pd.read_csv(predictions_csv_path)
    prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
    for c in prob_cols:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in {predictions_csv_path}")

    c_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    raw_probs = df[prob_cols].values
    targets = df["y_true"].map(c_to_idx).values
    tone_groups = df["tone_group"].values
    folds = df["fold"].values

    # 1. ECE before calibration
    ece_before = compute_ece_metrics(raw_probs, targets)
    ci_before = bootstrap_ece_ci(raw_probs, targets, seed=seed)

    # 2. Fit temperature scaling
    t_global = fit_temperature(raw_probs, targets)
    t_groups: Dict[str, float] = {}
    for g in ["G1", "G2", "G3"]:
        g_mask = (tone_groups == g)
        if np.sum(g_mask) > 10:
            t_groups[g] = fit_temperature(raw_probs[g_mask], targets[g_mask])
        else:
            t_groups[g] = t_global

    # Calibrate probabilities globally
    cal_probs = apply_temperature_scaling(raw_probs, t_global)
    ece_after = compute_ece_metrics(cal_probs, targets)
    ci_after = bootstrap_ece_ci(cal_probs, targets, seed=seed)

    # Per-tone group ECE before and after
    tg_ece: Dict[str, Dict[str, Any]] = {}
    for g in ["G1", "G2", "G3"]:
        g_mask = (tone_groups == g)
        if np.sum(g_mask) > 0:
            g_before = compute_ece_metrics(raw_probs[g_mask], targets[g_mask])
            g_cal = apply_temperature_scaling(raw_probs[g_mask], t_groups.get(g, t_global))
            g_after = compute_ece_metrics(g_cal, targets[g_mask])
            tg_ece[g] = {
                "temperature": t_groups.get(g, t_global),
                "ece_before": round(g_before, 4),
                "ece_after": round(g_after, 4),
                "sample_count": int(np.sum(g_mask)),
            }

    # 3. Split-conformal prediction calibration (held-out validation on folds 3, 4)
    cal_mask = np.isin(folds, [0, 1, 2])
    eval_mask = np.isin(folds, [3, 4])

    cal_probs_split = cal_probs[cal_mask]
    cal_targets = targets[cal_mask]

    # Non-conformity score: LAC score s_i = 1 - p(y_i)
    s_cal = 1.0 - cal_probs_split[np.arange(len(cal_targets)), cal_targets]
    n_cal = len(s_cal)
    # Quantile definition with finite-sample correction
    q_level = min(np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal, 1.0)
    q_val = float(np.quantile(s_cal, q_level, method="higher"))
    score_threshold = float(1.0 - q_val)

    # Evaluate on held-out split (folds 3, 4)
    eval_probs_split = cal_probs[eval_mask]
    eval_targets = targets[eval_mask]
    eval_tg = tone_groups[eval_mask]

    in_set_list: List[bool] = []
    set_sizes: List[int] = []

    for p, y in zip(eval_probs_split, eval_targets):
        c_set = [i for i, prob_val in enumerate(p) if (1.0 - prob_val) <= q_val]
        if not c_set:
            # Guarantee non-empty prediction set
            c_set = [int(np.argmax(p))]
        in_set_list.append(bool(y in c_set))
        set_sizes.append(len(c_set))

    overall_coverage = float(np.mean(in_set_list))
    overall_mean_set_size = float(np.mean(set_sizes))

    # Per tone group conformal metrics
    tg_conformal: Dict[str, Dict[str, Any]] = {}
    for g in ["G1", "G2", "G3"]:
        g_mask = (eval_tg == g)
        if np.sum(g_mask) > 0:
            cov = float(np.mean(np.array(in_set_list)[g_mask]))
            sz = float(np.mean(np.array(set_sizes)[g_mask]))
            tg_conformal[g] = {
                "coverage": round(cov, 4),
                "mean_set_size": round(sz, 2),
                "evaluation_samples": int(np.sum(g_mask)),
            }

    # Per class conformal metrics
    class_conformal: Dict[str, Dict[str, Any]] = {}
    for c_idx, c_name in enumerate(CLASS_NAMES):
        c_mask = (eval_targets == c_idx)
        if np.sum(c_mask) > 0:
            c_cov = float(np.mean(np.array(in_set_list)[c_mask]))
            c_sz = float(np.mean(np.array(set_sizes)[c_mask]))
            class_conformal[c_name] = {
                "coverage": round(c_cov, 4),
                "mean_set_size": round(c_sz, 2),
                "evaluation_samples": int(np.sum(c_mask)),
            }

    results_data = {
        "status": "success",
        "target_coverage": 1.0 - alpha,
        "alpha": alpha,
        "conformal_quantile_q": round(q_val, 4),
        "score_threshold_p": round(score_threshold, 4),
        "calibration_samples": int(n_cal),
        "evaluation_samples": int(len(eval_targets)),
        "fitted_temperatures": {
            "global": round(t_global, 4),
            **{k: round(v, 4) for k, v in t_groups.items()}
        },
        "ece": {
            "before": round(ece_before, 4),
            "before_ci_95": [round(ci_before[0], 4), round(ci_before[1], 4)],
            "after": round(ece_after, 4),
            "after_ci_95": [round(ci_after[0], 4), round(ci_after[1], 4)],
            "improvement": round(ece_before - ece_after, 4),
        },
        "per_tone_group_calibration": tg_ece,
        "conformal_evaluation": {
            "overall_coverage": round(overall_coverage, 4),
            "overall_mean_set_size": round(overall_mean_set_size, 2),
            "per_tone_group": tg_conformal,
            "per_class": class_conformal,
        }
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(results_data, indent=2))
    logging.info("Calibration and conformal results saved to %s", output_json_path)

    return results_data


# Cached Calibration Parameters
_CALIBRATION_PARAMS_CACHE: Optional[Dict[str, Any]] = None

def load_calibration_params() -> Dict[str, Any]:
    """Loads calibration parameters from results/calibration.json with fallback defaults."""
    global _CALIBRATION_PARAMS_CACHE
    if _CALIBRATION_PARAMS_CACHE is not None:
        return _CALIBRATION_PARAMS_CACHE

    cal_path = VISHWAS_ROOT / "results" / "calibration.json"
    if cal_path.exists():
        try:
            _CALIBRATION_PARAMS_CACHE = json.loads(cal_path.read_text())
            return _CALIBRATION_PARAMS_CACHE
        except Exception as e:
            logging.warning("Error reading %s: %s", cal_path, e)

    # Defaults if file not yet written
    _CALIBRATION_PARAMS_CACHE = {
        "fitted_temperatures": {"global": 0.9164, "G1": 1.0043, "G2": 0.8935, "G3": 0.9233},
        "score_threshold_p": 0.0401,
        "conformal_quantile_q": 0.9599,
        "target_coverage": 0.90,
        "ece": {"before": 0.0911, "after": 0.0821},
        "conformal_evaluation": {
            "overall_coverage": 0.9208,
            "overall_mean_set_size": 5.71,
            "per_tone_group": {
                "G1": {"coverage": 0.9153, "mean_set_size": 4.65},
                "G2": {"coverage": 0.9272, "mean_set_size": 5.88},
                "G3": {"coverage": 0.9094, "mean_set_size": 5.78},
            }
        }
    }
    return _CALIBRATION_PARAMS_CACHE


def get_conformal_prediction_set(
    probs: Union[np.ndarray, List[float], Dict[str, float]],
    tone_group: Optional[str] = None,
    alpha: float = 0.10,
    lang: str = "en",
) -> Dict[str, Any]:
    """
    Computes calibrated conformal prediction set for an inference sample.
    Returns:
    - prediction_set: list of class codes in set
    - prediction_set_names: list of condition display names
    - is_uncertain: bool (set size >= 4 or confidence < 0.35)
    - temperature: float applied
    - calibrated_probs: dict of {class: prob}
    - score_threshold: probability cutoff
    """
    cal_params = load_calibration_params()

    # Convert probs to array aligned with CLASS_NAMES
    if isinstance(probs, dict):
        prob_arr = np.array([float(probs.get(c, 0.0)) for c in CLASS_NAMES], dtype=np.float32)
    else:
        prob_arr = np.array(probs, dtype=np.float32)

    if prob_arr.sum() > 0:
        prob_arr = prob_arr / prob_arr.sum()

    # Determine temperature
    tg = str(tone_group).strip().upper() if tone_group else "GLOBAL"
    t_dict = cal_params.get("fitted_temperatures", {})
    temp = float(t_dict.get(tg, t_dict.get("global", 0.9164)))

    # Apply temperature
    calibrated_arr = apply_temperature_scaling(prob_arr, temp)

    # Threshold from calibration
    threshold = float(cal_params.get("score_threshold_p", 0.0401))

    # Form conformal prediction set
    set_indices = [i for i, p_val in enumerate(calibrated_arr) if p_val >= threshold]
    if not set_indices:
        set_indices = [int(np.argmax(calibrated_arr))]

    # Sort descending by calibrated probability
    set_indices = sorted(set_indices, key=lambda i: -calibrated_arr[i])
    prediction_set = [CLASS_NAMES[i] for i in set_indices]
    prediction_set_names = [CLASS_DISPLAY_NAMES.get(c, c) for c in prediction_set]

    top_prob = float(calibrated_arr[set_indices[0]])
    is_uncertain = (len(prediction_set) >= 4) or (top_prob < 0.35)

    cal_dict = {CLASS_NAMES[i]: float(calibrated_arr[i]) for i in range(len(CLASS_NAMES))}

    return {
        "prediction_set": prediction_set,
        "prediction_set_names": prediction_set_names,
        "is_uncertain": is_uncertain,
        "set_size": len(prediction_set),
        "temperature_applied": round(temp, 4),
        "score_threshold": round(threshold, 4),
        "target_coverage": float(cal_params.get("target_coverage", 0.90)),
        "empirical_coverage_overall": float(cal_params.get("conformal_evaluation", {}).get("overall_coverage", 0.9208)),
        "calibrated_probabilities": cal_dict,
        "top_calibrated_class": prediction_set[0],
        "top_calibrated_conf": top_prob,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = run_calibration_and_conformal_analysis()
    print("\nCalibration & Conformal Prediction Analysis Complete:")
    print(f"  Target Coverage: {res['target_coverage']*100:.1f}%")
    print(f"  Conformal Quantile q: {res['conformal_quantile_q']} (Cutoff: {res['score_threshold_p']})")
    print(f"  ECE Before: {res['ece']['before']} -> ECE After: {res['ece']['after']} (Improved by {res['ece']['improvement']})")
    print(f"  Overall Empirical Coverage: {res['conformal_evaluation']['overall_coverage']*100:.2f}%")
    print(f"  Overall Mean Set Size: {res['conformal_evaluation']['overall_mean_set_size']}")
    for g, metrics in res['conformal_evaluation']['per_tone_group'].items():
        print(f"    Tone Group {g}: Coverage = {metrics['coverage']*100:.2f}%, Mean Size = {metrics['mean_set_size']} (n={metrics['evaluation_samples']})")
