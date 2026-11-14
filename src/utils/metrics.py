"""
Vishwas Utilities Module: metrics.py

Implements evaluation metrics for binary and partial ROC analysis:
- compute_pauc: Official ISIC 2024 metric (pAUC above min_tpr=0.80)
- compute_roc_auc: Standard ROC-AUC
- bootstrap_patient_metrics: Patient-grouped bootstrap 95% confidence intervals
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve


def compute_pauc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    min_tpr: float = 0.80,
) -> float:
    """
    Computes Partial Area Under the ROC Curve (pAUC) above min_tpr (default 0.80).
    Official metric of the ISIC 2024 Challenge (Skin Cancer Detection with 3D-TBP).

    Parameters:
        y_true: Binary ground truth array (0 or 1).
        y_score: Predicted probabilities/scores for positive class (1).
        min_tpr: Minimum True Positive Rate threshold (default 0.80).

    Returns:
        Partial AUC in [0, 1 - min_tpr] (max value is 0.20 when min_tpr=0.80).
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_score_arr = np.asarray(y_score, dtype=np.float64)

    if len(np.unique(y_true_arr)) < 2:
        return 0.0

    # Align with scikit-learn max_fpr convention by inverting labels and negating scores
    v_gt = abs(y_true_arr - 1.0)
    v_pred = -1.0 * y_score_arr
    max_fpr = abs(1.0 - min_tpr)

    fpr, tpr, _ = roc_curve(v_gt, v_pred)

    if max_fpr >= 1.0:
        return float(auc(fpr, tpr))
    if max_fpr <= 0.0:
        raise ValueError(f"min_tpr must be in (0, 1], got {min_tpr}")

    stop = int(np.searchsorted(fpr, max_fpr, "right"))
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr_interp = float(np.interp(max_fpr, x_interp, y_interp))

    tpr_cut = np.append(tpr[:stop], tpr_interp)
    fpr_cut = np.append(fpr[:stop], max_fpr)

    return float(auc(fpr_cut, tpr_cut))


def compute_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Computes standard ROC-AUC, returning 0.5 if only one class is present."""
    y_true_arr = np.asarray(y_true)
    if len(np.unique(y_true_arr)) < 2:
        return 0.5
    return float(roc_auc_score(y_true_arr, y_score))


def bootstrap_patient_metrics(
    df: pd.DataFrame,
    target_col: str = "target",
    prob_col: str = "prob_malignant",
    patient_col: str = "patient_id",
    n_bootstrap: int = 1000,
    seed: int = 42,
    min_tpr: float = 0.80,
) -> Dict[str, Any]:
    """
    Computes point estimates and 95% bootstrap confidence intervals for ROC-AUC
    and pAUC (min_tpr=0.80) with resampling clustered at the patient level.
    """
    rng = np.random.RandomState(seed)

    y_true = df[target_col].to_numpy()
    y_score = df[prob_col].to_numpy()

    point_auc = compute_roc_auc(y_true, y_score)
    point_pauc = compute_pauc(y_true, y_score, min_tpr=min_tpr)
    prevalence = float(np.mean(y_true))

    patient_ids = df[patient_col].unique()
    n_patients = len(patient_ids)

    # Group dataframe by patient for fast lookup
    patient_groups = {pid: grp[[target_col, prob_col]] for pid, grp in df.groupby(patient_col)}

    boot_aucs: List[float] = []
    boot_paucs: List[float] = []

    for _ in range(n_bootstrap):
        sampled_pids = rng.choice(patient_ids, size=n_patients, replace=True)
        sampled_dfs = [patient_groups[pid] for pid in sampled_pids]
        b_df = pd.concat(sampled_dfs, axis=0)

        b_true = b_df[target_col].to_numpy()
        b_score = b_df[prob_col].to_numpy()

        if len(np.unique(b_true)) < 2:
            continue

        boot_aucs.append(compute_roc_auc(b_true, b_score))
        boot_paucs.append(compute_pauc(b_true, b_score, min_tpr=min_tpr))

    if boot_aucs:
        auc_ci = (float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5)))
        pauc_ci = (float(np.percentile(boot_paucs, 2.5)), float(np.percentile(boot_paucs, 97.5)))
    else:
        auc_ci = (point_auc, point_auc)
        pauc_ci = (point_pauc, point_pauc)

    return {
        "roc_auc": point_auc,
        "roc_auc_ci": auc_ci,
        "pauc": point_pauc,
        "pauc_ci": pauc_ci,
        "prevalence": prevalence,
        "n_samples": len(df),
        "n_patients": n_patients,
        "n_malignant": int(np.sum(y_true == 1)),
        "n_benign": int(np.sum(y_true == 0)),
    }
