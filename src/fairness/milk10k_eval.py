"""
Vishwas Fairness Evaluation Module: milk10k_eval.py

Evaluates predictions for MILK10k 5-fold cross-validation or single-fold runs:
- Overall: Accuracy, Macro-F1 (headline 10-class excluding MAL_OTH, plus 11-class),
  Balanced Accuracy, Macro-AUC (one-vs-rest, skipping classes without positive samples),
  per-class Recall and Precision, Confusion Matrix.
- Per Tone Group (G1, G2, G3): n lesions, Macro-F1, Balanced Accuracy, per-class Recall,
  worst-group value and best-worst gap, Expected Calibration Error (ECE), reliability curves.
- Bootstrap 95% Confidence Intervals (1,000 resamples) resampling LESIONS.
- Warnings for every group x class cell with fewer than 10 lesions.
- Separate reporting of MAL_OTH (retained out of headline macro metrics).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("vishwas.fairness.milk10k")

# Ground truth diagnostic classes in exact order
MILK10K_CLASSES: List[str] = [
    "AKIEC",
    "BCC",
    "BEN_OTH",
    "BKL",
    "DF",
    "INF",
    "MAL_OTH",
    "MEL",
    "NV",
    "SCCKA",
    "VASC",
]

# Headline classes exclude MAL_OTH (only 9 samples total in dataset, 0 in G3)
HEADLINE_CLASSES: List[str] = [c for c in MILK10K_CLASSES if c != "MAL_OTH"]


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Computes Expected Calibration Error (ECE) and bin calibration points."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_accs, bin_confs, bin_sizes = [], [], []

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        if i < n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)

        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = float(np.mean(predictions[in_bin] == targets[in_bin]))
            conf_in_bin = float(np.mean(confidences[in_bin]))
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
            bin_accs.append(acc_in_bin)
            bin_confs.append(conf_in_bin)
            bin_sizes.append(int(np.sum(in_bin)))
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)
            bin_sizes.append(0)

    return float(ece), np.array(bin_accs), np.array(bin_confs), np.array(bin_sizes)


def compute_macro_auc(
    y_true: np.ndarray,
    probs: np.ndarray,
    classes: List[str],
) -> Tuple[float, List[str], Dict[str, float]]:
    """
    Computes One-vs-Rest Macro-AUC, skipping classes with no positive samples.
    Returns: (macro_auc, skipped_classes, per_class_auc)
    """
    per_class_auc: Dict[str, float] = {}
    skipped_classes: List[str] = []

    for i, c in enumerate(classes):
        bin_targets = (y_true == c).astype(int)
        pos_count = np.sum(bin_targets)
        neg_count = len(bin_targets) - pos_count

        if pos_count == 0:
            skipped_classes.append(f"{c} (0 positive samples)")
            continue
        elif neg_count == 0:
            skipped_classes.append(f"{c} (0 negative samples)")
            continue

        try:
            auc = roc_auc_score(bin_targets, probs[:, i])
            per_class_auc[c] = float(auc)
        except Exception as e:
            skipped_classes.append(f"{c} (AUC error: {e})")

    macro_auc = float(np.mean(list(per_class_auc.values()))) if per_class_auc else 0.0
    return macro_auc, skipped_classes, per_class_auc


def bootstrap_lesion_metrics(
    df: pd.DataFrame,
    classes: List[str],
    headline_classes: List[str],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Computes 95% bootstrap confidence intervals resampling LESIONS with replacement.
    Includes Accuracy, Headline Macro-F1, Balanced Accuracy, and ECE.
    """
    n = len(df)
    if n < 5:
        return {
            "accuracy": (0.0, 1.0),
            "macro_f1": (0.0, 1.0),
            "balanced_accuracy": (0.0, 1.0),
            "ece": (0.0, 1.0),
        }

    rng = np.random.RandomState(seed)
    accs, f1s, b_accs, eces = [], [], [], []

    prob_cols = [f"prob_{c}" for c in classes if f"prob_{c}" in df.columns]
    has_probs = len(prob_cols) == len(classes)

    for _ in range(n_resamples):
        boot_idx = rng.choice(n, size=n, replace=True)
        sub = df.iloc[boot_idx]
        yt = sub["y_true"].values
        yp = sub["y_pred"].values

        accs.append(accuracy_score(yt, yp))
        # Headline macro-F1 (excluding MAL_OTH)
        f1s.append(f1_score(yt, yp, labels=headline_classes, average="macro", zero_division=0))
        # Balanced accuracy across classes present
        b_accs.append(balanced_accuracy_score(yt, yp))

        if has_probs:
            sub_probs = sub[prob_cols].values
            sub_confs = np.max(sub_probs, axis=1)
            b_ece, _, _, _ = compute_ece(sub_confs, yp, yt, n_bins=10)
            eces.append(b_ece)

    res = {
        "accuracy": (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))),
        "macro_f1": (float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))),
        "balanced_accuracy": (float(np.percentile(b_accs, 2.5)), float(np.percentile(b_accs, 97.5))),
    }
    if eces:
        res["ece"] = (float(np.percentile(eces, 2.5)), float(np.percentile(eces, 97.5)))
    else:
        res["ece"] = (0.0, 1.0)
    return res


def evaluate_milk10k_predictions(
    predictions_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    classes: Optional[List[str]] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of MILK10k predictions:
    - Overall performance (Accuracy, Balanced Accuracy, Headline Macro-F1, 11-Class Macro-F1, Macro-AUC)
    - Subgroup evaluation across tone_group (G1, G2, G3)
    - Calibration error (ECE) and reliability diagrams
    - Disparity analysis (worst group score and best-worst gap)
    - Bootstrap 95% CIs resampling lesions
    - Low sample size warnings for (group x class) cells with < 10 lesions
    - Separate reporting for MAL_OTH
    """
    df = predictions_df.copy()
    all_classes = classes or MILK10K_CLASSES
    headline_classes = [c for c in all_classes if c != "MAL_OTH"]

    # Extract probability matrix
    prob_cols = [f"prob_{c}" for c in all_classes if f"prob_{c}" in df.columns]
    if len(prob_cols) == len(all_classes):
        probs = df[prob_cols].values
    else:
        # Fallback if probability columns are not named prob_<class>
        probs = np.zeros((len(df), len(all_classes)))
        for i, c in enumerate(all_classes):
            if c in df.columns:
                probs[:, i] = df[c].values

    y_true = df["y_true"].astype(str).values
    y_pred = df["y_pred"].astype(str).values

    # Max confidence for ECE
    confidences = np.max(probs, axis=1) if probs.shape[1] > 0 else np.ones(len(df))

    # 1. Overall Metrics
    overall_acc = float(accuracy_score(y_true, y_pred))
    overall_bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    headline_macro_f1 = float(f1_score(y_true, y_pred, labels=headline_classes, average="macro", zero_division=0))
    full_macro_f1 = float(f1_score(y_true, y_pred, labels=all_classes, average="macro", zero_division=0))
    overall_auc, skipped_auc_classes, per_class_auc = compute_macro_auc(y_true, probs, all_classes)
    overall_ece, bin_accs, bin_confs, bin_sizes = compute_ece(confidences, y_pred, y_true, n_bins=10)

    # Per-class recall and precision
    per_class_recall: Dict[str, float] = {}
    per_class_precision: Dict[str, float] = {}
    for c in all_classes:
        per_class_recall[c] = float(recall_score(y_true == c, y_pred == c, zero_division=0))
        per_class_precision[c] = float(precision_score(y_true == c, y_pred == c, zero_division=0))

    # Bootstrap CIs for overall
    overall_cis = bootstrap_lesion_metrics(df, all_classes, headline_classes, n_resamples=n_bootstrap, seed=seed)

    # 2. MAL_OTH Dedicated Summary
    mal_mask = (y_true == "MAL_OTH")
    mal_n = int(np.sum(mal_mask))
    mal_recall = per_class_recall.get("MAL_OTH", 0.0)
    mal_precision = per_class_precision.get("MAL_OTH", 0.0)

    # 3. Low-count cell warnings (< 10 lesions)
    low_count_warnings: List[str] = []
    tone_groups_present = [g for g in ["G1", "G2", "G3"] if g in df["tone_group"].values]
    if "unknown" in df["tone_group"].values:
        tone_groups_present.append("unknown")

    ct_group_class = pd.crosstab(df["tone_group"], df["y_true"])
    for g in tone_groups_present:
        for c in all_classes:
            cnt = int(ct_group_class.loc[g, c]) if g in ct_group_class.index and c in ct_group_class.columns else 0
            if cnt < 10:
                msg = f"Warning: Too few samples for {g} x {c} (n = {cnt} lesions < 10)"
                low_count_warnings.append(msg)
                print(f"[FAIRNESS WARNING] {msg}")

    # 4. Per Tone Group Metrics
    per_group_metrics: Dict[str, Dict[str, Any]] = {}
    for g in tone_groups_present:
        sub_df = df[df["tone_group"] == g].copy()
        n_g = len(sub_df)
        if n_g == 0:
            continue

        yt_g = sub_df["y_true"].values
        yp_g = sub_df["y_pred"].values
        sub_probs = probs[df["tone_group"] == g]
        sub_confs = confidences[df["tone_group"] == g]

        g_acc = float(accuracy_score(yt_g, yp_g))
        g_bal_acc = float(balanced_accuracy_score(yt_g, yp_g))
        g_headline_f1 = float(f1_score(yt_g, yp_g, labels=headline_classes, average="macro", zero_division=0))
        g_full_f1 = float(f1_score(yt_g, yp_g, labels=all_classes, average="macro", zero_division=0))
        g_ece, g_bin_acc, g_bin_conf, _ = compute_ece(sub_confs, yp_g, yt_g, n_bins=10)

        # Per class recall for this group
        g_recall = {c: float(recall_score(yt_g == c, yp_g == c, zero_division=0)) for c in all_classes}
        g_cis = bootstrap_lesion_metrics(sub_df, all_classes, headline_classes, n_resamples=n_bootstrap, seed=seed)

        per_group_metrics[g] = {
            "n_lesions": n_g,
            "accuracy": g_acc,
            "balanced_accuracy": g_bal_acc,
            "headline_macro_f1": g_headline_f1,
            "full_macro_f1": g_full_f1,
            "ece": g_ece,
            "per_class_recall": g_recall,
            "cis": g_cis,
        }

    # Disparity Gap & Worst Group (among primary evaluation groups G1, G2, G3)
    eval_groups = [g for g in ["G1", "G2", "G3"] if g in per_group_metrics]
    if eval_groups:
        group_f1s = {g: per_group_metrics[g]["headline_macro_f1"] for g in eval_groups}
        worst_group = min(group_f1s, key=group_f1s.get)
        best_group = max(group_f1s, key=group_f1s.get)
        worst_f1 = group_f1s[worst_group]
        f1_gap = group_f1s[best_group] - group_f1s[worst_group]

        group_bal_accs = {g: per_group_metrics[g]["balanced_accuracy"] for g in eval_groups}
        worst_bal_group = min(group_bal_accs, key=group_bal_accs.get)
        bal_acc_gap = max(group_bal_accs.values()) - min(group_bal_accs.values())
    else:
        worst_group = "N/A"
        worst_f1 = 0.0
        f1_gap = 0.0
        bal_acc_gap = 0.0

    # 5. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)

    # 6. Save Plots & Markdown Report if output_dir provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save Confusion Matrix Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=all_classes,
            yticklabels=all_classes,
            ax=ax,
        )
        ax.set_title("MILK10k Confusion Matrix", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted Diagnosis")
        ax.set_ylabel("True Diagnosis")
        plt.tight_layout()
        cm_path = output_dir / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=300)
        plt.close(fig)

        # Save Reliability Curves Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.plot(bin_confs, bin_accs, "s-", color="#1f77b4", label=f"Overall (n={len(df):,}, ECE: {overall_ece:.3f})")

        palette = {"G1": "#2ca02c", "G2": "#ff7f0e", "G3": "#d62728"}
        group_plot_labels = {
            "G1": "G1 = {0,1,2}",
            "G2": "G2 = {3}",
            "G3": "G3 = {4,5}",
            "unknown": "unknown",
        }
        for g in eval_groups:
            sub_probs = probs[df["tone_group"] == g]
            sub_confs = confidences[df["tone_group"] == g]
            sub_preds = y_pred[df["tone_group"] == g]
            sub_true = y_true[df["tone_group"] == g]
            n_g = len(sub_true)
            g_ece, g_accs, g_confs, _ = compute_ece(sub_confs, sub_preds, sub_true, n_bins=10)
            color = palette.get(g, "#9467bd")
            ax.plot(g_confs, g_accs, "o--", color=color, label=f"{group_plot_labels.get(g, g)} (n={n_g:,}, ECE: {g_ece:.3f})")

        ax.set_title("Reliability Diagrams Across Skin Tone Groups (G1={0,1,2}, G2={3}, G3={4,5})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower right")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        rc_path = output_dir / "reliability_curves.png"
        fig.savefig(rc_path, dpi=300)
        plt.close(fig)

        # Generate Markdown Audit Report
        report_text = generate_evaluation_markdown(
            overall_acc=overall_acc,
            overall_bal_acc=overall_bal_acc,
            headline_macro_f1=headline_macro_f1,
            full_macro_f1=full_macro_f1,
            overall_auc=overall_auc,
            overall_ece=overall_ece,
            overall_cis=overall_cis,
            skipped_auc_classes=skipped_auc_classes,
            per_class_recall=per_class_recall,
            per_class_precision=per_class_precision,
            mal_n=mal_n,
            mal_recall=mal_recall,
            mal_precision=mal_precision,
            per_group_metrics=per_group_metrics,
            worst_group=worst_group,
            worst_f1=worst_f1,
            f1_gap=f1_gap,
            bal_acc_gap=bal_acc_gap,
            low_count_warnings=low_count_warnings,
            classes=all_classes,
            cm=cm,
        )
        report_path = output_dir / "fairness_audit_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

    return {
        "overall": {
            "accuracy": overall_acc,
            "balanced_accuracy": overall_bal_acc,
            "headline_macro_f1": headline_macro_f1,
            "full_macro_f1": full_macro_f1,
            "macro_auc": overall_auc,
            "ece": overall_ece,
            "cis": overall_cis,
            "skipped_auc_classes": skipped_auc_classes,
            "per_class_recall": per_class_recall,
            "per_class_precision": per_class_precision,
        },
        "mal_oth": {
            "n_lesions": mal_n,
            "recall": mal_recall,
            "precision": mal_precision,
        },
        "per_group": per_group_metrics,
        "disparity": {
            "worst_group": worst_group,
            "worst_f1": worst_f1,
            "f1_gap": f1_gap,
            "balanced_acc_gap": bal_acc_gap,
        },
        "low_count_warnings": low_count_warnings,
    }


def generate_evaluation_markdown(
    overall_acc: float,
    overall_bal_acc: float,
    headline_macro_f1: float,
    full_macro_f1: float,
    overall_auc: float,
    overall_ece: float,
    overall_cis: Dict[str, Tuple[float, float]],
    skipped_auc_classes: List[str],
    per_class_recall: Dict[str, float],
    per_class_precision: Dict[str, float],
    mal_n: int,
    mal_recall: float,
    mal_precision: float,
    per_group_metrics: Dict[str, Dict[str, Any]],
    worst_group: str,
    worst_f1: float,
    f1_gap: float,
    bal_acc_gap: float,
    low_count_warnings: List[str],
    classes: List[str],
    cm: np.ndarray,
) -> str:
    """Formats fairness and diagnostic audit results into Markdown."""
    lines: List[str] = []

    lines.append("# MILK10k Fairness Auditing & Subgroup Evaluation Report")
    lines.append("")
    lines.append("> **Evaluation Engine**: Vishwas Fairness Auditor | **Resamples**: 1,000 (Bootstrap on Lesions)")
    lines.append("")

    # Executive Summary Table
    lines.append("## 1. Overall Performance Metrics")
    lines.append("")
    lines.append("| Metric | Headline Value | 95% Bootstrap CI | Notes |")
    lines.append("| :--- | :---: | :---: | :--- |")
    lines.append(f"| **Overall Accuracy** | `{overall_acc * 100:.2f}%` | [{overall_cis['accuracy'][0]*100:.2f}%, {overall_cis['accuracy'][1]*100:.2f}%] | All evaluated lesions |")
    lines.append(f"| **Headline Macro-F1** | `{headline_macro_f1:.4f}` | [{overall_cis['macro_f1'][0]:.4f}, {overall_cis['macro_f1'][1]:.4f}] | **Excludes `MAL_OTH`** (10 diagnostic classes) |")
    lines.append(f"| **Full Macro-F1** | `{full_macro_f1:.4f}` | — | Includes all 11 classes |")
    lines.append(f"| **Balanced Accuracy** | `{overall_bal_acc * 100:.2f}%` | [{overall_cis['balanced_accuracy'][0]*100:.2f}%, {overall_cis['balanced_accuracy'][1]*100:.2f}%] | Average recall across classes |")
    lines.append(f"| **Macro-AUC (OvR)** | `{overall_auc:.4f}` | — | One-vs-Rest ROC-AUC |")
    lines.append(f"| **Calibration (ECE)** | `{overall_ece:.4f}` | [{overall_cis.get('ece', (0.0, 1.0))[0]:.4f}, {overall_cis.get('ece', (0.0, 1.0))[1]:.4f}] | Expected Calibration Error |")
    lines.append("")

    if skipped_auc_classes:
        lines.append("> [!NOTE]")
        lines.append("> **AUC Calculation Note**: The following classes lacked positive samples in the evaluated slice and were skipped from Macro-AUC averaging:")
        for sc in skipped_auc_classes:
            lines.append(f"> - `{sc}`")
        lines.append("")

    # Dedicated MAL_OTH section
    lines.append("## 2. Rare Malignancy Audit: `MAL_OTH`")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append(f"> As required by the evaluation protocol, **`MAL_OTH` is excluded from headline macro metrics** due to extreme dataset sparsity ($n = {mal_n}$ lesions in total, 0 lesions in G3). It is audited independently below:")
    lines.append("")
    lines.append("| Rare Diagnosis | Evaluated Lesions ($n$) | Recall | Precision | Subgroup Observations |")
    lines.append("| :--- | :---: | :---: | :---: | :--- |")
    lines.append(f"| **`MAL_OTH`** | $n = {mal_n}$ | `{mal_recall * 100:.2f}%` | `{mal_precision * 100:.2f}%` | Zero representation in G3; 9 samples globally |")
    lines.append("")

    # Per Tone Group Table
    lines.append("## 3. Subgroup Fairness Metrics Across Tone Groups")
    lines.append("")
    lines.append("> **Tone Groups defined by `skin_tone_class` values**: `G1 = {0, 1, 2}`, `G2 = {3}`, `G3 = {4, 5}`.")
    lines.append("")
    lines.append("| Tone Group (`skin_tone_class`) | Lesions ($n$) | Headline Macro-F1 (95% CI) | Balanced Accuracy (95% CI) | Accuracy (95% CI) | Calibration ECE (95% CI) |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
    group_labels_full = {
        "G1": "**`G1`** (`skin_tone_class: {0, 1, 2}`)",
        "G2": "**`G2`** (`skin_tone_class: {3}`)",
        "G3": "**`G3`** (`skin_tone_class: {4, 5}`)",
        "unknown": "**`unknown`** (missing)",
    }
    for g in ["G1", "G2", "G3", "unknown"]:
        if g in per_group_metrics:
            gm = per_group_metrics[g]
            ci_f1 = gm["cis"]["macro_f1"]
            ci_bacc = gm["cis"]["balanced_accuracy"]
            ci_acc = gm["cis"]["accuracy"]
            ci_ece = gm["cis"].get("ece", (0.0, 1.0))
            lines.append(
                f"| {group_labels_full.get(g, g)} | $n = {gm['n_lesions']:,}$ | "
                f"**`{gm['headline_macro_f1']:.4f}`** ([{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]) | "
                f"`{gm['balanced_accuracy']*100:.2f}%` ([{ci_bacc[0]*100:.2f}%, {ci_bacc[1]*100:.2f}%]) | "
                f"`{gm['accuracy']*100:.2f}%` ([{ci_acc[0]*100:.2f}%, {ci_acc[1]*100:.2f}%]) | "
                f"`{gm['ece']:.4f}` ([{ci_ece[0]:.4f}, {ci_ece[1]:.4f}]) |"
            )
    lines.append("")

    # Disparity Gap Box
    lines.append("### 3.1 Subgroup Disparity & Worst-Group Score")
    lines.append("")
    worst_n = per_group_metrics.get(worst_group, {}).get("n_lesions", 0)
    lines.append(f"- **Worst Performing Group**: **`{worst_group}`** ($n = {worst_n:,}$, Macro-F1: `{worst_f1:.4f}`)")
    lines.append(f"- **Best-Worst Headline Macro-F1 Gap**: **`{f1_gap:.4f}`**")
    lines.append(f"- **Balanced Accuracy Gap**: **`{bal_acc_gap * 100:.2f}%`**")
    lines.append("")

    # Per Class Recall Across Groups
    lines.append("## 4. Per-Class Recall Breakdown Across Groups")
    lines.append("")
    lines.append("| Diagnosis Class | Overall Recall | Overall Precision | G1 Recall | G2 Recall | G3 Recall |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for c in classes:
        rec_overall = per_class_recall.get(c, 0.0)
        prec_overall = per_class_precision.get(c, 0.0)
        rec_g1 = per_group_metrics.get("G1", {}).get("per_class_recall", {}).get(c, np.nan)
        rec_g2 = per_group_metrics.get("G2", {}).get("per_class_recall", {}).get(c, np.nan)
        rec_g3 = per_group_metrics.get("G3", {}).get("per_class_recall", {}).get(c, np.nan)

        rec_g1_str = f"{rec_g1*100:.1f}%" if pd.notna(rec_g1) else "N/A"
        rec_g2_str = f"{rec_g2*100:.1f}%" if pd.notna(rec_g2) else "N/A"
        rec_g3_str = f"{rec_g3*100:.1f}%" if pd.notna(rec_g3) else "N/A"

        c_name = f"**`{c}`**" if c != "MAL_OTH" else "`MAL_OTH` *(non-headline)*"
        lines.append(f"| {c_name} | {rec_overall*100:.2f}% | {prec_overall*100:.2f}% | {rec_g1_str} | {rec_g2_str} | {rec_g3_str} |")
    lines.append("")

    # Low Count Warnings
    if low_count_warnings:
        lines.append("## 5. Sample Size Sparsity Warnings")
        lines.append("")
        lines.append(f"> [!WARNING]")
        lines.append(f"> The following **{len(low_count_warnings)} group x class cells** contained fewer than 10 lesions:")
        for w in low_count_warnings:
            lines.append(f"> - {w}")
        lines.append("")

    return "\n".join(lines)
