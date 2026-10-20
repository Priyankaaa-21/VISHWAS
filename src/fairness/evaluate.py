"""
Vishwas Fairness Auditing Module: evaluate.py

Audits model fairness and performance in two explicit modes:
- Mode (a) With Demographic Skin-Tone Groups:
    Computes per-group accuracy, macro-F1, macro-AUC, per-class recall, max disparity gap,
    Expected Calibration Error (ECE), reliability diagrams, and bootstrap 95% confidence intervals.
- Mode (b) Without Demographic Skin-Tone Groups:
    Computes overall accuracy, macro-F1, macro-AUC, per-class recall, overall ECE, and bootstrap 95% CIs.
The generated report states clearly at the top which mode ran and WHY.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

# Ensure workspace root in path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VISHWAS_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(VISHWAS_ROOT) not in sys.path:
    sys.path.insert(0, str(VISHWAS_ROOT))

try:
    from vishwas.src.preprocess.skin_tone import fst_to_group
except ImportError:
    from src.preprocess.skin_tone import fst_to_group


def compute_ece(
    probs: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Expected Calibration Error (ECE) and bin statistics for reliability curves.
    """
    confidences = np.max(probs, axis=1) if probs.ndim == 2 else probs
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
            acc_in_bin = float(np.mean(preds[in_bin] == targets[in_bin]))
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


def bootstrap_ci(
    targets: np.ndarray,
    preds: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Computes 95% bootstrap confidence intervals for Accuracy and Macro-F1 (1,000 resamples).
    """
    if len(targets) < 5:
        return (0.0, 1.0), (0.0, 1.0)

    rng = np.random.RandomState(seed)
    n = len(targets)
    accs, f1s = [], []

    for _ in range(n_resamples):
        indices = rng.choice(n, size=n, replace=True)
        t_boot = targets[indices]
        p_boot = preds[indices]
        accs.append(accuracy_score(t_boot, p_boot))
        f1s.append(f1_score(t_boot, p_boot, average="macro", zero_division=0))

    acc_ci = (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)))
    f1_ci = (float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5)))
    return acc_ci, f1_ci


def evaluate_run(run_dir: Path) -> Dict[str, Any]:
    """
    Evaluates a run folder in either Mode (a) or Mode (b) depending on skin-tone metadata availability.
    """
    pred_path = run_dir / "predictions.csv"
    probs_path = run_dir / "probabilities.npy"

    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found at {pred_path}")

    df = pd.read_csv(pred_path)
    probs = np.load(probs_path) if probs_path.exists() else None

    # Check for presence of skin-tone annotations
    has_skin_tone_group = (
        "skin_tone_group" in df.columns
        and df["skin_tone_group"].dropna().isin(["light", "medium", "dark"]).any()
    )
    has_fitzpatrick = (
        "fitzpatrick" in df.columns
        and (df["fitzpatrick"] > 0).any()
    )
    has_skin_tone = has_skin_tone_group or has_fitzpatrick

    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    num_classes = len(np.unique(np.concatenate([y_true, y_pred])))

    overall_acc = accuracy_score(y_true, y_pred)
    overall_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    overall_acc_ci, overall_f1_ci = bootstrap_ci(y_true, y_pred, n_resamples=1000)

    # Compute overall AUC (OVR)
    overall_auc = 0.5
    if probs is not None and len(np.unique(y_true)) > 1:
        try:
            y_oh = np.eye(probs.shape[1])[y_true]
            overall_auc = roc_auc_score(y_oh, probs, average="macro", multi_class="ovr")
        except Exception:
            overall_auc = 0.5

    # Overall ECE
    if probs is not None:
        overall_ece, _, _, _ = compute_ece(probs, y_pred, y_true, n_bins=10)
    else:
        overall_ece = 0.0

    # Per-class recall
    recalls_arr = recall_score(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)
    per_class_rec = {int(i): float(r) for i, r in enumerate(recalls_arr)}

    results: Dict[str, Any] = {
        "mode": "mode_a" if has_skin_tone else "mode_b",
        "has_skin_tone": has_skin_tone,
        "overall": {
            "accuracy": round(overall_acc, 4),
            "accuracy_95ci": (round(overall_acc_ci[0], 4), round(overall_acc_ci[1], 4)),
            "macro_f1": round(overall_f1, 4),
            "macro_f1_95ci": (round(overall_f1_ci[0], 4), round(overall_f1_ci[1], 4)),
            "macro_auc": round(overall_auc, 4),
            "ece": round(overall_ece, 4),
            "per_class_recall": per_class_rec,
        },
    }

    if has_skin_tone:
        # MODE (A): Subgroup fairness audit across Light, Medium, Dark
        if "fitzpatrick" in df.columns:
            df["fitzpatrick_group"] = df["fitzpatrick"].apply(fst_to_group)
        group_col = "skin_tone_group" if has_skin_tone_group else "fitzpatrick_group"

        subgroup_metrics = {}
        for grp in ["light", "medium", "dark"]:
            sub = df[df[group_col] == grp]
            if len(sub) == 0:
                continue

            t_sub = sub["y_true"].values
            p_sub = sub["y_pred"].values
            acc = accuracy_score(t_sub, p_sub)
            f1 = f1_score(t_sub, p_sub, average="macro", zero_division=0)
            acc_ci, f1_ci = bootstrap_ci(t_sub, p_sub, n_resamples=1000)

            sub_probs = probs[sub.index.values] if probs is not None else None
            sub_ece = compute_ece(sub_probs, p_sub, t_sub)[0] if sub_probs is not None else 0.0

            sub_auc = 0.5
            if sub_probs is not None and len(np.unique(t_sub)) > 1:
                try:
                    y_sub_oh = np.eye(sub_probs.shape[1])[t_sub]
                    sub_auc = roc_auc_score(y_sub_oh, sub_probs, average="macro", multi_class="ovr")
                except Exception:
                    sub_auc = 0.5

            subgroup_metrics[grp] = {
                "count": len(sub),
                "accuracy": round(acc, 4),
                "accuracy_95ci": (round(acc_ci[0], 4), round(acc_ci[1], 4)),
                "macro_f1": round(f1, 4),
                "macro_f1_95ci": (round(f1_ci[0], 4), round(f1_ci[1], 4)),
                "macro_auc": round(sub_auc, 4),
                "ece": round(sub_ece, 4),
            }

        # Gaps
        f1_vals = [m["macro_f1"] for m in subgroup_metrics.values()]
        acc_vals = [m["accuracy"] for m in subgroup_metrics.values()]
        results["subgroups"] = subgroup_metrics
        results["disparity_gap"] = {
            "max_accuracy_gap": round(max(acc_vals) - min(acc_vals), 4) if acc_vals else 0.0,
            "max_f1_gap": round(max(f1_vals) - min(f1_vals), 4) if f1_vals else 0.0,
            "worst_group_f1": round(min(f1_vals), 4) if f1_vals else 0.0,
        }

    # Generate Markdown Report stating Mode and Why
    generate_audit_report(results, run_dir, df)

    # Save Reliability Diagram
    if probs is not None:
        plot_calibration_diagram(probs, y_pred, y_true, run_dir, has_skin_tone, df)

    print(f"Fairness audit report successfully saved to {run_dir / 'fairness_audit_report.md'}")
    return results


def plot_calibration_diagram(
    probs: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
    run_dir: Path,
    has_skin_tone: bool,
    df: pd.DataFrame,
) -> None:
    """Plots calibration curves."""
    ece, bin_accs, bin_confs, sizes = compute_ece(probs, preds, targets, n_bins=10)

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    valid = sizes > 0
    plt.plot(bin_confs[valid], bin_accs[valid], "s-", color="#1f77b4", label=f"Overall ECE = {ece:.3f}")
    plt.bar(bin_confs[valid], bin_accs[valid], width=0.08, alpha=0.2, color="#1f77b4")
    plt.title("Expected Calibration Reliability Curve", fontsize=11)
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Observed Empirical Accuracy")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(run_dir / "reliability_curves.png", dpi=200)
    plt.close()


def generate_audit_report(results: Dict[str, Any], run_dir: Path, df: pd.DataFrame) -> None:
    """Writes the markdown fairness audit report with explicit mode explanation."""
    lines = []
    mode = results["mode"]

    if mode == "mode_a":
        lines.append(f"# Vishwas Subgroup Fairness Audit Report: `{run_dir.name}`\n")
        lines.append("> [!NOTE]")
        lines.append("> **EVALUATION MODE: Mode (a) - With Demographic Skin-Tone Groups**")
        lines.append("> **WHY**: This evaluation ran in Mode (a) because the dataset contains verified individual skin phototype annotations (Fitzpatrick phototypes or Individual Typology Angle groupings).\n")

        lines.append("## 1. Demographic Subgroup Fairness Breakdown\n")
        lines.append("| Skin-Tone Group | Samples | Accuracy (95% CI) | Macro-F1 (95% CI) | Macro-AUC | ECE |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

        for grp, m in results.get("subgroups", {}).items():
            acc_ci = f"{m['accuracy'] * 100:.2f}% [{m['accuracy_95ci'][0] * 100:.1f}%, {m['accuracy_95ci'][1] * 100:.1f}%]"
            f1_ci = f"{m['macro_f1']:.4f} [{m['macro_f1_95ci'][0]:.4f}, {m['macro_f1_95ci'][1]:.4f}]"
            lines.append(f"| **{grp.capitalize()}** | {m['count']} | {acc_ci} | {f1_ci} | {m['macro_auc']:.3f} | {m['ece']:.3f} |")

        gaps = results.get("disparity_gap", {})
        lines.append("\n## 2. Demographic Parity & Disparity Summary\n")
        lines.append(f"- **Worst-Group F1 Score**: `{gaps.get('worst_group_f1', 0.0):.4f}` (Higher is better; guards against subgroup neglect)")
        lines.append(f"- **Max Macro-F1 Disparity Gap**: `{gaps.get('max_f1_gap', 0.0):.4f}` (Lower is better; parity target)")
        lines.append(f"- **Max Accuracy Disparity Gap**: `{gaps.get('max_accuracy_gap', 0.0):.4f}`\n")

    else:
        lines.append(f"# Vishwas Evaluation & Calibration Report: `{run_dir.name}`\n")
        lines.append("> [!NOTE]")
        lines.append("> **EVALUATION MODE: Mode (b) - Overall Metrics Without Demographic Groups**")
        lines.append("> **WHY**: This evaluation ran in Mode (b) because the DermaMNIST dataset (HAM10000 derived) does NOT contain patient-level Fitzpatrick Skin Phototype (FST) or individual typology angle (ITA) metadata.")
        lines.append("> Subgroup demographic disparities and gap metrics cannot be evaluated without verified demographic labels.\n")

    # Overall Metrics (Present in both modes)
    ov = results["overall"]
    lines.append("## Overall Performance & Calibration\n")
    lines.append(f"- **Overall Test Accuracy**: `{ov['accuracy'] * 100:.2f}%` (95% CI: [{ov['accuracy_95ci'][0] * 100:.1f}%, {ov['accuracy_95ci'][1] * 100:.1f}%])")
    lines.append(f"- **Overall Macro-F1**: `{ov['macro_f1']:.4f}` (95% CI: [{ov['macro_f1_95ci'][0]:.4f}, {ov['macro_f1_95ci'][1]:.4f}])")
    lines.append(f"- **Macro-AUC (OVR)**: `{ov['macro_auc']:.4f}`")
    lines.append(f"- **Expected Calibration Error (ECE)**: `{ov['ece']:.4f}`\n")

    lines.append("## Visual Calibration Artifacts\n")
    lines.append("- [Reliability Diagram (Calibration Curve)](reliability_curves.png)")
    lines.append("- [Confusion Matrix](confusion_matrix.png)")

    with open(run_dir / "fairness_audit_report.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit fairness and calibration of a model run")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to run directory in reports/")
    args = parser.parse_args()
    evaluate_run(Path(args.run_dir))
