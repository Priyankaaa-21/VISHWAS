"""
Vishwas Fairness Cross-Run Comparison Module: compare_runs.py

Builds a comprehensive comparison of models and fairness interventions:
- Compares:
    Model 1: two_image_tabular_fusion (d_cw_128)
    Model 2: dermoscopic_only (a_cw_128)
- Overall metrics with bootstrap 95% CIs (lesion resampling): Accuracy, Headline Macro-F1, Balanced Acc, ECE
- Tone groups labelled by skin_tone_class values: G1 = {0,1,2}, G2 = {3}, G3 = {4,5}
- Per tone group metrics with sample size n and 95% CIs for Macro-F1, Balanced Acc, Acc, ECE
- Worst-group value, Best-Worst Disparity Gap
- Per-class recall breakdown across all 11 diagnostic classes
- Outputs publication-ready markdown to reports/model_fairness_comparison.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VISHWAS_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(VISHWAS_ROOT) not in sys.path:
    sys.path.insert(0, str(VISHWAS_ROOT))

try:
    from vishwas.src.fairness.milk10k_eval import (
        HEADLINE_CLASSES,
        MILK10K_CLASSES,
        bootstrap_lesion_metrics,
        compute_ece,
        compute_macro_auc,
        evaluate_milk10k_predictions,
    )
except ImportError:
    from src.fairness.milk10k_eval import (
        HEADLINE_CLASSES,
        MILK10K_CLASSES,
        bootstrap_lesion_metrics,
        compute_ece,
        compute_macro_auc,
        evaluate_milk10k_predictions,
    )

TONE_GROUP_LABELS = {
    "G1": "G1 = {0,1,2}",
    "G2": "G2 = {3}",
    "G3": "G3 = {4,5}",
    "unknown": "unknown",
}


def load_and_evaluate_run(run_dir: Path, n_bootstrap: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """Loads run predictions.csv and computes full evaluation metrics with bootstrap CIs."""
    pred_path = run_dir / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file at {pred_path}")

    df = pd.read_csv(pred_path)
    eval_res = evaluate_milk10k_predictions(
        predictions_df=df,
        output_dir=run_dir,
        classes=MILK10K_CLASSES,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return eval_res


def build_comparison_report(
    run_dirs: List[Path],
    output_file: Optional[Path] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> str:
    """Builds a comprehensive side-by-side comparison report for the evaluated runs."""
    reports_data: List[Tuple[str, Dict[str, Any]]] = []

    for r_dir in run_dirs:
        print(f"[*] Evaluating run: {r_dir.name} ...", flush=True)
        res = load_and_evaluate_run(r_dir, n_bootstrap=n_bootstrap, seed=seed)
        reports_data.append((r_dir.name, res))

    lines: List[str] = []
    lines.append("# Vishwas Multi-Model Fairness & Calibration Comparison Benchmark")
    lines.append("")
    lines.append("> **Evaluation Protocol**: Bootstrap 95% Confidence Intervals (1,000 resamples of lesions).")
    lines.append("> **Tone Group Definitions**: Groups defined by `skin_tone_class` values (`G1 = {0,1,2}`, `G2 = {3}`, `G3 = {4,5}`).")
    lines.append("> **Headline Macro-F1**: Evaluated across 10 diagnostic classes excluding `MAL_OTH` (extreme dataset sparsity, $n=9$ total).")
    lines.append("")

    # Section 1: Executive Summary Comparison Table
    lines.append("## 1. Primary Model Comparison Table")
    lines.append("")
    lines.append("| Metric / Dimension | " + " | ".join(f"**`{name}`**" for name, _ in reports_data) + " |")
    lines.append("| :--- | " + " | ".join(":---:" for _ in reports_data) + " |")

    # Architectures
    model_types = []
    for name, _ in reports_data:
        if "two_image_tabular_fusion" in name or name.startswith("d_"):
            model_types.append("Two-Image + Tabular Fusion")
        elif "dermoscopic_only" in name or name.startswith("a_"):
            model_types.append("Dermoscopic Only")
        elif "clinical_only" in name or name.startswith("b_"):
            model_types.append("Clinical Only")
        elif "two_image_fusion" in name or name.startswith("c_"):
            model_types.append("Two-Image Fusion")
        else:
            model_types.append(name)
    lines.append("| **Architecture** | " + " | ".join(model_types) + " |")
    lines.append("| **Mitigation Strategy** | " + " | ".join("`class_weighted` (cap=10.0)" for _ in reports_data) + " |")

    # Overall Acc
    acc_strs = []
    for _, res in reports_data:
        ov = res["overall"]
        ci = ov["cis"]["accuracy"]
        acc_strs.append(f"**`{ov['accuracy']*100:.2f}%`**<br>[{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]")
    lines.append("| **Overall Accuracy (95% CI)** | " + " | ".join(acc_strs) + " |")

    # Headline Macro-F1
    f1_strs = []
    for _, res in reports_data:
        ov = res["overall"]
        ci = ov["cis"]["macro_f1"]
        f1_strs.append(f"**`{ov['headline_macro_f1']:.4f}`**<br>[{ci[0]:.4f}, {ci[1]:.4f}]")
    lines.append("| **Headline Macro-F1 (95% CI)** *(10 classes)* | " + " | ".join(f1_strs) + " |")

    # Full Macro-F1
    full_f1_strs = [f"`{res['overall']['full_macro_f1']:.4f}`" for _, res in reports_data]
    lines.append("| **Full Macro-F1** *(11 classes)* | " + " | ".join(full_f1_strs) + " |")

    # Balanced Accuracy
    bal_strs = []
    for _, res in reports_data:
        ov = res["overall"]
        ci = ov["cis"]["balanced_accuracy"]
        bal_strs.append(f"`{ov['balanced_accuracy']*100:.2f}%`<br>[{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]")
    lines.append("| **Balanced Accuracy (95% CI)** | " + " | ".join(bal_strs) + " |")

    # Calibration ECE
    ece_strs = []
    for _, res in reports_data:
        ov = res["overall"]
        ci = ov["cis"].get("ece", (0.0, 1.0))
        ece_strs.append(f"`{ov['ece']:.4f}`<br>[{ci[0]:.4f}, {ci[1]:.4f}]")
    lines.append("| **Calibration ECE (95% CI)** | " + " | ".join(ece_strs) + " |")

    # Macro-AUC
    auc_strs = [f"`{res['overall']['macro_auc']:.4f}`" for _, res in reports_data]
    lines.append("| **Macro-AUC (OvR)** | " + " | ".join(auc_strs) + " |")

    lines.append("| **--- Subgroup Fairness & Disparity ---** | " + " | ".join("---" for _ in reports_data) + " |")

    # Worst Group
    worst_strs = []
    for _, res in reports_data:
        disp = res["disparity"]
        wg = disp["worst_group"]
        wf1 = disp["worst_f1"]
        wn = res["per_group"].get(wg, {}).get("n_lesions", 0)
        wg_lbl = TONE_GROUP_LABELS.get(wg, wg)
        worst_strs.append(f"**`{wg_lbl}`**<br>($n={wn:,}$, F1: `{wf1:.4f}`)")
    lines.append("| **Worst-Performing Tone Group** | " + " | ".join(worst_strs) + " |")

    # Disparity Gap F1
    gap_f1_strs = [f"**`{res['disparity']['f1_gap']:.4f}`**" for _, res in reports_data]
    lines.append("| **Headline Macro-F1 Disparity Gap** | " + " | ".join(gap_f1_strs) + " |")

    # Balanced Acc Gap
    gap_bal_strs = [f"`{res['disparity']['balanced_acc_gap']*100:.2f}%`" for _, res in reports_data]
    lines.append("| **Balanced Accuracy Disparity Gap** | " + " | ".join(gap_bal_strs) + " |")
    lines.append("")

    # Section 2: Detailed Subgroup Performance by Tone Group
    lines.append("## 2. Subgroup Performance Across Tone Groups (`skin_tone_class`)")
    lines.append("")
    lines.append("Every group shows sample size $n$, Headline Macro-F1 (95% CI), Balanced Accuracy (95% CI), Accuracy (95% CI), and Calibration ECE (95% CI).")
    lines.append("")

    for g_key in ["G1", "G2", "G3"]:
        g_label = TONE_GROUP_LABELS.get(g_key, g_key)
        lines.append(f"### 2.{['G1', 'G2', 'G3'].index(g_key)+1} Tone Group: **`{g_label}`**")
        lines.append("")
        lines.append("| Metric | " + " | ".join(f"**`{name}`**" for name, _ in reports_data) + " |")
        lines.append("| :--- | " + " | ".join(":---:" for _ in reports_data) + " |")

        # Sample size n
        n_strs = []
        for _, res in reports_data:
            n_g = res["per_group"].get(g_key, {}).get("n_lesions", 0)
            n_strs.append(f"$n = {n_g:,}$")
        lines.append("| **Evaluated Lesions ($n$)** | " + " | ".join(n_strs) + " |")

        # Macro-F1
        f1_g_strs = []
        for _, res in reports_data:
            gm = res["per_group"].get(g_key, {})
            if gm:
                ci = gm["cis"]["macro_f1"]
                f1_g_strs.append(f"**`{gm['headline_macro_f1']:.4f}`** [{ci[0]:.4f}, {ci[1]:.4f}]")
            else:
                f1_g_strs.append("N/A")
        lines.append("| **Headline Macro-F1 (95% CI)** | " + " | ".join(f1_g_strs) + " |")

        # Balanced Accuracy
        bal_g_strs = []
        for _, res in reports_data:
            gm = res["per_group"].get(g_key, {})
            if gm:
                ci = gm["cis"]["balanced_accuracy"]
                bal_g_strs.append(f"`{gm['balanced_accuracy']*100:.2f}%` [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")
            else:
                bal_g_strs.append("N/A")
        lines.append("| **Balanced Accuracy (95% CI)** | " + " | ".join(bal_g_strs) + " |")

        # Accuracy
        acc_g_strs = []
        for _, res in reports_data:
            gm = res["per_group"].get(g_key, {})
            if gm:
                ci = gm["cis"]["accuracy"]
                acc_g_strs.append(f"`{gm['accuracy']*100:.2f}%` [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")
            else:
                acc_g_strs.append("N/A")
        lines.append("| **Accuracy (95% CI)** | " + " | ".join(acc_g_strs) + " |")

        # ECE
        ece_g_strs = []
        for _, res in reports_data:
            gm = res["per_group"].get(g_key, {})
            if gm:
                ci = gm["cis"].get("ece", (0.0, 1.0))
                ece_g_strs.append(f"`{gm['ece']:.4f}` [{ci[0]:.4f}, {ci[1]:.4f}]")
            else:
                ece_g_strs.append("N/A")
        lines.append("| **Calibration ECE (95% CI)** | " + " | ".join(ece_g_strs) + " |")
        lines.append("")

    # Section 3: Per-Class Diagnostic Recall Breakdown
    lines.append("## 3. Per-Class Diagnostic Recall Breakdown")
    lines.append("")
    lines.append("| Diagnosis Class | Description | " + " | ".join(f"**`{name}` Recall**" for name, _ in reports_data) + " |")
    lines.append("| :--- | :--- | " + " | ".join(":---:" for _ in reports_data) + " |")

    class_descriptions = {
        "AKIEC": "Actinic Keratosis / Intraepithelial Carcinoma",
        "BCC": "Basal Cell Carcinoma",
        "BEN_OTH": "Other Benign Lesions",
        "BKL": "Benign Keratotic Lesions",
        "DF": "Dermatofibroma",
        "INF": "Inflammatory / Infectious Conditions",
        "MAL_OTH": "Other Malignant Lesions *(non-headline, n=9)*",
        "MEL": "Melanoma",
        "NV": "Melanocytic Nevus",
        "SCCKA": "Squamous Cell Carcinoma / Keratoacanthoma",
        "VASC": "Vascular Lesions",
    }

    for c in MILK10K_CLASSES:
        c_label = f"**`{c}`**" if c != "MAL_OTH" else "`MAL_OTH` *(non-headline)*"
        desc = class_descriptions.get(c, "")
        row_recalls = []
        for _, res in reports_data:
            rec = res["overall"]["per_class_recall"].get(c, 0.0)
            row_recalls.append(f"`{rec*100:.2f}%`")
        lines.append(f"| {c_label} | {desc} | " + " | ".join(row_recalls) + " |")
    lines.append("")

    # Section 4: Audit & Calibration Figures
    lines.append("## 4. Audit & Calibration Artifacts")
    lines.append("")
    for name, _ in reports_data:
        lines.append(f"### Run: `{name}`")
        lines.append(f"- [Fairness Audit Report](file://{WORKSPACE_ROOT}/vishwas/reports/{name}/fairness_audit_report.md)")
        lines.append(f"- [Confusion Matrix Image](file://{WORKSPACE_ROOT}/vishwas/reports/{name}/confusion_matrix.png)")
        lines.append(f"- [Reliability Diagram (Calibration Curves)](file://{WORKSPACE_ROOT}/vishwas/reports/{name}/reliability_curves.png)")
        lines.append(f"- [Predictions CSV](file://{WORKSPACE_ROOT}/vishwas/reports/{name}/predictions.csv)")
        lines.append("")

    report_str = "\n".join(lines)

    out_path = output_file or (VISHWAS_ROOT / "reports" / "model_fairness_comparison.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_str)

    print(f"\n[+] Successfully generated model fairness comparison report at: {out_path}", flush=True)
    return report_str


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Vishwas model runs across fairness and performance dimensions")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Reports root directory")
    parser.add_argument("--runs", nargs="+", default=["d_cw_128", "a_cw_128"], help="Run folder names to compare")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap resamples on lesions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap reproducibility")
    parser.add_argument("--output", type=str, default=None, help="Output markdown path")
    args = parser.parse_args()

    rep_base = Path(args.reports_dir)
    if not rep_base.is_absolute():
        rep_base = VISHWAS_ROOT / rep_base

    target_run_dirs: List[Path] = []
    for r in args.runs:
        r_path = rep_base / r
        if r_path.exists() and (r_path / "predictions.csv").exists():
            target_run_dirs.append(r_path)
        else:
            print(f"[!] Warning: Run directory {r_path} not found or missing predictions.csv")

    if not target_run_dirs:
        # Fallback to scanning all dirs in reports with predictions.csv
        target_run_dirs = [p for p in sorted(rep_base.iterdir()) if p.is_dir() and (p / "predictions.csv").exists()]

    if not target_run_dirs:
        print(f"[!] No completed run directories found in {rep_base}")
        sys.exit(1)

    out_path = Path(args.output) if args.output else rep_base / "model_fairness_comparison.md"
    build_comparison_report(run_dirs=target_run_dirs, output_file=out_path, n_bootstrap=args.n_bootstrap, seed=args.seed)


if __name__ == "__main__":
    main()
