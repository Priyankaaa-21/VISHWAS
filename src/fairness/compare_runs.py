"""
Vishwas Fairness Cross-Run Comparison Module: compare_runs.py

Aggregates multiple experimental run directories to benchmark fairness interventions:
- Compares Baseline vs Class-Weighted vs Balanced Sampler vs Group-Weighted Loss vs Fusion
- Compares Standard Augmentation vs Colour-Preserving Augmentation
- Highlights Worst-Group F1 and Disparity Gap (Max Group F1 - Min Group F1)
- Outputs publication-ready comparison table to reports/model_fairness_comparison.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VISHWAS_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(VISHWAS_ROOT) not in sys.path:
    sys.path.insert(0, str(VISHWAS_ROOT))

from vishwas.src.preprocess.skin_tone import fst_to_group


def compare_all_runs(reports_dir: Path = Path("vishwas/reports")) -> pd.DataFrame:
    """Scans all run folders and aggregates performance & fairness disparity metrics."""
    run_dirs = [p for p in reports_dir.iterdir() if p.is_dir() and (p / "predictions.csv").exists()]

    if not run_dirs:
        print(f"No completed run directories found in {reports_dir}")
        return pd.DataFrame()

    results: List[Dict[str, Any]] = []

    for r_dir in sorted(run_dirs):
        df = pd.read_csv(r_dir / "predictions.csv")
        df["fitzpatrick_group"] = df["fitzpatrick"].apply(fst_to_group)
        if "ita_group" not in df.columns:
            df["ita_group"] = "medium"

        # Overall metrics
        acc = accuracy_score(df["y_true"], df["y_pred"])
        macro_f1 = f1_score(df["y_true"], df["y_pred"], average="macro", zero_division=0)

        # Subgroup metrics by ITA
        grp_f1s = {}
        for grp in ["light", "medium", "dark"]:
            sub = df[df["ita_group"] == grp]
            if len(sub) > 0:
                f1_grp = f1_score(sub["y_true"], sub["y_pred"], average="macro", zero_division=0)
                grp_f1s[grp] = f1_grp
            else:
                grp_f1s[grp] = np.nan

        valid_f1s = [v for v in grp_f1s.values() if not np.isnan(v)]
        worst_f1 = min(valid_f1s) if valid_f1s else 0.0
        best_f1 = max(valid_f1s) if valid_f1s else 0.0
        disparity_gap = best_f1 - worst_f1

        # Determine run configuration from name
        name_parts = r_dir.name.split("_")
        model_type = name_parts[0] if len(name_parts) > 0 else "image_only"
        mitigation = name_parts[1] if len(name_parts) > 1 else "none"

        results.append({
            "Run Name": r_dir.name,
            "Architecture": model_type,
            "Fairness Mitigation": mitigation,
            "Overall Acc": round(acc, 4),
            "Overall F1": round(macro_f1, 4),
            "Light F1": round(grp_f1s.get("light", 0.0), 4),
            "Medium F1": round(grp_f1s.get("medium", 0.0), 4),
            "Dark F1": round(grp_f1s.get("dark", 0.0), 4),
            "Worst-Group F1": round(worst_f1, 4),
            "Disparity Gap (Max-Min)": round(disparity_gap, 4),
        })

    summary_df = pd.DataFrame(results)

    # Format Markdown Table
    md_lines = [
        "# Vishwas Cross-Run Fairness & Performance Benchmark\n",
        "Comparison of model configurations across overall accuracy, subgroup F1 scores, worst-case performance, and disparity gaps.\n",
        "```",
        summary_df.to_string(index=False),
        "```\n",
        "## Key Auditing Observations:\n",
        "- **Worst-Group F1**: Higher is better (ensures vulnerable demographic subgroups are not left behind).",
        "- **Disparity Gap**: Lower is better (closer to demographic equality).",
        "- **Fairness-Utility Tradeoff**: Demonstrates whether mitigation reduces overall accuracy while equalizing group outcomes.",
    ]

    out_md = reports_dir / "model_fairness_comparison.md"
    with open(out_md, "w") as f:
        f.write("\n".join(md_lines))

    print("\n" + "=" * 80)
    print("           MODEL FAIRNESS & DISPARITY COMPARISON")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80 + "\n")
    print(f"Comparison report saved to {out_md}")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare fairness across model runs")
    parser.add_argument("--reports-dir", type=str, default="vishwas/reports")
    args = parser.parse_args()
    compare_all_runs(Path(args.reports_dir))
