"""
Vishwas Skin Tone Estimation Module: skin_tone.py

Extracts Individual Typology Angle (ITA) in CIELAB color space from healthy skin pixels
(excluding the central lesion and extreme specular/shadow pixels).
Computes agreement between annotators (fitzpatrick, fitzpatrick_centaur) and objective ITA groups,
generating data/ita.csv and reports/skin_tone_label_agreement.md.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import yaml


def rgb_to_cielab(rgb_norm: np.ndarray) -> np.ndarray:
    """
    Converts normalized sRGB [0, 1] array to CIELAB space using D65 reference white.
    Input: (H, W, 3) in [0, 1]. Output: (H, W, 3) where L in [0, 100], a in [-128, 127], b in [-128, 127].
    """
    # Linearize sRGB (gamma correction)
    mask = rgb_norm > 0.04045
    linear = np.zeros_like(rgb_norm)
    linear[mask] = ((rgb_norm[mask] + 0.055) / 1.055) ** 2.4
    linear[~mask] = rgb_norm[~mask] / 12.92

    # sRGB to XYZ (D65 matrix)
    transform = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = np.dot(linear, transform.T)

    # Reference white D65
    xn, yn, zn = 0.950489, 1.000000, 1.088840
    xyz[..., 0] /= xn
    xyz[..., 1] /= yn
    xyz[..., 2] /= zn

    # Non-linear transformation for Lab
    delta = 6.0 / 29.0
    mask_xyz = xyz > delta**3
    f_xyz = np.zeros_like(xyz)
    f_xyz[mask_xyz] = np.cbrt(xyz[mask_xyz])
    f_xyz[~mask_xyz] = (xyz[~mask_xyz] / (3 * delta**2)) + (4.0 / 29.0)

    l_channel = 116.0 * f_xyz[..., 1] - 16.0
    a_channel = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])
    b_channel = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])

    return np.stack([l_channel, a_channel, b_channel], axis=-1)


def compute_ita(l_val: float, b_val: float) -> float:
    """
    Computes Individual Typology Angle:
    ITA = arctan((L - 50) / b) * (180 / pi)
    """
    if abs(b_val) < 1e-6:
        b_val = 1e-6 if b_val >= 0 else -1e-6
    radians = math.atan2(l_val - 50.0, b_val)
    return float(math.degrees(radians))


def ita_to_group(ita_val: float) -> str:
    """
    Maps ITA degree value to 3-tier skin tone group:
    - Light: > 41 deg (corresponds to Fitzpatrick I & II)
    - Medium: 28 to 41 deg (corresponds to Fitzpatrick III & IV)
    - Dark: < 28 deg (corresponds to Fitzpatrick V & VI)
    """
    if ita_val > 41.0:
        return "light"
    elif ita_val >= 28.0:
        return "medium"
    else:
        return "dark"


def fst_to_group(fst_val: Optional[float]) -> str:
    """Maps Fitzpatrick scale 1-6 to Light/Medium/Dark (or unknown for <=0)."""
    if pd.isna(fst_val):
        return "unknown"
    try:
        val = int(fst_val)
        if val in (1, 2):
            return "light"
        elif val in (3, 4):
            return "medium"
        elif val in (5, 6):
            return "dark"
        else:
            return "unknown"
    except (ValueError, TypeError):
        return "unknown"


def extract_skin_ita_from_image(image_path: Path) -> Tuple[float, str]:
    """
    Extracts ITA from peripheral healthy skin, excluding the central lesion region
    and extreme specular/shadow pixels.
    """
    with Image.open(image_path) as pil_img:
        img_rgb = pil_img.convert("RGB")
        arr = np.array(img_rgb, dtype=np.float32) / 255.0

    h, w, _ = arr.shape
    lab = rgb_to_cielab(arr)
    l_chan = lab[..., 0]
    b_chan = lab[..., 2]

    # Create central lesion exclusion mask (inner 40% radius)
    y_idx, x_idx = np.ogrid[:h, :w]
    center_y, center_x = h / 2.0, w / 2.0
    radius_y, radius_x = 0.35 * h, 0.35 * w
    lesion_mask = (((x_idx - center_x) / radius_x) ** 2 + ((y_idx - center_y) / radius_y) ** 2) <= 1.0

    # Filter out extreme pixels (specular highlights L > 95, deep shadows L < 15)
    valid_skin_mask = (~lesion_mask) & (l_chan >= 15.0) & (l_chan <= 95.0)

    # Fallback if lesion mask removes too much
    if np.sum(valid_skin_mask) < 100:
        valid_skin_mask = (l_chan >= 15.0) & (l_chan <= 95.0)

    if np.sum(valid_skin_mask) == 0:
        mean_l = float(np.mean(l_chan))
        mean_b = float(np.mean(b_chan))
    else:
        mean_l = float(np.mean(l_chan[valid_skin_mask]))
        mean_b = float(np.mean(b_chan[valid_skin_mask]))

    ita = compute_ita(mean_l, mean_b)
    group = ita_to_group(ita)
    return round(ita, 2), group


def generate_agreement_report(
    merged_df: pd.DataFrame,
    report_path: Path,
) -> None:
    """Generates markdown confusion tables and agreement statistics."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter to known groups for fair agreement comparisons
    valid_df = merged_df[
        (merged_df["fitzpatrick_group"].isin(["light", "medium", "dark"]))
        & (merged_df["ita_group"].isin(["light", "medium", "dark"]))
    ].copy()

    lines = []
    lines.append("# Skin-Tone Label Agreement & Auditing Report\n")
    lines.append("Analysis of inter-annotator agreement and objective CIELAB Individual Typology Angle (ITA) skin-tone grouping.\n")

    lines.append("## 1. Group Mapping Definitions")
    lines.append("- **Light**: Fitzpatrick I & II | ITA > 41°")
    lines.append("- **Medium**: Fitzpatrick III & IV | 28° ≤ ITA ≤ 41°")
    lines.append("- **Dark**: Fitzpatrick V & VI | ITA < 28°\n")

    lines.append(f"**Total evaluated samples with valid annotations**: {len(valid_df)} / {len(merged_df)}\n")

    # Confusion matrix: Fitzpatrick vs ITA Group
    cm_fitz_ita = pd.crosstab(
        valid_df["fitzpatrick_group"],
        valid_df["ita_group"],
        rownames=["Fitzpatrick Label"],
        colnames=["Objective ITA Group"],
    )
    accuracy_fitz_ita = np.diag(cm_fitz_ita).sum() / max(1, cm_fitz_ita.values.sum())

    lines.append("## 2. Agreement: Fitzpatrick Label vs. Objective ITA Group")
    lines.append(f"**Overall Group Agreement**: `{accuracy_fitz_ita * 100:.1f}%`\n")
    lines.append("```")
    lines.append(cm_fitz_ita.to_string())
    lines.append("```\n")

    # Confusion matrix: Fitzpatrick (Annotator 1) vs Centaur (Annotator 2)
    has_centaur = "centaur_group" in valid_df.columns and valid_df["centaur_group"].isin(["light", "medium", "dark"]).any()
    if has_centaur:
        centaur_valid = valid_df[valid_df["centaur_group"].isin(["light", "medium", "dark"])]
        cm_fitz_centaur = pd.crosstab(
            centaur_valid["fitzpatrick_group"],
            centaur_valid["centaur_group"],
            rownames=["Fitzpatrick (Ann 1)"],
            colnames=["Centaur (Ann 2)"],
        )
        accuracy_ann = np.diag(cm_fitz_centaur).sum() / max(1, cm_fitz_centaur.values.sum())

        lines.append("## 3. Inter-Annotator Agreement: Fitzpatrick (Ann 1) vs Centaur (Ann 2)")
        lines.append(f"**Inter-Annotator Group Agreement**: `{accuracy_ann * 100:.1f}%`\n")
        lines.append("```")
        lines.append(cm_fitz_centaur.to_string())
        lines.append("```\n")

        # Centaur vs ITA Group
        cm_centaur_ita = pd.crosstab(
            centaur_valid["centaur_group"],
            centaur_valid["ita_group"],
            rownames=["Centaur (Ann 2)"],
            colnames=["Objective ITA Group"],
        )
        acc_centaur_ita = np.diag(cm_centaur_ita).sum() / max(1, cm_centaur_ita.values.sum())
        lines.append("## 4. Agreement: Centaur (Ann 2) vs Objective ITA Group")
        lines.append(f"**Agreement**: `{acc_centaur_ita * 100:.1f}%`\n")
        lines.append("```")
        lines.append(cm_centaur_ita.to_string())
        lines.append("```\n")

    lines.append("## 5. Clinical & Fairness Insights")
    lines.append("- Visual inspection of subjective Fitzpatrick scale annotations often suffers from significant inter-annotator variance.")
    lines.append("- Objective CIELAB ITA offers a standardized, reproducible skin-tone metric that avoids subjective annotator drift.")
    lines.append("- In downstream fairness auditing (`evaluate.py`), metrics are reported across **both** Fitzpatrick annotations and objective ITA groupings.\n")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Agreement report successfully saved to {report_path}")


def run_skin_tone_extraction(
    config_path: str = "vishwas/configs/config.yaml",
) -> None:
    """Main execution function for Phase 2 ITA extraction."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    clean_csv_path = Path(cfg["data"].get("clean_metadata_csv", "data/metadata_clean.csv"))
    orig_meta_path = Path(cfg["data"].get("orig_metadata_csv", "data/fitzpatrick17k.csv"))
    ita_csv_path = Path(cfg["data"].get("ita_csv", "data/ita.csv"))
    report_path = Path(cfg["reports"].get("output_dir", "vishwas/reports")) / "skin_tone_label_agreement.md"

    if not clean_csv_path.exists():
        raise FileNotFoundError(f"Clean metadata CSV not found at {clean_csv_path}")

    clean_df = pd.read_csv(clean_csv_path)
    if clean_df.empty:
        raise ValueError(f"Clean metadata CSV is empty at {clean_csv_path}")

    print(f"Processing ITA extraction for {len(clean_df)} images...")
    ita_records = []

    for _, row in clean_df.iterrows():
        img_path = Path(str(row["image_path"]))
        md5hash = str(row["md5hash"]).strip()

        if img_path.exists():
            ita_val, ita_grp = extract_skin_ita_from_image(img_path)
        else:
            ita_val, ita_grp = 0.0, "unknown"

        fitz_val = row.get("fitzpatrick", row.get("fst"))
        fitz_grp = fst_to_group(fitz_val)

        ita_records.append({
            "image_path": str(img_path),
            "md5hash": md5hash,
            "ita_value": ita_val,
            "ita_group": ita_grp,
            "fitzpatrick_group": fitz_grp,
        })

    ita_df = pd.DataFrame(ita_records)

    # Merge with original metadata to extract fitzpatrick_centaur
    if orig_meta_path.exists():
        orig_df = pd.read_csv(orig_meta_path)
        if "fitzpatrick_centaur" in orig_df.columns:
            orig_subset = orig_df[["md5hash", "fitzpatrick_centaur"]].drop_duplicates(subset=["md5hash"])
            ita_df = ita_df.merge(orig_subset, on="md5hash", how="left")
            ita_df["centaur_group"] = ita_df["fitzpatrick_centaur"].apply(fst_to_group)
        else:
            ita_df["centaur_group"] = "unknown"
    else:
        ita_df["centaur_group"] = "unknown"

    ita_csv_path.parent.mkdir(parents=True, exist_ok=True)
    ita_df.to_csv(ita_csv_path, index=False)
    print(f"Saved ITA values to {ita_csv_path}")

    # Generate agreement report
    generate_agreement_report(ita_df, report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ITA and analyze skin-tone agreement")
    parser.add_argument("--config", type=str, default="vishwas/configs/config.yaml")
    args = parser.parse_args()
    run_skin_tone_extraction(args.config)
