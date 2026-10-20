"""
Vishwas Dataset Verification Guard: check_images.py

Audits local image completeness against the official Fitzpatrick17k-C split CSV (11,394 images).
Ensures training refuses to start if fewer than 95% of expected images exist locally.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd


TOTAL_EXPECTED_FITZPATRICK = 11394
REQUIRED_COMPLETENESS_RATIO = 0.95


def check_fitzpatrick_images(
    split_csv_path: str = "Fitzpatrick17k/Fitzpatrick17k_Analysis/DatasetSplits/SimThresh_T_A2_T_0.99_0.70_FC_T_KeepOne_Out_T_OutThresh_None_0FST_F.csv",
    images_dir: str = "data/fitzpatrick_images",
    min_ratio: float = REQUIRED_COMPLETENESS_RATIO,
    enforce: bool = False,
) -> Tuple[int, int, float, bool]:
    """
    Checks how many expected images from the split CSV exist in images_dir.

    Returns:
        (num_found, num_expected, completeness_pct, passes_threshold)
    """
    csv_file = Path(split_csv_path)
    img_dir = Path(images_dir)

    if not csv_file.exists():
        raise FileNotFoundError(f"Split CSV not found at {csv_file}")

    df = pd.read_csv(csv_file)
    expected_count = len(df)
    found_count = 0

    print("\n" + "=" * 65)
    print("        FITZPATRICK17K IMAGE INTEGRITY & AVAILABILITY AUDIT")
    print("=" * 65)
    print(f"Target split CSV: {csv_file.name} (Expected rows: {expected_count})")
    print(f"Image directory : {img_dir.resolve()}")

    if img_dir.exists():
        for filepath in df["filepath"]:
            target_file = img_dir / str(filepath).strip()
            if target_file.is_file() and target_file.stat().st_size > 0:
                found_count += 1

    ratio = (found_count / expected_count) if expected_count > 0 else 0.0
    pct = ratio * 100.0
    passes = ratio >= min_ratio

    print(f"Found images    : {found_count} / {expected_count} ({pct:.2f}%)")
    print(f"Threshold req.  : {min_ratio * 100:.1f}%")
    print("-" * 65)

    if passes:
        print("STATUS: PASSED. Sufficient images exist to proceed with Fitzpatrick training.")
    else:
        print(f"STATUS: REFUSED. Only {found_count} images found ({pct:.2f}% < {min_ratio * 100:.1f}% threshold).")
        print("Training cannot start until the official image archive is extracted into:")
        print(f"  --> {img_dir.resolve()}")
    print("=" * 65 + "\n")

    if enforce and not passes:
        sys.exit(f"ABORTED: Fitzpatrick image availability ({pct:.2f}%) below {min_ratio * 100:.1f}% threshold.")

    return found_count, expected_count, pct, passes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit Fitzpatrick17k local image availability")
    parser.add_argument("--split-csv", type=str, default="Fitzpatrick17k/Fitzpatrick17k_Analysis/DatasetSplits/SimThresh_T_A2_T_0.99_0.70_FC_T_KeepOne_Out_T_OutThresh_None_0FST_F.csv")
    parser.add_argument("--image-dir", type=str, default="data/fitzpatrick_images")
    parser.add_argument("--enforce", action="store_true", help="Exit with code 1 if threshold is not met")
    args = parser.parse_args()

    check_fitzpatrick_images(args.split_csv, args.image_dir, enforce=args.enforce)
