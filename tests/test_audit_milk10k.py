"""
Unit tests for MILK10k dataset audit module: audit_milk10k.py
"""

from pathlib import Path
import pandas as pd
import pytest

from src.ingest.audit_milk10k import (
    DIAGNOSIS_CLASSES,
    resolve_path,
    scan_non_image_files,
    run_audit,
)


@pytest.fixture(scope="session")
def milk10k_dir() -> Path:
    return resolve_path("data/milk10k", must_exist=True)


@pytest.fixture(scope="session")
def audit_report_path() -> Path:
    return resolve_path("reports/milk10k_audit.md", must_exist=False)


def test_milk10k_files_exist(milk10k_dir: Path):
    """Verify that expected MILK10k CSV files and training directory exist."""
    assert (milk10k_dir / "MILK10k_Training_Metadata.csv").is_file()
    assert (milk10k_dir / "MILK10k_Training_GroundTruth.csv").is_file()
    assert (milk10k_dir / "train_images" / "MILK10k_Training_Input").is_dir()


def test_non_image_files_discovered(milk10k_dir: Path):
    """Verify extra non-image files are identified correctly."""
    train_dir = milk10k_dir / "train_images" / "MILK10k_Training_Input"
    non_images = scan_non_image_files(train_dir)
    assert "attribution.txt" in non_images
    assert "licenses/CC-BY-NC.txt" in non_images


def test_metadata_and_groundtruth_counts(milk10k_dir: Path):
    """Verify exact counts: 10,480 images, 5,240 lesions, 11 diagnosis columns."""
    meta = pd.read_csv(milk10k_dir / "MILK10k_Training_Metadata.csv")
    gt = pd.read_csv(milk10k_dir / "MILK10k_Training_GroundTruth.csv")

    assert len(meta) == 10480
    assert meta["lesion_id"].nunique() == 5240
    assert len(gt) == 5240
    assert gt["lesion_id"].nunique() == 5240

    # Ensure lesion_id sets match 100%
    assert set(meta["lesion_id"]) == set(gt["lesion_id"])

    # Ensure each lesion has exactly 2 rows: 1 clinical, 1 dermoscopic
    types_per_lesion = meta.groupby("lesion_id")["image_type"].agg(list)
    assert (types_per_lesion.apply(len) == 2).all()
    assert (types_per_lesion.apply(lambda l: sorted(l) == ["clinical: close-up", "dermoscopic"])).all()


def test_groundtruth_single_label(milk10k_dir: Path):
    """Verify ground truth has valid single-label one-hot vectors."""
    gt = pd.read_csv(milk10k_dir / "MILK10k_Training_GroundTruth.csv")
    label_sums = gt[DIAGNOSIS_CLASSES].sum(axis=1)
    assert (label_sums == 1.0).all(), "Ground truth contains non-single-label rows"


def test_metadata_consistency_across_pairs(milk10k_dir: Path):
    """Verify no metadata (age, sex, site, skin tone) contradicts within lesion pairs."""
    meta = pd.read_csv(milk10k_dir / "MILK10k_Training_Metadata.csv")
    for col in ["age_approx", "sex", "site", "skin_tone_class"]:
        grouped = meta.groupby("lesion_id")[col]
        def has_mismatch(s: pd.Series) -> bool:
            v = list(s)
            if len(v) != 2:
                return True
            if pd.isna(v[0]) and pd.isna(v[1]):
                return False
            return v[0] != v[1]
        mismatches = grouped.apply(has_mismatch)
        assert mismatches.sum() == 0, f"Found mismatches in column '{col}'"


def test_audit_report_generation(milk10k_dir: Path, audit_report_path: Path):
    """Verify that audit_milk10k generates a valid report with all required sections."""
    res = run_audit(milk10k_dir, audit_report_path, num_workers=4)
    assert res["total_images"] == 10480
    assert res["total_lesions"] == 5240
    assert res["corrupt_count"] == 0
    assert res["low_count_cells"] > 0
    assert audit_report_path.exists()

    report_text = audit_report_path.read_text(encoding="utf-8")
    required_keywords = [
        "Executive Summary",
        "Non-Image Files Discovered",
        "attribution.txt",
        "licenses/CC-BY-NC.txt",
        "Lesion Image Pairing",
        "Image Size Distribution",
        "600 x 450",
        "Demographic & Clinical Attribute Distributions",
        "skin_tone_class",
        "age_approx",
        "Ground Truth Diagnosis Distribution",
        "Crosstab: Diagnosis x Skin Tone Class",
        "Low-Count Cell Warnings",
    ]
    for kw in required_keywords:
        assert kw in report_text, f"Missing '{kw}' in audit report"
