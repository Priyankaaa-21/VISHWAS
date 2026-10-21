"""
Unit tests for MILK10k 5-fold cross-validation partition: make_folds.py
"""

from pathlib import Path
import pandas as pd
import pytest

from src.preprocess.make_folds import (
    DIAGNOSIS_CLASSES,
    DEFAULT_TONE_GROUPS,
    resolve_path,
    map_skin_tone_to_group,
)


@pytest.fixture(scope="session")
def folds_csv_path() -> Path:
    return resolve_path("data/milk10k_folds.csv", must_exist=True)


@pytest.fixture(scope="session")
def milk10k_dir() -> Path:
    return resolve_path("data/milk10k", must_exist=True)


@pytest.fixture(scope="session")
def folds_report_path() -> Path:
    return resolve_path("reports/folds_report.md", must_exist=True)


def test_folds_csv_exists_and_valid_columns(folds_csv_path: Path):
    """Verify milk10k_folds.csv exists with exact required columns and row count."""
    df = pd.read_csv(folds_csv_path)
    expected_cols = ["lesion_id", "fold", "diagnosis", "skin_tone_class", "tone_group"]
    assert list(df.columns) == expected_cols
    assert len(df) == 5240
    assert set(df["fold"].unique()) == {0, 1, 2, 3, 4}


def test_no_lesion_in_multiple_folds(folds_csv_path: Path):
    """Requirement: No lesion_id appears in more than one fold."""
    df = pd.read_csv(folds_csv_path)
    assert df["lesion_id"].is_unique, "lesion_id must be unique per row in folds CSV"
    folds_per_lesion = df.groupby("lesion_id")["fold"].nunique()
    assert (folds_per_lesion == 1).all()


def test_both_images_of_lesion_in_same_fold(folds_csv_path: Path, milk10k_dir: Path):
    """Requirement: Both images of a lesion are always in the same fold."""
    folds_df = pd.read_csv(folds_csv_path)
    meta_df = pd.read_csv(milk10k_dir / "MILK10k_Training_Metadata.csv")

    assert len(meta_df) == 10480
    merged = pd.merge(meta_df, folds_df[["lesion_id", "fold"]], on="lesion_id", how="inner")
    assert len(merged) == 10480

    # Ensure every lesion's images have identical fold assignment
    folds_per_lesion_meta = merged.groupby("lesion_id")["fold"].nunique()
    assert (folds_per_lesion_meta == 1).all()

    # Verify each lesion has exactly 2 images in the same fold
    images_per_lesion = merged.groupby(["lesion_id", "fold"]).size()
    assert (images_per_lesion == 2).all()


def test_fold_sizes_within_5_percent(folds_csv_path: Path):
    """Requirement: Fold sizes are within 5% of each other."""
    df = pd.read_csv(folds_csv_path)
    fold_counts = df["fold"].value_counts()
    min_size = fold_counts.min()
    max_size = fold_counts.max()

    variation_pct = (max_size - min_size) / min_size
    assert variation_pct <= 0.05, f"Fold size variation {variation_pct * 100:.2f}% exceeds 5% limit: {fold_counts.to_dict()}"


def test_tone_groups_mapping():
    """Requirement: {0,1,2}->G1, {3}->G2, {4,5}->G3, missing->unknown, no light/medium/dark."""
    mapping = DEFAULT_TONE_GROUPS

    assert map_skin_tone_to_group(0, mapping) == "G1"
    assert map_skin_tone_to_group(1, mapping) == "G1"
    assert map_skin_tone_to_group(2, mapping) == "G1"
    assert map_skin_tone_to_group(3, mapping) == "G2"
    assert map_skin_tone_to_group(4, mapping) == "G3"
    assert map_skin_tone_to_group(5, mapping) == "G3"
    assert map_skin_tone_to_group(None, mapping) == "unknown"
    assert map_skin_tone_to_group(float("nan"), mapping) == "unknown"

    # Verify no subjective names
    for group_name in mapping.values():
        assert group_name.lower() not in ["light", "medium", "dark"]


def test_all_diagnoses_present_across_folds(folds_csv_path: Path):
    """Verify all 11 classes exist in each fold (at least 1 per fold)."""
    df = pd.read_csv(folds_csv_path)
    ct = pd.crosstab(df["diagnosis"], df["fold"])
    for diag in DIAGNOSIS_CLASSES:
        assert diag in ct.index
        for f in range(5):
            assert ct.loc[diag, f] >= 1, f"Diagnosis {diag} missing in fold {f}"


def test_folds_report_contains_required_sections(folds_report_path: Path):
    """Verify reports/folds_report.md documents all required information."""
    assert folds_report_path.is_file()
    text = folds_report_path.read_text(encoding="utf-8")

    required_phrases = [
        "Lesion Counts Per Fold",
        "Tone Group Counts Per Fold",
        "Diagnosis Class Counts Per Fold",
        "Subgroup Extremes & Sparsity Audit Per Fold",
        "Patient-Level Leakage Caveat",
        "patient_id",
        "FLAG: `MAL_OTH` (1)",
        "Cross-Validation & Evaluation Protocol for Steps 3 and 4",
    ]
    for phrase in required_phrases:
        assert phrase in text, f"Missing phrase '{phrase}' in folds report"
