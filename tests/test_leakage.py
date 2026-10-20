"""
Unit tests for data partition leakage in Vishwas.

Verifies that no md5hash or image filepath appears in more than one partition (train, val, test)
in both the primary Fitzpatrick17k-C split CSV and any downloaded metadata_clean.csv.
"""

from pathlib import Path
import pandas as pd
import pytest


SPLIT_CSV_PATH = Path(
    "Fitzpatrick17k/Fitzpatrick17k_Analysis/DatasetSplits/SimThresh_T_A2_T_0.99_0.70_FC_T_KeepOne_Out_T_OutThresh_None_0FST_F.csv"
)
if not SPLIT_CSV_PATH.exists() and (Path("..") / SPLIT_CSV_PATH).exists():
    SPLIT_CSV_PATH = Path("..") / SPLIT_CSV_PATH
CLEAN_METADATA_PATH = Path("data/metadata_clean.csv")
if not CLEAN_METADATA_PATH.exists() and (Path("vishwas") / CLEAN_METADATA_PATH).exists():
    CLEAN_METADATA_PATH = Path("vishwas") / CLEAN_METADATA_PATH


def test_split_csv_exists():
    """Verify that the official split CSV file exists."""
    assert SPLIT_CSV_PATH.exists(), f"Split CSV not found at {SPLIT_CSV_PATH}"


def test_no_md5hash_leakage_in_split_csv():
    """
    Ensure no md5hash appears in more than one partition (train, val, test)
    in the official Fitzpatrick17k-C split file.
    """
    df = pd.read_csv(SPLIT_CSV_PATH)

    assert "md5hash" in df.columns, "Missing 'md5hash' column in split CSV"
    assert "partition" in df.columns, "Missing 'partition' column in split CSV"

    # Filter out empty or null hashes if any
    valid_df = df.dropna(subset=["md5hash", "partition"]).copy()
    valid_df["md5hash"] = valid_df["md5hash"].astype(str).str.strip()

    # Check partitions are standard
    unique_partitions = set(valid_df["partition"].unique())
    expected_partitions = {"train", "val", "test"}
    assert unique_partitions.issubset(
        expected_partitions
    ), f"Unexpected partitions found: {unique_partitions - expected_partitions}"

    # Group by md5hash and count unique partitions
    partitions_per_hash = valid_df.groupby("md5hash")["partition"].nunique()
    leaked_hashes = partitions_per_hash[partitions_per_hash > 1]

    assert len(leaked_hashes) == 0, (
        f"Data leakage detected! {len(leaked_hashes)} md5hash(es) appear in multiple partitions: "
        f"{leaked_hashes.head(10).to_dict()}"
    )


def test_partition_hash_sets_mutually_disjoint():
    """Verify train, val, and test partition sets of hashes are strictly disjoint."""
    df = pd.read_csv(SPLIT_CSV_PATH)

    train_hashes = set(df[df["partition"] == "train"]["md5hash"].dropna().astype(str).str.strip())
    val_hashes = set(df[df["partition"] == "val"]["md5hash"].dropna().astype(str).str.strip())
    test_hashes = set(df[df["partition"] == "test"]["md5hash"].dropna().astype(str).str.strip())

    train_val_overlap = train_hashes.intersection(val_hashes)
    train_test_overlap = train_hashes.intersection(test_hashes)
    val_test_overlap = val_hashes.intersection(test_hashes)

    assert len(train_val_overlap) == 0, f"Overlap between train and val: {len(train_val_overlap)}"
    assert len(train_test_overlap) == 0, f"Overlap between train and test: {len(train_test_overlap)}"
    assert len(val_test_overlap) == 0, f"Overlap between val and test: {len(val_test_overlap)}"


def test_filepaths_mutually_disjoint_across_partitions():
    """Verify that relative filepaths are unique across partitions."""
    df = pd.read_csv(SPLIT_CSV_PATH)
    if "filepath" in df.columns:
        train_fps = set(df[df["partition"] == "train"]["filepath"].dropna())
        val_fps = set(df[df["partition"] == "val"]["filepath"].dropna())
        test_fps = set(df[df["partition"] == "test"]["filepath"].dropna())

        assert len(train_fps.intersection(val_fps)) == 0, "Filepath leakage between train and val"
        assert len(train_fps.intersection(test_fps)) == 0, "Filepath leakage between train and test"
        assert len(val_fps.intersection(test_fps)) == 0, "Filepath leakage between val and test"


def test_no_leakage_in_clean_metadata_if_present():
    """If metadata_clean.csv has been generated, verify zero leakage there as well."""
    if not CLEAN_METADATA_PATH.exists():
        pytest.skip("data/metadata_clean.csv does not exist yet.")

    df = pd.read_csv(CLEAN_METADATA_PATH)
    if df.empty or "md5hash" not in df.columns or "partition" not in df.columns:
        pytest.skip("data/metadata_clean.csv is empty or lacks required columns.")

    partitions_per_hash = df.groupby("md5hash")["partition"].nunique()
    leaked = partitions_per_hash[partitions_per_hash > 1]
    assert len(leaked) == 0, f"Leakage found in metadata_clean.csv: {len(leaked)} hashes appear in multiple partitions."
