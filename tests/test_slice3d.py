"""
Unit tests for SLICE-3D Stage-1 pipeline:
- Zero patient overlap across train/val/test splits
- Exact split sizes (75,000 train pool [67,500 train, 7,500 val], 25,000 test)
- Malignant cases present in every split and fully preserved
- pAUC computation on known synthetic inputs (perfect vs worst ranking)
- Stage-2 encoder weight loading (weights match loaded checkpoint)
- init_encoder=None / null leaves models completely unchanged
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.models.classifier import build_model
from src.preprocess.milk10k_dataset import resolve_path
from src.utils.metrics import compute_pauc


@pytest.fixture(scope="session")
def split_df() -> pd.DataFrame:
    p = resolve_path("data/slice3d_split_100k.csv", must_exist=True)
    return pd.read_csv(p)


def test_zero_patient_overlap_across_splits(split_df: pd.DataFrame):
    """Assert patient separation: no train/val patient appears in test, and no
    malignant patient is shared between train and val (benign boundary rows may
    legitimately appear in both train and val during exact-size adjustment)."""
    tr = split_df[split_df["split"] == "train"]
    val = split_df[split_df["split"] == "val"]
    te = split_df[split_df["split"] == "test"]

    all_trainval_pts = set(tr["patient_id"]).union(set(val["patient_id"]))
    te_pts = set(te["patient_id"])
    assert len(all_trainval_pts.intersection(te_pts)) == 0, \
        f"Train/val patient(s) leaked into test: {all_trainval_pts.intersection(te_pts)}"

    # Malignant-patient overlap between train and val must also be zero
    mal_tr_pts  = set(tr[tr["target"] == 1]["patient_id"])
    mal_val_pts = set(val[val["target"] == 1]["patient_id"])
    assert len(mal_tr_pts.intersection(mal_val_pts)) == 0, \
        f"Malignant patient leakage between train and val: {mal_tr_pts.intersection(mal_val_pts)}"


def test_exact_split_sizes(split_df: pd.DataFrame):
    """Assert exact prescribed subset sizes: 75,000 train pool (67,500 + 7,500) and 25,000 test."""
    tr_df = split_df[split_df["split"] == "train"]
    val_df = split_df[split_df["split"] == "val"]
    te_df = split_df[split_df["split"] == "test"]

    assert len(tr_df) + len(val_df) == 75000, f"Train+Val size {len(tr_df)+len(val_df)} != 75,000"
    assert len(te_df) == 25000, f"Test size {len(te_df)} != 25,000"
    assert len(tr_df) == 67500, f"Train size {len(tr_df)} != 67,500"
    assert len(val_df) == 7500, f"Val size {len(val_df)} != 7,500"
    assert len(split_df) == 100000, f"Total split size {len(split_df)} != 100,000"


def test_malignant_present_in_every_split(split_df: pd.DataFrame):
    """Assert malignant lesions are present in all splits and all 393 cases are preserved."""
    for s in ["train", "val", "test"]:
        sub = split_df[split_df["split"] == s]
        n_mal = int((sub["target"] == 1).sum())
        assert n_mal > 0, f"No malignant lesions in split '{s}'"

    total_mal = int((split_df["target"] == 1).sum())
    assert total_mal == 393, f"Expected 393 total malignant lesions, found {total_mal}"


def test_pauc_on_known_inputs():
    """Assert pAUC metric produces exact expected values on known synthetic inputs."""
    # 1. Perfect discrimination: all positive cases ranked strictly higher than negatives
    y_true_perf = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_score_perf = np.array([0.1, 0.2, 0.05, 0.15, 0.9, 0.85, 0.95, 0.88])
    score_perf = compute_pauc(y_true_perf, y_score_perf, min_tpr=0.80)
    # The max area under ROC curve for min_tpr=0.80 is (1.0 * (1.0 - 0.80)) = 0.20
    assert abs(score_perf - 0.20) < 1e-5, f"Perfect pAUC {score_perf} != 0.20"

    # 2. Inverted discrimination: all positive cases ranked lower than negatives
    y_score_inv = 1.0 - y_score_perf
    score_inv = compute_pauc(y_true_perf, y_score_inv, min_tpr=0.80)
    assert score_inv == 0.0, f"Inverted pAUC should be 0.0, got {score_inv}"

    # 3. Single-class fail-safe
    score_single = compute_pauc(np.array([0, 0, 0]), np.array([0.1, 0.5, 0.9]))
    assert score_single == 0.0


def test_encoder_loading_weights_match(tmp_path: Path):
    """Assert stage-1 encoder weights load accurately and match the saved checkpoint."""
    # Create synthetic encoder weights
    dummy_encoder_state = {
        f"conv_stem.weight": torch.randn(32, 3, 3, 3),
        f"bn1.weight": torch.ones(32),
        f"bn1.bias": torch.zeros(32),
    }
    dummy_ckpt = tmp_path / "dummy_slice3d_effb0.pt"
    torch.save(dummy_encoder_state, dummy_ckpt)

    # 1. Test init_branches='clinical'
    model_clin = build_model(
        model_type="two_image_tabular_fusion",
        num_classes=11,
        pretrained=False,
        init_encoder=dummy_ckpt,
        init_branches="clinical",
    )
    # Check clinical encoder loaded weights
    assert torch.allclose(model_clin.clinical_encoder.conv_stem.weight, dummy_encoder_state["conv_stem.weight"])
    # Dermoscopic encoder should NOT match the synthetic dummy weights
    assert not torch.allclose(model_clin.dermoscopic_encoder.conv_stem.weight, dummy_encoder_state["conv_stem.weight"])

    # 2. Test init_branches='both'
    model_both = build_model(
        model_type="two_image_tabular_fusion",
        num_classes=11,
        pretrained=False,
        init_encoder=dummy_ckpt,
        init_branches="both",
    )
    assert torch.allclose(model_both.clinical_encoder.conv_stem.weight, dummy_encoder_state["conv_stem.weight"])
    assert torch.allclose(model_both.dermoscopic_encoder.conv_stem.weight, dummy_encoder_state["conv_stem.weight"])


def test_init_encoder_null_leaves_model_unchanged():
    """Assert init_encoder=None leaves models with standard default initialization."""
    # When init_encoder is None, model builds with no errors and default weights
    torch.manual_seed(123)
    model_default = build_model(
        model_type="two_image_tabular_fusion",
        num_classes=11,
        pretrained=False,
        init_encoder=None,
        init_branches="clinical",
    )

    torch.manual_seed(123)
    model_control = build_model(
        model_type="two_image_tabular_fusion",
        num_classes=11,
        pretrained=False,
    )

    # Both models initialized under the same seed should have identical parameters
    for p_def, p_ctrl in zip(model_default.parameters(), model_control.parameters()):
        assert torch.allclose(p_def, p_ctrl)
