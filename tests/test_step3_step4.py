"""
Unit tests for Step 3 (dataset and models) and Step 4 (fairness evaluation)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.fairness.milk10k_eval import (
    HEADLINE_CLASSES,
    MILK10K_CLASSES,
    compute_ece,
    compute_macro_auc,
    evaluate_milk10k_predictions,
)
from src.models.classifier import build_model
from src.preprocess.milk10k_dataset import resolve_path


@pytest.fixture(scope="session")
def folds_df() -> pd.DataFrame:
    p = resolve_path("data/milk10k_folds.csv", must_exist=True)
    return pd.read_csv(p)


@pytest.fixture(scope="session")
def meta_df() -> pd.DataFrame:
    p = resolve_path("data/milk10k/MILK10k_Training_Metadata.csv", must_exist=True)
    return pd.read_csv(p)


def test_no_lesion_in_both_train_and_heldout_for_any_fold(folds_df: pd.DataFrame):
    """Verify that for every fold k, the held-out set and train set of lesions are disjoint."""
    for k in range(5):
        heldout_lesions = set(folds_df[folds_df["fold"] == k]["lesion_id"])
        train_lesions = set(folds_df[folds_df["fold"] != k]["lesion_id"])

        assert len(heldout_lesions) == 1048
        assert len(train_lesions) == 4192
        intersection = heldout_lesions.intersection(train_lesions)
        assert len(intersection) == 0, f"Fold {k} has overlapping lesions between train and held-out: {intersection}"


def test_both_images_of_lesion_always_together(folds_df: pd.DataFrame, meta_df: pd.DataFrame):
    """Verify that both images (clinical & dermoscopic) of every lesion are strictly in the same fold."""
    merged = pd.merge(meta_df, folds_df[["lesion_id", "fold"]], on="lesion_id", how="inner")
    assert len(merged) == 10480

    for k in range(5):
        fold_images = merged[merged["fold"] == k]
        other_images = merged[merged["fold"] != k]

        fold_lids = set(fold_images["lesion_id"])
        other_lids = set(other_images["lesion_id"])

        assert len(fold_lids.intersection(other_lids)) == 0, f"Lesion images split across folds in fold {k}"


def test_auc_skips_absent_classes_without_crashing():
    """Requirement: AUC skips absent classes without crashing."""
    n = 40
    # Create target array with only 3 classes present (the other 8 classes have 0 samples)
    y_true = np.array(["BCC"] * 25 + ["NV"] * 10 + ["MEL"] * 5)
    probs = np.random.dirichlet(np.ones(len(MILK10K_CLASSES)), size=n)

    auc, skipped, per_class = compute_macro_auc(y_true, probs, MILK10K_CLASSES)

    assert isinstance(auc, float)
    assert 0.0 <= auc <= 1.0
    assert len(skipped) > 0, "Expected absent classes to be logged in skipped list"
    assert any("AKIEC" in s for s in skipped)
    assert any("MAL_OTH" in s for s in skipped)
    assert "BCC" in per_class
    assert "NV" in per_class
    assert "MEL" in per_class


def test_metric_functions_on_synthetic_predictions():
    """Requirement: Metric functions work on a tiny synthetic predictions file."""
    n = 60
    classes = MILK10K_CLASSES
    rng = np.random.RandomState(42)

    y_true = rng.choice(classes, size=n)
    y_pred = rng.choice(classes, size=n)
    tone_groups = rng.choice(["G1", "G2", "G3"], size=n)
    probs = rng.dirichlet(np.ones(len(classes)), size=n)

    data = {
        "lesion_id": [f"L_{i}" for i in range(n)],
        "fold": 0,
        "y_true": y_true,
        "y_pred": y_pred,
        "tone_group": tone_groups,
        "skin_tone_class": rng.choice([0, 1, 2, 3, 4, 5], size=n),
    }
    for idx, c in enumerate(classes):
        data[f"prob_{c}"] = probs[:, idx]

    synth_df = pd.DataFrame(data)

    results = evaluate_milk10k_predictions(
        predictions_df=synth_df,
        output_dir=None,
        n_bootstrap=50,
        seed=42,
    )

    ov = results["overall"]
    assert "accuracy" in ov
    assert "headline_macro_f1" in ov
    assert "balanced_accuracy" in ov
    assert "ece" in ov
    assert 0.0 <= ov["accuracy"] <= 1.0
    assert 0.0 <= ov["headline_macro_f1"] <= 1.0

    # Subgroup metrics
    assert "G1" in results["per_group"] or "G2" in results["per_group"] or "G3" in results["per_group"]
    assert "disparity" in results
    assert "worst_group" in results["disparity"]
    assert "f1_gap" in results["disparity"]


def test_models_instantiation_and_forward():
    """Verify all 4 MILK10k models instantiate and run forward pass without error."""
    b = 2
    c_img = torch.randn(b, 3, 128, 128)
    d_img = torch.randn(b, 3, 128, 128)
    tab = torch.randn(b, 13)

    for m_name in ["dermoscopic_only", "clinical_only", "two_image_fusion", "two_image_tabular_fusion"]:
        model = build_model(
            model_type=m_name,
            num_classes=11,
            backbone="efficientnet_b0",
            pretrained=False,
            tabular_dim=13,
        )
        if m_name == "dermoscopic_only":
            out = model(dermoscopic_img=d_img)
        elif m_name == "clinical_only":
            out = model(clinical_img=c_img)
        elif m_name == "two_image_fusion":
            out = model(clinical_img=c_img, dermoscopic_img=d_img)
        else:
            out = model(clinical_img=c_img, dermoscopic_img=d_img, tabular=tab)

        assert out.shape == (b, 11), f"Expected shape ({b}, 11) for {m_name}, got {out.shape}"


def test_tone_groups_naming_and_no_forbidden_words(tmp_path: Path):
    """Verify that reports and tables use G1 = {0,1,2}, G2 = {3}, G3 = {4,5} and no forbidden words."""
    from src.fairness.compare_runs import build_comparison_report

    n = 60
    classes = MILK10K_CLASSES
    rng = np.random.RandomState(42)

    data = {
        "lesion_id": [f"L_{i}" for i in range(n)],
        "fold": 0,
        "y_true": rng.choice(classes, size=n),
        "y_pred": rng.choice(classes, size=n),
        "tone_group": rng.choice(["G1", "G2", "G3"], size=n),
        "skin_tone_class": rng.choice([0, 1, 2, 3, 4, 5], size=n),
    }
    probs = rng.dirichlet(np.ones(len(classes)), size=n)
    for idx, c in enumerate(classes):
        data[f"prob_{c}"] = probs[:, idx]

    df = pd.DataFrame(data)

    run_dir1 = tmp_path / "d_cw_128"
    run_dir2 = tmp_path / "a_cw_128"
    run_dir1.mkdir()
    run_dir2.mkdir()
    df.to_csv(run_dir1 / "predictions.csv", index=False)
    df.to_csv(run_dir2 / "predictions.csv", index=False)

    out_md = tmp_path / "comparison.md"
    report_text = build_comparison_report(
        run_dirs=[run_dir1, run_dir2],
        output_file=out_md,
        n_bootstrap=20,
        seed=42,
    )

    # Check required tone group labels
    assert "G1 = {0,1,2}" in report_text
    assert "G2 = {3}" in report_text
    assert "G3 = {4,5}" in report_text

    # Check forbidden words are absent
    forbidden = ["fitzpatrick", "light", "medium", "dark"]
    lower_report = report_text.lower()
    for word in forbidden:
        assert word not in lower_report, f"Forbidden word '{word}' found in comparison report!"

    # Check ECE CIs and sample size n
    assert "Calibration ECE" in report_text
    assert "$n = " in report_text

