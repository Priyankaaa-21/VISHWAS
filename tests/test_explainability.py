"""
Unit tests for VISHWAS Phase C: Explainability, ABCD Features, Grad-CAM, and Sanity Checks
tests/test_explainability.py
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from src.explain.features import (
    compute_asymmetry,
    compute_border_irregularity,
    compute_feature_agreement,
    extract_dominant_colors,
    extract_lesion_features,
    segment_lesion,
)
from src.explain.gradcam import GradCAM, overlay_cam_on_image
from src.explain.sanity_checks import (
    load_explain_sanity_results,
    run_model_randomization_test,
)
from src.models.milk10k_classifier import ClinicalOnlyClassifier
from src.preprocess.milk10k_dataset import MILK10K_CLASSES


def _create_test_lesion_image() -> Image.Image:
    """Creates a synthetic 128x128 test skin lesion with brown core on tan background."""
    img_arr = np.full((128, 128, 3), fill_value=[215, 180, 150], dtype=np.uint8)
    # Add dark brown elliptical core
    y, x = np.ogrid[:128, :128]
    mask = (((y - 64) ** 2) / (25 ** 2) + ((x - 64) ** 2) / (35 ** 2)) <= 1.0
    img_arr[mask] = [70, 40, 25]
    # Add some darker focal dots
    focal = (((y - 55) ** 2) / (6 ** 2) + ((x - 70) ** 2) / (6 ** 2)) <= 1.0
    img_arr[focal] = [20, 15, 10]
    return Image.fromarray(img_arr)


def test_lesion_segmentation_and_abcd_features():
    """Verify lesion segmentation produces valid binary mask and ABCD features."""
    img = _create_test_lesion_image()
    mask, meta = segment_lesion(img)

    assert isinstance(mask, np.ndarray)
    assert mask.shape == (128, 128)
    assert mask.dtype == bool
    assert np.sum(mask) > 100, "Lesion mask should contain segmented lesion pixels"

    # Extract ABCD features
    feats = extract_lesion_features(img, mask=mask)
    assert "asymmetry_score" in feats
    assert 0.0 <= feats["asymmetry_score"] <= 1.0
    assert "border_irregularity_score" in feats
    assert feats["border_irregularity_score"] >= 0.8
    assert "num_dominant_colors" in feats
    assert feats["num_dominant_colors"] >= 1
    assert isinstance(feats["dominant_colors"], list)
    assert feats["equivalent_diameter_px"] > 10.0
    assert "methodology" in feats
    assert len(feats["known_weaknesses"]) >= 3


def test_feature_agreement_computation():
    """Verify feature agreement score correctly maps criteria against disease feature profiles."""
    img = _create_test_lesion_image()
    feats = extract_lesion_features(img)

    for c in ["MEL", "BCC", "NV"]:
        fa = compute_feature_agreement(feats, candidate_class=c)
        assert 0.0 <= fa["agreement_percentage"] <= 100.0
        assert len(fa["checklist"]) >= 3
        assert fa["matched_count"] <= fa["total_count"]
        assert all("feature" in item and "seen" in item and "weight" in item for item in fa["checklist"])


def test_disease_features_json_schema_and_completeness():
    """Verify data/disease_features.json contains all 11 classes, valid sources, and equity notes."""
    feat_path = Path(__file__).resolve().parent.parent / "data" / "disease_features.json"
    assert feat_path.exists(), "data/disease_features.json does not exist"

    data = json.loads(feat_path.read_text(encoding="utf-8"))
    for c in MILK10K_CLASSES:
        assert c in data, f"Class {c} missing in disease_features.json"
        c_info = data[c]
        assert len(c_info["clinical_features"]) >= 2
        assert len(c_info["dermoscopic_features"]) >= 2
        assert len(c_info["sources"]) >= 1
        assert "darker_skin_considerations" in c_info
        assert len(c_info["darker_skin_considerations"]) > 10
        assert isinstance(c_info["needs_dermatologist_review"], bool)


def test_gradcam_and_gradcam_plus_plus():
    """Verify Grad-CAM and Grad-CAM++ produce normalized saliency maps matching image dimensions."""
    model = ClinicalOnlyClassifier(num_classes=11, pretrained=False)
    model.eval()

    cam_gen = GradCAM(model, branch="clinical")
    t_img = torch.randn(1, 3, 128, 128)

    # Standard Grad-CAM
    cam_std = cam_gen.generate_cam(clin_tensor=t_img, target_class_idx=1, use_gradcam_plus_plus=False)
    assert cam_std.shape == (128, 128)
    assert 0.0 <= cam_std.min() <= cam_std.max() <= 1.0

    # Grad-CAM++
    cam_plus = cam_gen.generate_cam(clin_tensor=t_img, target_class_idx=1, use_gradcam_plus_plus=True)
    assert cam_plus.shape == (128, 128)
    assert 0.0 <= cam_plus.min() <= cam_plus.max() <= 1.0

    # Overlay blending
    pil_img = Image.new("RGB", (128, 128), color=(200, 160, 130))
    blended = overlay_cam_on_image(pil_img, cam_std, alpha=0.5)
    assert isinstance(blended, Image.Image)
    assert blended.size == (128, 128)


def test_explain_sanity_results_file():
    """Verify explainability sanity results JSON exists and validates randomization + faithfulness."""
    sanity_data = load_explain_sanity_results()
    assert sanity_data["status"] == "success"

    rand_test = sanity_data["model_randomization_test"]
    assert rand_test["mean_spearman_correlation"] < rand_test["correlation_threshold_pass"]
    assert "PASSED" in rand_test["verdict"]

    faith = sanity_data["faithfulness_curves"]
    assert faith["mean_area_under_insertion_curve_auic"] > faith["mean_area_under_deletion_curve_audc"]
    assert "FAITHFUL" in faith["verdict"]
