"""
Unit and integration tests for Phase A: Single-Photo Mismatch & Input-Type Routing.
Verifies:
1. Input-type detection: 'clinical only', 'dermoscopic only', 'both'
2. ClinicalOnlyClassifier loading and inference with clinical image only (no dermoscopic duplication)
3. predict_ensemble with experiment 'a_cw_128' evaluates folds [0..4] with clinical tensor only
4. Low reliability flag and hedged medical wording for single-photo inputs
5. Error/warning handling when checkpoints are missing
6. Trilingual strings for Phase A across English, Hindi, and Kannada
"""

from pathlib import Path
import pytest
import torch
import numpy as np
from PIL import Image

from src.models.milk10k_classifier import ClinicalOnlyClassifier
from src.app.strings import STRINGS, t
from src.app.helpers import compute_input_hash
from src.app.app import (
    IMAGE_TRANSFORMS,
    get_available_checkpoint_folds,
    load_checkpoint_model,
    predict_ensemble,
)

VISHWAS_ROOT = Path(__file__).resolve().parent.parent


def test_phase_a_strings_trilingual_keys():
    """Verify all Phase A user-facing keys exist across EN, HI, KN."""
    required_keys = [
        "input_mode_label",
        "input_mode_both",
        "input_mode_clinical_only",
        "input_mode_dermo_only",
        "single_photo_clinical_note",
        "single_photo_dermo_note",
        "no_clinical_model_warn",
        "no_dermo_model_warn",
        "low_reliability_badge",
        "hedged_finding_prefix",
        "expert_model_route_clinical",
        "expert_model_route_fusion",
        "expert_model_route_dermo",
    ]
    for lang in ["en", "hi", "kn"]:
        for k in required_keys:
            assert k in STRINGS[lang], f"Missing key '{k}' in language '{lang}'"
            assert len(STRINGS[lang][k]) > 0


def test_clinical_only_model_loading():
    """Verify that clinical-only checkpoints (a_cw_128) load into ClinicalOnlyClassifier."""
    avail_folds = get_available_checkpoint_folds("a_cw_128")
    assert len(avail_folds) == 5, f"Expected 5 folds for a_cw_128, found {len(avail_folds)}"

    for fold in avail_folds:
        model = load_checkpoint_model("a_cw_128", fold)
        assert model is not None, f"Failed to load a_cw_128 fold {fold}"
        assert type(model).__name__ == "ClinicalOnlyClassifier" or isinstance(model, ClinicalOnlyClassifier)

        # Forward pass with clinical image only
        dummy_img = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            out = model(clinical_img=dummy_img)
            assert out.shape == (1, 11), f"Expected (1, 11), got {out.shape}"


def test_predict_ensemble_clinical_only_routing():
    """Verify predict_ensemble routes clinical-only input without dermoscopic or tabular tensors."""
    avail_folds = get_available_checkpoint_folds("a_cw_128")
    dummy_c = Image.new("RGB", (128, 128), color=(200, 120, 100))
    c_tensor = IMAGE_TRANSFORMS(dummy_c).unsqueeze(0)

    # Call predict_ensemble with clin_tensor only (dermo_tensor and tab_tensor are None)
    probs = predict_ensemble("a_cw_128", clin_tensor=c_tensor, folds=avail_folds)
    assert len(probs) == 11
    assert np.isclose(np.sum(probs), 1.0, atol=1e-3)
    assert not np.allclose(probs, [1.0/11.0]*11), "Returned uniform fake probabilities!"


def test_missing_model_raises_runtime_error():
    """Verify that requesting a missing experiment raises RuntimeError."""
    dummy_tensor = torch.randn(1, 3, 128, 128)
    with pytest.raises(RuntimeError) as exc_info:
        predict_ensemble("nonexistent_exp", clin_tensor=dummy_tensor, folds=[0])
    assert "Model files not found" in str(exc_info.value)


def test_input_hash_distinguishes_input_types():
    """Verify that hashing distinguishes clinical-only vs both views."""
    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    h_both = compute_input_hash(img, img, 50.0, "male", "torso", model_type="both")
    h_clin = compute_input_hash(img, None, 50.0, "male", "torso", model_type="clinical only")
    assert h_both != h_clin, "Hash must differentiate 'both' from 'clinical only'"
