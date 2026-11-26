"""
Unit and Integration Tests for VISHWAS App Helpers, Strings, and Simple Mode Flow.
Tests:
- Image quality checks (blur, brightness, resolution)
- Confidence labeling (High / Medium / Low) in en and hi
- Next steps clinical advice by condition category in en and hi
- Input cache key hashing determinism and sensitivity
- Strings dictionary key parity and non-emptiness
- Condition descriptions completeness for all 11 MILK10k classes
- PDF diagnostic report generation
- AppTest smoke test for Simple Mode initialization
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure vishwas root is on sys.path
VISHWAS_ROOT = Path(__file__).resolve().parent.parent
if str(VISHWAS_ROOT) not in sys.path:
    sys.path.insert(0, str(VISHWAS_ROOT))

from src.app.strings import CONDITION_DESCRIPTIONS, STRINGS, t
from src.app.helpers import (
    DEFAULT_APP_CONFIG,
    MALIGNANT_OR_PREMALIGNANT_CLASSES,
    check_image_quality,
    get_confidence_label,
    get_next_steps,
    compute_input_hash,
    generate_pdf_report,
)


# ==============================================================================
# 1. Image Quality Checks
# ==============================================================================

def test_check_image_quality_sharp_and_clear():
    # Checkerboard pattern around luminance 128 provides strong Laplacian variance and adequate brightness
    arr = np.full((128, 128, 3), 128, dtype=np.uint8)
    arr[::2, ::2] = 220
    arr[1::2, 1::2] = 40
    img = Image.fromarray(arr)

    is_ok, warnings = check_image_quality(img, DEFAULT_APP_CONFIG, lang="en")
    assert is_ok is True
    assert len(warnings) == 0


def test_check_image_quality_blurry():
    # Uniform solid image has zero Laplacian variance (extremely blurry / out of focus)
    arr = np.full((128, 128, 3), 120, dtype=np.uint8)
    img = Image.fromarray(arr)

    is_ok, warnings = check_image_quality(img, DEFAULT_APP_CONFIG, lang="en")
    assert is_ok is False
    assert any("blurry" in w.lower() or "focus" in w.lower() for w in warnings)


def test_check_image_quality_too_dark():
    # Luminance < 30
    arr = np.full((128, 128, 3), 10, dtype=np.uint8)
    img = Image.fromarray(arr)

    is_ok, warnings = check_image_quality(img, DEFAULT_APP_CONFIG, lang="en")
    assert is_ok is False
    assert any("dark" in w.lower() for w in warnings)


def test_check_image_quality_too_bright():
    # Luminance > 225
    arr = np.full((128, 128, 3), 245, dtype=np.uint8)
    img = Image.fromarray(arr)

    is_ok, warnings = check_image_quality(img, DEFAULT_APP_CONFIG, lang="en")
    assert is_ok is False
    assert any("bright" in w.lower() for w in warnings)


def test_check_image_quality_low_resolution():
    # 32x32 is below min_resolution=64
    img = Image.new("RGB", (32, 32), color=(100, 100, 100))

    is_ok, warnings = check_image_quality(img, DEFAULT_APP_CONFIG, lang="en")
    assert is_ok is False
    assert any("resolution" in w.lower() or "px" in w.lower() for w in warnings)


# ==============================================================================
# 2. Confidence Labeling
# ==============================================================================

def test_confidence_labels_english():
    assert "High" in get_confidence_label(0.85, DEFAULT_APP_CONFIG, "en")
    assert "High" in get_confidence_label(0.70, DEFAULT_APP_CONFIG, "en")
    assert "Medium" in get_confidence_label(0.55, DEFAULT_APP_CONFIG, "en")
    assert "Medium" in get_confidence_label(0.40, DEFAULT_APP_CONFIG, "en")
    assert "Low" in get_confidence_label(0.25, DEFAULT_APP_CONFIG, "en")


def test_confidence_labels_hindi():
    hi_high = get_confidence_label(0.85, DEFAULT_APP_CONFIG, "hi")
    hi_med = get_confidence_label(0.55, DEFAULT_APP_CONFIG, "hi")
    hi_low = get_confidence_label(0.25, DEFAULT_APP_CONFIG, "hi")
    assert "उच्च" in hi_high
    assert "मध्यम" in hi_med
    assert "कम" in hi_low


# ==============================================================================
# 3. Next Steps Clinical Guidance
# ==============================================================================

def test_get_next_steps_categories_en():
    # Malignant
    mel_advice = get_next_steps("MEL", high_risk=False, lang="en")
    assert "doctor soon" in mel_advice.lower()

    # Premalignant
    aki_advice = get_next_steps("AKIEC", high_risk=False, lang="en")
    assert "doctor" in aki_advice.lower()

    # Benign
    nv_advice = get_next_steps("NV", high_risk=False, lang="en")
    assert "harmless" in nv_advice.lower()

    # Non-tumour
    inf_advice = get_next_steps("INF", high_risk=False, lang="en")
    assert "treatment" in inf_advice.lower()

    # High-risk override for benign with high malignant sum
    nv_high_risk = get_next_steps("NV", high_risk=True, lang="en")
    assert "doctor soon" in nv_high_risk.lower()


def test_get_next_steps_categories_hi():
    mel_advice_hi = get_next_steps("MEL", high_risk=False, lang="hi")
    nv_advice_hi = get_next_steps("NV", high_risk=False, lang="hi")
    assert "डॉक्टर" in mel_advice_hi
    assert "हानिरहित" in nv_advice_hi or "डॉक्टर" in nv_advice_hi


# ==============================================================================
# 4. Input Hash Determinism & Sensitivity
# ==============================================================================

def test_compute_input_hash():
    img1 = Image.new("RGB", (64, 64), color=(50, 100, 150))
    img2 = Image.new("RGB", (64, 64), color=(60, 110, 160))

    h1 = compute_input_hash(img1, None, 50.0, "male", "torso")
    h2 = compute_input_hash(img1, None, 50.0, "male", "torso")
    assert h1 == h2  # Deterministic

    # Changing age
    h_age = compute_input_hash(img1, None, 55.0, "male", "torso")
    assert h1 != h_age

    # Changing sex
    h_sex = compute_input_hash(img1, None, 50.0, "female", "torso")
    assert h1 != h_sex

    # Changing image
    h_img = compute_input_hash(img2, None, 50.0, "male", "torso")
    assert h1 != h_img


# ==============================================================================
# 5. Strings Dictionary Parity
# ==============================================================================

def test_strings_key_parity():
    en_keys = set(STRINGS["en"].keys())
    hi_keys = set(STRINGS["hi"].keys())

    missing_in_hi = en_keys - hi_keys
    missing_in_en = hi_keys - en_keys

    assert not missing_in_hi, f"Keys in 'en' missing from 'hi': {missing_in_hi}"
    assert not missing_in_en, f"Keys in 'hi' missing from 'en': {missing_in_en}"


def test_strings_non_empty():
    for lang in ["en", "hi"]:
        for k, v in STRINGS[lang].items():
            assert isinstance(v, str) and len(v.strip()) > 0, f"Empty string for key '{k}' in lang '{lang}'"


# ==============================================================================
# 6. Condition Descriptions Completeness (11 Classes)
# ==============================================================================

EXPECTED_CLASSES = {
    "AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"
}

def test_condition_descriptions_completeness():
    assert set(CONDITION_DESCRIPTIONS.keys()) == EXPECTED_CLASSES
    valid_categories = {"malignant", "premalignant", "benign", "nontumour"}

    for c_code, meta in CONDITION_DESCRIPTIONS.items():
        assert "en_name" in meta and len(meta["en_name"]) > 0
        assert "hi_name" in meta and len(meta["hi_name"]) > 0
        assert "en_desc" in meta and len(meta["en_desc"]) > 0
        assert "hi_desc" in meta and len(meta["hi_desc"]) > 0
        assert meta["category"] in valid_categories, f"Invalid category '{meta.get('category')}' for {c_code}"


def test_malignant_classes_consistency():
    for c_code in MALIGNANT_OR_PREMALIGNANT_CLASSES:
        assert c_code in EXPECTED_CLASSES
        cat = CONDITION_DESCRIPTIONS[c_code]["category"]
        assert cat in {"malignant", "premalignant"}


# ==============================================================================
# 7. PDF Report Generation
# ==============================================================================

def test_generate_pdf_report():
    img = Image.new("RGB", (100, 100), color=(180, 80, 80))
    sample_result = {
        "pred_class": "MEL",
        "pred_prob": 0.82,
        "probs": [0.82, 0.10, 0.05] + [0.003] * 8,
        "top3": [
            ("MEL", 0.82, "Melanoma"),
            ("NV", 0.10, "Melanocytic Nevus"),
            ("BCC", 0.05, "Basal Cell Carcinoma"),
        ],
        "high_risk": True,
        "age": 55.0,
        "sex": "male",
        "site": "torso",
        "model_type": "two_image_tabular_fusion",
        "fold": 0,
        "single_photo_fallback": False,
    }

    pdf_bytes = generate_pdf_report(sample_result, c_img=img, d_img=img, lang="en")
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")


# ==============================================================================
# 8. AppTest Smoke Test
# ==============================================================================

def test_apptest_smoke_simple_mode():
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        return  # AppTest optional if environment does not support it

    app_path = VISHWAS_ROOT / "src" / "app" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception, f"AppTest raised an unhandled exception: {at.exception}"
