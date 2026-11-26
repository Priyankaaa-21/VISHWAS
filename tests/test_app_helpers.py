"""
Unit and Integration Tests for VISHWAS App Helpers, Strings, and Triage Flow.
Tests:
- Image quality checks (blur, brightness, resolution)
- Confidence labeling with empirical OOF thresholds
- Traffic-light triage verdict logic (Red, Amber, Green, Grey all reachable)
- OOF predictions evidence-line computation
- Next steps clinical advice by condition category in en, hi, and kn
- Input cache key hashing determinism and sensitivity
- Strings dictionary key parity and non-emptiness across en, hi, and kn
- Condition descriptions completeness across all 11 MILK10k classes
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
    compute_triage_verdict,
    load_oof_prediction_bins,
    get_evidence_line,
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
    assert any("bright" in w.lower() or "glare" in w.lower() or "overexposed" in w.lower() for w in warnings)


def test_check_image_quality_low_resolution():
    # 32x32 is below min_resolution=64
    img = Image.new("RGB", (32, 32), color=(100, 100, 100))

    is_ok, warnings = check_image_quality(img, DEFAULT_APP_CONFIG, lang="en")
    assert is_ok is False
    assert any("resolution" in w.lower() or "px" in w.lower() for w in warnings)


# ==============================================================================
# 2. Confidence Labeling (Empirical OOF Cut-offs: High >=0.55, Medium >=0.30, Low <0.30)
# ==============================================================================

def test_confidence_labels_multilingual():
    # English
    assert "High" in get_confidence_label(0.70, DEFAULT_APP_CONFIG, "en")
    assert "High" in get_confidence_label(0.55, DEFAULT_APP_CONFIG, "en")
    assert "Medium" in get_confidence_label(0.40, DEFAULT_APP_CONFIG, "en")
    assert "Medium" in get_confidence_label(0.30, DEFAULT_APP_CONFIG, "en")
    assert "Low" in get_confidence_label(0.25, DEFAULT_APP_CONFIG, "en")

    # Hindi
    assert "उच्च" in get_confidence_label(0.60, DEFAULT_APP_CONFIG, "hi")
    assert "मध्यम" in get_confidence_label(0.40, DEFAULT_APP_CONFIG, "hi")
    assert "कम" in get_confidence_label(0.20, DEFAULT_APP_CONFIG, "hi")

    # Kannada
    assert "ಹೆಚ್ಚಿನ" in get_confidence_label(0.60, DEFAULT_APP_CONFIG, "kn")
    assert "ಮಧ್ಯಮ" in get_confidence_label(0.40, DEFAULT_APP_CONFIG, "kn")
    assert "ಕಡಿಮೆ" in get_confidence_label(0.20, DEFAULT_APP_CONFIG, "kn")


# ==============================================================================
# 3. Traffic-Light Triage Verdict Logic (All 4 Colours Reachable)
# ==============================================================================

def test_triage_verdict_red_malignant_top1():
    # Top-1 is Melanoma (malignant)
    res = compute_triage_verdict("MEL", conf=0.70, malignant_prob=0.75, lang="en")
    assert res["level"] == "red"
    assert res["badge_color"] == "#dc2626"
    assert "doctor soon" in res["title"].lower()


def test_triage_verdict_red_high_malignant_prob():
    # Top-1 is benign (NV), but combined malignant probability >= 0.35 threshold
    res = compute_triage_verdict("NV", conf=0.45, malignant_prob=0.40, lang="en")
    assert res["level"] == "red"
    assert res["badge_color"] == "#dc2626"


def test_triage_verdict_amber_inflammatory():
    # Top-1 is inflammatory (INF), moderate confidence
    res = compute_triage_verdict("INF", conf=0.45, malignant_prob=0.10, lang="en")
    assert res["level"] == "amber"
    assert res["badge_color"] == "#d97706"
    assert "checked" in res["title"].lower()


def test_triage_verdict_amber_low_confidence():
    # Top-1 is benign (NV), but confidence is low (0.28 < 0.30)
    res = compute_triage_verdict("NV", conf=0.28, malignant_prob=0.15, lang="en")
    assert res["level"] == "amber"
    assert res["badge_color"] == "#d97706"


def test_triage_verdict_green_benign_safe():
    # Top-1 is benign (NV), high confidence, low malignant probability
    res = compute_triage_verdict("NV", conf=0.60, malignant_prob=0.08, lang="en")
    assert res["level"] == "green"
    assert res["badge_color"] == "#16a34a"
    assert "harmless" in res["title"].lower()


def test_triage_verdict_grey_quality_warning():
    # Photo quality warnings fired
    res = compute_triage_verdict("MEL", conf=0.70, malignant_prob=0.75, has_quality_warnings=True, lang="en")
    assert res["level"] == "grey"
    assert res["badge_color"] == "#64748b"
    assert "retake" in res["title"].lower()


def test_triage_verdict_grey_very_low_confidence():
    # Confidence is below very_low_threshold (0.18 < 0.20)
    res = compute_triage_verdict("NV", conf=0.18, malignant_prob=0.10, lang="en")
    assert res["level"] == "grey"
    assert res["badge_color"] == "#64748b"


# ==============================================================================
# 4. Evidence Line from OOF Predictions
# ==============================================================================

def test_evidence_line_from_oof_bins():
    bins = load_oof_prediction_bins()
    assert bins is not None and len(bins) >= 3

    # High confidence evidence line
    ev_en = get_evidence_line(0.70, bins, lang="en")
    assert ev_en is not None
    assert "In testing, when the model was this confident" in ev_en
    assert "% of the time" in ev_en

    # Hindi evidence line
    ev_hi = get_evidence_line(0.70, bins, lang="hi")
    assert ev_hi is not None
    assert "% बार सही था" in ev_hi

    # Kannada evidence line
    ev_kn = get_evidence_line(0.70, bins, lang="kn")
    assert ev_kn is not None
    assert "% ಬಾರಿ ಸರಿಯಾಗಿತ್ತು" in ev_kn


# ==============================================================================
# 5. Next Steps Clinical Guidance
# ==============================================================================

def test_get_next_steps_multilingual():
    # English
    mel_en = get_next_steps("MEL", high_risk=False, lang="en")
    assert "doctor soon" in mel_en.lower()

    # Hindi
    mel_hi = get_next_steps("MEL", high_risk=False, lang="hi")
    assert "डॉक्टर" in mel_hi

    # Kannada
    mel_kn = get_next_steps("MEL", high_risk=False, lang="kn")
    assert "ವೈದ್ಯರನ್ನು" in mel_kn


# ==============================================================================
# 6. Input Hash Determinism & Sensitivity
# ==============================================================================

def test_compute_input_hash():
    img1 = Image.new("RGB", (64, 64), color=(50, 100, 150))
    img2 = Image.new("RGB", (64, 64), color=(60, 110, 160))

    h1 = compute_input_hash(img1, None, 50.0, "male", "torso")
    h2 = compute_input_hash(img1, None, 50.0, "male", "torso")
    assert h1 == h2

    h_age = compute_input_hash(img1, None, 55.0, "male", "torso")
    assert h1 != h_age

    h_sex = compute_input_hash(img1, None, 50.0, "female", "torso")
    assert h1 != h_sex

    h_img = compute_input_hash(img2, None, 50.0, "male", "torso")
    assert h1 != h_img


# ==============================================================================
# 7. Strings Dictionary Key Parity Across English, Hindi, and Kannada
# ==============================================================================

def test_strings_key_parity_trilingual():
    en_keys = set(STRINGS["en"].keys())
    hi_keys = set(STRINGS["hi"].keys())
    kn_keys = set(STRINGS["kn"].keys())

    assert en_keys == hi_keys, f"Diff en vs hi: {en_keys ^ hi_keys}"
    assert en_keys == kn_keys, f"Diff en vs kn: {en_keys ^ kn_keys}"

    for lang in ["en", "hi", "kn"]:
        for k, v in STRINGS[lang].items():
            assert isinstance(v, str) and len(v.strip()) > 0, f"Empty string for key '{k}' in lang '{lang}'"


# ==============================================================================
# 8. Condition Descriptions Completeness Across All 11 Classes
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
        assert "kn_name" in meta and len(meta["kn_name"]) > 0
        assert "en_desc" in meta and len(meta["en_desc"]) > 0
        assert "hi_desc" in meta and len(meta["hi_desc"]) > 0
        assert "kn_desc" in meta and len(meta["kn_desc"]) > 0
        assert meta["category"] in valid_categories


# ==============================================================================
# 9. PDF Report Generation
# ==============================================================================

def test_generate_pdf_report():
    img = Image.new("RGB", (100, 100), color=(180, 80, 80))
    sample_result = {
        "pred_class": "MEL",
        "pred_prob": 0.82,
        "conf": 0.82,
        "probs": [0.82, 0.10, 0.05] + [0.003] * 8,
        "top3": [
            {"code": "MEL", "prob": 0.82},
            {"code": "NV", "prob": 0.10},
            {"code": "BCC", "prob": 0.05},
        ],
        "high_risk": True,
        "age": 55.0,
        "sex": "male",
        "site": "torso",
        "model_type": "two_image_tabular_fusion",
        "fold": 0,
    }

    pdf_bytes = generate_pdf_report(sample_result, c_img=img, d_img=img, lang="en")
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")


# ==============================================================================
# 10. AppTest Smoke Test
# ==============================================================================

def test_apptest_smoke_simple_mode():
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        return

    app_path = VISHWAS_ROOT / "src" / "app" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception, f"AppTest raised an unhandled exception: {at.exception}"
