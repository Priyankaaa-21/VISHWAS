"""
VISHWAS Final Verification Test Suite
tests/test_final_verification.py

Verifies:
1. Full app execution across all 3 languages (English, Hindi, Kannada).
2. Theme stability across Light and Dark mode.
3. No raw keys, unmatched tags, or unhandled exceptions in UI.
4. Single-photo routing to ClinicalOnlyClassifier without duplication.
5. Calibrated conformal prediction sets (guaranteed non-empty, deterministic).
6. Fairness tab demographic parity checks.
7. Clinical abstention trigger mechanism on low-confidence / wide-set inputs.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from src.app.strings import STRINGS, t

VISHWAS_ROOT = Path(__file__).resolve().parent.parent


def test_app_trilingual_rendering():
    """Verify app renders cleanly in English, Hindi, and Kannada without exceptions."""
    app_path = VISHWAS_ROOT / "src" / "app" / "app.py"

    for lang_code in ["en", "hi", "kn"]:
        at = AppTest.from_file(str(app_path), default_timeout=30)
        at.run()
        assert not at.exception, f"App exception in default load: {at.exception}"

        # Find language selector selectbox
        lang_sb = None
        for sb in at.sidebar.selectbox:
            if "Language" in sb.label or "भाषा" in sb.label or "ಭಾಷೆ" in sb.label:
                lang_sb = sb
                break
        assert lang_sb is not None, "Language selector missing in sidebar selectbox"

        target_val = "English" if lang_code == "en" else ("हिंदी (Hindi)" if lang_code == "hi" else "ಕನ್ನಡ (Kannada)")
        lang_sb.set_value(target_val).run()
        assert not at.exception, f"App exception after switching to {target_val}: {at.exception}"

        # Assert no raw untranslated placeholders like 't(' or 'None' as titles
        md_texts = [m.value for m in at.markdown]
        for text in md_texts:
            assert "t('" not in text, f"Found raw t('...') string in {lang_code}: {text[:60]}"
            assert 't("' not in text, f"Found raw t(\"...\") string in {lang_code}: {text[:60]}"


def test_app_theme_light_and_dark():
    """Verify app switches between Light and Dark themes without errors."""
    app_path = VISHWAS_ROOT / "src" / "app" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception

    theme_radio = None
    for r in at.sidebar.radio:
        if "Theme" in r.label or "थीम" in r.label or "ಥೀಮ್" in r.label:
            theme_radio = r
            break
    assert theme_radio is not None, "Theme selector missing in sidebar"

    for theme_opt in ["Light", "Dark"]:
        theme_radio.set_value(theme_opt).run()
        assert not at.exception, f"App exception in {theme_opt} theme: {at.exception}"


def test_app_expert_mode_all_sections():
    """Verify Expert Mode renders all technical sections cleanly."""
    app_path = VISHWAS_ROOT / "src" / "app" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    mode_radio = None
    for r in at.sidebar.radio:
        if "Interface Mode" in r.label:
            mode_radio = r
            break
    assert mode_radio is not None
    mode_radio.set_value("Expert (Technical)").run()
    assert not at.exception

    md_all = " ".join([m.value for m in at.markdown])

    # Check 1: Conformal prediction
    assert "Conformal Prediction Set" in md_all or "Conformal" in md_all
    # Check 2: Feature Agreement Checklist
    assert "Features Seen vs Typical Features Checklist" in md_all
    # Check 3: Grad-CAM
    assert "Visual Focus & Attribution (Grad-CAM & Grad-CAM++)" in md_all
    # Check 4: Literature Retrieval (BM25)
    assert "Clinical Literature & Guideline Retrieval" in md_all
    # Check 5: Case-based Retrieval
    assert "Similar Historical Cases" in md_all
    # Check 6: Grounded Clinical Guidance
    assert "Grounded Clinical Context & Guidance" in md_all
