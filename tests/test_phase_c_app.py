"""
Headless Streamlit AppTest for Phase C Explainability UI verification
tests/test_phase_c_app.py
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


def test_phase_c_app_simple_mode_features():
    """Verify Streamlit app in Simple Mode renders observed feature summary."""
    app_path = Path(__file__).resolve().parent.parent / "src" / "app" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception, f"App raised exception: {at.exception}"

    # Click Melanoma example button
    mel_btn = None
    for b in at.button:
        if "Melanoma" in b.label or "मेलेनोमा" in b.label or "ಮೆಲನೋಮಾ" in b.label:
            mel_btn = b
            break
    assert mel_btn is not None
    mel_btn.click().run()
    assert not at.exception, f"App raised exception: {at.exception}"

    # Check for feature summary line in markdown
    md_texts = [m.value for m in at.markdown]
    has_feature_note = any("observed surface features commonly seen" in m or "लक्षणों को पहचाना" in m or "ಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಿದೆ" in m for m in md_texts)
    assert has_feature_note, "Observed feature summary line not found in Simple mode result"


def test_phase_c_app_expert_mode_explainability():
    """Verify Streamlit app in Expert Mode renders Feature Checklist and Grad-CAM section."""
    app_path = Path(__file__).resolve().parent.parent / "src" / "app" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception, f"App raised exception: {at.exception}"

    # Switch to Expert Mode
    mode_radio = None
    for r in at.sidebar.radio:
        if "Interface Mode" in r.label or "मोड" in r.label or "ಮೋಡ್" in r.label:
            mode_radio = r
            break
    assert mode_radio is not None
    mode_radio.set_value("Expert (Technical)").run()
    assert not at.exception, f"App raised exception: {at.exception}"

    md_texts = [m.value for m in at.markdown]
    # Check for Features Seen vs Typical Criteria
    has_features_checklist = any("Features Seen vs Typical Features" in m or "लक्षण बनाम रोग" in m or "ವಿರುದ್ಧ ವಿಶಿಷ್ಟ ಲಕ್ಷಣಗಳ" in m for m in md_texts)
    assert has_features_checklist, "Feature agreement checklist header missing in Expert mode"

    # Check for Grad-CAM section
    has_gradcam = any("Visual Focus & Attribution" in m or "Grad-CAM" in m for m in md_texts)
    assert has_gradcam, "Grad-CAM explainability header missing in Expert mode"

    # Check for Sanity checks expander
    expander_labels = [e.label for e in at.expander]
    has_sanity_expander = any("Explainability Sanity Checks" in l or "विश्वसनीयता जांच" in l or "ನಂಬಿಕಸ್ಥತೆ ಪರೀಕ್ಷೆಗಳು" in l for l in expander_labels)
    assert has_sanity_expander, "Explainability sanity checks expander missing in Expert mode"
