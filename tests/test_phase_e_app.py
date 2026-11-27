"""
Headless Streamlit AppTest for Phase E Grounded Explanation and Abstention UI verification
tests/test_phase_e_app.py
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


def test_phase_e_app_grounded_explanation_expert():
    """Verify Streamlit app in Expert Mode renders Grounded Explanation and Faithfulness Audit."""
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
    assert not at.exception, f"App raised exception in Expert Mode: {at.exception}"

    md_texts = [m.value for m in at.markdown]

    # Verify Grounded Explanation section header exists
    has_grounded_hdr = any("Grounded Clinical Context & Guidance" in m or "साक्ष्य-आधारित नैदानिक संदर्भ" in m or "ಆಧಾರಿತ ಕ್ಲಿನಿಕಲ್ ಸಂದರ್ಭ" in m for m in md_texts)
    assert has_grounded_hdr, "Grounded Clinical Context & Guidance section missing in Expert mode"

    # Verify metric for Sentence Grounding exists
    metric_labels = [met.label for met in at.metric]
    assert any("Sentence Grounding" in lbl or "Retrieved Chunks Used" in lbl or "Hallucination Risk" in lbl for lbl in metric_labels), "Faithfulness audit metrics missing in Expert mode"


def test_phase_e_app_simple_mode():
    """Verify Streamlit app in Simple Mode runs without errors."""
    app_path = Path(__file__).resolve().parent.parent / "src" / "app" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception, f"App raised exception: {at.exception}"
