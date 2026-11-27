"""
Headless Streamlit AppTest for Phase B UI verification
tests/test_phase_b_app.py
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


def test_phase_b_app_simple_mode_and_conformal():
    """Verify Streamlit app in Simple Mode loads conformal prediction and displays it."""
    app_path = Path(__file__).resolve().parent.parent / "src" / "app" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception, f"App raised exception: {at.exception}"

    # Initial state: disclaimer should be present
    warnings = [w.value for w in at.warning]
    assert any("Research prototype" in w or "चिकित्सा उपकरण नहीं" in w or "ವೈದ್ಯಕೀಯ ಸಾಧನವಲ್ಲ" in w for w in warnings)

    # Click Melanoma example button
    # Buttons: find button for example mel
    mel_btn = None
    for b in at.button:
        if "Melanoma" in b.label or "मेलेनोमा" in b.label or "ಮೆಲನೋಮಾ" in b.label:
            mel_btn = b
            break
    
    assert mel_btn is not None, "Melanoma example button not found"
    mel_btn.click().run()
    assert not at.exception, f"App raised exception after clicking example: {at.exception}"

    # Check that conformal prediction set is displayed in markdown
    md_texts = [m.value for m in at.markdown]
    has_conformal_likely = any("The AI thinks it is most likely one of:" in m or "AI का अनुमान है" in m or "AI ಇದು ಈ ಕೆಳಗಿನವುಗಳಲ್ಲಿ ಒಂದಾಗಿರಬಹುದು" in m for m in md_texts)
    assert has_conformal_likely, "Conformal prediction set sentence not found in result markdown"


def test_phase_b_app_fairness_tab():
    """Verify Tab 2 (How Fair Is It?) displays Conformal Equity section and table."""
    app_path = Path(__file__).resolve().parent.parent / "src" / "app" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception, f"App raised exception: {at.exception}"

    # Check for Conformal Prediction Coverage header in markdown
    md_texts = [m.value for m in at.markdown]
    has_conformal_fairness_hdr = any("Conformal Prediction Coverage & Calibration Equity" in m for m in md_texts)
    assert has_conformal_fairness_hdr, "Conformal fairness section header not found in app markdown"

    # Check that dataframe for conformal equity is rendered
    dfs = [df.value for df in at.dataframe]
    assert len(dfs) > 0, "No dataframes found in app"
    # Find dataframe with Target Coverage or Empirical Coverage
    equity_df_found = False
    for df in dfs:
        cols = list(df.columns)
        if any("Coverage" in c for c in cols):
            equity_df_found = True
            break
    assert equity_df_found, "Conformal equity comparison dataframe not found"


def test_phase_b_app_expert_mode_conformal():
    """Verify Streamlit app in Expert Mode renders Conformal Prediction section."""
    app_path = Path(__file__).resolve().parent.parent / "src" / "app" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception, f"App raised exception: {at.exception}"

    # Switch to Expert Mode via sidebar radio
    mode_radio = None
    for r in at.sidebar.radio:
        if "Interface Mode" in r.label or "मोड" in r.label or "ಮೋಡ್" in r.label:
            mode_radio = r
            break
    assert mode_radio is not None, "Mode selector radio not found in sidebar"
    mode_radio.set_value("Expert (Technical)").run()
    assert not at.exception, f"App raised exception in Expert Mode: {at.exception}"

    # Check for Calibrated Conformal Prediction header in markdown
    md_texts = [m.value for m in at.markdown]
    has_expert_hdr = any("Calibrated Conformal Prediction" in m for m in md_texts)
    assert has_expert_hdr, "Expert conformal section header not found"

    # Check metrics (Set Size, Threshold, Temperature)
    metric_labels = [m.label for m in at.metric]
    assert any("Conformal Set Size" in l for l in metric_labels), "Conformal Set Size metric missing"
    assert any("Cutoff Probability Threshold" in l for l in metric_labels), "Cutoff Threshold metric missing"
    assert any("Temperature Applied" in l for l in metric_labels), "Temperature Applied metric missing"

