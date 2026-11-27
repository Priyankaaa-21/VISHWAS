"""
Headless Streamlit AppTest for Phase D RAG and Case Retrieval UI verification
tests/test_phase_d_app.py
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


def test_phase_d_app_rag_and_cases():
    """Verify Streamlit app in Expert Mode renders BM25 knowledge retrieval and similar cases."""
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

    # Verify Clinical Literature Retrieval (BM25) header
    has_kb_hdr = any("Clinical Literature & Guideline Retrieval" in m or "नैदानिक साहित्य" in m or "ಕ್ಲಿನಿಕಲ್ ಸಾಹಿತ್ಯ" in m for m in md_texts)
    assert has_kb_hdr, "Clinical Literature Retrieval (BM25) section missing in Expert mode"

    # Verify Similar Historical Cases header
    has_cases_hdr = any("Similar Historical Cases" in m or "समान ऐतिहासिक मामले" in m or "ಹೋಲುವ ಐತಿಹಾಸಿಕ ಪ್ರಕರಣಗಳು" in m for m in md_texts)
    assert has_cases_hdr, "Similar Historical Cases section missing in Expert mode"
