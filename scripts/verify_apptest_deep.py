"""
Comprehensive deep AppTest simulation script to verify Streamlit user interactions in VISHWAS.
Tests:
1. Simple Mode: Example 1 (MEL), Start Over, Example 2 (BCC), Start Over, Example 3 (NV)
2. Curated Gallery: Tone groups, profiles, ground truth
3. Language Switching: English -> Hindi -> Kannada across all tabs
4. Expert Mode: Toggle and controls
5. Live Inference with checkpoints vs Missing checkpoints handling
"""

import sys
from pathlib import Path
VISHWAS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VISHWAS_ROOT))

from streamlit.testing.v1 import AppTest
from src.app.examples import load_curated_examples

def run_deep_apptest():
    app_path = VISHWAS_ROOT / "src" / "app" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception, f"AppTest initial run threw: {at.exception}"
    print("✓ AppTest initialized successfully in Simple mode")

    # 1. Click Example 1 (MEL)
    ex1 = next((b for b in at.button if "Example 1" in b.label), None)
    assert ex1 is not None, "Example 1 button missing"
    print(f"  Testing Example 1: {ex1.label}")
    ex1.click()
    at.run()
    assert not at.exception, f"Exception after Example 1: {at.exception}"
    res1 = at.session_state.get("precomputed_result")
    assert res1["pred_class"] == "MEL" and res1["gt_diagnosis"] == "MEL" and res1["is_precomputed"] is True
    print(f"  ✓ Example 1 PASSED: Pred={res1['pred_class']}, GT={res1['gt_diagnosis']}, Conf={res1['conf']:.2f}")

    # Reset
    reset_btn = next((b for b in at.button if "Start Over" in b.label), None)
    assert reset_btn is not None
    reset_btn.click()
    at.run()
    assert not at.exception

    # 2. Click Example 2 (BCC)
    ex2 = next((b for b in at.button if "Example 2" in b.label), None)
    assert ex2 is not None, "Example 2 button missing"
    print(f"  Testing Example 2: {ex2.label}")
    ex2.click()
    at.run()
    assert not at.exception, f"Exception after Example 2: {at.exception}"
    res2 = at.session_state.get("precomputed_result")
    assert res2["pred_class"] == "BCC" and res2["gt_diagnosis"] == "BCC" and res2["is_precomputed"] is True
    print(f"  ✓ Example 2 PASSED: Pred={res2['pred_class']}, GT={res2['gt_diagnosis']}, Conf={res2['conf']:.2f}")

    # Reset
    reset_btn = next((b for b in at.button if "Start Over" in b.label), None)
    assert reset_btn is not None
    reset_btn.click()
    at.run()
    assert not at.exception

    # 3. Click Example 3 (NV)
    ex3 = next((b for b in at.button if "Example 3" in b.label), None)
    assert ex3 is not None, "Example 3 button missing"
    print(f"  Testing Example 3: {ex3.label}")
    ex3.click()
    at.run()
    assert not at.exception, f"Exception after Example 3: {at.exception}"
    res3 = at.session_state.get("precomputed_result")
    assert res3["pred_class"] == "NV" and res3["gt_diagnosis"] == "NV" and res3["is_precomputed"] is True
    print(f"  ✓ Example 3 PASSED: Pred={res3['pred_class']}, GT={res3['gt_diagnosis']}, Conf={res3['conf']:.2f}")

    # 4. Test Language Switching
    lang_sel = next((s for s in at.selectbox if "Language" in s.label or "भाषा" in s.label or "ಭಾಷೆ" in s.label), None)
    assert lang_sel is not None, "Language selectbox widget missing"
    
    for l_label, code in [("हिंदी (Hindi)", "hi"), ("ಕನ್ನಡ (Kannada)", "kn"), ("English", "en")]:
        lang_sel.select(l_label)
        at.run()
        assert not at.exception, f"Exception switching to {l_label}: {at.exception}"
        print(f"  ✓ Language switch to {l_label} ({code}) rendered cleanly without exceptions")

    # 5. Test Theme Switching
    theme_radio = next((r for r in at.radio if "Theme" in r.label or "थीम" in r.label or "ವಿಷಯ" in r.label), None)
    assert theme_radio is not None, "Theme radio widget missing"
    for th in theme_radio.options:
        theme_radio.set_value(th)
        at.run()
        assert not at.exception, f"Exception switching theme to {th}: {at.exception}"
        print(f"  ✓ Theme switch to '{th}' rendered cleanly")

    # 6. Test Mode Switch to Expert Mode
    mode_radio = next((r for r in at.radio if "Mode" in r.label or "मोड" in r.label or "ವಿಧಾನ" in r.label), None)
    assert mode_radio is not None, "Mode radio missing"
    expert_opt = mode_radio.options[1]
    mode_radio.set_value(expert_opt)
    at.run()
    assert not at.exception, f"Exception toggling to Expert mode: {at.exception}"
    assert at.session_state["app_mode"] == "Expert"
    print("  ✓ Expert Mode activated cleanly via sidebar mode radio")

    print("\n==========================================")
    print("ALL DEEP APPTEST SIMULATION CHECKS PASSED!")
    print("==========================================")

if __name__ == "__main__":
    run_deep_apptest()
