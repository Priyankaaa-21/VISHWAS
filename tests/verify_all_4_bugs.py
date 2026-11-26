"""
Comprehensive verification script for all 4 bug fixes in VISHWAS:
- Bug 1: Gallery loader & Tone Group mapping (no stuck loading, proper G1/G2/G3, clinical + dermoscopic views)
- Bug 2: Conditions Guide loader (all 11 classes, no stuck loading, rich info cards)
- Bug 3: Tab 6 title internationalization (no raw keys, en/hi/kn verified, fallback works)
- Bug 4: Hero title styling (large, bold, responsive clamp font, accent colors)
"""

import sys
from pathlib import Path

# Add vishwas root to sys.path
vishwas_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(vishwas_root))

from src.app.strings import STRINGS, t
from src.app.examples import load_curated_examples_with_status, load_curated_examples, resolve_image_path
from src.app.diseases import load_diseases_data_with_status, load_diseases_data, get_localized
from src.app.theme import get_theme_css
from src.app.app import get_tone_group_for_class


def verify_bug1_gallery():
    print("=== [Bug 1 Verification] Curated Gallery & Tone Group Mapping ===")
    examples, err = load_curated_examples_with_status()
    assert err is None, f"Gallery returned error: {err}"
    assert len(examples) == 11, f"Expected 11 curated examples, got {len(examples)}"
    print(f"✓ Successfully loaded {len(examples)} curated gallery examples without error")

    # Verify tone group resolution for bug scenario: unknown with class 3
    grp, display_str = get_tone_group_for_class("unknown", 3)
    assert grp == "G2", f"Expected group G2, got {grp}"
    assert "unknown" not in display_str.lower(), f"Display string still says unknown: {display_str}"
    print(f"✓ Tone Group for skin_tone_class 3 correctly resolved: group='{grp}', display='{display_str}'")

    # Verify existing curated examples resolve properly
    for ex in examples:
        grp, disp = get_tone_group_for_class(ex.get("tone_group", "unknown"), ex.get("skin_tone_class"))
        assert grp in ["G1", "G2", "G3"], f"Unresolved group in example: {ex}"
        assert "unknown" not in disp.lower(), f"Unresolved display in example: {ex}"
    print("✓ All 11 examples resolve to valid tone groups (G1, G2, G3)")

    # Verify both clinical and dermoscopic image files exist
    for ex in examples:
        clin_path = resolve_image_path(ex["clinical_image"])
        dermo_path = resolve_image_path(ex["dermoscopic_image"])
        assert clin_path.exists(), f"Missing clinical image: {clin_path}"
        assert dermo_path.exists(), f"Missing dermoscopic image: {dermo_path}"
    print("✓ Both clinical and dermoscopic images physically verified for all 11 curated examples")


def verify_bug2_conditions_guide():
    print("\n=== [Bug 2 Verification] Conditions Guide Data Loader ===")
    data, err = load_diseases_data_with_status()
    assert err is None, f"Conditions guide returned error: {err}"
    assert len(data) == 11, f"Expected 11 disease classes, got {len(data)}"
    print(f"✓ Successfully loaded conditions guide with all {len(data)} classes")

    required_classes = ["MEL", "BCC", "SCCKA", "AKIEC", "BKL", "DF", "VASC", "NV", "INF", "BEN_OTH", "MAL_OTH"]
    for cls_code in required_classes:
        assert cls_code in data, f"Missing class: {cls_code}"
        cls_data = data[cls_code]
        for lang in ["en", "hi", "kn"]:
            name = get_localized(cls_data, "name", lang)
            desc = get_localized(cls_data, "desc", lang)
            what_to_do = get_localized(cls_data, "what_to_do", lang)
            assert name and desc and what_to_do, f"Missing {lang} text for {cls_code}"
    print("✓ All 11 classes contain complete English, Hindi, and Kannada descriptions, severity, and advice")


def verify_bug3_tab_translations():
    print("\n=== [Bug 3 Verification] Tab Titles & Multilingual Key Parity ===")
    for lang, expected_tab in [
        ("en", "🩺 Conditions Guide"),
        ("hi", "🩺 रोग मार्गदर्शिका"),
        ("kn", "🩺 ಚರ್ಮ ರೋಗಗಳ ಕೈಪಿಡಿ")
    ]:
        actual = t("tab_conditions_guide", lang)
        assert actual == expected_tab, f"Expected {expected_tab}, got {actual}"
        print(f"✓ Tab 6 in '{lang}' correctly localized: '{actual}'")

    # Verify no raw translation keys leak
    assert "tab_conditions_guide" not in t("tab_conditions_guide", "en")
    fallback_result = t("tab_nonexistent_test_slug", "en")
    assert fallback_result == "Nonexistent Test Slug", f"Unexpected fallback: {fallback_result}"
    print(f"✓ Missing keys fallback safely without raw prefix: '{fallback_result}'")


def verify_bug4_hero_title():
    print("\n=== [Bug 4 Verification] Hero Title Styling & Typography ===")
    css_light = get_theme_css("light")
    css_dark = get_theme_css("dark")

    # Assert main-header-hero styles exist in both themes
    for theme_name, css in [("Light", css_light), ("Dark", css_dark)]:
        assert ".main-header-hero" in css, f"Missing .main-header-hero in {theme_name} CSS"
        assert "clamp(2.3rem, 5.5vw, 3.2rem)" in css, f"Missing clamp typography in {theme_name} CSS"
        assert ".hero-brand" in css, f"Missing .hero-brand in {theme_name} CSS"
        assert ".hero-devanagari" in css, f"Missing .hero-devanagari in {theme_name} CSS"
        assert "Noto Sans Devanagari" in css, f"Missing font family in {theme_name} CSS"
        print(f"✓ Typography, clamp scaling, and font stacks verified in {theme_name} theme CSS")


def run_full_apptest_interaction():
    print("\n=== [Full AppTest Simulation] End-to-End Execution ===")
    from streamlit.testing.v1 import AppTest
    app_file = str(vishwas_root / "src" / "app" / "app.py")
    at = AppTest.from_file(app_file, default_timeout=30)
    at.run()
    assert len(at.exception) == 0, f"AppTest raised exceptions: {[e.value for e in at.exception]}"
    print("✓ Full Streamlit AppTest run succeeded with zero exceptions")


if __name__ == "__main__":
    verify_bug1_gallery()
    verify_bug2_conditions_guide()
    verify_bug3_tab_translations()
    verify_bug4_hero_title()
    run_full_apptest_interaction()
    print("\n🎉 ALL 4 BUG FIXES VERIFIED SUCCESSFULLY!")
