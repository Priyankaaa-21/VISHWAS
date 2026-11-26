import re
from pathlib import Path
import pytest
from src.app.strings import STRINGS, t

def test_translation_fallback_behavior():
    """Verify missing keys fall back to English or human-readable generated titles."""
    # 1. Existing key returns correct translation
    assert t("tab_check", "en") == "🔬 Check a Skin Spot"
    assert t("tab_check", "hi") == "🔬 त्वचा के धब्बे की जांच"
    assert t("tab_check", "kn") == "🔬 ಚರ್ಮದ ಮಚ್ಚೆಯ ತಪಾಸಣೆ"

    # 2. Missing key in hi/kn falls back to English if present in en
    # Simulate a key only in en
    test_key = "tab_temp_test_xyz"
    STRINGS["en"][test_key] = "Temporary Tab Test"
    try:
        assert t(test_key, "hi") == "Temporary Tab Test"
        assert t(test_key, "kn") == "Temporary Tab Test"
    finally:
        del STRINGS["en"][test_key]

    # 3. Missing key entirely falls back to clean title-cased string, never raw key
    assert t("tab_conditions_guide_unknown", "en") == "Conditions Guide Unknown"
    assert t("msg_upload_warning_error", "hi") == "Msg Upload Warning Error"
    assert t("simple_step_photo", "kn") == "Simple Step Photo"


def test_translation_coverage_across_all_languages():
    """All translation keys present in 'en' must also be present in 'hi' and 'kn'."""
    en_keys = set(STRINGS["en"].keys())
    hi_keys = set(STRINGS["hi"].keys())
    kn_keys = set(STRINGS["kn"].keys())

    missing_in_hi = en_keys - hi_keys
    missing_in_kn = en_keys - kn_keys

    assert not missing_in_hi, f"Keys in 'en' missing from 'hi': {missing_in_hi}"
    assert not missing_in_kn, f"Keys in 'en' missing from 'kn': {missing_in_kn}"


def test_codebase_translation_keys_exist():
    """Scans src/app/*.py for all t('key_name'...) calls and asserts they exist in STRINGS."""
    app_dir = Path(__file__).resolve().parent.parent / "src" / "app"
    pattern = re.compile(r'\bt\(\s*["\']([a-zA-Z0-9_-]+)["\']')

    found_keys = set()
    for py_file in app_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        matches = pattern.findall(content)
        for m in matches:
            found_keys.add(m)

    en_keys = set(STRINGS["en"].keys())
    missing_from_strings = found_keys - en_keys

    assert not missing_from_strings, f"Translation keys called in code but missing from STRINGS['en']: {missing_from_strings}"
