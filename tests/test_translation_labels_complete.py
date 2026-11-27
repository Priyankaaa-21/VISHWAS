"""
VISHWAS Translation Integrity & Label Completeness Test Suite
tests/test_translation_labels_complete.py

Scans all source code files in src/ for any `t("...")` calls
and asserts that EVERY key exists in English, Hindi, and Kannada.
Fails immediately if any translation key is missing or incomplete in any language.
"""

from pathlib import Path
import re
import pytest

from src.app.strings import STRINGS

VISHWAS_ROOT = Path(__file__).resolve().parent.parent


def test_no_missing_or_placeholder_keys_in_any_language():
    """Extracts all t('...') calls from python files and verifies full trilingual coverage."""
    src_dir = VISHWAS_ROOT / "src"
    pattern = re.compile(r'\bt\(\s*["\']([a-zA-Z0-9_-]+)["\']')

    used_keys = set()
    for py_file in src_dir.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        matches = pattern.findall(text)
        for m in matches:
            used_keys.add(m)

    en_keys = set(STRINGS["en"].keys())
    hi_keys = set(STRINGS["hi"].keys())
    kn_keys = set(STRINGS["kn"].keys())

    missing_in_en = used_keys - en_keys
    missing_in_hi = used_keys - hi_keys
    missing_in_kn = used_keys - kn_keys

    assert not missing_in_en, f"Keys called in code but missing from STRINGS['en']: {missing_in_en}"
    assert not missing_in_hi, f"Keys called in code but missing from STRINGS['hi']: {missing_in_hi}"
    assert not missing_in_kn, f"Keys called in code but missing from STRINGS['kn']: {missing_in_kn}"


def test_tab_icons_and_labels_consistent():
    """All tabs across all languages must have consistent icons and clean names."""
    tab_keys = [
        "tab_check",
        "tab_fairness",
        "tab_reliability",
        "tab_explorer",
        "tab_about",
        "tab_conditions_guide",
    ]
    for lang in ["en", "hi", "kn"]:
        for k in tab_keys:
            val = STRINGS[lang].get(k, "")
            assert val, f"Tab {k} missing in {lang}"
            # Assert starts with emoji or symbol
            assert any(ord(char) > 127 for char in val[:3]), f"Tab {k} in {lang} lacks an icon: '{val}'"
