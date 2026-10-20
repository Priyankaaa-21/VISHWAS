"""
Unit tests for Individual Typology Angle (ITA) extraction and skin-tone mapping.
"""

import math
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VISHWAS_ROOT = Path(__file__).resolve().parent.parent
if str(VISHWAS_ROOT) not in sys.path:
    sys.path.insert(0, str(VISHWAS_ROOT))

try:
    from vishwas.src.preprocess.skin_tone import (
        rgb_to_cielab,
        compute_ita,
        ita_to_group,
        fst_to_group,
        extract_skin_ita_from_image,
    )
except ImportError:
    from src.preprocess.skin_tone import (
        rgb_to_cielab,
        compute_ita,
        ita_to_group,
        fst_to_group,
        extract_skin_ita_from_image,
    )


def test_compute_ita_known_values():
    """Verify standard geometric angles for ITA calculation."""
    # When L = 50, (L - 50) = 0 -> arctan(0) = 0 degrees
    assert pytest.approx(compute_ita(50.0, 10.0), abs=1e-2) == 0.0

    # When L = 60, b = 10 -> (60-50)/10 = 1.0 -> arctan(1) = 45 degrees
    assert pytest.approx(compute_ita(60.0, 10.0), abs=1e-2) == 45.0

    # When L = 40, b = 10 -> (40-50)/10 = -1.0 -> arctan(-1) = -45 degrees
    assert pytest.approx(compute_ita(40.0, 10.0), abs=1e-2) == -45.0


def test_ita_to_group_thresholds():
    """Verify group categorization thresholds."""
    assert ita_to_group(55.0) == "light"
    assert ita_to_group(41.1) == "light"
    assert ita_to_group(41.0) == "medium"
    assert ita_to_group(35.0) == "medium"
    assert ita_to_group(28.0) == "medium"
    assert ita_to_group(27.9) == "dark"
    assert ita_to_group(-15.0) == "dark"


def test_fst_to_group_mapping():
    """Verify Fitzpatrick scale 1-6 mapping to Light/Medium/Dark."""
    assert fst_to_group(1) == "light"
    assert fst_to_group(2) == "light"
    assert fst_to_group(3) == "medium"
    assert fst_to_group(4) == "medium"
    assert fst_to_group(5) == "dark"
    assert fst_to_group(6) == "dark"
    assert fst_to_group(0) == "unknown"
    assert fst_to_group(-1) == "unknown"
    assert fst_to_group(None) == "unknown"


def test_rgb_to_cielab_properties():
    """Verify conversion of pure white and pure black."""
    white = np.ones((10, 10, 3), dtype=np.float32)
    black = np.zeros((10, 10, 3), dtype=np.float32)

    lab_white = rgb_to_cielab(white)
    lab_black = rgb_to_cielab(black)

    # L for white should be near 100, L for black near 0
    assert pytest.approx(lab_white[0, 0, 0], abs=1.0) == 100.0
    assert pytest.approx(lab_black[0, 0, 0], abs=1.0) == 0.0


def test_extract_skin_ita_synthetic(tmp_path):
    """Verify ITA extraction on an in-memory synthetic image file."""
    # Create a light skin patch
    arr = np.full((100, 100, 3), (240, 210, 190), dtype=np.uint8)
    img_path = tmp_path / "test_light_skin.jpg"
    Image.fromarray(arr).save(img_path)

    ita, group = extract_skin_ita_from_image(img_path)
    assert isinstance(ita, float)
    assert not math.isnan(ita)
    assert group in {"light", "medium", "dark"}
