"""
Unit tests for VISHWAS V2 Modernized Features:
1. Diseases Knowledge System (data/diseases.json)
2. Theme System & WCAG AA Contrast Variables (src/app/theme.py)
3. Offline Precomputed Summaries (results/*.json)
4. Curated Example Cases & Images (assets/examples/)
5. Interactive Plotly Charts (src/app/charts.py)
6. Example Loading & Placeholder Fallback (src/app/examples.py)
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from PIL import Image
import pytest

from src.app.theme import get_theme_css, get_plotly_theme, LIGHT_VARS, DARK_VARS
from src.app.diseases import load_diseases_data, get_disease_info, get_localized, render_severity_chip
from src.app.examples import load_curated_examples, get_example_by_id, load_image_safe
from src.app.charts import (
    plot_reliability_chart,
    plot_confusion_recall_chart,
    plot_confusion_detail_chart,
    plot_fairness_bars,
    plot_top3_horizontal,
)

ALL_11_CLASSES = [
    "AKIEC", "BCC", "BEN_OTH", "BKL", "DF",
    "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"
]


# ==============================================================================
# 1. Diseases Knowledge System
# ==============================================================================

def test_diseases_json_contains_all_11_classes():
    data = load_diseases_data()
    assert len(data) == 11
    for cls in ALL_11_CLASSES:
        assert cls in data, f"Missing class {cls} in data/diseases.json"
        entry = data[cls]
        for lang in ["en", "hi", "kn"]:
            name = get_localized(entry, "name", lang)
            desc = get_localized(entry, "desc", lang) or get_localized(entry, "description", lang)
            assert len(name) > 0, f"Missing {lang} name for {cls}"
            assert len(desc) > 0, f"Missing {lang} description for {cls}"
        assert entry["severity"] in ["cancer", "pre-cancerous", "benign", "not-skin-cancer"]


def test_render_severity_chip():
    chip_cancer = render_severity_chip("cancer", lang="en")
    assert "severity-cancer" in chip_cancer
    assert "Cancer" in chip_cancer

    chip_benign_hi = render_severity_chip("benign", lang="hi")
    assert "severity-benign" in chip_benign_hi
    assert "हानिरहित" in chip_benign_hi


# ==============================================================================
# 2. Theme System
# ==============================================================================

def test_theme_css_generation():
    css_light = get_theme_css("Light")
    assert "--bg: #ffffff" in css_light
    assert "--text: #0f172a" in css_light
    assert "[data-testid=\"stHeader\"]" in css_light
    assert "[data-testid=\"stFileUploaderDropzone\"]" in css_light
    assert "div[role=\"dialog\"]" in css_light

    css_dark = get_theme_css("Dark")
    assert "--bg: #0b1120" in css_dark
    assert "--text: #f8fafc" in css_dark

    css_auto = get_theme_css("Auto")
    assert "@media (prefers-color-scheme: dark)" in css_auto


def test_get_plotly_theme():
    p_light = get_plotly_theme("Light")
    assert p_light["paper_bgcolor"] == "rgba(0,0,0,0)"
    assert p_light["plot_bgcolor"] == "rgba(0,0,0,0)"
    assert p_light["font"]["color"] == "#0f172a"

    p_dark = get_plotly_theme("Dark")
    assert p_dark["font"]["color"] == "#f8fafc"


# ==============================================================================
# 3. Offline Precomputed Summaries
# ==============================================================================

def test_offline_summaries_exist_and_valid():
    summary_files = [
        "results/fairness_summary.json",
        "results/reliability_summary.json",
        "results/confusion_summary.json",
        "results/dataset_summary.json",
    ]
    for sf in summary_files:
        assert os.path.exists(sf), f"Missing summary file {sf}"
        with open(sf) as f:
            data = json.load(f)
            assert isinstance(data, dict)
            assert len(data) > 0

    # Check fairness summary specifics
    with open("results/fairness_summary.json") as f:
        f_data = json.load(f)
        assert "overall" in f_data
        assert "by_tone_group" in f_data
        assert "G1" in f_data["by_tone_group"]
        assert "G2" in f_data["by_tone_group"]
        assert "G3" in f_data["by_tone_group"]
        assert "disparity" in f_data

    # Check reliability summary specifics
    with open("results/reliability_summary.json") as f:
        r_data = json.load(f)
        assert "overall_ece" in r_data
        assert "overall_rating" in r_data
        assert len(r_data.get("bins", [])) == 5

    # Check confusion summary specifics
    with open("results/confusion_summary.json") as f:
        c_data = json.load(f)
        assert len(c_data["by_disease"]) == 11
        for cls in ALL_11_CLASSES:
            assert cls in c_data["by_disease"]


# ==============================================================================
# 4. Curated Example Cases
# ==============================================================================

def test_curated_examples_and_images():
    examples = load_curated_examples()
    assert len(examples) == 11
    
    # Check that each example has an existing, non-empty image file
    for ex in examples:
        assert "diagnosis" in ex
        assert "clinical_image" in ex
        clin_path = ex["clinical_image"]
        assert os.path.exists(clin_path), f"Missing image {clin_path}"
        assert os.path.getsize(clin_path) > 0
        img = Image.open(clin_path)
        assert img.size[0] > 0 and img.size[1] > 0

    # Test get_example_by_id
    mel_ex = get_example_by_id("mel")
    assert mel_ex is not None
    assert mel_ex["diagnosis"] == "MEL"


def test_load_image_safe_fallback():
    # Valid image loads
    examples = load_curated_examples()
    if examples:
        img = load_image_safe(examples[0]["clinical_image"])
        assert isinstance(img, Image.Image)

    # Missing image returns placeholder rather than crashing
    placeholder = load_image_safe("non_existent_path_12345.jpg")
    assert isinstance(placeholder, Image.Image)
    assert placeholder.size == (300, 300)


# ==============================================================================
# 5. Interactive Plotly Charts
# ==============================================================================

def test_plotly_charts_render_successfully():
    with open("results/reliability_summary.json") as f:
        rel_data = json.load(f)
    fig_rel = plot_reliability_chart(rel_data, tone_group="All", theme_mode="Dark")
    assert len(fig_rel.data) >= 2

    with open("results/confusion_summary.json") as f:
        conf_data = json.load(f)
    fig_recall = plot_confusion_recall_chart(conf_data, theme_mode="Light")
    assert len(fig_recall.data) == 1
    assert len(fig_recall.data[0].y) == 11

    fig_detail = plot_confusion_detail_chart(conf_data, "MEL", theme_mode="Dark")
    assert fig_detail is not None

    with open("results/fairness_summary.json") as f:
        fair_data = json.load(f)
    fig_fair = plot_fairness_bars(fair_data, theme_mode="Auto")
    assert len(fig_fair.data) == 2

    top3_sample = [
        {"name": "Melanoma", "class": "MEL", "prob": 0.65},
        {"name": "Dysplastic Nevus", "class": "NV", "prob": 0.20},
        {"name": "Seborrheic Keratosis", "class": "BKL", "prob": 0.10},
    ]
    fig_top3 = plot_top3_horizontal(top3_sample, theme_mode="Dark")
    assert len(fig_top3.data) == 1
