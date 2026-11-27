"""
VISHWAS Phase F: Evaluation and Equity Checks Verification Tests
tests/test_phase_f.py
"""

import json
from pathlib import Path
import pytest
import pandas as pd
from streamlit.testing.v1 import AppTest

VISHWAS_ROOT = Path(__file__).resolve().parent.parent


def test_rag_eval_results_structure():
    """Verify results/rag_eval.json exists and satisfies Phase F benchmark constraints."""
    eval_file = VISHWAS_ROOT / "results" / "rag_eval.json"
    assert eval_file.exists(), f"Missing {eval_file}"

    with open(eval_file, "r") as f:
        data = json.load(f)

    # 1. Retrieval Recall
    r_eval = data.get("retrieval_evaluation", {})
    assert r_eval.get("macro_recall@1", 0) >= 0.70, "Macro Recall@1 below 70%"
    assert r_eval.get("macro_recall@3", 0) >= 0.85, "Macro Recall@3 below 85%"
    assert len(r_eval.get("per_class", {})) == 11, "Must benchmark all 11 classes"

    # 2. Citation Precision & Faithfulness
    f_eval = data.get("faithfulness_evaluation", {})
    assert f_eval.get("citation_precision", 0) == 1.0, "Citation precision must be 100% (real citations only)"
    assert f_eval.get("faithfulness_rate", 0) == 1.0, "Faithfulness rate must be 100% for extractive generator"

    # 3. Tone Group Equity Checks
    eq_eval = data.get("conformal_and_equity_evaluation", {})
    ptg = eq_eval.get("per_tone_group", {})
    assert set(ptg.keys()) == {"G1", "G2", "G3"}, "Must audit G1, G2, and G3"

    for g, g_data in ptg.items():
        assert g_data.get("conformal_coverage", 0) >= 0.88, f"Coverage for group {g} below 88%"
        assert g_data.get("retrieval_hit_rate", 0) >= 0.95, f"Retrieval hit rate for group {g} below 95%"
        assert g_data.get("skin_of_colour_note_share", 0) >= 0.95, f"SoC note share for group {g} below 95%"

    # Inter-group disparity gaps must be <= 5% (0.05)
    gaps = eq_eval.get("inter_group_gaps", {})
    assert gaps.get("conformal_coverage", 1.0) <= 0.05, "Coverage disparity exceeds 5% threshold"
    assert gaps.get("abstention_rate", 1.0) <= 0.05, "Abstention disparity exceeds 5% threshold"


def test_dermatologist_review_sample_50():
    """Verify 50-case review sample exported in CSV, JSON, and PDF formats."""
    json_path = VISHWAS_ROOT / "reports" / "review_sample_50.json"
    csv_path = VISHWAS_ROOT / "reports" / "review_sample_50.csv"
    pdf_path = VISHWAS_ROOT / "reports" / "review_sample_50.pdf"

    assert json_path.exists(), "review_sample_50.json missing"
    assert csv_path.exists(), "review_sample_50.csv missing"
    assert pdf_path.exists(), "review_sample_50.pdf missing"

    # Verify JSON structure
    with open(json_path, "r") as f:
        cases = json.load(f)
    assert len(cases) == 50, f"Expected 50 cases, got {len(cases)}"

    # Check class diversity
    classes_present = {c["ground_truth_diagnosis"] for c in cases}
    assert len(classes_present) == 11, f"Expected all 11 classes, got {len(classes_present)}"

    # Check tone group diversity
    tones_present = {c["tone_group"] for c in cases}
    assert tones_present == {"G1", "G2", "G3"}, f"Expected all 3 tone groups, got {tones_present}"

    # Verify CSV structure
    df = pd.read_csv(csv_path)
    assert len(df) == 50
    assert "rubric_overall_grade" in df.columns
    assert "rubric_safety_hedging" in df.columns
    assert "rubric_retrieval_grounding" in df.columns

    # Verify PDF size
    assert pdf_path.stat().st_size > 10000, "PDF review booklet appears too small"


def test_phase_f_app_fairness_and_about_tabs():
    """Verify Fairness tab and About tab render Phase F equity metrics in Streamlit."""
    app_path = VISHWAS_ROOT / "src" / "app" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception, f"App exception: {at.exception}"

    md_texts = [m.value for m in at.markdown]
    # Verify Tab 2 (Fairness) contains Phase F equity audit section
    has_equity_audit = any("Retrieval-Grounded & Multimodal Equity Audit" in m for m in md_texts)
    assert has_equity_audit, "Tab 2 missing Retrieval-Grounded & Multimodal Equity Audit section"

    # Verify Tab 5 (About) contains RAG faithfulness audit section
    has_rag_audit = any("Retrieval-Grounded Knowledge Base & Faithfulness Audit" in m for m in md_texts)
    assert has_rag_audit, "Tab 5 missing RAG Knowledge Base & Faithfulness Audit section"
