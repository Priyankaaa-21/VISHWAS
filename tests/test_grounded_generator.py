"""
Unit tests for VISHWAS Phase E: Grounded Explanation Generation & Clinical Abstention
tests/test_grounded_generator.py
"""

import json
from pathlib import Path
import pytest

from src.rag.generator import GroundedExplanationGenerator, get_grounded_generator, should_abstain
from src.preprocess.milk10k_dataset import MILK10K_CLASSES


def test_grounded_generator_confident_case():
    """Verify confident prediction generates all 4 slots with real citations from knowledge base."""
    generator = get_grounded_generator()
    top3 = [
        {"code": "MEL", "name": "Melanoma", "prob": 0.72},
        {"code": "BCC", "name": "Basal Cell Carcinoma", "prob": 0.15},
        {"code": "NV", "name": "Melanocytic Nevus", "prob": 0.08},
    ]
    conformal_info = {"set_size": 2, "is_uncertain": False, "prediction_set": ["MEL", "BCC"]}
    observed_features = {
        "is_asymmetric": True,
        "is_irregular_border": True,
        "is_multicolor": True,
        "has_blue_white_veil": True,
    }
    patient_metadata = {"site": "back", "tone_group": "G3"}

    res = generator.generate(
        pred_class="MEL",
        conf=0.72,
        top3=top3,
        conformal_info=conformal_info,
        observed_features=observed_features,
        patient_metadata=patient_metadata,
        lang="en",
    )

    assert not res["is_abstained"], "Confident prediction should not abstain"
    assert res["top_consideration_sentence"] is not None
    assert "[source:" in res["top_consideration_sentence"]
    assert res["differential_sentence"] is not None
    assert "[source:" in res["differential_sentence"]
    assert res["darker_skin_sentence"] is not None
    assert "[source:" in res["darker_skin_sentence"]
    assert res["next_step_sentence"] is not None
    assert "[source:" in res["next_step_sentence"]
    assert len(res["citations"]) >= 3
    assert len(res["retrieved_chunk_ids"]) >= 3


def test_grounded_generator_citations_match_kb():
    """Verify every citation in the generated text originates from an existing chunk in kb_chunks.jsonl."""
    kb_path = Path(__file__).resolve().parent.parent / "data" / "knowledge_base" / "kb_chunks.jsonl"
    assert kb_path.exists()

    known_sources = set()
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                known_sources.add(json.loads(line)["source"])

    generator = get_grounded_generator()
    res = generator.generate(
        pred_class="BCC",
        conf=0.68,
        top3=[{"code": "BCC", "name": "Basal Cell", "prob": 0.68}],
        conformal_info={"set_size": 1, "is_uncertain": False},
        lang="en",
    )

    for cit in res["citations"]:
        assert cit in known_sources, f"Hallucinated or unknown citation: {cit}"


def test_abstention_triggers_on_wide_conformal_set():
    """Verify abstention triggers when conformal set has >= 4 classes."""
    is_abs, reason = should_abstain(conf=0.55, conformal_set_size=4)
    assert is_abs is True
    assert "wide" in reason.lower()

    generator = get_grounded_generator()
    res = generator.generate(
        pred_class="BCC",
        conf=0.55,
        top3=[{"code": "BCC", "name": "BCC", "prob": 0.55}],
        conformal_info={"set_size": 4, "is_uncertain": True},
        lang="en",
    )

    assert res["is_abstained"] is True
    assert "cannot make a confident assessment" in res["full_text"]
    assert res["top_consideration_sentence"] is None
    assert len(res["citations"]) >= 1


def test_abstention_triggers_on_low_confidence():
    """Verify abstention triggers when top diagnostic confidence is < 35%."""
    is_abs, reason = should_abstain(conf=0.28, conformal_set_size=2)
    assert is_abs is True
    assert "low" in reason.lower()

    generator = get_grounded_generator()
    res = generator.generate(
        pred_class="NV",
        conf=0.28,
        top3=[{"code": "NV", "name": "Nevus", "prob": 0.28}],
        conformal_info={"set_size": 2, "is_uncertain": True},
        lang="en",
    )

    assert res["is_abstained"] is True
    assert "cannot make a confident assessment" in res["full_text"]


def test_abstention_triggers_on_severe_quality_warning():
    """Verify abstention triggers when image quality warnings are present."""
    is_abs, reason = should_abstain(conf=0.80, conformal_set_size=1, quality_warnings=["Image too dark"])
    assert is_abs is True
    assert "quality" in reason.lower()


def test_trilingual_grounded_generation():
    """Verify grounded explanation and abstention render accurately in Hindi and Kannada."""
    generator = get_grounded_generator()
    
    # 1. Hindi confident
    res_hi = generator.generate(
        pred_class="MEL",
        conf=0.65,
        top3=[{"code": "MEL", "name": "Melanoma", "prob": 0.65}],
        conformal_info={"set_size": 2, "is_uncertain": False},
        lang="hi",
    )
    assert "शीर्ष विचार" in res_hi["full_text"] or "लक्षणों" in res_hi["full_text"]
    assert "[स्रोत:" in res_hi["full_text"]

    # 2. Kannada confident
    res_kn = generator.generate(
        pred_class="MEL",
        conf=0.65,
        top3=[{"code": "MEL", "name": "Melanoma", "prob": 0.65}],
        conformal_info={"set_size": 2, "is_uncertain": False},
        lang="kn",
    )
    assert "ಪ್ರಮುಖ ಪರಿಗಣನೆ" in res_kn["full_text"] or "ಲಕ್ಷಣಗಳ" in res_kn["full_text"]
    assert "[ಮೂಲ:" in res_kn["full_text"]
