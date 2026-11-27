"""
Unit tests for VISHWAS Phase D: Knowledge Base, BM25, and Zero-Leakage Case Retrieval
tests/test_rag.py
"""

import json
from pathlib import Path
import time
import numpy as np
import pytest

from src.rag.bm25 import BM25KnowledgeRetriever, get_bm25_retriever
from src.rag.case_retriever import CaseRetriever, get_case_retriever
from src.preprocess.milk10k_dataset import MILK10K_CLASSES


def test_kb_schema_and_size():
    """Verify data/knowledge_base/kb_chunks.jsonl has 50-100 chunks and valid schema."""
    kb_path = Path(__file__).resolve().parent.parent / "data" / "knowledge_base" / "kb_chunks.jsonl"
    assert kb_path.exists(), "Knowledge base JSONL does not exist"

    chunks = []
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    assert 50 <= len(chunks) <= 100, f"Expected 50-100 chunks, got {len(chunks)}"

    valid_sections = {
        "typical_look", "dermoscopic_features", "differential",
        "risk_factors", "red_flags", "darker_skin_nuance", "next_steps"
    }

    seen_classes = set()
    for c in chunks:
        assert "id" in c and isinstance(c["id"], str)
        assert "class_code" in c and c["class_code"] in MILK10K_CLASSES
        assert "section" in c and c["section"] in valid_sections
        assert "text" in c and len(c["text"]) > 25
        assert "source" in c and len(c["source"]) > 5
        assert "tone_groups_relevant" in c and isinstance(c["tone_groups_relevant"], list)
        seen_classes.add(c["class_code"])

    assert len(seen_classes) == 11, f"Expected chunks for all 11 classes, found {len(seen_classes)}"


def test_bm25_latency_under_50ms():
    """Verify BM25 knowledge retrieval executes in <50ms per query."""
    retriever = get_bm25_retriever()
    test_queries = [
        "arborizing telangiectasia and blue gray ovoid nests in basal cell carcinoma",
        "atypical network and blue white veil in melanoma",
        "acral lentiginous parallel ridge pattern on darker skin",
        "milia-like cysts and comedo openings in seborrheic keratosis",
        "dimple sign positive dermal nodule central scar dermatofibroma",
    ]

    for q in test_queries:
        t0 = time.perf_counter()
        results = retriever.retrieve(q, top_k=3)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        assert latency_ms < 50.0, f"BM25 latency {latency_ms:.2f} ms exceeded 50ms limit"
        assert len(results) > 0, f"Expected non-empty results for query '{q}'"


def test_bm25_recall_at_k():
    """Verify Recall@3 on a comprehensive set of 20 realistic clinical queries exceeds 85%."""
    retriever = get_bm25_retriever()

    # 20 ground-truth queries: (query_string, expected_target_class)
    eval_benchmark = [
        ("atypical pigment network and irregular dots in melanoma", "MEL"),
        ("blue white veil and regression structures in high risk melanoma", "MEL"),
        ("arborizing tree-like telangiectasias in basal cell carcinoma", "BCC"),
        ("blue-gray ovoid nests and spoke-wheel structures in BCC", "BCC"),
        ("pigmented basal cell carcinoma with shiny white strands", "BCC"),
        ("strawberry pattern and keratotic plugs in actinic keratosis", "AKIEC"),
        ("Bowen's disease with clustered glomerular vessels and scale", "AKIEC"),
        ("milia-like cysts and comedo-like openings in seborrheic keratosis", "BKL"),
        ("cerebriform gyri and sulci waxy stuck-on keratosis", "BKL"),
        ("Dermatosis Papulosa Nigra on darker facial skin", "BKL"),
        ("dimple sign central white patch and delicate peripheral network", "DF"),
        ("firm lower extremity nodule button-like dermatofibroma", "DF"),
        ("red and purple vascular lagoons separated by fibrous septa", "VASC"),
        ("cherry angioma with thrombosed blood lacunae", "VASC"),
        ("volcano-like central keratin crater in keratoacanthoma", "SCCKA"),
        ("squamous cell carcinoma white circles and hairpin vessels", "SCCKA"),
        ("Wickham striae and violaceous flat-topped papules in lichen planus", "INF"),
        ("psoriasis with uniformly distributed dotted vessels on red background", "INF"),
        ("symmetric uniform brown mole with fading peripheral network", "NV"),
        ("acral lentiginous melanoma on the sole of foot with parallel ridge pattern", "MEL"),
    ]

    assert len(eval_benchmark) == 20

    hits = 0
    for query_str, expected_class in eval_benchmark:
        top_chunks = retriever.retrieve(query_str, top_k=3)
        retrieved_classes = [c["class_code"] for c in top_chunks]
        if expected_class in retrieved_classes:
            hits += 1

    recall_at_3 = hits / len(eval_benchmark)
    assert recall_at_3 >= 0.85, f"Recall@3 {recall_at_3*100:.1f}% below 85% requirement"


def test_case_retriever_zero_leakage():
    """
    CRITICAL Zero-Leakage Test:
    1. Query lesion_id is NEVER in retrieved set.
    2. Query patient_id is NEVER in retrieved set.
    3. Eval fold cases are NEVER in retrieved set (training folds only).
    4. Retrieved cases strictly belong to the specified tone group.
    """
    case_ret = get_case_retriever()
    assert case_ret.df is not None and len(case_ret.df) > 0

    # Test across multiple distinct folds and tone groups
    for eval_fold in [0, 1, 2, 3, 4]:
        # Pick a sample case from the evaluation fold
        fold_cases = case_ret.df[case_ret.df["fold"] == eval_fold]
        assert len(fold_cases) > 0
        sample_row = fold_cases.iloc[0]

        q_lid = str(sample_row["lesion_id"])
        q_pid = str(sample_row["patient_id"])
        q_tg = str(sample_row["tone_group"])
        prob_cols = [f"prob_{c}" for c in MILK10K_CLASSES]
        q_probs = sample_row[prob_cols].values

        # Perform retrieval strictly excluding eval_fold and query IDs
        retrieved = case_ret.retrieve_similar_cases(
            query_probs=q_probs,
            query_tone_group=q_tg,
            query_lesion_id=q_lid,
            query_patient_id=q_pid,
            eval_fold=eval_fold,
            top_k=3,
        )

        assert len(retrieved) > 0, f"Expected retrieved cases for fold {eval_fold}"

        for c in retrieved:
            # Rule 1: Never retrieve query lesion itself
            assert c["lesion_id"] != q_lid, f"LEAKAGE: Query lesion {q_lid} was retrieved"

            # Rule 2: Never retrieve same patient
            assert c["patient_id"] != q_pid, f"LEAKAGE: Query patient {q_pid} was retrieved"

            # Rule 3: Never retrieve from evaluation fold
            assert c["fold"] != eval_fold, f"LEAKAGE: Case from eval fold {eval_fold} was retrieved"

            # Rule 4: Tone group matches query tone group
            assert c["tone_group"] == q_tg, f"DISPARITY: Case tone group {c['tone_group']} does not match query {q_tg}"

            # Rule 5: Cosine similarity is valid
            assert 0.0 <= c["cosine_similarity"] <= 1.0


def test_query_builder_expansion():
    """Verify query builder expands top classes with metadata, tone group, and ABCD traits."""
    retriever = get_bm25_retriever()
    expanded = retriever.build_query(
        top_classes=["MEL", "BCC", "NV"],
        patient_metadata={"site": "foot", "tone_group": "G3"},
        observed_features={"is_asymmetric": True, "has_blue_white_veil": True},
    )

    assert "MEL" in expanded
    assert "foot" in expanded
    assert "darker skin" in expanded or "hyperpigmentation" in expanded
    assert "asymmetry" in expanded
    assert "blue-white veil" in expanded
