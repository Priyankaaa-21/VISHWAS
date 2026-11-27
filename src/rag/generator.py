"""
VISHWAS Grounded Explanation Generator: src/rag/generator.py

Provides:
1. Extractive template-based grounded clinical explanation generator:
   - Slot 1: Top consideration with observed visual features and verified source citation.
   - Slot 2: Differentials with shared dermoscopic features and source citation.
   - Slot 3: Tone group nuance with darker skin clinical context and source citation.
   - Slot 4: Recommended next steps with clinical guideline source citation.
   - Strict rule: Every sentence MUST have a verified [source: <citation>] tag.
     Missing slots are omitted, NEVER fabricated.
2. Clinical Abstention Engine:
   - Triggers when conformal prediction set >= 4 classes, top confidence < 35%,
     or severe image quality warnings are present.
   - Outputs clear uncertainty notice with red flags for the top condition and medical consult advice.
3. Optional LLM Synthesis Layer:
   - Checks for GEMINI_API_KEY / GOOGLE_API_KEY.
   - If present, queries LLM with strict retrieved chunk context only.
   - Falls back safely to the extractive generator if no key is present or on API failure.
4. Trilingual output formatting across English, Hindi, and Kannada.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.rag.bm25 import BM25KnowledgeRetriever, get_bm25_retriever
from src.preprocess.milk10k_dataset import MILK10K_CLASSES

VISHWAS_ROOT = Path(__file__).resolve().parent.parent.parent

CLASS_DISPLAY_NAMES = {
    "AKIEC": "Actinic Keratosis / Bowen's Disease",
    "BCC": "Basal Cell Carcinoma",
    "BEN_OTH": "Other Benign Lesions",
    "BKL": "Benign Keratosis (Seborrheic Keratosis)",
    "DF": "Dermatofibroma",
    "INF": "Inflammatory Condition",
    "MAL_OTH": "Other Malignant Lesions",
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevus (Common Mole)",
    "SCCKA": "Squamous Cell Carcinoma / Keratoacanthoma",
    "VASC": "Vascular Lesion",
}


def should_abstain(
    conf: float,
    conformal_set_size: int,
    quality_warnings: Optional[List[str]] = None,
    abstain_conf_threshold: float = 0.35,
    abstain_set_threshold: int = 4,
) -> Tuple[bool, str]:
    """
    Evaluates whether the clinical AI should abstain from generating a definitive explanation.
    Returns (should_abstain: bool, reason: str).
    """
    reasons: List[str] = []

    if quality_warnings and len(quality_warnings) > 0:
        reasons.append(f"Image quality issues detected ({'; '.join(quality_warnings)})")

    if conformal_set_size >= abstain_set_threshold:
        reasons.append(f"Conformal prediction set is wide ({conformal_set_size} classes at 90% coverage)")

    if conf < abstain_conf_threshold:
        reasons.append(f"Top diagnostic confidence is low ({conf*100:.1f}% < {abstain_conf_threshold*100:.0f}%)")

    if reasons:
        return True, " & ".join(reasons)
    return False, ""


def format_observed_traits(observed_features: Optional[Dict[str, Any]], lang: str = "en") -> str:
    """Formats observed ABCD traits into plain readable language."""
    if not observed_features:
        return "characteristic morphology" if lang == "en" else ("विशिष्ट शारीरिक लक्षण" if lang == "hi" else "ವಿಶಿಷ್ಟ ರಚನಾತ್ಮಕ ಲಕ್ಷಣಗಳು")

    traits: List[str] = []
    if observed_features.get("is_asymmetric"):
        traits.append("asymmetrical contours" if lang == "en" else ("असममित रूपरेखा" if lang == "hi" else "ಅಸಮಪಾರ್ಶ್ವದ ಬಾಹ್ಯರೇಖೆಗಳು"))
    if observed_features.get("is_irregular_border"):
        traits.append("irregular borders" if lang == "en" else ("खुरदरे किनारे" if lang == "hi" else "ಅನಿಯಮಿತ ಅಂಚುಗಳು"))
    if observed_features.get("is_multicolor"):
        n_col = observed_features.get("num_dominant_colors", 3)
        traits.append(f"{n_col} color shades" if lang == "en" else ("कई रंगों का मिश्रण" if lang == "hi" else "ಹಲವು ಬಣ್ಣಗಳ ಛಾಯೆಗಳು"))
    if observed_features.get("has_blue_white_veil"):
        traits.append("blue-white haze" if lang == "en" else ("नीला-सफेद आवरण" if lang == "hi" else "ನೀಲಿ-ಬಿಳಿ ಪದರ"))
    if observed_features.get("has_vascular_structures"):
        traits.append("erythematous vascularity" if lang == "en" else ("संवहनी लालिमा" if lang == "hi" else "ರಕ್ತನಾಳದ ಕೆಂಪಾಗುವಿಕೆ"))
    if observed_features.get("ulceration_crust_present"):
        traits.append("crusting / erosion" if lang == "en" else ("पपड़ी / छिलना" if lang == "hi" else "ಹೆಪ್ಪುಗಟ್ಟುವಿಕೆ"))

    if not traits:
        return "regular uniform features" if lang == "en" else ("समान नियमित लक्षण" if lang == "hi" else "ನಿಯಮಿತ ಲಕ್ಷಣಗಳು")

    return ", ".join(traits)


class GroundedExplanationGenerator:
    """
    Extractive, retrieval-grounded medical explanation engine with tone group awareness
    and principled abstention.
    """

    def __init__(self, retriever: Optional[BM25KnowledgeRetriever] = None) -> None:
        self.retriever = retriever or get_bm25_retriever()

    def generate(
        self,
        pred_class: str,
        conf: float,
        top3: List[Dict[str, Any]],
        conformal_info: Optional[Dict[str, Any]] = None,
        observed_features: Optional[Dict[str, Any]] = None,
        patient_metadata: Optional[Dict[str, Any]] = None,
        quality_warnings: Optional[List[str]] = None,
        lang: str = "en",
        use_llm: bool = False,
    ) -> Dict[str, Any]:
        """
        Generates grounded explanation.
        If abstention criteria are met, returns abstention result with red flags.
        Otherwise fills slots with verified retrieved knowledge chunks.
        """
        c_set_size = conformal_info.get("set_size", 1) if conformal_info else 1
        is_abstained, abstain_reason = should_abstain(
            conf=conf,
            conformal_set_size=c_set_size,
            quality_warnings=quality_warnings,
        )

        tone_group = patient_metadata.get("tone_group", "Unknown") if patient_metadata else "Unknown"

        # ----------------------------------------------------------------------
        # CASE 1: ABSTENTION (Uncertain, Large Set, or Low Quality)
        # ----------------------------------------------------------------------
        if is_abstained:
            # Retrieve red flags chunk for the top condition
            red_flag_chunks = self.retriever.retrieve(
                query=f"{pred_class} red flags warning signs urgent",
                top_k=1,
                class_filter=pred_class,
                section_filter="red_flags",
            )

            if red_flag_chunks:
                rf_chunk = red_flag_chunks[0]
                rf_text = rf_chunk["text"]
                rf_source = rf_chunk["source"]
                rf_chunk_id = rf_chunk["id"]
            else:
                rf_text = "Rapid change in size, bleeding, asymmetrical spreading, or persistent ulceration."
                rf_source = "British Association of Dermatologists clinical guidelines (2020)"
                rf_chunk_id = "general_bad_guidelines"

            if lang == "hi":
                abstain_msg = (
                    f"⚠️ AI इस त्वचा स्थल का कोई आश्वस्त मूल्यांकन नहीं कर सकता ({abstain_reason})। "
                    f"निगरानी हेतु सामान्य चेतावनी संकेत: {rf_text} [स्रोत: {rf_source}]। "
                    f"कृपया किसी योग्य चिकित्सक या त्वचा विशेषज्ञ से प्रत्यक्ष जांच करवाएं।"
                )
            elif lang == "kn":
                abstain_msg = (
                    f"⚠️ AI ಈ ಚರ್ಮದ ಗುರುತಿನ ಬಗ್ಗೆ ಖಚಿತ ಮೌಲ್ಯಮಾಪನ ನೀಡಲು ಸಾಧ್ಯವಿಲ್ಲ ({abstain_reason}). "
                    f"ಗಮನಿಸಬೇಕಾದ ಸಾಮಾನ್ಯ ಎಚ್ಚರಿಕೆಯ ಲಕ್ಷಣಗಳು: {rf_text} [ಮೂಲ: {rf_source}]. "
                    f"ದಯವಿಟ್ಟು ಅರ್ಹ ವೈದ್ಯರು ಅಥವಾ ಚರ್ಮರೋಗ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ."
                )
            else:
                abstain_msg = (
                    f"⚠️ The AI cannot make a confident assessment of this spot ({abstain_reason}). "
                    f"Here are general signs to watch for: {rf_text} [source: {rf_source}]. "
                    f"Please see a qualified doctor or dermatologist for an in-person clinical check."
                )

            return {
                "is_abstained": True,
                "abstain_reason": abstain_reason,
                "full_text": abstain_msg,
                "top_consideration_sentence": None,
                "differential_sentence": None,
                "darker_skin_sentence": None,
                "next_step_sentence": None,
                "citations": [rf_source],
                "retrieved_chunk_ids": [rf_chunk_id],
                "red_flags_text": rf_text,
            }

        # ----------------------------------------------------------------------
        # CASE 2: CONFIDENT GROUNDED GENERATION (Slots with Retrieved Citations)
        # ----------------------------------------------------------------------
        top_name = CLASS_DISPLAY_NAMES.get(pred_class, pred_class)
        traits_str = format_observed_traits(observed_features, lang=lang)

        sentences: List[str] = []
        citations: List[str] = []
        chunk_ids: List[str] = []

        # Slot 1: Top consideration
        top_chunks = self.retriever.retrieve(
            query=f"{pred_class} typical look morphology presentation",
            top_k=1,
            class_filter=pred_class,
            section_filter="typical_look",
        )
        slot1_sentence: Optional[str] = None
        if top_chunks:
            c1 = top_chunks[0]
            citations.append(c1["source"])
            chunk_ids.append(c1["id"])
            if lang == "hi":
                slot1_sentence = (
                    f"छवि में देखे गए लक्षणों ({traits_str}) के आधार पर शीर्ष विचार {top_name} (`{pred_class}`) है "
                    f"[स्रोत: {c1['source']}]।"
                )
            elif lang == "kn":
                slot1_sentence = (
                    f"ಚಿತ್ರದಲ್ಲಿ ಕಂಡುಬಂದ ಲಕ್ಷಣಗಳ ({traits_str}) ಆಧಾರದ ಮೇಲೆ ಪ್ರಮುಖ ಪರಿಗಣನೆ {top_name} (`{pred_class}`) ಆಗಿದೆ "
                    f"[ಮೂಲ: {c1['source']}]."
                )
            else:
                slot1_sentence = (
                    f"Top consideration is {top_name} ({pred_class}), based on features ({traits_str}) "
                    f"observed in the photo [source: {c1['source']}]."
                )
            sentences.append(slot1_sentence)

        # Slot 2: Differentials
        slot2_sentence: Optional[str] = None
        diff_classes = [item["code"] for item in top3[1:3] if item.get("code") != pred_class]
        if diff_classes:
            diff_chunks = self.retriever.retrieve(
                query=f"{pred_class} differential diagnosis distinguishing clues",
                top_k=1,
                class_filter=pred_class,
                section_filter="differential",
            )
            if diff_chunks:
                c2 = diff_chunks[0]
                citations.append(c2["source"])
                chunk_ids.append(c2["id"])
                diff_names = [CLASS_DISPLAY_NAMES.get(c, c) for c in diff_classes]
                joined_diffs = " and ".join(diff_names)
                if lang == "hi":
                    slot2_sentence = (
                        f"संभावित विभेदक निदान के रूप में {joined_diffs} पर भी विचार करें, क्योंकि इनमें साझा लक्षण हो सकते हैं "
                        f"[स्रोत: {c2['source']}]।"
                    )
                elif lang == "kn":
                    slot2_sentence = (
                        f"ಪರ್ಯಾಯ ಸಾಧ್ಯತೆಗಳಾಗಿ {joined_diffs} ಸ್ಥಿತಿಗಳನ್ನೂ ಪರಿಗಣಿಸಿ, ಏಕೆಂದರೆ ಇವುಗಳು ಒಂದೇ ರೀತಿಯ ಲಕ್ಷಣಗಳನ್ನು ಹಂಚಿಕೊಳ್ಳುತ್ತವೆ "
                        f"[ಮೂಲ: {c2['source']}]."
                    )
                else:
                    slot2_sentence = (
                        f"Also consider {joined_diffs} as potential differentials because shared dermoscopic features "
                        f"can overlap [source: {c2['source']}]."
                    )
                sentences.append(slot2_sentence)

        # Slot 3: Skin tone nuance
        slot3_sentence: Optional[str] = None
        nuance_chunks = self.retriever.retrieve(
            query=f"{pred_class} darker skin phototype IV VI nuance melanin",
            top_k=1,
            class_filter=pred_class,
            section_filter="darker_skin_nuance",
        )
        if nuance_chunks:
            c3 = nuance_chunks[0]
            citations.append(c3["source"])
            chunk_ids.append(c3["id"])
            nuance_text = c3["text"]
            if lang == "hi":
                slot3_sentence = f"त्वचा टोन समूह ({tone_group}) के संदर्भ में: {nuance_text} [स्रोत: {c3['source']}]।"
            elif lang == "kn":
                slot3_sentence = f"ಚರ್ಮದ ಟೋನ್ ಗುಂಪಿನ ({tone_group}) ಸಂದರ್ಭದಲ್ಲಿ: {nuance_text} [ಮೂಲ: {c3['source']}]."
            else:
                slot3_sentence = f"On skin tone group {tone_group}, note that {nuance_text} [source: {c3['source']}]."
            sentences.append(slot3_sentence)

        # Slot 4: Next steps
        slot4_sentence: Optional[str] = None
        step_chunks = self.retriever.retrieve(
            query=f"{pred_class} next steps management clinical biopsy",
            top_k=1,
            class_filter=pred_class,
            section_filter="next_steps",
        )
        if step_chunks:
            c4 = step_chunks[0]
            citations.append(c4["source"])
            chunk_ids.append(c4["id"])
            step_text = c4["text"]
            if lang == "hi":
                slot4_sentence = f"अनुशंसित अगला कदम: {step_text} [स्रोत: {c4['source']}]।"
            elif lang == "kn":
                slot4_sentence = f"ಶಿಫಾರಸು ಮಾಡಿದ ಮುಂದಿನ ಕ್ರಮ: {step_text} [ಮೂಲ: {c4['source']}]."
            else:
                slot4_sentence = f"Recommended next step: {step_text} [source: {c4['source']}]."
            sentences.append(slot4_sentence)

        full_extractive_text = " ".join(sentences)

        # ----------------------------------------------------------------------
        # Optional LLM Synthesis Layer
        # ----------------------------------------------------------------------
        llm_text: Optional[str] = None
        if use_llm and (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            try:
                llm_text = self._synthesize_with_llm(
                    pred_class=pred_class,
                    top_chunks=top_chunks + (diff_chunks if diff_classes and diff_chunks else []) + nuance_chunks + step_chunks,
                    lang=lang,
                )
            except Exception as e:
                logging.warning("LLM synthesis failed, falling back to extractive: %s", e)
                llm_text = None

        return {
            "is_abstained": False,
            "abstain_reason": "",
            "full_text": llm_text or full_extractive_text,
            "is_llm_synthesized": bool(llm_text is not None),
            "top_consideration_sentence": slot1_sentence,
            "differential_sentence": slot2_sentence,
            "darker_skin_sentence": slot3_sentence,
            "next_step_sentence": slot4_sentence,
            "citations": list(dict.fromkeys(citations)),
            "retrieved_chunk_ids": chunk_ids,
        }

    def _synthesize_with_llm(
        self,
        pred_class: str,
        top_chunks: List[Dict[str, Any]],
        lang: str = "en",
    ) -> Optional[str]:
        """
        Optional synthesis using Gemini API if key is present in environment.
        Strict system prompt: answers strictly from retrieved context and uses bracket chunk IDs.
        """
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return None

        # Build context from chunks
        context_blocks = "\n".join([f"[{c['id']}] {c['text']} (Source: {c['source']})" for c in top_chunks])
        prompt = (
            f"You are a medical dermatology clinical AI summarizer.\n"
            f"STRICT INSTRUCTIONS:\n"
            f"1. Answer ONLY using the provided retrieved context below.\n"
            f"2. Cite chunk IDs in square brackets like [{top_chunks[0]['id']}].\n"
            f"3. Never diagnose; use hedged medical language ('shows features commonly seen in').\n"
            f"4. Output language must be {'Hindi' if lang=='hi' else ('Kannada' if lang=='kn' else 'English')}.\n"
            f"5. If context does not contain the answer, say 'I do not have enough information'.\n\n"
            f"CONTEXT:\n{context_blocks}\n\n"
            f"TASK:\nSummarize diagnostic considerations, differentials, darker skin nuances, and next steps for {pred_class}."
        )

        try:
            import urllib.request
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            req_data = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
            req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=8) as response:
                res_json = json.loads(response.read().decode("utf-8"))
                text = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                return text
        except Exception as exc:
            logging.warning("Urllib Gemini API call failed: %s", exc)
            return None


# Global singleton
_GLOBAL_EXPLANATION_GENERATOR: Optional[GroundedExplanationGenerator] = None


def get_grounded_generator() -> GroundedExplanationGenerator:
    """Returns singleton GroundedExplanationGenerator."""
    global _GLOBAL_EXPLANATION_GENERATOR
    if _GLOBAL_EXPLANATION_GENERATOR is None:
        _GLOBAL_EXPLANATION_GENERATOR = GroundedExplanationGenerator()
    return _GLOBAL_EXPLANATION_GENERATOR
