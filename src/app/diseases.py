"""
VISHWAS Disease Knowledge & Explanations System
Loads single source of truth from data/diseases.json to provide trilingual condition
explainers, severity chips, differential diagnosis comparisons, and a full Skin Conditions Guide.
"""

from __future__ import annotations
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import streamlit as st


def get_vishwas_root() -> Path:
    """Finds the vishwas project root directory robustly."""
    p = Path(__file__).resolve().parent.parent.parent
    if (p / "data" / "diseases.json").exists():
        return p
    cwd = Path.cwd()
    if (cwd / "data" / "diseases.json").exists():
        return cwd
    if (cwd / "vishwas" / "data" / "diseases.json").exists():
        return cwd / "vishwas"
    return p


@st.cache_data
def load_diseases_data_with_status() -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Loads disease information from data/diseases.json.
    Returns (data_dict, error_message). Never swallows errors silently.
    """
    root = get_vishwas_root()
    path = root / "data" / "diseases.json"

    if not path.exists():
        err_msg = f"Could not find data/diseases.json at '{path}'"
        logging.error(err_msg)
        return {}, err_msg

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data, None
    except Exception as exc:
        err_msg = f"Failed to parse data/diseases.json at '{path}': {exc}"
        logging.error(err_msg)
        return {}, err_msg


def load_diseases_data() -> Dict[str, Any]:
    """Convenience helper returning the disease dictionary."""
    data, _ = load_diseases_data_with_status()
    return data


def get_disease_info(code: str) -> Dict[str, Any]:
    """Retrieves full details for a disease code."""
    data = load_diseases_data()
    return data.get(code, {
        "code": code,
        "severity": "benign",
        "en": {"name": code, "desc": "Condition details unavailable.", "typical_look": "", "what_to_do": ""},
        "hi": {"name": code, "desc": "विवरण उपलब्ध नहीं है।", "typical_look": "", "what_to_do": ""},
        "kn": {"name": code, "desc": "ವಿವರ ಲಭ್ಯವಿಲ್ಲ.", "typical_look": "", "what_to_do": ""},
    })


def get_localized(entry: Dict[str, Any], field: str, lang: str = "en") -> str:
    """Helper to retrieve localized field like name, desc, typical_look in en, hi, kn."""
    if not entry or not isinstance(entry, dict):
        return ""
    # 1. Nested structure: entry[lang][field]
    if lang in entry and isinstance(entry[lang], dict) and field in entry[lang]:
        return str(entry[lang][field])
    # 2. Flat structure: entry[f"{field}_{lang}"]
    flat_k = f"{field}_{lang}"
    if flat_k in entry and entry[flat_k]:
        return str(entry[flat_k])
    # 3. Fallback to English nested
    if "en" in entry and isinstance(entry["en"], dict) and field in entry["en"]:
        return str(entry["en"][field])
    # 4. Fallback to English flat
    fallback_flat = f"{field}_en"
    if fallback_flat in entry and entry[fallback_flat]:
        return str(entry[fallback_flat])
    return str(entry.get(field, ""))


def render_severity_chip(severity: str, lang: str = "en") -> str:
    """Generates an HTML severity chip badge with appropriate contrast colors."""
    sev = str(severity).lower()
    classes = {
        "cancer": "severity-chip severity-cancer",
        "pre-cancerous": "severity-chip severity-precancer",
        "benign": "severity-chip severity-benign",
        "not-skin-cancer": "severity-chip severity-nontumour"
    }
    badge_class = classes.get(sev, "severity-chip severity-benign")
    
    labels = {
        "en": {
            "cancer": "⚠️ Cancer (Malignant)",
            "pre-cancerous": "⚠️ Pre-Cancerous",
            "benign": "✅ Benign (Harmless)",
            "not-skin-cancer": "🩺 Non-Tumour / Inflammatory"
        },
        "hi": {
            "cancer": "⚠️ कैंसर (घातक)",
            "pre-cancerous": "⚠️ पूर्व-कैंसर",
            "benign": "✅ हानिरहित (सौम्य)",
            "not-skin-cancer": "🩺 गैर-ट्यूमर / सूजन"
        },
        "kn": {
            "cancer": "⚠️ ಕ್ಯಾನ್ಸರ್",
            "pre-cancerous": "⚠️ ಕ್ಯಾನ್ಸರ್-ಪೂರ್ವ",
            "benign": "✅ ನಿರುಪದ್ರವಿ",
            "not-skin-cancer": "🩺 ಗಡ್ಡೆಯಲ್ಲ / ಉರಿಯೂತ"
        }
    }
    label = labels.get(lang, labels["en"]).get(sev, sev.capitalize())
    return f'<span class="{badge_class}">{label}</span>'


def render_disease_card(code: str, lang: str = "en") -> None:
    """Renders the 'What is this condition?' expandable explainer card."""
    info = get_disease_info(code)
    plain_name = get_localized(info, "name", lang) or code
    desc = get_localized(info, "desc", lang) or get_localized(info, "description", lang)
    look = get_localized(info, "typical_look", lang)
    action = get_localized(info, "what_to_do", lang)
    learn = get_localized(info, "learn_more", lang)
    chip_html = render_severity_chip(info.get("severity", "benign"), lang)

    title_label = {
        "en": f"ℹ️ What is {plain_name} ({code})?",
        "hi": f"ℹ️ {plain_name} ({code}) क्या है?",
        "kn": f"ℹ️ {plain_name} ({code}) ಎಂದರೇನು?"
    }.get(lang, f"ℹ️ What is {plain_name}?")

    with st.expander(title_label, expanded=True):
        st.markdown(f"{chip_html} **{plain_name}** (`{code}`)", unsafe_allow_html=True)
        if desc:
            st.write(desc)
        if look:
            st.markdown(f"**🔍 Typical Appearance:** {look}")
        if action:
            st.markdown(f"**🩺 Recommended Next Step:** {action}")
        if learn:
            with st.expander("📚 Medical Context & Notes", expanded=False):
                st.caption(learn)


def render_top3_differentials(top3: List[Dict[str, Any]], lang: str = "en") -> None:
    """Renders plain-language explainers for the top-3 conditions considered."""
    if not top3:
        return

    expander_title = {
        "en": "🔍 Understanding the Top 3 Possibilities",
        "hi": "🔍 शीर्ष 3 संभावनाओं का अर्थ समझें",
        "kn": "🔍 ಅಗ್ರ 3 ಸಾಧ್ಯತೆಗಳ ವಿವರ ತಿಳಿಯಿರಿ"
    }.get(lang, "🔍 Understanding the Top 3 Possibilities")

    with st.expander(expander_title, expanded=False):
        for idx, item in enumerate(top3, 1):
            code = item.get("class", item.get("code", ""))
            prob = item.get("prob", 0.0)
            info = get_disease_info(code)
            name = get_localized(info, "name", lang) or code
            chip_html = render_severity_chip(info.get("severity", "benign"), lang)
            desc = get_localized(info, "desc", lang) or get_localized(info, "description", lang)
            
            st.markdown(
                f"**#{idx}. {name} (`{code}`) — {prob*100:.1f}% probability**<br>"
                f"{chip_html} {desc}",
                unsafe_allow_html=True
            )
            st.markdown("---")


def render_skin_conditions_guide(lang: str = "en") -> None:
    """Renders the comprehensive Skin Conditions Guide browser with all 11 conditions and thumbnails."""
    data, err_msg = load_diseases_data_with_status()
    if err_msg:
        st.error(f"⚠️ {err_msg}")
        return
    if not data:
        st.error("⚠️ No condition records found in data/diseases.json.")
        return

    header_titles = {
        "en": "📚 Comprehensive Skin Conditions Guide (11 Conditions)",
        "hi": "📚 संपूर्ण त्वचा रोग मार्गदर्शिका (11 स्थितियां)",
        "kn": "📚 ಸಮಗ್ರ ಚರ್ಮ ರೋಗಗಳ ಕೈಪಿಡಿ (11 ಸ್ಥಿತಿಗಳು)"
    }
    st.markdown(f"### {header_titles.get(lang, header_titles['en'])}")

    desc_texts = {
        "en": "Explore all 11 dermatological conditions evaluated by the VISHWAS model, categorized by clinical urgency. Each card includes typical clinical presentations, severity levels, and example images.",
        "hi": "VISHWAS मॉडल द्वारा मूल्यांकित सभी 11 त्वचा स्थितियों की जानकारी, चिकित्सीय गंभीरता के अनुसार वर्गीकृत। प्रत्येक स्थिति का विशिष्ट रूप, गंभीरता स्तर और उदाहरण चित्र देखें।",
        "kn": "ವಿಶ್ವಾಸ ಮಾದರಿಯು ಮೌಲ್ಯಮಾಪನ ಮಾಡುವ ಎಲ್ಲಾ 11 ಚರ್ಮದ ಸ್ಥಿತಿಗಳ ಮಾಹಿತಿ, ವೈದ್ಯಕೀಯ ತುರ್ತುಸ್ಥಿತಿಯ ಆಧಾರದ ಮೇಲೆ ವಿಂಗಡಿಸಲಾಗಿದೆ."
    }
    st.write(desc_texts.get(lang, desc_texts["en"]))

    # Filter by category
    filter_options = {
        "All": "All Conditions (11)" if lang == "en" else ("सभी स्थितियां (11)" if lang == "hi" else "ಎಲ್ಲಾ ಸ್ಥಿತಿಗಳು (11)"),
        "cancer": "Cancers (Melanoma, Basal Cell, Squamous Cell, Rare)" if lang == "en" else ("कैंसर (मेलेनोमा, बेसल सेल, आदि)" if lang == "hi" else "ಕ್ಯಾನ್ಸರ್ (ಮೆಲನೋಮ, ಬೇಸಲ್ ಸೆಲ್)"),
        "pre-cancerous": "Pre-Cancerous (Actinic Keratosis)" if lang == "en" else ("पूर्व-कैंसर (एक्टिनिक केराटोसिस)" if lang == "hi" else "ಕ್ಯಾನ್ಸರ್-ಪೂರ್ವ"),
        "benign": "Benign / Harmless (Moles, Seborrheic Keratoses, Cysts)" if lang == "en" else ("हानिरहित (सामान्य तिल, मस्से)" if lang == "hi" else "ನಿರುಪದ್ರವಿ (ಸಾಮಾನ್ಯ ಮಚ್ಚೆಗಳು)"),
        "not-skin-cancer": "Non-Tumour / Inflammatory (Eczema, Psoriasis, Vascular)" if lang == "en" else ("गैर-ट्यूमर / सूजन संबंधी" if lang == "hi" else "ಗಡ್ಡೆಯಲ್ಲ / ಉರಿಯೂತ")
    }
    chosen_cat = st.segmented_control(
        "Filter by Urgency Level" if lang == "en" else ("गंभीरता के अनुसार फ़िल्टर करें" if lang == "hi" else "ತುರ್ತು ಮಟ್ಟದ ಪ್ರಕಾರ ಫಿಲ್ಟರ್ ಮಾಡಿ"),
        options=list(filter_options.keys()),
        default="All",
        format_func=lambda k: filter_options[k]
    )

    # Search bar
    search_prompt = "🔍 Search conditions by name or code" if lang == "en" else ("🔍 नाम या कोड से खोजें" if lang == "hi" else "🔍 ಹೆಸರು ಅಥವಾ ಕೋಡ್ ಮೂಲಕ ಹುಡುಕಿ")
    search_query = st.text_input(search_prompt, "").strip().lower()

    # Filter conditions
    filtered_items = []
    for code, info in data.items():
        cat = info.get("severity", "benign")
        if chosen_cat != "All" and cat != chosen_cat:
            continue
        name = get_localized(info, "name", lang).lower()
        if search_query and (search_query not in code.lower() and search_query not in name):
            continue
        filtered_items.append((code, info))

    st.caption(f"Showing {len(filtered_items)} of {len(data)} skin conditions")

    root = get_vishwas_root()

    # Display as grid
    for code, info in filtered_items:
        name = get_localized(info, "name", lang) or code
        desc = get_localized(info, "desc", lang) or get_localized(info, "description", lang)
        look = get_localized(info, "typical_look", lang)
        action = get_localized(info, "what_to_do", lang)
        chip_html = render_severity_chip(info.get("severity", "benign"), lang)
        
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                thumb_path = root / "assets" / "examples" / f"{code}_clinical.jpg"
                if thumb_path.exists():
                    st.image(str(thumb_path), width=220)
                else:
                    st.caption("No thumbnail")
            with col2:
                st.markdown(f"#### {name} (`{code}`)")
                st.markdown(f"{chip_html}", unsafe_allow_html=True)
                if desc:
                    st.write(desc)
                if look:
                    st.caption(f"**Typical look:** {look}")
                if action:
                    st.caption(f"**Recommended action:** {action}")
            st.markdown("---")
