"""
Vishwas (विश्वास) - User-Friendly & Multimodal Dermatological AI Platform
Streamlit Application: src/app/app.py

Modes:
- Simple Mode (Default): Guided 3-step flow (Add photos -> Patient details -> Plain-language result),
  trilingual (English, Hindi, Kannada), photo quality checks, side-by-side previews, and PDF report.
- Expert Mode: Technical research interface with dataset gallery, model architecture comparison,
  fold selection, raw probability charts, calibration analysis, and subgroup fairness auditing.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torchvision.transforms as transforms
import yaml

# Set page config as first Streamlit call
st.set_page_config(
    page_title="VISHWAS (विश्वास) - Skin AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Resolve workspace directories
APP_DIR = Path(__file__).resolve().parent
VISHWAS_ROOT = APP_DIR.parent.parent
if str(VISHWAS_ROOT) not in sys.path:
    sys.path.insert(0, str(VISHWAS_ROOT))

# Safe imports for modular helper packages
try:
    from src.models.classifier import build_model
    from src.preprocess.milk10k_dataset import (
        HEADLINE_CLASSES,
        MILK10K_CLASSES,
        MILK10K_CLASS_TO_IDX,
        MILK10K_IDX_TO_CLASS,
        SITE_CATEGORIES,
        SEX_CATEGORIES,
        TabularScaler,
    )
    from src.app.strings import STRINGS, CONDITION_DESCRIPTIONS, t
    from src.app.helpers import (
        DEFAULT_APP_CONFIG,
        MALIGNANT_OR_PREMALIGNANT_CLASSES,
        check_image_quality,
        get_confidence_label,
        get_next_steps,
        compute_input_hash,
        generate_pdf_report,
        compute_triage_verdict,
        load_oof_prediction_bins,
        get_evidence_line,
        create_top3_horizontal_chart,
    )
    from src.app.theme import inject_theme, get_plotly_theme
    from src.app.examples import (
        get_vishwas_root,
        load_curated_examples,
        load_curated_examples_with_status,
        get_example_by_id,
        load_image_safe,
        render_image_preview_card,
    )
    from src.app.diseases import (
        load_diseases_data,
        load_diseases_data_with_status,
        get_disease_info,
        get_localized,
        render_disease_card,
        render_top3_differentials,
        render_skin_conditions_guide,
    )
    from src.app.charts import (
        plot_reliability_chart,
        plot_confusion_recall_chart,
        plot_confusion_detail_chart,
        plot_fairness_bars,
        plot_top3_horizontal,
    )
except ImportError:
    from vishwas.src.models.classifier import build_model
    from vishwas.src.preprocess.milk10k_dataset import (
        HEADLINE_CLASSES,
        MILK10K_CLASSES,
        MILK10K_CLASS_TO_IDX,
        MILK10K_IDX_TO_CLASS,
        SITE_CATEGORIES,
        SEX_CATEGORIES,
        TabularScaler,
    )
    from vishwas.src.app.strings import STRINGS, CONDITION_DESCRIPTIONS, t
    from vishwas.src.app.helpers import (
        DEFAULT_APP_CONFIG,
        MALIGNANT_OR_PREMALIGNANT_CLASSES,
        check_image_quality,
        get_confidence_label,
        get_next_steps,
        compute_input_hash,
        generate_pdf_report,
        compute_triage_verdict,
        load_oof_prediction_bins,
        get_evidence_line,
        create_top3_horizontal_chart,
    )
    from vishwas.src.app.theme import inject_theme, get_plotly_theme
    from vishwas.src.app.examples import (
        get_vishwas_root,
        load_curated_examples,
        load_curated_examples_with_status,
        get_example_by_id,
        load_image_safe,
        render_image_preview_card,
    )
    from vishwas.src.app.diseases import (
        load_diseases_data,
        load_diseases_data_with_status,
        get_disease_info,
        get_localized,
        render_disease_card,
        render_top3_differentials,
        render_skin_conditions_guide,
    )
    from vishwas.src.app.charts import (
        plot_reliability_chart,
        plot_confusion_recall_chart,
        plot_confusion_detail_chart,
        plot_fairness_bars,
        plot_top3_horizontal,
    )

# Load App Configuration
@st.cache_data
def load_app_config() -> Dict[str, Any]:
    root = get_vishwas_root()
    cfg_path = root / "configs" / "config.yaml"
    cfg: Dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            full_yaml = yaml.safe_load(f) or {}
            cfg = full_yaml.get("app", {})
    return {**DEFAULT_APP_CONFIG, **cfg}


# Load Precomputed JSON Summaries
@st.cache_data
def load_summary_json(name: str) -> Dict[str, Any]:
    root = get_vishwas_root()
    path = root / "results" / f"{name}_summary.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logging.error("Failed to parse summary %s: %s", path, exc)
    return {}


# Human-Readable Class Metadata
CLASS_INFO = {
    "AKIEC": {"name": "Actinic Keratosis / Bowen's Disease", "type": "Pre-cancerous", "badge": "⚠️ Pre-Cancerous"},
    "BCC": {"name": "Basal Cell Carcinoma", "type": "Malignant", "badge": "🚨 Malignant (Common)"},
    "BEN_OTH": {"name": "Other Benign Lesions", "type": "Benign", "badge": "✅ Benign"},
    "BKL": {"name": "Benign Keratotic Lesions", "type": "Benign", "badge": "✅ Benign"},
    "DF": {"name": "Dermatofibroma", "type": "Benign", "badge": "✅ Benign"},
    "INF": {"name": "Inflammatory / Infectious Conditions", "type": "Non-tumour", "badge": "🩺 Non-Tumour"},
    "MAL_OTH": {"name": "Other Rare Malignant Lesions", "type": "Malignant", "badge": "🚨 Malignant (Rare)"},
    "MEL": {"name": "Melanoma", "type": "Malignant", "badge": "🚨 Malignant (High Risk)"},
    "NV": {"name": "Melanocytic Nevus (Mole)", "type": "Benign", "badge": "✅ Benign (Mole)"},
    "SCCKA": {"name": "Squamous Cell Carcinoma / Keratoacanthoma", "type": "Malignant", "badge": "🚨 Malignant"},
    "VASC": {"name": "Vascular Lesions", "type": "Non-tumour", "badge": "🩺 Vascular"},
}

TONE_GROUP_INFO = {
    "G1": {"label": "G1 = {0, 1, 2} (Light Tones)", "n": 617},
    "G2": {"label": "G2 = {3} (Medium Tones)", "n": 3174},
    "G3": {"label": "G3 = {4, 5} (Darker Tones)", "n": 1449},
}

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def get_tone_group_for_class(tone_group: str, skin_tone_class: Any) -> Tuple[str, str]:
    """Resolves tone group code (G1, G2, G3) and human-friendly label."""
    tg = str(tone_group).strip().upper()
    if tg in ("UNKNOWN", "", "NONE"):
        try:
            c = int(skin_tone_class)
            if c in (0, 1, 2):
                tg = "G1"
            elif c == 3:
                tg = "G2"
            elif c in (4, 5):
                tg = "G3"
        except (ValueError, TypeError):
            pass
    info = TONE_GROUP_INFO.get(tg)
    if info:
        return tg, info["label"]
    return tg, f"{tg} (Skin Tone Class {skin_tone_class})"


# Model and Checkpoint Cache
@st.cache_resource
def load_checkpoint_model(experiment_name: str, fold: int) -> Optional[torch.nn.Module]:
    """Loads a specific trained checkpoint fold onto MPS or CPU."""
    root = get_vishwas_root()
    ckpt_path = root / "checkpoints" / f"{experiment_name}_fold_{fold}.pt"
    if not ckpt_path.exists():
        logging.warning("Checkpoint %s does not exist.", ckpt_path)
        return None

    try:
        model_type = "fusion" if experiment_name.startswith("d_") else "image_only"
        model = build_model(
            model_type=model_type,
            backbone="resnet18",
            num_classes=11,
            num_tabular_features=18 if model_type == "fusion" else 0,
            pretrained=False,
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        return model
    except Exception as exc:
        logging.error("Failed loading checkpoint %s: %s", ckpt_path, exc)
        return None


def encode_tabular_vector(
    age: Optional[float] = 50.0,
    sex: str = "unknown",
    site: str = "unknown",
) -> torch.Tensor:
    """Encodes tabular features matching the training TabularScaler specification."""
    age_val = 50.0 if age is None else age
    age_norm = (age_val - 50.0) / 20.0
    age_missing = 1.0 if age is None else 0.0

    sex_one_hot = [0.0] * len(SEX_CATEGORIES)
    if sex in SEX_CATEGORIES:
        sex_one_hot[SEX_CATEGORIES.index(sex)] = 1.0
    else:
        sex_one_hot[SEX_CATEGORIES.index("unknown")] = 1.0

    site_one_hot = [0.0] * len(SITE_CATEGORIES)
    if site in SITE_CATEGORIES:
        site_one_hot[SITE_CATEGORIES.index(site)] = 1.0
    else:
        site_one_hot[SITE_CATEGORIES.index("unknown")] = 1.0

    feat_list = [age_norm, age_missing] + sex_one_hot + site_one_hot
    return torch.tensor(feat_list, dtype=torch.float32).unsqueeze(0).to(device)


def predict_single_model(
    model: torch.nn.Module,
    clin_tensor: torch.Tensor,
    dermo_tensor: torch.Tensor,
    tab_tensor: torch.Tensor,
    is_fusion: bool,
) -> np.ndarray:
    """Runs single model forward pass, outputting 11-class probabilities."""
    with torch.no_grad():
        if is_fusion:
            logits = model(clin_tensor, dermo_tensor, tab_tensor)
        else:
            logits = model(dermo_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def predict_ensemble(
    experiment_name: str,
    clin_tensor: torch.Tensor,
    dermo_tensor: torch.Tensor,
    tab_tensor: torch.Tensor,
    folds: List[int] = [0, 1, 2, 3, 4],
) -> np.ndarray:
    """Computes mean ensemble prediction across available checkpoints."""
    is_fusion = experiment_name.startswith("d_")
    accum_probs = np.zeros(11, dtype=np.float32)
    models_evaluated = 0

    for f in folds:
        m = load_checkpoint_model(experiment_name, f)
        if m is not None:
            p = predict_single_model(m, clin_tensor, dermo_tensor, tab_tensor, is_fusion)
            accum_probs += p
            models_evaluated += 1

    if models_evaluated == 0:
        return np.ones(11, dtype=np.float32) / 11.0

    return accum_probs / float(models_evaluated)


# ==============================================================================
# RESULT MODAL DIALOG (ST.DIALOG)
# ==============================================================================

if hasattr(st, "dialog"):
    @st.dialog("Diagnostic Assessment Result", width="large")
    def show_result_dialog(res_data: Dict[str, Any], lang_code: str, theme_name: str, oof_bins: Optional[List[Dict[str, Any]]]):
        pred_c = res_data["pred_class"]
        conf_val = res_data["conf"]
        c_meta = CONDITION_DESCRIPTIONS.get(pred_c, {})
        display_name = c_meta.get(f"{lang_code}_name", c_meta.get("en_name", pred_c))
        display_desc = c_meta.get(f"{lang_code}_desc", c_meta.get("en_desc", ""))
        triage = res_data.get("triage", {})

        app_cfg = load_app_config()

        # 1. Large Triage Banner
        strip_class = f"result-strip-{triage.get('level', 'grey')}"
        st.markdown(f"""
        <div class="result-strip {strip_class}" style="padding:16px 20px; margin-bottom:14px;">
            <div style="display:flex; align-items:center; gap:12px;">
                <span style="font-size:2.2rem;">{triage.get('icon', '🩺')}</span>
                <div>
                    <h3 style="margin:0; font-size:1.35rem; font-weight:800;">{triage.get('title', '')}</h3>
                    <div style="font-size:0.95rem; opacity:0.95; margin-top:3px; line-height:1.4;">{triage.get('text', '')}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Predicted Condition & Confidence
        title_tag = t("most_likely_uncertain", lang_code) if triage.get("is_uncertain") else t("most_likely", lang_code)
        st.markdown(f"#### {title_tag}: **{display_name}** (`{pred_c}`)")
        st.caption(display_desc)

        conf_word = get_confidence_label(conf_val, app_cfg, lang_code)
        st.markdown(f"**{t('confidence_level', lang_code)}**: {conf_word} ({conf_val*100:.1f}%)")
        st.progress(float(min(max(conf_val, 0.0), 1.0)))

        # 3. Evidence Line
        ev_line = get_evidence_line(conf_val, oof_bins, lang_code)
        if ev_line:
            st.caption(f"📈 *{ev_line}*")

        # 4. Next Step Sentence
        advice = get_next_steps(pred_c, res_data.get("is_high_risk", False), lang_code)
        st.info(f"💡 {advice}")

        # 5. Top-3 Horizontal Ranked Bars Chart (Plotly)
        st.markdown(f"##### 📊 {t('top3_title', lang_code)}")
        top3_fig = plot_top3_horizontal(res_data.get("top3", []), theme_mode=theme_name, lang=lang_code)
        st.plotly_chart(top3_fig, use_container_width=True)

        # 6. Condition Explainer Card
        render_disease_card(pred_c, lang_code)

        # 7. Action Buttons
        p_col1, p_col2 = st.columns([1, 1])
        with p_col1:
            pdf_bytes = generate_pdf_report(res_data, res_data.get("c_img"), res_data.get("d_img"), lang_code)
            st.download_button(
                label=t("download_pdf_btn", lang_code),
                data=pdf_bytes,
                file_name=f"VISHWAS_Result_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="popup_pdf_btn",
            )
        with p_col2:
            if st.button(t("popup_close_btn", lang_code), use_container_width=True, key="popup_close_btn"):
                st.rerun()
else:
    show_result_dialog = None


# ==============================================================================
# MAIN STREAMLIT APP APPLICATION ENTRY
# ==============================================================================

def main():
    app_cfg = load_app_config()
    oof_bins = load_oof_prediction_bins()

    # Session State Initialization
    if "app_mode" not in st.session_state:
        st.session_state["app_mode"] = "Simple"
    if "lang" not in st.session_state:
        st.session_state["lang"] = app_cfg.get("default_language", "en")
    if "theme" not in st.session_state:
        st.session_state["theme"] = app_cfg.get("default_theme", "Auto")
    if "cache" not in st.session_state:
        st.session_state["cache"] = {}
    if "popup_shown_for" not in st.session_state:
        st.session_state["popup_shown_for"] = None
    if "open_popup_now" not in st.session_state:
        st.session_state["open_popup_now"] = False
    if "dismiss_banner" not in st.session_state:
        st.session_state["dismiss_banner"] = False
    if "active_c_img" not in st.session_state:
        st.session_state["active_c_img"] = None
    if "active_d_img" not in st.session_state:
        st.session_state["active_d_img"] = None
    if "active_age" not in st.session_state:
        st.session_state["active_age"] = 50
    if "active_dont_know_age" not in st.session_state:
        st.session_state["active_dont_know_age"] = False
    if "active_sex" not in st.session_state:
        st.session_state["active_sex"] = "unknown"
    if "active_site" not in st.session_state:
        st.session_state["active_site"] = "unknown"
    if "active_gt" not in st.session_state:
        st.session_state["active_gt"] = None
    if "active_tone" not in st.session_state:
        st.session_state["active_tone"] = "Unknown"
    if "example_name" not in st.session_state:
        st.session_state["example_name"] = None

    lang = st.session_state["lang"]
    theme = st.session_state["theme"]

    # Inject Central Theme CSS
    inject_theme(theme)

    # --------------------------------------------------------------------------
    # SIDEBAR CONFIGURATION & GLOSSARY
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            '<h2 class="sidebar-brand">🩺 <span class="hero-brand">VISHWAS</span> <span class="hero-devanagari">(विश्वास)</span></h2>',
            unsafe_allow_html=True,
        )

        # Interface Mode Toggle
        mode_options = [t("mode_simple", lang), t("mode_expert", lang)]
        current_mode_idx = 0 if st.session_state["app_mode"] == "Simple" else 1

        selected_mode_str = st.radio(
            t("mode_selector", lang),
            mode_options,
            index=current_mode_idx,
            help="Choose 'Simple' for a clean, guided check, or 'Expert' for technical model benchmarking and research tools.",
        )
        st.session_state["app_mode"] = "Simple" if selected_mode_str == mode_options[0] else "Expert"

        # Language Selector (English, Hindi, Kannada)
        lang_options = ["English", "हिंदी (Hindi)", "ಕನ್ನಡ (Kannada)"]
        lang_idx = 0 if lang == "en" else (1 if lang == "hi" else 2)
        lang_choice = st.selectbox(
            t("language_selector", lang),
            lang_options,
            index=lang_idx,
            help="Select preferred language for Simple Mode guidance and condition explanations.",
        )
        new_lang = "en" if "English" in lang_choice else ("hi" if "Hindi" in lang_choice else "kn")
        if new_lang != st.session_state["lang"]:
            st.session_state["lang"] = new_lang
            st.rerun()

        # Theme Selector (Auto / Light / Dark)
        theme_options = [t("theme_auto", lang), t("theme_light", lang), t("theme_dark", lang)]
        current_theme_idx = 0 if theme == "Auto" else (1 if theme == "Light" else 2)
        selected_theme_str = st.radio(
            t("theme_selector", lang),
            theme_options,
            index=current_theme_idx,
            horizontal=True,
            help="Switch between Auto (follows device), Light, or Dark theme.",
        )
        new_theme = "Auto" if selected_theme_str == theme_options[0] else ("Light" if selected_theme_str == theme_options[1] else "Dark")
        if new_theme != st.session_state["theme"]:
            st.session_state["theme"] = new_theme
            st.rerun()

        st.markdown("---")

        # Privacy Assurance
        st.info(t("privacy_badge", lang))

        # Start Over Button
        if st.button(t("start_over_btn", lang), use_container_width=True, type="secondary"):
            st.session_state["active_c_img"] = None
            st.session_state["active_d_img"] = None
            st.session_state["active_gt"] = None
            st.session_state["active_tone"] = "Unknown"
            st.session_state["example_name"] = None
            st.session_state["active_dont_know_age"] = False
            st.session_state["active_age"] = 50
            st.session_state["active_sex"] = "unknown"
            st.session_state["active_site"] = "unknown"
            st.session_state["popup_shown_for"] = None
            st.session_state["open_popup_now"] = False
            st.rerun()

        # Plain-Language Glossary Expander
        with st.expander(t("glossary_title", lang), expanded=False):
            st.markdown(t("glossary_lesion", lang))
            st.markdown(t("glossary_dermoscopy", lang))
            st.markdown(t("glossary_melanoma", lang))
            st.markdown(t("glossary_bcc", lang))
            st.markdown(t("glossary_tone_group", lang))
            st.markdown(t("glossary_f1", lang))
            st.markdown(t("glossary_calibration", lang))

    # Prominent Hero Title with Devanagari & Kannada Support
    if lang == "kn":
        st.markdown(
            '<div class="main-header-hero">'
            '<span class="hero-brand">ವಿಶ್ವಾಸ</span> '
            '<span class="hero-devanagari">(VISHWAS)</span>'
            '<span class="hero-colon">:</span> '
            '<span class="hero-tagline">ಚರ್ಮ ಆರೋಗ್ಯ ತಪಾಸಣೆ</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    elif lang == "hi":
        st.markdown(
            '<div class="main-header-hero">'
            '<span class="hero-brand">VISHWAS</span> '
            '<span class="hero-devanagari">(विश्वास)</span>'
            '<span class="hero-colon">:</span> '
            '<span class="hero-tagline">त्वचा स्वास्थ्य जांच</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="main-header-hero">'
            '<span class="hero-brand">VISHWAS</span> '
            '<span class="hero-devanagari">(विश्वास)</span>'
            '<span class="hero-colon">:</span> '
            '<span class="hero-tagline">Skin AI Check</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Subtitle
    if st.session_state["app_mode"] == "Simple":
        st.markdown(f'<div class="main-header-sub">{t("app_subtitle_simple", lang)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="main-header-sub">{t("app_subtitle_expert", lang)}</div>', unsafe_allow_html=True)

    # Medical Disclaimer Banner
    st.warning(t("medical_disclaimer", lang))

    # Optional local metadata loading if present
    root = get_vishwas_root()
    folds_path = root / "data" / "milk10k_folds.csv"
    meta_path = root / "data" / "MILK10k_Training_Metadata.csv"
    folds_df = pd.read_csv(folds_path) if folds_path.exists() else pd.DataFrame()
    meta_df = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()

    # --------------------------------------------------------------------------
    # TABS: RENAMED WITH FRIENDLY LABELS
    # --------------------------------------------------------------------------
    tabs = st.tabs([
        t("tab_check", lang),
        t("tab_fairness", lang),
        t("tab_reliability", lang),
        t("tab_explorer", lang),
        t("tab_about", lang),
        t("tab_conditions_guide", lang),
    ])

    # ==========================================================================
    # TAB 1: CHECK A SKIN SPOT (SIMPLE & EXPERT MODES)
    # ==========================================================================
    with tabs[0]:
        # ----------------------------------------------------------------------
        # SIMPLE MODE: GUIDED FLOW (PINNED STRIP -> RESULT -> PHOTOS -> DETAILS)
        # ----------------------------------------------------------------------
        if st.session_state["app_mode"] == "Simple":
            c_img_active = st.session_state.get("active_c_img")
            d_img_active = st.session_state.get("active_d_img")
            age_val = None if st.session_state.get("active_dont_know_age") else float(st.session_state.get("active_age", 50))
            sex_val = str(st.session_state.get("active_sex", "unknown"))
            site_val = str(st.session_state.get("active_site", "unknown"))

            # Fallback rule: if single photo given, use for both branches
            img_for_clin = c_img_active if c_img_active is not None else d_img_active
            img_for_dermo = d_img_active if d_img_active is not None else c_img_active

            # ------------------------------------------------------------------
            # AUTOMATIC INFERENCE ON INPUT CHANGE
            # ------------------------------------------------------------------
            if c_img_active is not None or d_img_active is not None:
                input_hash = compute_input_hash(img_for_clin, img_for_dermo, age_val, sex_val, site_val)

                if input_hash in st.session_state["cache"]:
                    res = st.session_state["cache"][input_hash]
                else:
                    with st.status(t("progress_looking", lang), expanded=True) as status_box:
                        status_box.update(label=t("progress_models", lang))
                        c_tensor = IMAGE_TRANSFORMS(img_for_clin.convert("RGB")).unsqueeze(0).to(device)
                        d_tensor = IMAGE_TRANSFORMS(img_for_dermo.convert("RGB")).unsqueeze(0).to(device)
                        tab_tensor = encode_tabular_vector(age_val, sex_val, site_val)

                        status_box.update(label=t("progress_preparing", lang))
                        probs = predict_ensemble("d_cw_128", c_tensor, d_tensor, tab_tensor, folds=[0, 1, 2, 3, 4])
                        top_idx = int(np.argmax(probs))
                        pred_class = MILK10K_IDX_TO_CLASS[top_idx]
                        conf = float(probs[top_idx])

                        sorted_indices = np.argsort(probs)[::-1]
                        top3 = [{"code": MILK10K_IDX_TO_CLASS[i], "class": MILK10K_IDX_TO_CLASS[i], "name": CLASS_INFO[MILK10K_IDX_TO_CLASS[i]]["name"], "prob": float(probs[i])} for i in sorted_indices[:3]]

                        mal_pre_prob = sum(float(probs[MILK10K_CLASS_TO_IDX[c]]) for c in MALIGNANT_OR_PREMALIGNANT_CLASSES if c in MILK10K_CLASS_TO_IDX)

                        # Photo Quality Checks
                        warns_c = check_image_quality(c_img_active, app_cfg, lang)[1] if c_img_active else []
                        warns_d = check_image_quality(d_img_active, app_cfg, lang)[1] if d_img_active else []
                        has_quality_warns = (len(warns_c) > 0 or len(warns_d) > 0)

                        triage = compute_triage_verdict(pred_class, conf, mal_pre_prob, has_quality_warns, app_cfg, lang)
                        is_high_risk = (triage["level"] == "red") or (mal_pre_prob >= float(app_cfg.get("high_risk_threshold", 0.35)))

                        res = {
                            "pred_class": pred_class,
                            "conf": conf,
                            "probs": probs,
                            "top3": top3,
                            "mal_pre_prob": mal_pre_prob,
                            "is_high_risk": is_high_risk,
                            "triage": triage,
                            "c_img": img_for_clin,
                            "d_img": img_for_dermo,
                            "age": age_val,
                            "sex": sex_val,
                            "site": site_val,
                            "gt_diagnosis": st.session_state.get("active_gt"),
                            "tone_group": st.session_state.get("active_tone", "Unknown"),
                            "example_name": st.session_state.get("example_name"),
                        }

                        st.session_state["cache"][input_hash] = res
                        status_box.update(label=t("toast_done", lang), state="complete", expanded=False)
                        st.toast(t("toast_done", lang))

                # Localize triage verdict
                if res and "triage" in res:
                    res["triage"] = compute_triage_verdict(
                        res["pred_class"],
                        res["conf"],
                        res.get("mal_pre_prob", 0.0),
                        has_quality_warnings=False,
                        cfg=app_cfg,
                        lang=lang,
                    )

                # Gate Result Popup Dialog: open once per new result
                if show_result_dialog and (st.session_state.get("popup_shown_for") != input_hash or st.session_state.get("open_popup_now")):
                    st.session_state["popup_shown_for"] = input_hash
                    st.session_state["open_popup_now"] = False
                    show_result_dialog(res, lang, theme, oof_bins)

                # --------------------------------------------------------------
                # PINNED COMPACT RESULT STRIP AT THE VERY TOP OF TAB 1
                # --------------------------------------------------------------
                triage_info = res["triage"]
                pred_c = res["pred_class"]
                conf_val = res["conf"]
                c_meta = CONDITION_DESCRIPTIONS.get(pred_c, {})
                display_name = c_meta.get(f"{lang}_name", c_meta.get("en_name", pred_c))
                display_desc = c_meta.get(f"{lang}_desc", c_meta.get("en_desc", ""))
                conf_word = get_confidence_label(conf_val, app_cfg, lang)
                advice = get_next_steps(pred_c, res.get("is_high_risk", False), lang)

                strip_level = triage_info.get("level", "grey")
                st.markdown('<div id="result-anchor"></div>', unsafe_allow_html=True)
                strip_c1, strip_c2 = st.columns([4, 1])
                with strip_c1:
                    st.markdown(f"""
                    <div class="result-strip result-strip-{strip_level}">
                        <div>
                            <span style="font-size:1.15rem; font-weight:800;">
                                {triage_info['icon']} {triage_info['title']} — {display_name} ({pred_c})
                            </span>
                            <div style="font-size:0.9rem; font-weight:600; opacity:0.9; margin-top:2px;">
                                {conf_word} ({conf_val*100:.1f}%) • {advice}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with strip_c2:
                    if st.button(t("view_details_btn", lang), key="btn_reopen_popup_strip", use_container_width=True):
                        st.session_state["open_popup_now"] = True
                        st.rerun()

                # --------------------------------------------------------------
                # SIDE-BY-SIDE RESULT & IMAGE PREVIEW
                # --------------------------------------------------------------
                col_preview, col_result = st.columns([5, 7])

                # Left Column: Image Previews & Demographics
                with col_preview:
                    st.markdown("#### 📸 Analyzed Lesion Photos")
                    is_ex = bool(st.session_state.get("example_name"))
                    ex_cap = st.session_state.get("example_name", "")
                    render_image_preview_card(
                        is_example=is_ex,
                        caption=ex_cap,
                        clin_img=img_for_clin,
                        derm_img=img_for_dermo if (c_img_active is not None and d_img_active is not None) else None,
                        badge_text=t("example_badge", lang) if is_ex else t("user_photo_badge", lang)
                    )

                    # Quality Warnings
                    warns_c = check_image_quality(c_img_active, app_cfg, lang)[1] if c_img_active else []
                    warns_d = check_image_quality(d_img_active, app_cfg, lang)[1] if d_img_active else []
                    for w in warns_c + warns_d:
                        st.warning(w)

                    # Patient Details Summary Card
                    with st.container():
                        st.markdown(f"""
                        <div class="vishwas-card" style="padding:12px 16px; margin-top:10px;">
                            <div style="font-weight:700; font-size:0.9rem; margin-bottom:4px;">👤 Patient Information</div>
                            <div style="font-size:0.85rem; color:var(--muted);">
                                • <b>Age</b>: {f'{age_val:.0f} yrs' if age_val else 'Unspecified'}<br>
                                • <b>Sex</b>: {sex_val.capitalize()}<br>
                                • <b>Body Site</b>: {site_val.capitalize()}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Right Column: Step 3 Result Details
                with col_result:
                    st.markdown(f'<div class="simple-step-header">{t("step3_title", lang)}</div>', unsafe_allow_html=True)
                    card_title_tag = t("most_likely_uncertain", lang) if triage_info.get("is_uncertain") else t("most_likely", lang)

                    st.markdown(f"""
                    <div class="result-strip result-strip-{strip_level}" style="display:block; padding:18px 22px; margin-bottom:14px;">
                        <span style="font-weight:700; font-size:0.85rem; letter-spacing:0.5px;">
                            {triage_info['icon']} {card_title_tag.upper()}
                        </span>
                        <div style="font-size:1.75rem; font-weight:800; margin-top:2px;">
                            {display_name} <span style="font-size:1.1rem; opacity:0.8; font-weight:600;">({pred_c})</span>
                        </div>
                        <p style="font-size:1.0rem; opacity:0.95; margin-top:6px; margin-bottom:10px; line-height:1.5;">
                            {display_desc}
                        </p>
                        <div style="font-size:0.85rem; opacity:0.8;">
                            {t('ai_estimate_tag', lang)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Confidence meter
                    c_col1, c_col2 = st.columns([1, 2])
                    with c_col1:
                        st.metric(
                            t("confidence_level", lang),
                            f"{conf_val*100:.1f}%",
                            f"{conf_word}",
                            delta_color="normal" if conf_val >= 0.55 else "off",
                        )
                    with c_col2:
                        st.markdown(f"**{t('confidence_level', lang)}: {conf_word}**")
                        st.progress(float(min(max(conf_val, 0.0), 1.0)))

                    # Evidence Line from Out-of-Fold Predictions
                    ev_line = get_evidence_line(conf_val, oof_bins, lang)
                    if ev_line:
                        st.markdown(f"📈 *{ev_line}*")

                    # High-Risk Alert Banner
                    if res.get("is_high_risk"):
                        st.error(t("clinical_review_banner", lang))

                    # What you can do next
                    st.markdown(f"#### 💡 {t('next_steps_title', lang)}")
                    st.info(advice)

                    # Top-3 Horizontal Ranked Bars Chart (Plotly)
                    st.markdown(f"#### 📊 {t('top3_title', lang)}")
                    top3_chart = plot_top3_horizontal(res["top3"], theme_mode=theme, lang=lang)
                    st.plotly_chart(top3_chart, use_container_width=True)

                    # Action Buttons
                    act_col1, act_col2 = st.columns(2)
                    with act_col1:
                        if st.button(t("rerun_btn", lang), use_container_width=True, key="btn_rerun_simple"):
                            st.session_state["cache"].pop(input_hash, None)
                            st.session_state["popup_shown_for"] = None
                            st.rerun()
                    with act_col2:
                        pdf_bytes = generate_pdf_report(res, img_for_clin, img_for_dermo, lang)
                        st.download_button(
                            label=t("download_pdf_btn", lang),
                            data=pdf_bytes,
                            file_name=f"VISHWAS_Result_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key="btn_pdf_simple",
                        )

                # Explanations Expanders below the side-by-side row
                st.markdown("---")
                render_disease_card(pred_c, lang)
                render_top3_differentials(res["top3"], lang)

                # Technical Expander for curious visitors
                with st.expander(t("show_details", lang), expanded=False):
                    st.markdown("##### 11-Class Complete Differential Softmax Probabilities")
                    all_classes_df = pd.DataFrame({
                        "Diagnosis Code": MILK10K_CLASSES,
                        "Full Condition Name": [CLASS_INFO[c]["name"] for c in MILK10K_CLASSES],
                        "Category": [CLASS_INFO[c]["type"] for c in MILK10K_CLASSES],
                        "Probability (%)": [res["probs"][i] * 100 for i in range(11)],
                    }).set_index("Diagnosis Code")
                    st.dataframe(all_classes_df, use_container_width=True)

                    if res.get("gt_diagnosis"):
                        st.markdown(f"**Dataset Reference Ground Truth**: `{res['gt_diagnosis']}` ({'✅ Match' if pred_c == res['gt_diagnosis'] else '⚠️ Discrepancy'})")

                st.markdown("---")

            # ------------------------------------------------------------------
            # EMPTY STATE BANNER & EXAMPLES (when no photos uploaded yet)
            # ------------------------------------------------------------------
            if c_img_active is None and d_img_active is None:
                if not st.session_state["dismiss_banner"]:
                    with st.container():
                        st.markdown(f"""
                        <div class="vishwas-card" style="padding:18px 24px; margin-bottom:18px;">
                            <h4 style="margin-top:0;">{t('how_it_works_title', lang)}</h4>
                            <p style="margin-bottom:6px;">{t('how_it_works_step1', lang)}</p>
                            <p style="margin-bottom:6px;">{t('how_it_works_step2', lang)}</p>
                            <p style="margin-bottom:12px;">{t('how_it_works_step3', lang)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button(t("dismiss_banner", lang), key="btn_dismiss_guide"):
                            st.session_state["dismiss_banner"] = True
                            st.rerun()

                st.markdown(f"#### {t('try_example_title', lang)}")
                ex_col1, ex_col2, ex_col3 = st.columns(3)

                curated_exs, gallery_err = load_curated_examples_with_status()
                if gallery_err:
                    st.error(f"⚠️ {gallery_err}")

                with ex_col1:
                    if st.button(t("example_mel", lang), use_container_width=True):
                        ex_m = get_example_by_id("mel")
                        if ex_m:
                            st.session_state["active_c_img"] = load_image_safe(ex_m.get("clinical_image"))
                            st.session_state["active_d_img"] = load_image_safe(ex_m.get("dermoscopic_image"))
                            st.session_state["active_age"] = int(ex_m.get("age", 55))
                            st.session_state["active_sex"] = ex_m.get("sex", "male")
                            st.session_state["active_site"] = ex_m.get("site", "torso")
                            st.session_state["active_gt"] = ex_m.get("diagnosis", "MEL")
                            st.session_state["active_tone"] = ex_m.get("tone_group", "G1")
                            st.session_state["example_name"] = ex_m.get("caption", "Melanoma (MEL, Tone G1)")
                            st.rerun()

                with ex_col2:
                    if st.button(t("example_bcc", lang), use_container_width=True):
                        ex_b = get_example_by_id("bcc")
                        if ex_b:
                            st.session_state["active_c_img"] = load_image_safe(ex_b.get("clinical_image"))
                            st.session_state["active_d_img"] = load_image_safe(ex_b.get("dermoscopic_image"))
                            st.session_state["active_age"] = int(ex_b.get("age", 65))
                            st.session_state["active_sex"] = ex_b.get("sex", "male")
                            st.session_state["active_site"] = ex_b.get("site", "head/neck")
                            st.session_state["active_gt"] = ex_b.get("diagnosis", "BCC")
                            st.session_state["active_tone"] = ex_b.get("tone_group", "G3")
                            st.session_state["example_name"] = ex_b.get("caption", "Basal Cell Carcinoma (BCC, Tone G3)")
                            st.rerun()

                with ex_col3:
                    if st.button(t("example_nv", lang), use_container_width=True):
                        ex_n = get_example_by_id("nv")
                        if ex_n:
                            st.session_state["active_c_img"] = load_image_safe(ex_n.get("clinical_image"))
                            st.session_state["active_d_img"] = load_image_safe(ex_n.get("dermoscopic_image"))
                            st.session_state["active_age"] = int(ex_n.get("age", 40))
                            st.session_state["active_sex"] = ex_n.get("sex", "male")
                            st.session_state["active_site"] = ex_n.get("site", "torso")
                            st.session_state["active_gt"] = ex_n.get("diagnosis", "NV")
                            st.session_state["active_tone"] = ex_n.get("tone_group", "G3")
                            st.session_state["example_name"] = ex_n.get("caption", "Normal Mole (NV, Tone G3)")
                            st.rerun()

                # Curated gallery dropdown for all 11 conditions
                if curated_exs:
                    with st.expander("🔍 Or pick from all 11 curated conditions (Skin tone groups G1-G3):", expanded=False):
                        labels = [f"{ex['diagnosis']} — {ex['diagnosis_name']} ({ex['tone_group']})" for ex in curated_exs]
                        chosen_idx = st.selectbox("Select condition:", range(len(labels)), format_func=lambda i: labels[i])
                        if st.button("Load Selected Example", key="btn_load_dropdown_ex"):
                            chosen_ex = curated_exs[chosen_idx]
                            st.session_state["active_c_img"] = load_image_safe(chosen_ex.get("clinical_image"))
                            st.session_state["active_d_img"] = load_image_safe(chosen_ex.get("dermoscopic_image"))
                            st.session_state["active_age"] = int(chosen_ex.get("age", 50))
                            st.session_state["active_sex"] = chosen_ex.get("sex", "unknown")
                            st.session_state["active_site"] = chosen_ex.get("site", "torso")
                            st.session_state["active_gt"] = chosen_ex.get("diagnosis")
                            st.session_state["active_tone"] = chosen_ex.get("tone_group")
                            st.session_state["example_name"] = chosen_ex.get("caption")
                            st.rerun()

                st.markdown("---")

            # ------------------------------------------------------------------
            # STEP 1: ADD PHOTOS
            # ------------------------------------------------------------------
            st.markdown(f'<div class="simple-step-header">{t("step1_title", lang)}</div>', unsafe_allow_html=True)
            st.markdown(t("step1_desc", lang))

            # Photography Tips Expander
            with st.expander(t("photo_tips_title", lang), expanded=False):
                st.markdown(t("photo_tips_content", lang))

            col_u1, col_u2 = st.columns(2)

            with col_u1:
                st.markdown(f"**{t('label_closeup', lang)}**")
                up_c_file = st.file_uploader(
                    "Upload Close-up Image",
                    type=["jpg", "jpeg", "png"],
                    key="simple_c_file",
                    label_visibility="collapsed",
                    help=t("help_closeup", lang),
                )
                if up_c_file:
                    st.session_state["active_c_img"] = Image.open(up_c_file)
                    st.session_state["example_name"] = None

                # Optional Camera Input
                if hasattr(st, "camera_input"):
                    cam_c = st.camera_input(t("camera_closeup", lang), key="cam_closeup")
                    if cam_c:
                        st.session_state["active_c_img"] = Image.open(cam_c)
                        st.session_state["example_name"] = None

            with col_u2:
                st.markdown(f"**{t('label_dermo', lang)}**")
                up_d_file = st.file_uploader(
                    "Upload Magnified Image",
                    type=["jpg", "jpeg", "png"],
                    key="simple_d_file",
                    label_visibility="collapsed",
                    help=t("help_dermo", lang),
                )
                if up_d_file:
                    st.session_state["active_d_img"] = Image.open(up_d_file)
                    st.session_state["example_name"] = None

                # Optional Camera Input for Dermoscopic / Macro
                if hasattr(st, "camera_input"):
                    cam_d = st.camera_input(t("camera_dermo", lang), key="cam_dermo")
                    if cam_d:
                        st.session_state["active_d_img"] = Image.open(cam_d)
                        st.session_state["example_name"] = None

            st.markdown("---")

            # ------------------------------------------------------------------
            # STEP 2: PATIENT DETAILS (OPTIONAL)
            # ------------------------------------------------------------------
            st.markdown(f'<div class="simple-step-header">{t("step2_title", lang)}</div>', unsafe_allow_html=True)
            st.markdown(t("step2_desc", lang))

            col_p1, col_p2, col_p3 = st.columns(3)

            with col_p1:
                st.markdown(f"**{t('label_age', lang)}**")
                dont_know_age = st.checkbox(
                    t("dont_know_age", lang),
                    value=st.session_state.get("active_dont_know_age", False),
                    key="chk_dont_know_age",
                )
                st.session_state["active_dont_know_age"] = dont_know_age

                current_age = int(st.session_state.get("active_age", 50))
                age_input = st.slider(
                    "Patient Age",
                    min_value=1,
                    max_value=100,
                    value=current_age,
                    disabled=dont_know_age,
                    label_visibility="collapsed",
                    help=t("help_age", lang),
                )
                if not dont_know_age:
                    st.session_state["active_age"] = age_input

            with col_p2:
                st.markdown(f"**{t('label_sex', lang)}**")
                sex_options = [t("sex_male", lang), t("sex_female", lang), t("sex_unknown", lang)]
                current_sex = st.session_state.get("active_sex", "unknown")
                sex_idx = 0 if current_sex == "male" else (1 if current_sex == "female" else 2)

                if hasattr(st, "segmented_control"):
                    chosen_sex = st.segmented_control(
                        "Sex Assigned at Birth",
                        sex_options,
                        default=sex_options[sex_idx],
                        label_visibility="collapsed",
                        help=t("help_sex", lang),
                    )
                else:
                    chosen_sex = st.radio(
                        "Sex Assigned at Birth",
                        sex_options,
                        index=sex_idx,
                        horizontal=True,
                        label_visibility="collapsed",
                        help=t("help_sex", lang),
                    )
                if chosen_sex:
                    st.session_state["active_sex"] = "male" if chosen_sex == sex_options[0] else ("female" if chosen_sex == sex_options[1] else "unknown")

            with col_p3:
                st.markdown(f"**{t('label_site', lang)}**")
                site_map = {
                    "torso": t("site_trunk", lang),
                    "head/neck": t("site_head_neck", lang),
                    "lower extremity": t("site_lower_ext", lang),
                    "upper extremity": t("site_upper_ext", lang),
                    "palms/soles": t("site_foot", lang),
                    "genital": t("site_genital", lang),
                    "unknown": t("site_unknown", lang),
                }
                inv_site_map = {v: k for k, v in site_map.items()}
                site_options = list(site_map.values())
                current_site = st.session_state.get("active_site", "unknown")
                site_idx = site_options.index(site_map.get(current_site, t("site_unknown", lang)))

                chosen_site_display = st.selectbox(
                    "Anatomical Location",
                    site_options,
                    index=site_idx,
                    label_visibility="collapsed",
                    help=t("help_site", lang),
                )
                site_val = inv_site_map[chosen_site_display]
                st.session_state["active_site"] = site_val

        # ----------------------------------------------------------------------
        # EXPERT MODE: TECHNICAL RESEARCH INTERFACE
        # ----------------------------------------------------------------------
        else:
            st.subheader("Multimodal Clinical + Dermoscopic Lesion Diagnostic Inference")
            st.markdown(
                "Test live predictions using either our **Two-Image + Tabular Fusion Model (`d_cw_128`)** "
                "or the **Dermoscopic Baseline (`a_cw_128`)**, with individual checkpoints or 5-fold ensemble."
            )

            col_input_left, col_input_right = st.columns([1, 1])

            with col_input_left:
                st.markdown("#### 1. Input Modalities")
                input_mode = st.radio("Choose Input Method:", ["Select from Dataset Gallery", "Upload Custom Images"], horizontal=True)

                selected_c_img: Optional[Image.Image] = None
                selected_d_img: Optional[Image.Image] = None
                gt_diagnosis: Optional[str] = None
                tone_group_val: str = "G2"
                skin_tone_val: int = 3
                sample_age = 55.0
                sample_sex = "male"
                sample_site = "trunk"

                if input_mode == "Select from Dataset Gallery":
                    curated, gal_err = load_curated_examples_with_status()
                    if gal_err:
                        st.error(f"⚠️ {gal_err}")
                    elif curated:
                        options = [f"{ex['diagnosis']} — {ex['diagnosis_name']} (Tone {ex['tone_group']})" for ex in curated]
                        chosen_i = st.selectbox("Pick a Sample Lesion:", range(len(options)), format_func=lambda i: options[i])
                        chosen_ex = curated[chosen_i]
                        selected_c_img = load_image_safe(chosen_ex.get("clinical_image"))
                        selected_d_img = load_image_safe(chosen_ex.get("dermoscopic_image"))
                        gt_diagnosis = chosen_ex.get("diagnosis")
                        skin_tone_val = int(chosen_ex.get("fitzpatrick", 3))
                        tone_group_val, _ = get_tone_group_for_class(chosen_ex.get("tone_group", ""), skin_tone_val)
                        sample_age = float(chosen_ex.get("age", 50))
                        sample_sex = str(chosen_ex.get("sex", "male"))
                        sample_site = str(chosen_ex.get("site", "torso"))
                    else:
                        st.error("No example records found in assets/examples/examples.json.")

                else:
                    c_file = st.file_uploader("Upload Clinical Close-up Image", type=["jpg", "jpeg", "png"], key="exp_c_file")
                    d_file = st.file_uploader("Upload Dermoscopic Image", type=["jpg", "jpeg", "png"], key="exp_d_file")
                    if c_file:
                        selected_c_img = Image.open(c_file)
                    if d_file:
                        selected_d_img = Image.open(d_file)

                    m_c1, m_c2, m_c3 = st.columns(3)
                    with m_c1:
                        sample_age = float(st.number_input("Patient Approximate Age:", min_value=0, max_value=110, value=55))
                    with m_c2:
                        sample_sex = st.selectbox("Patient Sex:", SEX_CATEGORIES)
                    with m_c3:
                        sample_site = st.selectbox("Anatomical Site:", SITE_CATEGORIES)

                # Display images side by side
                img_c1, img_c2 = st.columns(2)
                with img_c1:
                    st.markdown("**Clinical Close-up View**")
                    if selected_c_img:
                        st.image(selected_c_img, caption="Clinical Close-up View", use_container_width=True)
                    else:
                        st.caption("No clinical image selected.")
                with img_c2:
                    st.markdown("**Dermoscopic Microscopic View**")
                    if selected_d_img:
                        st.image(selected_d_img, caption="Dermoscopic Microscopic View", use_container_width=True)
                    else:
                        st.caption("No dermoscopic image selected.")

            with col_input_right:
                st.markdown("#### 2. Inference Configuration & Execution")

                model_choice = st.radio(
                    "Model Architecture:",
                    [
                        "Two-Image + Tabular Fusion (d_cw_128)",
                        "Dermoscopic Only Baseline (a_cw_128)",
                    ],
                )

                ckpt_choice = st.selectbox(
                    "Checkpoint Selection:",
                    ["5-Fold Cross-Validation Ensemble (Recommended)", "Fold 0", "Fold 1", "Fold 2", "Fold 3", "Fold 4"],
                )

                folds_to_eval = [0, 1, 2, 3, 4] if "Ensemble" in ckpt_choice else [int(ckpt_choice.split(" ")[1])]
                tabular_tensor = encode_tabular_vector(sample_age, sample_sex, sample_site)

                # Robust tone group label resolution (Never "Tone Group: unknown")
                _, tg_label = get_tone_group_for_class(tone_group_val, skin_tone_val)
                st.markdown(f"""
                <div class="vishwas-card" style="padding:14px 18px; margin-bottom:12px;">
                    <b>Patient Clinical Profile</b><br>
                    • <b>Tone Group</b>: <code>{tg_label}</code> (skin_tone_class: {skin_tone_val})<br>
                    • <b>Demographics</b>: Age {sample_age:.0f} | Sex: {sample_sex.capitalize()} | Site: {sample_site}<br>
                    • <b>Ground Truth Diagnosis</b>: <code>{gt_diagnosis if gt_diagnosis else 'External / Unknown'}</code>
                </div>
                """, unsafe_allow_html=True)

                run_clicked = st.button("🚀 Run Diagnostic Inference", type="primary", use_container_width=True)

                if run_clicked or (selected_c_img is not None and selected_d_img is not None):
                    if selected_d_img is None and selected_c_img is None:
                        st.error("Please provide at least one image to run inference.")
                    else:
                        c_img_to_use = selected_c_img if selected_c_img is not None else selected_d_img
                        d_img_to_use = selected_d_img if selected_d_img is not None else selected_c_img

                        with st.spinner("Executing neural inference on Apple Silicon MPS/CPU..."):
                            c_tensor = IMAGE_TRANSFORMS(c_img_to_use.convert("RGB")).unsqueeze(0).to(device)
                            d_tensor = IMAGE_TRANSFORMS(d_img_to_use.convert("RGB")).unsqueeze(0).to(device)

                            exp_target = "d_cw_128" if "Fusion" in model_choice else "a_cw_128"
                            probs = predict_ensemble(exp_target, c_tensor, d_tensor, tabular_tensor, folds_to_eval)

                            top_idx = int(np.argmax(probs))
                            pred_class = MILK10K_IDX_TO_CLASS[top_idx]
                            conf = float(probs[top_idx])

                            st.success(f"**Predicted Finding**: `{pred_class}` — **{CLASS_INFO[pred_class]['name']}** ({conf*100:.1f}%)")

                            # Top-3 chart
                            sorted_idx = np.argsort(probs)[::-1]
                            top3_exp = [{"class": MILK10K_IDX_TO_CLASS[i], "prob": float(probs[i]), "name": CLASS_INFO[MILK10K_IDX_TO_CLASS[i]]["name"]} for i in sorted_idx[:3]]
                            st.plotly_chart(plot_top3_horizontal(top3_exp, theme_mode=theme, lang=lang), use_container_width=True)

                            # 11-class probability table
                            with st.expander("📊 Complete 11-Class Softmax Differential Distribution", expanded=False):
                                df_exp = pd.DataFrame({
                                    "Code": MILK10K_CLASSES,
                                    "Condition": [CLASS_INFO[c]["name"] for c in MILK10K_CLASSES],
                                    "Probability (%)": [probs[i] * 100 for i in range(11)]
                                }).set_index("Code")
                                st.dataframe(df_exp, use_container_width=True)

    # ==========================================================================
    # TAB 2: HOW FAIR IS IT? (FAIRNESS BENCHMARK)
    # ==========================================================================
    with tabs[1]:
        st.subheader("⚖️ Multi-Model Fairness Auditing & Subgroup Disparity Benchmark")
        st.markdown(
            "Evaluation of all **5,240 held-out lesions** across full 5-fold cross-validation. "
            "Tone groups are labeled strictly by confirmed `skin_tone_class` values: "
            "`G1 = {0,1,2}`, `G2 = {3}`, `G3 = {4,5}`. Confidence intervals calculated via 1,000 bootstrap lesion resamples."
        )

        fairness_data = load_summary_json("fairness")
        disparity_info = fairness_data.get("disparity", {})

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Headline Macro-F1", "0.3773", "+0.0190 vs Baseline", delta_color="normal")
        with col_m2:
            st.metric("Overall Accuracy", f"{fairness_data.get('overall', {}).get('accuracy', 0.4771)*100:.1f}%", "+2.58% vs Baseline", delta_color="normal")
        with col_m3:
            gap_pct = disparity_info.get("accuracy_gap_pct", 4.26)
            st.metric("Demographic Disparity Gap", f"{gap_pct:.2f}%", "-37% compression vs Baseline", delta_color="normal")
        with col_m4:
            st.metric("Worst-Group F1 (G2)", "0.3611", "+0.0248 vs Baseline", delta_color="normal")

        st.markdown("---")
        st.markdown("### 1. Interactive Subgroup Parity Across Skin Tone Groups")
        
        if fairness_data:
            st.plotly_chart(plot_fairness_bars(fairness_data, theme_mode=theme), use_container_width=True)
            if "clinical_verdict" in fairness_data:
                st.info(f"💡 **Clinical Fairness Verdict**: {fairness_data['clinical_verdict']}")

        with st.expander("💡 What does this subgroup parity chart mean?", expanded=False):
            st.markdown(t("chart_explain_fairness", lang))

        # Technical Multi-Model Comparison Table
        with st.expander("🔬 Comprehensive Multi-Model Benchmarking Table (Expert)", expanded=False):
            comp_table_data = {
                "Metric / Dimension": [
                    "Architecture",
                    "Input Modalities",
                    "Mitigation Strategy",
                    "Overall Accuracy (95% CI)",
                    "Headline Macro-F1 (95% CI) (10 classes)",
                    "Full Macro-F1 (11 classes)",
                    "Balanced Accuracy (95% CI)",
                    "Calibration ECE (95% CI)",
                    "Macro-AUC (OvR)",
                    "Worst-Performing Tone Group",
                    "Headline Macro-F1 Disparity Gap",
                    "Balanced Accuracy Disparity Gap",
                ],
                "Two-Image Tabular Fusion (d_cw_128)": [
                    "Two-Image + Tabular Fusion",
                    "Clinical + Dermoscopic + Age/Sex/Site",
                    "class_weighted (cap=10.0)",
                    "47.71% [46.30%, 49.07%]",
                    "0.3773 [0.3611, 0.3938]",
                    "0.3430",
                    "46.19% [43.57%, 48.76%]",
                    "0.0911 [0.0791, 0.1049]",
                    "0.8090",
                    "G2 = {3} (n=3,174, F1: 0.3611)",
                    "0.0461 (Best - Worst)",
                    "3.47%",
                ],
                "Dermoscopic Only (a_cw_128)": [
                    "Dermoscopic Only",
                    "Dermoscopic View Only",
                    "class_weighted (cap=10.0)",
                    "45.13% [43.87%, 46.55%]",
                    "0.3583 [0.3423, 0.3754]",
                    "0.3257",
                    "43.27% [41.02%, 45.90%]",
                    "0.0857 [0.0750, 0.1008]",
                    "0.8030",
                    "G2 = {3} (n=3,174, F1: 0.3363)",
                    "0.0732 (Best - Worst)",
                    "8.87%",
                ],
            }
            st.dataframe(pd.DataFrame(comp_table_data).set_index("Metric / Dimension"), use_container_width=True)

    # ==========================================================================
    # TAB 3: HOW RELIABLE IS IT? (CALIBRATION & CONFUSION EXPLORER)
    # ==========================================================================
    with tabs[2]:
        st.subheader("📈 Model Reliability & Diagnostic Confusion Explorer")
        st.markdown(
            "An AI tool must be both accurate and well-calibrated. Below, inspect how closely the AI's "
            "confidence scores match real-world accuracy, and explore which conditions it most frequently confuses."
        )

        rel_data = load_summary_json("reliability")
        conf_data = load_summary_json("confusion")

        st.markdown("### 1. Confidence Calibration: 'When the AI says X% sure, how often is it right?'")
        
        # Tone Group Selector for Calibration
        tg_opts = ["All", "G1", "G2", "G3"]
        tg_labels = {
            "All": "All Skin Tones (5,240 held-out lesions)",
            "G1": "G1 — Lighter Tones (n=617)",
            "G2": "G2 — Medium Tones (n=3,174)",
            "G3": "G3 — Darker Tones (n=1,449)",
        }
        chosen_tg = st.segmented_control("Tone Group Filter:", tg_opts, default="All", format_func=lambda k: tg_labels[k])

        if rel_data:
            st.plotly_chart(plot_reliability_chart(rel_data, tone_group=chosen_tg, theme_mode=theme), use_container_width=True)
            if "clinical_verdict" in rel_data:
                st.caption(f"ℹ️ **Verdict**: {rel_data['clinical_verdict']}")

        with st.expander("💡 What does model calibration mean?", expanded=False):
            st.markdown(t("chart_explain_calibration", lang))

        st.markdown("---")
        st.markdown("### 2. Diagnostic Sensitivity Ranking & Confusion Analysis")
        st.write("Ranked sensitivity across all 11 conditions. Critical malignant conditions (Melanoma, Basal Cell) achieve high recall.")

        if conf_data:
            st.plotly_chart(plot_confusion_recall_chart(conf_data, theme_mode=theme), use_container_width=True)

            # Interactive Disease Confusion Inspector
            st.markdown("#### 🔍 What does the AI confuse it with?")
            all_classes = conf_data.get("classes", MILK10K_CLASSES)
            disease_picker = st.selectbox(
                "Select a skin condition to inspect its confusion profile:",
                all_classes,
                format_func=lambda c: f"{c} — {conf_data.get('by_disease', {}).get(c, {}).get('name', c)}"
            )
            st.plotly_chart(plot_confusion_detail_chart(conf_data, disease_picker, theme_mode=theme), use_container_width=True)

            entry = conf_data.get("by_disease", {}).get(disease_picker, {})
            if "plain_explanation" in entry:
                st.info(f"💡 {entry['plain_explanation']}")

        # Technical Expander with static matrix PNGs
        with st.expander("🔬 Static Confusion Matrices & Detailed ROC Curves (Expert)", expanded=False):
            root = get_vishwas_root()
            rep_d = root / "reports" / "d_cw_128"
            cm_p1 = rep_d / "confusion_matrix.png"
            if cm_p1.exists():
                st.image(str(cm_p1), caption="Multimodal Tabular Fusion 11x11 Confusion Matrix", use_container_width=True)

    # ==========================================================================
    # TAB 4: DATA EXPLORER
    # ==========================================================================
    with tabs[3]:
        st.subheader("🗂️ MILK10k 5-Fold Cross-Validation Stratification Explorer")
        st.markdown(
            "Every lesion contains exactly two views (clinical close-up and dermoscopic) which are **strictly grouped together** "
            "within the same fold to prevent data leakage. Fold sizes are balanced to exactly 1,048 lesions each."
        )

        dataset_data = load_summary_json("dataset")
        if dataset_data:
            d_c1, d_c2, d_c3 = st.columns(3)
            with d_c1:
                st.metric("Total Unique Lesions", f"{dataset_data.get('total_lesions', 5240):,}")
            with d_c2:
                st.metric("Total Evaluated Images", f"{dataset_data.get('total_images', 10480):,} images")
            with d_c3:
                st.metric("Lesions per Fold", "1,048 exactly")

            st.markdown("#### Skin Tone Subgroup Distribution")
            tones = dataset_data.get("tone_distribution", {})
            t_df = pd.DataFrame([
                {"Tone Group": k, "Lesion Count": v["count"], "Percentage": f"{v['percentage']}%"}
                for k, v in tones.items()
            ]).set_index("Tone Group")
            st.dataframe(t_df, use_container_width=True)

        if not folds_df.empty:
            st.markdown("#### Cross-Validation Lesions by Fold and Tone Group")
            ct = pd.crosstab(folds_df["fold"], folds_df["tone_group"])
            st.dataframe(ct, use_container_width=True)

    # ==========================================================================
    # TAB 5: ABOUT THE DATA
    # ==========================================================================
    with tabs[4]:
        st.subheader("📚 Integrated Project Datasets Overview")
        st.markdown(
            "The Vishwas framework integrates **110,480 total images** across two specialized stages: "
            "**Stage-1 SLICE-3D (100,000 images)** for self-supervised binary pretraining, and "
            "**Stage-2 MILK10k (10,480 images / 5,240 pairs)** for multimodal fusion and skin-tone fairness auditing."
        )

        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Total Project Images", "110,480")
        with d2:
            st.metric("Stage-1 SLICE-3D", "100,000 images", "Pretraining")
        with d3:
            st.metric("Stage-2 MILK10k", "10,480 images", "5-Fold Evaluation")

        st.markdown("---")
        st.markdown("#### Cohort Partition Specifications")
        cohort_data = [
            {"Dataset": "SLICE-3D Train", "Role": "Stage-1 Pretraining", "Images": "67,500", "Patients": "700", "Labels": "Binary (Benign/Malignant)"},
            {"Dataset": "SLICE-3D Val", "Role": "Validation (10%)", "Images": "7,500", "Patients": "100", "Labels": "Binary (Benign/Malignant)"},
            {"Dataset": "SLICE-3D Test", "Role": "Held-out Test", "Images": "25,000", "Patients": "256", "Labels": "Binary (Benign/Malignant)"},
            {"Dataset": "MILK10k Folds", "Role": "Multimodal + Fairness", "Images": "10,480 (5,240 pairs)", "Patients": "Strictly Grouped", "Labels": "11 Classes + Tone Groups"},
        ]
        st.dataframe(pd.DataFrame(cohort_data).set_index("Dataset"), use_container_width=True)

    # ==========================================================================
    # TAB 6: SKIN CONDITIONS GUIDE
    # ==========================================================================
    with tabs[5]:
        render_skin_conditions_guide(lang)


if __name__ == "__main__":
    main()
