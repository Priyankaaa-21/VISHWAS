"""
Vishwas (विश्वास) - User-Friendly & Multimodal Dermatological AI Platform
Streamlit Application: src/app/app.py

Modes:
- Simple Mode (Default): Guided 3-step flow (Add photos -> Patient details -> Plain-language result),
  bilingual (English & Hindi), friendly progress, photo quality checks, session history, and PDF download.
- Expert Mode: Technical research interface with dataset gallery, model architecture comparison,
  fold selection, raw probability charts, calibration analysis, and subgroup fairness auditing.
"""

from __future__ import annotations

import io
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
    )

# Load App Configuration
@st.cache_data
def load_app_config() -> Dict[str, Any]:
    cfg_path = VISHWAS_ROOT / "configs" / "config.yaml"
    cfg: Dict[str, Any] = {}
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f) or {}
                cfg = raw_cfg.get("app", {})
        except Exception as e:
            logging.warning(f"Failed to parse config.yaml: {e}")
    merged = dict(DEFAULT_APP_CONFIG)
    merged.update(cfg)
    return merged


# Class descriptions & clinical notes
CLASS_INFO = {
    "AKIEC": {"name": "Actinic Keratosis / Intraepithelial Carcinoma", "type": "Pre-cancerous / In-situ", "color": "#e67e22"},
    "BCC": {"name": "Basal Cell Carcinoma", "type": "Malignant Skin Cancer", "color": "#e74c3c"},
    "BEN_OTH": {"name": "Other Benign Lesions", "type": "Benign", "color": "#27ae60"},
    "BKL": {"name": "Benign Keratotic Lesions", "type": "Benign", "color": "#2ecc71"},
    "DF": {"name": "Dermatofibroma", "type": "Benign", "color": "#1abc9c"},
    "INF": {"name": "Inflammatory / Infectious Conditions", "type": "Benign / Reactive", "color": "#3498db"},
    "MAL_OTH": {"name": "Other Rare Malignancies", "type": "Malignant (Sparse, non-headline)", "color": "#c0392b"},
    "MEL": {"name": "Melanoma", "type": "High-Risk Malignant Cancer", "color": "#8e44ad"},
    "NV": {"name": "Melanocytic Nevus (Mole)", "type": "Benign Common Mole", "color": "#16a085"},
    "SCCKA": {"name": "Squamous Cell Carcinoma / Keratoacanthoma", "type": "Malignant Skin Cancer", "color": "#d35400"},
    "VASC": {"name": "Vascular Lesions", "type": "Benign", "color": "#2980b9"},
}

TONE_GROUP_INFO = {
    "G1": {"label": "G1 = {0, 1, 2}", "desc": "skin_tone_class values 0, 1, 2 (Lighter tones)", "color": "#2ca02c"},
    "G2": {"label": "G2 = {3}", "desc": "skin_tone_class value 3 (Medium tones)", "color": "#ff7f0e"},
    "G3": {"label": "G3 = {4, 5}", "desc": "skin_tone_class values 4, 5 (Darker tones)", "color": "#d62728"},
    "unknown": {"label": "unknown", "desc": "Missing tone annotation", "color": "#7f7f7f"},
}


@st.cache_resource
def get_device() -> torch.device:
    """Detects best available inference device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@st.cache_data
def load_dataset_metadata() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads folds mapping, image metadata, and SLICE-3D split for explorer."""
    folds_path = VISHWAS_ROOT / "data" / "milk10k_folds.csv"
    meta_path = VISHWAS_ROOT / "data" / "milk10k" / "MILK10k_Training_Metadata.csv"
    slice3d_path = VISHWAS_ROOT / "data" / "slice3d_split_100k.csv"

    folds_df = pd.DataFrame()
    meta_df = pd.DataFrame()
    slice3d_df = pd.DataFrame()

    try:
        folds_df = pd.read_csv(folds_path)
    except (OSError, PermissionError, Exception) as e:
        logging.warning(f"Failed to load folds_path: {e}")

    try:
        meta_df = pd.read_csv(meta_path)
    except (OSError, PermissionError, Exception) as e:
        logging.warning(f"Failed to load meta_path: {e}")

    try:
        slice3d_df = pd.read_csv(slice3d_path)
    except (OSError, PermissionError, Exception) as e:
        logging.warning(f"Failed to load slice3d_path: {e}")

    return folds_df, meta_df, slice3d_df


@st.cache_resource
def load_checkpoint_model(
    model_type: str,
    fold: int,
    device_name: str,
) -> Optional[torch.nn.Module]:
    """Loads a specific model fold checkpoint from disk."""
    dev = torch.device(device_name)
    run_prefix = "d_cw_128" if "fusion" in model_type else "a_cw_128"
    ckpt_path = VISHWAS_ROOT / "checkpoints" / f"best_{run_prefix}_fold{fold}.pt"

    if not ckpt_path.exists():
        return None

    try:
        model = build_model(
            model_type=model_type,
            num_classes=11,
            backbone="efficientnet_b0",
            pretrained=False,
            tabular_dim=13,
        ).to(dev)
        state = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception as e:
        logging.warning(f"Error loading {ckpt_path.name}: {e}")
        return None


def get_image_transform(img_size: int = 128) -> transforms.Compose:
    """Standard evaluation image transform."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def encode_tabular_vector(
    age: Optional[float],
    sex: str,
    site: str,
    tabular_scaler: Optional[TabularScaler] = None,
) -> torch.Tensor:
    """Encodes clinical metadata into standardized 13-dim tabular tensor."""
    scaler = tabular_scaler or TabularScaler(use_monet=False)
    age_norm, age_missing = scaler.transform_age(age)

    # Sex one-hot (male, female, unknown)
    sex_clean = str(sex).lower().strip()
    sex_one_hot = [
        1.0 if sex_clean == "male" else 0.0,
        1.0 if sex_clean == "female" else 0.0,
        1.0 if sex_clean not in ["male", "female"] else 0.0,
    ]

    # Site one-hot
    site_clean = str(site).lower().strip()
    site_one_hot = [1.0 if site_clean == s else 0.0 for s in SITE_CATEGORIES[:-1]]
    is_known_site = any(site_one_hot)
    site_one_hot.append(0.0 if is_known_site else 1.0)

    vec = [age_norm, age_missing] + sex_one_hot + site_one_hot
    return torch.tensor(vec, dtype=torch.float32).unsqueeze(0)


def predict_single_model(
    model: torch.nn.Module,
    model_type: str,
    clinical_img: Image.Image,
    dermoscopic_img: Image.Image,
    tabular_tensor: torch.Tensor,
    device: torch.device,
    img_size: int = 128,
) -> np.ndarray:
    """Runs single model forward pass and returns softmax probabilities."""
    transform = get_image_transform(img_size)
    c_tensor = transform(clinical_img.convert("RGB")).unsqueeze(0).to(device)
    d_tensor = transform(dermoscopic_img.convert("RGB")).unsqueeze(0).to(device)
    t_tensor = tabular_tensor.to(device)

    with torch.no_grad():
        if "fusion" in model_type:
            logits = model(clinical_img=c_tensor, dermoscopic_img=d_tensor, tabular=t_tensor)
        else:
            logits = model(dermoscopic_img=d_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def predict_ensemble(
    model_type: str,
    clinical_img: Image.Image,
    dermoscopic_img: Image.Image,
    tabular_tensor: torch.Tensor,
    device: torch.device,
    folds: List[int],
    img_size: int = 128,
) -> Tuple[np.ndarray, bool]:
    """Averages softmax probabilities across cross-validation checkpoints. Returns (probs, checkpoints_found)."""
    all_probs = []
    for f in folds:
        m = load_checkpoint_model(model_type, f, str(device))
        if m is not None:
            p = predict_single_model(m, model_type, clinical_img, dermoscopic_img, tabular_tensor, device, img_size)
            all_probs.append(p)

    if not all_probs:
        return np.ones(11) / 11.0, False
    return np.mean(all_probs, axis=0), True


FALLBACK_EXAMPLES: Dict[str, Dict[str, Any]] = {
    "IL_0192026": {
        "lesion_id": "IL_0192026",
        "clin_isic": "ISIC_1666674",
        "dermo_isic": "ISIC_6581571",
        "gt_diagnosis": "MEL",
        "tone_group": "G3",
        "skin_tone_class": 5,
        "age": 60.0,
        "sex": "female",
        "site": "lower extremity",
        "fold": 4,
    },
    "IL_0000652": {
        "lesion_id": "IL_0000652",
        "clin_isic": "ISIC_8149219",
        "dermo_isic": "ISIC_4671410",
        "gt_diagnosis": "BCC",
        "tone_group": "G1",
        "skin_tone_class": 1,
        "age": 65.0,
        "sex": "male",
        "site": "head/neck",
        "fold": 0,
    },
    "IL_0047878": {
        "lesion_id": "IL_0047878",
        "clin_isic": "ISIC_2308076",
        "dermo_isic": "ISIC_0433706",
        "gt_diagnosis": "NV",
        "tone_group": "G2",
        "skin_tone_class": 3,
        "age": 40.0,
        "sex": "male",
        "site": "torso",
        "fold": 0,
    },
}

# Helper to fetch cached image for example cases
def load_example_lesion(lesion_id: str, folds_df: pd.DataFrame, meta_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    cache_128 = VISHWAS_ROOT / "data" / "cache" / "128"

    # Check fallback definitions first if metadata is unavailable
    if (folds_df.empty or meta_df.empty) and lesion_id in FALLBACK_EXAMPLES:
        fb = FALLBACK_EXAMPLES[lesion_id]
        c_img = None
        d_img = None
        if cache_128.exists():
            cp = cache_128 / f"{fb['clin_isic']}.jpg"
            dp = cache_128 / f"{fb['dermo_isic']}.jpg"
            if cp.exists():
                c_img = Image.open(cp)
            if dp.exists():
                d_img = Image.open(dp)
        return {
            "lesion_id": fb["lesion_id"],
            "c_img": c_img,
            "d_img": d_img,
            "gt_diagnosis": fb["gt_diagnosis"],
            "tone_group": fb["tone_group"],
            "skin_tone_class": fb["skin_tone_class"],
            "age": fb["age"],
            "sex": fb["sex"],
            "site": fb["site"],
            "fold": fb["fold"],
        }

    if folds_df.empty or meta_df.empty:
        return None

    sub = folds_df[folds_df["lesion_id"] == lesion_id]
    if sub.empty:
        if lesion_id in FALLBACK_EXAMPLES:
            return load_example_lesion(lesion_id, pd.DataFrame(), pd.DataFrame())
        return None
    row = sub.iloc[0]
    gt_diagnosis = str(row["diagnosis"])
    tone_group_val = str(row["tone_group"])
    skin_tone_val = int(row["skin_tone_class"])

    meta_lesion = meta_df[meta_df["lesion_id"] == lesion_id]
    clin_row = meta_lesion[meta_lesion["image_type"] == "clinical: close-up"]
    dermo_row = meta_lesion[meta_lesion["image_type"] == "dermoscopic"]

    sample_age = float(clin_row.iloc[0]["age_approx"]) if not clin_row.empty and pd.notna(clin_row.iloc[0]["age_approx"]) else 50.0
    sample_sex = str(clin_row.iloc[0]["sex"]) if not clin_row.empty and pd.notna(clin_row.iloc[0]["sex"]) else "unknown"
    sample_site = str(clin_row.iloc[0]["site"]) if not clin_row.empty and pd.notna(clin_row.iloc[0]["site"]) else "unknown"

    c_img = None
    d_img = None
    if not clin_row.empty and cache_128.exists():
        cp = cache_128 / f"{clin_row.iloc[0]['isic_id']}.jpg"
        if cp.exists():
            c_img = Image.open(cp)
    if not dermo_row.empty and cache_128.exists():
        dp = cache_128 / f"{dermo_row.iloc[0]['isic_id']}.jpg"
        if dp.exists():
            d_img = Image.open(dp)

    return {
        "lesion_id": lesion_id,
        "c_img": c_img,
        "d_img": d_img,
        "gt_diagnosis": gt_diagnosis,
        "tone_group": tone_group_val,
        "skin_tone_class": skin_tone_val,
        "age": sample_age,
        "sex": sample_sex,
        "site": sample_site,
        "fold": int(row["fold"]),
    }


# ==============================================================================
# UI RENDERING
# ==============================================================================

def main():
    app_cfg = load_app_config()

    # Session State Initialization
    if "app_mode" not in st.session_state:
        st.session_state["app_mode"] = "Simple"
    if "lang" not in st.session_state:
        st.session_state["lang"] = app_cfg.get("default_language", "en")
    if "cache" not in st.session_state:
        st.session_state["cache"] = {}
    if "history" not in st.session_state:
        st.session_state["history"] = []
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

    # --------------------------------------------------------------------------
    # SIDEBAR CONFIGURATION & GLOSSARY
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown(f"## 🩺 {t('app_title', lang).split(':')[0]}")

        # Interface Mode Toggle
        mode_options = [t("mode_simple", lang), t("mode_expert", lang)]
        current_mode_idx = 0 if st.session_state["app_mode"] == "Simple" else 1

        selected_mode_str = st.radio(
            t("mode_selector", lang),
            mode_options,
            index=current_mode_idx,
            help="Choose 'Simple' for a clean, guided 3-step check, or 'Expert' for technical model benchmarking and research tools.",
        )
        st.session_state["app_mode"] = "Simple" if selected_mode_str == mode_options[0] else "Expert"

        # Language Selector
        lang_choice = st.selectbox(
            t("language_selector", lang),
            ["English", "हिंदी (Hindi)"],
            index=0 if lang == "en" else 1,
            help="Select preferred language for Simple Mode guidance and condition explanations.",
        )
        new_lang = "en" if "English" in lang_choice else "hi"
        if new_lang != st.session_state["lang"]:
            st.session_state["lang"] = new_lang
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

        # Persistent Medical Disclaimer
        st.caption(t("medical_disclaimer", lang))

    # Responsive CSS Styling
    st.markdown("""
        <style>
        .simple-step-header {
            font-size: 1.25rem;
            font-weight: 700;
            color: #1e3a8a;
            margin-top: 1rem;
            margin-bottom: 0.3rem;
        }
        .main-header-title {
            font-size: 2.1rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.1rem;
        }
        .main-header-sub {
            font-size: 1.05rem;
            color: #475569;
            margin-bottom: 1.2rem;
        }
        .result-card {
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 16px;
        }
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main Header
    st.markdown(f'<div class="main-header-title">{t("app_title", lang)}</div>', unsafe_allow_html=True)
    if st.session_state["app_mode"] == "Simple":
        st.markdown(f'<div class="main-header-sub">{t("app_subtitle_simple", lang)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="main-header-sub">{t("app_subtitle_expert", lang)}</div>', unsafe_allow_html=True)

    device = get_device()
    folds_df, meta_df, slice3d_df = load_dataset_metadata()

    # Friendly Missing Checkpoints Verification
    sample_ckpt = VISHWAS_ROOT / "checkpoints" / "best_d_cw_128_fold0.pt"
    if not sample_ckpt.exists():
        st.error(
            "⚠️ **Model Checkpoints Not Found in `checkpoints/`**: Running in demonstration mode. "
            "Please ensure trained fold checkpoints (`best_d_cw_128_fold*.pt`) are placed in the `checkpoints/` folder."
        )

    # --------------------------------------------------------------------------
    # TABS: RENAMED WITH FRIENDLY LABELS
    # --------------------------------------------------------------------------
    tabs = st.tabs([
        t("tab_check", lang),
        t("tab_fairness", lang),
        t("tab_reliability", lang),
        t("tab_explorer", lang),
        t("tab_about", lang),
    ])

    # ==========================================================================
    # TAB 1: CHECK A SKIN SPOT (SIMPLE & EXPERT MODES)
    # ==========================================================================
    with tabs[0]:
        # ----------------------------------------------------------------------
        # RECENT CHECKS STRIP (SESSION HISTORY)
        # ----------------------------------------------------------------------
        if st.session_state["history"]:
            st.markdown(f"##### {t('history_title', lang)}")
            h_cols = st.columns(len(st.session_state["history"]))
            for idx, item in enumerate(st.session_state["history"]):
                with h_cols[idx]:
                    st.caption(f"{item['time']} • **{item['pred_class']}** ({item['conf']*100:.0f}%)")
                    if item.get("thumb"):
                        st.image(item["thumb"], width=70)
                    if st.button(f"{t('restore_btn', lang)} #{idx+1}", key=f"hist_btn_{idx}", use_container_width=True):
                        st.session_state["active_c_img"] = item["c_img"]
                        st.session_state["active_d_img"] = item["d_img"]
                        st.session_state["active_age"] = item.get("age", 50)
                        st.session_state["active_sex"] = item.get("sex", "unknown")
                        st.session_state["active_site"] = item.get("site", "unknown")
                        st.session_state["active_gt"] = item.get("gt_diagnosis")
                        st.session_state["example_name"] = item.get("example_name")
                        st.rerun()
            st.markdown("---")

        # ----------------------------------------------------------------------
        # SIMPLE MODE: GUIDED 3-STEP FLOW
        # ----------------------------------------------------------------------
        if st.session_state["app_mode"] == "Simple":

            # EMPTY STATE BANNER & EXAMPLES (when no photos uploaded)
            if st.session_state["active_c_img"] is None and st.session_state["active_d_img"] is None:
                if not st.session_state["dismiss_banner"]:
                    with st.container():
                        st.markdown(f"""
                        <div style="background-color:#eff6ff; border:1px solid #bfdbfe; border-radius:10px; padding:18px 24px; margin-bottom:18px;">
                            <h4 style="color:#1e40af; margin-top:0;">{t('how_it_works_title', lang)}</h4>
                            <p style="color:#1e3a8a; margin-bottom:6px;">{t('how_it_works_step1', lang)}</p>
                            <p style="color:#1e3a8a; margin-bottom:6px;">{t('how_it_works_step2', lang)}</p>
                            <p style="color:#1e3a8a; margin-bottom:12px;">{t('how_it_works_step3', lang)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button(t("dismiss_banner", lang), key="btn_dismiss_guide"):
                            st.session_state["dismiss_banner"] = True
                            st.rerun()

                st.markdown(f"#### {t('try_example_title', lang)}")
                ex_col1, ex_col2, ex_col3 = st.columns(3)

                # Example 1: Melanoma (Tone G3)
                with ex_col1:
                    if st.button(t("example_mel", lang), use_container_width=True):
                        ex_data = load_example_lesion("IL_0192026", folds_df, meta_df)
                        if ex_data:
                            st.session_state["active_c_img"] = ex_data["c_img"]
                            st.session_state["active_d_img"] = ex_data["d_img"]
                            st.session_state["active_age"] = int(ex_data["age"])
                            st.session_state["active_sex"] = ex_data["sex"]
                            st.session_state["active_site"] = ex_data["site"]
                            st.session_state["active_gt"] = ex_data["gt_diagnosis"]
                            st.session_state["active_tone"] = ex_data["tone_group"]
                            st.session_state["example_name"] = "Melanoma (MEL, Tone G3)"
                            st.rerun()

                # Example 2: BCC (Tone G1)
                with ex_col2:
                    if st.button(t("example_bcc", lang), use_container_width=True):
                        ex_data = load_example_lesion("IL_0000652", folds_df, meta_df)
                        if ex_data:
                            st.session_state["active_c_img"] = ex_data["c_img"]
                            st.session_state["active_d_img"] = ex_data["d_img"]
                            st.session_state["active_age"] = int(ex_data["age"])
                            st.session_state["active_sex"] = ex_data["sex"]
                            st.session_state["active_site"] = ex_data["site"]
                            st.session_state["active_gt"] = ex_data["gt_diagnosis"]
                            st.session_state["active_tone"] = ex_data["tone_group"]
                            st.session_state["example_name"] = "Basal Cell Carcinoma (BCC, Tone G1)"
                            st.rerun()

                # Example 3: Normal Mole (Tone G2)
                with ex_col3:
                    if st.button(t("example_nv", lang), use_container_width=True):
                        ex_data = load_example_lesion("IL_0047878", folds_df, meta_df)
                        if ex_data:
                            st.session_state["active_c_img"] = ex_data["c_img"]
                            st.session_state["active_d_img"] = ex_data["d_img"]
                            st.session_state["active_age"] = int(ex_data["age"])
                            st.session_state["active_sex"] = ex_data["sex"]
                            st.session_state["active_site"] = ex_data["site"]
                            st.session_state["active_gt"] = ex_data["gt_diagnosis"]
                            st.session_state["active_tone"] = ex_data["tone_group"]
                            st.session_state["example_name"] = "Normal Mole (NV, Tone G2)"
                            st.rerun()

                st.markdown("---")

            # ------------------------------------------------------------------
            # STEP 1: ADD PHOTOS
            # ------------------------------------------------------------------
            st.markdown(f'<div class="simple-step-header">{t("step1_title", lang)}</div>', unsafe_allow_html=True)
            st.markdown(t("step1_desc", lang))

            # Photo Tips Expander
            with st.expander(t("photo_tips_title", lang), expanded=False):
                st.markdown(t("photo_tips_content", lang))

            use_camera = st.toggle(t("take_photo_btn", lang), value=False, help="Switch between uploading an image file or taking a live picture with your device's camera.")

            col_p1, col_p2 = st.columns(2)
            uploaded_c = None
            uploaded_d = None

            with col_p1:
                st.markdown(f"**{t('label_closeup', lang)}**")
                if use_camera:
                    c_cam = st.camera_input(t("camera_closeup", lang), key="cam_c", help=t("help_closeup", lang))
                    if c_cam:
                        uploaded_c = Image.open(c_cam)
                else:
                    c_file = st.file_uploader(
                        t("label_closeup", lang),
                        type=["jpg", "jpeg", "png"],
                        key="file_c",
                        help=t("help_closeup", lang),
                        label_visibility="collapsed",
                    )
                    if c_file:
                        try:
                            uploaded_c = Image.open(c_file)
                        except Exception as e:
                            st.error(f"Could not read image: {e}")

                if uploaded_c:
                    st.session_state["active_c_img"] = uploaded_c

            with col_p2:
                st.markdown(f"**{t('label_dermo', lang)}**")
                if use_camera:
                    d_cam = st.camera_input(t("camera_dermo", lang), key="cam_d", help=t("help_dermo", lang))
                    if d_cam:
                        uploaded_d = Image.open(d_cam)
                else:
                    d_file = st.file_uploader(
                        t("label_dermo", lang),
                        type=["jpg", "jpeg", "png"],
                        key="file_d",
                        help=t("help_dermo", lang),
                        label_visibility="collapsed",
                    )
                    if d_file:
                        try:
                            uploaded_d = Image.open(d_file)
                        except Exception as e:
                            st.error(f"Could not read image: {e}")

                if uploaded_d:
                    st.session_state["active_d_img"] = uploaded_d

            # Visual Preview & Quality Checks
            c_img_active = st.session_state.get("active_c_img")
            d_img_active = st.session_state.get("active_d_img")

            if c_img_active or d_img_active:
                prev_c1, prev_c2 = st.columns(2)
                with prev_c1:
                    if c_img_active:
                        st.image(c_img_active, caption="Close-up View", use_container_width=True)
                        ok_c, warns_c = check_image_quality(c_img_active, app_cfg, lang)
                        for w in warns_c:
                            st.warning(w)
                    else:
                        st.caption("No close-up provided (magnified photo will be used for both inputs).")
                with prev_c2:
                    if d_img_active:
                        st.image(d_img_active, caption="Magnified (Dermoscope) View", use_container_width=True)
                        ok_d, warns_d = check_image_quality(d_img_active, app_cfg, lang)
                        for w in warns_d:
                            st.warning(w)
                    else:
                        st.caption("No magnified photo provided (close-up photo will be used for both inputs).")

                # Single Photo Notice
                if (c_img_active is None and d_img_active is not None) or (c_img_active is not None and d_img_active is None):
                    st.info(t("single_photo_notice", lang))

            # ------------------------------------------------------------------
            # STEP 2: PATIENT DETAILS (OPTIONAL)
            # ------------------------------------------------------------------
            st.markdown(f'<div class="simple-step-header">{t("step2_title", lang)}</div>', unsafe_allow_html=True)
            st.markdown(t("step2_desc", lang))

            col_pt1, col_pt2, col_pt3 = st.columns(3)

            # Age Slider + Don't Know Checkbox
            with col_pt1:
                dont_know_age = st.checkbox(
                    t("dont_know_age", lang),
                    value=st.session_state.get("active_dont_know_age", False),
                    help=t("help_age", lang),
                )
                st.session_state["active_dont_know_age"] = dont_know_age

                if dont_know_age:
                    age_val = None
                    st.caption("Age: Unspecified / Missing")
                else:
                    age_val = float(st.slider(
                        t("label_age", lang),
                        min_value=1,
                        max_value=100,
                        value=int(st.session_state.get("active_age", 50)),
                        help=t("help_age", lang),
                    ))
                    st.session_state["active_age"] = age_val

            # Sex Radio / Segmented Control
            with col_pt2:
                sex_options = [t("sex_male", lang), t("sex_female", lang), t("sex_unknown", lang)]
                current_sex_raw = st.session_state.get("active_sex", "unknown")
                if current_sex_raw == "male":
                    s_idx = 0
                elif current_sex_raw == "female":
                    s_idx = 1
                else:
                    s_idx = 2

                chosen_sex_str = st.radio(
                    t("label_sex", lang),
                    sex_options,
                    index=s_idx,
                    help=t("help_sex", lang),
                    horizontal=True,
                )
                if chosen_sex_str == sex_options[0]:
                    sex_val = "male"
                elif chosen_sex_str == sex_options[1]:
                    sex_val = "female"
                else:
                    sex_val = "unknown"
                st.session_state["active_sex"] = sex_val

            # Body Site Friendly Picker with Emojis
            with col_pt3:
                site_map = {
                    "trunk": t("site_trunk", lang),
                    "head_neck_face": t("site_head_neck", lang),
                    "lower_extremity": t("site_lower_ext", lang),
                    "upper_extremity": t("site_upper_ext", lang),
                    "hand": t("site_hand", lang),
                    "foot": t("site_foot", lang),
                    "genital": t("site_genital", lang),
                    "unknown": t("site_unknown", lang),
                }
                inv_site_map = {v: k for k, v in site_map.items()}

                site_options = list(site_map.values())
                curr_site_key = st.session_state.get("active_site", "unknown")
                curr_site_display = site_map.get(curr_site_key, site_map["unknown"])
                site_idx = site_options.index(curr_site_display) if curr_site_display in site_options else 7

                chosen_site_display = st.selectbox(
                    t("label_site", lang),
                    site_options,
                    index=site_idx,
                    help=t("help_site", lang),
                )
                site_val = inv_site_map[chosen_site_display]
                st.session_state["active_site"] = site_val

            # ------------------------------------------------------------------
            # STEP 3: RESULT IN PLAIN LANGUAGE (AUTO-RUN ON INPUT CHANGE)
            # ------------------------------------------------------------------
            st.markdown(f'<div class="simple-step-header">{t("step3_title", lang)}</div>', unsafe_allow_html=True)

            img_for_clin = c_img_active if c_img_active else d_img_active
            img_for_dermo = d_img_active if d_img_active else c_img_active

            if img_for_clin is not None and img_for_dermo is not None:
                # Tabular vector encoding
                tabular_tensor = encode_tabular_vector(age_val, sex_val, site_val)

                # Input Hash for Instant Caching
                input_hash = compute_input_hash(img_for_clin, img_for_dermo, age_val, sex_val, site_val)

                if input_hash in st.session_state["cache"]:
                    res = st.session_state["cache"][input_hash]
                else:
                    # Friendly multi-step progress indicators
                    with st.status(t("progress_looking", lang), expanded=True) as status_box:
                        st.write(t("progress_looking", lang))
                        status_box.update(label=t("progress_models", lang), state="running")
                        st.write(t("progress_models", lang))

                        probs, ok_models = predict_ensemble(
                            "two_image_tabular_fusion",
                            img_for_clin,
                            img_for_dermo,
                            tabular_tensor,
                            device,
                            [0, 1, 2, 3, 4],
                        )

                        status_box.update(label=t("progress_preparing", lang), state="running")
                        st.write(t("progress_preparing", lang))

                        top_idx = int(np.argmax(probs))
                        pred_class = MILK10K_IDX_TO_CLASS[top_idx]
                        conf = float(probs[top_idx])

                        # Top-3 Differential List
                        sorted_indices = np.argsort(probs)[::-1]
                        top3 = [{"code": MILK10K_IDX_TO_CLASS[i], "prob": float(probs[i])} for i in sorted_indices[:3]]

                        # Calculate malignant + premalignant combined probability
                        mal_pre_prob = sum(float(probs[MILK10K_CLASS_TO_IDX[c]]) for c in MALIGNANT_OR_PREMALIGNANT_CLASSES if c in MILK10K_CLASS_TO_IDX)
                        is_high_risk = mal_pre_prob >= float(app_cfg.get("high_risk_threshold", 0.35)) or conf < float(app_cfg.get("confidence_medium", 0.40))

                        res = {
                            "pred_class": pred_class,
                            "conf": conf,
                            "probs": probs,
                            "top3": top3,
                            "mal_pre_prob": mal_pre_prob,
                            "is_high_risk": is_high_risk,
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

                        # Save to session history (up to max_history)
                        thumb_img = img_for_dermo.copy()
                        thumb_img.thumbnail((80, 80))
                        hist_item = {
                            "time": datetime.now().strftime("%H:%M"),
                            "pred_class": pred_class,
                            "conf": conf,
                            "thumb": thumb_img,
                            "c_img": img_for_clin,
                            "d_img": img_for_dermo,
                            "age": age_val,
                            "sex": sex_val,
                            "site": site_val,
                            "gt_diagnosis": st.session_state.get("active_gt"),
                            "example_name": st.session_state.get("example_name"),
                        }
                        # Prepend to history avoiding duplicates
                        st.session_state["history"] = [hist_item] + [h for h in st.session_state["history"] if h.get("pred_class") != pred_class or h.get("time") != hist_item["time"]][:int(app_cfg.get("max_history", 5))-1]

                # Render Step 3 Result Card
                pred_c = res["pred_class"]
                conf_val = res["conf"]
                c_meta = CONDITION_DESCRIPTIONS.get(pred_c, {})
                display_name = c_meta.get(f"{lang}_name", c_meta.get("en_name", pred_c))
                display_desc = c_meta.get(f"{lang}_desc", c_meta.get("en_desc", ""))
                cat_type = c_meta.get("category", "benign")

                is_mal = cat_type in ["malignant", "premalignant"]
                card_bg = "#fef2f2" if is_mal else "#f0fdf4"
                card_border = "#ef4444" if is_mal else "#10b981"
                card_badge = "🚨 " + t("most_likely", lang).upper() if is_mal else "✅ " + t("most_likely", lang).upper()

                # Big Plain-Language Card
                st.markdown(f"""
                <div style="background:{card_bg}; border: 2px solid {card_border}; border-radius:12px; padding:22px 26px; margin-bottom:18px;">
                    <span style="font-weight:700; color:{card_border}; font-size:0.85rem; letter-spacing:0.5px;">{card_badge}</span>
                    <div style="font-size:1.75rem; font-weight:800; color:#0f172a; margin-top:2px;">
                        {display_name} <span style="font-size:1.1rem; color:#64748b; font-weight:600;">({pred_c})</span>
                    </div>
                    <p style="font-size:1.0rem; color:#334155; margin-top:6px; margin-bottom:12px; line-height:1.5;">
                        {display_desc}
                    </p>
                    <div style="font-size:0.85rem; color:#64748b;">
                        {t('ai_estimate_tag', lang)}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Confidence meter in words + progress bar
                conf_word = get_confidence_label(conf_val, app_cfg, lang)
                c_col1, c_col2 = st.columns([1, 2])
                with c_col1:
                    st.metric(
                        t("confidence_level", lang),
                        f"{conf_val*100:.1f}%",
                        f"{conf_word}",
                        delta_color="normal" if conf_val >= 0.70 else "off",
                    )
                with c_col2:
                    st.markdown(f"**{t('confidence_level', lang)}: {conf_word}**")
                    st.progress(float(min(max(conf_val, 0.0), 1.0)))

                # High-Risk Alert Banner (if malignant probability is elevated or confidence is low)
                if res.get("is_high_risk"):
                    st.error(t("clinical_review_banner", lang))

                # What you can do next
                st.markdown(f"#### 💡 {t('next_steps_title', lang)}")
                st.info(get_next_steps(pred_c, res.get("is_high_risk", False), lang))

                # Ranked Top-3 Possibilities
                st.markdown(f"#### 📊 {t('top3_title', lang)}")
                top3_df = pd.DataFrame({
                    "Condition": [
                        f"{item['code']} - {CONDITION_DESCRIPTIONS.get(item['code'], {}).get(f'{lang}_name', item['code'])}"
                        for item in res["top3"]
                    ],
                    "Probability (%)": [item["prob"] * 100 for item in res["top3"]],
                }).set_index("Condition")
                st.bar_chart(top3_df, height=190)

                # Action Buttons: Re-run & Download PDF
                act_col1, act_col2 = st.columns(2)
                with act_col1:
                    if st.button(t("rerun_btn", lang), use_container_width=True):
                        st.session_state["cache"].pop(input_hash, None)
                        st.rerun()
                with act_col2:
                    pdf_bytes = generate_pdf_report(res, img_for_clin, img_for_dermo, lang)
                    st.download_button(
                        label=t("download_pdf_btn", lang),
                        data=pdf_bytes,
                        file_name=f"VISHWAS_Skin_Check_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

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

            else:
                st.caption("Please upload a photo in Step 1 to generate your instant AI skin assessment.")

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
                tone_group_val: str = "unknown"
                skin_tone_val: int = 3
                sample_age = 55.0
                sample_sex = "male"
                sample_site = "trunk"

                if input_mode == "Select from Dataset Gallery":
                    if not folds_df.empty and not meta_df.empty:
                        filter_col1, filter_col2 = st.columns(2)
                        with filter_col1:
                            diag_filter = st.selectbox("Filter Diagnosis:", ["All Diagnoses"] + MILK10K_CLASSES)
                        with filter_col2:
                            tone_filter = st.selectbox("Filter Tone Group:", ["All Tone Groups", "G1 = {0,1,2}", "G2 = {3}", "G3 = {4,5}"])

                        filtered = folds_df.copy()
                        if diag_filter != "All Diagnoses":
                            filtered = filtered[filtered["diagnosis"] == diag_filter]
                        if tone_filter != "All Tone Groups":
                            t_key = tone_filter.split(" ")[0]
                            filtered = filtered[filtered["tone_group"] == t_key]

                        if not filtered.empty:
                            lesion_options = [
                                f"{row['lesion_id']} | {row['diagnosis']} | Tone: {row['tone_group']} (Fold {row['fold']})"
                                for _, row in filtered.head(50).iterrows()
                            ]
                            chosen_lesion_str = st.selectbox("Pick a Sample Lesion:", lesion_options)
                            chosen_lid = chosen_lesion_str.split(" ")[0]

                            lesion_row = folds_df[folds_df["lesion_id"] == chosen_lid].iloc[0]
                            gt_diagnosis = lesion_row["diagnosis"]
                            tone_group_val = lesion_row["tone_group"]
                            skin_tone_val = int(lesion_row["skin_tone_class"])

                            meta_lesion = meta_df[meta_df["lesion_id"] == chosen_lid]
                            clin_row = meta_lesion[meta_lesion["image_type"] == "clinical: close-up"]
                            dermo_row = meta_lesion[meta_lesion["image_type"] == "dermoscopic"]

                            sample_age = float(clin_row.iloc[0]["age_approx"]) if not clin_row.empty and pd.notna(clin_row.iloc[0]["age_approx"]) else 50.0
                            sample_sex = str(clin_row.iloc[0]["sex"]) if not clin_row.empty and pd.notna(clin_row.iloc[0]["sex"]) else "unknown"
                            sample_site = str(clin_row.iloc[0]["site"]) if not clin_row.empty and pd.notna(clin_row.iloc[0]["site"]) else "unknown"

                            cache_128 = VISHWAS_ROOT / "data" / "cache" / "128"
                            if not clin_row.empty and cache_128.exists():
                                c_path = cache_128 / f"{clin_row.iloc[0]['isic_id']}.jpg"
                                if c_path.exists():
                                    selected_c_img = Image.open(c_path)
                            if not dermo_row.empty and cache_128.exists():
                                d_path = cache_128 / f"{dermo_row.iloc[0]['isic_id']}.jpg"
                                if d_path.exists():
                                    selected_d_img = Image.open(d_path)
                        else:
                            st.warning("No samples match this filter.")
                    else:
                        st.info("Dataset metadata file not found; use upload mode.")

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

                # Display images
                img_c1, img_c2 = st.columns(2)
                with img_c1:
                    st.markdown("**Clinical Close-up View**")
                    if selected_c_img:
                        st.image(selected_c_img, use_container_width=True)
                    else:
                        st.caption("No clinical image selected.")
                with img_c2:
                    st.markdown("**Dermoscopic Microscopic View**")
                    if selected_d_img:
                        st.image(selected_d_img, use_container_width=True)
                    else:
                        st.caption("No dermoscopic image selected.")

            with col_input_right:
                st.markdown("#### 2. Inference Configuration & Execution")

                model_choice = st.radio(
                    "Model Architecture:",
                    [
                        "Compare Both Models (Two-Image Fusion vs. Dermoscopic Only)",
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

                tg_label = TONE_GROUP_INFO.get(tone_group_val, {}).get("label", tone_group_val)
                st.markdown(f"""
                <div style="background-color:#f8fafc; border-left:5px solid #2b6cb0; padding:14px 18px; border-radius:8px; margin-bottom:12px;">
                    <b>Patient Clinical Profile</b><br>
                    • <b>Tone Group</b>: <code>{tg_label}</code> (skin_tone_class: {skin_tone_val})<br>
                    • <b>Demographics</b>: Age {sample_age:.0f} | Sex: {sample_sex.capitalize()} | Site: {sample_site}<br>
                    • <b>Ground Truth Diagnosis</b>: <code>{gt_diagnosis if gt_diagnosis else 'External / Unknown'}</code>
                </div>
                """, unsafe_allow_html=True)

                run_clicked = st.button("🚀 Run Diagnostic Inference", type="primary", use_container_width=True)

                if run_clicked or (selected_c_img and selected_d_img):
                    if not selected_d_img:
                        st.error("Please provide at least a dermoscopic image to run inference.")
                    else:
                        c_img_to_use = selected_c_img if selected_c_img else selected_d_img

                        with st.spinner("Executing neural inference on Apple Silicon MPS/CPU..."):
                            if "Compare Both" in model_choice:
                                probs_fusion, _ = predict_ensemble("two_image_tabular_fusion", c_img_to_use, selected_d_img, tabular_tensor, device, folds_to_eval)
                                probs_dermo, _ = predict_ensemble("dermoscopic_only", c_img_to_use, selected_d_img, tabular_tensor, device, folds_to_eval)

                                top_f_idx = int(np.argmax(probs_fusion))
                                top_d_idx = int(np.argmax(probs_dermo))
                                pred_f_class = MILK10K_IDX_TO_CLASS[top_f_idx]
                                pred_d_class = MILK10K_IDX_TO_CLASS[top_d_idx]

                                res_col1, res_col2 = st.columns(2)
                                with res_col1:
                                    st.markdown("##### 🔬 Two-Image + Tabular Fusion (`d_cw_128`)")
                                    is_correct_f = (pred_f_class == gt_diagnosis) if gt_diagnosis else None
                                    corr_badge_f = " ✅ Match" if is_correct_f else (" ❌ Diff" if is_correct_f is False else "")
                                    st.markdown(f"**Predicted**: `{pred_f_class}` ({probs_fusion[top_f_idx]*100:.1f}% confidence){corr_badge_f}")
                                    st.caption(f"{CLASS_INFO[pred_f_class]['name']} ({CLASS_INFO[pred_f_class]['type']})")

                                with res_col2:
                                    st.markdown("##### 🔍 Dermoscopic Only Baseline (`a_cw_128`)")
                                    is_correct_d = (pred_d_class == gt_diagnosis) if gt_diagnosis else None
                                    corr_badge_d = " ✅ Match" if is_correct_d else (" ❌ Diff" if is_correct_d is False else "")
                                    st.markdown(f"**Predicted**: `{pred_d_class}` ({probs_dermo[top_d_idx]*100:.1f}% confidence){corr_badge_d}")
                                    st.caption(f"{CLASS_INFO[pred_d_class]['name']} ({CLASS_INFO[pred_d_class]['type']})")

                                st.markdown("##### Comparative Probability Distribution Across Diagnoses")
                                comp_df = pd.DataFrame({
                                    "Diagnosis": MILK10K_CLASSES,
                                    "Multimodal Fusion": probs_fusion,
                                    "Dermoscopic Only": probs_dermo,
                                }).set_index("Diagnosis")
                                st.bar_chart(comp_df)

                            else:
                                m_type = "two_image_tabular_fusion" if "Fusion" in model_choice else "dermoscopic_only"
                                probs, _ = predict_ensemble(m_type, c_img_to_use, selected_d_img, tabular_tensor, device, folds_to_eval)
                                top_idx = int(np.argmax(probs))
                                pred_class = MILK10K_IDX_TO_CLASS[top_idx]
                                conf = probs[top_idx]

                                is_correct = (pred_class == gt_diagnosis) if gt_diagnosis else None
                                corr_badge = " ✅ Ground Truth Match!" if is_correct else (" ❌ Different from Ground Truth" if is_correct is False else "")

                                st.success(f"**Predicted Diagnosis**: `{pred_class}` ({conf*100:.1f}% confidence){corr_badge}")
                                st.markdown(f"**Condition**: *{CLASS_INFO[pred_class]['name']}* ({CLASS_INFO[pred_class]['type']})")

                                st.markdown("##### Full Class Probabilities")
                                prob_df = pd.DataFrame({"Diagnosis": MILK10K_CLASSES, "Probability": probs}).set_index("Diagnosis")
                                st.bar_chart(prob_df)

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

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Headline Macro-F1 (Multimodal)", "0.3773", "+0.0190 vs Baseline", delta_color="normal")
        with col_m2:
            st.metric("Overall Accuracy (Multimodal)", "47.71%", "+2.58% vs Baseline", delta_color="normal")
        with col_m3:
            st.metric("Demographic Disparity Gap", "0.0461", "-0.0271 vs Baseline (37% compression)", delta_color="normal")
        with col_m4:
            st.metric("Worst-Group F1 (G2)", "0.3611", "+0.0248 vs Baseline", delta_color="normal")

        st.markdown("---")
        st.markdown("### 1. Primary Model Comparison")

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

        with st.expander("💡 What does this comparison table mean?", expanded=False):
            st.markdown(t("chart_explain_fairness", lang))

        st.markdown("### 2. Subgroup Performance Across Tone Groups (`skin_tone_class`)")

        g_col1, g_col2, g_col3 = st.columns(3)
        with g_col1:
            st.markdown("""
            <div style="background-color:#f8fafc; border-left:5px solid #2b6cb0; padding:14px 18px; border-radius:8px;">
                <b>Tone Group: G1 = {0, 1, 2}</b><br>
                <i>Evaluated lesions: n = 617</i><br><br>
                • <b>Multimodal Fusion F1</b>: <code>0.4072</code> [0.3512, 0.4534]<br>
                • <b>Dermoscopic Only F1</b>: <code>0.4094</code> [0.3550, 0.4587]<br>
                • <b>Multimodal Accuracy</b>: 51.05%<br>
                • <b>Calibration ECE</b>: 0.0542
            </div>
            """, unsafe_allow_html=True)

        with g_col2:
            st.markdown("""
            <div style="background-color:#f8fafc; border-left:5px solid #2b6cb0; padding:14px 18px; border-radius:8px;">
                <b>Tone Group: G2 = {3}</b><br>
                <i>Evaluated lesions: n = 3,174 (Worst Group)</i><br><br>
                • <b>Multimodal Fusion F1</b>: <code>0.3611</code> [0.3386, 0.3809]<br>
                • <b>Dermoscopic Only F1</b>: <code>0.3363</code> [0.3146, 0.3580]<br>
                • <b>Multimodal Accuracy</b>: 47.48%<br>
                • <b>Calibration ECE</b>: 0.1122
            </div>
            """, unsafe_allow_html=True)

        with g_col3:
            st.markdown("""
            <div style="background-color:#f8fafc; border-left:5px solid #2b6cb0; padding:14px 18px; border-radius:8px;">
                <b>Tone Group: G3 = {4, 5}</b><br>
                <i>Evaluated lesions: n = 1,449</i><br><br>
                • <b>Multimodal Fusion F1</b>: <code>0.3662</code> [0.3337, 0.3970]<br>
                • <b>Dermoscopic Only F1</b>: <code>0.3422</code> [0.3094, 0.3699]<br>
                • <b>Multimodal Accuracy</b>: 46.79%<br>
                • <b>Calibration ECE</b>: 0.0759
            </div>
            """, unsafe_allow_html=True)

        # Visual Disparity Comparison Bar
        st.markdown("#### Subgroup Headline Macro-F1 Parity Visualization")
        chart_df = pd.DataFrame({
            "Tone Group": ["G1 = {0,1,2} (n=617)", "G2 = {3} (n=3,174)", "G3 = {4,5} (n=1,449)"],
            "Multimodal Tabular Fusion (d_cw_128)": [0.4072, 0.3611, 0.3662],
            "Dermoscopic Baseline (a_cw_128)": [0.4094, 0.3363, 0.3422],
        }).set_index("Tone Group")
        st.bar_chart(chart_df)

        with st.expander("💡 What does this subgroup parity chart mean?", expanded=False):
            st.markdown(t("chart_explain_fairness", lang))

        st.info(
            "💡 **Key Fairness Finding**: The baseline dermoscopic model suffered a severe performance drop on G2 (F1: 0.3363) and G3 (F1: 0.3422), "
            "leading to a wide disparity gap of 0.0732. Multimodal fusion with clinical metadata lifted G2 to 0.3611 and G3 to 0.3662, "
            "successfully compressing the demographic gap by 37% down to 0.0461."
        )

        st.markdown("### 3. Per-Class Diagnostic Recall Comparison")
        rec_data = {
            "Diagnosis": MILK10K_CLASSES,
            "Description": [CLASS_INFO[c]["name"] for c in MILK10K_CLASSES],
            "Fusion Recall (d_cw_128)": [54.46, 41.20, 34.09, 34.74, 50.00, 40.00, 0.00, 46.00, 70.51, 58.35, 78.72],
            "Dermoscopic Recall (a_cw_128)": [51.49, 39.10, 31.82, 27.39, 48.08, 26.00, 0.00, 44.89, 67.43, 58.99, 80.85],
        }
        st.dataframe(pd.DataFrame(rec_data).set_index("Diagnosis"), use_container_width=True)

    # ==========================================================================
    # TAB 3: HOW RELIABLE IS IT? (CALIBRATION)
    # ==========================================================================
    with tabs[2]:
        st.subheader("📈 Model Calibration & Diagnostic Confusion Matrices")
        st.markdown(
            "Expected Calibration Error (ECE) evaluates whether predicted confidence aligns with true diagnostic accuracy. "
            "Confidence intervals are bootstrapped over 1,000 resamples of lesions."
        )

        rep_d = VISHWAS_ROOT / "reports" / "d_cw_128"
        rep_a = VISHWAS_ROOT / "reports" / "a_cw_128"

        st.markdown("#### 1. Reliability Diagrams (Calibration Curves Across Tone Groups)")
        rc_c1, rc_c2 = st.columns(2)
        with rc_c1:
            st.markdown("##### Two-Image Tabular Fusion (`d_cw_128`) - Overall ECE: 0.0911")
            rc_p1 = rep_d / "reliability_curves.png"
            if rc_p1.exists():
                st.image(str(rc_p1), use_container_width=True)
            else:
                st.warning("Reliability curve image not found.")

        with rc_c2:
            st.markdown("##### Dermoscopic Only (`a_cw_128`) - Overall ECE: 0.0857")
            rc_p2 = rep_a / "reliability_curves.png"
            if rc_p2.exists():
                st.image(str(rc_p2), use_container_width=True)
            else:
                st.warning("Reliability curve image not found.")

        with st.expander("💡 What does model calibration mean?", expanded=False):
            st.markdown(t("chart_explain_calibration", lang))

        st.markdown("#### 2. Confusion Matrices (11 Diagnostic Classes)")
        cm_c1, cm_c2 = st.columns(2)
        with cm_c1:
            st.markdown("##### Two-Image Tabular Fusion (`d_cw_128`)")
            cm_p1 = rep_d / "confusion_matrix.png"
            if cm_p1.exists():
                st.image(str(cm_p1), use_container_width=True)
            else:
                st.warning("Confusion matrix image not found.")

        with cm_c2:
            st.markdown("##### Dermoscopic Only (`a_cw_128`)")
            cm_p2 = rep_a / "confusion_matrix.png"
            if cm_p2.exists():
                st.image(str(cm_p2), use_container_width=True)
            else:
                st.warning("Confusion matrix image not found.")

        with st.expander("💡 What does this confusion matrix mean?", expanded=False):
            st.markdown(t("chart_explain_confusion", lang))

    # ==========================================================================
    # TAB 4: DATA EXPLORER
    # ==========================================================================
    with tabs[3]:
        st.subheader("🗂️ MILK10k 5-Fold Cross-Validation Stratification Explorer")
        st.markdown(
            "Every lesion contains exactly two views (clinical close-up and dermoscopic) which are **strictly grouped together** "
            "within the same fold to prevent data leakage. Fold sizes are balanced to exactly 1,048 lesions each."
        )

        if not folds_df.empty:
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                st.metric("Total Unique Lesions", f"{len(folds_df):,}")
            with f_col2:
                st.metric("Total Image Pairs", f"{len(folds_df) * 2:,} images")
            with f_col3:
                st.metric("Lesions per Fold", "1,048 exactly")

            st.markdown("#### Cross-Validation Lesions by Fold and Tone Group")
            ct = pd.crosstab(folds_df["fold"], folds_df["tone_group"])
            st.dataframe(ct, use_container_width=True)

            st.markdown("#### Browse Lesion Fold Mappings")
            st.dataframe(folds_df, use_container_width=True, height=350)
        else:
            st.info("Fold mapping file `data/milk10k_folds.csv` not found.")

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


if __name__ == "__main__":
    main()
