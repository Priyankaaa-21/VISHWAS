"""
Vishwas (विश्वास) Interactive Diagnostic & Fairness Auditing Platform
Streamlit Application: src/app/app.py

Features:
1. Interactive Multimodal Dermatological Diagnosis (Clinical + Dermoscopic + Tabular Patient Metadata)
2. Side-by-Side Model Comparison (Two-Image Tabular Fusion vs. Dermoscopic Only Baseline)
3. 5-Fold Ensemble or Single-Fold Checkpoint Inference
4. Comprehensive Demographic Subgroup Fairness Auditing (G1 = {0,1,2}, G2 = {3}, G3 = {4,5})
5. Calibration Analysis (ECE & Reliability Diagrams) and Confusion Matrices
6. 5-Fold Stratification & Dataset Explorer
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import streamlit as st
import torch
import torchvision.transforms as transforms

# Set page config as first Streamlit call
st.set_page_config(
    page_title="Vishwas - Skin-Tone-Aware Multimodal Skin AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Resolve workspace directories
APP_DIR = Path(__file__).resolve().parent
VISHWAS_ROOT = APP_DIR.parent.parent
if str(VISHWAS_ROOT) not in sys.path:
    sys.path.insert(0, str(VISHWAS_ROOT))

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
    "G1": {"label": "G1 = {0, 1, 2}", "desc": "skin_tone_class values 0, 1, 2", "color": "#2ca02c"},
    "G2": {"label": "G2 = {3}", "desc": "skin_tone_class value 3", "color": "#ff7f0e"},
    "G3": {"label": "G3 = {4, 5}", "desc": "skin_tone_class values 4, 5", "color": "#d62728"},
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
def load_dataset_metadata() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads folds mapping and image metadata for sample explorer."""
    folds_path = VISHWAS_ROOT / "data" / "milk10k_folds.csv"
    meta_path = VISHWAS_ROOT / "data" / "milk10k" / "MILK10k_Training_Metadata.csv"

    folds_df = pd.read_csv(folds_path) if folds_path.exists() else pd.DataFrame()
    meta_df = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
    return folds_df, meta_df


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
        st.error(f"Error loading {ckpt_path.name}: {e}")
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
) -> np.ndarray:
    """Averages softmax probabilities across the 5 cross-validation checkpoints."""
    all_probs = []
    for f in folds:
        m = load_checkpoint_model(model_type, f, str(device))
        if m is not None:
            p = predict_single_model(m, model_type, clinical_img, dermoscopic_img, tabular_tensor, device, img_size)
            all_probs.append(p)
    if not all_probs:
        return np.ones(11) / 11.0
    return np.mean(all_probs, axis=0)


# ==============================================================================
# UI RENDERING
# ==============================================================================

def main():
    # Custom CSS Styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 800;
            color: #1a365d;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #4a5568;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #f8fafc;
            border-left: 5px solid #2b6cb0;
            padding: 16px 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            margin-bottom: 12px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 16px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 48px;
            font-weight: 600;
            font-size: 1.0rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🩺 VISHWAS (विश्वास): Skin-Tone-Aware Multimodal Dermatological AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Certified 5-Fold Cross-Validation, Multimodal Fusion (Clinical + Dermoscopic + Tabular), and Rigorous Subgroup Fairness Auditing across confirmed <code>skin_tone_class</code> groups.</div>', unsafe_allow_html=True)

    device = get_device()
    folds_df, meta_df = load_dataset_metadata()

    tabs = st.tabs([
        "🔬 Interactive Diagnosis",
        "⚖️ Fairness Audit Benchmark",
        "📈 Calibration & Reliability",
        "🗂️ 5-Fold Dataset Explorer",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: INTERACTIVE DIAGNOSIS
    # --------------------------------------------------------------------------
    with tabs[0]:
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
                    # Filter gallery by tone group or diagnosis
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

                        # Find clinical and dermoscopic ISIC IDs
                        meta_lesion = meta_df[meta_df["lesion_id"] == chosen_lid]
                        clin_row = meta_lesion[meta_lesion["image_type"] == "clinical: close-up"]
                        dermo_row = meta_lesion[meta_lesion["image_type"] == "dermoscopic"]

                        sample_age = float(clin_row.iloc[0]["age_approx"]) if not clin_row.empty and pd.notna(clin_row.iloc[0]["age_approx"]) else 50.0
                        sample_sex = str(clin_row.iloc[0]["sex"]) if not clin_row.empty and pd.notna(clin_row.iloc[0]["sex"]) else "unknown"
                        sample_site = str(clin_row.iloc[0]["site"]) if not clin_row.empty and pd.notna(clin_row.iloc[0]["site"]) else "unknown"

                        # Load images from cache
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
                c_file = st.file_uploader("Upload Clinical Close-up Image", type=["jpg", "jpeg", "png"])
                d_file = st.file_uploader("Upload Dermoscopic Image", type=["jpg", "jpeg", "png"])
                if c_file:
                    selected_c_img = Image.open(c_file)
                if d_file:
                    selected_d_img = Image.open(d_file)

                # Custom clinical inputs
                m_c1, m_c2, m_c3 = st.columns(3)
                with m_c1:
                    sample_age = float(st.number_input("Patient Approximate Age:", min_value=0, max_value=110, value=55))
                with m_c2:
                    sample_sex = st.selectbox("Patient Sex:", SEX_CATEGORIES)
                with m_c3:
                    sample_site = st.selectbox("Anatomical Site:", SITE_CATEGORIES)

            # Display uploaded/selected images
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

            # Encode tabular
            tabular_tensor = encode_tabular_vector(sample_age, sample_sex, sample_site)

            # Patient context card
            tg_label = TONE_GROUP_INFO.get(tone_group_val, {}).get("label", tone_group_val)
            st.markdown(f"""
            <div class="metric-card">
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
                    # Fallback clinical image if only dermoscopic provided
                    c_img_to_use = selected_c_img if selected_c_img else selected_d_img

                    with st.spinner("Executing neural inference on Apple Silicon MPS/CPU..."):
                        if "Compare Both" in model_choice:
                            probs_fusion = predict_ensemble("two_image_tabular_fusion", c_img_to_use, selected_d_img, tabular_tensor, device, folds_to_eval)
                            probs_dermo = predict_ensemble("dermoscopic_only", c_img_to_use, selected_d_img, tabular_tensor, device, folds_to_eval)

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

                            # Side-by-side probability distribution
                            st.markdown("##### Comparative Probability Distribution Across Diagnoses")
                            comp_df = pd.DataFrame({
                                "Diagnosis": MILK10K_CLASSES,
                                "Multimodal Fusion": probs_fusion,
                                "Dermoscopic Only": probs_dermo,
                            }).set_index("Diagnosis")
                            st.bar_chart(comp_df)

                        else:
                            m_type = "two_image_tabular_fusion" if "Fusion" in model_choice else "dermoscopic_only"
                            probs = predict_ensemble(m_type, c_img_to_use, selected_d_img, tabular_tensor, device, folds_to_eval)
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

    # --------------------------------------------------------------------------
    # TAB 2: FAIRNESS AUDIT BENCHMARK
    # --------------------------------------------------------------------------
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

        st.markdown("### 2. Subgroup Performance Across Tone Groups (`skin_tone_class`)")

        g_col1, g_col2, g_col3 = st.columns(3)
        with g_col1:
            st.markdown("""
            <div class="metric-card">
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
            <div class="metric-card">
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
            <div class="metric-card">
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

    # --------------------------------------------------------------------------
    # TAB 3: CALIBRATION & RELIABILITY
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # TAB 4: DATASET & 5-FOLD EXPLORER
    # --------------------------------------------------------------------------
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


if __name__ == "__main__":
    main()
