"""
Pure Helper Functions for VISHWAS Streamlit Application
Contains:
1. Image Quality Verification (blur, brightness, resolution)
2. Traffic-Light Triage Verdict Calculation (Red, Amber, Green, Grey)
3. Confidence & OOF Empirical Evidence Line Computation
4. Horizontal Altair Ranked Top-3 Chart
5. Deterministic Input Caching Hashes
6. PDF Diagnostic Result Report Generator
"""

from __future__ import annotations

import hashlib
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import laplace

try:
    from src.app.strings import CONDITION_DESCRIPTIONS, STRINGS, t
except ImportError:
    from vishwas.src.app.strings import CONDITION_DESCRIPTIONS, STRINGS, t


# Default threshold values if not loaded from config
DEFAULT_APP_CONFIG: Dict[str, Any] = {
    "blur_threshold": 40.0,
    "min_brightness": 30.0,
    "max_brightness": 225.0,
    "min_resolution": 64,
    "confidence_high": 0.55,
    "confidence_medium": 0.30,
    "high_risk_threshold": 0.35,
    "default_language": "en",
    "default_theme": "Auto",
    "triage": {
        "malignant_risk_threshold": 0.35,
        "low_confidence_threshold": 0.30,
        "very_low_threshold": 0.20,
    },
}

MALIGNANT_OR_PREMALIGNANT_CLASSES = {"AKIEC", "BCC", "MAL_OTH", "MEL", "SCCKA"}


def check_image_quality(
    image: Image.Image,
    cfg: Optional[Dict[str, Any]] = None,
    lang: str = "en",
) -> Tuple[bool, List[str]]:
    """
    Evaluates image quality:
    - Blur: Variance of Laplacian
    - Brightness: Mean grayscale luminance
    - Resolution: Minimum width and height
    Returns (is_acceptable, list_of_warning_strings).
    """
    if cfg is None:
        cfg = DEFAULT_APP_CONFIG

    blur_thresh = float(cfg.get("blur_threshold", 40.0))
    min_bright = float(cfg.get("min_brightness", 30.0))
    max_bright = float(cfg.get("max_brightness", 225.0))
    min_res = int(cfg.get("min_resolution", 64))

    warnings: List[str] = []

    # 1. Resolution Check
    width, height = image.size
    if width < min_res or height < min_res:
        warnings.append(t("warn_low_res", lang) + f" ({width}x{height}px)")

    # 2. Grayscale Array for Luminance & Blur
    gray = np.array(image.convert("L"), dtype=float)

    # 3. Brightness Check
    mean_luminance = float(gray.mean())
    if mean_luminance < min_bright:
        warnings.append(t("warn_too_dark", lang) + f" (lum: {mean_luminance:.1f})")
    elif mean_luminance > max_bright:
        warnings.append(t("warn_too_bright", lang) + f" (lum: {mean_luminance:.1f})")

    # 4. Blur Check via Laplacian Variance
    lap_var = float(laplace(gray).var())
    if lap_var < blur_thresh:
        warnings.append(t("warn_blurry", lang) + f" (focus score: {lap_var:.1f})")

    is_acceptable = len(warnings) == 0
    return is_acceptable, warnings


def get_confidence_label(
    conf: float,
    cfg: Optional[Dict[str, Any]] = None,
    lang: str = "en",
) -> str:
    """Returns high / medium / low confidence text descriptor based on OOF distribution."""
    if cfg is None:
        cfg = DEFAULT_APP_CONFIG

    c_high = float(cfg.get("confidence_high", 0.55))
    c_med = float(cfg.get("confidence_medium", 0.30))

    if conf >= c_high:
        return t("conf_high", lang)
    elif conf >= c_med:
        return t("conf_med", lang)
    else:
        return t("conf_low", lang)


def compute_triage_verdict(
    pred_class: str,
    conf: float,
    malignant_prob: float,
    has_quality_warnings: bool = False,
    cfg: Optional[Dict[str, Any]] = None,
    lang: str = "en",
) -> Dict[str, Any]:
    """
    Traffic-light triage verdict (Red, Amber, Green, Grey).
    Never relies on colour alone (provides icons, title, text, action guidance).

    Rules:
    - Grey, "Can't tell, retake the photo": photo-quality warnings fired or top-1 prob < very_low_threshold (0.20).
    - Red, "See a doctor soon": top-1 is malignant or pre-malignant, OR malignant+pre-malignant prob >= threshold (0.35).
    - Amber, "Get it checked": top-1 is inflammatory/non-benign, OR confidence is low (< 0.30).
    - Green, "Likely harmless, keep an eye on it": top-1 is benign AND malignant prob < threshold AND confidence is not low.
    """
    if cfg is None:
        cfg = DEFAULT_APP_CONFIG

    triage_cfg = cfg.get("triage", {}) if isinstance(cfg.get("triage"), dict) else {}
    mal_thresh = float(triage_cfg.get("malignant_risk_threshold", 0.35))
    low_thresh = float(triage_cfg.get("low_confidence_threshold", 0.30))
    very_low_thresh = float(triage_cfg.get("very_low_threshold", 0.20))

    cond_meta = CONDITION_DESCRIPTIONS.get(pred_class, {})
    cat = cond_meta.get("category", "benign")

    # 1. Grey: photo quality issues or very low confidence
    if has_quality_warnings or conf < very_low_thresh:
        return {
            "level": "grey",
            "icon": "🔄",
            "title": t("triage_grey_title", lang),
            "text": t("triage_grey_text", lang),
            "badge_color": "#64748b",
            "border_color": "#94a3b8",
            "bg_color_light": "#f1f5f9",
            "bg_color_dark": "#1e293b",
            "text_color_light": "#334155",
            "text_color_dark": "#f8fafc",
            "is_uncertain": True,
        }

    # 2. Red: top-1 is malignant/premalignant OR malignant probability >= threshold
    if pred_class in MALIGNANT_OR_PREMALIGNANT_CLASSES or malignant_prob >= mal_thresh:
        return {
            "level": "red",
            "icon": "🚨",
            "title": t("triage_red_title", lang),
            "text": t("triage_red_text", lang),
            "badge_color": "#dc2626",
            "border_color": "#ef4444",
            "bg_color_light": "#fef2f2",
            "bg_color_dark": "#450a0a",
            "text_color_light": "#991b1b",
            "text_color_dark": "#fecaca",
            "is_uncertain": conf < low_thresh,
        }

    # 3. Amber: non-benign / inflammatory OR low confidence
    if cat != "benign" or conf < low_thresh:
        return {
            "level": "amber",
            "icon": "⚠️",
            "title": t("triage_amber_title", lang),
            "text": t("triage_amber_text", lang),
            "badge_color": "#d97706",
            "border_color": "#f59e0b",
            "bg_color_light": "#fffbeb",
            "bg_color_dark": "#451a03",
            "text_color_light": "#92400e",
            "text_color_dark": "#fde68a",
            "is_uncertain": conf < low_thresh,
        }

    # 4. Green: benign AND low malignant probability AND not low confidence
    return {
        "level": "green",
        "icon": "✅",
        "title": t("triage_green_title", lang),
        "text": t("triage_green_text", lang),
        "badge_color": "#16a34a",
        "border_color": "#22c55e",
        "bg_color_light": "#f0fdf4",
        "bg_color_dark": "#052e16",
        "text_color_light": "#166534",
        "text_color_dark": "#bbf7d0",
        "is_uncertain": False,
    }


def load_oof_prediction_bins(pred_path: Optional[Path | str] = None) -> Optional[List[Dict[str, Any]]]:
    """
    Loads out-of-fold predictions (reports/d_cw_128/predictions.csv) and derives empirical accuracy bins.
    No hardcoded numbers.
    """
    if pred_path is None:
        pred_path = Path(__file__).resolve().parent.parent.parent / "reports" / "d_cw_128" / "predictions.csv"
    else:
        pred_path = Path(pred_path)

    if not pred_path.exists():
        return None

    try:
        import pandas as pd
        df = pd.read_csv(pred_path)
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        if not prob_cols or "y_true" not in df.columns or "y_pred" not in df.columns:
            return None

        df["conf"] = df[prob_cols].max(axis=1)
        df["correct"] = (df["y_true"] == df["y_pred"]).astype(float)

        # Standard bins matching Low (<0.30), Medium (0.30-0.55), High (>=0.55)
        bins = [0.0, 0.30, 0.55, 1.01]
        labels = [0, 1, 2]
        df["bin_idx"] = pd.cut(df["conf"], bins=bins, labels=labels, right=False)

        bin_stats = []
        for b_idx in labels:
            sub = df[df["bin_idx"] == b_idx]
            if len(sub) > 0:
                acc = float(sub["correct"].mean() * 100.0)
                bin_stats.append({
                    "min": float(bins[b_idx]),
                    "max": float(bins[b_idx + 1]),
                    "accuracy": round(acc, 1),
                    "count": len(sub),
                })
        return bin_stats
    except Exception as e:
        logging.warning(f"Could not compute OOF prediction bins: {e}")
        return None


def get_evidence_line(
    conf: float,
    bins: Optional[List[Dict[str, Any]]] = None,
    lang: str = "en",
) -> Optional[str]:
    """
    Returns localized evidence line:
    'In testing, when the model was this confident, it was right about N% of the time.'
    Returns None if predictions file is missing or bins are unavailable.
    """
    if not bins:
        return None

    matched_bin = None
    for b in bins:
        if b["min"] <= conf < b["max"]:
            matched_bin = b
            break
    if matched_bin is None and bins:
        matched_bin = bins[-1] if conf >= bins[-1]["max"] else bins[0]

    if matched_bin is None:
        return None

    acc_str = f"{matched_bin['accuracy']:.0f}"
    template = t("evidence_line", lang)
    return template.replace("{accuracy}", acc_str)


def get_next_steps(
    pred_class: str,
    high_risk: bool = False,
    lang: str = "en",
) -> str:
    """Returns the plain-language clinical next-step guidance by condition category."""
    cond_meta = CONDITION_DESCRIPTIONS.get(pred_class, {})
    cat = cond_meta.get("category", "benign")

    if high_risk or cat == "malignant":
        return t("advice_malignant", lang)
    elif cat == "premalignant":
        return t("advice_premalignant", lang)
    elif cat == "nontumour":
        return t("advice_nontumour", lang)
    else:
        return t("advice_benign", lang)


def compute_input_hash(
    c_img: Optional[Image.Image],
    d_img: Optional[Image.Image],
    age: Optional[float],
    sex: str,
    site: str,
    model_type: str = "two_image_tabular_fusion",
) -> str:
    """Calculates deterministic SHA-256 fingerprint for input caching."""
    hasher = hashlib.sha256()

    if c_img is not None:
        hasher.update(c_img.tobytes()[:4096])
        hasher.update(f"{c_img.size}".encode())
    if d_img is not None:
        hasher.update(d_img.tobytes()[:4096])
        hasher.update(f"{d_img.size}".encode())

    hasher.update(f"{age}_{sex}_{site}_{model_type}".encode())
    return hasher.hexdigest()[:16]


def create_top3_horizontal_chart(
    top3_data: List[Any],
    theme: str = "Light",
    lang: str = "en",
):
    """
    Creates an Altair horizontal ranked bar chart:
    - Label = code + short plain name
    - Value shown as percentage on the bar
    - Axis 0-100%, sorted descending
    - Theme-aware colors
    """
    import pandas as pd
    import altair as alt

    records = []
    for item in top3_data:
        if isinstance(item, dict):
            c_code = item.get("code", "")
            prob = float(item.get("prob", 0.0))
            c_name = item.get("name", "")
        elif isinstance(item, (tuple, list)):
            c_code = str(item[0])
            prob = float(item[1])
            c_name = str(item[2]) if len(item) > 2 else ""
        else:
            c_code = str(item)
            prob = 0.0
            c_name = ""

        if not c_name:
            cond_meta = CONDITION_DESCRIPTIONS.get(c_code, {})
            name_key = f"{lang}_name" if f"{lang}_name" in cond_meta else "en_name"
            c_name = cond_meta.get(name_key, c_code)

        display_label = f"{c_code} — {c_name}"
        records.append({
            "code": c_code,
            "label": display_label,
            "prob": prob,
            "percentage": f"{prob * 100:.1f}%",
        })

    df = pd.DataFrame(records)

    is_dark = str(theme).lower() == "dark"
    bar_color = "#38bdf8" if is_dark else "#2563eb"
    text_color = "#f8fafc" if is_dark else "#0f172a"
    axis_color = "#94a3b8" if is_dark else "#64748b"

    base = alt.Chart(df).encode(
        y=alt.Y(
            "label:N",
            sort="-x",
            axis=alt.Axis(title=None, labelLimit=280, labelColor=text_color, labelFontSize=12),
        ),
        x=alt.X(
            "prob:Q",
            scale=alt.Scale(domain=[0.0, 1.0]),
            axis=alt.Axis(
                format=".0%",
                title=None,
                labelColor=axis_color,
                gridColor=axis_color,
                gridDash=[2, 2],
                tickCount=5,
            ),
        ),
    )

    bars = base.mark_bar(cornerRadiusEnd=4, height=22, color=bar_color)
    text = base.mark_text(
        align="left",
        baseline="middle",
        dx=6,
        color=text_color,
        fontSize=12,
        fontWeight="bold",
    ).encode(text="percentage:N")

    chart = (bars + text).properties(
        height=130,
        width="container",
    ).configure_view(
        strokeOpacity=0,
    )
    return chart


def generate_pdf_report(
    result_data: Dict[str, Any],
    c_img: Optional[Image.Image] = None,
    d_img: Optional[Image.Image] = None,
    lang: str = "en",
) -> bytes:
    """
    Builds a clean 1-page PDF diagnostic summary report using ReportLab.
    Falls back gracefully if reportlab encounters rendering issues.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "Title",
            parent=styles["Heading1"],
            fontSize=18,
            textColor=colors.HexColor("#1e3a8a"),
            spaceAfter=4,
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.HexColor("#334155"),
            leading=14,
        )
        bold_style = ParagraphStyle(
            "Bold",
            parent=body_style,
            fontName="Helvetica-Bold",
        )
        disclaimer_style = ParagraphStyle(
            "Disclaimer",
            parent=body_style,
            fontSize=8,
            textColor=colors.HexColor("#64748b"),
            leading=11,
        )

        elements = []

        # 1. Header
        elements.append(Paragraph("<b>VISHWAS (विश्वास): Skin AI Assessment Report</b>", title_style))
        elements.append(Paragraph(
            f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} | Educational & Research Prototype",
            disclaimer_style,
        ))
        elements.append(Spacer(1, 10))

        # 2. Patient / Case Profile Table
        age_str = f"{result_data.get('age', 'Not specified')} years" if result_data.get('age') else "Not specified"
        sex_str = str(result_data.get("sex", "Not specified")).capitalize()
        site_str = str(result_data.get("site", "Not specified")).replace("_", " ").title()

        profile_data = [
            [Paragraph("<b>Patient Age:</b>", body_style), Paragraph(age_str, body_style),
             Paragraph("<b>Biological Sex:</b>", body_style), Paragraph(sex_str, body_style)],
            [Paragraph("<b>Anatomical Site:</b>", body_style), Paragraph(site_str, body_style),
             Paragraph("<b>Tone Group:</b>", body_style), Paragraph(str(result_data.get("tone_group", "Unknown")), body_style)],
        ]
        t_prof = Table(profile_data, colWidths=[90, 160, 90, 160])
        t_prof.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(t_prof)
        elements.append(Spacer(1, 12))

        # 3. Photos
        img_cells = []
        if c_img is not None:
            c_buf = io.BytesIO()
            c_img.resize((150, 150)).save(c_buf, format="JPEG")
            c_buf.seek(0)
            img_cells.append(RLImage(c_buf, width=130, height=130))
        else:
            img_cells.append(Paragraph("<i>Close-up photo not provided</i>", disclaimer_style))

        if d_img is not None:
            d_buf = io.BytesIO()
            d_img.resize((150, 150)).save(d_buf, format="JPEG")
            d_buf.seek(0)
            img_cells.append(RLImage(d_buf, width=130, height=130))
        else:
            img_cells.append(Paragraph("<i>Dermoscopic photo not provided</i>", disclaimer_style))

        img_table = Table([[Paragraph("<b>Close-up View</b>", bold_style), Paragraph("<b>Dermoscopic View</b>", bold_style)],
                           img_cells], colWidths=[250, 250])
        img_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(img_table)
        elements.append(Spacer(1, 12))

        # 4. Primary Finding & Triage
        pred = result_data.get("pred_class", "Unknown")
        conf = float(result_data.get("conf", result_data.get("pred_prob", 0.0)))
        cond_meta = CONDITION_DESCRIPTIONS.get(pred, {})
        full_name = cond_meta.get("en_name", pred)
        desc = cond_meta.get("en_desc", "")

        triage_info = result_data.get("triage")
        if not triage_info:
            triage_info = compute_triage_verdict(pred, conf, result_data.get("malignant_prob", 0.0), lang="en")

        is_mal = triage_info["level"] == "red"
        card_bg = colors.HexColor("#fee2e2") if is_mal else colors.HexColor("#dcfce7")
        card_border = colors.HexColor("#ef4444") if is_mal else colors.HexColor("#10b981")

        verdict_text = [
            Paragraph(f"<b>Triage Status: {triage_info['icon']} {triage_info['title']}</b>", bold_style),
            Paragraph(f"<b>Estimated Finding: {full_name} ({pred})</b>", bold_style),
            Paragraph(f"AI Confidence: <b>{conf*100:.1f}%</b> ({get_confidence_label(conf, None, 'en')})", body_style),
            Paragraph(f"<i>{desc}</i>", body_style),
        ]
        v_table = Table([[verdict_text]], colWidths=[500])
        v_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), card_bg),
            ("BOX", (0, 0), (-1, -1), 1.5, card_border),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(v_table)
        elements.append(Spacer(1, 10))

        # 5. Top 3 Differential Diagnoses Table
        top3_rows = [["Rank", "Condition Name", "Code", "Probability"]]
        top3_list = result_data.get("top3", [])
        for r_idx, item in enumerate(top3_list, 1):
            if isinstance(item, dict):
                c_code = item.get("code", "")
                c_prob = float(item.get("prob", 0.0))
            elif isinstance(item, (tuple, list)):
                c_code = str(item[0])
                c_prob = float(item[1])
            else:
                c_code = str(item)
                c_prob = 0.0
            c_name = CONDITION_DESCRIPTIONS.get(c_code, {}).get("en_name", c_code)
            top3_rows.append([str(r_idx), c_name, c_code, f"{c_prob*100:.1f}%"])

        t_top3 = Table(top3_rows, colWidths=[40, 260, 80, 120])
        t_top3.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(Paragraph("<b>Top Differential Possibilities Considered:</b>", bold_style))
        elements.append(Spacer(1, 4))
        elements.append(t_top3)
        elements.append(Spacer(1, 12))

        # 6. Action Recommendation
        advice = get_next_steps(pred, result_data.get("is_high_risk", False), "en")
        clean_advice = (
            advice.replace("<b>", "")
            .replace("</b>", "")
            .replace("**", "")
            .replace("🚨 ", "")
            .replace("⚠️ ", "")
            .replace("✅ ", "")
            .replace("🩺 ", "")
        )
        elements.append(Paragraph("<b>Next Steps:</b>", bold_style))
        elements.append(Paragraph(clean_advice, body_style))
        elements.append(Spacer(1, 14))

        # 7. Persistent Educational Disclaimer
        elements.append(Paragraph(
            "<b>IMPORTANT DISCLAIMER:</b> This assessment is generated by an experimental research AI prototype "
            "(VISHWAS). It does not provide medical diagnoses or replace clinical examination by a qualified doctor or dermatologist. "
            "If this spot is bleeding, rapidly growing, asymmetrical, or concerning, please consult a healthcare professional immediately.",
            disclaimer_style,
        ))

        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()

    except Exception as e:
        fallback_buf = io.BytesIO()
        fallback_buf.write(
            f"VISHWAS Skin AI Report\nCondition: {result_data.get('pred_class')}\nConfidence: {result_data.get('conf')}\n".encode()
        )
        fallback_buf.seek(0)
        return fallback_buf.getvalue()
