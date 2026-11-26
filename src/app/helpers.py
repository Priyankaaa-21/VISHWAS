"""
Pure Helper Functions for VISHWAS Streamlit Application
Contains:
1. Image Quality Verification (blur, brightness, resolution)
2. Confidence & Category Next-Step Mappings
3. Deterministic Input Caching Hashes
4. PDF & JSON Diagnostic Result Report Generators
"""

from __future__ import annotations

import hashlib
import io
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import laplace

try:
    from src.app.strings import CONDITION_DESCRIPTIONS, STRINGS, t
except ImportError:
    from vishwas.src.app.strings import CONDITION_DESCRIPTIONS, STRINGS, t


# Default threshold values if not loaded from config
DEFAULT_APP_CONFIG = {
    "blur_threshold": 40.0,
    "min_brightness": 30.0,
    "max_brightness": 225.0,
    "min_resolution": 64,
    "confidence_high": 0.70,
    "confidence_medium": 0.40,
    "high_risk_threshold": 0.35,
    "default_language": "en",
    "max_history": 5,
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
    """Returns high / medium / low confidence text descriptor."""
    if cfg is None:
        cfg = DEFAULT_APP_CONFIG

    c_high = float(cfg.get("confidence_high", 0.70))
    c_med = float(cfg.get("confidence_medium", 0.40))

    if conf >= c_high:
        return t("conf_high", lang)
    elif conf >= c_med:
        return t("conf_med", lang)
    else:
        return t("conf_low", lang)


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
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
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

        # 4. Primary Finding
        pred = result_data.get("pred_class", "Unknown")
        conf = float(result_data.get("conf", result_data.get("pred_prob", 0.0)))
        cond_meta = CONDITION_DESCRIPTIONS.get(pred, {})
        full_name = cond_meta.get("en_name", pred)
        desc = cond_meta.get("en_desc", "")

        is_mal = cond_meta.get("category") in ["malignant", "premalignant"]
        card_bg = colors.HexColor("#fee2e2") if is_mal else colors.HexColor("#dcfce7")
        card_border = colors.HexColor("#ef4444") if is_mal else colors.HexColor("#10b981")

        verdict_text = [
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
        clean_advice = advice.replace("**", "").replace("🚨 ", "").replace("⚠️ ", "").replace("✅ ", "").replace("🩺 ", "")
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
        # Fallback to plain text / simple PDF if error
        fallback_buf = io.BytesIO()
        fallback_buf.write(f"VISHWAS Skin AI Report\nCondition: {result_data.get('pred_class')}\nConfidence: {result_data.get('conf')}\n".encode())
        fallback_buf.seek(0)
        return fallback_buf.getvalue()
