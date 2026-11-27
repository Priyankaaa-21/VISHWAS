"""
VISHWAS Label-Free Lesion Feature Extraction Module: src/explain/features.py

Provides:
1. Lesion segmentation via adaptive/Otsu thresholding and morphological cleanup.
2. ABCD dermatological feature measurement:
   - Asymmetry (A): Biaxial IoU overlap score in [0, 1]
   - Border Irregularity (B): Normalized isoperimetric compactness index P^2 / (4*pi*A)
   - Color Variegation (C): Dominant color clustering and count across standard dermoscopy palettes
   - Diameter (D): Equivalent diameter in pixels
   - Heuristics: Blue-white veil detection and vascular / erythematous structure detection
3. Caching of extracted features to data/cache/features/<hash>.json.
4. Comprehensive documentation of methodology, assumptions, and known clinical weaknesses.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from scipy.ndimage import binary_fill_holes, center_of_mass, label
except ImportError:
    from scipy.ndimage.morphology import binary_fill_holes
    from scipy.ndimage.measurements import center_of_mass, label

VISHWAS_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = VISHWAS_ROOT / "data" / "cache" / "features"


def compute_image_hash(image: Image.Image) -> str:
    """Computes SHA-256 hash of RGB image bytes."""
    img_bytes = image.convert("RGB").tobytes()
    return hashlib.sha256(img_bytes).hexdigest()[:16]


def segment_lesion(image: Image.Image) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Segments skin lesion from background skin using Otsu-guided color contrast
    and morphological operations.

    Methodology:
    1. Converts image to RGB and calculates luminance / color contrast against mean corner skin.
    2. Computes Otsu threshold to separate central darker/pigmented or erythematous lesion from healthy skin.
    3. Performs morphological hole filling and extracts the connected component closest to image center.

    Known Weaknesses:
    - Illumination gradients and vignetting from dermatoscope lenses can distort the periphery.
    - High baseline melanin in Fitzpatrick IV-VI skin reduces contrast between lesion and background.
    - Thick dark hairs, ruler markings, or air bubbles in dermoscopy gel can produce artifactual mask protrusions.
    """
    rgb = np.array(image.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape

    # Estimate background healthy skin from corners (top-left, top-right, bottom-left, bottom-right)
    corner_size = max(4, min(h, w) // 10)
    corners = np.concatenate([
        rgb[:corner_size, :corner_size].reshape(-1, 3),
        rgb[:corner_size, -corner_size:].reshape(-1, 3),
        rgb[-corner_size:, :corner_size].reshape(-1, 3),
        rgb[-corner_size:, -corner_size:].reshape(-1, 3),
    ], axis=0)
    bg_color = np.median(corners, axis=0)

    # Euclidean color distance from background skin
    diff = np.linalg.norm(rgb - bg_color, axis=2)

    # Normalize contrast map
    min_d, max_d = diff.min(), diff.max()
    norm_diff = (diff - min_d) / (max_d - min_d + 1e-8)

    # Center-weighted Gaussian prior (lesions are clinically framed in center)
    y_coords, x_coords = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    sigma_y, sigma_x = h / 2.5, w / 2.5
    center_weight = np.exp(-(((y_coords - cy) ** 2) / (2 * sigma_y ** 2) + ((x_coords - cx) ** 2) / (2 * sigma_x ** 2)))

    weighted_diff = norm_diff * (0.6 + 0.4 * center_weight)

    # Otsu thresholding
    hist, bin_edges = np.histogram(weighted_diff, bins=64, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total_pixels = float(h * w)
    
    current_max = 0.0
    threshold = 0.35
    sum_total = np.dot(hist, bin_centers)
    sum_b = 0.0
    weight_b = 0.0

    for i in range(len(hist)):
        weight_b += hist[i]
        if weight_b == 0:
            continue
        weight_f = total_pixels - weight_b
        if weight_f == 0:
            break
        sum_b += hist[i] * bin_centers[i]
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        between_var = weight_b * weight_f * (mean_b - mean_f) ** 2
        if between_var > current_max:
            current_max = between_var
            threshold = float(bin_centers[i])

    # Initial binary mask
    raw_mask = weighted_diff >= threshold

    # Fill internal holes
    filled_mask = binary_fill_holes(raw_mask)

    # Connected component labeling - select central component
    labeled, num_features = label(filled_mask)
    if num_features == 0:
        # Fallback central elliptical mask
        fallback_mask = center_weight > 0.5
        return fallback_mask, {"segmented": False, "reason": "No high-contrast component found"}

    # Find component closest to image center
    raw_coms = center_of_mass(filled_mask, labeled, range(1, num_features + 1))
    if isinstance(raw_coms, tuple):
        com_list = [raw_coms]
    elif isinstance(raw_coms, list):
        com_list = raw_coms
    else:
        com_list = list(raw_coms)

    best_label = 1
    min_dist_to_center = float("inf")
    for idx, pt in enumerate(com_list, start=1):
        if isinstance(pt, (tuple, list, np.ndarray)) and len(pt) == 2:
            ry, rx = pt[0], pt[1]
        else:
            continue
        if np.isnan(ry) or np.isnan(rx):
            continue
        dist = np.hypot(ry - cy, rx - cx)
        area = np.sum(labeled == idx)
        if area >= (total_pixels * 0.01) and dist < min_dist_to_center:
            min_dist_to_center = dist
            best_label = idx

    final_mask = (labeled == best_label)

    return final_mask, {
        "segmented": True,
        "lesion_area_px": int(np.sum(final_mask)),
        "mask_fraction": float(np.mean(final_mask)),
        "threshold": round(threshold, 4),
    }


def compute_asymmetry(mask: np.ndarray) -> float:
    """
    Computes biaxial asymmetry index in [0.0, 1.0].
    Measures IoU difference when flipping along horizontal and vertical principal axes.
    0.0 = Perfectly symmetric; 1.0 = Maximally asymmetric.
    """
    if np.sum(mask) == 0:
        return 0.0

    # Horizontal flip IoU
    h_flipped = np.fliplr(mask)
    h_overlap = np.sum(mask & h_flipped) / (np.sum(mask | h_flipped) + 1e-8)
    h_asym = 1.0 - float(h_overlap)

    # Vertical flip IoU
    v_flipped = np.flipud(mask)
    v_overlap = np.sum(mask & v_flipped) / (np.sum(mask | v_flipped) + 1e-8)
    v_asym = 1.0 - float(v_overlap)

    asymmetry_score = float(np.clip(0.5 * (h_asym + v_asym), 0.0, 1.0))
    return round(asymmetry_score, 4)


def compute_border_irregularity(mask: np.ndarray) -> float:
    """
    Computes normalized compactness / isoperimetric quotient:
    Border Irregularity = Perimeter^2 / (4 * pi * Area).
    For a perfect circle, value = 1.0. Irregular, jagged, or scalloped borders yield values > 1.0.
    """
    area = float(np.sum(mask))
    if area < 10:
        return 1.0

    # Perimeter via morphological gradient (difference between dilation and erosion)
    y_shift = np.zeros_like(mask)
    y_shift[:-1, :] = mask[1:, :]
    x_shift = np.zeros_like(mask)
    x_shift[:, :-1] = mask[:, 1:]

    boundary = (mask != y_shift) | (mask != x_shift)
    perimeter = float(np.sum(boundary))

    compactness = (perimeter ** 2) / (4.0 * np.pi * area + 1e-8)
    # Normalized so circle is 1.0, typical lesions range 1.0 - 3.5
    return round(float(compactness), 4)


def extract_dominant_colors(rgb: np.ndarray, mask: np.ndarray) -> Tuple[int, List[str]]:
    """
    Extracts number of distinct color categories present within the lesion mask.
    Recognized dermoscopic color clusters:
    - Light brown / Tan
    - Dark brown
    - Black
    - Red / Pink
    - Blue-white / Gray
    - White / Depigmentation
    """
    lesion_pixels = rgb[mask]
    if len(lesion_pixels) < 10:
        return 1, ["Light brown / Tan"]

    r = lesion_pixels[:, 0]
    g = lesion_pixels[:, 1]
    b = lesion_pixels[:, 2]

    # Convert to approximate lightness and saturation
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    lightness = max_c / 255.0
    delta = (max_c - min_c) / 255.0
    saturation = np.where(max_c > 0, delta / (max_c / 255.0 + 1e-8), 0.0)

    detected_colors = set()

    # Black (very low lightness)
    black_frac = np.mean(lightness < 0.22)
    if black_frac > 0.03:
        detected_colors.add("Black")

    # Dark brown (low lightness, warm tone)
    dark_brown_frac = np.mean((lightness >= 0.22) & (lightness < 0.45) & (r > b))
    if dark_brown_frac > 0.05:
        detected_colors.add("Dark brown")

    # Light brown / Tan (moderate lightness, warm)
    light_brown_frac = np.mean((lightness >= 0.45) & (lightness < 0.70) & (r > b))
    if light_brown_frac > 0.08:
        detected_colors.add("Light brown / Tan")

    # Red / Pink (excess red channel r > g + 20 and r > b + 20)
    red_frac = np.mean((r > g + 25) & (r > b + 25))
    if red_frac > 0.04:
        detected_colors.add("Red / Pink (Vascular)")

    # Blue-white / Blue-gray (blue dominant or close to red, moderate lightness, low saturation)
    blue_gray_frac = np.mean((b > r) & (lightness >= 0.35) & (lightness < 0.75))
    if blue_gray_frac > 0.03:
        detected_colors.add("Blue-white / Gray")

    # White / Scar depigmentation (high lightness, low saturation)
    white_frac = np.mean((lightness >= 0.75) & (saturation < 0.15))
    if white_frac > 0.03:
        detected_colors.add("White / Depigmentation")

    if not detected_colors:
        detected_colors.add("Light brown / Tan")

    return len(detected_colors), sorted(list(detected_colors))


def detect_blue_white_veil(rgb: np.ndarray, mask: np.ndarray) -> Tuple[bool, float]:
    """
    Detects presence of blue-white veil (ground-glass bluish haze over pigmentation).
    Heuristic: Pixels inside mask where blue channel >= red channel and lightness >= 0.35.
    """
    lesion_pixels = rgb[mask]
    if len(lesion_pixels) < 10:
        return False, 0.0

    r = lesion_pixels[:, 0]
    b = lesion_pixels[:, 2]
    lightness = np.maximum(r, b) / 255.0

    bw_pixels = (b >= r) & (lightness >= 0.30)
    frac = float(np.mean(bw_pixels))
    has_veil = (frac >= 0.035)
    return has_veil, round(frac, 4)


def detect_vascular_structures(rgb: np.ndarray, mask: np.ndarray) -> Tuple[bool, float]:
    """
    Detects presence of erythematous / vascular patterns.
    Heuristic: Pixels where excess red (2R - G - B) exceeds a threshold and R > 110.
    """
    lesion_pixels = rgb[mask]
    if len(lesion_pixels) < 10:
        return False, 0.0

    r = lesion_pixels[:, 0].astype(np.float32)
    g = lesion_pixels[:, 1].astype(np.float32)
    b = lesion_pixels[:, 2].astype(np.float32)

    excess_red = (2.0 * r) - g - b
    vasc_pixels = (excess_red >= 50.0) & (r >= 100.0)
    frac = float(np.mean(vasc_pixels))
    has_vascular = (frac >= 0.05)
    return has_vascular, round(frac, 4)


def extract_lesion_features(
    image: Image.Image,
    mask: Optional[np.ndarray] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Extracts complete ABCD and heuristic dermoscopic features for a given image.
    Caches results in data/cache/features/<hash>.json.
    """
    img_hash = compute_image_hash(image)
    cache_file = CACHE_DIR / f"{img_hash}.json"

    if use_cache and cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception as e:
            logging.warning("Error reading feature cache %s: %s", cache_file, e)

    # 1. Lesion segmentation
    if mask is None:
        mask, seg_meta = segment_lesion(image)
    else:
        seg_meta = {"segmented": True, "lesion_area_px": int(np.sum(mask))}

    rgb = np.array(image.convert("RGB"))
    h, w, _ = rgb.shape

    # 2. Asymmetry
    asym_score = compute_asymmetry(mask)

    # 3. Border Irregularity
    border_score = compute_border_irregularity(mask)

    # 4. Dominant Colors
    num_colors, color_names = extract_dominant_colors(rgb, mask)

    # 5. Diameter in pixels
    area = float(np.sum(mask))
    equiv_diameter_px = round(float(2.0 * np.sqrt(area / np.pi)), 1)

    # Bounding box dimensions
    coords = np.argwhere(mask)
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        bbox_w = int(x_max - x_min + 1)
        bbox_h = int(y_max - y_min + 1)
    else:
        bbox_w, bbox_h = 0, 0

    # 6. Blue-white veil heuristic
    has_veil, veil_frac = detect_blue_white_veil(rgb, mask)

    # 7. Vascular / erythema heuristic
    has_vasc, vasc_frac = detect_vascular_structures(rgb, mask)

    # 8. Ulceration / crust heuristic (high-contrast central erosion or crust)
    ulceration_present = (veil_frac > 0.08 or vasc_frac > 0.12) and (border_score > 1.4)

    # Qualitative clinical interpretation
    is_high_asymmetry = asym_score >= 0.40
    is_irregular_border = border_score >= 1.45
    is_multicolor = num_colors >= 3

    features_dict = {
        "image_hash": img_hash,
        "asymmetry_score": asym_score,
        "is_asymmetric": is_high_asymmetry,
        "border_irregularity_score": border_score,
        "is_irregular_border": is_irregular_border,
        "num_dominant_colors": num_colors,
        "is_multicolor": is_multicolor,
        "dominant_colors": color_names,
        "equivalent_diameter_px": equiv_diameter_px,
        "bounding_box_w_px": bbox_w,
        "bounding_box_h_px": bbox_h,
        "has_blue_white_veil": has_veil,
        "blue_white_veil_fraction": veil_frac,
        "has_vascular_structures": has_vasc,
        "vascular_fraction": vasc_frac,
        "ulceration_crust_present": ulceration_present,
        "segmentation_meta": seg_meta,
        "methodology": (
            "Label-free automated lesion segmentation (Otsu color-contrast + morphological cleanup) "
            "followed by analytical measurement of ABCD criteria (biaxial IoU asymmetry, normalized "
            "isoperimetric boundary compactness, color-clustering, and color-space veil/vascular heuristics)."
        ),
        "known_weaknesses": [
            "Sensitivity to non-uniform lighting and peripheral vignetting from dermatoscope faceplates.",
            "Darker baseline skin tones (Fitzpatrick IV-VI) elevate background melanin and can reduce threshold contrast.",
            "Presence of dark hairs, surgical marker lines, or air bubbles in immersion fluid can artifactually distort border regularity.",
            "Does not measure microscopic subcellular atypia, which requires histological examination."
        ],
    }

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(features_dict, indent=2), encoding="utf-8")
    except Exception as exc:
        logging.warning("Could not cache features to %s: %s", cache_file, exc)

    return features_dict


def compute_feature_agreement(
    observed_features: Dict[str, Any],
    candidate_class: str,
    disease_features_db: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Computes a feature-agreement score and checklist comparing observed features
    against typical clinical and dermoscopic criteria for a candidate class.
    """
    if disease_features_db is None:
        feat_path = VISHWAS_ROOT / "data" / "disease_features.json"
        if feat_path.exists():
            try:
                disease_features_db = json.loads(feat_path.read_text(encoding="utf-8"))
            except Exception:
                disease_features_db = {}
        else:
            disease_features_db = {}

    target_info = disease_features_db.get(candidate_class, {})
    checklist: List[Dict[str, Any]] = []

    # Map observed features to disease criteria
    if candidate_class == "MEL":
        # Melanoma expects asymmetry, irregular border, multi-color, blue-white veil, vascularity
        checklist.append({"feature": "Asymmetry across axes", "seen": observed_features.get("is_asymmetric", False), "weight": 2.0})
        checklist.append({"feature": "Irregular jagged border", "seen": observed_features.get("is_irregular_border", False), "weight": 2.0})
        checklist.append({"feature": "Multiple colors (≥3 shades)", "seen": observed_features.get("is_multicolor", False), "weight": 2.0})
        checklist.append({"feature": "Blue-white veil or gray structures", "seen": observed_features.get("has_blue_white_veil", False), "weight": 1.5})
        checklist.append({"feature": "Atypical vascular / erythematous pattern", "seen": observed_features.get("has_vascular_structures", False), "weight": 1.0})

    elif candidate_class == "BCC":
        # BCC expects arborizing vessels / erythema, ulceration, discrete borders
        checklist.append({"feature": "Vascularity / Erythematous vessels", "seen": observed_features.get("has_vascular_structures", False), "weight": 2.5})
        checklist.append({"feature": "Ulceration or central crust", "seen": observed_features.get("ulceration_crust_present", False), "weight": 2.0})
        checklist.append({"feature": "Blue-gray ovoid structures / veil", "seen": observed_features.get("has_blue_white_veil", False), "weight": 1.5})
        checklist.append({"feature": "Irregular lobulated contour", "seen": observed_features.get("is_irregular_border", False), "weight": 1.0})

    elif candidate_class == "NV":
        # Nevus expects symmetry, smooth border, uniform color (1-2 shades)
        checklist.append({"feature": "Symmetric round/oval shape", "seen": not observed_features.get("is_asymmetric", False), "weight": 2.5})
        checklist.append({"feature": "Smooth, well-defined border", "seen": not observed_features.get("is_irregular_border", False), "weight": 2.0})
        checklist.append({"feature": "Uniform color distribution (≤2 colors)", "seen": not observed_features.get("is_multicolor", False), "weight": 2.0})
        checklist.append({"feature": "Absence of blue-white veil", "seen": not observed_features.get("has_blue_white_veil", False), "weight": 1.5})

    elif candidate_class == "BKL":
        # BKL expects verrucous/irregular border, brown/tan, lack of veil
        checklist.append({"feature": "Irregular verrucous border", "seen": observed_features.get("is_irregular_border", False), "weight": 2.0})
        checklist.append({"feature": "Multiple shades of brown / tan", "seen": observed_features.get("is_multicolor", False), "weight": 1.5})
        checklist.append({"feature": "Absence of prominent blue-white veil", "seen": not observed_features.get("has_blue_white_veil", False), "weight": 1.5})

    elif candidate_class == "VASC":
        # Vascular expects red/vascular presence, well-defined border
        checklist.append({"feature": "Prominent vascular / red pigmentation", "seen": observed_features.get("has_vascular_structures", False), "weight": 3.0})
        checklist.append({"feature": "Circumscribed border", "seen": not observed_features.get("is_irregular_border", False), "weight": 1.5})

    elif candidate_class == "AKIEC":
        # AK expects scale/crust, vascular erythema
        checklist.append({"feature": "Erythematous background", "seen": observed_features.get("has_vascular_structures", False), "weight": 2.0})
        checklist.append({"feature": "Surface scale / ulceration crust", "seen": observed_features.get("ulceration_crust_present", False), "weight": 2.0})
        checklist.append({"feature": "Irregular border / macule", "seen": observed_features.get("is_irregular_border", False), "weight": 1.5})

    else:
        # Generic checklist
        checklist.append({"feature": "Shape symmetry", "seen": not observed_features.get("is_asymmetric", False), "weight": 1.0})
        checklist.append({"feature": "Border regularity", "seen": not observed_features.get("is_irregular_border", False), "weight": 1.0})
        checklist.append({"feature": "Color uniformity", "seen": not observed_features.get("is_multicolor", False), "weight": 1.0})

    total_weight = sum(item["weight"] for item in checklist)
    matched_weight = sum(item["weight"] for item in checklist if item["seen"])
    agreement_pct = round((matched_weight / (total_weight + 1e-8)) * 100.0, 1)

    return {
        "candidate_class": candidate_class,
        "class_name": target_info.get("name", candidate_class),
        "agreement_percentage": agreement_pct,
        "checklist": checklist,
        "matched_count": sum(1 for item in checklist if item["seen"]),
        "total_count": len(checklist),
        "darker_skin_considerations": target_info.get("darker_skin_considerations", ""),
    }
