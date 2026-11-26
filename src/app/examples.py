"""
VISHWAS Curated Example Cases & Side-by-Side Image Previews
Manages curated examples from assets/examples/examples.json, ensures images are
displayed with captions and 'Example case' badges, and handles missing image fallbacks.
"""

from __future__ import annotations
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import streamlit as st
from PIL import Image, ImageDraw


def get_vishwas_root() -> Path:
    """Finds the vishwas project root directory robustly across working directories."""
    # 1. Check relative to this source file: src/app/examples.py -> vishwas
    p = Path(__file__).resolve().parent.parent.parent
    if (p / "assets" / "examples" / "examples.json").exists():
        return p
    # 2. Check current working directory
    cwd = Path.cwd()
    if (cwd / "assets" / "examples" / "examples.json").exists():
        return cwd
    # 3. Check cwd/vishwas
    if (cwd / "vishwas" / "assets" / "examples" / "examples.json").exists():
        return cwd / "vishwas"
    return p


@st.cache_data
def load_curated_examples_with_status() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Loads curated example cases from assets/examples/examples.json.
    Returns (examples_list, error_message).
    Never swallows exceptions silently.
    """
    root = get_vishwas_root()
    manifest_path = root / "assets" / "examples" / "examples.json"
    
    if not manifest_path.exists():
        err_msg = f"Could not find assets/examples/examples.json at '{manifest_path}'"
        logging.error(err_msg)
        return [], err_msg

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                err = f"Expected JSON list in {manifest_path}, got {type(data).__name__}"
                logging.error(err)
                return [], err
            return data, None
    except Exception as exc:
        err_msg = f"Failed to parse {manifest_path}: {exc}"
        logging.error(err_msg)
        return [], err_msg


def load_curated_examples() -> List[Dict[str, Any]]:
    """Convenience helper returning just the list of curated examples."""
    examples, _ = load_curated_examples_with_status()
    return examples


def get_example_by_id(example_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a specific curated example by ID or diagnosis code."""
    examples = load_curated_examples()
    target = example_id.strip().lower()
    for ex in examples:
        if ex.get("id", "").lower() == target or ex.get("diagnosis", "").lower() == target:
            return ex
    return None


def get_placeholder_image(text: str = "Image not found") -> Image.Image:
    """Generates a neutral placeholder image if a file is missing."""
    img = Image.new("RGB", (300, 300), color=(30, 41, 59))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 290, 290], outline=(71, 85, 105), width=2)
    draw.line([10, 10, 290, 290], fill=(51, 65, 85), width=1)
    draw.line([10, 290, 290, 10], fill=(51, 65, 85), width=1)
    draw.text((70, 140), text, fill=(241, 245, 249))
    return img


def resolve_image_path(path_str: Optional[str]) -> Optional[Path]:
    """Resolves an image file path against vishwas root, CWD, or absolute path."""
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    root = get_vishwas_root()
    cand1 = root / p
    if cand1.exists():
        return cand1
    cand2 = Path.cwd() / p
    if cand2.exists():
        return cand2
    cand3 = Path.cwd() / "vishwas" / p
    if cand3.exists():
        return cand3
    return None


def load_image_safe(path: Optional[str]) -> Image.Image:
    """
    Safely loads an image from disk, validating that the file exists and is readable.
    Falls back to a clean placeholder with a logged warning if missing.
    """
    resolved = resolve_image_path(path)
    if resolved and resolved.is_file():
        try:
            return Image.open(resolved)
        except Exception as exc:
            logging.warning("Failed to open image file at %s: %s", resolved, exc)
    else:
        logging.warning("Image file does not exist at requested path: %s", path)
    return get_placeholder_image("Image unavailable")


def render_image_preview_card(
    is_example: bool,
    caption: str,
    clin_img: Any,
    derm_img: Optional[Any] = None,
    badge_text: str = "🏷️ Example Case"
) -> None:
    """
    Renders the image preview block with appropriate badge, caption, and image previews.
    Always constrained to ~220px max height.
    """
    if is_example:
        st.markdown(f'<span class="example-badge">{badge_text}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="example-badge" style="background-color: var(--info);">📸 User Photo</span>', unsafe_allow_html=True)

    if caption:
        st.caption(f"**{caption}**")

    # Render image(s) side by side
    if derm_img is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.image(clin_img, caption="Close-up View", use_container_width=True)
        with c2:
            st.image(derm_img, caption="Magnified (Dermoscopic) View", use_container_width=True)
    elif clin_img is not None:
        st.image(clin_img, caption="Spot Photo", use_container_width=True)
    else:
        st.image(get_placeholder_image("No Image"), caption="No Image Selected", use_container_width=True)
