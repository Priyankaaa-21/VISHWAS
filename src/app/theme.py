"""
VISHWAS Centralized Design & Theme System
Provides light, dark, and auto CSS variable palettes adhering to WCAG AA contrast standards,
and injects comprehensive styling across ALL native Streamlit widgets:
- Headers, top bar, and sidebar
- Selectbox & popover menu, radio, checkbox, toggle, number input, slider, text input
- File uploader & camera input
- Tabs, expanders, tooltips, toasts, status boxes, modals/dialogs
- Dataframes, st.metric, alert boxes, code blocks, and custom cards.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import streamlit as st


LIGHT_VARS = """
    --bg: #ffffff;
    --surface: #f8fafc;
    --surface-hover: #f1f5f9;
    --surface-card: #ffffff;
    --text: #0f172a;
    --text-strong: #020617;
    --muted: #475569;
    --border: #cbd5e1;
    --accent: #2563eb;
    --accent-hover: #1d4ed8;
    --danger: #dc2626;
    --danger-bg: #fef2f2;
    --danger-text: #991b1b;
    --danger-border: #fca5a5;
    --warn: #d97706;
    --warn-bg: #fffbeb;
    --warn-text: #92400e;
    --warn-border: #fcd34d;
    --ok: #16a34a;
    --ok-bg: #f0fdf4;
    --ok-text: #166534;
    --ok-border: #86efac;
    --info: #0284c7;
    --info-bg: #f0f9ff;
    --info-text: #075985;
    --info-border: #7dd3fc;
    --grey-bg: #f1f5f9;
    --grey-text: #334155;
    --grey-border: #cbd5e1;
    --code-bg: #f1f5f9;
    --code-text: #0f172a;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.08);
"""

DARK_VARS = """
    --bg: #0b1120;
    --surface: #111827;
    --surface-hover: #1e293b;
    --surface-card: #141e33;
    --text: #f8fafc;
    --text-strong: #ffffff;
    --muted: #94a3b8;
    --border: #334155;
    --accent: #3b82f6;
    --accent-hover: #60a5fa;
    --danger: #ef4444;
    --danger-bg: #450a0a;
    --danger-text: #fecaca;
    --danger-border: #991b1b;
    --warn: #f59e0b;
    --warn-bg: #451a03;
    --warn-text: #fde68a;
    --warn-border: #92400e;
    --ok: #22c55e;
    --ok-bg: #052e16;
    --ok-text: #bbf7d0;
    --ok-border: #166534;
    --info: #38bdf8;
    --info-bg: #082f49;
    --info-text: #bae6fd;
    --info-border: #0369a1;
    --grey-bg: #1e293b;
    --grey-text: #e2e8f0;
    --grey-border: #475569;
    --code-bg: #0f172a;
    --code-text: #e2e8f0;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -2px rgba(0, 0, 0, 0.3);
    --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.4);
"""


def get_theme_css(theme_mode: str) -> str:
    """Generates complete CSS styles using CSS variables for light, dark, or auto mode."""
    theme_lower = str(theme_mode).strip().lower()
    
    if theme_lower == "dark":
        root_block = f":root {{\n{DARK_VARS}\n}}"
    elif theme_lower == "light":
        root_block = f":root {{\n{LIGHT_VARS}\n}}"
    else:  # auto
        root_block = f"""
        :root {{
            {LIGHT_VARS}
        }}
        @media (prefers-color-scheme: dark) {{
            :root {{
                {DARK_VARS}
            }}
        }}
        """

    component_rules = """
        html, body, .stApp {
            background-color: var(--bg) !important;
            color: var(--text) !important;
            font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans Devanagari", "Noto Sans Kannada", sans-serif !important;
        }

        /* ---------------------------------------------------------------------
           TYPOGRAPHY & HERO HEADER
           --------------------------------------------------------------------- */
        .main-header-hero {
            font-size: clamp(2.3rem, 5.5vw, 3.2rem) !important;
            font-weight: 900 !important;
            line-height: 1.15 !important;
            letter-spacing: -0.025em !important;
            color: var(--text) !important;
            margin-top: 0.1rem !important;
            margin-bottom: 0.25rem !important;
            display: flex;
            align-items: baseline;
            flex-wrap: wrap;
            gap: 0.4rem;
        }
        .main-header-hero .hero-brand {
            font-weight: 950 !important;
            color: var(--accent) !important;
            letter-spacing: -0.02em !important;
        }
        .main-header-hero .hero-devanagari {
            font-weight: 850 !important;
            color: var(--text) !important;
            opacity: 0.95 !important;
            font-family: "Noto Sans Devanagari", -apple-system, sans-serif !important;
        }
        .main-header-hero .hero-colon {
            color: var(--muted) !important;
            font-weight: 400 !important;
        }
        .main-header-hero .hero-tagline {
            font-weight: 700 !important;
            font-size: clamp(1.6rem, 3.8vw, 2.3rem) !important;
            color: var(--text) !important;
        }

        .main-header-sub {
            font-size: clamp(1.0rem, 2vw, 1.18rem) !important;
            font-weight: 400 !important;
            color: var(--muted) !important;
            margin-top: 0.1rem !important;
            margin-bottom: 1.25rem !important;
            line-height: 1.5 !important;
        }

        /* Sidebar Logo Branding */
        .sidebar-brand {
            font-weight: 900 !important;
            font-size: 1.45rem !important;
            letter-spacing: -0.015em !important;
            color: var(--text) !important;
            margin-top: 0 !important;
            margin-bottom: 0.8rem !important;
        }
        .sidebar-brand .hero-brand {
            color: var(--accent) !important;
            font-weight: 950 !important;
        }

        /* ---------------------------------------------------------------------
           APP HEADER & TOP TOOLBAR
           --------------------------------------------------------------------- */
        [data-testid="stHeader"], header[data-testid="stHeader"], .stAppHeader {
            background-color: var(--bg) !important;
            color: var(--text) !important;
            border-bottom: 1px solid var(--border) !important;
        }
        header[data-testid="stHeader"] svg, header[data-testid="stHeader"] button {
            color: var(--text) !important;
        }

        /* ---------------------------------------------------------------------
           SIDEBAR
           --------------------------------------------------------------------- */
        [data-testid="stSidebar"], section[data-testid="stSidebar"] {
            background-color: var(--surface) !important;
            border-right: 1px solid var(--border) !important;
            color: var(--text) !important;
        }
        [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
            color: var(--text) !important;
        }

        /* ---------------------------------------------------------------------
           BUTTONS (PRIMARY & SECONDARY)
           --------------------------------------------------------------------- */
        .stButton > button, button[kind="secondary"] {
            background-color: var(--surface) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.15s ease-in-out !important;
        }
        .stButton > button:hover, button[kind="secondary"]:hover {
            background-color: var(--surface-hover) !important;
            border-color: var(--accent) !important;
            color: var(--accent) !important;
        }
        button[kind="primary"], .stButton > button[kind="primary"] {
            background-color: var(--accent) !important;
            color: #ffffff !important;
            border: 1px solid var(--accent) !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
        }
        button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {
            background-color: var(--accent-hover) !important;
            border-color: var(--accent-hover) !important;
            color: #ffffff !important;
        }
        .stDownloadButton > button {
            background-color: var(--surface) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        .stDownloadButton > button:hover {
            background-color: var(--surface-hover) !important;
            border-color: var(--accent) !important;
            color: var(--accent) !important;
        }

        /* ---------------------------------------------------------------------
           FORM INPUTS (SELECTBOX, RADIO, CHECKBOX, TOGGLE, NUMBER, SLIDER, TEXT)
           --------------------------------------------------------------------- */
        /* Selectbox container and input */
        [data-testid="stSelectbox"], [data-baseweb="select"], [data-baseweb="select"] > div {
            background-color: var(--surface) !important;
            border-color: var(--border) !important;
            color: var(--text) !important;
            border-radius: 8px !important;
        }
        [data-baseweb="select"] span, [data-baseweb="select"] div {
            color: var(--text) !important;
        }
        [data-baseweb="select"] svg {
            fill: var(--text) !important;
        }
        /* Selectbox Popover Menu Dropdown */
        [data-baseweb="popover"], [data-baseweb="menu"], ul[role="listbox"] {
            background-color: var(--surface) !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
            border-radius: 8px !important;
            box-shadow: var(--shadow-lg) !important;
        }
        li[role="option"], [data-baseweb="menu"] li {
            background-color: var(--surface) !important;
            color: var(--text) !important;
        }
        li[role="option"]:hover, [data-baseweb="menu"] li:hover, li[aria-selected="true"] {
            background-color: var(--surface-hover) !important;
            color: var(--accent) !important;
        }

        /* Radio & Checkbox */
        [data-testid="stRadio"] label, [data-testid="stCheckbox"] label, [data-testid="stToggle"] label {
            color: var(--text) !important;
            font-weight: 500 !important;
        }
        [data-testid="stRadio"] [role="radiogroup"] {
            gap: 10px;
        }
        [data-testid="stRadio"] div[role="radio"], [data-testid="stCheckbox"] span {
            color: var(--text) !important;
        }

        /* Sliders */
        [data-testid="stSlider"] div {
            color: var(--text) !important;
        }

        /* Number & Text Input */
        [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input, textarea {
            background-color: var(--surface) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
        }
        [data-testid="stNumberInput"] button {
            background-color: var(--surface-hover) !important;
            color: var(--text) !important;
            border-color: var(--border) !important;
        }

        /* ---------------------------------------------------------------------
           FILE UPLOADER & CAMERA INPUT
           --------------------------------------------------------------------- */
        [data-testid="stFileUploader"] {
            background-color: transparent !important;
        }
        [data-testid="stFileUploaderDropzone"], [data-testid="stFileUploadDropzone"] {
            background-color: var(--surface) !important;
            border: 2px dashed var(--border) !important;
            border-radius: 12px !important;
            color: var(--text) !important;
            padding: 1.5rem !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover, [data-testid="stFileUploadDropzone"]:hover {
            border-color: var(--accent) !important;
        }
        [data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploadDropzone"] span,
        [data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploadDropzone"] small {
            color: var(--muted) !important;
        }
        [data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploadDropzone"] button {
            background-color: var(--surface-hover) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
        }
        [data-testid="stFileUploaderFileData"], [data-testid="stFileUploaderFileName"] {
            color: var(--text) !important;
        }
        [data-testid="stCameraInput"] {
            background-color: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            padding: 10px !important;
        }
        [data-testid="stCameraInput"] button {
            background-color: var(--surface-hover) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
        }

        /* ---------------------------------------------------------------------
           DIALOGS & MODALS (ST.DIALOG)
           --------------------------------------------------------------------- */
        div[role="dialog"], [data-testid="stDialog"], [data-testid="stModal"] {
            background-color: var(--surface-card) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
            box-shadow: var(--shadow-lg) !important;
        }
        div[role="dialog"] p, div[role="dialog"] span, div[role="dialog"] h1, div[role="dialog"] h2, div[role="dialog"] h3, div[role="dialog"] h4 {
            color: var(--text) !important;
        }
        div[role="dialog"] header, div[role="dialog"] [data-testid="stDialogHeader"] {
            background-color: var(--surface-card) !important;
            color: var(--text) !important;
            border-bottom: 1px solid var(--border) !important;
        }
        div[role="dialog"] button[aria-label="Close"], [data-testid="stDialogCloseButton"] {
            color: var(--text) !important;
        }

        /* ---------------------------------------------------------------------
           EXPANDERS, TABS, TOOLTIPS, TOASTS, STATUS BOXES
           --------------------------------------------------------------------- */
        /* Expanders */
        [data-testid="stExpander"], details[data-testid="stExpander"] {
            background-color: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            color: var(--text) !important;
        }
        [data-testid="stExpander"] summary {
            color: var(--text) !important;
            font-weight: 600 !important;
        }
        [data-testid="stExpander"] summary:hover {
            color: var(--accent) !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: transparent !important;
            border-bottom: 2px solid var(--border) !important;
            gap: 8px !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: var(--muted) !important;
            font-weight: 600 !important;
            background-color: transparent !important;
            border-radius: 8px 8px 0 0 !important;
            padding: 8px 16px !important;
        }
        .stTabs [aria-selected="true"] {
            color: var(--accent) !important;
            border-bottom: 2px solid var(--accent) !important;
        }

        /* Tooltips */
        [data-baseweb="tooltip"], [data-testid="stTooltipContent"] {
            background-color: var(--surface-card) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            box-shadow: var(--shadow) !important;
        }

        /* Toasts */
        [data-testid="stToast"] {
            background-color: var(--surface-card) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            box-shadow: var(--shadow-lg) !important;
        }

        /* Status box */
        [data-testid="stStatusWidget"] {
            background-color: var(--surface) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
        }

        /* ---------------------------------------------------------------------
           DATAFRAME, TABLES & METRICS
           --------------------------------------------------------------------- */
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            background-color: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
        }
        [data-testid="stDataFrame"] div, [data-testid="stTable"] th, [data-testid="stTable"] td {
            color: var(--text) !important;
        }
        [data-testid="stMetric"] {
            background-color: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px 16px;
        }
        [data-testid="stMetricLabel"] {
            color: var(--muted) !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
        }
        [data-testid="stMetricValue"] {
            color: var(--text) !important;
            font-size: 1.5rem !important;
            font-weight: 800 !important;
        }

        /* ---------------------------------------------------------------------
           ALERT BOXES & CODE BLOCKS
           --------------------------------------------------------------------- */
        .stAlert, [data-testid="stAlert"] {
            background-color: var(--surface) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
        }
        pre, code, [data-testid="stCodeBlock"] {
            background-color: var(--code-bg) !important;
            color: var(--code-text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
        }

        /* ---------------------------------------------------------------------
           IMAGES & PREVIEWS (PREVENT BLURRY STRETCHING)
           --------------------------------------------------------------------- */
        [data-testid="stImage"] img {
            max-height: 250px !important;
            max-width: 280px !important;
            object-fit: contain !important;
            border-radius: 8px !important;
            image-rendering: -webkit-optimize-contrast !important;
            image-rendering: crisp-edges !important;
            box-shadow: var(--shadow) !important;
        }

        /* ---------------------------------------------------------------------
           CUSTOM VISHWAS CARDS, RESULT STRIPS & PILLS
           --------------------------------------------------------------------- */
        .vishwas-card {
            background-color: var(--surface-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px 22px;
            margin-bottom: 16px;
            color: var(--text);
            box-shadow: var(--shadow);
        }
        .vishwas-card-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 8px;
            color: var(--text);
        }

        .result-strip {
            border-radius: 10px;
            padding: 14px 20px;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-weight: 600;
        }
        .result-strip-red {
            background-color: var(--danger-bg);
            border: 1px solid var(--danger-border);
            color: var(--danger-text);
        }
        .result-strip-amber {
            background-color: var(--warn-bg);
            border: 1px solid var(--warn-border);
            color: var(--warn-text);
        }
        .result-strip-green {
            background-color: var(--ok-bg);
            border: 1px solid var(--ok-border);
            color: var(--ok-text);
        }
        .result-strip-grey {
            background-color: var(--grey-bg);
            border: 1px solid var(--grey-border);
            color: var(--grey-text);
        }

        /* Badges & Pills */
        .example-badge {
            display: inline-block;
            background-color: var(--accent);
            color: #ffffff !important;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 6px;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .precomputed-note-pill {
            display: inline-block;
            background-color: var(--info-bg);
            color: var(--info-text) !important;
            border: 1px solid var(--info-border);
            font-size: 0.8rem;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 9999px;
            margin-bottom: 8px;
        }
        .severity-chip {
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 3px 9px;
            border-radius: 9999px;
            margin-right: 6px;
        }
        .severity-cancer {
            background-color: var(--danger-bg);
            color: var(--danger-text);
            border: 1px solid var(--danger-border);
        }
        .severity-precancer {
            background-color: var(--warn-bg);
            color: var(--warn-text);
            border: 1px solid var(--warn-border);
        }
        .severity-benign {
            background-color: var(--ok-bg);
            color: var(--ok-text);
            border: 1px solid var(--ok-border);
        }
        .severity-nontumour {
            background-color: var(--info-bg);
            color: var(--info-text);
            border: 1px solid var(--info-border);
        }
    """

    return f"<style>\n{root_block}\n{component_rules}\n</style>"


def inject_theme(theme_mode: str = "Auto") -> None:
    """Injects the theme CSS into the current Streamlit app session."""
    css = get_theme_css(theme_mode)
    st.markdown(css, unsafe_allow_html=True)


def get_plotly_theme(theme_mode: str) -> Dict[str, Any]:
    """
    Returns Plotly layout configuration parameters adapted to the active theme.
    Handles 'Dark', 'Light', and 'Auto' (adapting via st.context.theme or high-contrast neutral defaults).
    """
    theme_lower = str(theme_mode).strip().lower()

    if theme_lower == "dark":
        is_dark = True
    elif theme_lower == "light":
        is_dark = False
    else:  # auto
        ctx_theme = getattr(getattr(st, "context", None), "theme", None)
        if isinstance(ctx_theme, dict) and ctx_theme.get("type") == "dark":
            is_dark = True
        elif isinstance(ctx_theme, dict) and ctx_theme.get("type") == "light":
            is_dark = False
        else:
            # Neutral contrast that works on transparent backgrounds in both Light & Dark OS modes
            is_dark = False

    font_color = "#f8fafc" if is_dark else "#0f172a"
    grid_color = "#334155" if is_dark else "#e2e8f0"
    
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "family": "Inter, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
            "size": 14,
            "color": font_color,
        },
        "xaxis": {
            "gridcolor": grid_color,
            "zerolinecolor": grid_color,
            "tickfont": {"size": 13, "color": font_color},
        },
        "yaxis": {
            "gridcolor": grid_color,
            "zerolinecolor": grid_color,
            "tickfont": {"size": 13, "color": font_color},
        }
    }
