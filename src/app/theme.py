"""
VISHWAS Centralized Design & Theme System
Provides:
- WCAG AA compliant true-black dark theme (#000000 page/sidebar, #0B0B0B-#111111 surfaces, #222222 borders)
- Calm blue (#3B82F6) accent replacing default Streamlit red across Dark and Light modes
- Standardized typography: 16px base font, 1.5 line-height, readable hierarchy
- Comprehensive styling across all native Streamlit widgets (file uploader, camera, radio, tabs, modals, tables)
- Transparent Plotly chart themes with #222222 dark grid lines
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import streamlit as st


LIGHT_VARS = """
    --bg: #ffffff;
    --sidebar-bg: #f8fafc;
    --header-bg: #ffffff;
    --surface: #f8fafc;
    --surface-hover: #f1f5f9;
    --surface-card: #ffffff;
    --text: #0f172a;
    --text-strong: #020617;
    --muted: #64748b;
    --border: #e2e8f0;
    --border-hover: #cbd5e1;
    --accent: #3b82f6;
    --accent-hover: #2563eb;
    --accent-focus: rgba(59, 130, 246, 0.25);
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
    --grid: #e2e8f0;
    --code-bg: #f1f5f9;
    --code-text: #0f172a;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.06), 0 2px 4px -2px rgba(0, 0, 0, 0.04);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.06);
"""

DARK_VARS = """
    --bg: #000000;
    --sidebar-bg: #000000;
    --header-bg: #000000;
    --surface: #0b0b0b;
    --surface-hover: #141414;
    --surface-card: #101010;
    --text: #ffffff;
    --text-strong: #ffffff;
    --muted: #a3a3a3;
    --border: #222222;
    --border-hover: #333333;
    --accent: #3b82f6;
    --accent-hover: #60a5fa;
    --accent-focus: rgba(59, 130, 246, 0.25);
    --danger: #ef4444;
    --danger-bg: #1c0a0a;
    --danger-text: #fca5a5;
    --danger-border: #7f1d1d;
    --warn: #f59e0b;
    --warn-bg: #1c1303;
    --warn-text: #fde68a;
    --warn-border: #78350f;
    --ok: #10b981;
    --ok-bg: #061e12;
    --ok-text: #a7f3d0;
    --ok-border: #065f46;
    --info: #3b82f6;
    --info-bg: #081a36;
    --info-text: #bfdbfe;
    --info-border: #1e3a8a;
    --grey-bg: #141414;
    --grey-text: #e5e5e5;
    --grey-border: #262626;
    --grid: #222222;
    --code-bg: #0b0b0b;
    --code-text: #ffffff;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.8), 0 2px 4px -2px rgba(0, 0, 0, 0.6);
    --shadow-lg: 0 16px 24px -4px rgba(0, 0, 0, 0.9), 0 8px 10px -6px rgba(0, 0, 0, 0.8);
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
        /* ---------------------------------------------------------------------
           GLOBAL BASE TYPOGRAPHY & BODY
           --------------------------------------------------------------------- */
        html, body, .stApp {
            background-color: var(--bg) !important;
            color: var(--text) !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
            font-size: 16px !important;
            line-height: 1.5 !important;
            -webkit-font-smoothing: antialiased;
        }

        /* Centered content container for Simple mode */
        .simple-mode-container {
            max-width: 1100px;
            margin: 0 auto;
            padding: 0 8px;
        }

        /* Headings hierarchy */
        h1, .stMarkdown h1 {
            font-size: 1.85rem !important;
            font-weight: 800 !important;
            line-height: 1.25 !important;
            color: var(--text) !important;
            margin-top: 0.8rem !important;
            margin-bottom: 0.6rem !important;
        }
        h2, .stMarkdown h2 {
            font-size: 1.5rem !important;
            font-weight: 750 !important;
            line-height: 1.3 !important;
            color: var(--text) !important;
            margin-top: 0.8rem !important;
            margin-bottom: 0.5rem !important;
        }
        h3, .stMarkdown h3 {
            font-size: 1.25rem !important;
            font-weight: 700 !important;
            color: var(--text) !important;
            margin-top: 0.7rem !important;
            margin-bottom: 0.4rem !important;
        }
        h4, .stMarkdown h4 {
            font-size: 1.1rem !important;
            font-weight: 650 !important;
            color: var(--text) !important;
            margin-top: 0.6rem !important;
            margin-bottom: 0.35rem !important;
        }
        p, .stMarkdown p {
            font-size: 1rem !important;
            line-height: 1.55 !important;
            color: var(--text) !important;
        }

        /* ---------------------------------------------------------------------
           HEADER & TOP BAR
           --------------------------------------------------------------------- */
        [data-testid="stHeader"], header[data-testid="stHeader"], .stAppHeader {
            background-color: var(--header-bg) !important;
            border-bottom: 1px solid var(--border) !important;
            box-shadow: none !important;
        }
        [data-testid="stToolbar"] {
            background-color: var(--header-bg) !important;
        }

        /* ---------------------------------------------------------------------
           SIDEBAR
           --------------------------------------------------------------------- */
        [data-testid="stSidebar"], section[data-testid="stSidebar"] {
            background-color: var(--sidebar-bg) !important;
            border-right: 1px solid var(--border) !important;
        }
        [data-testid="stSidebar"] .stMarkdown p {
            color: var(--text) !important;
            font-size: 0.95rem !important;
        }

        /* ---------------------------------------------------------------------
           CARDS & SURFACES & DIALOGS
           --------------------------------------------------------------------- */
        .vishwas-card, div[role="dialog"] {
            background-color: var(--surface-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            padding: 22px !important;
            margin-bottom: 16px !important;
            box-shadow: var(--shadow) !important;
        }

        /* ---------------------------------------------------------------------
           BUTTONS (Large 44px+ touch targets, calm blue accent)
           --------------------------------------------------------------------- */
        button[kind="primary"], [data-testid="stBaseButton-primary"] {
            background-color: var(--accent) !important;
            border: 1px solid var(--accent) !important;
            color: #ffffff !important;
            font-size: 1.05rem !important;
            font-weight: 700 !important;
            min-height: 48px !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            transition: all 0.15s ease-in-out !important;
        }
        button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
            background-color: var(--accent-hover) !important;
            border-color: var(--accent-hover) !important;
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.35) !important;
        }
        button[kind="primary"]:disabled, [data-testid="stBaseButton-primary"]:disabled {
            background-color: var(--border) !important;
            border-color: var(--border) !important;
            color: var(--muted) !important;
            cursor: not-allowed !important;
            opacity: 0.6 !important;
        }

        button[kind="secondary"], [data-testid="stBaseButton-secondary"] {
            background-color: var(--surface) !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            min-height: 44px !important;
            border-radius: 8px !important;
            padding: 8px 18px !important;
        }
        button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover {
            background-color: var(--surface-hover) !important;
            border-color: var(--accent) !important;
            color: var(--accent) !important;
        }

        /* Primary action button in Simple Mode */
        .btn-check-my-photo button {
            font-size: 1.15rem !important;
            font-weight: 800 !important;
            min-height: 52px !important;
            letter-spacing: 0.01em !important;
        }

        /* ---------------------------------------------------------------------
           FOCUS RINGS & OUTLINES (Calm Blue, Never Red)
           --------------------------------------------------------------------- */
        *:focus-visible, input:focus, select:focus, textarea:focus, button:focus {
            outline: 2px solid var(--accent) !important;
            outline-offset: 1px !important;
        }

        /* ---------------------------------------------------------------------
           RADIO BUTTONS & SEGMENTED CONTROLS (Calm Blue, Never Red)
           --------------------------------------------------------------------- */
        div[data-testid="stRadio"] div[role="radiogroup"] label {
            color: var(--text) !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
            color: var(--text) !important;
            font-size: 1rem !important;
        }
        div[data-testid="stRadio"] input[type="radio"]:checked + div {
            border-color: var(--accent) !important;
            background-color: var(--accent) !important;
        }
        div[data-testid="stRadio"] input[type="radio"] + div {
            border-color: var(--border) !important;
        }

        /* Segmented control buttons */
        [data-testid="stSegmentedControl"] button[aria-selected="true"] {
            background-color: var(--accent) !important;
            color: #ffffff !important;
            border-color: var(--accent) !important;
        }

        /* ---------------------------------------------------------------------
           FILE UPLOADER (Neutral file chips, No Red icons)
           --------------------------------------------------------------------- */
        [data-testid="stFileUploader"] {
            background-color: transparent !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            border: 2px dashed var(--border) !important;
            background-color: var(--surface) !important;
            border-radius: 10px !important;
            padding: 24px !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--accent) !important;
            background-color: var(--surface-hover) !important;
        }
        [data-testid="stFileUploaderDropzone"] svg {
            fill: var(--accent) !important;
            color: var(--accent) !important;
        }
        [data-testid="stFileUploaderFileChip"] {
            background-color: var(--surface-hover) !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
            border-radius: 6px !important;
        }
        [data-testid="stFileUploaderFileChip"] svg {
            fill: var(--text) !important;
            color: var(--text) !important;
        }

        /* ---------------------------------------------------------------------
           TABS
           --------------------------------------------------------------------- */
        div[data-baseweb="tab-list"] {
            background-color: transparent !important;
            border-bottom: 1px solid var(--border) !important;
            gap: 12px !important;
        }
        button[data-baseweb="tab"] {
            color: var(--muted) !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            padding: 10px 16px !important;
            background-color: transparent !important;
            border: none !important;
        }
        button[data-baseweb="tab"]:hover {
            color: var(--text) !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--accent) !important;
            border-bottom: 2px solid var(--accent) !important;
        }
        div[data-baseweb="tab-highlight"] {
            background-color: var(--accent) !important;
        }
        div[data-baseweb="tab-border"] {
            background-color: var(--border) !important;
        }

        /* ---------------------------------------------------------------------
           EXPANDERS
           --------------------------------------------------------------------- */
        details[data-testid="stExpander"] {
            background-color: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            box-shadow: none !important;
            margin-bottom: 12px !important;
        }
        details[data-testid="stExpander"] summary {
            color: var(--text) !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            padding: 12px 18px !important;
        }
        details[data-testid="stExpander"] summary:hover {
            color: var(--accent) !important;
        }
        details[data-testid="stExpander"] summary svg {
            fill: var(--text) !important;
        }

        /* ---------------------------------------------------------------------
           SELECTBOX & DROPDOWN MENUS
           --------------------------------------------------------------------- */
        div[data-baseweb="select"] > div {
            background-color: var(--surface) !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
            border-radius: 8px !important;
            min-height: 44px !important;
        }
        div[data-baseweb="select"] span {
            color: var(--text) !important;
        }
        div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
            background-color: var(--surface) !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
        }
        li[role="option"] {
            background-color: var(--surface) !important;
            color: var(--text) !important;
            font-size: 0.95rem !important;
            padding: 10px 14px !important;
        }
        li[role="option"]:hover, li[role="option"][aria-selected="true"] {
            background-color: var(--surface-hover) !important;
            color: var(--accent) !important;
        }

        /* ---------------------------------------------------------------------
           METRICS & DATAFRAMES
           --------------------------------------------------------------------- */
        [data-testid="stMetric"] {
            background-color: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            padding: 14px 18px !important;
        }
        [data-testid="stMetricLabel"] {
            color: var(--muted) !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
        }
        [data-testid="stMetricValue"] {
            color: var(--text) !important;
            font-size: 1.5rem !important;
            font-weight: 800 !important;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
        }

        /* ---------------------------------------------------------------------
           STEP INDICATOR
           --------------------------------------------------------------------- */
        .step-progress-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 24px;
            padding: 14px 20px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
        }
        .step-item {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--muted);
        }
        .step-item.active {
            color: var(--accent);
            font-weight: 750;
        }
        .step-circle {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.85rem;
            font-weight: 750;
            background: var(--border);
            color: var(--text);
        }
        .step-item.active .step-circle {
            background: var(--accent);
            color: #ffffff;
        }
        .step-item.completed .step-circle {
            background: var(--ok);
            color: #ffffff;
        }

        /* ---------------------------------------------------------------------
           VERDICT CARDS
           --------------------------------------------------------------------- */
        .verdict-card {
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            border-left: 6px solid var(--accent);
            background: var(--surface-card);
            border: 1px solid var(--border);
            border-left-width: 6px;
        }
        .verdict-card-red {
            border-left-color: var(--danger);
            background: var(--danger-bg);
        }
        .verdict-card-amber {
            border-left-color: var(--warn);
            background: var(--warn-bg);
        }
        .verdict-card-green {
            border-left-color: var(--ok);
            background: var(--ok-bg);
        }

        /* Severity chip */
        .severity-badge {
            display: inline-block;
            font-size: 0.85rem;
            font-weight: 750;
            padding: 4px 12px;
            border-radius: 9999px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .severity-badge-red {
            background-color: var(--danger);
            color: #ffffff;
        }
        .severity-badge-amber {
            background-color: var(--warn);
            color: #000000;
        }
        .severity-badge-green {
            background-color: var(--ok);
            color: #ffffff;
        }

        /* Provenance pill */
        .provenance-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.82rem;
            font-weight: 600;
            padding: 4px 12px;
            border-radius: 9999px;
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--muted);
            margin-bottom: 12px;
        }

        /* ---------------------------------------------------------------------
           SLIM NOTICE BANNER
           --------------------------------------------------------------------- */
        .slim-disclaimer {
            font-size: 0.85rem;
            padding: 8px 14px;
            border-radius: 8px;
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--muted);
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* ---------------------------------------------------------------------
           EXPERT MODE TYPOGRAPHY PRESERVATION (No tiny 11px text)
           --------------------------------------------------------------------- */
        .expert-container, [data-testid="stMarkdownContainer"] table {
            font-size: 14px !important;
            line-height: 1.5 !important;
        }
        [data-testid="stMarkdownContainer"] th {
            font-size: 14px !important;
            font-weight: 700 !important;
            background: var(--surface) !important;
            color: var(--text) !important;
        }
        [data-testid="stMarkdownContainer"] td {
            font-size: 14px !important;
            color: var(--text) !important;
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
    Dark mode uses true black transparent canvas with #222222 grid lines and white text.
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
            is_dark = False

    font_color = "#ffffff" if is_dark else "#0f172a"
    grid_color = "#222222" if is_dark else "#e2e8f0"

    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "family": "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
            "size": 13,
            "color": font_color,
        },
        "xaxis": {
            "gridcolor": grid_color,
            "zerolinecolor": grid_color,
            "tickfont": {"size": 12, "color": font_color},
        },
        "yaxis": {
            "gridcolor": grid_color,
            "zerolinecolor": grid_color,
            "tickfont": {"size": 13, "color": font_color},
        },
        "line_color": "#333333" if is_dark else "#e2e8f0",
    }
