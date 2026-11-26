"""
VISHWAS Centralized Design & Theme System
Provides light, dark, and auto CSS variable palettes adhering to WCAG AA contrast standards,
and injects styling across native Streamlit widgets (headers, dialogs, buttons, uploaders, tabs).
"""

from __future__ import annotations
from typing import Dict, Any
import streamlit as st


LIGHT_VARS = """
    --bg: #ffffff;
    --surface: #f8fafc;
    --surface-hover: #f1f5f9;
    --text: #0f172a;
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
"""

DARK_VARS = """
    --bg: #0b1120;
    --surface: #1e293b;
    --surface-hover: #334155;
    --text: #f8fafc;
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
    --warn-text: #fef3c7;
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
    --grey-text: #f8fafc;
    --grey-border: #475569;
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

        /* Hero Header Styling - Large, Bold, Multilingual Typography */
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

        /* Top header bar */
        [data-testid="stHeader"], header[data-testid="stHeader"], .stAppHeader {
            background-color: var(--bg) !important;
            color: var(--text) !important;
            border-bottom: 1px solid var(--border) !important;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"], section[data-testid="stSidebar"] {
            background-color: var(--surface) !important;
            border-right: 1px solid var(--border) !important;
            color: var(--text) !important;
        }
        [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
            color: var(--text) !important;
        }

        /* Standard buttons */
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
        button[kind="primary"] {
            background-color: var(--accent) !important;
            color: #ffffff !important;
            border: 1px solid var(--accent) !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        button[kind="primary"]:hover {
            background-color: var(--accent-hover) !important;
        }

        /* File Uploaders */
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

        /* Dialogs / Modals */
        div[role="dialog"], [data-testid="stDialog"], [data-testid="stModal"] {
            background-color: var(--surface) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 10px 10px -5px rgba(0, 0, 0, 0.3) !important;
        }
        div[role="dialog"] p, div[role="dialog"] span, div[role="dialog"] h1, div[role="dialog"] h2, div[role="dialog"] h3, div[role="dialog"] h4 {
            color: var(--text) !important;
        }

        /* Expanders */
        [data-testid="stExpander"] {
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

        /* Inputs, Radios, Segmented controls */
        input, select, textarea, [data-baseweb="select"] {
            background-color: var(--surface) !important;
            color: var(--text) !important;
            border-color: var(--border) !important;
        }
        [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p {
            color: var(--text) !important;
            font-weight: 600 !important;
        }
        [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] span {
            color: var(--text);
        }

        /* Images Preview max-height constraint */
        [data-testid="stImage"] img {
            max-height: 220px !important;
            object-fit: contain !important;
            border-radius: 8px !important;
        }

        /* Custom VISHWAS Component Cards */
        .vishwas-card {
            background-color: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px 22px;
            margin-bottom: 16px;
            color: var(--text);
        }
        .vishwas-card-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 8px;
            color: var(--text);
        }

        .result-strip {
            border-radius: 10px;
            padding: 12px 18px;
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

        /* Badges */
        .example-badge {
            display: inline-block;
            background-color: var(--accent);
            color: #ffffff;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 6px;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
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
    """Returns Plotly layout configuration parameters adapted to the active theme."""
    is_dark = str(theme_mode).strip().lower() == "dark"
    font_color = "#f8fafc" if is_dark else "#0f172a"
    grid_color = "#334155" if is_dark else "#e2e8f0"
    
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "family": "Inter, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
            "size": 14,
            "color": font_color
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
