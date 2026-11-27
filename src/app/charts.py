"""
VISHWAS Interactive Graphical Visualizations
Theme-aware, WCAG AA compliant Plotly charts for:
1. Reliability & Calibration ("When the AI says X% sure, how often is it actually right?")
2. Per-disease Recall Ranking & Interactive Confusion Explorer
3. Subgroup Fairness & Disparity Comparison
4. Top-3 Differential Probability Bar Chart
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
from .theme import get_plotly_theme


def plot_reliability_chart(
    rel_data: Dict[str, Any],
    tone_group: str = "All",
    theme_mode: str = "Auto"
) -> go.Figure:
    """
    Creates an intuitive Plotly calibration bar chart:
    'When the AI says X% sure, how often is it actually right?'
    """
    t_cfg = get_plotly_theme(theme_mode)
    
    if tone_group != "All" and "by_tone_group" in rel_data and tone_group in rel_data["by_tone_group"]:
        bins = rel_data["by_tone_group"][tone_group]["bins"]
        ece_val = rel_data["by_tone_group"][tone_group]["ece"]
        rating = rel_data["by_tone_group"][tone_group]["rating"]
    else:
        bins = rel_data.get("bins", [])
        ece_val = rel_data.get("overall_ece", 0.0)
        rating = rel_data.get("overall_rating", "good")

    bin_labels = [b["bin_label"] for b in bins]
    actual_acc = [b["actual_accuracy"] * 100 for b in bins]
    expected_acc = [b["mean_confidence"] * 100 for b in bins]
    counts = [b["count"] for b in bins]

    fig = go.Figure()

    # Bar chart for actual accuracy
    fig.add_trace(go.Bar(
        x=bin_labels,
        y=actual_acc,
        name="Actual Accuracy (%)",
        marker=dict(
            color="#2563eb" if str(theme_mode).lower() != "dark" else "#3b82f6",
            line=dict(color="#1d4ed8", width=1.5)
        ),
        text=[f"{v:.1f}%" for v in actual_acc],
        textposition="outside",
        hovertemplate=(
            "<b>Confidence Range: %{x}</b><br>" +
            "Actual Correctness: %{y:.1f}%<br>" +
            "Expected: %{customdata[0]:.1f}%<br>" +
            "Lesion Count: %{customdata[1]:,}<extra></extra>"
        ),
        customdata=list(zip(expected_acc, counts))
    ))

    # Reference scatter line for ideal calibration
    fig.add_trace(go.Scatter(
        x=bin_labels,
        y=expected_acc,
        mode="lines+markers",
        name="Expected (Ideal AI)",
        line=dict(color="#10b981", width=3, dash="dot"),
        marker=dict(size=8, color="#10b981")
    ))

    xaxis_dict = dict(t_cfg["xaxis"])
    xaxis_dict["title"] = "When the AI says it is this confident..."
    
    yaxis_dict = dict(t_cfg["yaxis"])
    yaxis_dict["title"] = "Actual Accuracy (% Right)"
    yaxis_dict["range"] = [0, 115]

    fig.update_layout(
        title=dict(
            text=f"Confidence Calibration (Tone: {tone_group}) | ECE: {ece_val:.3f}",
            font=dict(size=16, color=t_cfg["font"]["color"])
        ),
        xaxis=xaxis_dict,
        yaxis=yaxis_dict,
        paper_bgcolor=t_cfg["paper_bgcolor"],
        plot_bgcolor=t_cfg["plot_bgcolor"],
        font=t_cfg["font"],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=t_cfg["font"]["color"])
        ),
        margin=dict(l=40, r=20, t=60, b=40),
        height=380
    )
    return fig


def plot_confusion_recall_chart(
    conf_data: Dict[str, Any],
    theme_mode: str = "Auto"
) -> go.Figure:
    """
    Ranks all 11 conditions by clinical recall (sensitivity) from highest to lowest.
    """
    t_cfg = get_plotly_theme(theme_mode)
    by_disease = conf_data.get("by_disease", {})
    ranked_codes = conf_data.get("ranked_by_recall", list(by_disease.keys()))

    # Sort descending for display (reversed so highest is at top of horizontal bar chart)
    codes = ranked_codes[::-1]
    names = [by_disease[c]["name"] for c in codes]
    recalls = [by_disease[c]["recall_pct"] for c in codes]
    totals = [by_disease[c]["total_cases"] for c in codes]
    corrects = [by_disease[c]["correct_cases"] for c in codes]

    # Gradient or category coloring: high recall green/blue, lower amber
    colors = [
        "#16a34a" if r >= 75 else ("#2563eb" if r >= 50 else "#f59e0b")
        for r in recalls
    ]

    fig = go.Figure(go.Bar(
        x=recalls,
        y=names,
        orientation="h",
        marker=dict(color=colors, line=dict(color="#334155", width=1)),
        text=[f"{r:.1f}%" for r in recalls],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "Diagnostic Recall: %{x:.1f}%<br>" +
            "Correctly Detected: %{customdata[0]:,} of %{customdata[1]:,} cases<extra></extra>"
        ),
        customdata=list(zip(corrects, totals))
    ))

    xaxis_dict = dict(t_cfg["xaxis"])
    xaxis_dict["title"] = "Recall (% of actual cases correctly detected)"
    xaxis_dict["range"] = [0, 115]

    yaxis_dict = dict(t_cfg["yaxis"])
    yaxis_dict["title"] = ""
    yaxis_dict["tickfont"] = dict(size=12, color=t_cfg["font"]["color"])

    fig.update_layout(
        title=dict(
            text="Diagnostic Sensitivity / Recall per Skin Condition",
            font=dict(size=16, color=t_cfg["font"]["color"])
        ),
        xaxis=xaxis_dict,
        yaxis=yaxis_dict,
        paper_bgcolor=t_cfg["paper_bgcolor"],
        plot_bgcolor=t_cfg["plot_bgcolor"],
        font=t_cfg["font"],
        margin=dict(l=160, r=40, t=50, b=40),
        height=420
    )
    return fig


def plot_confusion_detail_chart(
    conf_data: Dict[str, Any],
    disease_code: str,
    theme_mode: str = "Auto"
) -> go.Figure:
    """
    Shows what the AI occasionally confuses the selected condition with.
    """
    t_cfg = get_plotly_theme(theme_mode)
    by_disease = conf_data.get("by_disease", {})
    entry = by_disease.get(disease_code, {})
    
    top_conf = entry.get("top_confusions", [])
    if not top_conf:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=f"No significant confusions recorded for {entry.get('name', disease_code)}"),
            paper_bgcolor=t_cfg["paper_bgcolor"],
            plot_bgcolor=t_cfg["plot_bgcolor"]
        )
        return fig

    # Reverse so top confusion is at top of horizontal chart
    top_conf_rev = top_conf[::-1]
    labels = [c["confused_name"] for c in top_conf_rev]
    percentages = [c["percentage"] for c in top_conf_rev]
    counts = [c["count"] for c in top_conf_rev]

    fig = go.Figure(go.Bar(
        x=percentages,
        y=labels,
        orientation="h",
        marker=dict(color="#f59e0b", line=dict(color="#d97706", width=1.5)),
        text=[f"{p:.1f}% ({cnt} cases)" for p, cnt in zip(percentages, counts)],
        textposition="outside",
        hovertemplate="Mistaken for <b>%{y}</b>: %{x:.1f}% (%{customdata} cases)<extra></extra>",
        customdata=counts
    ))

    max_p = max(percentages) if percentages else 10
    xaxis_dict = dict(t_cfg["xaxis"])
    xaxis_dict["title"] = "% of cases mistaken for this condition"
    xaxis_dict["range"] = [0, max_p * 1.35 + 5]

    yaxis_dict = dict(t_cfg["yaxis"])
    yaxis_dict["title"] = ""

    fig.update_layout(
        title=dict(
            text=f"When mistaken, what is {entry.get('name', disease_code)} confused with?",
            font=dict(size=15, color=t_cfg["font"]["color"])
        ),
        xaxis=xaxis_dict,
        yaxis=yaxis_dict,
        paper_bgcolor=t_cfg["paper_bgcolor"],
        plot_bgcolor=t_cfg["plot_bgcolor"],
        font=t_cfg["font"],
        margin=dict(l=140, r=40, t=50, b=40),
        height=260
    )
    return fig


def plot_fairness_bars(
    fairness_data: Dict[str, Any],
    theme_mode: str = "Auto"
) -> go.Figure:
    """
    Subgroup fairness comparison: Accuracy and Macro-F1 across skin tone groups G1, G2, G3.
    """
    t_cfg = get_plotly_theme(theme_mode)
    by_tone = fairness_data.get("by_tone_group", {})
    
    groups = ["G1 (Light)", "G2 (Medium)", "G3 (Darker)"]
    keys = ["G1", "G2", "G3"]
    
    accuracies = [by_tone.get(k, {}).get("accuracy", 0.0) * 100 for k in keys]
    f1s = [by_tone.get(k, {}).get("macro_f1", 0.0) * 100 for k in keys]
    counts = [by_tone.get(k, {}).get("count", 0) for k in keys]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=groups,
        y=accuracies,
        name="Accuracy (%)",
        marker=dict(color="#2563eb", line=dict(color="#1d4ed8", width=1.5)),
        text=[f"{a:.1f}%" for a in accuracies],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.1f}%<br>Sample Size: %{customdata:,}<extra></extra>",
        customdata=counts
    ))

    fig.add_trace(go.Bar(
        x=groups,
        y=f1s,
        name="Macro F1-Score (%)",
        marker=dict(color="#10b981", line=dict(color="#059669", width=1.5)),
        text=[f"{f:.1f}%" for f in f1s],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Balanced F1: %{y:.1f}%<br>Sample Size: %{customdata:,}<extra></extra>",
        customdata=counts
    ))

    xaxis_dict = dict(t_cfg["xaxis"])
    xaxis_dict["title"] = "Skin Tone Subgroup"

    yaxis_dict = dict(t_cfg["yaxis"])
    yaxis_dict["title"] = "Score (%)"
    yaxis_dict["range"] = [0, 115]

    fig.update_layout(
        barmode="group",
        title=dict(
            text="Performance Parity Across Skin Tone Groups",
            font=dict(size=16, color=t_cfg["font"]["color"])
        ),
        xaxis=xaxis_dict,
        yaxis=yaxis_dict,
        paper_bgcolor=t_cfg["paper_bgcolor"],
        plot_bgcolor=t_cfg["plot_bgcolor"],
        font=t_cfg["font"],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=t_cfg["font"]["color"])
        ),
        margin=dict(l=40, r=20, t=60, b=40),
        height=380
    )
    return fig


def plot_top3_horizontal(
    top3_list: List[Dict[str, Any]],
    theme_mode: str = "Auto",
    lang: str = "en"
) -> go.Figure:
    """
    Horizontal bar chart for Top-3 differential predictions.
    """
    t_cfg = get_plotly_theme(theme_mode)
    
    if not top3_list:
        return go.Figure()

    # Invert order for top-to-bottom display
    rev_top3 = top3_list[::-1]
    names = []
    for x in rev_top3:
        code = x.get("code") or x.get("class", "")
        full_name = x.get("name", code)
        if full_name and full_name != code:
            names.append(f"{code} — {full_name}")
        else:
            names.append(code)
    probs = [x.get("prob", 0.0) * 100 for x in rev_top3]

    colors = ["#2563eb"] * len(names)
    if len(colors) >= 3:
        colors[-1] = "#10b981"  # Top 1 highlighted in green

    fig = go.Figure(go.Bar(
        x=probs,
        y=names,
        orientation="h",
        marker=dict(color=colors, line=dict(color=t_cfg.get("line_color", "#222222"), width=1)),
        text=[f"{p:.1f}%" for p in probs],
        textposition="auto",
        insidetextanchor="end",
        cliponaxis=False,
        textfont=dict(size=13, color=t_cfg["font"]["color"]),
        hovertemplate="<b>%{y}</b>: %{x:.1f}%<extra></extra>"
    ))

    max_p = max(probs) if probs else 50
    xaxis_dict = dict(t_cfg["xaxis"])
    xaxis_dict["title"] = "Probability (%)"
    xaxis_dict["range"] = [0, max(100.0, min(108.0, max_p * 1.15 + 8))]

    yaxis_dict = dict(t_cfg["yaxis"])
    yaxis_dict["title"] = ""
    yaxis_dict["tickfont"] = dict(size=13, color=t_cfg["font"]["color"])

    fig.update_layout(
        xaxis=xaxis_dict,
        yaxis=yaxis_dict,
        paper_bgcolor=t_cfg["paper_bgcolor"],
        plot_bgcolor=t_cfg["plot_bgcolor"],
        font=t_cfg["font"],
        margin=dict(l=150, r=65, t=20, b=30),
        height=210
    )
    return fig
