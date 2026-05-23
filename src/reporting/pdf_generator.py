"""PDF report generator using Matplotlib and ReportLab."""

import io
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

logger = logging.getLogger(__name__)

STRESS_COLORS = {
    "Low":      "#2ECC71",
    "Elevated": "#F39C12",
    "High":     "#E67E22",
    "Extreme":  "#E74C3C",
}

DARK_BG   = "#1a1a2e"
CARD_BG   = "#16213e"
TEXT_COL  = "#e0e0e0"
ACCENT    = "#0f3460"


def _gauge_figure(score: float, stress_class: str, figsize=(4, 2.2)) -> bytes:
    """Render a semicircular gauge for the stress score."""
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    theta_range = np.linspace(np.pi, 0, 300)
    zones = [(0, 25, "#2ECC71"), (25, 50, "#F39C12"), (50, 75, "#E67E22"), (75, 100, "#E74C3C")]
    for lo, hi, col in zones:
        t = np.linspace(np.pi * (1 - lo / 100), np.pi * (1 - hi / 100), 50)
        ax.fill_between(t, 0.6, 1.0, color=col, alpha=0.85)

    needle_angle = np.pi * (1 - score / 100)
    ax.annotate("", xy=(needle_angle, 0.9), xytext=(needle_angle, 0.0),
                arrowprops=dict(arrowstyle="->", color="white", lw=2))

    ax.set_ylim(0, 1.1)
    ax.set_theta_zero_location("E")
    ax.axis("off")
    ax.text(np.pi / 2, -0.25, f"{score:.1f}", ha="center", va="center",
            fontsize=20, color="white", fontweight="bold",
            transform=ax.transData)
    ax.text(np.pi / 2, -0.55, stress_class.upper(), ha="center", va="center",
            fontsize=10, color=STRESS_COLORS.get(stress_class, "white"),
            transform=ax.transData)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _history_figure(
    csi_history: pd.Series,
    predictions: Dict[int, dict],
    figsize=(8, 3),
    min_rows: int = 10,
) -> bytes:
    """CSI history chart with stress zone bands and prediction markers."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)

    tail = csi_history.dropna().tail(504)  # ~2 years
    if len(tail) < min_rows:
        plt.close(fig)
        return b""

    ax.fill_between(tail.index, 0, 25,  alpha=0.12, color="#2ECC71")
    ax.fill_between(tail.index, 25, 50, alpha=0.12, color="#F39C12")
    ax.fill_between(tail.index, 50, 75, alpha=0.12, color="#E67E22")
    ax.fill_between(tail.index, 75, 100, alpha=0.12, color="#E74C3C")

    ax.plot(tail.index, tail.values, color="#4fc3f7", lw=1.5, label="CSI")
    ax.axhline(float(tail.values[-1]), color="white", lw=0.5, ls="--", alpha=0.4)

    # Mark prediction points
    colors_h = {5: "#FFD700", 21: "#FF6B6B", 63: "#98FB98"}
    last_date = tail.index[-1]
    for h, col in colors_h.items():
        pred = predictions.get(h)
        if pred:
            ax.scatter([last_date], [pred["stress_score"]], color=col, zorder=5,
                       s=60, label=f"{h}d fwd: {pred['stress_score']:.0f}")

    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(colors=TEXT_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.set_ylabel("CSI Score", color=TEXT_COL, fontsize=9)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.3,
              labelcolor="white", facecolor=CARD_BG)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _shap_figure(attribution: List[dict], horizon: int, figsize=(6, 3)) -> bytes:
    """Horizontal bar chart of top SHAP contributors."""
    if not attribution:
        return b""
    labels = [a["label"][:35] for a in attribution[:8]]
    values = [a["shap_value"] for a in attribution[:8]]
    dims   = [a["dimension"] for a in attribution[:8]]

    dim_color = {
        "Equity": "#4fc3f7", "Credit": "#FF6B6B", "Rates": "#FFD700",
        "Liquidity": "#98FB98", "FX": "#DA70D6", "International": "#FFA07A",
    }
    bar_colors = [dim_color.get(d, "#aaa") for d in dims]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)

    y_pos = range(len(labels))
    ax.barh(list(y_pos), values, color=bar_colors, alpha=0.85)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8, color=TEXT_COL)
    ax.tick_params(colors=TEXT_COL)
    ax.set_title(f"Top SHAP Drivers — {horizon}d Horizon", color=TEXT_COL, fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.invert_yaxis()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _dimension_radar(dim_scores: dict, figsize=(4, 4)) -> bytes:
    """Radar chart of the 6 CSI dimension scores."""
    dims  = ["Equity", "Credit", "Rates", "Liquidity", "FX", "International"]
    keys  = ["csi_equity", "csi_credit", "csi_rates", "csi_liquidity", "csi_fx", "csi_international"]
    vals  = [float(dim_scores.get(k, 0)) for k in keys]
    vals += vals[:1]

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)

    ax.fill(angles, vals, alpha=0.25, color="#4fc3f7")
    ax.plot(angles, vals, color="#4fc3f7", lw=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, color=TEXT_COL, size=8)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75])
    ax.set_yticklabels(["25", "50", "75"], color="#888", size=7)
    ax.grid(color="#333", alpha=0.6)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_pdf(
    predictions: Dict[int, dict],
    attribution_by_horizon: Dict[int, List[dict]],
    csi: pd.DataFrame,
    narrative: str,
    output_dir: str = "reports",
    as_of: Optional[str] = None,
) -> str:
    """
    Generate the daily stress report PDF.

    Returns the path of the saved PDF.
    """
    os.makedirs(output_dir, exist_ok=True)
    date_str = as_of or datetime.today().strftime("%Y-%m-%d")
    output_path = os.path.join(output_dir, f"stress_report_{date_str}.pdf")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", fontSize=18, textColor=colors.HexColor("#4fc3f7"),
                         spaceAfter=6, alignment=TA_CENTER)
    h2 = ParagraphStyle("H2", fontSize=12, textColor=colors.HexColor("#4fc3f7"),
                         spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("Body", fontSize=9, textColor=colors.HexColor("#cccccc"),
                           leading=14)
    small = ParagraphStyle("Small", fontSize=8, textColor=colors.HexColor("#888888"))

    story = []

    # ── Title ──────────────────────────────────────────────────────────────
    story.append(Paragraph(f"Global Market Stress Report — {date_str}", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#444")))
    story.append(Spacer(1, 8))

    # ── Gauge + Radar side by side ─────────────────────────────────────────
    pred_5d = predictions.get(5, {})
    current_csi = pred_5d.get("current_csi", 0) or 0
    current_class = "Low"
    for thr, cls in [(75, "Extreme"), (50, "High"), (25, "Elevated")]:
        if current_csi > thr:
            current_class = cls
            break

    gauge_bytes = _gauge_figure(current_csi, current_class)
    gauge_img = Image(io.BytesIO(gauge_bytes), width=2.8 * inch, height=1.6 * inch)

    latest_csi_row = csi.iloc[-1] if len(csi) > 0 else pd.Series()
    dim_scores = latest_csi_row.to_dict() if not latest_csi_row.empty else {}
    radar_bytes = _dimension_radar(dim_scores)
    radar_img = Image(io.BytesIO(radar_bytes), width=2.5 * inch, height=2.5 * inch)

    story.append(Table(
        [[gauge_img, radar_img]],
        colWidths=[3.5 * inch, 3.5 * inch],
        style=TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")])
    ))
    story.append(Spacer(1, 6))

    # ── Prediction summary table ───────────────────────────────────────────
    story.append(Paragraph("Multi-Horizon Forecast Summary", h2))
    tbl_data = [["Horizon", "Stress Score", "Class", "Δ Score", "Market Direction"]]
    for h in [5, 21, 63]:
        p = predictions.get(h, {})
        tbl_data.append([
            f"{h}d",
            str(p.get("stress_score", "—")),
            p.get("stress_class", "—"),
            f"{p.get('stress_delta', 0):+.1f}" if p.get("stress_delta") is not None else "—",
            p.get("market_direction", "—") if h < 63 else "N/A",
        ])

    tbl = Table(tbl_data, colWidths=[0.7*inch, 1.1*inch, 1.1*inch, 0.9*inch, 1.4*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#0f3460")),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.HexColor("#4fc3f7")),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#1a1a2e"), colors.HexColor("#16213e")]),
        ("TEXTCOLOR",    (0, 1), (-1, -1), colors.HexColor("#cccccc")),
        ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#444")),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))

    # ── CSI history chart ──────────────────────────────────────────────────
    story.append(Paragraph("CSI Historical Trajectory (2-Year)", h2))
    hist_bytes = _history_figure(csi["csi_composite"].dropna(), predictions)
    if hist_bytes:
        hist_img = Image(io.BytesIO(hist_bytes), width=6.5 * inch, height=2.4 * inch)
        story.append(hist_img)
    else:
        story.append(Paragraph("(Insufficient history data for chart)", small))
    story.append(Spacer(1, 8))

    # ── SHAP charts ────────────────────────────────────────────────────────
    story.append(Paragraph("Key Risk Drivers (SHAP Attribution)", h2))
    shap_row = []
    for h in [5, 21]:
        attr = attribution_by_horizon.get(h, [])
        b = _shap_figure(attr, h)
        if b:
            shap_row.append(Image(io.BytesIO(b), width=3.1 * inch, height=2.0 * inch))
    if shap_row:
        story.append(Table(
            [shap_row],
            colWidths=[3.5 * inch] * len(shap_row),
            style=TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")])
        ))
    story.append(Spacer(1, 8))

    # ── Narrative ──────────────────────────────────────────────────────────
    story.append(Paragraph("Market Intelligence Briefing", h2))
    for para in narrative.strip().split("\n\n"):
        clean = para.replace("###", "").replace("**", "").strip()
        if clean:
            story.append(Paragraph(clean, body))
            story.append(Spacer(1, 6))

    # ── Footer ─────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#444")))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} | "
        "Global Market Stress Indicator v1.0 | For research purposes only.",
        small
    ))

    doc.build(story)
    logger.info("PDF report saved: %s", output_path)
    return output_path
