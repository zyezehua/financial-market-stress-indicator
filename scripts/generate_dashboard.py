"""
Comprehensive interactive HTML dashboard.

Includes four tabs:
  Snapshot   — current stress gauge, radar, forecast cards, direction chart
  History    — full 2000-present CSI with episode annotations & regime overlay
  Backtest   — in-sample predicted vs actual + OOS performance metrics
  Calendar   — monthly stress heatmap

Usage:
    python scripts/generate_dashboard.py
    python scripts/generate_dashboard.py --date 2025-08-01
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.features.builder import load_features, add_regime_features
from src.labels.composite_index import load_csi
from src.models.predictor import predict_latest, predict_historical
from src.labels.calibrator import STRESS_EPISODES

# ── OOS walk-forward results (h=5d, run 2026-05-23) ─────────────────────────
OOS_METRICS_5D = {
    "n": 6120, "mae": 4.38, "rmse": 5.66, "pearson_r": 0.890,
    "cls_acc": 0.678, "dir_acc": 0.42,
    "stress_cls": {
        "Low":      {"precision": 0.70, "recall": 0.82, "f1": 0.76, "support": 2000},
        "Elevated": {"precision": 0.33, "recall": 0.19, "f1": 0.24, "support":  705},
        "High":     {"precision": 0.30, "recall": 0.23, "f1": 0.26, "support":  903},
        "Extreme":  {"precision": 0.80, "recall": 0.86, "f1": 0.83, "support": 2512},
    },
    "dir_cls": {
        "Down":    {"precision": 0.23, "recall": 0.28, "f1": 0.25, "support": 1378},
        "Neutral": {"precision": 0.56, "recall": 0.48, "f1": 0.52, "support": 2625},
        "Up":      {"precision": 0.43, "recall": 0.44, "f1": 0.43, "support": 2117},
    },
}

# ── Style constants ──────────────────────────────────────────────────────────
BG       = "#1a1a2e"
CARD_BG  = "#16213e"
BORDER   = "#0f3460"
TEXT     = "#e0e0e0"
ACCENT   = "#4fc3f7"

STRESS_COLORS = {"Low": "#2ECC71", "Elevated": "#F39C12", "High": "#E67E22", "Extreme": "#E74C3C"}
DIM_COLORS = {
    "csi_equity":        "#4fc3f7",
    "csi_credit":        "#FF6B6B",
    "csi_rates":         "#FFD700",
    "csi_liquidity":     "#98FB98",
    "csi_fx":            "#DA70D6",
    "csi_international": "#FFA07A",
}
DIM_LABELS = {
    "csi_equity": "Equity", "csi_credit": "Credit", "csi_rates": "Rates",
    "csi_liquidity": "Liquidity", "csi_fx": "FX", "csi_international": "International",
}

DARK = dict(
    paper_bgcolor=BG, plot_bgcolor=CARD_BG,
    font=dict(color=TEXT, family="Inter, Arial, sans-serif"),
    margin=dict(l=50, r=30, t=50, b=40),
)


# ── Chart builders ───────────────────────────────────────────────────────────

def _gauge(score: float, stress_class: str) -> go.Figure:
    color = STRESS_COLORS.get(stress_class, "#888")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": TEXT,
                     "tickvals": [0, 25, 50, 75, 100]},
            "bar": {"color": color, "thickness": 0.25},
            "steps": [
                {"range": [0,  25], "color": "rgba(46,204,113,0.2)"},
                {"range": [25, 50], "color": "rgba(243,156,18,0.2)"},
                {"range": [50, 75], "color": "rgba(230,126,34,0.2)"},
                {"range": [75,100], "color": "rgba(231,76,60,0.2)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "value": score},
        },
        number={"font": {"color": color, "size": 44}},
        title={"text": f"Composite Stress Index<br><b style='color:{color}'>{stress_class.upper()}</b>",
               "font": {"color": TEXT, "size": 13}},
    ))
    fig.update_layout(**DARK, height=300)
    return fig


def _radar(latest_row: dict) -> go.Figure:
    dims  = list(DIM_LABELS.keys())
    vals  = [float(latest_row.get(d, 0)) for d in dims]
    names = [DIM_LABELS[d] for d in dims]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=names + [names[0]],
        fill="toself", fillcolor="rgba(79,195,247,0.15)",
        line=dict(color=ACCENT, width=2),
        hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
    ))
    fig.update_layout(
        **DARK,
        polar=dict(
            bgcolor=CARD_BG,
            radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75],
                            tickfont={"size": 8}, gridcolor="#333"),
            angularaxis=dict(tickfont={"size": 10}, gridcolor="#333"),
        ),
        title="Stress by Dimension",
        height=300, showlegend=False,
    )
    return fig


def _direction_chart(predictions: dict) -> go.Figure:
    horizons = [h for h in [5, 21, 63] if predictions.get(h, {}).get("market_dir_proba")]
    if not horizons:
        return go.Figure()
    fig = go.Figure()
    opacities = {5: 1.0, 21: 0.7, 63: 0.45}
    for h in horizons:
        proba = predictions[h]["market_dir_proba"]
        fig.add_trace(go.Bar(
            x=["Down", "Neutral", "Up"],
            y=[proba["Down"], proba["Neutral"], proba["Up"]],
            name=f"{h}d horizon",
            marker_color=["#E74C3C", "#F39C12", "#2ECC71"],
            opacity=opacities.get(h, 0.7),
        ))
    fig.update_layout(
        **DARK, title="Market Direction Probabilities",
        yaxis=dict(title="Probability", tickformat=".0%", range=[0, 1], gridcolor="#333"),
        xaxis=dict(gridcolor="#333"),
        barmode="group", height=280,
        legend=dict(bgcolor="rgba(22,33,62,0.8)", bordercolor="#444", borderwidth=1),
    )
    return fig


def _csi_3y(csi: pd.DataFrame, predictions: dict) -> go.Figure:
    tail = csi.tail(756)
    fig = _csi_line_base(tail, show_episodes=False, range_slider=False)
    # Forward prediction markers
    last_date = tail.index[-1]
    colors = {5: "#FFD700", 21: "#FF6B6B", 63: "#98FB98"}
    for h, col in colors.items():
        p = predictions.get(h)
        if p:
            fig.add_trace(go.Scatter(
                x=[last_date], y=[p["stress_score"]],
                mode="markers+text",
                marker=dict(color=col, size=12, symbol="diamond", line=dict(color="white", width=1)),
                text=[f"{h}d: {p['stress_score']:.0f}"],
                textposition="top center",
                name=f"{h}d forecast",
                hovertemplate=f"{h}d Forecast: {p['stress_score']:.1f} ({p['stress_class']})<extra></extra>",
            ))
    fig.update_layout(title="Composite Stress Index — 3-Year History", height=350)
    return fig


def _csi_line_base(data: pd.DataFrame, show_episodes: bool = True,
                   range_slider: bool = True) -> go.Figure:
    fig = go.Figure()
    # Zone bands as scatter fills (same approach as html_dashboard.py — avoids add_hrect rendering artifacts)
    zone_kw = dict(line_width=0, showlegend=False, hoverinfo="skip", mode="lines")
    fig.add_trace(go.Scatter(x=data.index, y=[25]*len(data),
                             fill="tozeroy", fillcolor="rgba(46,204,113,0.07)", **zone_kw))
    fig.add_trace(go.Scatter(x=data.index, y=[50]*len(data),
                             fill="tonexty", fillcolor="rgba(243,156,18,0.07)", **zone_kw))
    fig.add_trace(go.Scatter(x=data.index, y=[75]*len(data),
                             fill="tonexty", fillcolor="rgba(230,126,34,0.07)", **zone_kw))
    fig.add_trace(go.Scatter(x=data.index, y=[100]*len(data),
                             fill="tonexty", fillcolor="rgba(231,76,60,0.08)", **zone_kw))

    if show_episodes:
        for ep_name, (start, end) in STRESS_EPISODES.items():
            try:
                fig.add_vrect(
                    x0=pd.Timestamp(start), x1=pd.Timestamp(end),
                    fillcolor="rgba(231,76,60,0.12)", line_width=0,
                    layer="below",  # render behind all traces, not above them
                )
            except Exception:
                pass

    # Composite line — use .tolist() to force plain JSON array (avoids Plotly typed-array bdata format
    # which breaks when manually JSON.parse'd and passed to Plotly.newPlot)
    fig.add_trace(go.Scatter(
        x=data.index.tolist(), y=data["csi_composite"].tolist(),
        name="CSI Composite", line=dict(color=ACCENT, width=2),
        hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}<extra></extra>",
    ))

    # Dimension lines (hidden by default)
    for col, label in DIM_LABELS.items():
        if col in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index.tolist(), y=data[col].tolist(), name=label,
                line=dict(color=DIM_COLORS[col], width=1, dash="dot"),
                opacity=0.6, visible="legendonly",
                hovertemplate=f"{label} %{{x|%Y-%m-%d}}: %{{y:.1f}}<extra></extra>",
            ))

    xaxis_cfg = dict(showgrid=False, gridcolor="#333")
    if range_slider:
        xaxis_cfg["rangeslider"] = dict(visible=True, bgcolor=CARD_BG, thickness=0.05)
        xaxis_cfg["rangeselector"] = dict(
            buttons=[
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor=CARD_BG, activecolor=ACCENT, font=dict(color=TEXT, size=10),
        )

    fig.update_layout(
        **DARK,
        yaxis=dict(range=[0, 100], tickvals=[0, 25, 50, 75, 100], gridcolor="#333"),
        xaxis=xaxis_cfg,
        height=500,
        legend=dict(bgcolor="rgba(22,33,62,0.8)", bordercolor="#444", borderwidth=1,
                    orientation="h", y=-0.15 if range_slider else 1.02),
    )
    return fig


def _full_history(csi: pd.DataFrame) -> go.Figure:
    fig = _csi_line_base(csi, show_episodes=True, range_slider=True)

    # Regime mean overlay (toggleable)
    if "regime_csi_mean_252d" in csi.columns:
        fig.add_trace(go.Scatter(
            x=csi.index.tolist(), y=csi["regime_csi_mean_252d"].tolist(),
            name="Regime Mean (252d)", line=dict(color="#FF69B4", width=1.5, dash="dash"),
            visible="legendonly",
            hovertemplate="Regime Mean 252d %{x|%Y-%m-%d}: %{y:.1f}<extra></extra>",
        ))

    fig.update_layout(title="Composite Stress Index — Full History (2000–present)", height=550)
    return fig


def _backtest_chart(hist: dict) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=["5-Day Horizon", "21-Day Horizon", "63-Day Horizon"],
        row_heights=[1, 1, 1], vertical_spacing=0.06,
    )
    colors_pred = {5: "#FFD700", 21: "#FF6B6B", 63: "#98FB98"}
    for i, h in enumerate([5, 21, 63], 1):
        df = hist.get(h)
        if df is None:
            continue
        df = df.dropna(subset=["stress_actual"])
        fig.add_trace(go.Scatter(
            x=df.index.tolist(), y=df["stress_actual"].tolist(), name="Actual CSI",
            line=dict(color=ACCENT, width=1.2), opacity=0.9,
            showlegend=(i == 1),
            hovertemplate="%{x|%Y-%m-%d} Actual: %{y:.1f}<extra></extra>",
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=df.index.tolist(), y=df["stress_score_pred"].tolist(),
            name=f"{h}d Predicted",
            line=dict(color=colors_pred[h], width=1.2, dash="dash"), opacity=0.85,
            hovertemplate=f"%{{x|%Y-%m-%d}} Pred {h}d: %{{y:.1f}}<extra></extra>",
        ), row=i, col=1)
        for ep_name, (start, end) in STRESS_EPISODES.items():
            try:
                fig.add_vrect(
                    x0=pd.Timestamp(start), x1=pd.Timestamp(end),
                    fillcolor="rgba(231,76,60,0.08)", line_width=0,
                    layer="below", row=i, col=1,
                )
            except Exception:
                pass
        # Y-axis range
        fig.update_yaxes(range=[0, 100], row=i, col=1, gridcolor="#333")

    fig.update_layout(
        **DARK, height=700, title="In-Sample Backtest: Predicted vs Actual CSI",
        legend=dict(bgcolor="rgba(22,33,62,0.8)", bordercolor="#444", borderwidth=1),
    )
    fig.update_xaxes(showgrid=False, gridcolor="#333")
    return fig


def _perf_chart() -> go.Figure:
    m = OOS_METRICS_5D
    classes = list(m["stress_cls"].keys())
    colors  = [STRESS_COLORS[c] for c in classes]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Stress Classification (OOS h=5d)", "Market Direction (OOS h=5d)"],
        horizontal_spacing=0.12,
    )
    metrics = ["precision", "recall", "f1"]
    metric_colors = ["#4fc3f7", "#FFD700", "#98FB98"]

    for j, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric.capitalize(),
            x=classes,
            y=[m["stress_cls"][c][metric] for c in classes],
            marker_color=metric_colors[j],
            opacity=0.85,
            hovertemplate=f"{metric.capitalize()}: %{{y:.2f}}<extra></extra>",
        ), row=1, col=1)

    dir_classes = list(m["dir_cls"].keys())
    for j, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric.capitalize(), x=dir_classes,
            y=[m["dir_cls"][c][metric] for c in dir_classes],
            marker_color=metric_colors[j],
            opacity=0.85,
            showlegend=False,
            hovertemplate=f"{metric.capitalize()}: %{{y:.2f}}<extra></extra>",
        ), row=1, col=2)

    fig.update_layout(
        **DARK, height=340, barmode="group",
        legend=dict(bgcolor="rgba(22,33,62,0.8)", bordercolor="#444", borderwidth=1),
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%", gridcolor="#333")
    fig.update_xaxes(gridcolor="#333")
    return fig


def _error_dist(hist: dict) -> go.Figure:
    fig = go.Figure()
    colors = {5: "#FFD700", 21: "#FF6B6B", 63: "#98FB98"}
    for h in [5, 21, 63]:
        df = hist.get(h)
        if df is None:
            continue
        err = (df["stress_score_pred"] - df["stress_actual"]).dropna()
        fig.add_trace(go.Histogram(
            x=err, name=f"{h}d", nbinsx=60,
            marker_color=colors[h], opacity=0.6,
            hovertemplate=f"{h}d error: %{{x:.1f}}<extra></extra>",
        ))
    fig.update_layout(
        **DARK, height=280, barmode="overlay",
        title="Prediction Error Distribution (In-Sample)",
        xaxis=dict(title="Pred − Actual", gridcolor="#333"),
        yaxis=dict(title="Count", gridcolor="#333"),
        legend=dict(bgcolor="rgba(22,33,62,0.8)", bordercolor="#444", borderwidth=1),
    )
    return fig


def _calendar_heatmap(csi: pd.DataFrame) -> go.Figure:
    df = csi[["csi_composite"]].copy()
    df["year"]  = df.index.year
    df["month"] = df.index.month
    monthly = df.groupby(["year", "month"])["csi_composite"].mean().reset_index()
    years   = sorted(monthly["year"].unique())
    months  = list(range(1, 13))
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    z = []
    text = []
    for yr in years:
        row_z = []
        row_t = []
        for mo in months:
            val = monthly[(monthly["year"] == yr) & (monthly["month"] == mo)]["csi_composite"]
            v = float(val.iloc[0]) if len(val) else None
            row_z.append(v)
            row_t.append(f"{yr}-{month_labels[mo-1]}: {v:.1f}" if v is not None else "")
        z.append(row_z)
        text.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=z, x=month_labels, y=years,
        text=text, hoverinfo="text",
        colorscale=[
            [0.00, "#1a3a1a"],
            [0.25, "#2ECC71"],
            [0.50, "#F39C12"],
            [0.75, "#E67E22"],
            [1.00, "#E74C3C"],
        ],
        zmin=0, zmax=100,
        colorbar=dict(
            title=dict(text="CSI", font=dict(color=TEXT)),
            tickvals=[0,25,50,75,100], ticktext=["0","25","50","75","100"],
            tickfont=dict(color=TEXT),
        ),
    ))
    cal_layout = {**DARK}
    cal_layout["margin"] = dict(l=60, r=60, t=80, b=20)
    fig.update_layout(
        **cal_layout, height=600,
        title="Monthly Average Stress — Heatmap Calendar",
        xaxis=dict(side="top", tickfont=dict(size=11)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10), gridcolor="#333"),
    )
    return fig


# ── HTML assembly ────────────────────────────────────────────────────────────

def _fig_json(fig: go.Figure) -> str:
    return json.dumps(fig.to_json())


def _pred_card(h: int, p: dict) -> str:
    if not p:
        return ""
    score = p.get("stress_score", "—")
    cls   = p.get("stress_class", "—")
    delta = p.get("stress_delta")
    direc = p.get("market_direction", "N/A")
    proba = p.get("market_dir_proba", {})
    col   = STRESS_COLORS.get(cls, "#888")
    label = {5: "Short-Term (5d)", 21: "Medium-Term (21d)", 63: "Long-Term (63d)"}[h]

    if delta is not None:
        dcolor = "#E74C3C" if delta > 0 else "#2ECC71"
        delta_str = f"<span style='color:{dcolor}'>{delta:+.1f}</span>"
    else:
        delta_str = "—"

    dir_html = ""
    if proba and h < 63:
        best = max(proba, key=proba.get)
        dir_html = f"""
        <div class="dir-bar-wrap">
          <div class="dir-bar" style="width:{proba.get('Down',0)*100:.0f}%; background:#E74C3C" title="Down {proba.get('Down',0):.0%}"></div>
          <div class="dir-bar" style="width:{proba.get('Neutral',0)*100:.0f}%; background:#F39C12" title="Neutral {proba.get('Neutral',0):.0%}"></div>
          <div class="dir-bar" style="width:{proba.get('Up',0)*100:.0f}%; background:#2ECC71" title="Up {proba.get('Up',0):.0%}"></div>
        </div>
        <div class="dir-label">Direction: <b style="color:{'#E74C3C' if best=='Down' else '#2ECC71' if best=='Up' else '#F39C12'}">{direc}</b></div>"""

    return f"""
    <div class="card">
      <div class="card-label">{label}</div>
      <div class="stress-score" style="color:{col}">{score}</div>
      <div class="stress-class" style="color:{col}">{cls}</div>
      <div class="delta">vs current: {delta_str}</div>
      {dir_html}
    </div>"""


def _oos_metrics_table() -> str:
    m = OOS_METRICS_5D
    return f"""
    <table class="metrics-table">
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>Observations</td><td>{m['n']:,}</td></tr>
        <tr><td>MAE</td><td>{m['mae']}</td></tr>
        <tr><td>RMSE</td><td>{m['rmse']}</td></tr>
        <tr><td>Pearson R</td><td>{m['pearson_r']}</td></tr>
        <tr><td>Classification Accuracy</td><td>{m['cls_acc']:.1%}</td></tr>
        <tr><td>Direction Accuracy</td><td>{m['dir_acc']:.1%}</td></tr>
      </tbody>
    </table>"""


def build_html(predictions, hist, csi, date_str) -> str:
    current_csi   = predictions.get(5, {}).get("current_csi", 50) or 50
    latest_row    = csi.iloc[-1].to_dict()

    # Determine current class from CSI value
    current_class = "Low"
    for thr, cls in [(75, "Extreme"), (50, "High"), (25, "Elevated")]:
        if current_csi > thr:
            current_class = cls
            break

    # Build figures
    fig_gauge  = _gauge(current_csi, current_class)
    fig_radar  = _radar(latest_row)
    fig_3y     = _csi_3y(csi, predictions)
    fig_dir    = _direction_chart(predictions)
    fig_full   = _full_history(csi)
    fig_bt     = _backtest_chart(hist)
    fig_perf   = _perf_chart()
    fig_err    = _error_dist(hist)
    fig_cal    = _calendar_heatmap(csi)

    cards = "".join(_pred_card(h, predictions.get(h, {})) for h in [5, 21, 63])
    col = STRESS_COLORS.get(current_class, "#888")
    oos_table = _oos_metrics_table()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Market Stress Dashboard — {date_str}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:{BG};color:{TEXT};font-family:Inter,Arial,sans-serif;font-size:14px}}
  .container{{max-width:1280px;margin:0 auto;padding:20px 16px}}
  header{{text-align:center;margin-bottom:20px}}
  h1{{color:{ACCENT};font-size:24px;margin-bottom:4px}}
  .subtitle{{color:#888;font-size:12px}}
  .badge{{display:inline-block;padding:4px 14px;border-radius:20px;font-size:13px;
          font-weight:bold;letter-spacing:1px;margin-top:8px;
          color:#fff;background:{col}}}
  /* Tabs */
  .tabs{{display:flex;gap:4px;margin-bottom:20px;border-bottom:1px solid #333;padding-bottom:0}}
  .tab-btn{{background:none;border:none;color:#888;font-size:13px;padding:10px 20px;
            cursor:pointer;border-bottom:2px solid transparent;transition:.2s}}
  .tab-btn:hover{{color:{TEXT}}}
  .tab-btn.active{{color:{ACCENT};border-bottom-color:{ACCENT}}}
  .tab-content{{display:none}}.tab-content.active{{display:block}}
  /* Cards */
  .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
  .grid-3{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
  .card{{background:{CARD_BG};border:1px solid {BORDER};border-radius:10px;padding:18px;text-align:center}}
  .card-label{{font-size:10px;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}}
  .stress-score{{font-size:44px;font-weight:bold;line-height:1}}
  .stress-class{{font-size:12px;text-transform:uppercase;letter-spacing:2px;margin:4px 0}}
  .delta{{font-size:12px;color:#aaa;margin:4px 0}}
  .dir-bar-wrap{{display:flex;height:6px;border-radius:3px;overflow:hidden;margin:8px 0}}
  .dir-bar{{height:100%;transition:.3s}}
  .dir-label{{font-size:11px;color:#888}}
  .chart-box{{background:{CARD_BG};border:1px solid {BORDER};border-radius:8px;padding:8px;margin-bottom:16px}}
  h2{{color:{ACCENT};font-size:14px;margin:20px 0 10px;border-bottom:1px solid #333;padding-bottom:6px}}
  /* Metrics table */
  .metrics-table{{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:16px}}
  .metrics-table th{{background:#0f3460;color:{ACCENT};padding:8px 12px;text-align:left;font-weight:600}}
  .metrics-table td{{padding:7px 12px;border-bottom:1px solid #1e3060;color:{TEXT}}}
  .metrics-table tr:hover td{{background:rgba(79,195,247,0.05)}}
  .oos-note{{font-size:11px;color:#666;margin-bottom:12px;font-style:italic}}
  /* Legend */
  .legend-row{{display:flex;justify-content:center;gap:20px;margin-bottom:16px;flex-wrap:wrap}}
  .legend-item{{display:flex;align-items:center;gap:6px;font-size:11px}}
  .legend-dot{{width:12px;height:12px;border-radius:2px}}
  .footer{{text-align:center;color:#555;font-size:11px;margin-top:30px;padding-top:16px;border-top:1px solid #333}}
  @media(max-width:700px){{.grid-2,.grid-3{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="container">

<header>
  <h1>Global Market Stress Indicator</h1>
  <div class="subtitle">{date_str} &nbsp;|&nbsp; Cross-Asset Financial Stress Dashboard &nbsp;|&nbsp; 174 Features · Dynamic Regime Boundaries</div>
  <div class="badge">Current CSI: {current_csi:.1f} — {current_class}</div>
</header>

<div class="legend-row">
  <div class="legend-item"><div class="legend-dot" style="background:#2ECC71"></div>Low</div>
  <div class="legend-item"><div class="legend-dot" style="background:#F39C12"></div>Elevated</div>
  <div class="legend-item"><div class="legend-dot" style="background:#E67E22"></div>High</div>
  <div class="legend-item"><div class="legend-dot" style="background:#E74C3C"></div>Extreme</div>
</div>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('snapshot')">📊 Snapshot</button>
  <button class="tab-btn" onclick="showTab('history')">📈 Full History</button>
  <button class="tab-btn" onclick="showTab('backtest')">🔬 Backtest</button>
  <button class="tab-btn" onclick="showTab('calendar')">📅 Calendar</button>
</div>

<!-- TAB: SNAPSHOT -->
<div id="tab-snapshot" class="tab-content active">
  <h2>Current Stress Assessment</h2>
  <div class="grid-2">
    <div class="chart-box" id="gauge"></div>
    <div class="chart-box" id="radar"></div>
  </div>

  <h2>Multi-Horizon Forecast</h2>
  <div class="grid-3">{cards}</div>

  <h2>Market Direction Signals</h2>
  <div class="chart-box" id="direction"></div>

  <h2>Recent Trajectory (3 Years)</h2>
  <div class="chart-box" id="csi3y"></div>
</div>

<!-- TAB: HISTORY -->
<div id="tab-history" class="tab-content">
  <h2>Full CSI History (2000 – present)</h2>
  <p style="font-size:11px;color:#666;margin-bottom:10px">
    Red bands = historical stress episodes &nbsp;|&nbsp;
    Dotted lines = dimension sub-indices (toggle in legend) &nbsp;|&nbsp;
    Use range selector or slider to zoom
  </p>
  <div class="chart-box" id="fullhistory"></div>
</div>

<!-- TAB: BACKTEST -->
<div id="tab-backtest" class="tab-content">
  <h2>Walk-Forward OOS Performance — 5-Day Horizon</h2>
  <p class="oos-note">OOS walk-forward results: 98 folds · step=63d · expanding window · 174 features · dynamic regime class boundaries (30/50/70th percentile of training window)</p>
  {oos_table}

  <h2>OOS Classification & Direction Performance</h2>
  <div class="chart-box" id="perfchart"></div>

  <h2>In-Sample Predicted vs Actual (All Horizons)</h2>
  <p class="oos-note">In-sample only — shows model fit quality; red bands = stress episodes</p>
  <div class="chart-box" id="btchart"></div>

  <h2>Prediction Error Distribution (In-Sample)</h2>
  <div class="chart-box" id="errchart"></div>
</div>

<!-- TAB: CALENDAR -->
<div id="tab-calendar" class="tab-content">
  <h2>Monthly Stress Heatmap</h2>
  <p style="font-size:11px;color:#666;margin-bottom:10px">Average monthly CSI score by year — darker red = higher stress</p>
  <div class="chart-box" id="calendar"></div>
</div>

<div class="footer">
  Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
  Financial Market Stress Indicator &nbsp;|&nbsp; For research purposes only.
</div>
</div>

<script>
const LP = {{paper_bgcolor:'{BG}',plot_bgcolor:'{CARD_BG}',font:{{color:'{TEXT}'}}}};
const CFG = {{responsive:true,displayModeBar:true,
  modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d'],displaylogo:false}};

// Charts rendered eagerly (Snapshot tab is visible on page load)
function renderNow(id, json_str) {{
  if (!json_str) return;
  const spec = JSON.parse(json_str);
  if (spec && spec.layout) Object.assign(spec.layout, LP);
  Plotly.newPlot(id, spec.data, spec.layout, CFG);
}}

// Lazy chart store: render only when the tab is first shown
const LAZY = {{
  'fullhistory': {_fig_json(fig_full)},
  'btchart':     {_fig_json(fig_bt)},
  'perfchart':   {_fig_json(fig_perf)},
  'errchart':    {_fig_json(fig_err)},
  'calendar':    {_fig_json(fig_cal)},
}};
const _rendered = {{}};

function renderLazy(id) {{
  if (_rendered[id]) {{ Plotly.Plots.resize(document.getElementById(id)); return; }}
  const json_str = LAZY[id];
  if (!json_str) return;
  const spec = JSON.parse(json_str);
  if (spec && spec.layout) Object.assign(spec.layout, LP);
  Plotly.newPlot(id, spec.data, spec.layout, CFG);
  _rendered[id] = true;
}}

const TAB_LAZY = {{
  'history':  ['fullhistory'],
  'backtest': ['perfchart','btchart','errchart'],
  'calendar': ['calendar'],
}};

function showTab(name) {{
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
  // Render lazy charts after the tab's div is visible (50ms lets layout settle)
  setTimeout(function() {{
    (TAB_LAZY[name] || []).forEach(renderLazy);
  }}, 50);
}}

// Render snapshot charts immediately (tab is visible on page load)
renderNow('gauge',     {_fig_json(fig_gauge)});
renderNow('radar',     {_fig_json(fig_radar)});
renderNow('direction', {_fig_json(fig_dir)});
renderNow('csi3y',     {_fig_json(fig_3y)});
</script>
</body>
</html>"""


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",       type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="reports")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("dashboard")

    log.info("Loading data ...")
    features = load_features()
    csi      = load_csi()

    if "regime_csi_mean_504d" not in features.columns:
        log.info("Adding regime features ...")
        features = add_regime_features(features, csi, save=False)

    log.info("Running predictions ...")
    predictions = predict_latest(features, csi, horizons=[5, 21, 63], as_of=args.date)

    log.info("Generating in-sample historical predictions ...")
    hist = {}
    for h in [5, 21, 63]:
        try:
            hist[h] = predict_historical(features, csi, h)
            log.info("  h=%dd: %d rows", h, len(hist[h]))
        except FileNotFoundError as e:
            log.warning(str(e))

    date_str = args.date or features.index[-1].strftime("%Y-%m-%d")
    log.info("Building HTML dashboard ...")
    html = build_html(predictions, hist, csi, date_str)

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"stress_dashboard_interactive_{date_str}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("Dashboard saved: %s", path)
    print(f"\nDashboard ready: {path}")


if __name__ == "__main__":
    main()
