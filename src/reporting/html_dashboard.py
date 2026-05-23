"""
Standalone HTML dashboard generator using Plotly.

Produces a single self-contained HTML file with no server dependency.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

STRESS_COLORS = {
    "Low":      "#2ECC71",
    "Elevated": "#F39C12",
    "High":     "#E67E22",
    "Extreme":  "#E74C3C",
}

DARK_LAYOUT = dict(
    paper_bgcolor="#1a1a2e",
    plot_bgcolor="#16213e",
    font=dict(color="#e0e0e0", family="Inter, Arial, sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40),
)

DIM_COLORS = {
    "Equity":        "#4fc3f7",
    "Credit":        "#FF6B6B",
    "Rates":         "#FFD700",
    "Liquidity":     "#98FB98",
    "FX":            "#DA70D6",
    "International": "#FFA07A",
}


def _csi_history_chart(csi: pd.DataFrame, predictions: Dict[int, dict]) -> go.Figure:
    tail = csi.tail(756)  # 3 years
    fig = go.Figure()

    # Stress zone bands
    zone_kwargs = dict(line_width=0, showlegend=False)
    fig.add_trace(go.Scatter(x=tail.index, y=[25]*len(tail), fill="tozeroy",
                             fillcolor="rgba(46,204,113,0.08)", **zone_kwargs, name="Low"))
    fig.add_trace(go.Scatter(x=tail.index, y=[50]*len(tail), fill="tonexty",
                             fillcolor="rgba(243,156,18,0.10)", **zone_kwargs, name="Elevated"))
    fig.add_trace(go.Scatter(x=tail.index, y=[75]*len(tail), fill="tonexty",
                             fillcolor="rgba(230,126,34,0.10)", **zone_kwargs, name="High"))
    fig.add_trace(go.Scatter(x=tail.index, y=[100]*len(tail), fill="tonexty",
                             fillcolor="rgba(231,76,60,0.10)", **zone_kwargs, name="Extreme"))

    # CSI line
    fig.add_trace(go.Scatter(
        x=tail.index, y=tail["csi_composite"],
        name="CSI Composite", line=dict(color="#4fc3f7", width=2),
        hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}<extra></extra>"
    ))

    # Dimension lines (lighter)
    dim_cols = {
        "csi_equity": "Equity", "csi_credit": "Credit",
        "csi_rates": "Rates",   "csi_liquidity": "Liquidity",
    }
    for col, dim in dim_cols.items():
        if col in tail.columns:
            fig.add_trace(go.Scatter(
                x=tail.index, y=tail[col],
                name=dim, line=dict(color=DIM_COLORS[dim], width=1, dash="dot"),
                opacity=0.5, visible="legendonly",
                hovertemplate=f"{dim} %{{x|%Y-%m-%d}}: %{{y:.1f}}<extra></extra>"
            ))

    # Forward predictions
    pred_colors = {5: "#FFD700", 21: "#FF6B6B", 63: "#98FB98"}
    last_date = tail.index[-1]
    for h, col in pred_colors.items():
        p = predictions.get(h)
        if p:
            fig.add_trace(go.Scatter(
                x=[last_date], y=[p["stress_score"]],
                mode="markers+text",
                marker=dict(color=col, size=10, symbol="diamond"),
                text=[f"{h}d: {p['stress_score']:.0f}"],
                textposition="top center",
                name=f"{h}d forecast",
                hovertemplate=f"{h}d forecast: {p['stress_score']:.1f} ({p['stress_class']})<extra></extra>"
            ))

    fig.update_layout(
        **DARK_LAYOUT,
        title="Composite Stress Index — 3-Year History",
        yaxis=dict(range=[0, 100], tickvals=[0, 25, 50, 75, 100]),
        xaxis=dict(showgrid=False),
        height=380,
        legend=dict(bgcolor="rgba(22,33,62,0.8)", bordercolor="#444", borderwidth=1)
    )
    return fig


def _gauge_chart(score: float, stress_class: str) -> go.Figure:
    color = STRESS_COLORS.get(stress_class, "#888")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": 50, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#e0e0e0"},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 25],   "color": "rgba(46,204,113,0.25)"},
                {"range": [25, 50],  "color": "rgba(243,156,18,0.25)"},
                {"range": [50, 75],  "color": "rgba(230,126,34,0.25)"},
                {"range": [75, 100], "color": "rgba(231,76,60,0.25)"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "value": score},
        },
        number={"font": {"color": color, "size": 36}},
        title={"text": f"Current Stress Level<br><b style='color:{color}'>{stress_class}</b>",
               "font": {"color": "#e0e0e0", "size": 13}},
    ))
    fig.update_layout(**DARK_LAYOUT, height=280)
    return fig


def _shap_chart(attribution: List[dict], horizon: int) -> go.Figure:
    if not attribution:
        return go.Figure()
    labels = [a["label"][:40] for a in attribution[:8]]
    values = [a["shap_value"] for a in attribution[:8]]
    dims   = [a["dimension"] for a in attribution[:8]]
    bar_colors = [DIM_COLORS.get(d, "#aaa") for d in dims]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=bar_colors,
        hovertemplate="%{y}: SHAP=%{x:.4f}<extra></extra>"
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=f"SHAP Drivers — {horizon}-Day Horizon",
        yaxis={"autorange": "reversed", "tickfont": {"size": 9}},
        height=300,
        showlegend=False,
    )
    return fig


def _radar_chart(dim_scores: dict) -> go.Figure:
    dims = ["Equity", "Credit", "Rates", "Liquidity", "FX", "International"]
    keys = ["csi_equity", "csi_credit", "csi_rates", "csi_liquidity", "csi_fx", "csi_international"]
    vals = [float(dim_scores.get(k, 0)) for k in keys]

    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=dims + [dims[0]],
        fill="toself",
        fillcolor="rgba(79,195,247,0.2)",
        line=dict(color="#4fc3f7", width=2),
        name="Dimension Scores",
        hovertemplate="%{theta}: %{r:.1f}<extra></extra>"
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        polar=dict(
            bgcolor="#16213e",
            radialaxis=dict(visible=True, range=[0, 100],
                            tickvals=[25, 50, 75], tickfont={"size": 8}),
            angularaxis=dict(tickfont={"size": 9}),
        ),
        title="Stress by Dimension",
        height=300,
        showlegend=False,
    )
    return fig


def _direction_probability_chart(predictions: Dict[int, dict]) -> go.Figure:
    horizons = [h for h in [5, 21] if predictions.get(h, {}).get("market_dir_proba")]
    if not horizons:
        return go.Figure()

    fig = go.Figure()
    for h in horizons:
        proba = predictions[h]["market_dir_proba"]
        fig.add_trace(go.Bar(
            x=["Down", "Neutral", "Up"],
            y=[proba["Down"], proba["Neutral"], proba["Up"]],
            name=f"{h}d",
            marker_color=["#E74C3C", "#F39C12", "#2ECC71"],
            opacity=0.8 if h == 5 else 0.5,
        ))
    fig.update_layout(
        **DARK_LAYOUT,
        title="Market Direction Probabilities",
        yaxis=dict(title="Probability", tickformat=".0%", range=[0, 1]),
        barmode="group",
        height=280,
    )
    return fig


def _build_html(
    fig_gauge: go.Figure,
    fig_radar: go.Figure,
    fig_history: go.Figure,
    fig_shap_5: go.Figure,
    fig_shap_21: go.Figure,
    fig_dir: go.Figure,
    predictions: Dict[int, dict],
    narrative: str,
    date_str: str,
) -> str:
    def to_json(fig):
        return fig.to_json() if fig else "{}"

    def pred_card(h):
        p = predictions.get(h, {})
        score = p.get("stress_score", "—")
        cls   = p.get("stress_class", "—")
        delta = p.get("stress_delta")
        direc = p.get("market_direction", "N/A")
        col   = STRESS_COLORS.get(cls, "#888")
        label = {5: "Short-Term (5d)", 21: "Medium-Term (21d)", 63: "Long-Term (63d)"}[h]
        delta_str = f"<span style='color:{'#E74C3C' if delta and delta>0 else '#2ECC71'}'>{delta:+.1f}</span>" if delta is not None else "—"
        return f"""
        <div class="card">
          <div class="card-label">{label}</div>
          <div class="stress-score" style="color:{col}">{score}</div>
          <div class="stress-class" style="color:{col}">{cls}</div>
          <div class="delta">Δ {delta_str}</div>
          <div class="direction">Direction: {direc if h < 63 else 'N/A'}</div>
        </div>"""

    narrative_html = "".join(
        f"<p>{p.replace('**', '<b>').replace('**', '</b>')}</p>"
        for p in narrative.split("\n\n") if p.strip()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Global Market Stress Report — {date_str}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #1a1a2e; color: #e0e0e0; font-family: Inter, Arial, sans-serif; font-size: 14px; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
  h1 {{ color: #4fc3f7; text-align: center; font-size: 22px; margin-bottom: 4px; }}
  .subtitle {{ text-align: center; color: #888; font-size: 12px; margin-bottom: 20px; }}
  h2 {{ color: #4fc3f7; font-size: 15px; margin: 20px 0 8px; border-bottom: 1px solid #333; padding-bottom: 4px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
  .grid-4 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }}
  .card {{ background: #16213e; border: 1px solid #0f3460; border-radius: 8px; padding: 16px; text-align: center; }}
  .card-label {{ font-size: 11px; color: #888; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 1px; }}
  .stress-score {{ font-size: 36px; font-weight: bold; }}
  .stress-class {{ font-size: 13px; text-transform: uppercase; letter-spacing: 2px; margin: 2px 0; }}
  .delta {{ font-size: 13px; color: #aaa; margin: 4px 0; }}
  .direction {{ font-size: 12px; color: #888; }}
  .chart-box {{ background: #16213e; border: 1px solid #0f3460; border-radius: 8px; padding: 8px; }}
  .narrative {{ background: #16213e; border: 1px solid #0f3460; border-radius: 8px; padding: 20px; line-height: 1.7; }}
  .narrative h3 {{ color: #4fc3f7; font-size: 13px; margin: 10px 0 4px; }}
  .narrative p {{ margin-bottom: 10px; color: #cccccc; font-size: 13px; }}
  .footer {{ text-align: center; color: #555; font-size: 11px; margin-top: 30px; padding-top: 16px; border-top: 1px solid #333; }}
  .legend-row {{ display: flex; justify-content: center; gap: 20px; margin-bottom: 16px; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 11px; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 2px; }}
  @media (max-width: 700px) {{ .grid-2, .grid-3 {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="container">
  <h1>Global Market Stress Report</h1>
  <div class="subtitle">{date_str} &nbsp;|&nbsp; Cross-Asset Financial Stress Indicator</div>

  <div class="legend-row">
    <div class="legend-item"><div class="legend-dot" style="background:#2ECC71"></div>Low (0–25)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#F39C12"></div>Elevated (26–50)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#E67E22"></div>High (51–75)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#E74C3C"></div>Extreme (76–100)</div>
  </div>

  <h2>Current Stress Assessment</h2>
  <div class="grid-2">
    <div class="chart-box" id="gauge"></div>
    <div class="chart-box" id="radar"></div>
  </div>

  <h2>Multi-Horizon Forecast</h2>
  <div class="grid-3">
    {pred_card(5)}
    {pred_card(21)}
    {pred_card(63)}
  </div>

  <h2>Historical Trajectory</h2>
  <div class="chart-box" id="history"></div>

  <h2>Market Direction Signals</h2>
  <div class="chart-box" id="direction"></div>

  <h2>Key Risk Drivers (SHAP Attribution)</h2>
  <div class="grid-2">
    <div class="chart-box" id="shap5"></div>
    <div class="chart-box" id="shap21"></div>
  </div>

  <h2>Market Intelligence Briefing</h2>
  <div class="narrative">{narrative_html}</div>

  <div class="footer">
    Generated {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;|&nbsp;
    Global Market Stress Indicator v1.0 &nbsp;|&nbsp; For research purposes only.
  </div>
</div>

<script>
  const layout_patch = {{ paper_bgcolor: '#1a1a2e', plot_bgcolor: '#16213e', font: {{ color: '#e0e0e0' }} }};
  function render(id, json) {{
    if (!json || json === '{{}}') return;
    const spec = JSON.parse(json);
    Object.assign(spec.layout, layout_patch);
    Plotly.newPlot(id, spec.data, spec.layout, {{responsive: true, displayModeBar: false}});
  }}
  render('gauge',    {json.dumps(to_json(fig_gauge))});
  render('radar',    {json.dumps(to_json(fig_radar))});
  render('history',  {json.dumps(to_json(fig_history))});
  render('direction',{json.dumps(to_json(fig_dir))});
  render('shap5',    {json.dumps(to_json(fig_shap_5))});
  render('shap21',   {json.dumps(to_json(fig_shap_21))});
</script>
</body>
</html>"""


def generate_html(
    predictions: Dict[int, dict],
    attribution_by_horizon: Dict[int, List[dict]],
    csi: pd.DataFrame,
    narrative: str,
    output_dir: str = "reports",
    as_of: Optional[str] = None,
) -> str:
    """Generate the standalone HTML dashboard. Returns the output file path."""
    os.makedirs(output_dir, exist_ok=True)
    date_str = as_of or datetime.today().strftime("%Y-%m-%d")

    pred_5d = predictions.get(5, {})
    current_csi = pred_5d.get("current_csi", 50) or 50
    current_class = "Low"
    for thr, cls in [(75, "Extreme"), (50, "High"), (25, "Elevated")]:
        if current_csi > thr:
            current_class = cls
            break

    latest_row = csi.iloc[-1].to_dict() if len(csi) > 0 else {}

    fig_gauge   = _gauge_chart(current_csi, current_class)
    fig_radar   = _radar_chart(latest_row)
    fig_history = _csi_history_chart(csi, predictions)
    fig_shap_5  = _shap_chart(attribution_by_horizon.get(5, []), 5)
    fig_shap_21 = _shap_chart(attribution_by_horizon.get(21, []), 21)
    fig_dir     = _direction_probability_chart(predictions)

    html = _build_html(
        fig_gauge, fig_radar, fig_history,
        fig_shap_5, fig_shap_21, fig_dir,
        predictions, narrative, date_str,
    )

    output_path = os.path.join(output_dir, f"stress_dashboard_{date_str}.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("HTML dashboard saved: %s", output_path)
    return output_path
