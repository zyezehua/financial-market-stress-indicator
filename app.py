"""
Streamlit app for Financial Market Stress Indicator.
Deployable on Streamlit Cloud with model artifacts hosted on Hugging Face Hub.

Local usage:
    streamlit run app.py

Streamlit Cloud:
    Set secrets: FRED_API_KEY, HF_TOKEN (optional, for private repos)
    Artifacts are downloaded from HF Hub on cold start.
"""

import os
import sys
import logging
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Financial Market Stress Indicator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

HF_REPO = "zyezehua/fmsi-artifacts"

# ── Style constants ──────────────────────────────────────────────────────────
BG          = "#1a1a2e"
CARD_BG     = "#16213e"
BORDER      = "#0f3460"
TEXT        = "#e0e0e0"
ACCENT      = "#4fc3f7"

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

# ── Artifact loading ─────────────────────────────────────────────────────────

def _get_artifact_dir() -> str:
    """Return local artifact dir. Downloads pkl files from HF Hub if not present."""
    local = Path("models")
    if (local / "model_h5d.pkl").exists():
        return str(local)

    try:
        import shutil
        from huggingface_hub import hf_hub_download
        hf_token = st.secrets.get("HF_TOKEN", None) or os.getenv("HF_TOKEN")
        local.mkdir(parents=True, exist_ok=True)
        with st.spinner("Downloading model artifacts from Hugging Face Hub (~65 MB)..."):
            for fname in ["model_h5d.pkl", "model_h21d.pkl", "model_h63d.pkl"]:
                cached = hf_hub_download(
                    repo_id=HF_REPO, repo_type="dataset",
                    filename=fname, token=hf_token,
                )
                shutil.copy2(cached, local / fname)
        logger.info("Model artifacts downloaded to: %s", local)
        return str(local)
    except Exception as exc:
        st.error(f"Could not load model artifacts: {exc}")
        st.stop()


def _get_data_dir() -> str:
    """Download data files from HF Hub to their expected local paths."""
    import shutil
    from huggingface_hub import hf_hub_download

    # HF path → expected local path (matching load_features / load_csi conventions)
    mapping = [
        ("data/features.parquet", Path("data/processed/features.parquet")),
        ("data/csi.parquet",      Path("data/labels/csi.parquet")),
        ("data/targets.parquet",  Path("data/labels/targets.parquet")),
    ]

    if mapping[0][1].exists():
        return str(mapping[0][1].parent)

    try:
        hf_token = st.secrets.get("HF_TOKEN", None) or os.getenv("HF_TOKEN")
        with st.spinner("Downloading data from Hugging Face Hub (~7 MB)..."):
            for hf_fname, local_path in mapping:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                cached = hf_hub_download(
                    repo_id=HF_REPO, repo_type="dataset",
                    filename=hf_fname, token=hf_token,
                )
                shutil.copy2(cached, local_path)
        logger.info("Data downloaded to data/processed/ and data/labels/")
        return str(mapping[0][1].parent)
    except Exception as exc:
        st.error(f"Could not load data artifacts: {exc}")
        st.stop()


# ── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading features and CSI...")
def load_data():
    _get_data_dir()
    from src.features.builder import load_features, add_regime_features
    from src.labels.composite_index import load_csi
    features = load_features()
    csi      = load_csi()
    if "regime_csi_mean_504d" not in features.columns:
        features = add_regime_features(features, csi, save=False)
    return features, csi


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    _get_artifact_dir()


@st.cache_data(ttl=3600, show_spinner="Running predictions...")
def get_predictions(as_of: str | None = None):
    features, csi = load_data()
    load_artifacts()
    from src.models.predictor import predict_latest
    return predict_latest(features, csi, horizons=[5, 21, 63], as_of=as_of)


@st.cache_data(ttl=3600, show_spinner="Computing historical predictions...")
def get_hist_predictions():
    features, csi = load_data()
    load_artifacts()
    from src.models.predictor import predict_historical
    hist = {}
    for h in [5, 21, 63]:
        try:
            df = predict_historical(features, csi, h)
            hist[h] = df
        except Exception as exc:
            logger.warning("predict_historical h=%dd failed: %s", h, exc)
    return hist


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


def _csi_line_base(data: pd.DataFrame, show_episodes: bool = True,
                   range_slider: bool = True) -> go.Figure:
    from src.labels.calibrator import STRESS_EPISODES
    fig = go.Figure()
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
                    fillcolor="rgba(231,76,60,0.12)", line_width=0, layer="below",
                )
            except Exception:
                pass

    fig.add_trace(go.Scatter(
        x=data.index, y=data["csi_composite"],
        name="CSI Composite", line=dict(color=ACCENT, width=2),
        hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}<extra></extra>",
    ))

    for col, label in DIM_LABELS.items():
        if col in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data[col], name=label,
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


def _csi_3y(csi: pd.DataFrame, predictions: dict) -> go.Figure:
    tail = csi.tail(756)
    fig  = _csi_line_base(tail, show_episodes=False, range_slider=False)
    last_date = tail.index[-1]
    colors = {5: "#FFD700", 21: "#FF6B6B", 63: "#98FB98"}
    for h, col in colors.items():
        p = predictions.get(h)
        if p:
            fig.add_trace(go.Scatter(
                x=[last_date], y=[p["stress_score"]],
                mode="markers+text",
                marker=dict(color=col, size=12, symbol="diamond",
                            line=dict(color="white", width=1)),
                text=[f"{h}d: {p['stress_score']:.0f}"],
                textposition="top center",
                name=f"{h}d forecast",
                hovertemplate=f"{h}d Forecast: {p['stress_score']:.1f} ({p['stress_class']})<extra></extra>",
            ))
    fig.update_layout(title="Composite Stress Index — 3-Year History", height=380)
    return fig


def _full_history(csi: pd.DataFrame) -> go.Figure:
    fig = _csi_line_base(csi, show_episodes=True, range_slider=True)
    if "regime_csi_mean_252d" in csi.columns:
        fig.add_trace(go.Scatter(
            x=csi.index, y=csi["regime_csi_mean_252d"],
            name="Regime Mean (252d)", line=dict(color="#FF69B4", width=1.5, dash="dash"),
            visible="legendonly",
            hovertemplate="Regime Mean 252d %{x|%Y-%m-%d}: %{y:.1f}<extra></extra>",
        ))
    fig.update_layout(title="Composite Stress Index — Full History (2000–present)", height=550)
    return fig


def _backtest_chart(hist: dict) -> go.Figure:
    from src.labels.calibrator import STRESS_EPISODES
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=["5-Day Horizon", "21-Day Horizon", "63-Day Horizon"],
        row_heights=[1, 1, 1], vertical_spacing=0.06,
    )
    colors_pred = {5: "#FFD700", 21: "#FF6B6B", 63: "#98FB98"}
    for i, h in enumerate([5, 21, 63], 1):
        df = hist.get(h)
        if df is None or df.empty:
            continue
        df = df.dropna(subset=["stress_actual"])
        fig.add_trace(go.Scatter(
            x=df.index, y=df["stress_actual"], name="Actual CSI",
            line=dict(color=ACCENT, width=1.2), opacity=0.9,
            showlegend=(i == 1),
            hovertemplate="%{x|%Y-%m-%d} Actual: %{y:.1f}<extra></extra>",
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["stress_score_pred"],
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
        fig.update_yaxes(range=[0, 100], row=i, col=1, gridcolor="#333")

    fig.update_layout(
        **DARK, height=700,
        title="In-Sample Backtest: Predicted vs Actual CSI",
        legend=dict(bgcolor="rgba(22,33,62,0.8)", bordercolor="#444", borderwidth=1),
    )
    fig.update_xaxes(showgrid=False, gridcolor="#333")
    return fig


def _perf_chart() -> go.Figure:
    m = OOS_METRICS_5D
    classes = list(m["stress_cls"].keys())
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Stress Classification (OOS h=5d)", "Market Direction (OOS h=5d)"],
        horizontal_spacing=0.12,
    )
    metrics = ["precision", "recall", "f1"]
    metric_colors = ["#4fc3f7", "#FFD700", "#98FB98"]

    for j, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric.capitalize(), x=classes,
            y=[m["stress_cls"][c][metric] for c in classes],
            marker_color=metric_colors[j], opacity=0.85,
            hovertemplate=f"{metric.capitalize()}: %{{y:.2f}}<extra></extra>",
        ), row=1, col=1)

    dir_classes = list(m["dir_cls"].keys())
    for j, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric.capitalize(), x=dir_classes,
            y=[m["dir_cls"][c][metric] for c in dir_classes],
            marker_color=metric_colors[j], opacity=0.85,
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
        if df is None or df.empty:
            continue
        err = (df["stress_score_pred"] - df["stress_actual"]).dropna()
        fig.add_trace(go.Histogram(
            x=err, name=f"{h}d", nbinsx=60,
            marker_color=colors[h], opacity=0.6,
            hovertemplate=f"{h}d error: %{{x:.1f}}<extra></extra>",
        ))
    fig.update_layout(
        **DARK, height=300, barmode="overlay",
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

    z, text = [], []
    for yr in years:
        row_z, row_t = [], []
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
            [0.00, "#1a3a1a"], [0.25, "#2ECC71"],
            [0.50, "#F39C12"], [0.75, "#E67E22"], [1.00, "#E74C3C"],
        ],
        zmin=0, zmax=100,
        colorbar=dict(
            title=dict(text="CSI", font=dict(color=TEXT)),
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0", "25", "50", "75", "100"],
            tickfont=dict(color=TEXT),
        ),
    ))
    cal_layout = {**DARK}
    cal_layout["margin"] = dict(l=60, r=60, t=80, b=20)
    fig.update_layout(
        **cal_layout, height=620,
        title="Monthly Average Stress — Heatmap Calendar",
        xaxis=dict(side="top", tickfont=dict(size=11)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10), gridcolor="#333"),
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown(f"### 📊 FMSI Dashboard")
        st.markdown("*Financial Market Stress Indicator*")
        st.divider()

        as_of = st.date_input(
            "Prediction date",
            value=None,
            help="Leave blank for latest. Set a past date to simulate a historical prediction.",
        )
        as_of_str = as_of.strftime("%Y-%m-%d") if as_of else None

        st.divider()
        st.markdown("**Model info**")
        st.markdown("- 174 features · 6 dimensions")
        st.markdown("- 3-model ensemble per horizon")
        st.markdown("- Horizons: 5d · 21d · 63d")
        st.markdown("- Data: 2000–present")

        st.divider()
        m = OOS_METRICS_5D
        st.markdown("**OOS Backtest (h=5d)**")
        st.markdown(f"MAE: `{m['mae']}` · R: `{m['pearson_r']}`")
        st.markdown(f"Cls Acc: `{m['cls_acc']:.1%}` · Dir Acc: `{m['dir_acc']:.1%}`")

        st.divider()
        if st.button("🔄 Refresh predictions", use_container_width=True):
            get_predictions.clear()
            get_hist_predictions.clear()
            st.rerun()

    return as_of_str


# ── Tab renderers ─────────────────────────────────────────────────────────────

def render_snapshot(predictions: dict, csi: pd.DataFrame):
    if not predictions:
        st.warning("No predictions available. Check that model artifacts are loaded.")
        return

    p5 = predictions.get(5, {})
    current_csi   = p5.get("current_csi", 0) or 0
    stress_class  = p5.get("current_stress_class") or "Low"
    pred_date     = p5.get("prediction_date", "—")

    col_badge, _ = st.columns([3, 1])
    with col_badge:
        color = STRESS_COLORS.get(stress_class, "#888")
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<span style='background:{color};color:#fff;padding:4px 16px;"
            f"border-radius:20px;font-weight:bold;letter-spacing:1px;"
            f"font-size:13px'>{stress_class.upper()}</span>"
            f"<span style='color:#888;font-size:12px;margin-left:12px'>as of {pred_date}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Row 1: gauge + radar
    col_g, col_r = st.columns(2)
    with col_g:
        st.plotly_chart(_gauge(current_csi, stress_class),
                        use_container_width=True, key="gauge")
    with col_r:
        latest_row = csi.iloc[-1].to_dict()
        st.plotly_chart(_radar(latest_row), use_container_width=True, key="radar")

    # Row 2: forecast cards
    st.markdown("#### Forecast Horizons")
    card_cols = st.columns(3)
    for col, h in zip(card_cols, [5, 21, 63]):
        p = predictions.get(h, {})
        label = {5: "Short-Term (5d)", 21: "Medium-Term (21d)", 63: "Long-Term (63d)"}[h]
        with col:
            if p:
                score = p.get("stress_score", "—")
                cls   = p.get("stress_class", "—")
                delta = p.get("stress_delta")
                direc = p.get("market_direction", "N/A")
                col_c = STRESS_COLORS.get(cls, "#888")
                delta_color = "#E74C3C" if (delta or 0) > 0 else "#2ECC71"
                delta_str = (
                    f"<span style='color:{delta_color}'>{delta:+.1f}</span>"
                    if delta is not None else "—"
                )
                dir_color = "#E74C3C" if direc == "Down" else "#2ECC71" if direc == "Up" else "#F39C12"
                st.markdown(
                    f"<div style='background:{CARD_BG};border:1px solid {BORDER};"
                    f"border-radius:10px;padding:18px;text-align:center'>"
                    f"<div style='font-size:10px;color:#888;text-transform:uppercase;"
                    f"letter-spacing:1px;margin-bottom:8px'>{label}</div>"
                    f"<div style='font-size:40px;font-weight:bold;color:{col_c}'>{score}</div>"
                    f"<div style='font-size:12px;text-transform:uppercase;letter-spacing:2px;"
                    f"color:{col_c};margin:4px 0'>{cls}</div>"
                    f"<div style='font-size:12px;color:#aaa'>vs current: {delta_str}</div>"
                    f"<div style='font-size:11px;color:#888;margin-top:4px'>Direction: "
                    f"<b style='color:{dir_color}'>{direc}</b></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # Row 3: direction + 3-year chart
    st.markdown("#### Direction Probabilities")
    st.plotly_chart(_direction_chart(predictions), use_container_width=True, key="direction")

    st.markdown("#### 3-Year CSI Trajectory")
    st.plotly_chart(_csi_3y(csi, predictions), use_container_width=True, key="csi_3y")


def render_history(csi: pd.DataFrame):
    st.plotly_chart(_full_history(csi), use_container_width=True, key="full_history")


def render_backtest(hist: dict):
    m = OOS_METRICS_5D
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### OOS Backtest Results (h=5d)")
        st.caption("98 walk-forward folds · 6,120 out-of-sample observations")
        metrics_data = {
            "Metric": ["N obs", "MAE", "RMSE", "Pearson R", "Cls Accuracy", "Dir Accuracy"],
            "Value":  [f"{m['n']:,}", m['mae'], m['rmse'], m['pearson_r'],
                       f"{m['cls_acc']:.1%}", f"{m['dir_acc']:.1%}"],
        }
        st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)

        st.markdown("**Stress Classification**")
        cls_rows = [
            {"Class": c, **{k: f"{v:.2f}" if k != "support" else v
                             for k, v in vals.items()}}
            for c, vals in m["stress_cls"].items()
        ]
        st.dataframe(pd.DataFrame(cls_rows), hide_index=True, use_container_width=True)

    with col2:
        st.plotly_chart(_perf_chart(), use_container_width=True, key="perf_chart")

    if hist:
        st.plotly_chart(_backtest_chart(hist), use_container_width=True, key="bt_chart")
        st.plotly_chart(_error_dist(hist), use_container_width=True, key="err_dist")
    else:
        st.info("Historical predictions not available. Run the daily pipeline first.")


def render_calendar(csi: pd.DataFrame):
    st.plotly_chart(_calendar_heatmap(csi), use_container_width=True, key="calendar")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    as_of_str = render_sidebar()

    st.markdown(
        f"<h1 style='color:{ACCENT};text-align:center;margin-bottom:4px'>"
        f"📊 Financial Market Stress Indicator</h1>"
        f"<p style='text-align:center;color:#888;font-size:12px'>"
        f"Cross-asset ML stress forecasting · 26 years of daily data · "
        f"174 features · 3-horizon ensemble</p>",
        unsafe_allow_html=True,
    )

    tab_snapshot, tab_history, tab_backtest, tab_calendar = st.tabs(
        ["📡 Snapshot", "📈 Full History", "📋 Backtest", "📅 Calendar"]
    )

    with tab_snapshot:
        with st.spinner("Loading predictions..."):
            predictions = get_predictions(as_of_str)
            features, csi = load_data()
        render_snapshot(predictions, csi)

    with tab_history:
        features, csi = load_data()
        render_history(csi)

    with tab_backtest:
        hist = get_hist_predictions()
        render_backtest(hist)

    with tab_calendar:
        features, csi = load_data()
        render_calendar(csi)


if __name__ == "__main__":
    main()
