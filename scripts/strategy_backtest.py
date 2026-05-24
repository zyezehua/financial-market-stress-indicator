"""
scripts/strategy_backtest.py

S&P 500 trading strategy backtester using FMSI (Financial Market Stress Indicator) signals.
Strategy logic lives in src/strategy/engine.py; this script handles CLI, viz, and reporting.

Strategies:
  S1 — CSI Stress-Level Allocation
  S2 — CSI Trend Filter
  S3 — S1 + Model Direction Overlay (requires trained artifacts)
  S4 — Percentile Adaptive Threshold
  S5 — Trend Filter with Partial Re-entry

Usage:
    python scripts/strategy_backtest.py
    python scripts/strategy_backtest.py --start 2003-01-01 --optimize
    python scripts/strategy_backtest.py --no-model
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from src.strategy.engine import (
    load_spy_returns,
    run_backtest,
    build_weights,
    compute_metrics,
    rank_strategies,
    walkforward_optimize,
    run_all_strategies,
    subperiod_returns,
    STRATEGY_META,
    DEFAULT_PARAMS,
    METRIC_LABELS,
    HIGHER_IS_BETTER,
)

logger = logging.getLogger("strategy_backtest")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

TRADING_DAYS_PER_YEAR = 252
OUTPUT_DIR = Path("reports/strategy")

PALETTE = {
    "B&H SPY":                          "#4fc3f7",
    "S1: Stress Allocation":            "#FFD700",
    "S1: Stress Allocation (default)":  "#FFD7004D",
    "S1: Stress Allocation (opt)":      "#FFD700",
    "S2: Trend Filter":                 "#2ECC71",
    "S2: Trend Filter (default)":       "#2ECC714D",
    "S2: Trend Filter (opt)":           "#2ECC71",
    "S3: Stress + Direction":           "#E67E22",
    "S3: Stress + Direction (opt)":     "#E67E22",
    "S4: Percentile Adaptive":          "#BB86FC",
    "S4: Percentile Adaptive (opt)":    "#BB86FC",
    "S5: Trend + Re-entry":             "#FF6B9D",
    "S5: Trend + Re-entry (opt)":       "#FF6B9D",
}


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_csi_signal(csi_path: str = "data/labels/csi.parquet") -> pd.DataFrame:
    csi = pd.read_parquet(csi_path)
    sig = csi[["csi_composite"]].copy()
    if "csi_class" in csi.columns:
        sig["csi_class"] = csi["csi_class"]
    else:
        sig["csi_class"] = pd.cut(
            sig["csi_composite"],
            bins=[-1, 30, 50, 70, 101],
            labels=["Low", "Elevated", "High", "Extreme"],
        )
    return sig


def load_model_direction_signal(
    features_path: str = "data/processed/features.parquet",
    csi_path: str = "data/labels/csi.parquet",
    artifacts_dir: str = "models",
    horizon: int = 5,
    start: str = None,
) -> pd.DataFrame:
    from src.features.builder import load_features, add_regime_features
    from src.labels.composite_index import load_csi
    from src.models.trainer import load_artifact

    features = load_features(features_path)
    csi      = load_csi(csi_path)
    if "regime_csi_mean_504d" not in features.columns:
        features = add_regime_features(features, csi, save=False)
    if start:
        features = features.loc[start:]

    artifact = load_artifact(horizon, artifacts_dir)
    feat_names = artifact["feature_names"]
    X = features.reindex(columns=feat_names, fill_value=0)

    dir_ens = artifact.get("down_risk_ensemble") or artifact.get("direction_ensemble")
    if dir_ens is None:
        return pd.DataFrame()

    raw_preds  = dir_ens.predict(X)
    raw_probas = dir_ens.predict_proba(X)
    label_map  = {-1: "Down", 0: "Neutral", 1: "Up"}
    return pd.DataFrame({
        "dir_pred":      [label_map.get(int(p), "Neutral") for p in raw_preds],
        "dir_down_prob": raw_probas[:, 0],
        "dir_up_prob":   raw_probas[:, 2],
    }, index=X.index)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics printing
# ══════════════════════════════════════════════════════════════════════════════

def print_metrics_table(metrics_list: list, priority: list[str] = None):
    keys = [
        ("CAGR",         "cagr",          "{:.2%}"),
        ("Vol",          "vol",           "{:.2%}"),
        ("Sharpe",       "sharpe",        "{:.3f}"),
        ("Sortino",      "sortino",       "{:.3f}"),
        ("Calmar",       "calmar",        "{:.3f}"),
        ("Max DD",       "max_drawdown",  "{:.2%}"),
        ("Total Return", "total_return",  "{:.2%}"),
        ("% Invested",   "pct_invested",  "{:.1%}"),
        ("Alpha (CAGR)", "alpha_cagr",    "{:+.2%}"),
        ("Composite ★",  "composite_score", "{:.3f}"),
    ]
    names  = [m["name"] for m in metrics_list]
    col_w  = max(14, max(len(n) for n in names) + 2)
    header = f"{'Metric':<18}" + "".join(f"{n:>{col_w}}" for n in names)
    sep    = "-" * len(header)

    print(f"\n{sep}\nSTRATEGY PERFORMANCE COMPARISON")
    if priority:
        print(f"  Optimization objective: {' > '.join(priority)}")
    print(f"{sep}\n{header}\n{sep}")

    for label, key, fmt in keys:
        marker = " ★" if priority and key == priority[0] else "  "
        row = f"{label:<18}"
        for m in metrics_list:
            val = m.get(key)
            if val is None:
                row += f"{'—':>{col_w}}"
            else:
                try:
                    row += f"{fmt.format(val):>{col_w}}"
                except Exception:
                    row += f"{'N/A':>{col_w}}"
        print(f"{marker}{row}")
    print(sep)

    bh = next((m for m in metrics_list if m["name"] == "B&H SPY"), {})
    print(f"\nBenchmark (B&H SPY): CAGR {bh.get('bh_cagr',0):.2%}  "
          f"Sharpe {bh.get('bh_sharpe',0):.3f}  "
          f"Max DD {bh.get('bh_max_dd',0):.2%}")

    ranked = sorted([m for m in metrics_list if m.get("rank")], key=lambda x: x["rank"])
    print(f"\nRanking (by composite score):")
    for m in ranked:
        beats = []
        if m["cagr"] > bh.get("bh_cagr", 0):   beats.append("Return")
        if m["sharpe"] > bh.get("bh_sharpe", 0): beats.append("Sharpe")
        if m["calmar"] > bh.get("bh_calmar", 0): beats.append("Calmar")
        beat_str = ", ".join(beats) if beats else "none"
        print(f"  #{m['rank']} {m['name']} (score={m['composite_score']:.2f}) — beats B&H: {beat_str}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(
    results: dict,
    csi_sig: pd.DataFrame,
    metrics: list,
    output_dir: Path = OUTPUT_DIR,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 14), facecolor="#1a1a2e")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.28)

    ax_eq  = fig.add_subplot(gs[0, :])
    ax_dd  = fig.add_subplot(gs[1, :])
    ax_csi = fig.add_subplot(gs[2, 0])
    ax_bar = fig.add_subplot(gs[2, 1])

    for ax in [ax_eq, ax_dd, ax_csi, ax_bar]:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#e0e0e0")
        for sp in ax.spines.values():
            sp.set_color("#444")

    # Equity curves
    for name, bt in results.items():
        color = PALETTE.get(name, "#aaa")
        lw    = 2.2 if name == "B&H SPY" else 1.6
        ls    = "--" if name == "B&H SPY" else "-"
        ax_eq.plot(bt.index, bt["cum_strat"], label=name, color=color, lw=lw, ls=ls, alpha=0.92)

    ax_eq.set_ylabel("Portfolio Value ($1 start)", color="#e0e0e0", fontsize=10)
    ax_eq.set_title("Equity Curves — FMSI Strategies vs Buy & Hold SPY",
                    color="#4fc3f7", fontsize=12)
    ax_eq.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e",
                 framealpha=0.6, loc="upper left", ncol=3)
    ax_eq.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"${y:.1f}"))
    ax_eq.grid(axis="y", color="#333", lw=0.5)

    # Drawdown
    for name, bt in results.items():
        color = PALETTE.get(name, "#aaa")
        ax_dd.fill_between(bt.index, bt["drawdown"] * 100, 0, color=color, alpha=0.25)
        ax_dd.plot(bt.index, bt["drawdown"] * 100, color=color, lw=0.8, alpha=0.7, label=name)
    ax_dd.set_ylabel("Drawdown (%)", color="#e0e0e0", fontsize=10)
    ax_dd.set_title("Drawdown", color="#4fc3f7", fontsize=11)
    ax_dd.grid(axis="y", color="#333", lw=0.5)
    ax_dd.invert_yaxis()

    # CSI history
    csi_plot = csi_sig["csi_composite"].dropna()
    ax_csi.plot(csi_plot.index, csi_plot, color="#4fc3f7", lw=0.9, alpha=0.85)
    for thr, col, lbl in [(30, "#2ECC71", "Low/Elev"), (50, "#F39C12", "Elev/High"), (70, "#E74C3C", "High/Ext")]:
        ax_csi.axhline(thr, color=col, lw=0.7, ls="--", alpha=0.7, label=lbl)
    ax_csi.set_ylabel("CSI Composite", color="#e0e0e0", fontsize=10)
    ax_csi.set_title("CSI Signal History", color="#4fc3f7", fontsize=11)
    ax_csi.legend(fontsize=7, labelcolor="white", facecolor="#1a1a2e", framealpha=0.5)
    ax_csi.set_ylim(0, 100)
    ax_csi.grid(axis="y", color="#333", lw=0.5)

    # Bar chart (Sharpe / Calmar / CAGR)
    bar_keys   = ["sharpe", "calmar", "cagr"]
    bar_labels = ["Sharpe", "Calmar", "CAGR (%)"]
    n_s = len(results)
    bar_w = 0.7 / n_s
    x    = np.arange(len(bar_keys))

    for i, (name, bt) in enumerate(results.items()):
        m    = next((mm for mm in metrics if mm["name"] == name), compute_metrics(bt, name))
        ys   = [m.get("sharpe", 0), min(m.get("calmar", 0), 10), m.get("cagr", 0) * 100]
        xpos = x + (i - n_s / 2 + 0.5) * bar_w
        col  = PALETTE.get(name, "#aaa")
        bars = ax_bar.bar(xpos, ys, width=bar_w * 0.9, color=col, alpha=0.85, label=name)
        for bar, y in zip(bars, ys):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{y:.2f}", ha="center", va="bottom", color="#e0e0e0", fontsize=6.5)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(bar_labels, color="#e0e0e0", fontsize=9)
    ax_bar.set_title("Risk-Adjusted Performance", color="#4fc3f7", fontsize=11)
    ax_bar.axhline(0, color="#888", lw=0.8)
    ax_bar.legend(fontsize=7, labelcolor="white", facecolor="#1a1a2e",
                  framealpha=0.5, loc="upper right")
    ax_bar.grid(axis="y", color="#333", lw=0.5)

    plt.suptitle("FMSI Trading Strategies — Full Backtest Results",
                 color="#e0e0e0", fontsize=14, y=0.99)

    out_path = output_dir / "strategy_backtest.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    logger.info("Chart saved: %s", out_path)
    return str(out_path)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FMSI Trading Strategy Backtester")
    parser.add_argument("--start",      default="2003-01-01")
    parser.add_argument("--end",        default=None)
    parser.add_argument("--optimize",   action="store_true")
    parser.add_argument("--no-model",   action="store_true")
    parser.add_argument("--horizon",    type=int, default=5)
    parser.add_argument("--priority",   nargs="+",
                        default=["sharpe", "calmar", "cagr"],
                        help="Metric priority for optimization (e.g. sharpe calmar cagr)")
    parser.add_argument("--strategies", nargs="+",
                        default=["S1", "S2", "S4", "S5"],
                        help="Strategies to run (S1 S2 S3 S4 S5)")
    parser.add_argument("--output-dir", default="reports/strategy")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    spy_returns = load_spy_returns(args.start, args.end or "2026-12-31")
    csi_sig     = load_csi_signal()

    common_idx  = spy_returns.index.intersection(csi_sig.index)
    spy_returns = spy_returns.loc[common_idx]
    csi_sig     = csi_sig.loc[common_idx]

    logger.info("Backtest period: %s → %s (%d days)",
                common_idx[0].date(), common_idx[-1].date(), len(common_idx))

    # Direction signal for S3
    dir_sig = pd.DataFrame()
    strategies = [s.upper() for s in args.strategies]
    if "S3" in strategies and not args.no_model:
        try:
            dir_sig = load_model_direction_signal(horizon=args.horizon, start=args.start)
            logger.info("Direction signal loaded (%d rows).", len(dir_sig))
        except Exception as exc:
            logger.warning("Direction signal unavailable: %s — S3 will fall back to S1.", exc)

    priority = args.priority

    def progress(frac, msg):
        logger.info("[%.0f%%] %s", frac * 100, msg)

    output = run_all_strategies(
        spy_returns, csi_sig, strategies, priority,
        optimize=args.optimize,
        dir_sig=dir_sig if not dir_sig.empty else None,
        progress_cb=progress,
    )

    print_metrics_table(output["metrics"], priority=priority)

    chart_path = plot_results(
        output["results"], csi_sig, output["metrics"],
        output_dir=Path(args.output_dir),
    )
    print(f"Chart saved → {chart_path}")

    pd.DataFrame(output["metrics"]).to_csv(
        Path(args.output_dir) / "strategy_metrics.csv", index=False
    )

    # Sub-period analysis
    sub_df = subperiod_returns(output["results"])
    print("\n" + "=" * 60)
    print("SUB-PERIOD ANALYSIS (total return / CAGR if period > 6mo)")
    print("=" * 60)
    with pd.option_context("display.float_format", "{:.1%}".format,
                           "display.max_columns", None, "display.width", 160):
        print(sub_df.to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()
