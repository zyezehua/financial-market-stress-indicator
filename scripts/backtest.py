"""
Backtesting script: evaluates prediction accuracy over historical stress episodes.

Usage:
    python scripts/backtest.py                         # full backtest
    python scripts/backtest.py --horizon 5 --start 2020-01-01
    python scripts/backtest.py --episode covid         # specific episode
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.pipeline import DataPipeline
from src.features.builder import build_features, load_features
from src.labels.composite_index import build_csi, load_csi
from src.labels.calibrator import STRESS_EPISODES
from src.models.predictor import predict_historical

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backtest")

EPISODES = {
    "dotcom":    ("2000-03-01", "2002-10-31"),
    "gfc":       ("2008-09-01", "2009-03-31"),
    "covid":     ("2020-02-20", "2020-05-31"),
    "bear2022":  ("2022-01-01", "2022-12-31"),
}


def compute_metrics(pred_df: pd.DataFrame, horizon: int) -> dict:
    """Compute regression and classification metrics."""
    df = pred_df.dropna(subset=["stress_score_pred", "stress_actual"])
    if df.empty:
        return {}

    mae  = float(np.mean(np.abs(df["stress_score_pred"] - df["stress_actual"])))
    rmse = float(np.sqrt(np.mean((df["stress_score_pred"] - df["stress_actual"]) ** 2)))
    corr = float(df["stress_score_pred"].corr(df["stress_actual"]))

    # Classification accuracy (same stress bucket)
    def bucket(v):
        if v <= 25: return 0
        if v <= 50: return 1
        if v <= 75: return 2
        return 3

    pred_cls   = df["stress_score_pred"].apply(bucket)
    actual_cls = df["stress_actual"].apply(bucket)
    cls_acc    = float((pred_cls == actual_cls).mean())

    return {
        "horizon":          horizon,
        "n_obs":            len(df),
        "mae":              round(mae, 2),
        "rmse":             round(rmse, 2),
        "pearson_r":        round(corr, 3),
        "class_accuracy":   round(cls_acc, 3),
    }


def plot_backtest(pred_df: pd.DataFrame, horizon: int, output_dir: str = "reports"):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), facecolor="#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")

    # Panel 1: predicted vs actual
    axes[0].plot(pred_df.index, pred_df["stress_actual"], color="#4fc3f7",
                 lw=1.2, label="Actual CSI", alpha=0.9)
    axes[0].plot(pred_df.index, pred_df["stress_score_pred"], color="#FFD700",
                 lw=1.2, label=f"Predicted ({horizon}d ahead)", ls="--", alpha=0.85)
    axes[0].axhline(50, color="#888", lw=0.5, ls=":")
    axes[0].axhline(75, color="#E74C3C", lw=0.5, ls=":", alpha=0.6)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("CSI Score", color="#e0e0e0")
    axes[0].legend(fontsize=8, labelcolor="white", facecolor="#16213e", framealpha=0.5)
    axes[0].tick_params(colors="#e0e0e0")
    for ep_name, (start, end) in STRESS_EPISODES.items():
        try:
            axes[0].axvspan(pd.Timestamp(start), pd.Timestamp(end),
                            alpha=0.08, color="#E74C3C")
        except Exception:
            pass

    # Panel 2: prediction error
    error = pred_df["stress_score_pred"] - pred_df["stress_actual"]
    axes[1].fill_between(pred_df.index, error, 0,
                         where=(error >= 0), color="#E74C3C", alpha=0.5, label="Overestimate")
    axes[1].fill_between(pred_df.index, error, 0,
                         where=(error < 0),  color="#2ECC71", alpha=0.5, label="Underestimate")
    axes[1].axhline(0, color="#888", lw=0.8)
    axes[1].set_ylabel("Prediction Error", color="#e0e0e0")
    axes[1].legend(fontsize=8, labelcolor="white", facecolor="#16213e", framealpha=0.5)
    axes[1].tick_params(colors="#e0e0e0")

    plt.suptitle(f"Backtest — {horizon}-Day Horizon", color="#4fc3f7", fontsize=13)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"backtest_h{horizon}d.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    logger.info("Backtest chart saved: %s", path)


def run(horizon: int = None, start: str = None, episode: str = None,
        output_dir: str = "reports", artifacts_dir: str = "models"):

    # Load pre-computed features and CSI
    try:
        features = load_features()
        csi      = load_csi()
    except FileNotFoundError:
        logger.info("Pre-computed data not found. Running full pipeline ...")
        pipeline = DataPipeline()
        data = pipeline.run()
        features = build_features(data["prices"], data["macro"])
        csi = build_csi(features)

    ep_start, ep_end = None, None
    if episode and episode in EPISODES:
        ep_start, ep_end = EPISODES[episode]
        logger.info("Backtesting episode: %s (%s → %s)", episode, ep_start, ep_end)

    horizons = [horizon] if horizon else [5, 21, 63]
    all_metrics = []

    for h in horizons:
        try:
            pred_df = predict_historical(
                features, csi, h,
                start=start or ep_start,
                end=ep_end,
                artifacts_dir=artifacts_dir,
            )
            metrics = compute_metrics(pred_df, h)
            all_metrics.append(metrics)
            logger.info("Horizon %dd: %s", h, metrics)
            plot_backtest(pred_df, h, output_dir)
        except FileNotFoundError as exc:
            logger.error(str(exc))

    if all_metrics:
        summary = pd.DataFrame(all_metrics).set_index("horizon")
        print("\n" + "=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)
        print(summary.to_string())
        print("=" * 60 + "\n")

    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest stress prediction models")
    parser.add_argument("--horizon",  type=int,  default=None)
    parser.add_argument("--start",    type=str,  default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--episode",  type=str,  default=None,
                        choices=list(EPISODES.keys()))
    parser.add_argument("--output-dir",    default="reports")
    parser.add_argument("--artifacts-dir", default="models")
    args = parser.parse_args()

    run(
        horizon=args.horizon,
        start=args.start,
        episode=args.episode,
        output_dir=args.output_dir,
        artifacts_dir=args.artifacts_dir,
    )
