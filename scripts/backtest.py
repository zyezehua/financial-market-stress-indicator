"""
Backtesting script: evaluates prediction accuracy over historical stress episodes.

Two modes
─────────
  default      In-sample historical predictions (fast, for visual inspection only)
  --walkforward True OOS predictions via expanding-window retraining (slow but honest)

Usage
─────
    # Quick in-sample visual backtest
    python scripts/backtest.py --horizon 5

    # True OOS walk-forward (all horizons)
    python scripts/backtest.py --walkforward

    # OOS for a single horizon, starting from a given date
    python scripts/backtest.py --walkforward --horizon 5 --start 2015-01-01

    # In-sample for a named episode only
    python scripts/backtest.py --episode gfc

    # Market Crisis Signal lifecycle (entry + exit detection, all horizons)
    python scripts/backtest.py --crisis-signal

    # Crisis signal for a single horizon
    python scripts/backtest.py --crisis-signal --horizon 21
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
from sklearn.metrics import classification_report

from src.data.pipeline import DataPipeline
from src.features.builder import build_features, load_features, add_regime_features
from src.labels.composite_index import build_csi, load_csi, build_prediction_targets, load_targets
from src.labels.calibrator import STRESS_EPISODES
from src.models.predictor import predict_historical
from src.models.walk_forward import run_walk_forward, compute_oos_metrics, print_oos_summary
from src.labels.crisis_signal import analyze_all_episodes, compute_false_alarm_rate, print_lifecycle_report
from src.models.regime_detector import RegimeDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backtest")

# Named episodes for --episode flag (subset of STRESS_EPISODES for convenience)
NAMED_EPISODES = {
    "dotcom":    ("2000-03-01", "2002-09-30"),
    "iraq":      ("2003-01-01", "2003-04-30"),
    "subprime":  ("2007-06-01", "2008-08-31"),
    "gfc":       ("2008-09-01", "2009-03-31"),
    "flash2010": ("2010-05-01", "2010-06-30"),
    "euro":      ("2011-07-01", "2011-10-31"),
    "taper":     ("2013-05-01", "2013-09-30"),
    "china":     ("2015-08-01", "2016-02-29"),
    "brexit":    ("2016-06-01", "2016-07-31"),
    "volmaged":  ("2018-02-01", "2018-02-28"),
    "q42018":    ("2018-10-01", "2018-12-31"),
    "covid":     ("2020-02-20", "2020-04-30"),
    "inflation21":("2021-10-01", "2021-12-31"),
    "bear2022":  ("2022-01-01", "2022-12-31"),
    "svb":       ("2023-03-01", "2023-10-31"),
    "rates24":   ("2024-01-01", "2024-03-31"),
    "yencarry":  ("2024-07-31", "2024-08-16"),
    "tariff2025":("2025-04-02", "2025-05-12"),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _bucket(v: float) -> int:
    if v <= 25: return 0
    if v <= 50: return 1
    if v <= 75: return 2
    return 3


def compute_insample_metrics(pred_df: pd.DataFrame, horizon: int) -> dict:
    """Regression + classification metrics for in-sample predictions."""
    df = pred_df.dropna(subset=["stress_score_pred", "stress_actual"])
    if df.empty:
        return {}

    err  = df["stress_score_pred"] - df["stress_actual"]
    mae  = float(err.abs().mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    corr = float(df["stress_score_pred"].corr(df["stress_actual"]))

    pred_cls   = df["stress_score_pred"].apply(_bucket)
    actual_cls = df["stress_actual"].apply(_bucket)
    cls_acc    = float((pred_cls == actual_cls).mean())

    return {
        "horizon":        horizon,
        "n_obs":          len(df),
        "mae":            round(mae, 2),
        "rmse":           round(rmse, 2),
        "pearson_r":      round(corr, 3),
        "class_accuracy": round(cls_acc, 3),
    }


def plot_backtest(
    pred_df: pd.DataFrame,
    horizon: int,
    output_dir: str = "reports",
    tag: str = "",
):
    """Predicted-vs-actual + error panel chart."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor="#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")

    label_pred = f"Predicted ({horizon}d ahead)"
    axes[0].plot(pred_df.index, pred_df["stress_actual"], color="#4fc3f7",
                 lw=1.2, label="Actual CSI", alpha=0.9)
    axes[0].plot(pred_df.index, pred_df["stress_score_pred"], color="#FFD700",
                 lw=1.2, label=label_pred, ls="--", alpha=0.85)
    axes[0].axhline(25, color="#2ECC71", lw=0.5, ls=":", alpha=0.5)
    axes[0].axhline(50, color="#888",    lw=0.5, ls=":")
    axes[0].axhline(75, color="#E74C3C", lw=0.5, ls=":", alpha=0.6)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("CSI Score", color="#e0e0e0")
    axes[0].legend(fontsize=8, labelcolor="white", facecolor="#16213e", framealpha=0.5)
    axes[0].tick_params(colors="#e0e0e0")

    # Shade all known stress episodes
    for ep_name, (start, end) in STRESS_EPISODES.items():
        try:
            axes[0].axvspan(
                pd.Timestamp(start), pd.Timestamp(end),
                alpha=0.08, color="#E74C3C", label="_nolegend_",
            )
        except Exception:
            pass

    error = pred_df["stress_score_pred"] - pred_df["stress_actual"]
    axes[1].fill_between(pred_df.index, error, 0,
                         where=(error >= 0), color="#E74C3C", alpha=0.5, label="Over")
    axes[1].fill_between(pred_df.index, error, 0,
                         where=(error < 0),  color="#2ECC71", alpha=0.5, label="Under")
    axes[1].axhline(0, color="#888", lw=0.8)
    axes[1].set_ylabel("Prediction Error", color="#e0e0e0")
    axes[1].legend(fontsize=8, labelcolor="white", facecolor="#16213e", framealpha=0.5)
    axes[1].tick_params(colors="#e0e0e0")

    mode = "OOS Walk-forward" if tag == "oos" else "In-sample"
    plt.suptitle(f"Backtest [{mode}] — {horizon}-Day Horizon", color="#4fc3f7", fontsize=13)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fname = f"backtest_h{horizon}d{'_oos' if tag == 'oos' else ''}.png"
    path  = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    logger.info("Chart saved: %s", path)
    return path


# ── Crisis Signal mode ─────────────────────────────────────────────────────────

def run_crisis_signal(
    features, csi,
    horizons,
    start=None,
    artifacts_dir="models",
):
    """Run Market Crisis Signal lifecycle analysis for all stress episodes."""
    # Fit HMM regime detector on full CSI history
    regimes = None
    try:
        detector = RegimeDetector()
        detector.fit(csi["csi_composite"])
        regimes = detector.predict(csi["csi_composite"])
        reg_idx, reg_name = detector.current_regime(csi["csi_composite"])
        logger.info("HMM regime detector fitted. Current regime: %d (%s)", reg_idx, reg_name)
    except Exception as exc:
        logger.warning("RegimeDetector failed (%s) — using flat regime=1 fallback", exc)

    for h in horizons:
        logger.info("── Crisis Signal — Horizon %dd ──────────────────────", h)
        try:
            pred_df = predict_historical(
                features, csi, h,
                start=start,
                artifacts_dir=artifacts_dir,
            )
        except FileNotFoundError as exc:
            logger.error(str(exc))
            continue

        if pred_df.empty or "stress_score_pred" not in pred_df.columns:
            logger.warning("No predictions for h=%dd — skipping.", h)
            continue

        results = analyze_all_episodes(pred_df, STRESS_EPISODES, h, regimes=regimes)

        if not results:
            logger.warning("No episodes in prediction range for h=%dd.", h)
            continue

        pred_full = pred_df["stress_score_pred"].dropna()
        fa_stats  = compute_false_alarm_rate(pred_full, STRESS_EPISODES, regimes=regimes)

        print_lifecycle_report(results, h, fa_stats)


# ── In-sample mode ─────────────────────────────────────────────────────────────

def run_insample(
    features, csi,
    horizons, start=None, episode=None,
    output_dir="reports", artifacts_dir="models",
):
    ep_start, ep_end = None, None
    if episode and episode in NAMED_EPISODES:
        ep_start, ep_end = NAMED_EPISODES[episode]
        logger.info("Episode filter: %s (%s → %s)", episode, ep_start, ep_end)

    all_metrics = []
    for h in horizons:
        try:
            pred_df = predict_historical(
                features, csi, h,
                start=start or ep_start,
                end=ep_end,
                artifacts_dir=artifacts_dir,
            )
            metrics = compute_insample_metrics(pred_df, h)
            all_metrics.append(metrics)
            logger.info("In-sample h=%dd: %s", h, metrics)
            plot_backtest(pred_df, h, output_dir, tag="insample")
        except FileNotFoundError as exc:
            logger.error(str(exc))

    if all_metrics:
        summary = pd.DataFrame(all_metrics).set_index("horizon")
        print("\n" + "=" * 60)
        print("IN-SAMPLE BACKTEST SUMMARY  (⚠ not true OOS — use --walkforward)")
        print("=" * 60)
        print(summary.to_string())
        print("=" * 60 + "\n")

    return all_metrics


# ── Walk-forward OOS mode ──────────────────────────────────────────────────────

def run_oos(
    features, csi, targets,
    horizons, start=None, step=63,
    output_dir="reports",
    ensemble_weights=None,
):
    logger.info(
        "Starting walk-forward OOS backtest: horizons=%s, step=%dd", horizons, step
    )
    all_metrics = []

    for h in horizons:
        logger.info("── Horizon %dd ──────────────────────────────────", h)
        try:
            result = run_walk_forward(
                features, targets, csi,
                horizon=h,
                step=step,
                start_date=start,
                ensemble_weights=ensemble_weights,
            )
        except Exception as exc:
            logger.error("Walk-forward failed for h=%dd: %s", h, exc)
            continue

        stress_df = result["stress"]
        dir_df    = result["direction"]

        metrics = compute_oos_metrics(stress_df, dir_df, h)
        all_metrics.append(metrics)

        # Build a pred_df compatible with plot_backtest
        if not stress_df.empty:
            plot_df = stress_df[["stress_score_pred", "stress_actual"]].copy()
            plot_backtest(plot_df, h, output_dir, tag="oos")

    print_oos_summary(all_metrics)
    return all_metrics


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest stress prediction models")
    parser.add_argument(
        "--walkforward", action="store_true",
        help="Run true OOS walk-forward backtest (slow but correct)",
    )
    parser.add_argument(
        "--crisis-signal", action="store_true",
        help="Run Market Crisis Signal lifecycle analysis (entry + exit detection per episode)",
    )
    parser.add_argument("--horizon", type=int, default=None,
                        help="Single horizon to backtest (default: all)")
    parser.add_argument("--start",   type=str, default=None,
                        help="Start date for OOS predictions (YYYY-MM-DD)")
    parser.add_argument("--step",    type=int, default=63,
                        help="Retraining step size in trading days (default 63 ≈ quarterly)")
    parser.add_argument("--episode", type=str, default=None,
                        choices=list(NAMED_EPISODES.keys()),
                        help="Named episode filter (in-sample mode only)")
    parser.add_argument("--output-dir",    default="reports")
    parser.add_argument("--artifacts-dir", default="models")
    args = parser.parse_args()

    # ── Load or build data ─────────────────────────────────────────────────
    try:
        features = load_features()
        csi      = load_csi()
        logger.info("Loaded pre-computed features (%d rows) and CSI.", len(features))
    except FileNotFoundError:
        logger.info("Pre-computed data not found — running full pipeline ...")
        pipeline = DataPipeline()
        data     = pipeline.run()
        features = build_features(data["prices"], data["macro"])
        csi      = build_csi(features)

    # Add regime features if not already present (idempotent)
    if "regime_csi_mean_504d" not in features.columns:
        logger.info("Adding regime features from CSI ...")
        features = add_regime_features(features, csi, save=True)

    horizons = [args.horizon] if args.horizon else [5, 21, 63]

    if args.crisis_signal:
        run_crisis_signal(
            features, csi,
            horizons=horizons,
            start=args.start,
            artifacts_dir=args.artifacts_dir,
        )
    elif args.walkforward:
        # Load pre-saved targets; fall back to building from prices if missing
        try:
            targets = load_targets()
            logger.info("Loaded pre-computed targets (%d rows).", len(targets))
        except FileNotFoundError:
            logger.info("Targets not found — loading prices to build them ...")
            from src.data.pipeline import DataPipeline as _DP
            _pipe = _DP()
            _data = _pipe.update()
            targets = build_prediction_targets(_data["prices"], csi, save=True)

        run_oos(
            features, csi, targets,
            horizons=horizons,
            start=args.start,
            step=args.step,
            output_dir=args.output_dir,
        )
    else:
        run_insample(
            features, csi,
            horizons=horizons,
            start=args.start,
            episode=args.episode,
            output_dir=args.output_dir,
            artifacts_dir=args.artifacts_dir,
        )


if __name__ == "__main__":
    main()
