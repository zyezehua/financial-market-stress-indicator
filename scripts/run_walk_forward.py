"""
Run walk-forward OOS backtest for all horizons and save predictions to parquet.

This is the only correct way to generate direction signals for S3 strategy
backtesting — predictions are made exclusively on data the model has never seen.

Usage:
    python scripts/run_walk_forward.py                  # all horizons (5, 21, 63d)
    python scripts/run_walk_forward.py --horizon 5      # single horizon
    python scripts/run_walk_forward.py --step 21        # retrain every 21 days (finer)
    python scripts/run_walk_forward.py --start 2010-01-01  # skip early history

Output:
    data/processed/oos_stress_h{N}d.parquet      — stress score + class predictions
    data/processed/oos_direction_h{N}d.parquet   — direction + down/up probabilities

Runtime: ~10-20 minutes for all 3 horizons on a modern laptop (40+ retraining cycles).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from src.features.builder import load_features, add_regime_features
from src.labels.composite_index import load_csi
from src.models.walk_forward import run_walk_forward, compute_oos_metrics, print_oos_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_walk_forward")

OUTPUT_DIR = Path("data/processed")


def main():
    parser = argparse.ArgumentParser(description="Walk-forward OOS backtest")
    parser.add_argument(
        "--horizon", type=int, nargs="+", default=[5, 21, 63],
        help="Prediction horizons in days (default: 5 21 63)",
    )
    parser.add_argument(
        "--step", type=int, default=63,
        help="Trading days between retrainings (default: 63 = quarterly)",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Earliest prediction date, e.g. 2010-01-01 (default: min_train threshold)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading features and CSI ...")
    features = load_features()
    csi      = load_csi()
    if "regime_csi_mean_504d" not in features.columns:
        features = add_regime_features(features, csi, save=False)

    targets_path = Path("data/labels/targets.parquet")
    if not targets_path.exists():
        logger.error("targets.parquet not found at %s", targets_path)
        sys.exit(1)
    targets = pd.read_parquet(targets_path)

    logger.info(
        "Data loaded: features=%s, targets=%s, csi=%s",
        features.shape, targets.shape, csi.shape,
    )

    all_metrics = []
    for h in args.horizon:
        logger.info("=" * 60)
        logger.info("Walk-forward h=%dd (step=%d days) ...", h, args.step)
        logger.info("=" * 60)

        result = run_walk_forward(
            features=features,
            targets=targets,
            csi=csi,
            horizon=h,
            step=args.step,
            start_date=args.start,
        )

        stress_df = result["stress"]
        dir_df    = result["direction"]

        if not stress_df.empty:
            out = OUTPUT_DIR / f"oos_stress_h{h}d.parquet"
            stress_df.to_parquet(out)
            logger.info("Saved stress OOS → %s (%d rows)", out, len(stress_df))
        else:
            logger.warning("No stress predictions generated for h=%dd.", h)

        if not dir_df.empty:
            out = OUTPUT_DIR / f"oos_direction_h{h}d.parquet"
            dir_df.to_parquet(out)
            logger.info("Saved direction OOS → %s (%d rows)", out, len(dir_df))
        else:
            logger.warning("No direction predictions generated for h=%dd.", h)

        metrics = compute_oos_metrics(stress_df, dir_df, h)
        all_metrics.append(metrics)

    print_oos_summary(all_metrics)
    logger.info(
        "Done. Upload results with: python scripts/upload_artifacts.py"
    )


if __name__ == "__main__":
    main()
