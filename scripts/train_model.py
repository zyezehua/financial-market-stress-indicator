"""
Model training script.

Runs the full data pipeline, builds features and labels, then trains
ensemble models for all configured horizons.

Usage:
    python scripts/train_model.py                    # train from scratch
    python scripts/train_model.py --horizon 5        # single horizon
    python scripts/train_model.py --validate         # print alignment stats
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import yaml

from src.data.pipeline import DataPipeline
from src.features.builder import build_features
from src.labels.composite_index import build_csi, build_prediction_targets
from src.labels.fred_benchmark import load_fred_fsi
from src.labels.calibrator import compute_alignment, adjust_csi_weights
from src.models.trainer import train_all_horizons, train_horizon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_model")


def run(horizon: int = None, validate: bool = False):
    with open("config/settings.yaml") as f:
        cfg = yaml.safe_load(f)

    horizons    = [horizon] if horizon else cfg["models"]["horizons"]
    artifacts   = cfg["models"]["artifacts_dir"]
    weights_cfg = cfg["models"]["ensemble_weights"]

    # ── 1. Fetch full history ───────────────────────────────────────────────
    logger.info("Fetching full market data history (%s → present) ...",
                cfg["data"]["start_date"])
    pipeline = DataPipeline()
    data = pipeline.run()
    prices = data["prices"]
    macro  = data["macro"]

    # ── 2. Build features ──────────────────────────────────────────────────
    logger.info("Building feature matrix ...")
    features = build_features(prices, macro)

    # ── 3. Build labels ────────────────────────────────────────────────────
    logger.info("Building CSI labels ...")
    csi = build_csi(features)

    # Optionally calibrate weights against FRED FSI
    fred_fsi = load_fred_fsi(macro)
    if not fred_fsi.empty and validate:
        logger.info("Calibrating CSI against FRED FSI ...")
        stats = compute_alignment(csi["csi_composite"], fred_fsi)
        logger.info("Alignment stats: %s", stats)

    targets = build_prediction_targets(prices, csi, horizons)

    # ── 4. Train models ────────────────────────────────────────────────────
    logger.info("Training models for horizons: %s ...", horizons)
    results = {}
    for h in horizons:
        try:
            result = train_horizon(features, targets, h,
                                   weights=weights_cfg, artifacts_dir=artifacts)
            results[h] = result
            m = result.get("stress_metrics", {})
            dm = result.get("direction_metrics", {})
            logger.info(
                "Horizon %dd — Stress CV MAE: %.2f | Direction Acc: %s",
                h,
                m.get("cv_mae") or -1,
                dm.get("direction_accuracy_last_500", "N/A"),
            )
        except Exception as exc:
            logger.error("Training failed for horizon %dd: %s", h, exc)

    logger.info("Training complete. Artifacts saved to: %s/", artifacts)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stress prediction models")
    parser.add_argument("--horizon",  type=int, default=None, help="Single horizon to train (5, 21, or 63)")
    parser.add_argument("--validate", action="store_true", help="Run CSI vs FRED FSI validation")
    args = parser.parse_args()
    run(horizon=args.horizon, validate=args.validate)
