"""
Daily pipeline entry point.

Usage:
    python scripts/daily_run.py                    # run for today
    python scripts/daily_run.py --date 2024-08-15  # run for a specific date
    python scripts/daily_run.py --no-pdf           # skip PDF
    python scripts/daily_run.py --use-llm          # enable Claude narrative
"""

import argparse
import logging
import os
import sys

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import yaml
import pandas as pd

from src.data.pipeline import DataPipeline
from src.features.builder import build_features, load_features, get_feature_groups
from src.labels.composite_index import build_csi, build_prediction_targets, load_csi
from src.labels.fred_benchmark import load_fred_fsi
from src.models.predictor import predict_latest
from src.models.trainer import load_artifact
from src.explainability.shap_analyzer import compute_shap_values, top_factors, shap_summary
from src.explainability.factor_attribution import (
    build_attribution, current_feature_percentiles
)
from src.explainability.narrative import generate_narrative
from src.reporting.html_dashboard import generate_html
from src.reporting.pdf_generator import generate_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("daily_run")


def run(date: str = None, use_llm: bool = False, gen_pdf: bool = True, gen_html: bool = True):
    with open("config/settings.yaml") as f:
        cfg = yaml.safe_load(f)

    horizons    = cfg["models"]["horizons"]
    artifacts   = cfg["models"]["artifacts_dir"]
    n_top       = cfg["reporting"]["top_n_factors"]
    output_dir  = cfg["reporting"]["output_dir"]
    llm_model   = cfg["anthropic"]["model"]
    max_tokens  = cfg["anthropic"]["max_tokens"]

    # ── 1. Update data ──────────────────────────────────────────────────────
    logger.info("Step 1/6: Updating market data ...")
    pipeline = DataPipeline()
    data = pipeline.update()
    prices = data["prices"]
    macro  = data["macro"]

    # ── 2. Build / refresh features ────────────────────────────────────────
    logger.info("Step 2/6: Building features ...")
    features = build_features(prices, macro)
    feature_groups = get_feature_groups(features)

    # ── 3. Rebuild CSI + targets ───────────────────────────────────────────
    logger.info("Step 3/6: Building CSI labels ...")
    csi     = build_csi(features)
    targets = build_prediction_targets(prices, csi, horizons)

    # ── 4. Run multi-horizon predictions ──────────────────────────────────
    logger.info("Step 4/6: Running predictions ...")
    predictions = predict_latest(features, csi, horizons, artifacts_dir=artifacts, as_of=date)

    if not predictions:
        logger.error("No predictions generated. Ensure models are trained first (run train_model.py).")
        return

    # ── 5. Compute SHAP + attribution ──────────────────────────────────────
    logger.info("Step 5/6: Computing SHAP attribution ...")
    attribution_by_horizon = {}
    shap_row_date = pd.Timestamp(date) if date else features.index[-1]
    if shap_row_date not in features.index:
        shap_row_date = features.index[-1]

    for h in horizons:
        try:
            artifact     = load_artifact(h, artifacts)
            feat_names   = artifact["feature_names"]
            lgbm_model   = artifact["stress_ensemble"].models["lgbm"]
            X_window     = features.reindex(columns=feat_names, fill_value=0).tail(63)
            shap_df      = compute_shap_values(lgbm_model, X_window)
            top_shap     = top_factors(shap_df, shap_row_date if shap_row_date in shap_df.index
                                       else shap_df.index[-1], n=n_top)
            pct_ranks    = current_feature_percentiles(
                features.reindex(columns=feat_names, fill_value=0), shap_row_date
                if shap_row_date in features.index else features.index[-1]
            )
            attr = build_attribution(top_shap, features.loc[features.index[-1]], pct_ranks)
            attribution_by_horizon[h] = attr
        except Exception as exc:
            logger.warning("SHAP failed for horizon %dd: %s", h, exc)
            attribution_by_horizon[h] = []

    # ── 6. Generate narrative + reports ────────────────────────────────────
    logger.info("Step 6/6: Generating reports ...")
    narrative = generate_narrative(
        predictions,
        attribution_by_horizon.get(5, []),
        attribution_by_horizon.get(21, []),
        attribution_by_horizon.get(63, []),
        csi,
        use_llm=use_llm,
        model=llm_model,
        max_tokens=max_tokens,
    )

    report_date = date or features.index[-1].strftime("%Y-%m-%d")
    if gen_html:
        html_path = generate_html(predictions, attribution_by_horizon, csi, narrative,
                                  output_dir=output_dir, as_of=report_date)
        logger.info("HTML dashboard: %s", html_path)

    if gen_pdf:
        pdf_path = generate_pdf(predictions, attribution_by_horizon, csi, narrative,
                                output_dir=output_dir, as_of=report_date)
        logger.info("PDF report: %s", pdf_path)

    logger.info("Daily run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run daily stress indicator pipeline")
    parser.add_argument("--date",    type=str, default=None, help="As-of date (YYYY-MM-DD)")
    parser.add_argument("--use-llm", action="store_true", help="Enable Claude API narrative")
    parser.add_argument("--no-pdf",  action="store_true", help="Skip PDF generation")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML generation")
    args = parser.parse_args()

    run(
        date=args.date,
        use_llm=args.use_llm,
        gen_pdf=not args.no_pdf,
        gen_html=not args.no_html,
    )
