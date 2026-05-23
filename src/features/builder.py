"""Master feature builder: assembles all six dimensions into a single feature matrix."""

import logging
import os

import pandas as pd
import yaml

from .credit import compute_credit_features
from .equity import compute_equity_features
from .fx import compute_fx_features
from .international import compute_international_features
from .liquidity import compute_liquidity_features
from .rates import compute_rates_features

logger = logging.getLogger(__name__)

DIMENSION_PREFIXES = {
    "equity": "eq_",
    "credit": "cr_",
    "rates": "rt_",
    "liquidity": "lq_",
    "fx": "fx_",
    "international": "intl_",
}


def build_features(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    processed_dir: str = "data/processed",
    save: bool = True,
) -> pd.DataFrame:
    """
    Run all six feature modules and concatenate results.

    Parameters
    ----------
    prices       : output of DataPipeline (Yahoo Adj Close)
    macro        : output of DataPipeline (FRED series)
    processed_dir: where to save the feature matrix
    save         : whether to persist to disk

    Returns
    -------
    feature_matrix : DataFrame, shape (T, ~120 features)
    """
    logger.info("Building feature matrix ...")

    eq_feat   = compute_equity_features(prices, macro)
    cr_feat   = compute_credit_features(prices, macro)
    rt_feat   = compute_rates_features(prices, macro)
    lq_feat   = compute_liquidity_features(prices, macro)
    fx_feat   = compute_fx_features(prices, macro)
    intl_feat = compute_international_features(prices, macro)

    feature_matrix = pd.concat(
        [eq_feat, cr_feat, rt_feat, lq_feat, fx_feat, intl_feat], axis=1
    )

    # Drop columns that are all-NaN
    before = feature_matrix.shape[1]
    feature_matrix = feature_matrix.dropna(axis=1, how="all")
    after = feature_matrix.shape[1]
    if before != after:
        logger.warning("Dropped %d all-NaN columns", before - after)

    # Final forward-fill + backward-fill for remaining gaps
    feature_matrix = feature_matrix.ffill().bfill()

    logger.info(
        "Feature matrix: %d rows × %d features",
        len(feature_matrix),
        feature_matrix.shape[1],
    )

    if save:
        path = os.path.join(processed_dir, "features.parquet")
        os.makedirs(processed_dir, exist_ok=True)
        feature_matrix.to_parquet(path)
        logger.info("Saved feature matrix to %s", path)

    return feature_matrix


def get_feature_groups(feature_matrix: pd.DataFrame) -> dict:
    """Return a dict mapping dimension name → list of column names."""
    groups = {}
    for dim, prefix in DIMENSION_PREFIXES.items():
        groups[dim] = [c for c in feature_matrix.columns if c.startswith(prefix)]
    return groups


def load_features(processed_dir: str = "data/processed") -> pd.DataFrame:
    path = os.path.join(processed_dir, "features.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature matrix not found at {path}. Run build_features first.")
    return pd.read_parquet(path)
