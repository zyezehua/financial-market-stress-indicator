"""Load and normalize the St. Louis Fed Financial Stress Index (STLFSI4)."""

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

STLFSI_SERIES = "STLFSI4"


def load_fred_fsi(
    macro: pd.DataFrame,
    labels_dir: str = "data/labels",
    save: bool = True,
) -> pd.Series:
    """
    Extract the STLFSI4 series from the macro DataFrame and normalize to 0-100.

    The raw FSI is centered at 0 (positive = above-average stress).
    We rescale to 0-100 using rolling percentile rank so it's comparable to CSI.

    Returns
    -------
    pd.Series named 'fred_fsi_norm' with values in [0, 100]
    """
    if STLFSI_SERIES not in macro.columns:
        logger.warning(
            "STLFSI4 not found in macro DataFrame. "
            "Ensure FRED_API_KEY is set and data pipeline ran successfully."
        )
        return pd.Series(name="fred_fsi_norm", dtype=float)

    raw = macro[STLFSI_SERIES].copy()

    # STLFSI4 is weekly; already forward-filled to daily by FredFetcher
    # Normalize: rolling 3-year percentile rank → 0-100
    normalized = raw.rolling(756, min_periods=252).apply(
        lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]) * 100,
        raw=False,
    )
    normalized.name = "fred_fsi_norm"

    if save:
        os.makedirs(labels_dir, exist_ok=True)
        normalized.to_frame().to_parquet(os.path.join(labels_dir, "fred_fsi.parquet"))
        logger.info("Saved FRED FSI to %s/fred_fsi.parquet", labels_dir)

    return normalized


def load_fred_fsi_from_disk(labels_dir: str = "data/labels") -> pd.Series:
    path = os.path.join(labels_dir, "fred_fsi.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"FRED FSI not found at {path}. Run load_fred_fsi first.")
    return pd.read_parquet(path)["fred_fsi_norm"]
