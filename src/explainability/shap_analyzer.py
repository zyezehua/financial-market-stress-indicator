"""SHAP-based feature attribution for ensemble stress predictions."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def compute_shap_values(
    lgbm_model,
    X: pd.DataFrame,
    max_display: int = 20,
) -> pd.DataFrame:
    """
    Compute SHAP values using TreeExplainer on the LightGBM booster.

    Returns
    -------
    DataFrame of shape (n_samples, n_features) with SHAP values.
    """
    try:
        explainer = shap.TreeExplainer(lgbm_model.booster_)
        shap_values = explainer.shap_values(X)
        # For multi-output (classifier), shap_values is a list; take mean abs
        if isinstance(shap_values, list):
            shap_arr = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_arr = shap_values
        return pd.DataFrame(shap_arr, index=X.index, columns=X.columns)
    except Exception as exc:
        logger.warning("TreeExplainer failed (%s). Falling back to feature importance.", exc)
        fi = lgbm_model.feature_importances_
        fi_norm = fi / fi.sum() if fi.sum() > 0 else fi
        row = pd.Series(fi_norm, index=X.columns)
        return pd.DataFrame([row.values] * len(X), index=X.index, columns=X.columns)


def top_factors(
    shap_df: pd.DataFrame,
    row_index,
    n: int = 5,
) -> pd.Series:
    """
    Return the top-N SHAP contributors for a single prediction row.

    Returns pd.Series: feature_name → shap_value, sorted by absolute value.
    """
    row = shap_df.loc[row_index]
    return row.abs().nlargest(n)


def shap_summary(
    shap_df: pd.DataFrame,
    feature_groups: Dict[str, List[str]] = None,
    n_top: int = 5,
) -> dict:
    """
    Aggregate SHAP analysis across all rows.

    Returns
    -------
    dict with:
        top_global_features : top N features by mean |SHAP|
        group_importance    : dimension-level importance (if feature_groups provided)
    """
    mean_abs_shap = shap_df.abs().mean()
    top_global = mean_abs_shap.nlargest(n_top).to_dict()

    group_importance = {}
    if feature_groups:
        for group, cols in feature_groups.items():
            group_cols = [c for c in cols if c in shap_df.columns]
            if group_cols:
                group_importance[group] = float(shap_df[group_cols].abs().mean().mean())
        total = sum(group_importance.values()) or 1.0
        group_importance = {k: round(v / total * 100, 1) for k, v in group_importance.items()}

    return {
        "top_global_features": top_global,
        "group_importance":    group_importance,
    }
