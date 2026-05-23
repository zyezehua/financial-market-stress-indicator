"""
Walk-forward out-of-sample backtest.

At each step the model is trained exclusively on data prior to the cutoff date,
then evaluated on the following `step` trading days — data the model has never
seen.  This is the only correct way to measure true generalization.

Key design choices
──────────────────
- Min training window : 756 trading days (~3 years), same as trainer.py
- Step size           : 63 trading days (~1 quarter) by default
- Retrains the *full ensemble* (LGBM + XGB + Ridge) at every step
- Direction model is trained and evaluated only when direction targets exist
- Progress is logged so long runs remain observable
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False

from .base.lgbm_model import LGBMStressModel, LGBMStressClassModel, LGBMDirectionModel
from .base.ridge_model import RidgeStressModel, RidgeStressClassModel, RidgeDirectionModel
from .base.xgb_model import XGBStressModel, XGBStressClassModel, XGBDirectionModel
from .base.down_risk_model import LGBMDownRiskModel, XGBDownRiskModel, RidgeDownRiskModel
from .ensemble import StressEnsemble, StressClassEnsemble, DirectionEnsemble
from .down_ensemble import DownRiskEnsemble

logger = logging.getLogger(__name__)

MIN_TRAIN = 756   # trading days
DEFAULT_STEP = 63  # retraining frequency


_SMOTE_MIN_CLASS = 3   # skip SMOTE when any class has fewer than this many samples


def _smote_safe(
    X: pd.DataFrame,
    y: pd.Series,
    k_neighbors: int = 3,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE only when all classes have ≥ _SMOTE_MIN_CLASS samples.
    Falls back to the original data on any failure or sparse-class window.
    """
    if not _SMOTE_AVAILABLE:
        return X, y
    counts = y.value_counts()
    if len(counts) < 2 or int(counts.min()) < _SMOTE_MIN_CLASS:
        return X, y
    k = min(k_neighbors, int(counts.min()) - 1)
    try:
        sm = SMOTE(k_neighbors=k, random_state=42)
        X_r, y_r = sm.fit_resample(X.values, y.values)
        return pd.DataFrame(X_r, columns=X.columns), pd.Series(y_r)
    except Exception:
        return X, y


def _classify(score: float) -> str:
    if score <= 30:
        return "Low"
    if score <= 50:
        return "Elevated"
    if score <= 70:
        return "High"
    return "Extreme"


def _classify_dynamic(score: float, q30: float, q50: float, q70: float) -> str:
    """Regime-adaptive classification using training-window quantiles."""
    if pd.isna(score):
        return None
    if score <= q30:
        return "Low"
    if score <= q50:
        return "Elevated"
    if score <= q70:
        return "High"
    return "Extreme"


def _direction_label(v: int) -> str:
    return {-1: "Down", 0: "Neutral", 1: "Up"}.get(int(v), "Neutral")


def _drop_missing(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    mask = y.notna() & X.notna().all(axis=1)
    return X[mask], y[mask]


def run_walk_forward(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    csi: pd.DataFrame,
    horizon: int,
    min_train: int = MIN_TRAIN,
    step: int = DEFAULT_STEP,
    ensemble_weights: Optional[dict] = None,
    start_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run walk-forward OOS backtest for a single horizon.

    Parameters
    ----------
    features         : full feature matrix (T × F)
    targets          : DataFrame with columns stress_fwd_{h}d, market_dir_{h}d
    csi              : CSI DataFrame containing csi_composite column
    horizon          : prediction horizon in days (5, 21, or 63)
    min_train        : minimum number of training samples before first prediction
    step             : number of trading days between model retrainings
    ensemble_weights : optional dict of model weights for StressEnsemble
    start_date       : earliest prediction date (overrides min_train if later)

    Returns
    -------
    Dict with keys:
        "stress"    : DataFrame(stress_score_pred, stress_class_pred, stress_actual,
                                stress_class_actual) indexed by date
        "direction" : DataFrame(direction_pred, direction_actual) indexed by date,
                      or empty DataFrame if direction targets are unavailable
    """
    stress_col = f"stress_fwd_{horizon}d"
    dir_col    = f"market_dir_{horizon}d"

    if stress_col not in targets.columns:
        raise ValueError(f"Column '{stress_col}' not found in targets.")

    X_all, y_stress_all = _drop_missing(features, targets[stress_col])

    has_direction = dir_col in targets.columns
    if has_direction:
        X_dir_all, y_dir_all = _drop_missing(features, targets[dir_col])

    n = len(X_all)
    if n < min_train + step:
        raise ValueError(
            f"Not enough data for walk-forward: {n} samples, need >{min_train + step}."
        )

    # Determine the first test index
    first_test_idx = min_train
    if start_date:
        sd = pd.Timestamp(start_date)
        start_positional = int(X_all.index.searchsorted(sd))
        first_test_idx = max(first_test_idx, start_positional)

    stress_records   = []
    direction_records = []
    n_folds = 0

    for cutoff_idx in range(first_test_idx, n, step):
        X_train = X_all.iloc[:cutoff_idx]
        y_train = y_stress_all.iloc[:cutoff_idx]

        test_end = min(cutoff_idx + step, n)
        X_test   = X_all.iloc[cutoff_idx:test_end]

        if len(X_test) == 0:
            break

        # ── Train stress ensemble ───────────────────────────────────────────
        lgbm_s  = LGBMStressModel().fit(X_train, y_train)
        xgb_s   = XGBStressModel().fit(X_train, y_train)
        ridge_s = RidgeStressModel().fit(X_train, y_train)
        ens_s   = StressEnsemble(lgbm_s, xgb_s, ridge_s, ensemble_weights)

        # Dynamic class boundaries from training distribution (regime-adaptive)
        q30, q50, q70 = [float(np.nanpercentile(y_train.values, p)) for p in [30, 50, 70]]

        # Stress-class ensemble — SMOTE-augmented to recover Low/Extreme recall
        y_train_cls = y_train.apply(lambda v: _classify_dynamic(v, q30, q50, q70))
        # Ensure all 4 classes are present so XGB/LGBM classifiers don't fail on
        # early expanding windows where Low or Extreme haven't appeared yet.
        _ALL_STRESS = ["Low", "Elevated", "High", "Extreme"]
        _missing_cls = [c for c in _ALL_STRESS if c not in y_train_cls.values]
        if _missing_cls:
            _pad_X = pd.concat([X_train.iloc[[0]]] * len(_missing_cls), ignore_index=True)
            _pad_y = pd.Series(_missing_cls)
            _X_cls_in = pd.concat([X_train, _pad_X], ignore_index=True)
            _y_cls_in = pd.concat([y_train_cls.reset_index(drop=True), _pad_y], ignore_index=True)
        else:
            _X_cls_in, _y_cls_in = X_train, y_train_cls
        X_cls_sm, y_cls_sm = _smote_safe(_X_cls_in, _y_cls_in)
        try:
            lgbm_sc  = LGBMStressClassModel().fit(X_cls_sm, y_cls_sm)
            xgb_sc   = XGBStressClassModel().fit(X_cls_sm, y_cls_sm)
            ridge_sc = RidgeStressClassModel().fit(X_cls_sm, y_cls_sm)
            ens_sc   = StressClassEnsemble(lgbm_sc, xgb_sc, ridge_sc, ensemble_weights)
            class_preds = ens_sc.predict(X_test)
        except Exception as exc:
            logger.warning("StressClassEnsemble skipped (fold %d): %s", n_folds + 1, exc)
            class_preds = np.array([_classify(p) for p in ens_s.predict(X_test)])

        preds_s   = ens_s.predict(X_test)
        actuals_s = csi["csi_composite"].reindex(X_test.index).values

        for date, pred, cls_pred, actual in zip(X_test.index, preds_s, class_preds, actuals_s):
            stress_records.append({
                "date":               date,
                "stress_score_pred":  round(float(pred), 2),
                "stress_class_pred":  str(cls_pred),
                "stress_actual":      float(actual) if not np.isnan(actual) else np.nan,
                "stress_class_actual": _classify_dynamic(actual, q30, q50, q70) if not np.isnan(actual) else None,
            })

        # ── Train direction ensemble ────────────────────────────────────────
        if has_direction:
            # Align direction training data to the same cutoff
            mask_train = y_dir_all.index.isin(X_train.index) & y_dir_all.notna()
            X_d_train  = X_dir_all[mask_train]
            y_d_train  = y_dir_all[mask_train]

            mask_test  = y_dir_all.index.isin(X_test.index) & y_dir_all.notna()
            X_d_test   = X_dir_all[mask_test]
            y_d_test   = y_dir_all[mask_test]

            if len(X_d_train) >= min_train // 2 and len(X_d_test) > 0:
                # Ensure all 3 direction classes present before fitting
                _ALL_DIR = [-1, 0, 1]
                _missing_dir = [c for c in _ALL_DIR if c not in y_d_train.values]
                if _missing_dir:
                    _pad_dX = pd.concat([X_d_train.iloc[[0]]] * len(_missing_dir), ignore_index=True)
                    _pad_dy = pd.Series(_missing_dir)
                    X_d_train = pd.concat([X_d_train.reset_index(drop=True), _pad_dX], ignore_index=True)
                    y_d_train = pd.concat([y_d_train.reset_index(drop=True), _pad_dy], ignore_index=True)
                # SMOTE-augment direction data to improve Down recall
                X_d_sm, y_d_sm = _smote_safe(X_d_train, y_d_train)

                lgbm_d  = LGBMDirectionModel().fit(X_d_sm, y_d_sm)
                xgb_d   = XGBDirectionModel().fit(X_d_sm, y_d_sm)
                ridge_d = RidgeDirectionModel().fit(X_d_sm, y_d_sm)
                dir_ens = DirectionEnsemble(lgbm_d, xgb_d, ridge_d, ensemble_weights)

                # Down-risk specialists — may fail when no Down labels in window
                try:
                    lgbm_dr  = LGBMDownRiskModel().fit(X_d_sm, y_d_sm)
                    xgb_dr   = XGBDownRiskModel().fit(X_d_sm, y_d_sm)
                    ridge_dr = RidgeDownRiskModel().fit(X_d_sm, y_d_sm)
                    ens_d    = DownRiskEnsemble(lgbm_dr, xgb_dr, ridge_dr, dir_ens)
                except Exception as exc:
                    logger.warning("DownRiskEnsemble skipped (fold %d): %s", n_folds + 1, exc)
                    ens_d = dir_ens

                preds_d = ens_d.predict(X_d_test)
                if hasattr(ens_d, "predict_down_proba"):
                    down_probas = ens_d.predict_down_proba(X_d_test)
                else:
                    down_probas = ens_d.predict_proba(X_d_test)[:, 0]
                for date, pred, actual, dp in zip(
                    X_d_test.index, preds_d, y_d_test.values, down_probas
                ):
                    direction_records.append({
                        "date":             date,
                        "direction_pred":   _direction_label(pred),
                        "direction_actual": _direction_label(actual),
                        "down_proba":       round(float(dp), 4),
                    })

        n_folds += 1
        pct = 100.0 * cutoff_idx / n
        logger.info(
            "Walk-forward h=%dd | fold %d | cutoff=%s | test_rows=%d (%.0f%% of data)",
            horizon, n_folds, X_all.index[cutoff_idx - 1].date(), len(X_test), pct,
        )

    logger.info(
        "Walk-forward complete: horizon=%dd, folds=%d, stress_rows=%d, dir_rows=%d",
        horizon, n_folds, len(stress_records), len(direction_records),
    )

    stress_df = (
        pd.DataFrame(stress_records).set_index("date")
        if stress_records else pd.DataFrame()
    )
    dir_df = (
        pd.DataFrame(direction_records).set_index("date")
        if direction_records else pd.DataFrame()
    )
    return {"stress": stress_df, "direction": dir_df}


def compute_oos_metrics(
    stress_df: pd.DataFrame,
    direction_df: pd.DataFrame,
    horizon: int,
) -> dict:
    """
    Compute regression + classification metrics on OOS walk-forward predictions.

    Returns a dict with scalar metrics and full classification_report strings.
    """
    metrics: dict = {"horizon": horizon}

    # ── Stress regression metrics ───────────────────────────────────────────
    df = stress_df.dropna(subset=["stress_score_pred", "stress_actual"])
    if not df.empty:
        err = df["stress_score_pred"] - df["stress_actual"]
        metrics["n_stress_obs"] = len(df)
        metrics["stress_mae"]   = round(float(err.abs().mean()), 2)
        metrics["stress_rmse"]  = round(float(np.sqrt((err ** 2).mean())), 2)
        metrics["stress_pearson"] = round(
            float(df["stress_score_pred"].corr(df["stress_actual"])), 3
        )

    # ── Stress classification metrics ───────────────────────────────────────
    cls_df = stress_df.dropna(subset=["stress_class_pred", "stress_class_actual"])
    if not cls_df.empty:
        order = ["Low", "Elevated", "High", "Extreme"]
        report = classification_report(
            cls_df["stress_class_actual"],
            cls_df["stress_class_pred"],
            labels=[l for l in order if l in cls_df["stress_class_actual"].values],
            zero_division=0,
        )
        metrics["stress_class_report"] = report
        metrics["stress_class_accuracy"] = round(
            float((cls_df["stress_class_pred"] == cls_df["stress_class_actual"]).mean()), 3
        )

    # ── Direction classification metrics ────────────────────────────────────
    if direction_df is not None and not direction_df.empty:
        ddf = direction_df.dropna()
        if not ddf.empty:
            report_d = classification_report(
                ddf["direction_actual"],
                ddf["direction_pred"],
                labels=["Down", "Neutral", "Up"],
                zero_division=0,
            )
            metrics["direction_class_report"] = report_d
            metrics["direction_accuracy"] = round(
                float((ddf["direction_pred"] == ddf["direction_actual"]).mean()), 3
            )
            metrics["n_direction_obs"] = len(ddf)

    return metrics


def print_oos_summary(all_metrics: List[dict]) -> None:
    """Pretty-print a summary table plus full classification reports."""
    print("\n" + "=" * 70)
    print("WALK-FORWARD OUT-OF-SAMPLE BACKTEST RESULTS")
    print("=" * 70)

    header = f"{'Horizon':>8} {'N':>6} {'MAE':>6} {'RMSE':>7} {'PearsonR':>9} {'Cls Acc':>8} {'Dir Acc':>8}"
    print(header)
    print("-" * 70)
    for m in all_metrics:
        print(
            f"{str(m['horizon'])+'d':>8}"
            f"{m.get('n_stress_obs', '—'):>6}"
            f"{m.get('stress_mae', '—'):>7}"
            f"{m.get('stress_rmse', '—'):>8}"
            f"{m.get('stress_pearson', '—'):>10}"
            f"{m.get('stress_class_accuracy', '—'):>9}"
            f"{m.get('direction_accuracy', 'N/A'):>9}"
        )
    print("=" * 70)

    for m in all_metrics:
        h = m["horizon"]
        if "stress_class_report" in m:
            print(f"\n── Stress Classification Report — {h}d Horizon ──")
            print(m["stress_class_report"])
        if "direction_class_report" in m:
            print(f"── Market Direction Report — {h}d Horizon ──")
            print(m["direction_class_report"])
