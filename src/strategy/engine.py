"""
src/strategy/engine.py

Core strategy logic for FMSI trading strategies.
Used by both scripts/strategy_backtest.py and the Streamlit app.

Strategies
──────────
  S1 — CSI Stress-Level Allocation
         Position size = f(current CSI class): Low→100%, Elevated→75%, High→25%, Extreme→0%
  S2 — CSI Trend Filter
         Exit to cash when CSI > threshold AND n-day momentum > 0 (stress rising)
  S3 — S1 + Model Direction Overlay
         S1 base × direction prediction modifier (requires trained model artifacts)
  S4 — Adaptive Percentile Threshold
         Exit when rolling 252d CSI percentile > exit_pctile AND CSI rising
  S5 — Trend Filter with Partial Re-entry
         Like S2 but re-enters at 75% when CSI starts falling from above threshold
"""

import logging
import warnings
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

TRADING_DAYS = 252
COST_PER_SIDE = 0.0005     # 5 bps one-way
RISK_FREE_RATE = 0.04      # annualized

# Walk-forward optimization defaults
WF_TRAIN_YEARS = 5
WF_TEST_YEARS  = 1

STRATEGY_META = {
    "S1": {
        "label": "S1: Stress Allocation",
        "desc":  "Position size from CSI stress class: Low→100%, Elevated→75%, High→25%, Extreme→0%",
    },
    "S2": {
        "label": "S2: Trend Filter",
        "desc":  "Exit to cash when CSI > threshold AND stress is rising (n-day momentum > 0)",
    },
    "S3": {
        "label": "S3: Stress + Direction",
        "desc":  "S1 base allocation scaled by model direction prediction (requires model artifacts)",
    },
    "S4": {
        "label": "S4: Percentile Adaptive",
        "desc":  "Adaptive exit threshold via rolling 252d CSI percentile rank",
    },
    "S5": {
        "label": "S5: Trend + Re-entry",
        "desc":  "Like S2 but partially re-enters (75%) when stress starts falling from peaks",
    },
}

DEFAULT_PARAMS = {
    "S1": {"Low": 1.00, "Elevated": 0.75, "High": 0.25, "Extreme": 0.00},
    "S2": {"threshold": 55.0, "momentum_window": 10},
    "S2_OPT": {"threshold": 55.0, "momentum_window": 5},
    "S3": {"down_scale": 0.50, "up_scale": 1.25,
           "down_prob_thresh": 0.40, "up_prob_thresh": 0.45},
    "S4": {"roll_window": 252, "exit_pctile": 0.75, "enter_pctile": 0.50,
           "momentum_window": 10, "low_pctile": 0.25, "high_alloc": 1.10},
    "S5": {"threshold": 55.0, "momentum_window": 10,
           "reentry_drop": 3.0, "reentry_window": 5,
           "exit_weight": 0.0, "reentry_weight": 0.75, "base_weight": 1.0},
}

PARAM_GRIDS = {
    "S1": [
        {"Low": l, "Elevated": e, "High": h, "Extreme": x}
        for l in [1.00, 1.10, 1.25]
        for e in [0.50, 0.60, 0.75]
        for h in [0.10, 0.15, 0.25, 0.35]
        for x in [0.00, 0.10]
    ],
    "S2": [
        {"threshold": t, "momentum_window": m}
        for t in [45, 55, 62, 70]
        for m in [3, 5, 10, 21]
    ],
    "S3": [
        {"down_scale": d, "up_scale": u, "down_prob_thresh": dp, "up_prob_thresh": up}
        for d  in [0.25, 0.50, 0.75]
        for u  in [1.00, 1.25]
        for dp in [0.35, 0.40, 0.50]
        for up in [0.40, 0.50]
    ],
    "S4": [
        {"roll_window": 252, "exit_pctile": ep, "enter_pctile": 0.50,
         "momentum_window": m, "low_pctile": 0.25, "high_alloc": h}
        for ep in [0.65, 0.75, 0.80]
        for m  in [5, 10, 21]
        for h  in [1.00, 1.10]
    ],
    "S5": [
        {"threshold": t, "momentum_window": m,
         "reentry_drop": rd, "reentry_window": rw,
         "exit_weight": 0.0, "reentry_weight": rweight, "base_weight": 1.0}
        for t       in [50, 55, 62]
        for m       in [5, 10]
        for rd      in [2.0, 3.0, 5.0]
        for rw      in [5, 10]
        for rweight in [0.50, 0.75]
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# SPY data
# ══════════════════════════════════════════════════════════════════════════════

def load_spy_returns(start: str, end: str) -> pd.Series:
    """Download SPY daily log returns via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed")

    spy   = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if spy.empty:
        raise ValueError("SPY download returned empty DataFrame")
    close = spy["Close"].squeeze()
    ret   = np.log(close / close.shift(1)).dropna()
    ret.name = "spy_ret"
    return ret


# ══════════════════════════════════════════════════════════════════════════════
# Signal → position weights
# ══════════════════════════════════════════════════════════════════════════════

def s1_weights(csi_sig: pd.DataFrame, params: dict = None) -> pd.Series:
    """
    Position weight from CSI stress class.
    t-1 signal → t trade (no lookahead).
    """
    p = params or DEFAULT_PARAMS["S1"]
    w = csi_sig["csi_class"].map(p).astype(float)
    w.name = "weight"
    return w.shift(1)


def s2_weights(csi_sig: pd.DataFrame, params: dict = None) -> pd.Series:
    """
    Exit to cash when CSI > threshold AND n-day momentum > 0.
    t-1 signal → t trade.
    """
    p   = params or DEFAULT_PARAMS["S2"]
    csi = csi_sig["csi_composite"]
    mom = csi - csi.shift(p["momentum_window"])
    wgt = pd.Series(
        np.where((csi > p["threshold"]) & (mom > 0), 0.0, 1.0),
        index=csi_sig.index,
        name="weight",
    )
    return wgt.shift(1)


def s3_weights(
    csi_sig: pd.DataFrame,
    dir_sig: pd.DataFrame,
    params: dict = None,
) -> pd.Series:
    """
    S1 base allocation × model direction modifier.
    Down signal → scale down; Up signal → scale up; Neutral → no change.
    t-1 signal → t trade.
    Falls back to S1 if dir_sig is empty.
    """
    base = s1_weights(csi_sig, DEFAULT_PARAMS["S1"])
    if dir_sig is None or dir_sig.empty:
        logger.warning("S3: no direction signal — falling back to S1.")
        return base

    p     = params or DEFAULT_PARAMS["S3"]
    da    = dir_sig.reindex(base.index)
    dp    = da["dir_down_prob"].ffill().shift(1)
    up    = da["dir_up_prob"].ffill().shift(1)
    is_dn = dp >= p["down_prob_thresh"]
    is_up = (up >= p["up_prob_thresh"]) & ~is_dn
    scale = np.where(is_dn, p["down_scale"], np.where(is_up, p["up_scale"], 1.0))
    wgt   = (base * scale).clip(0.0, 1.50)
    wgt.name = "weight"
    return wgt


def s4_weights(csi_sig: pd.DataFrame, params: dict = None) -> pd.Series:
    """
    Adaptive exit: use rolling 252d CSI percentile rank instead of fixed CSI level.
    Adapts to post-2020 structural elevation.
    t-1 signal → t trade.
    """
    p      = params or DEFAULT_PARAMS["S4"]
    csi    = csi_sig["csi_composite"]
    pctile = csi.rolling(p["roll_window"], min_periods=p["roll_window"] // 2).rank(pct=True)
    mom    = csi - csi.shift(p["momentum_window"])

    exit_mask = (pctile > p["exit_pctile"]) & (mom > 0)
    calm_mask = pctile < p["low_pctile"]

    wgt = pd.Series(1.0, index=csi_sig.index, name="weight")
    wgt = wgt.where(~exit_mask, 0.0)
    wgt = wgt.where(~calm_mask | exit_mask, p["high_alloc"])
    return wgt.shift(1)


def s5_weights(csi_sig: pd.DataFrame, params: dict = None) -> pd.Series:
    """
    Trend filter with partial re-entry.
    Exits to 0% when stress rising above threshold.
    Returns to 75% when stress starts falling from above threshold.
    t-1 signal → t trade.
    """
    p   = params or DEFAULT_PARAMS["S5"]
    csi = csi_sig["csi_composite"]
    mom = csi - csi.shift(p["momentum_window"])
    drp = csi - csi.shift(p["reentry_window"])

    was_above = csi.shift(p["reentry_window"]) > p["threshold"]
    recovering = (drp < -p["reentry_drop"]) & was_above

    wgt = pd.Series(
        np.where(
            recovering,
            p["reentry_weight"],
            np.where(
                (csi > p["threshold"]) & (mom > 0) & ~recovering,
                p["exit_weight"],
                p["base_weight"],
            ),
        ),
        index=csi_sig.index,
        name="weight",
    )
    return wgt.shift(1)


_WEIGHT_FNS = {
    "S1": s1_weights,
    "S2": s2_weights,
    "S4": s4_weights,
    "S5": s5_weights,
}


def build_weights(
    strategy: str,
    csi_sig: pd.DataFrame,
    params: dict = None,
    dir_sig: pd.DataFrame = None,
) -> pd.Series:
    """Dispatch to the right weight function. Handles S3 separately."""
    if strategy == "S3":
        return s3_weights(csi_sig, dir_sig, params)
    fn = _WEIGHT_FNS.get(strategy)
    if fn is None:
        raise ValueError(f"Unknown strategy: {strategy}")
    return fn(csi_sig, params)


# ══════════════════════════════════════════════════════════════════════════════
# Backtest engine
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(returns: pd.Series, weights: pd.Series) -> pd.DataFrame:
    """
    Simulate daily rebalanced strategy.

    Returns DataFrame: weight, trade, cost, strat_ret, cum_strat, bh_ret, cum_bh, drawdown
    """
    idx  = returns.index.intersection(weights.index)
    ret  = returns.loc[idx]
    wgt  = weights.loc[idx].ffill().fillna(0.0).clip(0.0, 1.5)

    delta     = wgt.diff().abs().fillna(0.0)
    cost      = delta * COST_PER_SIDE
    strat_ret = wgt * ret - cost
    bh_ret    = ret.copy()

    cum_strat = np.exp(strat_ret.cumsum())
    cum_bh    = np.exp(bh_ret.cumsum())

    peak = cum_strat.cummax()
    dd   = (cum_strat - peak) / peak

    return pd.DataFrame({
        "weight":    wgt,
        "trade":     delta,
        "cost":      cost,
        "strat_ret": strat_ret,
        "cum_strat": cum_strat,
        "bh_ret":    bh_ret,
        "cum_bh":    cum_bh,
        "drawdown":  dd,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Performance metrics
# ══════════════════════════════════════════════════════════════════════════════

METRIC_LABELS = {
    "cagr":         "CAGR",
    "sharpe":       "Sharpe",
    "sortino":      "Sortino",
    "calmar":       "Calmar",
    "max_drawdown": "Max DD",
    "vol":          "Volatility",
    "total_return": "Total Return",
    "alpha_cagr":   "Alpha (CAGR)",
    "info_ratio":   "Info Ratio",
    "win_rate":     "Win Rate",
    "pct_invested": "% Invested",
}

HIGHER_IS_BETTER = {
    "cagr": True, "sharpe": True, "sortino": True, "calmar": True,
    "max_drawdown": False, "vol": False, "total_return": True,
    "alpha_cagr": True, "info_ratio": True, "win_rate": True, "pct_invested": True,
}


def compute_metrics(bt: pd.DataFrame, name: str = "Strategy") -> dict:
    """Compute comprehensive performance metrics."""
    s  = bt["strat_ret"]
    bh = bt["bh_ret"]
    n  = len(s)
    n_years = n / TRADING_DAYS

    total_return = float(bt["cum_strat"].iloc[-1] - 1)
    cagr         = float(bt["cum_strat"].iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0.0
    vol          = float(s.std() * np.sqrt(TRADING_DAYS))
    max_dd       = float(bt["drawdown"].min())

    daily_rf = RISK_FREE_RATE / TRADING_DAYS
    excess   = s - daily_rf
    sharpe   = float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)) if excess.std() > 0 else 0.0
    downside = float(s[s < 0].std() * np.sqrt(TRADING_DAYS)) if len(s[s < 0]) > 1 else 1e-9
    sortino  = float((cagr - RISK_FREE_RATE) / downside)
    calmar   = float(cagr / abs(max_dd)) if max_dd != 0 else np.inf

    bh_total = float(bt["cum_bh"].iloc[-1] - 1)
    bh_cagr  = float(bt["cum_bh"].iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0.0
    bh_vol   = float(bh.std() * np.sqrt(TRADING_DAYS))
    bh_exc   = bh - daily_rf
    bh_sharpe = float(bh_exc.mean() / bh_exc.std() * np.sqrt(TRADING_DAYS)) if bh_exc.std() > 0 else 0.0
    bh_peak  = bt["cum_bh"].cummax()
    bh_dd    = float(((bt["cum_bh"] - bh_peak) / bh_peak).min())
    bh_calmar = float(bh_cagr / abs(bh_dd)) if bh_dd != 0 else np.inf

    exc_daily = s - bh
    ir        = float(exc_daily.mean() / exc_daily.std() * np.sqrt(TRADING_DAYS)) if exc_daily.std() > 0 else 0.0

    return {
        "name":          name,
        "n_days":        n,
        "n_years":       round(n_years, 1),
        "total_return":  round(total_return, 4),
        "cagr":          round(cagr, 4),
        "vol":           round(vol, 4),
        "sharpe":        round(sharpe, 3),
        "sortino":       round(sortino, 3),
        "calmar":        round(calmar, 3),
        "max_drawdown":  round(max_dd, 4),
        "win_rate":      round(float((s > 0).mean()), 4),
        "pct_invested":  round(float(bt["weight"].mean()), 4),
        "n_trades":      int((bt["trade"] > 0.01).sum()),
        "total_costs":   round(float(bt["cost"].sum()), 6),
        "bh_cagr":       round(bh_cagr, 4),
        "bh_vol":        round(bh_vol, 4),
        "bh_sharpe":     round(bh_sharpe, 3),
        "bh_calmar":     round(bh_calmar, 3),
        "bh_max_dd":     round(bh_dd, 4),
        "alpha_cagr":    round(cagr - bh_cagr, 4),
        "info_ratio":    round(ir, 3),
    }


def rank_strategies(
    metrics_list: list[dict],
    priority: list[str],
) -> list[dict]:
    """
    Compute composite rank score based on user-defined metric priority.
    Priority is a list of metric keys in descending importance.
    Returns metrics_list with added 'composite_score' and 'rank' fields.
    """
    weights_map = {m: len(priority) - i for i, m in enumerate(priority)}
    strategies  = [m for m in metrics_list if m["name"] != "B&H SPY"]

    for key in priority:
        hi_better = HIGHER_IS_BETTER.get(key, True)
        vals      = np.array([s.get(key, 0.0) for s in strategies], dtype=float)
        finite    = vals[np.isfinite(vals)]
        if len(finite) < 2:
            for s in strategies:
                s[f"_rank_{key}"] = 1.0
            continue
        std = finite.std() or 1.0
        z   = (vals - finite.mean()) / std
        if not hi_better:
            z = -z
        for s, zi in zip(strategies, z):
            s[f"_rank_{key}"] = float(zi)

    for s in strategies:
        score = sum(weights_map[k] * s.get(f"_rank_{k}", 0.0) for k in priority)
        s["composite_score"] = round(score, 3)

    strategies.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, s in enumerate(strategies, 1):
        s["rank"] = i

    # Benchmark has no rank
    bh = next((m for m in metrics_list if m["name"] == "B&H SPY"), None)
    if bh:
        bh["rank"] = None
        bh["composite_score"] = None

    return metrics_list


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward optimization
# ══════════════════════════════════════════════════════════════════════════════

def walkforward_optimize(
    returns: pd.Series,
    csi_sig: pd.DataFrame,
    strategy: str,
    primary_metric: str = "sharpe",
    train_years: int = WF_TRAIN_YEARS,
    test_years:  int = WF_TEST_YEARS,
    dir_sig: pd.DataFrame = None,
    progress_cb=None,
) -> dict:
    """
    Walk-forward parameter optimization.
    Finds best params (by median OOS primary_metric) across expanding folds.

    progress_cb : optional callable(fold_idx, total_folds) for progress reporting
    Returns: {"best_params": ..., "folds": [...], "median_oos": ...}
    """
    grid = PARAM_GRIDS.get(strategy, [])
    if not grid:
        return {"best_params": DEFAULT_PARAMS.get(strategy, {}), "folds": [], "median_oos": None}

    start_year = returns.index[0].year
    end_year   = returns.index[-1].year
    fold_years = list(range(start_year, end_year - train_years - test_years + 1, test_years))

    fold_results = []
    for fi, fs in enumerate(fold_years):
        if progress_cb:
            progress_cb(fi, len(fold_years))

        r_tr = returns.loc[f"{fs}-01-01": f"{fs + train_years}-12-31"]
        r_te = returns.loc[f"{fs + train_years + 1}-01-01": f"{fs + train_years + test_years}-12-31"]
        c_tr = csi_sig.loc[f"{fs}-01-01": f"{fs + train_years}-12-31"]
        c_te = csi_sig.loc[f"{fs + train_years + 1}-01-01": f"{fs + train_years + test_years}-12-31"]

        if len(r_tr) < 100 or len(r_te) < 20:
            continue

        best_val, best_par = -np.inf, grid[0]
        for params in grid:
            try:
                d_tr = dir_sig.reindex(c_tr.index) if dir_sig is not None else None
                w    = build_weights(strategy, c_tr, params, dir_sig=d_tr)
                bt   = run_backtest(r_tr, w)
                val  = compute_metrics(bt).get(primary_metric, -np.inf)
                if np.isfinite(val) and val > best_val:
                    best_val, best_par = val, params
            except Exception:
                continue

        try:
            d_te = dir_sig.reindex(c_te.index) if dir_sig is not None else None
            w_te = build_weights(strategy, c_te, best_par, dir_sig=d_te)
            bt_t = run_backtest(r_te, w_te)
            m_t  = compute_metrics(bt_t)
            fold_results.append({
                "fold":      fs,
                "params":    best_par,
                "train_val": round(best_val, 3),
                "oos_val":   round(m_t.get(primary_metric, np.nan), 3),
                "oos_cagr":  m_t["cagr"],
                "oos_dd":    m_t["max_drawdown"],
            })
        except Exception:
            pass

    if not fold_results:
        return {"best_params": DEFAULT_PARAMS.get(strategy, {}), "folds": [], "median_oos": None}

    # Best params = most frequent among folds
    from collections import Counter
    votes     = Counter(str(sorted(f["params"].items())) for f in fold_results)
    best_str  = votes.most_common(1)[0][0]
    best_idx  = [str(sorted(f["params"].items())) for f in fold_results].index(best_str)
    best_par  = fold_results[best_idx]["params"]
    oos_vals  = [f["oos_val"] for f in fold_results if np.isfinite(f["oos_val"])]
    med_oos   = float(np.median(oos_vals)) if oos_vals else None

    return {"best_params": best_par, "folds": fold_results, "median_oos": med_oos}


# ══════════════════════════════════════════════════════════════════════════════
# Run all strategies in one call
# ══════════════════════════════════════════════════════════════════════════════

def run_all_strategies(
    spy_returns: pd.Series,
    csi_sig: pd.DataFrame,
    strategies: list[str],
    priority: list[str],
    optimize: bool = True,
    dir_sig: pd.DataFrame = None,
    progress_cb=None,
) -> dict:
    """
    Run backtests (and optionally optimize) for all requested strategies.
    Returns {
        "results":      {name: bt_df},
        "metrics":      [metric_dict],
        "opt_params":   {strategy: params},
        "opt_folds":    {strategy: folds},
    }
    """
    primary = priority[0] if priority else "sharpe"
    bh_w    = pd.Series(1.0, index=spy_returns.index)
    bt_bh   = run_backtest(spy_returns, bh_w)

    results    = {"B&H SPY": bt_bh}
    opt_params = {}
    opt_folds  = {}
    total_ops  = len(strategies) * (2 if optimize else 1)
    op_idx     = [0]

    for sid in strategies:
        def _prog(fi, total, _sid=sid):
            if progress_cb:
                op_idx[0] += 1 / (total or 1)
                progress_cb(op_idx[0] / total_ops, f"Optimizing {_sid}… fold {fi+1}/{total}")

        if optimize:
            d_sig_use = dir_sig if sid == "S3" else None
            opt = walkforward_optimize(
                spy_returns, csi_sig, sid,
                primary_metric=primary,
                dir_sig=d_sig_use,
                progress_cb=_prog,
            )
            params = opt["best_params"]
            opt_params[sid] = params
            opt_folds[sid]  = opt["folds"]
        else:
            params = DEFAULT_PARAMS.get(sid, {})

        # Also run default version for comparison (only when optimizing)
        if optimize:
            default_w  = build_weights(sid, csi_sig, DEFAULT_PARAMS.get(sid, {}),
                                       dir_sig=dir_sig if sid == "S3" else None)
            bt_default = run_backtest(spy_returns, default_w)
            results[f"{STRATEGY_META[sid]['label']} (default)"] = bt_default

        opt_w  = build_weights(sid, csi_sig, params,
                               dir_sig=dir_sig if sid == "S3" else None)
        label  = STRATEGY_META[sid]["label"] + (" (opt)" if optimize else "")
        bt_opt = run_backtest(spy_returns, opt_w)
        results[label] = bt_opt

        if progress_cb:
            op_idx[0] = strategies.index(sid) + 1
            progress_cb(op_idx[0] / len(strategies), f"Done: {sid}")

    metrics = [compute_metrics(bt, name) for name, bt in results.items()]
    metrics = rank_strategies(metrics, priority)

    return {
        "results":    results,
        "metrics":    metrics,
        "opt_params": opt_params,
        "opt_folds":  opt_folds,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Sub-period analysis
# ══════════════════════════════════════════════════════════════════════════════

STRESS_EPISODES_BACKTEST = {
    "GFC 2008-09":       ("2008-09-01", "2009-03-31"),
    "Euro Crisis 2011":  ("2011-07-01", "2011-10-31"),
    "China / Oil 2015":  ("2015-08-01", "2016-02-29"),
    "COVID Crash 2020":  ("2020-02-20", "2020-04-30"),
    "Bear Market 2022":  ("2022-01-01", "2022-12-31"),
    "SVB / Rates 2023":  ("2023-03-01", "2023-10-31"),
    "Tariff Shock 2025": ("2025-04-02", "2025-05-12"),
    "Bull 2013-2014":    ("2013-01-01", "2014-12-31"),
    "Bull 2017":         ("2017-01-01", "2017-12-31"),
    "Bull 2019":         ("2019-01-01", "2019-12-31"),
}


def subperiod_returns(results: dict) -> pd.DataFrame:
    """Return DataFrame of sub-period performance (total return, not CAGR for short periods)."""
    rows = []
    for period, (ps, pe) in STRESS_EPISODES_BACKTEST.items():
        row = {"Period": period}
        for name, bt in results.items():
            try:
                sub = bt.loc[ps:pe]
                if len(sub) < 5:
                    row[name] = np.nan
                    continue
                sub_ret = sub["cum_strat"].iloc[-1] / sub["cum_strat"].iloc[0] - 1
                ny      = len(sub) / TRADING_DAYS
                # Show total return for periods < 6 months; CAGR otherwise
                if ny < 0.5:
                    row[name] = sub_ret
                else:
                    row[name] = (1 + sub_ret) ** (1 / ny) - 1
            except Exception:
                row[name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("Period")
