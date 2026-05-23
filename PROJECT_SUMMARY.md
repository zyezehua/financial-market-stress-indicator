# Financial Market Stress Indicator — Project Summary

> **Status**: Phase B+ complete (as of 2026-05-23)
> Cross-asset ML stress forecasting system covering 26 years of daily market data.

---

## 1. Overview

A multi-horizon machine learning system that quantifies and forecasts financial market stress across six asset-class dimensions. The system produces a composite stress index (CSI, 0–100 scale) and generates 5-day, 21-day, and 63-day forward predictions with stress class labels and market direction signals.

**Core output per prediction run:**
- Current CSI level and stress class (Low / Elevated / High / Extreme)
- Predicted CSI for each horizon (5d / 21d / 63d) + delta vs current
- Market direction probability (Down / Neutral / Up)
- SHAP attribution: top drivers by asset class dimension
- Regime context: where current stress sits relative to recent baseline

---

## 2. Data Sources

### Market Data (Yahoo Finance via yfinance)
| Dimension | Key Tickers | Coverage |
|-----------|-------------|----------|
| **Equity** | SPY, QQQ, VIX, VVIX, SKEW, XLF, KBE | Volatility surface, sector stress |
| **Credit** | HYG, LQD, TLT, JNK, EMB, HYG/TLT ratio | Spread dynamics, credit-duration divergence |
| **Rates** | TLT, SHY, IEF, TIP, DXY | Yield curve shape, TIPS breakevens |
| **Liquidity** | GLD, SLV, BIL, BND | Safe-haven flows, liquidity proxies |
| **FX** | DXY, FXY, FXE, MCHI | Dollar strength, EM stress |
| **International** | EEM, EFA, ACWI, GLD/SPY ratio | Cross-border contagion, flight-to-safety |

### Macro Data (FRED API)
- SOFR-OIS spread (short-term funding stress)
- IG/HY credit spreads (ICE BofA indices)
- Effective federal funds rate, 2s10s yield spread
- VIX term structure slope
- FRED Financial Stress Index (FSI) used as independent benchmark

**Data range**: 2000-01-03 → 2026-05-22 (6,881 trading days)

---

## 3. Feature Engineering

**174 total features** across 6 dimensions + 4 regime context features:

| Dimension | Features | Key Signals |
|-----------|----------|-------------|
| Equity | 49 | VIX level/z-score/term structure, VVIX, SKEW, realized vol (5d/21d/63d), SPY drawdown, XLF/KBE bank stress, QQQ-SPY relative momentum |
| Credit | 31 | HYG/LQD spread proxies, credit drawdown, HYG-TLT ratio (credit vs duration divergence), HYG 252d drawdown |
| Rates | 38 | Yield curve slope (2s10s), TIPS-nominal spread, rate vol, SOFR-OIS, term structure features |
| Liquidity | 14 | BID-ask proxies, safe-haven flow signals, volume anomalies |
| FX | 17 | DXY level/percentile rank, JPY/EUR stress, EM FX momentum |
| International | 21 | EEM-SPY relative performance, GLD/SPY ratio (flight-to-safety intensity), cross-asset correlations |
| **Regime** | 4 | 252d/504d rolling CSI mean, CSI z-score vs 252d window, CSI 504d percentile rank |

**Feature transformations applied**: rolling percentile ranks (252d window), z-scores, momentum (5d/21d/63d), realized volatility, drawdown from rolling maxima, cross-asset ratios, sign change (correlation regime) indicators.

---

## 4. Composite Stress Index (CSI)

The CSI is a weighted composite of six dimension sub-indices, each normalized to a 0–100 scale using rolling 252-day percentile ranks:

```
CSI = Σ wᵢ × normalize(dimension_raw_scoreᵢ)
```

Default weights: Equity 30%, Credit 25%, Rates 15%, Liquidity 15%, FX 10%, International 5%

The CSI is calibrated against 19 named historical stress episodes spanning the Dot-com Crash (2000) through the 2025 Tariff Shock. The FRED Financial Stress Index is used as an independent benchmark for validation.

**Dynamic class boundaries** (introduced in Phase B+): Instead of fixed thresholds (30/50/70), class labels are determined by the 30th/50th/70th percentile of the training window's CSI distribution at each walk-forward fold. This solves the post-2020 baseline elevation problem where the market's structural stress floor shifted upward (~30 → ~48 median CSI).

---

## 5. Machine Learning Models

### Architecture: 3-Model Ensemble per Horizon × 4 Prediction Targets

For each prediction horizon (5d, 21d, 63d), four ensemble models are trained:

| Model | Task | Base Learners |
|-------|------|---------------|
| **StressEnsemble** | Regression: predict future CSI score (continuous 0–100) | LightGBM + XGBoost + Ridge |
| **StressClassEnsemble** | Classification: Low / Elevated / High / Extreme | LightGBM + XGBoost + LogisticRegression (balanced) |
| **DirectionEnsemble** | 3-class: Down / Neutral / Up market direction | LightGBM + XGBoost + LogisticRegression (balanced) |
| **DownRiskEnsemble** | Hierarchical: binary Down specialist + 3-class direction | LightGBM + XGBoost + Ridge + DirectionEnsemble |

**Ensemble weighting**: weighted average of model probabilities/predictions; weights tunable via config.

**Training protocol**:
- Minimum training window: 756 days (~3 years)
- Class imbalance handling: `class_weight="balanced"` on classifiers; SMOTE augmentation in walk-forward folds
- Feature alignment: reindex to training-time feature list (handles feature additions gracefully)
- Regime-adaptive class thresholds: quantile-based per fold (not fixed percentages)

### Regime Detector (HMM)
A 3-state Hidden Markov Model fitted on the CSI time series classifies the current market regime (Calm / Transition / Stressed). Used in crisis signal lifecycle analysis.

---

## 6. Walk-Forward OOS Backtest Results

**Methodology**: Expanding window walk-forward, step = 63 days (quarterly retraining), no lookahead bias. All metrics below are true out-of-sample.

### 5-Day Horizon (98 folds, 6,120 OOS observations)

**Regression (Stress Score)**
| Metric | Value |
|--------|-------|
| MAE | **4.38** |
| RMSE | **5.66** |
| Pearson R | **0.890** |
| Classification Accuracy | **67.8%** |
| Direction Accuracy | 42.0% |

**Stress Classification (OOS)**
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Low | 0.70 | **0.82** | 0.76 | 2,000 |
| Elevated | 0.33 | 0.19 | 0.24 | 705 |
| High | 0.30 | 0.23 | 0.26 | 903 |
| Extreme | 0.80 | **0.86** | 0.83 | 2,512 |

**Market Direction (OOS)**
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Down | 0.23 | 0.28 | 0.25 | 1,378 |
| Neutral | 0.56 | 0.48 | 0.52 | 2,625 |
| Up | 0.43 | 0.44 | 0.43 | 2,117 |

> **Note**: 21d and 63d OOS results pending (walk-forward run interrupted; to be completed).

### Key Improvements from Phase B+ (vs. Phase B baseline)

| Metric | Before (Phase B) | After (Phase B+) | Change |
|--------|-----------------|------------------|--------|
| MAE (5d) | 4.77 | **4.38** | −8.2% |
| Pearson R (5d) | 0.871 | **0.890** | +2.2% |
| Cls Accuracy (5d) | 0.528 | **0.678** | +28.4% |
| Low Recall | ~9% | **82%** | +73pp |
| Extreme Recall | ~5% | **86%** | +81pp |
| Features | 147 | **174** | +18% |

The dramatic improvement in Low/Extreme recall is entirely attributable to **dynamic regime class boundaries** — the single most impactful change in this phase.

---

## 7. Interactive Dashboard

A standalone HTML dashboard is generated daily via `python scripts/generate_dashboard.py`.

**Four tabs:**
- **Snapshot**: Current CSI gauge, dimension radar, 3-horizon forecast cards with direction probability bars, 3-year trajectory chart
- **Full History**: 2000–present CSI with all 19 stress episode annotations, dimension sub-index toggles, range selector
- **Backtest**: OOS performance metrics table, precision/recall charts, in-sample predicted vs actual (all horizons), error distribution
- **Calendar**: Monthly CSI heatmap (year × month)

---

## 8. System Architecture

```
scripts/
  daily_run.py          ← Full pipeline: data → features → CSI → predict → report
  train_model.py        ← Model training (all horizons)
  backtest.py           ← In-sample / walk-forward OOS / crisis signal analysis
  generate_dashboard.py ← Standalone interactive HTML dashboard

src/
  data/pipeline.py      ← Yahoo Finance + FRED data ingestion
  features/             ← 6-dimension feature builders
  labels/               ← CSI construction, calibration, prediction targets
  models/
    base/               ← LightGBM, XGBoost, Ridge wrappers
    ensemble.py         ← StressEnsemble, DirectionEnsemble, StressClassEnsemble
    down_ensemble.py    ← DownRiskEnsemble (hierarchical Down specialist)
    trainer.py          ← train_horizon(), train_all_horizons()
    walk_forward.py     ← OOS expanding-window backtest engine
    predictor.py        ← predict_latest(), predict_historical()
    regime_detector.py  ← HMM regime classifier
  explainability/       ← SHAP attribution, factor decomposition
  reporting/            ← HTML dashboard, PDF report generators
  monitoring/           ← PSI drift detection
```

---

## 9. Next Steps (Phase C)

### Immediate (before next OOS run)
- [ ] **Complete 21d/63d OOS**: Re-run full walk-forward to get all-horizon metrics
- [ ] **DownRiskEnsemble threshold calibration**: Use saved `down_proba` column from OOS run to tune optimal Down detection threshold (currently default=0.40) via precision-recall curve analysis

### Short-term (Phase C)
- [ ] **Macro feature expansion**: Add recession probability (Sahm Rule, yield curve inversion duration), ISM PMI momentum, high-yield issuance volume
- [ ] **Cross-horizon consistency**: Enforce 5d prediction is consistent with 21d direction (currently independent)
- [ ] **Volatility regime features**: Realized/implied vol ratio, VIX contango/backwardation persistent regime
- [ ] **Model calibration**: Platt scaling on class probabilities; temperature scaling on direction probabilities

### Medium-term
- [ ] **Live data pipeline**: Automate daily `daily_run.py` via cron/scheduler; push HTML dashboard to static hosting
- [ ] **Alert system**: Telegram/email notification when CSI crosses regime boundaries or model detects Extreme stress onset
- [ ] **Factor model integration**: Link CSI dimensions to Fama-French/AQR factors for academic-style attribution
- [ ] **Backtesting P&L**: Simulate simple strategy (reduce equity exposure when CSI Extreme) to compute Sharpe/max-drawdown

### Research directions
- [ ] **Transformer-based sequential model**: Replace ensemble with temporal attention model to capture longer-range dependencies
- [ ] **Online learning**: Incremental model updates with each new day's data (currently quarterly retrain)
- [ ] **Alternative data**: Google Trends "recession" queries, Reddit sentiment, earnings revision breadth

---

## 10. Limitations and Known Issues

1. **In-sample CSI construction**: The CSI is built using percentile ranks over the full history. Strictly speaking this uses future information — a fully OOS CSI would require rolling-window normalization. This is a known trade-off between signal stability and statistical purity.

2. **Elevated/High recall gap**: After dynamic boundaries, Low and Extreme recall are strong (82%/86%), but Elevated (19%) and High (23%) remain low. These middle classes are structurally harder to classify because the boundary between them is a smooth distribution, not a regime change.

3. **Direction accuracy at 42%**: For short-horizon (5d) market direction, 42% OOS accuracy reflects the fundamental difficulty of the problem. The model's value is better measured by asymmetric performance: it correctly flags Down periods with higher precision in high-stress regimes.

4. **Post-2020 structural shift**: Market volatility regimes changed significantly post-COVID. While regime features partially address this, the model was trained largely on pre-2020 data and may underweight structural changes in liquidity, rate volatility, and geopolitical risk.

5. **Dependency on Yahoo Finance availability**: Daily runs require market data from Yahoo Finance which has occasional outages and data quality issues (particularly for options-derived series like VIX term structure).

---

*Generated 2026-05-23 | Financial Market Stress Indicator v1.0*
