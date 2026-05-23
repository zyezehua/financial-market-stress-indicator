# Methodology Note: Global Market Stress Indicator (v1.0)

> **For research purposes only. This system does not constitute investment advice.**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Data Sources](#2-data-sources)
3. [Feature Engineering](#3-feature-engineering)
4. [Label Construction: Composite Stress Index](#4-label-construction-composite-stress-index)
5. [Model Architecture](#5-model-architecture)
6. [Explainability Layer](#6-explainability-layer)
7. [Key Assumptions](#7-key-assumptions)
8. [Model Limitations](#8-model-limitations)
9. [Phase 2 & 3 Roadmap](#9-phase-2--3-roadmap)

---

## 1. Overview

The Global Market Stress Indicator (GMSI) is a quantitative system that estimates and forecasts the level of financial market stress across multiple asset classes and time horizons. It combines classical financial signal construction with gradient-boosted ensemble models, post-hoc SHAP explainability, and an optional large-language-model narrative layer.

**Primary goal:** Provide a daily, interpretable, multi-horizon assessment of systemic market risk — not merely describing what markets did, but anticipating where stress is heading.

**Prediction targets per run:**

| Horizon | Stress Score | Stress Direction | Market Direction |
|---|---|---|---|
| 5-day (short-term) | ✓ | ✓ | ✓ |
| 21-day (medium-term) | ✓ | ✓ | ✓ |
| 63-day (long-term) | ✓ | ✓ | — |

---

## 2. Data Sources

All data is sourced from free, publicly available providers. No proprietary or purchased data feeds are used.

### 2.1 Yahoo Finance (via `yfinance`)

| Asset Class | Tickers | Rationale |
|---|---|---|
| US Equities | SPY, QQQ, ^GSPC | Broad market level and momentum |
| Volatility | ^VIX, ^VIX3M, ^VVIX | Implied vol level, term structure, vol-of-vol |
| Sector ETFs | XLF, XLK, XLE, XLU | Cross-sector divergence as stress signal |
| Fixed Income | TLT, HYG, LQD, AGG | Rates and credit market conditions |
| FX | UUP, USDJPY=X, EURUSD=X, GBPUSD=X | USD strength, safe-haven demand |
| Commodities | GLD, USO, ^OVX | Gold as safe-haven; oil volatility |
| International | EEM, VGK, EWJ | EM contagion; European/Japan stress |

**Data range:** January 2000 – present (daily adjusted close prices).

### 2.2 FRED API (Federal Reserve Bank of St. Louis)

| Series | ID | Description |
|---|---|---|
| 2-Year Treasury Yield | DGS2 | Front-end rates |
| 10-Year Treasury Yield | DGS10 | Long-end rates and risk-free rate |
| 3-Month Treasury Yield | DGS3MO | T-bill rate for funding spread calculation |
| 10Y–2Y Spread | T10Y2Y | Yield curve shape |
| HY Option-Adjusted Spread | BAMLH0A0HYM2 | High-yield credit stress (ICE BofA) |
| IG Option-Adjusted Spread | BAMLC0A0CM | Investment-grade credit stress (ICE BofA) |
| SOFR | SOFR | Overnight funding rate (replaced LIBOR) |
| St. Louis FSI | STLFSI4 | External benchmark for CSI validation |

**Frequency handling:** FRED series with weekly or monthly frequency (e.g., STLFSI4) are forward-filled to business-day frequency for alignment with daily price data.

### 2.3 Caching Strategy

All raw data is cached to disk (`data/raw/`) with a 24-hour expiry. Daily updates fetch only data since the last processed date and append to existing processed files, avoiding full re-downloads.

---

## 3. Feature Engineering

Features are computed across six thematic dimensions. Each dimension captures a distinct facet of financial stress. All features use only information available at time *t* (no look-ahead).

### 3.1 Rolling Windows

For each raw signal, the following rolling statistics are computed:

| Window | Trading Days | Approximate Calendar |
|---|---|---|
| Short | 5d | 1 week |
| Medium | 21d | 1 month |
| Long-medium | 63d | 1 quarter |
| Long | 252d | 1 year |

Key feature types per signal: **level**, **z-score** (rolling mean/std), **momentum** (return), **percentile rank** (rolling empirical CDF), **drawdown from rolling high**.

### 3.2 Equity Dimension (~20 features)

| Feature | Construction | Interpretation |
|---|---|---|
| VIX level + z-score | Raw VIX and (VIX − μ₂₁) / σ₂₁ | Absolute and relative implied vol |
| VIX term structure ratio | VIX / VIX3M | >1 = backwardation = acute stress; <1 = contango = calm |
| VVIX (vol-of-vol) | Raw level + percentile rank | Tail risk demand; spikes before major dislocations |
| SPY realized vol | Rolling std × √252 over 5d, 21d, 63d | Actual market turbulence |
| SPY drawdown | (Price − 252d high) / 252d high | Peak-to-trough loss magnitude |
| SPY–TLT correlation | Rolling 21d Pearson correlation | Negative = normal; near-zero/positive = flight-to-quality breakdown |

*Note: VIX3M data only available from 2011. Pre-2011 values are proxied as VIX × 1.05 (approximating the typical term structure premium).*

### 3.3 Credit Dimension (~15 features)

| Feature | Construction | Interpretation |
|---|---|---|
| HY OAS | BAMLH0A0HYM2 level + z-score + momentum | Absolute credit risk premium |
| IG OAS | BAMLC0A0CM level + z-score | Investment-grade funding cost |
| HY − IG spread | HY OAS − IG OAS + percentile rank | Pure credit risk premium (removes duration) |
| HYG realized vol | Rolling std × √252 | Credit market turbulence |
| HYG/SPY relative return | Normalized relative performance | Credit-equity divergence (leads equity in stress) |
| HYG drawdown | (Price − 63d high) / 63d high | Credit market deterioration |

### 3.4 Rates Dimension (~12 features)

| Feature | Construction | Interpretation |
|---|---|---|
| 10Y yield level + velocity | DGS10 + 5d/21d change | Rate trend and pace of moves |
| 10Y absolute change (pct rank) | |Δ₅DGS10| percentile rank | Sudden rate moves signal stress |
| 2s10s spread | T10Y2Y or DGS10 − DGS2 | Yield curve shape |
| Inversion depth | max(−T10Y2Y, 0) + percentile rank | Recessions historically follow deep inversions |
| TLT realized vol | Rolling std × √252 | Proxy for MOVE Index (bond market volatility) |

*Note: The MOVE Index is not freely available via Yahoo Finance. TLT realized volatility serves as a proxy, capturing bond market uncertainty through price action rather than options pricing.*

### 3.5 Liquidity Dimension (~10 features)

| Feature | Construction | Interpretation |
|---|---|---|
| SOFR − 3M T-bill spread | SOFR − DGS3MO + percentile rank | Funding/repo stress (successor to TED spread) |
| Average cross-asset correlation | Mean pairwise 21d correlation: {SPY, TLT, HYG, GLD, EEM} | Contagion indicator: high correlation = systemic stress |
| SPY–TLT correlation | 21d rolling Pearson + percentile rank | Equity-bond relationship; breakdown signals flight-to-quality failure |
| HYG–TLT correlation | 21d rolling Pearson | Credit-rates decoupling |

*The average pairwise cross-asset correlation is the system's primary contagion indicator. During normal markets, asset classes are relatively uncorrelated. Stress episodes (GFC 2008, COVID March 2020) cause correlations to spike toward 1.0 as liquidity becomes the single dominant factor.*

### 3.6 FX Dimension (~12 features)

| Feature | Construction | Interpretation |
|---|---|---|
| USD index (UUP ETF) | Level + 21d return + realized vol + percentile rank | Dollar demand (safe-haven or carry unwind) |
| JPY strength | Inverted USDJPY + percentile rank | JPY appreciation = risk-off signal; strongest safe-haven indicator |
| JPY distance from 52w high | 1 − (1/USDJPY) / (1/USDJPY)₅₂ₘₐₓ | Measures how far JPY is from its stress peak |
| EUR/USD realized vol | Rolling std × √252 + percentile rank | European risk premium |
| FX vol composite | Mean of USDJPY vol + EURUSD vol | Aggregate currency market turbulence |

### 3.7 International Dimension (~12 features)

| Feature | Construction | Interpretation |
|---|---|---|
| EEM realized vol + drawdown | Rolling std × √252 + (Price − 63d high) / 63d high | EM financial conditions |
| EEM/SPY relative performance | EEM / SPY momentum (21d) + percentile rank | EM underperformance = global stress propagation |
| VGK realized vol + momentum | Rolling std × √252 + 21d return | European stress |
| EWJ realized vol + momentum | Rolling std × √252 + 21d return | Japan-specific stress |
| International vol composite | Mean of EEM, VGK, EWJ 21d realized vol | Aggregate global stress |
| Gold level (percentile rank) | GLD 252d percentile rank | Safe-haven demand as global stress proxy |

**Total feature count (Phase 1): ~104 features** after removing all-NaN columns.

---

## 4. Label Construction: Composite Stress Index

### 4.1 Design Philosophy

We construct our own Composite Stress Index (CSI) rather than directly predicting an external index (e.g., STLFSI4) for three reasons:

1. **Timeliness:** External FSI indices are published with lags (STLFSI4 is weekly). CSI is daily.
2. **Customizability:** We can weight dimensions to reflect the specific asset universe we care about.
3. **Transparency:** Every component and weight is fully observable and auditable.

The STLFSI4 serves as a **validation benchmark**, not the prediction target.

### 4.2 Dimension Score Construction

For each dimension *d*, raw signals are processed as follows:

```
Step 1: Compute rolling percentile rank for each signal sᵢ:
        PRᵢ(t) = Rank of sᵢ(t) within [t − W, t] × 100
        where W = 756 trading days (≈ 3 years)

Step 2: Average available signals within dimension:
        DimScore_d(t) = mean({PRᵢ(t) : sᵢ ∈ dimension d, PRᵢ(t) is non-null})

Step 3: Clip to [0, 100]
```

The 3-year rolling window is chosen to balance:
- **Long enough** to capture full stress cycles (including GFC tail)
- **Short enough** to adapt to structural shifts in market volatility regimes

### 4.3 Composite Score

```
CSI(t) = Σ_d [ w_d × DimScore_d(t) ]

Dimension weights:
  Equity:        20.0%
  Credit:        20.0%
  Rates:         17.5%
  Liquidity:     17.5%
  FX:            12.5%
  International: 12.5%
```

Equity and credit receive the highest weights as they are empirically the most responsive and leading indicators of systemic stress across historical episodes.

### 4.4 Classification Thresholds

| Score Range | Label | Description |
|---|---|---|
| 0 – 25 | **Low** | Below-average stress; risk appetite broadly intact |
| 26 – 50 | **Elevated** | Above-average tension; monitoring warranted |
| 51 – 75 | **High** | Significant stress; multiple risk premia elevated |
| 76 – 100 | **Extreme** | Systemic dislocation; comparable to GFC / COVID shock |

Thresholds are fixed (not data-driven) to preserve interpretability and consistency across time.

### 4.5 Validation Against STLFSI4

CSI is validated against the St. Louis Fed FSI using:
- **Pearson correlation** (linear co-movement)
- **Spearman rank correlation** (ordinal co-movement, more robust)
- **Episode recall:** Fraction of days during 7 known historical stress episodes where CSI > 50

**Phase 1 results (2000–2026):**

| Metric | Value |
|---|---|
| Pearson r (vs STLFSI4) | 0.49 |
| Spearman r (vs STLFSI4) | 0.49 |
| Episode recall (CSI > 50) | 69.9% |

Notable episode detection rates:
- GFC 2008–2009: 93% of days correctly flagged as High/Extreme
- COVID crash 2020: 100% of days correctly flagged
- 2022 Bear Market: 95% of days correctly flagged
- Dot-com crash 2000–2002: only 16% — see [Limitations §8.2](#82-csi-calibration-and-historical-data-coverage)

### 4.6 Prediction Targets

For each horizon *h* ∈ {5, 21, 63} trading days, the following targets are constructed:

| Target | Formula | Type |
|---|---|---|
| `stress_fwd_hd` | CSI(t+h) | Regression |
| `stress_delta_hd` | CSI(t+h) − CSI(t) | Regression |
| `stress_dir_hd` | sign(Δ) with ±5 pt deadband | 3-class |
| `market_ret_hd` | log(SPY(t+h) / SPY(t)) | Regression |
| `market_dir_hd` | sign(ret) with ±1% deadband | 3-class (Up/Neutral/Down) |

---

## 5. Model Architecture

### 5.1 Ensemble Design

Three base learners are combined via fixed weighted average:

| Model | Weight | Role |
|---|---|---|
| LightGBM | 50% | Primary learner; fast, handles nonlinearity and interactions |
| XGBoost | 30% | Secondary tree-based learner; redundancy and ensemble diversity |
| Ridge Regression | 20% | Linear baseline; anchors predictions in low-data regimes |

For the **stress score** (regression): outputs are averaged with the fixed weights above.

For **market direction** (3-class classification): soft voting — predicted probabilities from each model are weighted and averaged before argmax.

### 5.2 Training Procedure

**Expanding window cross-validation** is used for model evaluation:

```
Fold 1: Train [t₀, t₁]          → Validate [t₁, t₂]
Fold 2: Train [t₀, t₂]          → Validate [t₂, t₃]
...
Fold 5: Train [t₀, t₅]          → Validate [t₅, t₆]
Final:  Train [t₀, T]           → Production model
```

Minimum training size: **756 samples (~3 years)** to ensure sufficient stress episode coverage.

**No walk-forward leakage:** All features at time *t* use only data available at *t*. Forward-looking targets are constructed separately and never used in feature computation.

### 5.3 Model Hyperparameters

**LightGBM (stress):**
```
n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=6
min_child_samples=20, subsample=0.8, colsample_bytree=0.8
reg_alpha=0.1, reg_lambda=1.0
```

**XGBoost (stress):**
```
n_estimators=500, learning_rate=0.05, max_depth=5
min_child_weight=5, subsample=0.8, colsample_bytree=0.8
reg_alpha=0.1, reg_lambda=1.0, tree_method=hist
```

**Ridge regression:** α = 10.0 with StandardScaler normalization.

### 5.4 Phase 1 Performance Metrics

Cross-validated on the held-out final fold of training data:

| Horizon | Stress MAE (CV) | Notes |
|---|---|---|
| 5-day | **6.70 pts** | On 0–100 CSI scale |
| 21-day | **9.28 pts** | Natural increase with horizon |
| 63-day | **10.08 pts** | Structural trend captured more than level |

*Direction accuracy figures reported in training logs are in-sample and should not be interpreted as out-of-sample accuracy. A proper out-of-sample backtest is available via `scripts/backtest.py`.*

### 5.5 Retraining Schedule

Models are designed for **monthly retraining** using an expanding window (new data added, no data discarded). This adapts to:
- Structural breaks in volatility regimes
- New market instruments or regime changes
- Gradual drift in feature-target relationships

---

## 6. Explainability Layer

### 6.1 SHAP Attribution

SHAP (SHapley Additive exPlanations) values are computed using the `TreeExplainer` algorithm on the LightGBM booster, which provides exact SHAP values in polynomial time for tree-based models.

For each prediction, the top-*N* features by absolute SHAP value are extracted and mapped to human-readable labels (e.g., `eq_vix_pct_rank_252d` → "VIX (1Y percentile rank)") along with:
- **Current value** of the feature at time *t*
- **Percentile rank** of the current value within the trailing 252-day window

This combination answers three questions simultaneously:
- *What* factor is driving the prediction (SHAP attribution)?
- *Where* is that factor currently (absolute level)?
- *How unusual* is the current reading (historical context)?

### 6.2 Narrative Generation

The narrative layer has two modes:

**Rule-based (default, no API key required):** Generates structured text directly from prediction outputs and SHAP attribution data using templated prose. Fully deterministic and reproducible.

**LLM-enhanced (requires `ANTHROPIC_API_KEY`):** Passes the structured SHAP attribution and prediction data to Claude (claude-opus-4-7) via a structured expert-persona prompt. The model is instructed to synthesize — not merely recite — the quantitative outputs into a 3-paragraph financial briefing.

Both modes produce output in three sections: current regime assessment, short/medium-term outlook, and long-term structural view.

---

## 7. Key Assumptions

The model rests on several explicit assumptions. Violations of these assumptions may degrade prediction quality.

### 7.1 Stationarity of Stress Dynamics

**Assumption:** The statistical relationship between features (e.g., VIX percentile rank, HY spread z-score) and future stress levels is sufficiently stable across time to allow supervised learning.

**Basis:** Financial markets exhibit persistent behavioral patterns (risk-on/risk-off cycles, credit spread mean-reversion, volatility clustering) that have been documented across multiple decades.

**Risk:** Structural breaks — such as the 2022 end of the zero-interest-rate policy (ZIRP) era — can shift the feature-target relationship in ways that historical training data cannot capture.

### 7.2 Rolling Percentile Rank as Normalization

**Assumption:** Normalizing raw signals via their 3-year rolling empirical percentile rank removes level effects and makes features regime-agnostic.

**Basis:** A VIX of 30 meant something different in 2003 vs. 2020. Percentile rank captures "high relative to recent history" rather than "high in absolute terms."

**Risk:** During prolonged stress regimes (e.g., European Debt Crisis 2011–2012), the rolling window itself becomes contaminated with stress observations, compressing the upper end of the distribution and understating current stress.

### 7.3 Stale-Data Forward-Fill

**Assumption:** Forward-filling FRED series from their last published date is an adequate approximation for daily feature computation.

**Basis:** Many macro series (yield curve, credit spreads) exhibit mean-reversion and do not change discontinuously between publication dates.

**Risk:** During fast-moving markets (e.g., March 2020, September 2008), stale macro data introduces measurement error precisely when accuracy is most critical.

### 7.4 Fixed Dimension Weights

**Assumption:** The equal-ish weighting across CSI dimensions (20/20/17.5/17.5/12.5/12.5) reasonably reflects the relative information content of each asset class for systemic stress.

**Basis:** Initial weights reflect academic and practitioner consensus on which markets lead stress dynamics. Calibration against STLFSI4 was used to confirm the ordering is reasonable.

**Risk:** Optimal weights may shift across regimes. For example, FX stress was more important during the 2015 EM currency crisis, while rates stress dominates in 2022.

### 7.5 Ensemble Weight Stability

**Assumption:** Fixed ensemble weights (LightGBM 50%, XGBoost 30%, Ridge 20%) are appropriate throughout the model's lifecycle.

**Basis:** These weights were chosen a priori based on expected relative model quality and serve primarily to dampen individual model overfitting.

**Risk:** If market regime shifts cause LightGBM to degrade relative to Ridge (e.g., in very low-variance periods), fixed weights may not be optimal.

---

## 8. Model Limitations

### 8.1 Short-Term Prediction Is Noise-Dominated

The 5-day stress score prediction (MAE ≈ 6.7 on a 0–100 scale) operates in a regime where market noise is comparable to the signal. Week-ahead market moves are well-established to be largely unpredictable; the model captures directional tendencies rather than precise magnitudes.

**Implication:** The 5-day stress score should be interpreted as a *probability-weighted scenario* (e.g., "stress is more likely to increase than decrease over the next week") rather than a precise forecast.

### 8.2 CSI Calibration and Historical Data Coverage

The low episode recall (16%) for the dot-com crash (2000–2002) reflects a structural data limitation: several key features are unavailable or of poor quality in 2000–2003:
- **VIX3M** was not introduced until 2011
- **VVIX** was not introduced until 2012
- **SOFR** replaced LIBOR in 2023; funding stress data is sparse pre-2018
- **HY OAS data** from ICE BofA begins in 2004 in the current FRED series

As a result, the CSI has fewer signals with which to characterize stress during the early 2000s, causing systematic under-detection of that episode. The model trained on 2000–present inherits this data sparsity for pre-2005 periods.

### 8.3 No Options Market Microstructure

The model does not incorporate:
- **Options skew / put/call ratio** (indicative of tail risk hedging demand)
- **VIX futures term structure** (shape and slope carry additional information beyond VIX/VIX3M ratio)
- **Options open interest and gamma exposure** (dealer positioning can exacerbate or dampen moves)

These signals require either paid data subscriptions or bespoke CBOE data parsing.

### 8.4 No Macroeconomic Flow or Positioning Data

The model is entirely based on **price and spread data**. It does not incorporate:
- **Fund flow data** (ICI, EPFR)
- **Futures positioning** (CFTC Commitment of Traders)
- **Retail sentiment** (AAII survey, CNN Fear & Greed)
- **Credit impulse / economic activity** (PMI, ISM, GDP growth)

Macro-fundamental data would likely improve long-horizon (63-day) stress prediction, where price momentum becomes less informative.

### 8.5 Survivorship and Selection Bias

- **ETF history:** Many ETFs in the universe (HYG, LQD, EEM) launched after 2000. Early periods use alternative proxies or have fewer active features, creating implicit feature sparsity in early training windows.
- **Yield curve:** The model's yield curve features are derived from the US Treasury. Non-US sovereign stress (e.g., EU periphery spreads during 2011–2012) is captured only indirectly via VGK and EEM performance.

### 8.6 Regime Sensitivity of Direction Models

The 3-class market direction models (Up / Neutral / Down) showed high in-sample accuracy but this should not be mistaken for out-of-sample predictive power. The deadband (±1%) means that in low-volatility periods, most observations are "Neutral," and a naive model that always predicts Neutral would appear accurate. Proper evaluation requires checking **precision / recall per class** on held-out data, particularly for minority classes (Up and Down during strong trending environments).

### 8.7 Computational Dependencies

- **SHAP computation** via `TreeExplainer` scales approximately O(n × d × depth) where n = number of samples and d = number of features. For large prediction windows (e.g., 756-row rolling SHAP), computation can take 30–60 seconds.
- **No GPU acceleration:** All models use CPU inference. Scaling to intraday frequency or significantly expanding the feature set would require architectural changes.

### 8.8 Single-Country Regulatory and Structural Factors

The system is calibrated primarily on US asset markets. Events driven by regulatory interventions (e.g., SEC trading halts, Fed emergency facilities, FDIC actions) or fiscal policy announcements may cause stress to spike or drop discontinuously in ways that are not well-represented in historical feature distributions.

---

## 9. Phase 2 & 3 Roadmap

### Phase 2: Direction Prediction and Narrative Enhancement

**Target timeline:** ~2–3 months after Phase 1 validation

#### 9.1 Improved Direction Modeling

- Replace 3-class classification with **calibrated probability regression** (predicting log returns directly, then deriving direction from predicted distribution)
- Add **return magnitude head**: predict expected magnitude conditional on direction (e.g., "Up with expected +2.3%")
- Introduce **regime-conditioned models**: train separate directional models for High/Extreme stress vs. Low/Elevated stress regimes, given the distinct dynamics in each

#### 9.2 Feature Expansion

| New Feature | Source | Rationale |
|---|---|---|
| VIX futures term structure slope | CBOE (free daily data) | More granular vol surface information |
| Put/call ratio (equity + index) | CBOE (free daily data) | Sentiment / hedging demand |
| AAII Bull-Bear Spread | AAII (free weekly) | Retail sentiment as contrarian indicator |
| US 5Y5Y inflation breakeven | FRED: T5YIFR | Inflation tail risk |
| US credit card delinquency rate | FRED: DRCCLACBS | Consumer credit stress (monthly) |

#### 9.3 Ensemble Weight Optimization

Replace fixed ensemble weights with **dynamic weighting** using a meta-learner (isotonic regression or simple ridge) trained on rolling validation errors. This allows the ensemble to up-weight Ridge in low-volatility periods and LightGBM in trend-driven regimes.

#### 9.4 LLM Narrative Integration

- Activate Claude API integration with a refined prompt that includes:
  - Historical comparison to similar episodes ("Current stress profile resembles Q4 2018 most closely")
  - Actionable risk monitoring thresholds ("Watch for HY OAS exceeding 500bp as confirmation")
  - Scenario framing for each horizon

---

### Phase 3: International Sub-Models and Platform Hardening

**Target timeline:** ~4–6 months after Phase 1 validation

#### 9.5 Regional Sub-Models

Promote international features from auxiliary inputs to independent sub-models:

| Region | New Data Sources | Key Stress Signals |
|---|---|---|
| Europe | German Bund yield (^TNX proxy via FRED), VSTOXX, EUR sovereign spreads | ECB policy divergence, banking sector stress |
| Asia / Japan | JGB yield, Nikkei implied vol (^VXJ on Bloomberg; proxy via EWJ vol) | BOJ yield curve control risks, EM contagion |
| Emerging Markets | EMBI spread (FRED: BAMLEMCBPIOAS), EM local currency ETF (ELD) | Dollar funding stress for EM corporates |

Each regional sub-model produces its own stress score (0–100), which feeds into a revised global CSI with adjustable regional weights.

#### 9.6 Intraday Monitoring Mode

Add an intraday "alert" mode that runs on a configurable schedule (e.g., every 30 minutes) using:
- Real-time VIX and futures data from Yahoo Finance
- A lightweight "fast" model (Ridge only) for near-instantaneous stress delta estimates
- Push notification interface when stress delta exceeds a configurable threshold

#### 9.7 Platform Hardening

- **Automated retraining pipeline:** Monthly cron job that retrains all horizon models, computes alignment statistics against STLFSI4, and logs model drift metrics
- **Model versioning:** Store artifact checksums and training metadata; allow rollback to previous model versions
- **Test suite:** Unit tests for feature computation determinism, label construction reproducibility, and report generation integrity
- **Interactive configuration UI:** Web-based settings panel for adjusting CSI dimension weights, stress thresholds, and report preferences

#### 9.8 Alternative Label Exploration

Investigate alternative ground truth labels for the prediction target:

| Alternative | Pros | Cons |
|---|---|---|
| OFR Financial Stress Index | Daily, broader than STLFSI4 | Less commonly referenced |
| BIS Credit-to-GDP Gap | Long run macro stress | Quarterly; too low-frequency |
| Systemic Risk Measure (SRISK) | Bank-specific; captures contagion | Complex to compute; requires market cap data |
| Custom PCA-based FSI | Fully controlled; daily | Requires periodic re-estimation of factor loadings |

A multi-label training approach (predicting several FSI variants simultaneously) could improve robustness to individual index idiosyncrasies.

---

## Appendix: Stress Episode Reference

The following historical episodes are used for model validation:

| Episode | Date Range | Peak CSI (Phase 1) |
|---|---|---|
| Dot-com Crash | Mar 2000 – Oct 2002 | Moderate (data-limited) |
| 9/11 Shock | Sep 2001 – Oct 2001 | Elevated |
| Global Financial Crisis | Sep 2008 – Mar 2009 | Extreme (93% days > 50) |
| European Debt Crisis | Jul 2011 – Oct 2011 | High (90% days > 50) |
| China Growth Scare | Aug 2015 – Sep 2015 | High (67% days > 50) |
| COVID-19 Crash | Feb 2020 – Apr 2020 | Extreme (100% days > 50) |
| 2022 Inflation Bear Market | Jan 2022 – Dec 2022 | High (95% days > 50) |

---

*Last updated: 2026-05-22 | Model version: Phase 1.0*
