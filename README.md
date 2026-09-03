# Forecasting Food Prices in Kenyan Markets

Month-ahead forecasting of staple food commodity prices in WFP-monitored Kenyan markets,
with a rigorous out-of-sample comparison of classical time-series and machine-learning
approaches against a naive benchmark.

**Headline result: no fitted model beat the naive forecast.** ARIMA was statistically
indistinguishable from it; pooled XGBoost was significantly worse. This is a genuine
finding about the data, predicted in advance by the stationarity analysis, and is
explained in full below.

AnalystLab Africa — Machine Learning Internship, Week 8 Capstone.

---

## Contents

- [Problem](#problem)
- [Data](#data)
- [Approach](#approach)
- [Results](#results)
- [Why the naive forecast wins](#why-the-naive-forecast-wins)
- [Repository structure](#repository-structure)
- [Running the notebook](#running-the-notebook)
- [Running the API](#running-the-api)
- [Limitations](#limitations)
- [Reproducibility](#reproducibility)

---

## Problem

Households, traders and humanitarian agencies in Kenya make decisions against food prices
they cannot see in advance. Prices are observed only after the fact and published with a
reporting lag, so by the time a sharp increase is visible the decisions it should have
informed have already been made.

**Research question**

> Can next month's staple food price in a given Kenyan market be forecast from information
> observable at the time of prediction, and does a machine-learning model that pools
> information across many markets outperform classical time-series models fitted to each
> market separately?

**Why it matters.** Food is roughly a third of the Kenyan consumer price index, and staple
prices are the largest driver of short-run changes in household purchasing power. In the
markets studied here, the World Food Programme sets cash-transfer values against observed
prices — if prices rise between observation and transfer, the transfer under-delivers in
real terms.

---

## Data

**Source:** World Food Programme Global Food Prices Database, Kenya
→ https://data.humdata.org/dataset/wfp-food-prices-for-kenya

Distributed through the Humanitarian Data Exchange. Contributing agencies include the Kenya
National Bureau of Statistics, the Ministry of Agriculture, the National Drought Management
Authority, the Energy and Petroleum Regulatory Authority, and FAO GIEWS.
Licence: Creative Commons Attribution for Intergovernmental Organisations.

**Coverage:** ~28,000 monthly observations, January 2006 onward, across 226 markets and
multiple commodities, price types and units.

### Why this source rather than the alternatives

The World Bank Real-Time Food Prices dataset covers Kenya over a longer period with better
documentation, but it is compiled partly by **machine-learning imputation** of missing
values. Forecasting it would mean training a model to predict another model's output, and
its imputation may draw on information from after the month being imputed — leakage that
cannot be detected or removed. WFP data is used because it records **directly observed
prices**.

### The central data finding

The dataset is **two disjoint collection programmes, not one**:

| | Long-history series | Current series |
|---|---|---|
| Count | 33 | 827 |
| History | 120–189 months | ≤ 59 months |
| Ends | 2019–2022 | 2024–2026 |
| Markets | Commercial wholesale + arid retail | Camp and settlement |

There is **no overlap** — no series is both long and currently reporting. A design that is
simultaneously long-history and current is impossible with this data, which forced a
two-study structure:

- **Study A** — 15 series, 2006–2020, ~165 months each. Long history, supports seasonal
  models and an extended test window. Retrospective methods benchmark.
- **Study B** — 73 series, 2023–2026, ~37 months each. Balanced 12-market × 7-commodity
  panel running to the most recent month. The operational half.

---

## Approach

### Phase 0 — data validation before any modelling

No model was fitted until the data had been validated. This occupies roughly half the
notebook and is a deliberate choice.

1. **Series-level inventory.** The forecastable unit is `market × commodity × pricetype ×
   unit × currency`, not "commodity". Aggregating at a coarser grain mixes retail with
   wholesale and kilograms with 90 kg bags.
2. **Coverage and continuity.** Density (observed months ÷ months spanned), longest gap,
   and gap positions per series. Revealed a panel-wide suspension of camp price monitoring
   through most of 2022 — identical across all seven commodities within each market,
   indicating a market-level interruption rather than commodity dropout.
3. **Duplicate-label testing.** "Maize" and "Maize (white)" at the same market proved to be
   the same underlying price measured twice (level correlation 0.95–0.96, median difference
   4%); one was dropped. "Beans" and "Beans (dry)" proved to be genuinely different products
   (correlation 0.72, 23% difference); both retained.
4. **Unit normalisation.** All prices converted to KES/kg or KES/L. Weight and volume kept
   on separate bases.
5. **Outlier screening.** A spike-and-rebound rule flagged candidates; each was inspected
   in a seven-month window. **Five observations of ~5,600 were removed**, each with an
   identified mechanism. Several large movements were flagged and deliberately **retained**
   — including five markets recording identical bean prices in August 2025, which is a
   market-wide event rather than five coincident errors.
6. **Stationarity.** ADF and KPSS on every series, in levels and first differences.
   **All 15 Study A series are I(1)** with both tests agreeing on every one.
7. **Seasonality.** STL decomposition. Seasonal strength peaks at 0.37 and falls to 0.00
   for arid-market retail series.

### Modelling

Four models forming a ladder, each testing one additional idea:

| Model | Fitted | Tests |
|---|---|---|
| Naive | — | The benchmark. Optimal one-step forecast for a random walk |
| Seasonal naive | — | Whether annual price structure is exploitable |
| ARIMA / SARIMA | Per series | Temporal dependence within each series' own history |
| XGBoost | Pooled across the panel | Whether cross-series pooling and covariates help |

**Deep learning was considered and rejected.** With ~120 training months in Study A and ~30
in Study B, recurrent networks would overfit and produce results indistinguishable from
noise.

### Guarding against leakage

- **Chronological splits only.** No `train_test_split`.
- **Backward-only features.** Rolling statistics computed on the already-lagged series.
  Verified by an assertion that rebuilds features on a truncated panel and fails if any
  feature value at time *t* changes when later rows are deleted.
- **Order selection on training data only.** ARIMA orders chosen by AICc within the
  training block; hyperparameters tuned by expanding-window CV on training rows.
- **Fixed eligible-series set.** Every model scored on identical series, so a model that
  silently skipped harder series cannot appear to outperform one that did not.
- **No imputation.** Missing months are left as `NaN` and handled by the Kalman filter and
  by XGBoost's native missing-value branching. Linear interpolation would have used
  *future* values to fill past gaps.

### Metrics

**MASE** (median across series) as the headline — scaled per series so markets at different
price levels contribute equally. **MAE and RMSE in KES/kg** for interpretability.
**Diebold–Mariano** to test whether differences are meaningful. **Per-series win counts**,
because a model with a good average that wins overwhelmingly on three series is not the
same as one winning narrowly everywhere.

R² is deliberately omitted — it is uninformative for forecasts against a persistent
baseline.

---

## Results

### Model comparison — median MASE (lower is better)

| Model | Study A (15 series) | Study B (73 series) |
|---|---|---|
| **Naive** | **0.890** | **0.547** |
| ARIMA | 0.887 | 0.644 |
| XGBoost | 1.027 | 0.739 |
| Seasonal naive | 2.595 | — |

### Diebold–Mariano against naive

| Model | Study A | Study B |
|---|---|---|
| ARIMA | p = 0.937 — **no significant difference** | p = 0.246 — **no significant difference** |
| XGBoost | p < 0.0001 — **worse** | p = 0.002 — **worse** |
| Seasonal naive | p < 0.0001 — **worse** | — |

### Key findings

1. **No model beats naive.** ARIMA ties it; XGBoost is significantly worse in both studies.
2. **The ARIMA order search independently selected the naive forecast.** Chosen by AICc
   with no knowledge of test performance: every series selected `d = 1`, and in Study B
   **12 of 73 selected `(0,1,0)` — a pure random walk, which *is* the naive forecast.** A
   further 44 selected a single term.
3. **Seasonality is absent.** Seasonal naive is ~3× worse than naive and beats it on zero
   series; the ARIMA search declined seasonal terms for 13 of 15 Study A series. This runs
   against what Kenya's bimodal rainfall and maize marketing calendar would predict.
4. **Markets are partially integrated, but pooling did not pay.** The cross-market feature
   `xmkt` earned 17.6% of gain in Study B and ranks third — so markets do carry information
   about each other — yet the pooled model still lost. The signal is real but small, and the
   variance from estimating eighteen features outweighed it.
5. **Panel width matters more than pooling.** A controlled truncation experiment cut Study
   A's series to 43 months, holding markets, commodities, era and test window fixed. ARIMA
   *improved* (−22.5% MAE); XGBoost degraded (+10.6%). With only 15 series there is too
   little cross-sectional information to compensate for lost temporal depth.
6. **Regime change is the dominant failure mode.** One salt series held at exactly
   50.0 KES/kg for 34 consecutive months, then began moving. No model trained on the fixed
   period could have anticipated it.
7. **Measurement error bounds achievable accuracy.** Two independent recordings of the same
   maize price differ by a median of 4%. No model should be expected to forecast its target
   more precisely than the target is observed.

---

## Why the naive forecast wins

This result was **predicted before any model was fitted**, and that is what makes it a
finding rather than a failure.

Phase 0 established that every series is I(1) — non-stationary in levels, stationary in
first differences, with ADF and KPSS agreeing on all 15. For a random walk, the last
observed value is the theoretically optimal one-step prediction. The modelling section
stated this explicitly before fitting anything.

**Four independent methods agree:** stationarity testing, STL decomposition, automated
ARIMA order selection, and out-of-sample forecast comparison. Any one alone would be weak;
together they are difficult to dismiss.

**What this does not mean.** It does not mean forecasting is impossible here. It means that
at one-month horizon the useful output is not the point forecast — which the naive method
already provides — but the **calibrated prediction interval** around it. That is what the
deployed service provides, and it is why the service reports the naive forecast as its
primary value rather than deploying a model the evaluation found to be worse.

---

## Repository structure

```
.
├── README.md
├── notebooks/
│   └── WEEK_8_capstone_complete.ipynb    # full analysis, Phase 0 through deployment
├── deploy/
│   ├── app.py                            # Flask forecasting service
│   ├── export_artifact.py                # builds model_artifact.pkl from the notebook
│   ├── test_app.py                       # smoke tests for the API
│   ├── requirements.txt
│   └── Dockerfile
├── data/
│   └── raw/                              # frozen WFP CSV (not committed)
└── figs/                                 # generated figures
```

`data/raw/` is gitignored. The dataset is downloaded on first run and cached, and its
SHA-256 is recorded so results can be traced to an exact file version.

---

## Running the notebook

```bash
git clone <repository-url>
cd <repository>
pip install -r deploy/requirements.txt
jupyter notebook notebooks/WEEK_8_capstone_complete.ipynb
```

Or open it in Google Colab and run all cells. The data-loading cell downloads the WFP CSV
once and caches it locally; subsequent runs read from disk so the dataset stays frozen while
modelling.

**Runtime:** roughly 15–25 minutes end to end. The ARIMA cells fit a small order grid per
series and account for most of it.

---

## Running the API

The service requires `model_artifact.pkl`, which is produced by the notebook — it bundles
the trained model with the recent price history needed to construct features.

```bash
# 1. Run the notebook through Section 9, then run export_artifact.py in a cell.
#    This writes deploy/model_artifact.pkl

cd deploy
pip install -r requirements.txt
python test_app.py          # smoke tests
python app.py               # serve on :5000
```

### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Service description |
| `/health` | GET | Status, data vintage, and the evidence behind the method choice |
| `/series` | GET | Available market–commodity series |
| `/forecast` | POST | Month-ahead forecast with an 80% interval |

### Example

```bash
curl -X POST http://localhost:5000/forecast \
     -H "Content-Type: application/json" \
     -d '{"market": "Kakuma 3", "commodity": "Maize flour", "horizon": 1}'
```

```json
{
  "market": "Kakuma 3",
  "commodity": "Maize flour",
  "target_month": "2026-09",
  "forecast_kes_per_kg": 88.40,
  "interval_80": {"low": 79.10, "high": 97.70},
  "method": "naive (last observed price)",
  "model_comparison": {
    "xgboost_kes_per_kg": 89.15,
    "note": "Shown for transparency. Significantly worse than naive out of sample."
  },
  "data_through": "2026-08"
}
```

The service reports the **naive forecast as its primary output**, with the XGBoost forecast
alongside for transparency. Deploying a model the evaluation found to be significantly
worse than the benchmark would contradict the analysis.

### Docker

```bash
cd deploy
docker build -t kenya-food-forecast .
docker run -p 5000:5000 kenya-food-forecast
```

---

## Limitations

- **Humanitarian markets only.** The operational panel covers 12 camp and settlement markets
  in Turkana and Garissa. Conclusions do not transfer to Kenya's commercial food economy.
- **Study A is retrospective.** Its series end in 2019–2020; it evaluates methods and cannot
  be deployed.
- **Study B cannot test seasonality.** ~43 months is 3.5 annual cycles.
- **No true real-time backtest.** WFP revises history retroactively and no public vintage
  archive exists, so the evaluation uses the latest version of history — slightly cleaner
  than what a forecaster would have had. Reported accuracy is a mild upper bound.
- **Supply-side variables are absent.** No traded volumes, stock levels, or usable farm-gate
  prices. These are plausibly the most important omitted drivers of price formation.
- **Prices are coarsely recorded.** Camp prices cluster on round figures, placing a floor on
  achievable precision independent of the model.
- **Pooled significance testing is approximate.** Diebold–Mariano treats forecast errors as
  independent across series, but markets share national shocks. Per-series win counts are
  reported alongside for this reason.
- **This is not a food-security indicator.** Forecasting a price is not forecasting hunger.
- **No causal claims.** The models are predictive. Feature importance describes what the
  model used, not what moves prices.

---

## Reproducibility

- The WFP CSV is downloaded once and cached; its SHA-256 is printed and recorded.
- All splits are chronological, with boundaries defined as named constants.
- Outlier removals are listed individually with a stated mechanism, and set to `NaN` rather
  than deleted so the calendar position is preserved.
- Random seeds are fixed for all model fitting.
- A leakage assertion fails loudly if any feature at time *t* depends on later data.

**Known caveat.** HDX revises the WFP file retroactively, so re-downloading may produce
slightly different series counts. Freeze the CSV before running the analysis and record its
digest.

---

## Acknowledgements

Data: World Food Programme, via the Humanitarian Data Exchange, with contributions from the
Kenya National Bureau of Statistics, the Ministry of Agriculture, the National Drought
Management Authority, EPRA and FAO GIEWS.

Prepared for the AnalystLab Africa Machine Learning Internship.
