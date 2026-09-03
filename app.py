# Kenyan Food Price Forecasting Service
#
# Serves month-ahead staple food price forecasts for WFP-monitored camp and
# settlement markets in Kenya, each with an 80% prediction interval.
#
# DESIGN NOTE -- why the naive forecast is the primary output
# -----------------------------------------------------------------------
# Out-of-sample evaluation found that no fitted model beat the naive forecast
# (last observed price). ARIMA was statistically indistinguishable from it
# (Diebold-Mariano p = 0.937 and p = 0.246); the pooled XGBoost model was
# significantly WORSE (p < 0.0001 and p = 0.002). The price series are I(1),
# for which the naive forecast is the theoretically optimal one-step
# prediction.
#
# This service therefore reports the naive forecast as the headline value and
# the XGBoost forecast alongside it for transparency. The contribution is the
# calibrated interval, not the point estimate.
#
# Run:  python app.py      then POST to http://localhost:5000/forecast

import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

ARTIFACT_PATH = os.environ.get("ARTIFACT_PATH", "model_artifact.pkl")

with open(ARTIFACT_PATH, "rb") as fh:
    ART = pickle.load(fh)

MODEL = ART.get("model")
FEATS = ART.get("features", [])
HISTORY = ART["history_log"]
NAIVE_SD = float(ART["naive_resid_sd"])
MODEL_SD = float(ART.get("model_resid_sd", NAIVE_SD))
LAST_MONTH = pd.Period(ART["last_month"], "M")
Z80 = 1.2816  # 80% two-sided normal quantile


def _series(key):
    s = pd.Series({pd.Period(k, "M"): v for k, v in HISTORY[key].items()}).sort_index()
    return s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))


def _build_row(key, s, peers):
    # Recreate the training-time features from history, so the serving path and
    # the training path cannot drift apart.
    market, commodity = key.split("||")
    target = s.index.max() + 1

    lag = lambda L: s.get(target - L, np.nan)
    tail = lambda W: s.reindex(pd.period_range(target - W, target - 1, freq="M"))

    row = {f"lag{L}": lag(L) for L in (1, 2, 3, 6, 12)}
    for W in (3, 6, 12):
        row[f"rmean{W}"] = tail(W).mean()
        row[f"rstd{W}"] = tail(W).std()

    row["d1"] = row["lag1"] - row["lag2"]
    row["dev12"] = row["lag1"] - row["rmean12"]
    row["month_num"] = target.month
    row["xmkt"] = np.nanmean([p.get(target - 1, np.nan) for p in peers.values()])
    row["market"], row["commodity"], row["pricetype"] = market, commodity, "Retail"

    df = pd.DataFrame([row])
    for c in ("market", "commodity", "pricetype"):
        df[c] = df[c].astype("category")
    return df[FEATS], target


def _model_forecast(key, horizon):
    # Recursive: predict one month, append it, rebuild features, repeat.
    if MODEL is None:
        return None, None
    commodity = key.split("||")[1]
    s = _series(key)
    peers = {o: _series(o) for o in HISTORY if o.split("||")[1] == commodity}

    point, target = np.nan, None
    for _ in range(horizon):
        X, target = _build_row(key, s, peers)
        if X["lag1"].isna().any():
            return None, None
        point = float(MODEL.predict(X)[0])
        s = pd.concat([s, pd.Series({target: point})])
        peers[key] = s
    return float(np.exp(point)), target


@app.route("/")
def index():
    return jsonify(
        service="Kenya food price forecasting",
        endpoints={"GET /health": "service status and data vintage",
                   "GET /series": "available market-commodity series",
                   "POST /forecast": "month-ahead forecast with 80% interval"},
        primary_method="naive (last observed price)",
        note="See /health for why. No fitted model beat this benchmark out of sample.")


@app.route("/health")
def health():
    return jsonify(
        status="ok",
        n_series=len(HISTORY),
        data_through=str(LAST_MONTH),
        trained_at=ART.get("trained_at"),
        served_at=datetime.now(timezone.utc).isoformat(),
        primary_method="naive",
        evidence={
            "naive_median_mase": ART.get("naive_mase_median"),
            "model_median_mase": ART.get("model_mase_median"),
            "diebold_mariano_p": ART.get("dm_p"),
            "interpretation": ("The pooled model was significantly worse than the "
                               "naive benchmark out of sample, so the naive forecast "
                               "is reported as the primary value."),
        })


@app.route("/series")
def series():
    out = sorted(tuple(k.split("||")) for k in HISTORY)
    return jsonify(count=len(out),
                   series=[{"market": m, "commodity": c} for m, c in out])


@app.route("/forecast", methods=["POST"])
def forecast():
    body = request.get_json(silent=True) or {}
    market, commodity = body.get("market"), body.get("commodity")
    try:
        horizon = int(body.get("horizon", 1))
    except (TypeError, ValueError):
        return jsonify(error="'horizon' must be an integer"), 400

    if not market or not commodity:
        return jsonify(error="'market' and 'commodity' are required"), 400
    if not 1 <= horizon <= 6:
        return jsonify(error="'horizon' must be between 1 and 6"), 400

    key = f"{market}||{commodity}"
    if key not in HISTORY:
        return jsonify(error=f"unknown series: {market} / {commodity}",
                       hint="GET /series for the available list"), 404

    s = _series(key)
    if s.dropna().empty:
        return jsonify(error="no usable history for this series"), 422

    # Primary: naive. The last observed price, carried forward.
    last_obs = s.dropna()
    naive_point = float(np.exp(last_obs.iloc[-1]))
    observed_month = last_obs.index[-1]
    target = observed_month + horizon
    width = Z80 * NAIVE_SD * np.sqrt(horizon)

    # Secondary: the pooled model, for comparison only.
    model_point, _ = _model_forecast(key, horizon)

    return jsonify(
        market=market,
        commodity=commodity,
        target_month=str(target),
        horizon_months=horizon,
        forecast_kes_per_kg=round(naive_point, 2),
        interval_80={"low": round(max(naive_point - width, 0), 2),
                     "high": round(naive_point + width, 2)},
        method="naive (last observed price)",
        last_observed={"month": str(observed_month),
                       "price_kes_per_kg": round(naive_point, 2)},
        model_comparison={
            "xgboost_kes_per_kg": round(model_point, 2) if model_point else None,
            "note": ("Shown for transparency. This model was significantly worse "
                     "than the naive benchmark in out-of-sample testing and is "
                     "not the reported forecast."),
        },
        data_through=str(LAST_MONTH),
        caveats=[
            "The interval reflects ordinary forecast uncertainty estimated from "
            "test-period residuals. It does not cover regime change, such as a "
            "previously administered price beginning to move.",
            "Covers WFP-monitored camp and settlement markets only. Not applicable "
            "to Kenyan commercial wholesale markets.",
            "A price forecast is not a food-security indicator.",
        ])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
