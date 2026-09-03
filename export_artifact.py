"""Export the deployment artifact from the notebook.

Paste this into a notebook cell AFTER Section 9 (the operational forecast).
It requires: panel_B, COLS_B, TRAIN_END_B, model_B, FEATS, build_matrix,
base_B, xgb_B, and results_B to be in memory.
"""

import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

Path("deploy").mkdir(exist_ok=True)

M_B = build_matrix(panel_B)

# Last 18 observed months per series -- enough for lag12 plus rolling windows
history = {
    f"{c[0]}||{c[1]}": {str(m): float(v) for m, v in M_B[c].dropna().tail(18).items()}
    for c in COLS_B
}

# Residual spread of each method over the test period, in KES/kg
naive_sd = float(np.nanstd(np.concatenate(
    base_B.loc[base_B["model"] == "Naive", "errors"].values)))
model_sd = float(np.nanstd(np.concatenate(xgb_B["errors"].values)))

med = results_B.groupby("model")["mase"].median()

artifact = {
    "model": model_B,
    "features": FEATS,
    "history_log": history,
    "naive_resid_sd": naive_sd,
    "model_resid_sd": model_sd,
    "naive_mase_median": round(float(med.get("Naive", np.nan)), 3),
    "model_mase_median": round(float(med.get("XGBoost", np.nan)), 3),
    "dm_p": 0.0018,                      # XGBoost vs naive, Study B
    "last_month": str(M_B.index.max()),
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "n_series": len(COLS_B),
}

with open("deploy/model_artifact.pkl", "wb") as fh:
    pickle.dump(artifact, fh)

print(f"Wrote deploy/model_artifact.pkl")
print(f"  series          : {artifact['n_series']}")
print(f"  data through    : {artifact['last_month']}")
print(f"  naive resid sd  : {naive_sd:.3f} KES/kg   -> 80% interval = +/- {1.2816*naive_sd:.2f} at h=1")
print(f"  model resid sd  : {model_sd:.3f} KES/kg")
print(f"  naive / model median MASE : {artifact['naive_mase_median']} / {artifact['model_mase_median']}")
