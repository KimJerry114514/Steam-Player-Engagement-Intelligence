"""Step 2-3: fit Platt calibration and fix the segment cut rules on 2022 ONLY, then freeze them.

Uses only validation (2022) rows of the prediction file and training-period
hours from the cleaned table. No 2023 label is used for any fitted or chosen value.
Writes frozen_rules.json; evaluate_test.py refuses to run without it.
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from steam_reviews.paths import ROOT
MODEL_DIR = Path(os.environ.get("STEAM_MODEL_DIR", ROOT / "models" / "lightgbm"))
MODEL = MODEL_DIR / "training_full_model.txt"
CLEANED = Path(os.environ.get("STEAM_CLEANED", ROOT / "data" / "processed" / "negative_reviews_cleaned.parquet"))
NLP = Path(os.environ.get("STEAM_NLP", ROOT / "data" / "processed" / "nlp" / "nlp_factor_scores_2023.parquet"))
OUT = Path(os.environ.get("STEAM_RESULT_DIR", ROOT / "runs" / "calibration"))
OUT.mkdir(parents=True, exist_ok=True)
TARGET = "still_playing_d30"
EPS = 1e-6


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


pred_path = OUT / "predictions_val_test_raw.parquet"
val = pq.read_table(pred_path, filters=[("split", "==", "val")],
                    columns=["hours_at_review", TARGET, "prob_raw"]).to_pandas()
y = val[TARGET].astype(int).to_numpy()
z = logit(val["prob_raw"].to_numpy())

# Platt: p_cal = sigmoid(a + b * logit(p_raw)), fitted on 2022 only
platt = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(z.reshape(-1, 1), y)
a, b = float(platt.intercept_[0]), float(platt.coef_[0, 0])
p_cal = 1 / (1 + np.exp(-(a + b * z)))

# Cut 1: prior engagement = hours at review, split at the training-period median
# (training cohort = the final model's cohort: 2010-10-15 <= date < 2022-01-01, label in {0,1}, edited included)
tr = pq.read_table(CLEANED, columns=["review_created_utc", TARGET, "hours_at_review"]).to_pandas()
d = pd.to_datetime(tr["review_created_utc"], utc=True)
tr = tr[d.ge(pd.Timestamp("2010-10-15", tz="UTC")) & d.lt(pd.Timestamp("2022-01-01", tz="UTC"))
        & pd.to_numeric(tr[TARGET], errors="coerce").isin([0, 1])]
assert len(tr) == 10_763_783
hours_cut = float(tr["hours_at_review"].median())
del tr

# Cut 2: later engagement = top 50% of predicted probability WITHIN each scored period
# (rank-based, so Platt does not change it; applied separately to 2022 and to 2023)
prob_cut_val = float(np.median(val["prob_raw"]))
high_prior = val["hours_at_review"].to_numpy() > hours_cut
high_later = val["prob_raw"].to_numpy() >= prob_cut_val
names = np.select([high_prior & high_later, high_prior & ~high_later, ~high_prior & high_later],
                  ["Hardcore", "Dropout", "New Blood"], "Nobody")
seg = (pd.DataFrame({"segment": names, "y": y})
       .groupby("segment")["y"].agg(rows="size", actual_d30="mean"))
seg["share"] = seg["rows"] / seg["rows"].sum()

rules = {
    "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "note": "Fixed on 2022 validation data only, before any 2023 label was read.",
    "model": {"file": "training_full_model.txt", "sha256": sha256(MODEL_DIR / "training_full_model.txt")},
    "predictions_file_sha256": sha256(pred_path),
    "cohort": "review date in range, still_playing_d30 in {0,1}, edited reviews included (same as the final model)",
    "platt": {"formula": "p_cal = 1 / (1 + exp(-(a + b * logit(p_raw))))", "a": round(a, 6), "b": round(b, 6),
              "fitted_on": "2022 validation, all rows", "logit_clip_eps": EPS},
    "segments": {
        "prior_engagement_cut_hours": hours_cut,
        "prior_engagement_rule": "High if hours_at_review > cut (cut = median of the 2010-2021 training cohort)",
        "later_engagement_rule": "High if predicted probability >= the median prediction of the same period "
                                 "(top 50% within 2022, top 50% within 2023); rank-based, unaffected by Platt",
        "labels": {"High prior + High later": "Hardcore", "High prior + Low later": "Dropout",
                   "Low prior + High later": "New Blood", "Low prior + Low later": "Nobody"},
    },
    "test_metrics_planned": ["AUC", "AP", "log loss", "Brier", "mean prediction vs actual (raw and Platt)",
                             "calibration by decile", "precision/recall/lift at top 10%/20%/50%",
                             "same metrics on unedited reviews only"],
    "validation_evidence": {
        "rows": int(len(val)), "actual_rate": round(float(y.mean()), 4),
        "mean_pred_raw": round(float(val["prob_raw"].mean()), 4), "mean_pred_platt": round(float(p_cal.mean()), 4),
        "auc": round(float(roc_auc_score(y, p_cal)), 4),
        "log_loss_raw": round(float(log_loss(y, val["prob_raw"])), 4), "log_loss_platt": round(float(log_loss(y, p_cal)), 4),
        "brier_raw": round(float(brier_score_loss(y, val["prob_raw"])), 4),
        "brier_platt": round(float(brier_score_loss(y, p_cal)), 4),
        "median_pred_2022": round(prob_cut_val, 4),
        "segments_2022": {k: {"share": round(float(r.share), 4), "actual_d30": round(float(r.actual_d30), 4)}
                          for k, r in seg.iterrows()},
    },
}
rules_path = OUT / "frozen_rules.json"
if rules_path.exists():
    raise SystemExit("frozen_rules.json already exists; refusing to overwrite frozen rules")
rules_path.write_text(json.dumps(rules, indent=2))
print(json.dumps(rules, indent=2))
