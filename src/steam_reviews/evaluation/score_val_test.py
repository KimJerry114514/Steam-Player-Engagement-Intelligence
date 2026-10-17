"""Score 2022 (validation) and 2023 (test) with the team's full-refit LightGBM+NLP model.

Rebuilds the 96 input features exactly as described in the model's metadata
(training_full_model.json, final_training.py). Prints validation-set metrics
only. 2023 labels are carried into the prediction file but no test-label
metric is computed here: the calibration and cut rules are frozen first.
"""
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from steam_reviews.paths import ROOT
MODEL_DIR = Path(os.environ.get("STEAM_MODEL_DIR", ROOT / "models" / "lightgbm"))
MODEL = MODEL_DIR / "training_full_model.txt"
CLEANED = Path(os.environ.get("STEAM_CLEANED", ROOT / "data" / "processed" / "negative_reviews_cleaned.parquet"))
NLP = Path(os.environ.get("STEAM_NLP", ROOT / "data" / "processed" / "nlp" / "nlp_factor_scores_2023.parquet"))
OUT = Path(os.environ.get("STEAM_RESULT_DIR", ROOT / "runs" / "calibration"))
OUT.mkdir(parents=True, exist_ok=True)
TARGET = "still_playing_d30"
TRAIN_START = pd.Timestamp("2010-10-15", tz="UTC")
VAL_START = pd.Timestamp("2022-01-01", tz="UTC")
TEST_START = pd.Timestamp("2023-01-01", tz="UTC")
DATE_FEATURES = ["review_year", "review_month", "review_dayofweek", "review_quarter_of_year"]

t0 = time.time()
OUT.mkdir(parents=True, exist_ok=True)

# 1. Model and its feature contract
meta = json.loads((MODEL_DIR / "training_full_model.json").read_text())
booster = lgb.Booster(model_file=str(MODEL_DIR / "training_full_model.txt"))
feats = meta["transformed_features"]
assert booster.num_trees() == meta["rounds_trained"] == 1000
assert [f.replace(" ", "_") for f in feats] == booster.feature_name(), "feature order mismatch"
p_cols = [f for f in feats if f.startswith("p_")]
plain_cols = [f for f in feats if f not in p_cols and f not in DATE_FEATURES
              and f not in ("language", "log_hours_at_review")]

# 2. Data: cleaned table + NLP scores (same join as the team's merged file)
read_cols = ["appid", "recommendationid", "review_created_utc", "edited", TARGET,
             "hours_at_review", "language", *plain_cols]
df = pq.read_table(CLEANED, columns=read_cols).to_pandas()
dates = pd.to_datetime(df["review_created_utc"], utc=True)
label = pd.to_numeric(df[TARGET], errors="coerce")
df = df[dates.ge(TRAIN_START) & label.isin([0, 1])].copy()
df["split"] = np.where(df["review_created_utc"] >= TEST_START, "test",
                       np.where(df["review_created_utc"] >= VAL_START, "val", "train"))
# training rows: reproduce the full-refit cohort count as a data check, keep a sample for a sanity check
counts = df["split"].value_counts().to_dict()
train_pos = int(df.loc[df["split"].eq("train"), TARGET].sum())
assert counts["train"] == meta["rows"], (counts["train"], meta["rows"])
assert train_pos == meta["positive_rows"], (train_pos, meta["positive_rows"])
train_sample = df[df["split"].eq("train")].sample(n=500_000, random_state=42)
df = pd.concat([df[df["split"].ne("train")], train_sample], ignore_index=True)

nlp = pq.read_table(NLP, columns=["appid", "recommendationid", *p_cols]).to_pandas()
before = len(df)
df = df.merge(nlp, on=["appid", "recommendationid"], how="left", validate="one_to_one")
del nlp
assert len(df) == before
nlp_missing = int(df[p_cols[0]].isna().sum())

# 3. Feature engineering, mirroring final_training.py
d = pd.to_datetime(df["review_created_utc"], utc=True)
df["review_year"] = d.dt.year.astype("float32")
df["review_month"] = d.dt.month.astype("float32")
df["review_dayofweek"] = d.dt.dayofweek.astype("float32")
df["review_quarter_of_year"] = d.dt.quarter.astype("float32")
df["log_hours_at_review"] = np.log1p(
    pd.to_numeric(df["hours_at_review"], errors="coerce").clip(lower=0)).astype("float32")
codes = {name: i for i, name in enumerate(meta["language_categories"])}
lang = df["language"].astype(object).where(df["language"].notna(), meta["language_fill_value"])
unknown_lang = int((~lang.isin(codes)).sum())
df["language_code"] = lang.map(codes).fillna(meta["unknown_language_code"]).astype("float32")

X = np.empty((len(df), len(feats)), dtype=np.float32)
for j, f in enumerate(feats):
    src = "language_code" if f == "language" else f
    X[:, j] = pd.to_numeric(df[src], errors="coerce").to_numpy(dtype=np.float32, na_value=np.nan)

# 4. Score
df["prob_raw"] = booster.predict(X, num_threads=8)
del X

# 5. Save predictions (val + test only)
keep = ["recommendationid", "appid", "review_created_utc", "split", "edited",
        "hours_at_review", TARGET, "prob_raw"]
pred = df.loc[df["split"].ne("train"), keep].reset_index(drop=True)
pred.to_parquet(OUT / "predictions_val_test_raw.parquet", index=False)


def metrics(frame):
    y, p = frame[TARGET].astype(int).to_numpy(), frame["prob_raw"].to_numpy()
    return {"rows": len(frame), "actual_rate": round(y.mean(), 4), "mean_pred": round(p.mean(), 4),
            "auc": round(roc_auc_score(y, p), 4), "ap": round(average_precision_score(y, p), 4),
            "log_loss": round(log_loss(y, p), 4), "brier": round(brier_score_loss(y, p), 4)}


tr = df[df["split"].eq("train")]
val = df[df["split"].eq("val")]
edited = val["edited"].fillna(False).astype(bool)
report = {
    "data_checks": {
        "train_rows_match_model_metadata": True,
        "train_rows": counts["train"], "train_positive_rows": train_pos,
        "val_rows": counts.get("val", 0), "test_rows": counts.get("test", 0),
        "nlp_missing_rows": nlp_missing, "unknown_language_rows": unknown_lang,
    },
    "sanity_train_sample_in_sample": metrics(tr),
    "val_2022_all": metrics(val),
    "val_2022_unedited": metrics(val[~edited]),
    "test_2023": "scored, not evaluated (rules not frozen yet)",
    "seconds": round(time.time() - t0, 1),
}
(OUT / "score_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
