"""Supplementary test diagnostics added 2026-10-07 after an independent audit.

Descriptive only: nothing here changes the frozen Platt parameters, cut rules,
or the headline test metrics in test_evaluation.json.
- unedited-only ranking + decile calibration (promised in frozen_rules.json test_metrics_planned)
- game-clustered bootstrap 95% CI for test AUC (reviews cluster within games)
- AUC for games seen vs not seen in the training years, and without the largest game
- calibration gap by follow-up length (the label is open-ended up to the Nov-2023 snapshot)
- monthly AUC
- four segments in 2022 and 2023 under the frozen rules (reference values for the segmentation step)
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from steam_reviews.paths import ROOT
MODEL_DIR = Path(os.environ.get("STEAM_MODEL_DIR", ROOT / "models" / "lightgbm"))
MODEL = MODEL_DIR / "training_full_model.txt"
CLEANED = Path(os.environ.get("STEAM_CLEANED", ROOT / "data" / "processed" / "negative_reviews_cleaned.parquet"))
NLP = Path(os.environ.get("STEAM_NLP", ROOT / "data" / "processed" / "nlp" / "nlp_factor_scores_2023.parquet"))
OUT = Path(os.environ.get("STEAM_RESULT_DIR", ROOT / "runs" / "calibration"))
OUT.mkdir(parents=True, exist_ok=True)
TARGET = "still_playing_d30"
N_BOOT = 500

rules = json.loads((OUT / "frozen_rules.json").read_text())
df = pd.read_parquet(OUT / "predictions_val_test_calibrated.parquet")
extra = pq.read_table(CLEANED, columns=["appid", "recommendationid", "review_age_days_at_snapshot"],
                      filters=[("review_created_utc", ">=", pd.Timestamp("2022-01-01", tz="UTC"))]).to_pandas()
df = df.merge(extra, on=["appid", "recommendationid"], how="left", validate="one_to_one")
train_games = set(pq.read_table(CLEANED, columns=["appid"], filters=[
    ("review_created_utc", "<", pd.Timestamp("2022-01-01", tz="UTC"))]).column("appid").to_pylist())


def auc(frame, col="prob_raw"):
    return round(float(roc_auc_score(frame[TARGET].astype(int), frame[col])), 4)


def top_k(frame, col="prob_platt"):
    y = frame[TARGET].astype(int).to_numpy()
    order = np.argsort(-frame[col].to_numpy(), kind="stable")
    out = {}
    for f in (0.10, 0.20, 0.50):
        hit = y[order[:int(round(len(y) * f))]]
        out[f"top_{int(f * 100)}pct"] = {"precision": round(hit.mean(), 4), "recall": round(hit.sum() / y.sum(), 4),
                                         "lift": round(hit.mean() / y.mean(), 3)}
    return out


def deciles(frame, col="prob_platt"):
    g = frame.groupby(pd.qcut(frame[col], 10, labels=False)).agg(mean_pred=(col, "mean"), actual=(TARGET, "mean"))
    return [{"decile": int(i) + 1, "mean_pred": round(r.mean_pred, 4), "actual": round(r.actual, 4)} for i, r in g.iterrows()]


def game_bootstrap_ci(frame, seed=0):
    y = frame[TARGET].astype(int).to_numpy()
    p = frame["prob_raw"].to_numpy()
    codes, uniq = pd.factorize(frame["appid"])
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(len(uniq)))
    ends = np.append(starts[1:], len(codes))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([order[starts[k]:ends[k]] for k in pick])
        stats.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"auc": auc(frame), "ci95": [round(lo, 4), round(hi, 4)], "games": len(uniq), "resamples": N_BOOT}


def segments(frame):
    hours_cut = rules["segments"]["prior_engagement_cut_hours"]
    median = float(frame["prob_raw"].median())
    hi_prior = frame["hours_at_review"].to_numpy() > hours_cut
    hi_later = frame["prob_raw"].to_numpy() >= median
    name = np.select([hi_prior & hi_later, hi_prior & ~hi_later, ~hi_prior & hi_later],
                     ["Hardcore", "Dropout", "New Blood"], "Nobody")
    g = pd.DataFrame({"segment": name, "y": frame[TARGET].astype(int).to_numpy()}).groupby("segment")["y"]
    return {"hours_cut": hours_cut, "prob_median_this_period": round(median, 4),
            "segments": {k: {"share": round(v.size / len(frame), 4), "actual_d30": round(v.mean(), 4)}
                         for k, v in g}}


val, test = df[df["split"].eq("val")], df[df["split"].eq("test")].copy()
unedited = test[~test["edited"].fillna(False).astype(bool)]
seen = test["appid"].isin(train_games)
largest = test["appid"].value_counts().index[0]
test["followup_quartile"] = pd.qcut(test["review_age_days_at_snapshot"], 4, labels=["Q1 shortest", "Q2", "Q3", "Q4 longest"])

result = {
    "note": "Added 2026-10-07 after audit. Descriptive only; frozen rules and test_evaluation.json unchanged.",
    "test_period_covered": [str(test["review_created_utc"].min()), str(test["review_created_utc"].max())],
    "test_auc_game_clustered": game_bootstrap_ci(test),
    "val_auc_game_clustered": game_bootstrap_ci(val),
    "test_games_seen_vs_new": {
        "seen_in_training_share_of_rows": round(float(seen.mean()), 4),
        "auc_seen": auc(test[seen]), "auc_new": auc(test[~seen]),
    },
    "test_largest_game": {"appid": int(largest), "share_of_rows": round(float(test["appid"].eq(largest).mean()), 4),
                          "auc_this_game": auc(test[test["appid"].eq(largest)]),
                          "auc_without_it": auc(test[test["appid"].ne(largest)])},
    "test_calibration_by_followup_quartile": [
        {"quartile": str(q), "median_days": float(g["review_age_days_at_snapshot"].median()),
         "actual": round(g[TARGET].mean(), 4), "mean_platt": round(g["prob_platt"].mean(), 4),
         "gap_pp": round((g["prob_platt"].mean() - g[TARGET].mean()) * 100, 1), "auc": auc(g)}
        for q, g in test.groupby("followup_quartile", observed=True)],
    "test_monthly_auc": {str(m): auc(g) for m, g in test.groupby(test["review_created_utc"].dt.strftime("%Y-%m"))},
    "test_platt_gap_closed": round(1 - (test["prob_platt"].mean() - test[TARGET].mean())
                                   / (test["prob_raw"].mean() - test[TARGET].mean()), 3),
    "test_unedited_ranking": top_k(unedited),
    "test_unedited_calibration_by_decile": deciles(unedited),
    "segments_2022_frozen_rule": segments(val),
    "segments_2023_frozen_rule": segments(test),
}
(OUT / "supplement_diagnostics_20261007.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
