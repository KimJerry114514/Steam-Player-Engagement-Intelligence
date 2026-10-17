"""Steps 4-5: apply the frozen Platt calibration to 2023 and run the final, one-time test evaluation.

Refuses to run unless frozen_rules.json exists and the model and prediction
files still match the fingerprints recorded when the rules were frozen.
Also writes calibrated 2022 + 2023 predictions for the segmentation step.
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from steam_reviews.paths import ROOT
MODEL_DIR = Path(os.environ.get("STEAM_MODEL_DIR", ROOT / "models" / "lightgbm"))
MODEL = MODEL_DIR / "training_full_model.txt"
CLEANED = Path(os.environ.get("STEAM_CLEANED", ROOT / "data" / "processed" / "negative_reviews_cleaned.parquet"))
NLP = Path(os.environ.get("STEAM_NLP", ROOT / "data" / "processed" / "nlp" / "nlp_factor_scores_2023.parquet"))
OUT = Path(os.environ.get("STEAM_RESULT_DIR", ROOT / "runs" / "calibration"))
OUT.mkdir(parents=True, exist_ok=True)
TARGET = "still_playing_d30"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if (OUT / "test_evaluation.json").exists() and os.environ.get("ALLOW_RERUN") != "1":
    raise SystemExit("test_evaluation.json already exists: the test set is evaluated once. Set ALLOW_RERUN=1 only to reproduce.")
rules = json.loads((OUT / "frozen_rules.json").read_text())
pred_path = OUT / "predictions_val_test_raw.parquet"
assert sha256(MODEL) == rules["model"]["sha256"], "model file changed after freezing"
assert sha256(pred_path) == rules["predictions_file_sha256"], "prediction file changed after freezing"

a, b, eps = rules["platt"]["a"], rules["platt"]["b"], rules["platt"]["logit_clip_eps"]
df = pd.read_parquet(pred_path)
p = np.clip(df["prob_raw"].to_numpy(), eps, 1 - eps)
df["prob_platt"] = 1 / (1 + np.exp(-(a + b * np.log(p / (1 - p)))))
df.to_parquet(OUT / "predictions_val_test_calibrated.parquet", index=False)


def scores(frame, col):
    y, q = frame[TARGET].astype(int).to_numpy(), frame[col].to_numpy()
    return {"mean_pred": round(q.mean(), 4), "auc": round(roc_auc_score(y, q), 4),
            "ap": round(average_precision_score(y, q), 4), "log_loss": round(log_loss(y, q), 4),
            "brier": round(brier_score_loss(y, q), 4)}


def top_k(frame, col, fractions=(0.10, 0.20, 0.50)):
    y = frame[TARGET].astype(int).to_numpy()
    order = np.argsort(-frame[col].to_numpy(), kind="stable")
    base, out = y.mean(), {}
    for f in fractions:
        k = int(round(len(y) * f))
        hit = y[order[:k]]
        out[f"top_{int(f * 100)}pct"] = {"precision": round(hit.mean(), 4), "recall": round(hit.sum() / y.sum(), 4),
                                         "lift": round(hit.mean() / base, 3)}
    return out


def deciles(frame, col):
    bins = pd.qcut(frame[col], 10, labels=False, duplicates="drop")
    g = frame.groupby(bins).agg(mean_pred=(col, "mean"), actual=(TARGET, "mean"), rows=(TARGET, "size"))
    return [{"decile": int(i) + 1, "mean_pred": round(r.mean_pred, 4), "actual": round(r.actual, 4)}
            for i, r in g.iterrows()]


test = df[df["split"].eq("test")]
unedited = test[~test["edited"].fillna(False).astype(bool)]
result = {
    "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "rules_frozen_at_utc": rules["frozen_at_utc"],
    "test_2023_all": {"rows": len(test), "actual_rate": round(test[TARGET].mean(), 4),
                      "raw": scores(test, "prob_raw"), "platt": scores(test, "prob_platt"),
                      "ranking": top_k(test, "prob_platt"),
                      "calibration_by_decile_platt": deciles(test, "prob_platt")},
    "test_2023_unedited": {"rows": len(unedited), "actual_rate": round(unedited[TARGET].mean(), 4),
                           "raw": scores(unedited, "prob_raw"), "platt": scores(unedited, "prob_platt")},
}
(OUT / "test_evaluation.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    for col, label in [("prob_raw", "Raw"), ("prob_platt", "Platt (fit on 2022)")]:
        d = deciles(test, col)
        ax.plot([r["mean_pred"] for r in d], [r["actual"] for r in d], marker="o", label=label)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set(xlabel="Mean predicted probability (decile)", ylabel="Actual D30 rate",
           title="LightGBM + NLP: 2023 test calibration", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "test_calibration_2023.png", dpi=180)
except ImportError:
    print("matplotlib not available; skipped figure")
