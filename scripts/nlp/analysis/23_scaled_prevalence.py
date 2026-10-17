"""Classifier-tier prevalence per scope with misclassification adjustment, compared with the LLM-labeled sample tier.
Adjustment (Rogan-Gladen): p_adj = (p_obs - FPR) / (TPR - FPR), clipped to [0,1], with TPR/FPR estimated on the held-out
test set against the LLM (two-pass) labels. Only factors with test F1 >= 0.40 are scaled ("classifier tier").
Usage: python 23_scaled_prevalence.py <factor_matrix.parquet>  -> results/scaled_prevalence.csv"""
import sys, json, duckdb, numpy as np, pandas as pd
ROOT = "."
fm = sys.argv[1]
import os
TAGC = os.environ.get("CLS_TAG", "ft")
th = json.load(open(f"{ROOT}/results/{TAGC}_thresholds.json"))["thresholds"]
met = json.load(open(f"{ROOT}/results/{TAGC}_metrics.json"))
P = pd.read_parquet(f"{ROOT}/results/{TAGC}_test_predictions.parquet")
L = pd.read_parquet(os.environ.get("LABELS_PATH", f"{ROOT}/results/labels/final/labels_primary.parquet")).set_index("rid")  # must be the labels the classifier was trained/tested on
keep = [f for f, m in met.items() if m["test_f1"] >= 0.40 and m["n_pos_test"] >= 5]
rates = {}
for f in keep:
    y = L.loc[P.rid, f].values.astype(bool); yhat = (P[f].values >= th[f])
    rates[f] = {"tpr": (yhat & y).sum() / max(y.sum(), 1), "fpr": (yhat & ~y).sum() / max((~y).sum(), 1), "test_f1": met[f]["test_f1"]}
GAMES = {730: "Counter-Strike", 578080: "PUBG", 1172470: "Apex Legends", 1091500: "Cyberpunk 2077", 105600: "Terraria"}
con = duckdb.connect(); con.execute("SET threads=8")
sel = ", ".join(f'avg("{f}") AS "{f}"' for f in keep)
rows = []
for scope, where in [(os.environ.get("OTHER_SCOPE", "All games (1M random sample)"), "f.appid NOT IN (730,578080,1172470,1091500,105600)")] + [(g, f"f.appid = {a}") for a, g in GAMES.items()]:
    r = con.execute(f"""SELECT count(*) n, {sel} FROM (SELECT m.*, a.appid FROM '{fm}' m JOIN '{ROOT}/data/neg_all.parquet' a USING(rid)) f WHERE {where}""").df().iloc[0]
    for f in keep:
        tpr, fpr = rates[f]["tpr"], rates[f]["fpr"]; po = float(r[f])
        adj = float(np.clip((po - fpr) / (tpr - fpr), 0, 1)) if tpr - fpr > 0.05 else np.nan
        rows.append({"scope": scope, "factor": f, "n_reviews": int(r.n), "p_observed": po, "p_adjusted": adj, "tpr": tpr, "fpr": fpr, "test_f1": rates[f]["test_f1"]})
S = pd.DataFrame(rows)
st = pd.read_csv(f"{ROOT}/results/sample_tier_prevalence.csv").rename(columns={"prevalence": "p_sample", "lo90": "sample_lo90", "hi90": "sample_hi90"})
st["scope"] = st.scope.replace({"All games (random sample)": os.environ.get("OTHER_SCOPE", "All games (1M random sample)"), "All games (random 1,500)": os.environ.get("OTHER_SCOPE", "All games (1M random sample)")})
S = S.merge(st[["scope", "factor", "p_sample", "sample_lo90", "sample_hi90", "k", "n"]], on=["scope", "factor"], how="left")
S.to_csv(os.environ.get("OUT_CSV", f"{ROOT}/results/scaled_prevalence.csv"), index=False)
ok = S.dropna(subset=["p_adjusted", "p_sample"])
print("classifier-tier factors:", len(keep), "| corr(adjusted, sample-tier) =", round(ok.p_adjusted.corr(ok.p_sample), 3),
      "| corr(observed, sample) =", round(ok.p_observed.corr(ok.p_sample), 3), "| n pairs", len(ok))
print(ok.sort_values(["scope", "p_adjusted"], ascending=[True, False]).groupby("scope").head(4)[["scope", "factor", "p_observed", "p_adjusted", "p_sample"]].round(3).to_string(index=False))
