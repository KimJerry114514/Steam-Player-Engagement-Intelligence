"""Sample tier: estimates from the representative, high-quality (two-pass LLM) labeled samples.
- prevalence per focus game (400 random each) and all-games (1,500 random) with Wilson 90% CIs
- factor -> still_playing_d30 on the pooled random samples: LPM with game FE + review-year FE + log hours at review,
  HC1 robust SE, factors with >= 25 positives; used to cross-check the direction of the classifier-scaled estimates.
Outputs: results/sample_tier_prevalence.csv, results/sample_tier_effects.csv"""
import json, numpy as np, pandas as pd, duckdb, statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
ROOT = "."
import sys
L = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/results/labels/final/labels_primary.parquet")
FIDS = [c for c in L.columns if "." in c]
cb = {f["id"]: f["label"] for f in json.load(open(f"{ROOT}/llm_jobs/out/revise/codebook_v1_1.json"))["factors"]}
GAMES = {730: "Counter-Strike", 578080: "PUBG", 1172470: "Apex Legends", 1091500: "Cyberpunk 2077", 105600: "Terraria"}
rows = []
for scope, sub in [("All games (random sample)", L[L.source == "random_all"])] + [(g, L[(L.source == "random_focus") & (L.appid == a)]) for a, g in GAMES.items()]:
    n = len(sub)
    for f in FIDS:
        k = int(sub[f].sum())
        if k == 0: continue
        lo, hi = proportion_confint(k, n, alpha=0.10, method="wilson")
        rows.append({"scope": scope, "factor": f, "label": cb.get(f, f), "n": n, "k": k, "prevalence": k / n, "lo90": lo, "hi90": hi})
P = pd.DataFrame(rows); P.to_csv(f"{ROOT}/results/sample_tier_prevalence.csv", index=False)
# also review-level shares
lvl = L[L.source.isin(["random_focus", "random_all"])].assign(neg=lambda d: d.polarity.isin(["negative", "mixed"]))
print("share really negative/mixed:", round(lvl.neg.mean(), 3), "| attributable:", round(lvl.attributable.mean(), 3),
      "| self-reported quit:", round(lvl.quit.mean(), 3), "| refund:", round(lvl.refund.mean(), 3))

con = duckdb.connect()
beh = con.execute(f"SELECT rid, ret_d30, hours_at_review, year(to_timestamp(created)) yr FROM '{ROOT}/data/neg_behavior.parquet' WHERE rid IN ({','.join(map(str, lvl.rid))})").df()
D = lvl.merge(beh, on="rid").dropna(subset=["ret_d30"])
D["game"] = np.where(D.appid.isin(GAMES), D.appid.map(GAMES), "other")
use = [f for f in FIDS if D[f].sum() >= 25]
X = pd.concat([D[use].astype(float), np.log1p(D.hours_at_review).rename("log_hours"),
               pd.get_dummies(D.game, prefix="g", drop_first=True, dtype=float), pd.get_dummies(D.yr, prefix="y", drop_first=True, dtype=float)], axis=1)
m = sm.OLS(D.ret_d30.astype(float), sm.add_constant(X)).fit(cov_type="HC1")
E = pd.DataFrame({"factor": use, "label": [cb.get(f, f) for f in use], "n_pos": [int(D[f].sum()) for f in use],
                  "beta_pp": [m.params[f] * 100 for f in use], "lo90": [(m.params[f] - 1.645 * m.bse[f]) * 100 for f in use],
                  "hi90": [(m.params[f] + 1.645 * m.bse[f]) * 100 for f in use], "p": [m.pvalues[f] for f in use]})
E.sort_values("beta_pp").to_csv(f"{ROOT}/results/sample_tier_effects.csv", index=False)
print("n =", len(D), "| factors with >=25 positives:", len(use))
print(E.sort_values("beta_pp").round(3).to_string(index=False))
top = P[P.scope != "All games (random sample)"].sort_values(["scope", "prevalence"], ascending=[True, False]).groupby("scope").head(6)
print(top[["scope", "label", "k", "n", "prevalence", "lo90", "hi90"]].round(3).to_string(index=False))
