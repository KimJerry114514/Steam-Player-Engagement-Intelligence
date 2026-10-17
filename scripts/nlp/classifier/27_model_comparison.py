"""(1) Classifier comparison on the same 802 round-1 test reviews, scored against the v2 labels, on the factors all four models
share (and on each model's own factor set). (2) Review-level shares from the LLM-labeled random samples, unweighted and
weighted to the population (5 deep-dive titles + all other games). Output: results/model_comparison.json"""
import json, numpy as np, pandas as pd, duckdb
from sklearn.metrics import f1_score
ROOT = "."
L = pd.read_parquet(f"{ROOT}/results/labels/v2/labels_train.parquet").set_index("rid")
MODELS = {"linear probe (frozen e5-base)": "model_pass1", "e5-base fine-tune": "ft", "e5-large, round-1 labels": "ft_large_v1", "e5-large, all labels (final)": "ft_large_v2"}
sub = pd.read_parquet(f"{ROOT}/results/ft_test_predictions.parquet").rid.values
P = {k: pd.read_parquet(f"{ROOT}/results/{t}_test_predictions.parquet").set_index("rid").loc[sub] for k, t in MODELS.items()}
TH = {k: json.load(open(f"{ROOT}/results/{t}_thresholds.json"))["thresholds"] for k, t in MODELS.items()}
usable = {k: {f for f in TH[k] if f in L.columns and f in P[k].columns and L.loc[sub, f].sum() > 0} for k in MODELS}
common = set.intersection(*usable.values())


def wf1(k, fs):
    fs = sorted(fs); y = [L.loc[sub, f].values.astype(int) for f in fs]
    return float(np.average([f1_score(yy, (P[k][f].values >= TH[k][f]).astype(int), zero_division=0) for yy, f in zip(y, fs)], weights=[yy.sum() for yy in y]))


comp = {k: {"common_factors_wf1": wf1(k, common), "own_factors_wf1": wf1(k, usable[k]), "n_own_factors": len(usable[k])} for k in MODELS}
R = pd.read_parquet(f"{ROOT}/results/labels/v2/labels_primary_random.parquet")
FOCUS = [730, 578080, 1172470, 1091500, 105600]
pop = duckdb.connect().execute(f"SELECT CASE WHEN appid IN ({','.join(map(str, FOCUS))}) THEN appid ELSE 0 END g, count(*) n FROM '{ROOT}/data/neg_all.parquet' GROUP BY 1").df().set_index("g").n
R["g"] = np.where(R.source == "random_all", 0, R.appid)
R["neg"] = R.polarity.isin(["negative", "mixed"]); R["att"] = R.attributable.astype(bool); R["q"] = R.quit.astype(bool); R["rf"] = R.refund.astype(bool)
w = R.g.map(pop / R.g.value_counts())
shares = {c: {"unweighted": float(R[c].mean()), "population_weighted": float(np.average(R[c], weights=w))} for c in ["neg", "att", "q", "rf"]}
out = {"n_test_reviews": int(len(sub)), "n_common_factors": len(common), "models": comp, "random_sample_n": int(len(R)),
       "random_all_in_focus_titles": int(((R.source == "random_all") & R.appid.isin(FOCUS)).sum()),
       "review_level_shares": {"negative_or_mixed": shares["neg"], "attributable": shares["att"], "self_reported_quit": shares["q"], "refund": shares["rf"]}}
json.dump(out, open(f"{ROOT}/results/model_comparison.json", "w"), indent=1)
print(json.dumps(out, indent=1))
