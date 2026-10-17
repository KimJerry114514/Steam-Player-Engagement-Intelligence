"""Temporal out-of-sample check for the factor measurement (frozen-embedding probe, same protocol as w10_train_score.py).
Round-1 labels (3,730 reviews) with e5-base passage embeddings (results/timesplit/chunk_0000_*.npy).
- temporal split: train on reviews created before 2022-01-01, test on 2022-2023 reviews
- random splits: same train/test sizes, 5 seeds
Per factor: logistic regression (C=2, balanced), threshold chosen by 5-fold CV on the training part to maximise F1;
factors need >= 8 training positives. Metric: support-weighted F1 over factors with test positives.
Output: results/timesplit/timesplit_metrics.json"""
import json, numpy as np, pandas as pd, duckdb
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_recall_fscore_support
ROOT = "."
rid = np.load(f"{ROOT}/results/timesplit/chunk_0000_rid.npy"); X = np.load(f"{ROOT}/results/timesplit/chunk_0000_emb.npy").astype(np.float32)
L = pd.read_parquet(f"{ROOT}/results/labels/final/labels_primary.parquet").set_index("rid").loc[rid]
FIDS = [c for c in L.columns if "." in c]
yr = duckdb.connect().execute(f"SELECT rid, year(to_timestamp(created)) yr FROM '{ROOT}/data/neg_all.parquet' WHERE rid IN ({','.join(map(str, rid))})").df().set_index("rid").yr.loc[rid].values
GRID = np.linspace(0.05, 0.95, 91)


def one(f, tr, te):
    y = L[f].values.astype(int)
    if y[tr].sum() < 8 or y[te].sum() == 0: return None
    clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000)
    p_cv = cross_val_predict(clf, X[tr], y[tr], cv=StratifiedKFold(5, shuffle=True, random_state=20260927), method="predict_proba")[:, 1]
    thr = GRID[int(np.argmax([precision_recall_fscore_support(y[tr], p_cv >= t, average="binary", zero_division=0)[2] for t in GRID]))]
    clf.fit(X[tr], y[tr])
    return f, float(precision_recall_fscore_support(y[te], clf.predict_proba(X[te])[:, 1] >= thr, average="binary", zero_division=0)[2]), int(y[te].sum())


def wf1(tr, te):
    r = [x for x in Parallel(n_jobs=8)(delayed(one)(f, tr, te) for f in FIDS) if x]
    return float(np.average([x[1] for x in r], weights=[x[2] for x in r])), {x[0]: x[1] for x in r}, len(r)


tr_t = yr < 2022; te_t = ~tr_t
t_f1, t_per, t_n = wf1(tr_t, te_t)
rand = []
for seed in range(5):
    idx = np.random.default_rng(seed).permutation(len(rid)); tr = np.zeros(len(rid), bool); tr[idx[:tr_t.sum()]] = True
    rand.append(wf1(tr, ~tr)[0])
out = {"n_train_pre2022": int(tr_t.sum()), "n_test_2022_2023": int(te_t.sum()), "temporal_weighted_f1": t_f1, "temporal_n_factors": t_n,
       "random_split_weighted_f1_mean": float(np.mean(rand)), "random_split_weighted_f1_each": rand, "temporal_per_factor_f1": t_per}
json.dump(out, open(f"{ROOT}/results/timesplit/timesplit_metrics.json", "w"), indent=1)
print({k: v for k, v in out.items() if k != "temporal_per_factor_f1"})
