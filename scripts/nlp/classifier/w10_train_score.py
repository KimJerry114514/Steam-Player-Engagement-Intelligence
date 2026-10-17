"""Train per-factor classifiers on frozen e5-base embeddings using LLM (LLM) labels; score all negative reviews.

Inputs (gpu_work):
  data/labels_train.parquet : rid, split ('train'|'test'), then one int8 column per factor id (0/1)
  out/focus/*, out/rest/*    : chunk_XXXX_emb.npy (float16, L2-normalized) + chunk_XXXX_rid.npy
Outputs:
  model/metrics.json         : per-factor n_pos, CV/test precision, recall, F1, AUPRC, chosen threshold
  model/scores/part_XXXX.parquet : rid + uint8 probability (0-255) per factor for every embedded review
Method: one-vs-rest logistic regression (class_weight=balanced), threshold chosen on 5-fold CV of the train
split to maximize F1; frozen before scoring the test split and the full corpus.
"""
import os, sys, json, glob, time, numpy as np, pyarrow as pa, pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

BASE = "gpu_work"
MODEL = os.environ.get("MODEL_DIR", "model")
os.makedirs(os.path.join(BASE, MODEL, "scores"), exist_ok=True)
lab = pq.read_table(os.path.join(BASE, "data", "labels_train.parquet")).to_pandas()
factors = [c for c in lab.columns if c not in ("rid", "split")]

# load embeddings for labeled rids
SCORE_DIRS = sys.argv[1:] or ["focus", "sample"]
chunks = sorted(glob.glob(os.path.join(BASE, "out", "labeled", "chunk_*_rid.npy")))  # all labeled items live here
need = set(lab.rid.tolist()); emb_map = {}
for rf in chunks:
    r = np.load(rf); m = np.isin(r, list(need))
    if m.any():
        e = np.load(rf.replace("_rid.npy", "_emb.npy"))
        for rid, v in zip(r[m], e[m]): emb_map[int(rid)] = v.astype(np.float32)
lab = lab[lab.rid.isin(emb_map)].reset_index(drop=True)
X = np.stack([emb_map[r] for r in lab.rid]); tr = (lab.split == "train").values; te = ~tr
print("labeled with embeddings:", len(lab), "train", tr.sum(), "test", te.sum(), flush=True)

models, metrics = {}, {}
for f in factors:
    y = lab[f].values.astype(int); npos = int(y[tr].sum())
    if npos < 8:
        metrics[f] = {"n_pos_train": npos, "status": "too_few_positives"}; continue
    clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260927)
    p_cv = cross_val_predict(clf, X[tr], y[tr], cv=cv, method="predict_proba")[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [precision_recall_fscore_support(y[tr], p_cv >= t, average="binary", zero_division=0)[2] for t in grid]
    thr = float(grid[int(np.argmax(f1s))])
    clf.fit(X[tr], y[tr]); models[f] = (clf, thr)
    p_te = clf.predict_proba(X[te])[:, 1] if te.any() else np.array([])
    pr, rc, f1, _ = precision_recall_fscore_support(y[te], p_te >= thr, average="binary", zero_division=0) if te.any() else (0, 0, 0, 0)
    metrics[f] = {"n_pos_train": npos, "n_pos_test": int(y[te].sum()), "threshold": thr,
                  "cv_f1": float(max(f1s)), "cv_auprc": float(average_precision_score(y[tr], p_cv)),
                  "test_precision": float(pr), "test_recall": float(rc), "test_f1": float(f1),
                  "test_auprc": float(average_precision_score(y[te], p_te)) if te.any() and y[te].sum() else None}
    print(f, metrics[f], flush=True)
json.dump(metrics, open(os.path.join(BASE, MODEL, "metrics.json"), "w"), indent=1)
# save held-out predictions (incl. human-gold items) for evaluation against human labels on the Mac
tp = {"rid": pa.array(lab.rid.values[te])}
for f, (clf, thr) in models.items(): tp[f] = pa.array(clf.predict_proba(X[te])[:, 1].astype(np.float32))
pq.write_table(pa.table(tp), os.path.join(BASE, MODEL, "test_predictions.parquet"))

# score the full corpus
W = np.stack([models[f][0].coef_[0] for f in models]).astype(np.float32)
b = np.array([models[f][0].intercept_[0] for f in models], dtype=np.float32)
names = list(models)
t0 = time.time(); total = 0
score_chunks = [c for d in SCORE_DIRS for c in sorted(glob.glob(os.path.join(BASE, "out", d, "chunk_*_rid.npy")))]
for i, rf in enumerate(score_chunks):
    tag = os.path.basename(os.path.dirname(rf)) + "_" + os.path.basename(rf).replace("_rid.npy", "")
    outp = os.path.join(BASE, MODEL, "scores", f"{tag}.parquet")
    if os.path.exists(outp): continue
    r = np.load(rf); e = np.load(rf.replace("_rid.npy", "_emb.npy")).astype(np.float32)
    p = 1 / (1 + np.exp(-(e @ W.T + b)))
    cols = {"rid": pa.array(r)}
    for j, f in enumerate(names): cols[f] = pa.array(np.round(p[:, j] * 255).astype(np.uint8))
    pq.write_table(pa.table(cols), outp, compression="zstd"); total += len(r)
print("scored", total, "rows in", round(time.time() - t0), "s", flush=True)
json.dump({"factors": names, "thresholds": {f: models[f][1] for f in names}}, open(os.path.join(BASE, MODEL, "thresholds.json"), "w"), indent=1)
