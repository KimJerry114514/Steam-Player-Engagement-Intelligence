"""Cluster negative-review embeddings of each focus game (full population, not a sample) for bottom-up factor discovery.
Input: out/focus/chunk_*  (all 2.19M focus embeddings), data/focus_rid_app.parquet (rid, appid)
Output: model/clusters/<appid>.json : clusters with size, share, 15 reps nearest the centroid, 10 random members
        model/clusters/<appid>_assign.parquet : rid -> cluster id
"""
import os, glob, json, numpy as np, pyarrow.parquet as pq, pyarrow as pa
from sklearn.cluster import MiniBatchKMeans
BASE = "gpu_work"; OUT = os.path.join(BASE, "model", "clusters"); os.makedirs(OUT, exist_ok=True)
app = pq.read_table(os.path.join(BASE, "data", "focus_rid_app.parquet")).to_pandas().set_index("rid").appid
R, E = [], []
for rf in sorted(glob.glob(os.path.join(BASE, "out", "focus", "chunk_*_rid.npy"))):
    R.append(np.load(rf)); E.append(np.load(rf.replace("_rid.npy", "_emb.npy")))
R = np.concatenate(R); E = np.concatenate(E)
A = app.reindex(R).values
K = {730: 80, 578080: 80, 1172470: 60, 1091500: 60, 105600: 40}
rng = np.random.default_rng(20260927)
for a, k in K.items():
    m = np.where(A == a)[0]; r = R[m]
    fit_idx = np.sort(rng.choice(len(r), size=min(len(r), 250000), replace=False))
    km = MiniBatchKMeans(n_clusters=k, batch_size=8192, n_init=3, random_state=20260927).fit(E[m[fit_idx]].astype(np.float32))
    C = km.cluster_centers_.astype(np.float32); lab = np.empty(len(r), np.int32); d = np.empty(len(r), np.float32)
    for s0 in range(0, len(r), 100000):
        x = E[m[s0:s0 + 100000]].astype(np.float32)
        dist = (x * x).sum(1)[:, None] - 2 * x @ C.T + (C * C).sum(1)[None, :]
        lab[s0:s0 + len(x)] = dist.argmin(1); d[s0:s0 + len(x)] = dist.min(1)
    cl = []
    for c in range(k):
        idx = np.where(lab == c)[0]
        if not len(idx): continue
        near = idx[np.argsort(d[idx])[:15]]; rand = rng.choice(idx, size=min(10, len(idx)), replace=False)
        cl.append({"cluster": int(c), "size": int(len(idx)), "share": float(len(idx) / len(r)),
                   "rep_rids": [int(x) for x in r[near]], "random_rids": [int(x) for x in r[rand]]})
    json.dump({"appid": a, "n": int(len(r)), "k": k, "clusters": cl}, open(os.path.join(OUT, f"{a}.json"), "w"))
    pq.write_table(pa.table({"rid": pa.array(r), "cluster": pa.array(lab.astype(np.int16))}), os.path.join(OUT, f"{a}_assign.parquet"))
    print(a, len(r), "clusters", len(cl), flush=True)
