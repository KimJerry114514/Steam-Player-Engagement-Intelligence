"""Resumable e5-base embedding of all negative reviews. Focus games first, then the rest.
Output: gpu_work/out/{focus,rest}/chunk_XXXX_{emb.npy (float16, L2-normalized), rid.npy}
"""
import os, time, numpy as np, pyarrow.parquet as pq, torch
from sentence_transformers import SentenceTransformer

BASE = "gpu_work"
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "embed_progress.log")


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


model = SentenceTransformer("intfloat/multilingual-e5-base", device="cuda",
                            model_kwargs={"torch_dtype": torch.float16})
model.max_seq_length = 256


def embed_file(path, outdir, skip_rids=None):
    os.makedirs(outdir, exist_ok=True)
    pf = pq.ParquetFile(path)
    for i in range(pf.num_row_groups):
        tag = os.path.join(outdir, f"chunk_{i:04d}")
        if os.path.exists(tag + "_emb.npy") and os.path.exists(tag + "_rid.npy"):
            continue
        t = pf.read_row_group(i, columns=["rid", "text"])
        rid = np.asarray(t.column("rid").to_numpy(), dtype=np.int64)
        txt = t.column("text").to_pylist()
        if skip_rids is not None:
            keep = ~np.isin(rid, skip_rids)
            rid = rid[keep]
            txt = [x for x, k in zip(txt, keep) if k]
        t0 = time.time()
        if len(rid):
            emb = model.encode(["passage: " + (x or "") for x in txt], batch_size=256,
                               normalize_embeddings=True, convert_to_numpy=True,
                               show_progress_bar=False).astype(np.float16)
        else:
            emb = np.zeros((0, 768), dtype=np.float16)
        np.save(tag + "_emb.tmp.npy", emb)
        np.save(tag + "_rid.tmp.npy", rid)
        os.replace(tag + "_emb.tmp.npy", tag + "_emb.npy")
        os.replace(tag + "_rid.tmp.npy", tag + "_rid.npy")
        dt = time.time() - t0
        log(f"{os.path.basename(outdir)} rg {i + 1}/{pf.num_row_groups} rows {len(rid)} "
            f"secs {dt:.1f} rate {len(rid) / max(dt, 1e-6):.0f}/s")


log("START")
embed_file(os.path.join(BASE, "data", "neg_text_focus.parquet"), os.path.join(OUT, "focus"))
log("FOCUS DONE")
focus_rids = np.sort(np.asarray(
    pq.read_table(os.path.join(BASE, "data", "neg_text_focus.parquet"), columns=["rid"]).column("rid").to_numpy(),
    dtype=np.int64))
sample_path = os.path.join(BASE, "data", "neg_text_sample1m.parquet")
skip = focus_rids
if os.path.exists(sample_path):  # representative 1M sample of all other games first, so all-game analysis is not blocked
    embed_file(sample_path, os.path.join(OUT, "sample"))
    log("SAMPLE DONE")
    skip = np.union1d(focus_rids, np.asarray(pq.read_table(sample_path, columns=["rid"]).column("rid").to_numpy(), dtype=np.int64))
embed_file(os.path.join(BASE, "data", "neg_text.parquet"), os.path.join(OUT, "rest"), skip_rids=skip)
log("ALL DONE")
