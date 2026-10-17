"""Embed the LLM/gold labeling items right away (same model/settings as the corpus run) so classifier training
does not wait for the full-corpus embedding. Input: gpu_work/data/label_items.json (list with rid, text).
Output: gpu_work/out/labeled/chunk_0000_emb.npy + chunk_0000_rid.npy
"""
import os, json, numpy as np, torch
from sentence_transformers import SentenceTransformer
BASE = "gpu_work"
items = json.load(open(os.path.join(BASE, "data", "label_items.json"), encoding="utf-8"))
rid = np.array([x["rid"] for x in items], dtype=np.int64)
m = SentenceTransformer("intfloat/multilingual-e5-base", device="cuda", model_kwargs={"torch_dtype": torch.float16})
m.max_seq_length = 256
emb = m.encode(["passage: " + (x["text"] or "") for x in items], batch_size=128, normalize_embeddings=True,
               convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
os.makedirs(os.path.join(BASE, "out", "labeled"), exist_ok=True)
np.save(os.path.join(BASE, "out", "labeled", "chunk_0000_emb.npy"), emb)
np.save(os.path.join(BASE, "out", "labeled", "chunk_0000_rid.npy"), rid)
print("embedded", len(rid))
