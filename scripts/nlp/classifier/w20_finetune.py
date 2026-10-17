"""Fine-tune multilingual-e5-base end-to-end as a multi-label factor classifier on the LLM labels, then score reviews.

Train on split=='train' (a 12% validation slice is carved out by hash for epoch selection and per-factor thresholds);
evaluate on split=='test' (includes all human-gold items). Factors with < MIN_POS training positives are not modeled.
Outputs (gpu_work/ft/): metrics.json, thresholds.json, test_predictions.parquet, scores/<set>_<i>.parquet (uint8 probs)
Usage: python w20_finetune.py [score sets...]   (sets: focus sample rest)
"""
import os, sys, json, time, math, hashlib, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

BASE = "gpu_work"; OUT = os.path.join(BASE, os.environ.get("FT_DIR", "ft")); os.makedirs(os.path.join(OUT, "scores"), exist_ok=True)
MODEL = os.environ.get("FT_MODEL", "intfloat/multilingual-e5-base"); MIN_POS = 8; EPOCHS = int(os.environ.get("EPOCHS", 6)); LR = float(os.environ.get("LR", 3e-5))
BS = int(os.environ.get("BS", 32)); MAXLEN_TRAIN = int(os.environ.get("MAXLEN_TRAIN", 256)); MAXLEN_INFER = int(os.environ.get("MAXLEN_INFER", 160)); FREEZE = int(os.environ.get("FREEZE_LAYERS", 0))
ITEMS = os.environ.get("ITEMS", "label_items.json"); LABELS = os.environ.get("LABELS", "labels_train.parquet")
torch.manual_seed(20260927); np.random.seed(20260927)
dev = "cuda"
lab = pq.read_table(os.path.join(BASE, "data", LABELS)).to_pandas()
items = {x["rid"]: x["text"] for x in json.load(open(os.path.join(BASE, "data", ITEMS), encoding="utf-8"))}
lab = lab[lab.rid.isin(items)].reset_index(drop=True)
factors = [c for c in lab.columns if c not in ("rid", "split") and c != "other" and lab.loc[lab.split == "train", c].sum() >= MIN_POS]
is_val = lab.rid.map(lambda r: int(hashlib.md5(f"v{r}".encode()).hexdigest(), 16) % 100 < 12).values
tr = (lab.split.values == "train") & ~is_val; va = (lab.split.values == "train") & is_val; te = lab.split.values == "test"
Y = lab[factors].values.astype(np.float32); T = ["passage: " + (items[r] or "") for r in lab.rid]
print(f"factors {len(factors)} | train {tr.sum()} val {va.sum()} test {te.sum()}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)


class Clf(nn.Module):
    def __init__(self, n):
        super().__init__(); self.enc = AutoModel.from_pretrained(MODEL); self.drop = nn.Dropout(0.1); self.head = nn.Linear(self.enc.config.hidden_size, n)
    def forward(self, ids, att):
        h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        m = att.unsqueeze(-1).float(); pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.drop(pooled))


def batches(idx, bs, maxlen, shuffle):
    idx = np.array(idx); idx = np.random.permutation(idx) if shuffle else idx
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]; enc = tok([T[i] for i in b], truncation=True, max_length=maxlen, padding=True, return_tensors="pt")
        yield b, enc["input_ids"].to(dev), enc["attention_mask"].to(dev)


def predict(model, idx, maxlen=MAXLEN_TRAIN):
    model.eval(); out = np.zeros((len(idx), len(factors)), np.float32); pos = {v: k for k, v in enumerate(idx)}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for b, ids, att in batches(idx, 128, maxlen, False):
            p = torch.sigmoid(model(ids, att).float()).cpu().numpy()
            for j, i in enumerate(b): out[pos[i]] = p[j]
    return out


SCORE_ONLY = os.environ.get("SCORE_ONLY") == "1"
if SCORE_ONLY:
    factors = json.load(open(os.path.join(OUT, "thresholds.json")))["factors"]
    model = Clf(len(factors)).to(dev); model.load_state_dict(torch.load(os.path.join(OUT, "model.pt"), map_location=dev)); model.half()
    print("score-only: loaded", OUT, len(factors), "factors", flush=True)
else:
    model = Clf(len(factors)).to(dev)
    if FREEZE:
        for p_ in model.enc.embeddings.parameters(): p_.requires_grad = False
        for layer in model.enc.encoder.layer[:FREEZE]:
            for p_ in layer.parameters(): p_.requires_grad = False
        model.enc.gradient_checkpointing_enable()
    print("model", MODEL, "trainable params", sum(p_.numel() for p_ in model.parameters() if p_.requires_grad), flush=True)
    pos = Y[tr].sum(0); pw = torch.tensor(np.clip((tr.sum() - pos) / np.maximum(pos, 1), 1, 30) ** 0.5, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW([{"params": [p_ for p_ in model.enc.parameters() if p_.requires_grad], "lr": LR}, {"params": model.head.parameters(), "lr": LR * 20}], weight_decay=0.01)
    steps = EPOCHS * math.ceil(tr.sum() / BS); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR, LR * 20], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    tr_idx, va_idx, te_idx = np.where(tr)[0], np.where(va)[0], np.where(te)[0]
    best, best_state = -1, None
    for ep in range(EPOCHS):
        model.train(); t0 = time.time(); tot = 0
        for b, ids, att in batches(tr_idx, BS, MAXLEN_TRAIN, True):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(model(ids, att).float(), torch.tensor(Y[b], device=dev))
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()
        pv = predict(model, va_idx)
        f1s = [precision_recall_fscore_support(Y[va_idx, j], pv[:, j] >= 0.5, average="binary", zero_division=0)[2] for j in range(len(factors))]
        auprc = np.nanmean([average_precision_score(Y[va_idx, j], pv[:, j]) if Y[va_idx, j].sum() else np.nan for j in range(len(factors))])
        print(f"epoch {ep + 1} loss {tot:.1f} val macroF1@0.5 {np.mean(f1s):.3f} val mean AUPRC {auprc:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if auprc > best: best, best_state = auprc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pv = predict(model, va_idx); grid = np.linspace(0.05, 0.95, 91); thr = {}
    for j, f in enumerate(factors):
        if Y[va_idx, j].sum() == 0: thr[f] = 0.5; continue
        f1s = [precision_recall_fscore_support(Y[va_idx, j], pv[:, j] >= t, average="binary", zero_division=0)[2] for t in grid]
        thr[f] = float(grid[int(np.argmax(f1s))])
    pt = predict(model, te_idx); metrics = {}
    for j, f in enumerate(factors):
        y = Y[te_idx, j]; p, r, f1, _ = precision_recall_fscore_support(y, pt[:, j] >= thr[f], average="binary", zero_division=0)
        metrics[f] = {"n_pos_train": int(Y[tr_idx, j].sum()), "n_pos_test": int(y.sum()), "threshold": thr[f], "test_precision": float(p), "test_recall": float(r),
                      "test_f1": float(f1), "test_auprc": float(average_precision_score(y, pt[:, j])) if y.sum() else None}
    json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
    json.dump({"factors": factors, "thresholds": thr}, open(os.path.join(OUT, "thresholds.json"), "w"), indent=1)
    cols = {"rid": pa.array(lab.rid.values[te_idx])}; [cols.__setitem__(f, pa.array(pt[:, j])) for j, f in enumerate(factors)]
    pq.write_table(pa.table(cols), os.path.join(OUT, "test_predictions.parquet"))
    ws = [m["test_f1"] for m in metrics.values() if m["n_pos_test"]]; sup = [m["n_pos_test"] for m in metrics.values() if m["n_pos_test"]]
    print("TEST median F1", round(float(np.median(ws)), 3), "support-weighted F1", round(float(np.average(ws, weights=sup)), 3), flush=True)
    torch.save(model.state_dict(), os.path.join(OUT, "model.pt"))

# ---- scoring ----
SETS = {"focus": "neg_text_focus.parquet", "sample": "neg_text_sample1m.parquet", "rest": "neg_text.parquet"}
for s in sys.argv[1:]:
    pf = pq.ParquetFile(os.path.join(BASE, "data", SETS[s])); model.eval()
    for i in range(pf.num_row_groups):
        outp = os.path.join(OUT, "scores", f"{s}_{i:04d}.parquet")
        if os.path.exists(outp): continue
        t0 = time.time(); t = pf.read_row_group(i, columns=["rid", "text"]); rid = t.column("rid").to_numpy(); txt = t.column("text").to_pylist()
        order = np.argsort([len(x or "") for x in txt]); probs = np.zeros((len(txt), len(factors)), np.uint8)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for k in range(0, len(order), 256):
                b = order[k:k + 256]; enc = tok(["passage: " + (txt[j] or "") for j in b], truncation=True, max_length=MAXLEN_INFER, padding=True, return_tensors="pt")
                p = torch.sigmoid(model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev)).float()).cpu().numpy()
                probs[b] = np.round(p * 255).astype(np.uint8)
        c = {"rid": pa.array(rid)}; [c.__setitem__(f, pa.array(probs[:, j])) for j, f in enumerate(factors)]
        pq.write_table(pa.table(c), outp + ".tmp", compression="zstd"); os.replace(outp + ".tmp", outp)
        print(f"scored {s} rg {i + 1}/{pf.num_row_groups} rows {len(rid)} {len(rid) / (time.time() - t0):.0f}/s", flush=True)
print("DONE", flush=True)
