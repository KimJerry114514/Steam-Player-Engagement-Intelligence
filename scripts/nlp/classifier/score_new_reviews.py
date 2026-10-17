"""Score review texts with the trained complaint-factor classifier (ft_large_v2) and write the same NLP columns as
the team file nlp_factor_scores_2023.parquet: n_factors_flagged, p_<factor>, flag_<factor> for the 35 reliable factors.

Usage:
    python score_new_reviews.py reviews.csv scored.parquet                 # text column "review" (or --text-col)
    python score_new_reviews.py reviews.parquet scored.parquet --all-64    # also keep the 29 less reliable factors

Input: a CSV or Parquet file with a text column; every other column (e.g. appid, recommendationid) is copied through.
Needs: torch, transformers, pandas, pyarrow. A CUDA GPU is fastest; Apple MPS and CPU also work (slower).
Default assets: models/nlp/model.pt, models/nlp/ft_large_v2_thresholds.json, and results/nlp/ft_large_v2_metrics.json.
The tokenizer and encoder configuration are included in models/nlp/tokenizer/; inference needs no network.
Paths are resolved from this script, so the command works from any working directory.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_MODEL = "intfloat/multilingual-e5-large"
TOKENIZER_DIR = PROJECT_ROOT / "models/nlp/tokenizer"
MAXLEN = 128          # tokens per review at scoring time (same as the run that produced the team file)
TAG = re.compile(r"\[/?[a-zA-Z0-9*]+(=[^\]]*)?\]")


def clean(text):
    """Same cleaning as 01b_build_text_files.sql: drop Steam BBCode tags, collapse spaces, keep 1,500 characters."""
    return re.sub(r"\s+", " ", TAG.sub(" ", text or "")).strip()[:1500]


class Clf(nn.Module):
    """multilingual-e5-large encoder -> mean pooling -> one linear output per factor (as in w20_finetune.py)."""
    def __init__(self, n):
        super().__init__()
        self.enc = AutoModel.from_config(AutoConfig.from_pretrained(TOKENIZER_DIR, local_files_only=True))   # weights come from model.pt
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(self.enc.config.hidden_size, n)

    def forward(self, ids, att):
        h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        m = att.unsqueeze(-1).float()
        return self.head(self.drop((h * m).sum(1) / m.sum(1).clamp(min=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--text-col", default="review")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--all-64", action="store_true", help="also output the 29 factors that did not pass the F1 >= 0.40 rule")
    ap.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models/nlp/model.pt")
    ap.add_argument("--thresholds", type=Path, default=PROJECT_ROOT / "models/nlp/ft_large_v2_thresholds.json")
    ap.add_argument("--metrics", type=Path, default=PROJECT_ROOT / "results/nlp/ft_large_v2_metrics.json")
    a = ap.parse_args()
    if a.batch < 1:
        ap.error("--batch must be positive")
    for asset in (a.model_path, a.thresholds, a.metrics,
                  TOKENIZER_DIR / "config.json", TOKENIZER_DIR / "tokenizer.json",
                  TOKENIZER_DIR / "tokenizer_config.json"):
        if not asset.is_file():
            ap.error(f"Required asset not found: {asset}")
    if Path(a.input).resolve() == Path(a.output).resolve():
        ap.error("Input and output must be different files")

    df = pd.read_parquet(a.input) if a.input.endswith(".parquet") else pd.read_csv(a.input)
    texts = ["passage: " + clean(t) for t in df[a.text_col].astype(str).where(df[a.text_col].notna(), "")]

    with a.thresholds.open(encoding="utf-8") as handle:
        th = json.load(handle)
    with a.metrics.open(encoding="utf-8") as handle:
        met = json.load(handle)
    factors = th["factors"]
    keep = factors if a.all_64 else [f for f in factors if met[f]["test_f1"] >= 0.40 and met[f]["n_pos_test"] >= 5]

    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    model = Clf(len(factors))
    model.load_state_dict(torch.load(a.model_path, map_location="cpu", weights_only=True))
    model.to(dev).eval()
    if dev == "cuda":
        model.half()

    scores = np.zeros((len(texts), len(factors)), np.uint8)     # 0-255, as stored for the team file
    order = np.argsort([len(t) for t in texts])                  # similar lengths per batch = faster
    with torch.no_grad():
        for k in range(0, len(order), a.batch):
            b = order[k:k + a.batch]
            enc = tok([texts[j] for j in b], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
            p = torch.sigmoid(model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev)).float()).cpu().numpy()
            scores[b] = np.round(p * 255).astype(np.uint8)
            print(f"\rscored {min(k + a.batch, len(order)):,} / {len(order):,}", end="", flush=True)
    print()

    out = df.drop(columns=[a.text_col]).copy()
    col = {f: f.replace(".", "_") for f in factors}
    idx = {f: j for j, f in enumerate(factors)}
    flags = {f: (scores[:, idx[f]] >= int(round(th["thresholds"][f] * 255))).astype(np.int8) for f in keep}
    out["n_factors_flagged"] = np.sum([flags[f] for f in keep], axis=0).astype(np.int16)
    for f in keep:
        out["p_" + col[f]] = np.round(scores[:, idx[f]] / 255.0, 3).astype(np.float32)
    for f in keep:
        out["flag_" + col[f]] = flags[f]
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.output, index=False) if a.output.endswith(".parquet") else out.to_csv(a.output, index=False)
    print(f"wrote {a.output}: {len(out):,} reviews, {len(keep)} factors, device {dev}")


if __name__ == "__main__":
    main()
