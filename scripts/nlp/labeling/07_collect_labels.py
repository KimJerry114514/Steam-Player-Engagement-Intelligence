"""Collect LLM labels, measure reliability, build the classifier training table.

Outputs (results/labels/):
  labels_primary.parquet     one row per labeled review (model_a coder), factor multi-hot + review-level fields
  agreement_terra_luna.csv   per-factor agreement on the 20% double-coded batches (kappa, positive-agreement F1)
  gold_B_vs_llm.csv        per-factor precision/recall/F1 of LLM vs the human gold B (when available)
  gold_fields_vs_llm.csv   polarity / attributable / quit / refund agreement vs human gold A+B
  prevalence.csv             factor prevalence in the representative random samples (per focus game and all-games)
  new_factor_candidates.csv  'other' reasons proposed by the coder (active discovery input)
  labels_train.parquet       rid, split, factor columns -> input of 4_classifier/w10_train_score.py
"""
import json, glob, os, hashlib, collections, numpy as np, pandas as pd
ROOT = "."
OUT = f"{ROOT}/results/labels"; os.makedirs(OUT, exist_ok=True)
cb = json.load(open(f"{ROOT}/llm_jobs/out/revise/codebook_v1_1.json"))
FIDS = [f["id"] for f in cb["factors"]]
items = {x["rid"]: x for x in json.load(open(f"{ROOT}/labels/llm_label/items.json"))}
import sys
APPLY_RECALL = "--recall" in sys.argv
OUT = OUT + ("/final" if APPLY_RECALL else "/pass1"); os.makedirs(OUT, exist_ok=True)


def load(stage):
    rows, unknown = [], collections.Counter()
    for f in sorted(glob.glob(f"{ROOT}/llm_jobs/out/{stage}/*.json")):
        rec = {}
        rp = f.replace(f"/out/{stage}/", "/out/recall/")
        if stage == "label" and APPLY_RECALL and os.path.exists(rp):
            rec = {x["rid"]: x for x in json.load(open(rp))["reviews"]}
        for r in json.load(open(f))["reviews"]:
            if r["rid"] in rec:
                rm = set(rec[r["rid"]]["remove"])
                r["factors"] = [x for x in r["factors"] if x["id"] not in rm] + [x for x in rec[r["rid"]]["add"] if x["id"] not in {y["id"] for y in r["factors"]}]
            if r["rid"] not in items: continue
            ids = []
            for x in r["factors"]:
                if x["id"] in FIDS or x["id"] == "other": ids.append(x["id"])
                else: unknown[x["id"]] += 1
            rows.append({"rid": r["rid"], "factors": sorted(set(ids)), "primary": r["primary_factor"], "polarity": r["content_polarity"],
                         "attributable": r["attributable"], "quit": r["self_reported_quit"], "refund": r["refund_intent"],
                         "new_candidate": r["new_factor_candidate"], "confidence": r["confidence"],
                         "evidence": {x["id"]: x["evidence"] for x in r["factors"]}})
    df = pd.DataFrame(rows).drop_duplicates("rid")
    return df, unknown


def multihot(df):
    m = pd.DataFrame({f: df.factors.apply(lambda s, f=f: int(f in s)) for f in FIDS + ["other"]})
    return pd.concat([df[["rid"]].reset_index(drop=True), m.reset_index(drop=True)], axis=1)


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b); po = (a == b).mean(); pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def prf(pred, gold):
    pred, gold = np.asarray(pred), np.asarray(gold); tp = (pred & gold).sum()
    p = tp / pred.sum() if pred.sum() else np.nan; r = tp / gold.sum() if gold.sum() else np.nan
    return p, r, (2 * p * r / (p + r) if p and r and (p + r) else np.nan)


L1, unk1 = load("label")
print("primary labels", len(L1), "unknown ids", unk1.most_common(10))
L1["source"] = L1.rid.map(lambda r: items[r].get("source")); L1["appid"] = L1.rid.map(lambda r: items[r]["appid"])
M1 = multihot(L1)
full = L1.drop(columns=["factors", "evidence"]).merge(M1, on="rid")
full.to_parquet(f"{OUT}/labels_primary.parquet")

# agreement model_a vs model_b
L2, unk2 = load("label2")
if len(L2):
    M2 = multihot(L2).set_index("rid"); A1 = M1.set_index("rid").loc[M2.index.intersection(M1.rid)]; M2 = M2.loc[A1.index]
    rows = []
    for f in FIDS + ["other"]:
        a, b = A1[f].values.astype(bool), M2[f].values.astype(bool)
        if a.sum() + b.sum() == 0: continue
        rows.append({"factor": f, "n_both_coded": len(a), "pos_terra": int(a.sum()), "pos_luna": int(b.sum()),
                     "kappa": kappa(a, b), "pos_agreement_f1": prf(a, b)[2]})
    ag = pd.DataFrame(rows).sort_values("pos_terra", ascending=False); ag.to_csv(f"{OUT}/agreement_terra_luna.csv", index=False)
    both = L1.set_index("rid").loc[A1.index]; two = L2.set_index("rid").loc[A1.index]
    print("double-coded", len(A1), "| exact factor-set match", round((both.factors.apply(tuple) == two.factors.apply(tuple)).mean(), 3),
          "| polarity agreement", round((both.polarity == two.polarity).mean(), 3),
          "| micro pos-agreement", round(prf(A1[FIDS].values.astype(bool).ravel(), M2[FIDS].values.astype(bool).ravel())[2], 3))

# prevalence in representative random samples
prev = []
for scope, sub in [("all_games_random", full[full.source == "random_all"])] + \
        [(f"game_{a}", full[(full.source == "random_focus") & (full.appid == a)]) for a in (730, 578080, 1172470, 1091500, 105600)]:
    for f in FIDS + ["other"]:
        k = int(sub[f].sum()); n = len(sub)
        if n: prev.append({"scope": scope, "factor": f, "n": n, "k": k, "prevalence": k / n})
pd.DataFrame(prev).to_csv(f"{OUT}/prevalence.csv", index=False)
pd.DataFrame([{"rid": r.rid, "game": items[r.rid]["game"], "candidate": r.new_candidate} for r in L1.itertuples() if r.new_candidate]).to_csv(
    f"{OUT}/new_factor_candidates.csv", index=False)

# human gold comparisons
gf = []
for s in ("A", "B"):
    p = f"{ROOT}/labels/gold/gold_{s}_labels.json"
    if not os.path.exists(p): continue
    G = json.load(open(p)); gi = {x["gid"]: x for x in json.load(open(f"{ROOT}/labels/gold/gold_{s}_items.json"))}
    for gid, g in G.items():
        rid = gi[gid]["rid"]; c = L1[L1.rid == rid]
        if not len(c): continue
        c = c.iloc[0]
        gf.append({"set": s, "rid": rid, "h_pol": g["polarity"], "c_pol": c.polarity, "h_att": g.get("attributable") == "yes", "c_att": c.attributable,
                   "h_quit": g.get("self_reported_quit", False), "c_quit": c.quit, "h_refund": g.get("refund_intent", False), "c_refund": c.refund,
                   "h_factors": g.get("factors"), "c_factors": list(c.factors), "h_unsure": g.get("unsure", False)})
if gf:
    G = pd.DataFrame(gf); G.to_json(f"{OUT}/gold_vs_llm_rows.json", orient="records", force_ascii=False)
    summ = {"n": len(G), "polarity_exact": (G.h_pol == G.c_pol).mean(),
            "negative_vs_not": ((G.h_pol.isin(["negative", "mixed"])) == (G.c_pol.isin(["negative", "mixed"]))).mean(),
            "attributable": (G.h_att == G.c_att).mean(), "quit": (G.h_quit == G.c_quit).mean(), "refund": (G.h_refund == G.c_refund).mean()}
    pd.DataFrame([summ]).to_csv(f"{OUT}/gold_fields_vs_llm.csv", index=False); print("gold fields vs llm", {k: round(v, 3) for k, v in summ.items()})
    B = G[G.set == "B"]
    if len(B):
        rows = []
        for f in FIDS:
            h = B.h_factors.apply(lambda s, f=f: f in (s or [])).values; c = B.c_factors.apply(lambda s, f=f: f in s).values
            if h.sum() + c.sum() == 0: continue
            p, r, f1 = prf(c, h); rows.append({"factor": f, "n": len(B), "human_pos": int(h.sum()), "llm_pos": int(c.sum()), "precision": p, "recall": r, "f1": f1})
        gb = pd.DataFrame(rows).sort_values("human_pos", ascending=False); gb.to_csv(f"{OUT}/gold_B_vs_llm.csv", index=False)
        hh = np.concatenate([B.h_factors.apply(lambda s, f=f: f in (s or [])).values for f in FIDS]); cc = np.concatenate([B.c_factors.apply(lambda s, f=f: f in s).values for f in FIDS])
        print("gold B micro P/R/F1 llm vs human", [round(x, 3) for x in prf(cc, hh)])

# training table: gold items -> test; others hashed 80/20
gold_rids = {x["rid"] for s in ("A", "B") for x in json.load(open(f"{ROOT}/labels/gold/gold_{s}_items.json"))}
def split(r):
    if r in gold_rids: return "test"
    return "test" if int(hashlib.md5(str(r).encode()).hexdigest(), 16) % 5 == 0 else "train"
T = M1.copy(); T.insert(1, "split", T.rid.map(split))
T = T[T.rid.isin(L1[L1.polarity.isin(["negative", "mixed", "neutral", "positive"])].rid)]
for c in FIDS + ["other"]: T[c] = T[c].astype("int8")
T.to_parquet(f"{OUT}/labels_train.parquet", index=False)
print("train table", T.shape, T.split.value_counts().to_dict())
