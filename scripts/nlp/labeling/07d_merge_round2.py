"""Merge round-1 (two-pass) and round-2 (active-learning, single pass with recall guidance) LLM labels into the
training table v2. Test split stays representative: round-1 test (random + human gold) unchanged; round-2 random items
are split 80/20 by hash; round-2 uncertainty / rare-factor selections go to train only.
Outputs results/labels/v2/labels_train.parquet and data/label_items_v2.json (review texts for the classifier)."""
import json, glob, os, hashlib, pandas as pd
ROOT = "."; OUT = f"{ROOT}/results/labels/v2"; os.makedirs(OUT, exist_ok=True)
cb = json.load(open(f"{ROOT}/llm_jobs/out/revise/codebook_v1_1.json")); FIDS = [f["id"] for f in cb["factors"]]
T1 = pd.read_parquet(f"{ROOT}/results/labels/final/labels_train.parquet")
items2 = {x["rid"]: x for x in json.load(open(f"{ROOT}/labels/llm_label2/items.json"))}
rows, unknown = [], 0
for f in sorted(glob.glob(f"{ROOT}/llm_jobs/out/label_al/*.json")):
    for r in json.load(open(f))["reviews"]:
        if r["rid"] not in items2: continue
        ids = {x["id"] for x in r["factors"] if x["id"] in FIDS or x["id"] == "other"}; unknown += sum(1 for x in r["factors"] if x["id"] not in FIDS and x["id"] != "other")
        src = items2[r["rid"]]["source"]
        split = "train" if not src.startswith("random") else ("test" if int(hashlib.md5(str(r["rid"]).encode()).hexdigest(), 16) % 5 == 0 else "train")
        rows.append({"rid": r["rid"], "split": split, **{c: int(c in ids) for c in FIDS + ["other"]}})
T2 = pd.DataFrame(rows).drop_duplicates("rid")
T = pd.concat([T1, T2[~T2.rid.isin(T1.rid)]], ignore_index=True)
for c in FIDS + ["other"]: T[c] = T[c].fillna(0).astype("int8")
T.to_parquet(f"{OUT}/labels_train.parquet", index=False)
it1 = json.load(open(f"{ROOT}/labels/llm_label/items.json"))
json.dump([{"rid": x["rid"], "text": x["text"]} for x in it1] + [{"rid": x["rid"], "text": x["text"]} for x in items2.values()],
          open(f"{ROOT}/data/label_items_v2.json", "w"), ensure_ascii=False)
print("round2 labeled", len(T2), "unknown ids", unknown, "| v2 table", T.shape, T.split.value_counts().to_dict())
print("train positives per factor (min/median):", int(T[T.split == 'train'][FIDS].sum().min()), int(T[T.split == 'train'][FIDS].sum().median()))
