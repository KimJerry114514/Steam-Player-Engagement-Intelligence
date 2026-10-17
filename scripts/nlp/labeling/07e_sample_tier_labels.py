"""Combine the RANDOM components of both labeling rounds for sample-tier estimates (active-learning selections excluded).
Output: results/labels/v2/labels_primary_random.parquet (same columns as results/labels/final/labels_primary.parquet)."""
import json, glob, os, pandas as pd
ROOT = "."; os.makedirs(f"{ROOT}/results/labels/v2", exist_ok=True)
cb = json.load(open(f"{ROOT}/llm_jobs/out/revise/codebook_v1_1.json")); FIDS = [f["id"] for f in cb["factors"]]
L1 = pd.read_parquet(f"{ROOT}/results/labels/final/labels_primary.parquet")
L1 = L1[L1.source.isin(["random_focus", "random_all"])]
items2 = {x["rid"]: x for x in json.load(open(f"{ROOT}/labels/llm_label2/items.json"))}
rows = []
for f in sorted(glob.glob(f"{ROOT}/llm_jobs/out/label_al/*.json")):
    for r in json.load(open(f))["reviews"]:
        it = items2.get(r["rid"])
        if not it or not it["source"].startswith("random"): continue
        ids = {x["id"] for x in r["factors"]}
        rows.append({"rid": r["rid"], "primary": r["primary_factor"], "polarity": r["content_polarity"], "attributable": r["attributable"],
                     "quit": r["self_reported_quit"], "refund": r["refund_intent"], "new_candidate": r["new_factor_candidate"],
                     "confidence": r["confidence"], "source": it["source"], "appid": it["appid"], **{c: int(c in ids) for c in FIDS + ["other"]}})
L2 = pd.DataFrame(rows).drop_duplicates("rid")
L = pd.concat([L1, L2[~L2.rid.isin(L1.rid)]], ignore_index=True)
L.to_parquet(f"{ROOT}/results/labels/v2/labels_primary_random.parquet", index=False)
print("random-sample labels:", len(L), L.source.value_counts().to_dict())
