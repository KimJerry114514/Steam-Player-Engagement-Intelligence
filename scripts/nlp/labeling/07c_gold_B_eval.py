"""Human gold set B (codebook checkboxes, draft ids) vs LLM labels (v1.1), via the LLM-produced draft->v1.1 mapping.
A human-tagged draft factor matches when LLM tagged ANY of its mapped v1.1 factors. Reports per-factor-group and micro
precision / recall / F1, and review-level field agreement. Output: results/labels/gold_B_eval.json"""
import json, collections, pandas as pd
ROOT = "."
M = {x["draft_id"]: x["v11_ids"] for x in json.load(open(f"{ROOT}/llm_jobs/out/map/draft_to_v11.json"))["mapping"]}
import sys
SUB = sys.argv[1] if len(sys.argv) > 1 else "final"
lab = pd.read_parquet(f"{ROOT}/results/labels/{SUB}/labels_primary.parquet"); FIDS = [c for c in lab.columns if "." in c]
items = {x["gid"]: x for x in json.load(open(f"{ROOT}/labels/gold/gold_B_items.json"))}
G = json.load(open(f"{ROOT}/labels/gold/gold_B_labels.json"))
per = collections.defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0}); rows = []
TP = FN = FP = 0
for gid, g in G.items():
    c = lab[lab.rid == items[gid]["rid"]]
    if not len(c): continue
    c = c.iloc[0]; cset = {f for f in FIDS if c[f] == 1}
    groups = {d: set(M.get(d, [])) for d in g["factors"]}
    covered = set().union(*groups.values()) if groups else set()
    for d, grp in groups.items():
        hit = bool(cset & grp); per[d]["tp" if hit else "fn"] += 1; TP += hit; FN += (not hit)
    for f in cset - covered: per["(llm-only) " + f]["fp"] += 1; FP += 1
    rows.append({"gid": gid, "h_pol": g["polarity"], "c_pol": c.polarity, "h_att": g.get("attributable") == "yes" or bool(groups),
                 "c_att": bool(c.attributable), "h_quit": g["self_reported_quit"], "c_quit": bool(c.quit),
                 "h_ref": g["refund_intent"], "c_ref": bool(c.refund), "n_h": len(groups), "n_c": len(cset)})
R = pd.DataFrame(rows)
P = TP / (TP + FP) if TP + FP else float("nan"); Rc = TP / (TP + FN) if TP + FN else float("nan")
summ = {"n": len(R), "micro_precision": P, "micro_recall": Rc, "micro_f1": 2 * P * Rc / (P + Rc) if P + Rc else float("nan"),
        "polarity_exact": (R.h_pol == R.c_pol).mean(), "negative_or_mixed_agreement": (R.h_pol.isin(["negative", "mixed"]) == R.c_pol.isin(["negative", "mixed"])).mean(),
        "attributable_agreement": (R.h_att == R.c_att).mean(), "quit_agreement": (R.h_quit == R.c_quit).mean(), "refund_agreement": (R.h_ref == R.c_ref).mean()}
json.dump({"summary": summ, "per_factor": per, "rows": rows}, open(f"{ROOT}/results/labels/{SUB}/gold_B_eval.json", "w"), ensure_ascii=False, indent=1, default=float)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()})
for k, v in sorted(per.items(), key=lambda kv: -sum(kv[1].values()))[:20]: print(k, v)
