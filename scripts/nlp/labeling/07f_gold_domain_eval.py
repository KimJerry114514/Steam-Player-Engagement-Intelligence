"""Domain-level agreement of the LLM labels with the human reference (gold A: 80 reviews, gold B: 60 reviews).
Same matching rules as 07b/07c, rolled up to the codebook's 8 domains (tech, gameplay, progression, content, social, ux,
business, service): a human reason counts as matched when the LLM tagged any factor in any domain the reason maps to; an
LLM domain not covered by any human reason is a false positive. Output: results/labels/final/gold_domain_eval.json"""
import json, importlib.util, pandas as pd
ROOT = "."
DOM = {f["id"]: f["domain"] for f in json.load(open(f"{ROOT}/llm_jobs/out/revise/codebook_v1_1.json"))["factors"]}
spec = importlib.util.spec_from_file_location("ga", f"{ROOT}/scripts/07b_gold_A_eval.py"); src = open(spec.origin).read()
CHIP2V11 = eval(src.split("CHIP2V11 = ")[1].split("\n}\n")[0] + "\n}"); FREE2V11 = eval(src.split("FREE2V11 = ")[1].split("\n")[0])
lab = pd.read_parquet(f"{ROOT}/results/labels/final/labels_primary.parquet"); FIDS = [c for c in lab.columns if "." in c]
M = {x["draft_id"]: x["v11_ids"] for x in json.load(open(f"{ROOT}/llm_jobs/out/map/draft_to_v11.json"))["mapping"]}


def score(sets):
    tp = fn = fp = 0
    for hum_groups, cdom in sets:
        covered = set().union(*hum_groups) if hum_groups else set()
        for g in hum_groups:
            hit = bool(cdom & g); tp += hit; fn += (not hit)
        fp += len(cdom - covered)
    p = tp / (tp + fp) if tp + fp else float("nan"); r = tp / (tp + fn) if tp + fn else float("nan")
    return {"tp": tp, "fn": fn, "fp": fp, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else float("nan")}


def llm_domains(rid):
    c = lab[lab.rid == rid]
    return None if not len(c) else {DOM[f] for f in FIDS if c.iloc[0][f] == 1 and f in DOM}


out = {}
for name, items_f, labels_f in [("A", "gold_A_items.json", "gold_A_labels.json"), ("B", "gold_B_items.json", "gold_B_labels.json")]:
    items = {x["gid"]: x for x in json.load(open(f"{ROOT}/labels/gold/{items_f}"))}; G = json.load(open(f"{ROOT}/labels/gold/{labels_f}"))
    sets = []
    for gid, g in G.items():
        cd = llm_domains(items[gid]["rid"])
        if cd is None: continue
        if name == "A":
            groups = [CHIP2V11[ch] for ch in g["chips"] if ch in CHIP2V11] + [FREE2V11.get(fr, []) for fr in g.get("free", [])]
        else:
            groups = [M.get(d, []) for d in g["factors"]]
        hum = [{DOM[f] for f in grp if f in DOM} for grp in groups]
        sets.append(([h for h in hum if h], cd))
    out[name] = {"n": len(sets), **score(sets)}
json.dump(out, open(f"{ROOT}/results/labels/final/gold_domain_eval.json", "w"), indent=1)
print({k: {m: (round(v, 3) if isinstance(v, float) else v) for m, v in d.items()} for k, d in out.items()})
