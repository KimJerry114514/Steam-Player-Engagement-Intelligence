"""Convert codebook v1.1 into the B-page checklist (labels/gold/codebook_for_ui.json).
Family-level factors are mapped onto our 25 gameplay-family ids so the page shows them for the right games."""
import json, re, difflib
ROOT = "."
import sys
CBP = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/llm_jobs/out/revise/codebook_v1_1.json"
cb = json.load(open(CBP))
FAMS = sorted({g["family"] for g in json.load(open(f"{ROOT}/data/app_families.json"))["games"]})


def norm_family(s):
    s = (s or "").lower().replace("fam.", "").replace("-", "_").replace(" ", "_")
    if s in FAMS: return s
    hit = [f for f in FAMS if f in s or s in f]
    if hit: return hit[0]
    m = difflib.get_close_matches(s, FAMS, n=1, cutoff=0.4)
    return m[0] if m else s


out = {"version": "v1.1" if "v1_1" in CBP else "v1", "domains": cb["domains"], "factors": []}
unmapped = []
for f in cb["factors"]:
    fam = norm_family(f.get("family")) if f["level"] == "family" else ""
    if f["level"] == "family" and fam not in FAMS: unmapped.append((f["id"], f.get("family")))
    out["factors"].append({"id": f["id"], "label": f["label"], "domain": f["domain"], "level": f["level"], "family": fam,
                           "game_appid": int(f.get("game_appid") or 0), "status": f["status"],
                           "definition": f["definition"] + ("  | Include: " + "; ".join(f["include_rules"][:3]) if f["include_rules"] else "")
                                         + ("  | Exclude: " + "; ".join(f["exclude_rules"][:3]) if f["exclude_rules"] else "")})
json.dump(out, open(f"{ROOT}/labels/gold/codebook_for_ui.json", "w"), ensure_ascii=False, indent=1)
from collections import Counter
print("factors", len(out["factors"]), Counter(f["level"] for f in out["factors"]), Counter(f["status"] for f in out["factors"]))
print("family factors by family", Counter(f["family"] for f in out["factors"] if f["level"] == "family"))
print("unmapped family names", unmapped)
