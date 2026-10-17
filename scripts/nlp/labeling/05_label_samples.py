"""Samples for (1) the human gold set B (150, zh/en) and (2) the LLM labeling set (representative random draws).
Disjoint from the discovery sample and from gold A. Deterministic.
LLM labeling set = 400 random negatives per focus game + 1,500 random negatives from all other games
                     + all gold A and gold B items (so LLM can be scored against the human labels).
"""
import json, glob, os, duckdb, random
ROOT = "."
FOCUS = (730, 578080, 1172470, 1091500, 105600)
used = set()
for f in glob.glob(f"{ROOT}/labels/discovery/batch_*.json"):
    used |= {r["rid"] for r in json.load(open(f))}
goldA = json.load(open(f"{ROOT}/labels/gold/gold_A_items.json"))
used |= {x["rid"] for x in goldA}
fam = {g["appid"]: g["family"] for g in json.load(open(f"{ROOT}/data/app_families.json"))["games"]}
con = duckdb.connect(); con.execute("SET threads=8")
con.execute("CREATE TABLE used(rid BIGINT)"); con.executemany("INSERT INTO used VALUES (?)", [(r,) for r in used])
con.execute("CREATE TABLE fam(appid BIGINT, family VARCHAR)"); con.executemany("INSERT INTO fam VALUES (?,?)", list(fam.items()))
BASE = f"""SELECT n.rid, n.appid, n.game, n.language, round(n.pt_at_review/60.0,1) h, strftime(to_timestamp(n.created),'%Y-%m-%d') d,
  t.text, coalesce(f.family,'long_tail') AS family FROM '{ROOT}/data/neg_all.parquet' n JOIN '{ROOT}/data/neg_text.parquet' t USING(rid)
  LEFT JOIN fam f USING(appid) WHERE n.rid NOT IN (SELECT rid FROM used)"""


def rec(r, prefix, i):
    return {"gid": f"{prefix}{i:03d}", "rid": int(r["rid"]), "appid": int(r["appid"]), "game": r["game"], "lang": r["language"],
            "hours_at_review": float(r["h"]), "date": r["d"], "family": r["family"], "text": r["text"]}


# gold B: 20 per focus game (10 zh + 10 en) + 50 other games (25 zh + 25 en), length 30..1200
B = []
for a in FOCUS:
    for lang in ("schinese", "english"):
        B += con.execute(f"SELECT * FROM ({BASE}) WHERE appid={a} AND language='{lang}' AND length(text) BETWEEN 30 AND 1200 "
                         f"ORDER BY hash(rid*31337+1) LIMIT 10").df().to_dict("records")
for lang in ("schinese", "english"):
    B += con.execute(f"SELECT * FROM ({BASE}) WHERE appid NOT IN {FOCUS} AND language='{lang}' AND length(text) BETWEEN 30 AND 1200 "
                     f"ORDER BY hash(rid*31337+2) LIMIT 25").df().to_dict("records")
random.Random(7).shuffle(B)
goldB = [rec(r, "B", i + 1) for i, r in enumerate(B)]
json.dump(goldB, open(f"{ROOT}/labels/gold/gold_B_items.json", "w"), ensure_ascii=False, indent=0)
usedB = {x["rid"] for x in goldB}

# LLM labeling set: representative random (any language, any length >= 1 char)
con.execute("CREATE TABLE usedb(rid BIGINT)"); con.executemany("INSERT INTO usedb VALUES (?)", [(r,) for r in usedB])
L = []
for a in FOCUS:
    L += con.execute(f"SELECT * FROM ({BASE}) WHERE appid={a} AND rid NOT IN (SELECT rid FROM usedb) AND length(trim(text))>0 "
                     f"ORDER BY hash(rid*7777+3) LIMIT 400").df().to_dict("records")
L += con.execute(f"SELECT * FROM ({BASE}) WHERE appid NOT IN {FOCUS} AND rid NOT IN (SELECT rid FROM usedb) AND length(trim(text))>0 "
                 f"ORDER BY hash(rid*7777+4) LIMIT 1500").df().to_dict("records")
lab = [rec(r, "L", i + 1) for i, r in enumerate(L)]
for x in lab: x["source"] = "random_focus" if x["appid"] in FOCUS else "random_all"
for x in goldA: lab.append(dict(x, source="goldA"))
for x in goldB: lab.append(dict(x, source="goldB"))
random.Random(11).shuffle(lab)
os.makedirs(f"{ROOT}/labels/llm_label", exist_ok=True)
for x in lab: x["text"] = x["text"][:1500]
json.dump(lab, open(f"{ROOT}/labels/llm_label/items.json", "w"), ensure_ascii=False, indent=0)
BATCH = 50
man = []
for b in range(0, len(lab), BATCH):
    p = f"{ROOT}/labels/llm_label/batch_{b // BATCH + 1:03d}.json"
    json.dump([{k: x[k] for k in ("rid", "appid", "game", "lang", "hours_at_review", "text")} for x in lab[b:b + BATCH]],
              open(p, "w"), ensure_ascii=False, indent=0)
    man.append({"path": p, "n": len(lab[b:b + BATCH])})
json.dump(man, open(f"{ROOT}/labels/llm_label/manifest.json", "w"), indent=1)
import collections
print("goldB", len(goldB), collections.Counter(x["lang"] for x in goldB))
print("llm label items", len(lab), collections.Counter(x["source"] for x in lab), "batches", len(man))
