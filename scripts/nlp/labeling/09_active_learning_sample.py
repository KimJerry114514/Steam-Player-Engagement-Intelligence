"""Active-learning batch for the second labeling round (6,000 reviews, disjoint from everything labeled so far).
 - 2,000 focus-game reviews with the most uncertain fine-tuned predictions (max over factors of min(p, 1-p), p in [0.3, 0.7])
 - 1,500 focus-game candidates for rare factors (< 40 training positives): top-scored reviews per rare factor
 - 500 random focus-game reviews + 2,000 random reviews from the 1M all-games sample (keeps a representative component)
Writes labels/llm_label2/items.json, batch_XXX.json (50 each), manifest.json. Selection source is recorded per item."""
import json, glob, os, random, duckdb, numpy as np, pandas as pd
ROOT = "."; OUT = f"{ROOT}/labels/llm_label2"; os.makedirs(OUT, exist_ok=True)
met = json.load(open(f"{ROOT}/results/ft_metrics.json")); th = json.load(open(f"{ROOT}/results/ft_thresholds.json"))["thresholds"]
rare = [f for f, m in met.items() if m["n_pos_train"] < 40]
used = {x["rid"] for x in json.load(open(f"{ROOT}/labels/llm_label/items.json"))}
for f in glob.glob(f"{ROOT}/labels/discovery/batch_*.json"): used |= {r["rid"] for r in json.load(open(f))}
con = duckdb.connect(); con.execute("SET threads=8")
con.execute("CREATE TABLE used(rid BIGINT)"); con.executemany("INSERT INTO used VALUES (?)", [(r,) for r in used])
files = sorted(glob.glob(f"{ROOT}/results/scores_ft/focus_*.parquet"))
F = list(th)
unc = " , ".join(f'least("{f}", 255-"{f}")' for f in F)
con.execute(f"""CREATE TABLE s AS SELECT rid, greatest({unc})/255.0 AS unc, {", ".join('"%s"' % f for f in F)}
               FROM read_parquet({files}) WHERE rid NOT IN (SELECT rid FROM used)""")
pick = {}
for r in con.execute("SELECT rid FROM s WHERE unc >= 0.3 ORDER BY hash(rid*13+1) LIMIT 2000").fetchall(): pick[r[0]] = "uncertain"
per = max(1, 1500 // max(len(rare), 1))
for f in rare:
    for r in con.execute(f'SELECT rid FROM s ORDER BY "{f}" DESC, hash(rid) LIMIT {per * 3}').fetchall():
        if r[0] not in pick and sum(1 for v in pick.values() if v == f"rare:{f}") < per: pick[r[0]] = f"rare:{f}"
for r in con.execute("SELECT rid FROM s ORDER BY hash(rid*17+5) LIMIT 3000").fetchall():
    if r[0] not in pick and sum(1 for v in pick.values() if v == "random_focus") < 500: pick[r[0]] = "random_focus"
samp = con.execute(f"""SELECT rid FROM '{ROOT}/data/neg_text_sample1m.parquet' WHERE rid NOT IN (SELECT rid FROM used)
                       ORDER BY hash(rid*19+7) LIMIT 2000""").fetchall()
for r in samp: pick.setdefault(r[0], "random_all")
con.execute("CREATE TABLE p(rid BIGINT, src VARCHAR)"); con.executemany("INSERT INTO p VALUES (?,?)", list(pick.items()))
rows = con.execute(f"""SELECT p.rid, p.src, n.appid, n.game, n.language, round(n.pt_at_review/60.0,1) h, t.text FROM p
    JOIN '{ROOT}/data/neg_all.parquet' n USING(rid) JOIN '{ROOT}/data/neg_text.parquet' t USING(rid) WHERE length(trim(t.text))>0""").df()
items = [{"rid": int(r.rid), "appid": int(r.appid), "game": r.game, "lang": r.language, "hours_at_review": float(r.h), "text": r.text[:1500], "source": r.src} for r in rows.itertuples()]
random.Random(3).shuffle(items)
json.dump(items, open(f"{OUT}/items.json", "w"), ensure_ascii=False)
man = []
for b in range(0, len(items), 50):
    p = f"{OUT}/batch_{b // 50 + 1:03d}.json"
    json.dump([{k: x[k] for k in ("rid", "appid", "game", "lang", "hours_at_review", "text")} for x in items[b:b + 50]], open(p, "w"), ensure_ascii=False)
    man.append({"path": p, "n": len(items[b:b + 50])})
json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1)
from collections import Counter
print(len(items), "items,", len(man), "batches;", Counter(x["source"].split(":")[0] for x in items), "| rare factors:", len(rare))
