"""Stratified discovery sample for LLM open coding.

Strata: 5 focus games (4 batches x 60 each), each gameplay family among the top-600 apps
excluding focus games (2 batches if >=20 apps else 1), and the long tail outside the top 600 (2 batches).
Within a stratum: 50% random, 25% most helpful, 25% long (>=400 chars); min 40 chars cleaned text;
per-app caps keep breadth. Deterministic (hash-ordered). Writes labels/discovery/batch_*.json + manifest.
"""
import json, os, duckdb

BASE = "."
OUT = f"{BASE}/labels/discovery"
os.makedirs(OUT, exist_ok=True)
FOCUS = {730: "cs", 105600: "terraria", 578080: "pubg", 1091500: "cyberpunk", 1172470: "apex"}
BATCH = 60

con = duckdb.connect()
con.execute("SET threads=8")
fam = json.load(open(f"{BASE}/data/app_families.json"))["games"]
con.execute("CREATE TABLE fam(appid BIGINT, family VARCHAR)")
con.executemany("INSERT INTO fam VALUES (?, ?)", [(g["appid"], g["family"]) for g in fam])

con.execute(f"""
CREATE TABLE pool AS
SELECT n.rid, n.appid, n.game, n.language, n.pt_at_review/60.0 AS hours_at_review, n.weighted_vote_score, n.votes_up,
       t.text, length(t.text) AS clen, hash(n.rid * 7919 + 20260927) AS h,
       coalesce(f.family, CASE WHEN n.appid IN (SELECT appid FROM fam) THEN 'unknown' ELSE 'long_tail' END) AS family
FROM '{BASE}/data/neg_all.parquet' n
JOIN '{BASE}/data/neg_text.parquet' t USING (rid)
LEFT JOIN fam f USING (appid)
WHERE length(t.text) >= 40
""")

def draw(where, n, per_app_cap):
    """50% random, 25% helpful, 25% long; dedupe; per-app cap."""
    parts = [("random", "h", n // 2), ("helpful", "weighted_vote_score DESC, votes_up DESC, h", n // 4),
             ("long", "h", n - n // 2 - n // 4)]
    chosen, seen, per_app = [], set(), {}
    for kind, order, k in parts:
        extra = " AND clen >= 400" if kind == "long" else (" AND votes_up >= 5" if kind == "helpful" else "")
        rows = con.execute(f"""SELECT rid, appid, game, language, hours_at_review, text FROM pool
                              WHERE {where}{extra} ORDER BY {order} LIMIT {k * 40}""").fetchall()
        got = 0
        for r in rows:
            if got >= k: break
            if r[0] in seen or per_app.get(r[1], 0) >= per_app_cap: continue
            seen.add(r[0]); per_app[r[1]] = per_app.get(r[1], 0) + 1
            chosen.append({"rid": r[0], "appid": r[1], "game": r[2], "lang": r[3],
                           "hours_at_review": round(r[4] or 0, 1), "sample_kind": kind, "text": r[5][:1000]})
            got += 1
    return chosen

manifest = []
def write(stratum, rows):
    for b in range(0, len(rows), BATCH):
        chunk = rows[b:b + BATCH]
        path = f"{OUT}/batch_{stratum}_{b // BATCH + 1}.json"
        json.dump(chunk, open(path, "w"), ensure_ascii=False, indent=0)
        manifest.append({"stratum": stratum, "path": path, "n": len(chunk)})

for appid, key in FOCUS.items():
    write(f"game_{key}", draw(f"appid = {appid}", 4 * BATCH, 10 ** 9))

counts = dict(con.execute("SELECT family, count(DISTINCT appid) FROM pool WHERE family NOT IN ('long_tail','unknown') "
                          f"AND appid NOT IN ({','.join(map(str, FOCUS))}) GROUP BY 1").fetchall())
for family, napps in sorted(counts.items()):
    nb = 2 if napps >= 20 else 1
    write(f"family_{family}", draw(f"family = '{family}' AND appid NOT IN ({','.join(map(str, FOCUS))})", nb * BATCH, 4))

write("long_tail", draw("family = 'long_tail'", 2 * BATCH, 1))
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
print(len(manifest), "batches;", sum(m["n"] for m in manifest), "reviews")
for m in manifest: print(m["stratum"], m["n"])
