"""Write the team's per-review NLP factor file (nlp_factor_scores_2023.parquet + _meta.json), run on 2026-09-28.

Recovered verbatim from the session log (it was run inline, not saved as a script); only ROOT/out paths made relative.
Keeps the 35 factors with held-out test F1 >= 0.40 and >= 5 held-out positives (ft_large_v2 = multilingual-e5-large
fine-tuned on 9,720 LLM-labeled reviews). Scores on disk are uint8 (0-255): p_ = score/255 rounded to 3 decimals;
flag_ = score >= round(threshold * 255), i.e. computed from the stored 8-bit score, so use flag_ as given.
"""
import duckdb, json, glob, os
ROOT = "."
th = json.load(open(f"{ROOT}/results/ft_large_v2_thresholds.json"))["thresholds"]
met = json.load(open(f"{ROOT}/results/ft_large_v2_metrics.json"))
cb = {f["id"]: f for f in json.load(open(f"{ROOT}/llm_jobs/out/revise/codebook_v1_1.json"))["factors"]}
rel = [f for f in th if met[f]["test_f1"] >= 0.40 and met[f]["n_pos_test"] >= 5]
files = sorted(glob.glob(f"{ROOT}/results/scores_ft_large_v2/rest_*.parquet")); assert len(files) == 82
col = lambda f: f.replace(".", "_")
prob = ", ".join(f'round("{f}"/255.0, 3)::FLOAT AS "p_{col(f)}"' for f in rel)
flag = ", ".join(f'("{f}" >= {int(round(th[f] * 255))})::TINYINT AS "flag_{col(f)}"' for f in rel)
nflag = " + ".join(f'("{f}" >= {int(round(th[f] * 255))})::INT' for f in rel)
out = "./team_data/nlp_factor_scores_2023.parquet"
con = duckdb.connect(); con.execute("SET threads=8"); con.execute("SET memory_limit='16GB'")
con.execute(f"""COPY (SELECT a.appid, a.recommendationid, ({nflag})::SMALLINT AS n_factors_flagged, {prob}, {flag}
  FROM (SELECT DISTINCT ON (rid) * FROM read_parquet({files})) s JOIN '{ROOT}/data/neg_all.parquet' a USING(rid)
  ORDER BY a.appid, a.recommendationid) TO '{out}.tmp' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)""")
os.replace(out + ".tmp", out)
n, u = con.execute(f"SELECT count(*), count(DISTINCT (appid, recommendationid)) FROM '{out}'").fetchone()
bad = con.execute(f"SELECT count(*) FROM '{out}' WHERE n_factors_flagged <> {' + '.join('flag_' + col(f) for f in rel)}").fetchone()[0]
meta = {"rows": n, "key": ["appid", "recommendationid"], "model": "multilingual-e5-large fine-tuned on 9,720 LLM-labeled reviews (ft_large_v2)",
        "inclusion_rule": "held-out test F1 >= 0.40 and at least 5 held-out test positives",
        "note": "p_ is stored with 8-bit precision (256 levels); flag_ was computed from the full-precision probability, so use flag_ as given rather than recomputing it from p_.",
        "factors": [{"column": col(f), "factor_id": f, "label": cb.get(f, {}).get("label", f), "threshold": th[f],
                     "test_f1": round(met[f]["test_f1"], 3), "test_precision": round(met[f]["test_precision"], 3), "test_recall": round(met[f]["test_recall"], 3),
                     "n_pos_test": met[f]["n_pos_test"], "n_pos_train": met[f]["n_pos_train"],
                     "caution": "fewer than 10 held-out positives: F1 is imprecise" if met[f]["n_pos_test"] < 10 else ""} for f in rel]}
json.dump(meta, open(out.replace(".parquet", "_meta.json"), "w"), indent=1, ensure_ascii=False)
print("rows", n, "unique", u, "factors", len(rel), "n_flag mismatch", bad, "caution:", [m["column"] for m in meta["factors"] if m["caution"]])
