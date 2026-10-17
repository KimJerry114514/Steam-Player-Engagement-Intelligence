# Original-language annotation keys and search terms below are research data, not submission instructions.
"""Keyword check behind the report's PUBG onboarding sentence: among PUBG negative reviews the classifier flags for
ux.onboarding (all languages), the share whose text mentions the tutorial explicitly, and the share that mentions the
tutorial or new players / learning. Output: results/pubg_onboarding_keywords.json"""
import json, duckdb
ROOT = "."
TUT = ["tutorial", "教程", "教學", "教学", "训练场", "訓練場", "обучени", "туториал", "tutorial"]
NEW = TUT + ["new player", "newbie", "beginner", "noob", "learn", "新手", "萌新", "上手", "новичк", "новичок", "научит"]
like = lambda ws: " OR ".join(f"lower(t.text) LIKE '%{w}%'" for w in ws)
con = duckdb.connect()
n, tut, new = con.execute(f"""SELECT count(*), avg(({like(TUT)})::INT), avg(({like(NEW)})::INT)
  FROM '{ROOT}/results/ft_large_v2/factor_matrix_16m.parquet' m JOIN '{ROOT}/data/neg_all.parquet' a USING (rid)
  JOIN '{ROOT}/data/neg_text.parquet' t USING (rid) WHERE a.appid = 578080 AND m."ux.onboarding" = 1""").fetchone()
out = {"n_flagged_pubg": n, "share_tutorial_terms": tut, "share_tutorial_or_new_player_terms": new, "tutorial_terms": TUT, "broad_terms": NEW}
json.dump(out, open(f"{ROOT}/results/pubg_onboarding_keywords.json", "w"), ensure_ascii=False, indent=1); print(out)
