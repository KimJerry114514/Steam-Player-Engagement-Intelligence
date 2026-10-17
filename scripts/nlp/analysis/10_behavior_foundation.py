"""Behavior foundation for negative Steam reviews (no factors yet).

Outputs (results/behavior/, report/figs/):
 T1 reviewer segments per focus game: playtime-at-review segment x post-review engagement
 T2 review-age cohort check: how D30/D90 and naive 'extra hours' depend on review age (teacher's concern)
 F1 post-review activity curves: P(last play >= review + t days | observed >= t+30 days), per focus game
Definitions
 snapshot S_app = max(created, last_played) over all reviews of the app (~early Nov 2023 for focus games)
 ret_h = 1[last_played >= created + h days], defined only if S_app - created >= h + 30 days
"""
import os, duckdb, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "."
OUT = f"{ROOT}/results/behavior"; FIG = f"{ROOT}/report/figs"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
FOCUS = {730: "Counter-Strike", 578080: "PUBG", 1172470: "Apex Legends", 1091500: "Cyberpunk 2077", 105600: "Terraria"}
con = duckdb.connect(); con.execute("SET threads=8")
con.execute(f"CREATE VIEW b AS SELECT * FROM '{ROOT}/data/neg_behavior.parquet'")
seg = """CASE WHEN hours_at_review < 2 THEN '1_<2h' WHEN hours_at_review < 20 THEN '2_2-20h'
              WHEN hours_at_review < 200 THEN '3_20-200h' ELSE '4_200h+' END"""

# T1 segments
t1 = con.execute(f"""
SELECT appid, {seg} AS seg, count(*) n,
  avg(ret_d30) d30, avg(ret_d90) d90, avg(no_play_after_review) no_play_after,
  avg(active_last_2w_at_snapshot) active_at_snapshot, median(hours_at_review) med_h_at_review
FROM b WHERE appid IN ({','.join(map(str, FOCUS))}) GROUP BY 1,2 ORDER BY 1,2""").df()
t1["game"] = t1.appid.map(FOCUS)
t1["share"] = t1.n / t1.groupby("appid").n.transform("sum")
t1.to_csv(f"{OUT}/T1_segments_focus.csv", index=False)
allg = con.execute(f"""SELECT {seg} seg, count(*) n, avg(ret_d30) d30, avg(ret_d90) d90, avg(no_play_after_review) no_play_after
                       FROM b GROUP BY 1 ORDER BY 1""").df()
allg["share"] = allg.n / allg.n.sum()
allg.to_csv(f"{OUT}/T1_segments_allgames.csv", index=False)

# T2 review-age cohort check (focus games), yearly cohorts
t2 = con.execute(f"""
SELECT appid, year(to_timestamp(created)) AS yr, count(*) n, avg(age_days) mean_age_days,
  avg(ret_d30) d30, avg(ret_d90) d90, median(extra_hours) med_extra_hours, avg(extra_hours/(age_days/30.4)) mean_extra_hours_per_month
FROM b WHERE appid IN ({','.join(map(str, FOCUS))}) GROUP BY 1,2 HAVING count(*) >= 500 ORDER BY 1,2""").df()
t2["game"] = t2.appid.map(FOCUS)
t2.to_csv(f"{OUT}/T2_age_cohorts_focus.csv", index=False)

# F1 activity curves
ts = [1, 7, 14, 30, 60, 90, 180, 365]
rows = []
for appid, name in FOCUS.items():
    for t in ts:
        r = con.execute(f"""SELECT avg((last_played >= created + {t}*86400)::INT), count(*) FROM b
                            WHERE appid={appid} AND age_days >= {t} + 30""").fetchone()
        rows.append({"appid": appid, "game": name, "t_days": t, "p_active_beyond_t": r[0], "n": r[1]})
f1 = pd.DataFrame(rows); f1.to_csv(f"{OUT}/F1_activity_curves.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4.2))
for name, g in f1.groupby("game"):
    ax.plot(g.t_days, g.p_active_beyond_t, marker="o", label=name)
ax.set_xscale("log"); ax.set_xticks(ts); ax.set_xticklabels(ts)
ax.set_xlabel("days after the negative review (log scale)"); ax.set_ylabel("share still playing beyond day t")
ax.set_title("Post-review activity of negative reviewers\n(each point uses only reviews observed for >= t+30 days)")
ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{FIG}/F1_activity_curves.png", dpi=160)

pd.set_option("display.width", 200)
print(t1[["game", "seg", "n", "share", "d30", "d90", "no_play_after", "active_at_snapshot"]].round(3).to_string(index=False))
print(allg.round(3).to_string(index=False))
print(t2[["game", "yr", "n", "mean_age_days", "d30", "d90", "med_extra_hours", "mean_extra_hours_per_month"]].round(3).to_string(index=False))
print(f1.pivot(index="t_days", columns="game", values="p_active_beyond_t").round(3).to_string())
