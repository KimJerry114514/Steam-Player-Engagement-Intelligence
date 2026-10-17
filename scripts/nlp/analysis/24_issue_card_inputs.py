"""Evidence packets for investigation cards (per focus game): top factors by sample-tier prevalence plus the most
retention-relevant factors, with prevalence (sample + adjusted classifier), D30 association (game scope, classifier tier),
yearly share trend (classifier tier) and verbatim evidence quotes from the two-pass LLM labels.
Usage: python 24_issue_card_inputs.py <factor_matrix.parquet> <effects.csv>  -> labels/cards/packets.json"""
import sys, json, glob, os, collections, duckdb, pandas as pd
ROOT = "."; os.makedirs(f"{ROOT}/labels/cards", exist_ok=True)
fm, effp = sys.argv[1], sys.argv[2]
GAMES = {730: "Counter-Strike", 578080: "PUBG", 1172470: "Apex Legends", 1091500: "Cyberpunk 2077", 105600: "Terraria"}
cb = {f["id"]: f for f in json.load(open(f"{ROOT}/llm_jobs/out/revise/codebook_v1_1.json"))["factors"]}
items = {x["rid"]: x for x in json.load(open(f"{ROOT}/labels/llm_label/items.json"))}
ev = collections.defaultdict(list)  # (appid, factor) -> quotes
for f in sorted(glob.glob(f"{ROOT}/llm_jobs/out/label/*.json")):
    rec = {}
    rp = f.replace("/out/label/", "/out/recall/")
    if os.path.exists(rp): rec = {x["rid"]: x for x in json.load(open(rp))["reviews"]}
    for r in json.load(open(f))["reviews"]:
        it = items.get(r["rid"]);
        if not it: continue
        facs = r["factors"] + (rec[r["rid"]]["add"] if r["rid"] in rec else [])
        for x in facs:
            if x["evidence"]: ev[(it["appid"], x["id"])].append({"quote": x["evidence"][:220], "lang": it["lang"], "hours_at_review": it["hours_at_review"], "date": it["date"]})
sp = pd.read_csv(f"{ROOT}/results/sample_tier_prevalence.csv")
SCALED_PATH = os.environ.get("SCALED_CSV", f"{ROOT}/results/scaled_prevalence.csv")
scaled = pd.read_csv(SCALED_PATH) if os.path.exists(SCALED_PATH) else pd.DataFrame()
eff = pd.read_csv(effp)
MET = json.load(open(f"{ROOT}/results/{os.environ.get('CLS_TAG', 'ft')}_metrics.json"))
OK = {f for f, m in MET.items() if m["test_f1"] >= 0.40 and m["n_pos_test"] >= 5}  # classifier-tier numbers only for reliable factors
con = duckdb.connect(); con.execute("SET threads=8")
fcols = [c for c in con.execute(f"DESCRIBE SELECT * FROM '{fm}'").df().column_name if c != "rid"]
packets = []
for a, g in GAMES.items():
    s = sp[sp.scope == g].sort_values("prevalence", ascending=False)
    top = list(s.factor.head(3))
    e = eff[(eff.scope == g) & (eff.outcome == "ret_d30") & (eff.n_flagged >= 300) & eff.term.isin(OK)]
    neg = e[e.hi90 < 0].sort_values("beta").factor.tolist() if "factor" in e else e[e.hi90 < 0].sort_values("beta").term.tolist()
    kk = dict(zip(s.factor, s.k))
    for f in [x for x in neg if kk.get(x, 0) >= 5][:2]:  # a retention-relevant factor gets its own card only with >= 5 sample-tier reviews as evidence
        if f not in top: top.append(f)
    trend = {}
    tf = [f for f in top if f in fcols and f in OK]
    if tf:
        q = ", ".join(f'avg("{f}") AS "{f}"' for f in tf)
        T = con.execute(f"SELECT year(to_timestamp(a.created)) yr, count(*) n, {q} FROM '{fm}' m JOIN '{ROOT}/data/neg_all.parquet' a USING(rid) WHERE a.appid={a} GROUP BY 1 HAVING count(*)>=1000 ORDER BY 1").df()
        trend = {f: {int(r.yr): round(float(r[f]), 4) for _, r in T.iterrows()} for f in tf}
    for f in top:
        row = s[s.factor == f]
        er = e[e.term == f] if ("term" in e and f in OK) else pd.DataFrame()
        sc = scaled[(scaled.scope == g) & (scaled.factor == f)] if (len(scaled) and f in OK) else pd.DataFrame()
        packets.append({"game": g, "appid": a, "factor_id": f, "factor_label": cb[f]["label"] if f in cb else f,
                        "definition": cb[f]["definition"] if f in cb else "", "owner_team": cb[f].get("owner_team", "") if f in cb else "",
                        "sample_prevalence": None if row.empty else {"k": int(row.k.iloc[0]), "n": int(row.n.iloc[0]), "p": round(float(row.prevalence.iloc[0]), 3),
                                                                   "lo90": round(float(row.lo90.iloc[0]), 3), "hi90": round(float(row.hi90.iloc[0]), 3)},
                        "classifier_adjusted_prevalence": None if sc.empty or pd.isna(sc.p_adjusted.iloc[0]) else round(float(sc.p_adjusted.iloc[0]), 3),
                        "d30_effect_pp": None if er.empty else {"beta": float(er.beta.iloc[0]), "lo90": float(er.lo90.iloc[0]), "hi90": float(er.hi90.iloc[0]), "clusters": int(er.clusters.iloc[0]),
                                                              "n_flagged": int(er.n_flagged.iloc[0]), "raw_d30_flagged": round(float(er.raw_rate_flagged.iloc[0]), 3), "raw_d30_unflagged": round(float(er.raw_rate_unflagged.iloc[0]), 3)},
                        "yearly_share_classifier": trend.get(f), "classifier_test_f1": MET.get(f, {}).get("test_f1"), "evidence": ev.get((a, f), [])[:6]})
pc = lambda x: f"{x * 100:.1f}%"


def readable(p):
    """Pre-format numbers so the card writer copies reader-friendly values (percentages, 2-decimal F1)."""
    sp_, ap, d = p.pop("sample_prevalence"), p.pop("classifier_adjusted_prevalence"), p.pop("d30_effect_pp")
    p["sample_tier_share"] = None if sp_ is None else f"{pc(sp_['p'])} ({sp_['k']} of {sp_['n']} random negative reviews; 90% CI {pc(sp_['lo90'])}–{pc(sp_['hi90'])})"
    p["classifier_adjusted_share"] = None if ap is None else (pc(ap) if ap > 0 else "≈0% (the pooled misclassification correction is uninformative for this title)")
    p["d30_retention_difference"] = None if d is None else (f"{d['beta']:+.1f} pp (90% CI {d['lo90']:+.1f} to {d['hi90']:+.1f}); raw share active past day 30: "
                                                            f"{pc(d['raw_d30_flagged'])} of {d['n_flagged']:,} flagged vs {pc(d['raw_d30_unflagged'])} of other negative reviewers of the game")
    p["retention_ci_caution"] = None if d is None or d["clusters"] >= 20 else f"the single-title model has only {d['clusters']} quarter clusters, so this CI may be too narrow; treat it as descriptive"
    y = p.pop("yearly_share_classifier")
    p["yearly_flagged_share_unadjusted"] = None if not y else {k: pc(v) for k, v in y.items()}
    f1 = p.pop("classifier_test_f1"); p["classifier_test_f1"] = None if f1 is None else f"{float(f1):.2f}"
    return p


packets = [readable(p) for p in packets]
json.dump(packets, open(f"{ROOT}/labels/cards/packets.json", "w"), ensure_ascii=False, indent=1)
print(len(packets), "packets:", [(p["game"], p["factor_id"]) for p in packets])
