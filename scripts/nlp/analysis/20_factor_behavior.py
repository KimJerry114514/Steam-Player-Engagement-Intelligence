"""Factor -> post-review behavior, with fixed-horizon outcomes and app x review-quarter fixed effects.

Input: a factor matrix parquet (rid + 0/1 int8 columns, one per factor), joined to data/neg_behavior.parquet.
For each outcome (ret_d30, ret_d90, no_play_after_review) and each scope (pooled all games, each focus game):
  y = sum_k beta_k * F_k + controls + FE(app x review_quarter) + e     (linear probability model)
  controls: log1p(hours_at_review), log1p(review_len), log1p(num_games_owned), log1p(num_reviews),
            received_for_free, steam_purchase, early_access, edited, language group dummies
  SE clustered by app (pooled) or by review month (single game). Reports beta in percentage points,
  90% and 95% CIs, p-values and Benjamini-Hochberg q-values across factors within a scope/outcome.
Also reports prevalence and raw retention of flagged vs unflagged reviews.
Usage: python 20_factor_behavior.py <factor_matrix.parquet> <out_prefix>
"""
import sys, os, duckdb, numpy as np, pandas as pd
from scipy import stats

ROOT = "."
FOCUS = {730: "Counter-Strike", 578080: "PUBG", 1172470: "Apex Legends", 1091500: "Cyberpunk 2077", 105600: "Terraria"}
fm_path, out_prefix = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
con = duckdb.connect(); con.execute("SET threads=8")
fm_cols = [c for c in con.execute(f"DESCRIBE SELECT * FROM '{fm_path}'").df().column_name if c != "rid"]
# Options (env): RELIABLE_ONLY=1 keeps only factors with held-out F1 >= 0.40 for classifier CLS_TAG; DROP_EDITED=1 removes the
# post-review 'edited' flag from the controls (it is a bad control); SUBSET=unedited keeps unedited reviews only (leakage check);
# CLUSTER_FE=1 clusters the deep-dive pool at game x quarter and single titles at review quarter (not finer than the FE cells);
# DROP_SNAPSHOT=1 drops library size and reviewer activity (author snapshots taken at data collection, i.e. after the review);
# NOPLAY_WINDOWED=1 estimates no_play_after_review only on reviews observed >= 60 days (same sample as ret_d30).
import json
if os.environ.get("RELIABLE_ONLY") == "1":
    _met = json.load(open(f"{ROOT}/results/{os.environ.get('CLS_TAG', 'ft')}_metrics.json"))
    fm_cols = [c for c in fm_cols if _met.get(c, {}).get("test_f1", 0) >= 0.40 and _met.get(c, {}).get("n_pos_test", 0) >= 5]
df = con.execute(f"""
SELECT b.rid, b.appid, b.review_quarter, month(to_timestamp(b.created)) + 12*year(to_timestamp(b.created)) AS ym,
  b.ret_d30, b.ret_d90, b.no_play_after_review, ln(1+b.hours_at_review) lh, ln(1+b.review_len) llen,
  ln(1+coalesce(b.num_games_owned,0)) lgames, ln(1+coalesce(b.num_reviews,0)) lrev,
  coalesce(b.received_for_free,0) free, coalesce(b.steam_purchase,0) purchase, coalesce(b.early_access,0) ea, b.edited,
  b.language, {', '.join('f."' + c + '"' for c in fm_cols)}
FROM '{ROOT}/data/neg_behavior.parquet' b JOIN '{fm_path}' f USING (rid)""").df()
top_lang = df.language.value_counts().index[:12]
for l in top_lang[1:]:
    df[f"lang_{l}"] = (df.language == l).astype(np.float32)
CONTROLS = ["lh", "llen"] + ([] if os.environ.get("DROP_SNAPSHOT") == "1" else ["lgames", "lrev"]) + ["free", "purchase", "ea"] + ([] if os.environ.get("DROP_EDITED") == "1" else ["edited"]) + [f"lang_{l}" for l in top_lang[1:]]
if os.environ.get("SUBSET") == "unedited": df = df[df.edited == 0].copy()
df["fe"] = df.appid.astype(str) + "_" + df.review_quarter.astype(str)
df["app_month"] = df.appid.astype(np.int64) * 100000 + df.ym.astype(np.int64)
print("rows", len(df), "factors", len(fm_cols), flush=True)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def fe_ols(d, ycol, xcols, cluster):
    d = d[d[ycol].notna()]
    if len(d) < 1000: return None
    cols = [ycol] + xcols
    M = d[cols].astype(np.float64)
    M = M - M.groupby(d["fe"].values).transform("mean")
    y = M[ycol].values; X = M[xcols].values
    keep = X.std(0) > 0; X = X[:, keep]; xs = [c for c, k in zip(xcols, keep) if k]
    XtX_inv = np.linalg.pinv(X.T @ X); beta = XtX_inv @ (X.T @ y); u = y - X @ beta
    S = pd.DataFrame(X * u[:, None]).groupby(d[cluster].values).sum().values
    G = S.shape[0]; n, k = X.shape
    V = XtX_inv @ (S.T @ S) @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.diag(V))
    return pd.DataFrame({"term": xs, "beta": beta, "se": se}).assign(n=len(d), clusters=G)


rows = []
CFE = os.environ.get("CLUSTER_FE") == "1"
scopes = [("ALL", df, "appid" if df.appid.nunique() >= 30 else "app_month")] + [(FOCUS[a], df[df.appid == a], "review_quarter" if CFE else "ym") for a in FOCUS]
_other = df[~df.appid.isin(list(FOCUS))]
if _other.appid.nunique() >= 30: scopes.append(("OTHER_GAMES_SAMPLE", _other, "appid"))
_foc = df[df.appid.isin(list(FOCUS))]
if len(_foc) and len(_other): scopes.append(("FOCUS_POOLED", _foc, "fe" if CFE else "app_month"))
for scope, d, cl in scopes:
    fcols = [c for c in fm_cols if d[c].sum() >= 50]
    for y in ["ret_d30", "ret_d90", "no_play_after_review"]:
        if y == "no_play_after_review" and os.environ.get("NOPLAY_WINDOWED") == "1": d_y = d[d.ret_d30.notna()]
        else: d_y = d
        r = fe_ols(d_y, y, fcols + CONTROLS, cl)
        if r is None: continue
        r = r[r.term.isin(fcols)].copy()
        dof = max(int(r.clusters.iloc[0]) - 1, 1)  # t critical values with G-1 df (few clusters in single-title models)
        r["p"] = 2 * stats.t.sf(np.abs(r.beta / r.se), dof); r["q_bh"] = bh(r.p.values)
        for lvl, q in (("90", 0.95), ("95", 0.975)):
            z = stats.t.ppf(q, dof); r[f"lo{lvl}"] = r.beta - z * r.se; r[f"hi{lvl}"] = r.beta + z * r.se
        dd = d_y[d_y[y].notna()]
        r["prevalence"] = [dd[t].mean() for t in r.term]
        r["n_flagged"] = [int(dd[t].sum()) for t in r.term]
        r["raw_rate_flagged"] = [dd.loc[dd[t] == 1, y].mean() for t in r.term]
        r["raw_rate_unflagged"] = [dd.loc[dd[t] == 0, y].mean() for t in r.term]
        r["scope"], r["outcome"] = scope, y
        rows.append(r)
res = pd.concat(rows)
for c in ["beta", "se", "lo90", "hi90", "lo95", "hi95"]: res[c] = res[c] * 100  # percentage points
res.to_csv(out_prefix + "_effects.csv", index=False)
pd.set_option("display.width", 220)
show = res[(res.outcome == "ret_d30")].sort_values(["scope", "beta"])[["scope", "term", "n_flagged", "prevalence", "raw_rate_flagged", "raw_rate_unflagged", "beta", "lo90", "hi90", "q_bh"]]
print(show.round(4).to_string(index=False))
