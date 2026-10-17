"""Recompute p-values and CIs of an effects CSV with t critical values (df = clusters - 1), exactly as the current
20_factor_behavior.py does, for outputs produced before that change. Usage: python 20b_t_ci.py <effects.csv> (in place)"""
import sys, numpy as np, pandas as pd
from scipy import stats
p = sys.argv[1]; e = pd.read_csv(p); dof = (e.clusters - 1).clip(lower=1)
e["p"] = 2 * stats.t.sf(np.abs(e.beta / e.se), dof)
for lvl, q in (("90", 0.95), ("95", 0.975)):
    z = stats.t.ppf(q, dof); e[f"lo{lvl}"] = e.beta - z * e.se; e[f"hi{lvl}"] = e.beta + z * e.se
def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]; return np.minimum(q, 1)
e["q_bh"] = e.groupby(["scope", "outcome"]).p.transform(lambda x: bh(x.values))
e.to_csv(p, index=False); print("t-based CIs written", p, "min clusters", int(e.clusters.min()))
