"""Report figures from the analysis outputs.
Usage: python 30_figures.py <effects_csv> <prevalence_csv> <codebook_json>
F2 prevalence heatmap (factor x game)   F3 forest plot of D30 effects (pp, 90% CI) per focus game
F4 priority matrix (prevalence vs D30 effect)   F5 reviewer segments (share and D30 by hours-at-review)
"""
import sys, json, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "."; FIG = f"{ROOT}/report/figs"; os.makedirs(FIG, exist_ok=True)
eff = pd.read_csv(sys.argv[1]); cb = json.load(open(sys.argv[2])); TAG = sys.argv[3] if len(sys.argv) > 3 else ""
import os
_mp = f"{ROOT}/results/{os.environ.get('CLS_TAG', 'ft')}_metrics.json"
if os.path.exists(_mp):
    _m = json.load(open(_mp)); eff = eff[eff.term.isin([f for f, v in _m.items() if v["test_f1"] >= 0.40 and v["n_pos_test"] >= 5])]
lab = {f["id"]: f["label"].split(" / ")[0] for f in cb["factors"]}
prev = eff[eff.outcome == "ret_d30"].rename(columns={"term": "factor"})[["scope", "factor", "prevalence"]]
GAMES = ["Counter-Strike", "PUBG", "Apex Legends", "Cyberpunk 2077", "Terraria"]
plt.rcParams.update({"font.size": 9})

# F3 forest per focus game (top factors by |effect| among flagged >= 200)
d30 = eff[(eff.outcome == "ret_d30") & (eff.n_flagged >= 200)]
fig, axes = plt.subplots(2, 3, figsize=(16, 11), sharex=False); axes = axes.ravel(); axes[-1].set_visible(False)
for ax, g in zip(axes, GAMES):
    s = d30[d30.scope == g].copy()
    if not len(s): ax.set_visible(False); continue
    s = s.reindex(s.beta.abs().sort_values(ascending=False).index).head(12).sort_values("beta")
    y = np.arange(len(s)); col = ["#c0392b" if (h < 0) else ("#1f8a4c" if l > 0 else "#888") for l, h in zip(s.lo90, s.hi90)]
    ax.errorbar(s.beta, y, xerr=[s.beta - s.lo90, s.hi90 - s.beta], fmt="none", ecolor=col, elinewidth=2)
    ax.scatter(s.beta, y, c=col, zorder=3); ax.axvline(0, color="k", lw=.8)
    ax.set_yticks(y); ax.set_yticklabels([lab.get(t, t)[:34] for t in s.term]); ax.set_title(g); ax.set_xlabel("Δ share active past day 30 (pp)")
fig.suptitle("Complaint factors and activity past day 30 (LPM, game×review-quarter FE, 90% CI, t with quarter clusters)", y=1.0)
fig.tight_layout(); fig.savefig(f"{FIG}/F3_forest_d30{TAG}.png", dpi=150, bbox_inches="tight")

# F4 priority matrix: focus-pooled (or ALL) and, if present, the other-games sample
def f4(scope, tag):
    p = d30[d30.scope == scope].copy()
    if not len(p): return
    fig, ax = plt.subplots(figsize=(9, 6.5))
    col = np.where(p.hi90 < 0, "#c0392b", np.where(p.lo90 > 0, "#1f8a4c", "#999999"))
    ax.scatter(p.prevalence * 100, p.beta, s=np.clip(p.n_flagged / 300, 10, 600), c=col, alpha=.7)
    for _, r in p.iterrows():
        if (r.prevalence > .02 or abs(r.beta) > 1.5) and (r.hi90 < 0 or r.lo90 > 0): ax.annotate(lab.get(r.term, r.term)[:30], (r.prevalence * 100, r.beta), fontsize=7.5, xytext=(4, 2), textcoords="offset points")
    ax.axhline(0, color="k", lw=.8); ax.set_xscale("log")
    ax.set_xlabel("prevalence among negative reviews (%, classifier, log scale)"); ax.set_ylabel("Δ still playing at day 30 (pp, adjusted)")
    ax.set_title(f"Priority matrix — {scope}\nred = significantly lower retention (90% CI), green = higher, grey = n.s.")
    fig.tight_layout(); fig.savefig(f"{FIG}/F4_priority_matrix{TAG}{tag}.png", dpi=150)
f4("FOCUS_POOLED" if (d30.scope == "FOCUS_POOLED").any() else "ALL", "")
f4("OTHER_GAMES_SAMPLE", "_other_games")

# F2 prevalence heatmap from classifier-scored corpus (scope rows per game)
pv = prev.pivot_table(index="factor", columns="scope", values="prevalence")
if len(pv):
    pv = pv.loc[pv.max(axis=1).sort_values(ascending=False).index[:35]]
    fig, ax = plt.subplots(figsize=(8, 11)); im = ax.imshow(pv.values * 100, aspect="auto", cmap="Blues")
    ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns, rotation=30, ha="right")
    ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels([lab.get(i, i)[:40] for i in pv.index])
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j] * 100
            if not np.isnan(v): ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5, color="white" if v > 15 else "black")
    fig.colorbar(im, ax=ax, label="% of negative reviews"); ax.set_title("Complaint factor prevalence")
    fig.tight_layout(); fig.savefig(f"{FIG}/F2_prevalence_heatmap{TAG}.png", dpi=150)

# F5 segments
seg = pd.read_csv(f"{ROOT}/results/behavior/T1_segments_focus.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for g, s in seg.groupby("game"):
    axes[0].plot(s.seg.str[2:], s.share * 100, marker="o", label=g); axes[1].plot(s.seg.str[2:], s.d30 * 100, marker="o", label=g)
axes[0].set_ylabel("% of the game's negative reviews"); axes[1].set_ylabel("% still playing 30 days after review")
for a in axes: a.set_xlabel("hours played when the review was written"); a.grid(alpha=.3)
axes[1].legend(fontsize=8); fig.suptitle("Not all negative reviews are equal: quick rejectors vs frustrated loyalists")
fig.tight_layout(); fig.savefig(f"{FIG}/F5_segments{TAG}.png", dpi=150)
print("figures written to", FIG)
