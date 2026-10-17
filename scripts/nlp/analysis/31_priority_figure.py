"""Publication-quality priority matrix: short labels, no overlaps, readable title. Usage:
python 31_priority_figure.py <effects_csv> <metrics_json> <scope> <title> <out_png>"""
import sys, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text
eff_p, met_p, scope, title, out = sys.argv[1:6]
SHORT = {"tech.performance": "Performance", "tech.stability": "Crashes / launch", "tech.functionality": "Broken functions", "tech.save_integrity": "Save / data loss",
         "tech.network.responsiveness": "Network lag / desync", "tech.network.session_access": "Connection failures", "tech.platform_compatibility": "Compatibility",
         "gameplay.core_loop": "Play rhythm / feel", "gameplay.balance": "Balance", "gameplay.challenge": "Difficulty", "gameplay.ai_simulation": "AI / simulation",
         "gameplay.agency": "Meaningful choices", "gameplay.randomness": "Randomness", "progression.effort_reward": "Grind / rewards", "content.scope_variety": "Content breadth",
         "content.narrative": "Story", "content.world_level": "World / level design", "content.audiovisual": "Audiovisual", "content.franchise_identity": "Originality",
         "social.cheating": "Cheating", "social.skill_matchmaking": "Matchmaking", "social.match_availability": "Few players / queues", "social.cooperation": "Co-op",
         "social.conduct": "Toxic players", "social.sanctions": "Disputed bans*", "ux.interface": "UI", "ux.controls": "Controls", "ux.onboarding": "Onboarding / goals",
         "ux.localization": "Localization", "ux.accessibility": "Comfort / accessibility", "business.price_value": "Price / value", "business.paid_advantage": "Pay-to-win",
         "business.paywall": "Paywall", "business.paid_randomness": "Loot boxes", "business.sales_pressure": "Sales pressure", "business.access_requirements": "Access requirements",
         "business.ownership": "Ownership", "business.account_security": "Stolen accounts*", "business.privacy_security": "Privacy", "service.maintenance": "Neglected maintenance",
         "service.marketing_trust": "Misleading marketing", "service.communication": "Dev communication", "service.support": "Support", "service.release_readiness": "Unfinished release",
         "service.update_regression": "Updates made it worse", "social.governance": "Admin abuse",
         "game.cs.free_account_cheating": "CS: free-account cheaters", "game.cyberpunk.city_interactivity": "Cyberpunk: city interaction",
         "game.cyberpunk.police_response": "Cyberpunk: police response"}
import os
MIN_ABS = float(os.environ.get("LABEL_MIN_ABS", "0"))  # label only effects at least this large (pp)
met = json.load(open(met_p)); ok = {f for f, m in met.items() if m["test_f1"] >= 0.40 and m["n_pos_test"] >= 5}
e = pd.read_csv(eff_p); p = e[(e.outcome == "ret_d30") & (e.scope == scope) & e.term.isin(ok) & (e.n_flagged >= 300)].copy()
sig = (p.hi90 < 0) | (p.lo90 > 0)
col = np.where(p.hi90 < 0, "#c0392b", np.where(p.lo90 > 0, "#1f8a4c", "#b0b0b0"))
plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(figsize=(10, 6.2))
ax.scatter(p.prevalence * 100, p.beta, s=25 + 875 * p.n_flagged / p.n_flagged.max(), c=col, alpha=.75, edgecolors="white", linewidths=.6, zorder=3)
texts = [ax.text(r.prevalence * 100, r.beta, SHORT.get(r.term, r.term), fontsize=10.5, color="#222") for _, r in p[sig & ((p.beta.abs() >= MIN_ABS) | (p.prevalence >= 0.10))].iterrows()]
ax.axhline(0, color="#444", lw=.9, zorder=1); ax.set_xscale("log"); ax.grid(alpha=.25, zorder=0)
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#888", lw=.6), expand=(1.3, 1.6), force_text=(0.4, 0.8))
ax.set_xlabel("Share of negative reviews flagged for the complaint (%, classifier, log scale)")
ax.set_ylabel("Δ share active past day 30\n(percentage points, adjusted)")
ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.text(0.01, 0.012, "Red = lower share active past day 30, green = higher (90% CI excludes 0); grey = not significant. Bubble area ∝ number of flagged reviews."
         + (f"\nLabels: |effect| ≥ {MIN_ABS:g} pp or flagged share ≥ 10%." if MIN_ABS else "") + "\n"
         "* Bans may partly limit play mechanically; the stolen-account association is not confirmed outside the deep-dive titles.", ha="left", va="bottom", fontsize=9, color="#555")
fig.savefig(out, dpi=200); print("saved", out, "labels", len(texts))
