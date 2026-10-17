# Original-language annotation keys and search terms below are research data, not submission instructions.
"""Human gold set A (80 reviews, annotator's own reason chips) vs LLM labels (codebook v1.1).
Chips are mapped to groups of v1.1 factors with a fixed, published table; a chip counts as matched when LLM tagged ANY
factor in its group. Also compares review-level fields. Output: results/labels/gold_A_eval.json (+ printed summary)."""
import json, collections, numpy as np, pandas as pd, os
ROOT = "."
CHIP2V11 = {
 "外挂/作弊": ["social.cheating", "game.cs.free_account_cheating"], "反作弊不作为/封得慢": ["social.cheating", "game.cs.free_account_cheating"],
 "误封/封号争议": ["social.sanctions"], "账号被盗/安全": ["business.account_security"],
 "服务器/延迟/掉线": ["tech.network.responsiveness", "tech.network.session_access"], "网络同步/命中判定": ["tech.network.responsiveness"],
 "优化差/掉帧/卡顿": ["tech.performance"], "崩溃/闪退/无法启动": ["tech.stability"], "Bug/故障": ["tech.functionality", "tech.stability", "game.cyberpunk.quest_state"],
 "存档/进度丢失": ["tech.save_integrity"], "匹配机制/段位不公/炸鱼": ["social.skill_matchmaking", "fam.ranked_competitive.rating"],
 "平衡性（武器/角色/职业）": ["gameplay.balance", "fam.hero_team.role_balance", "fam.competitive_shooter.input_balance"],
 "太难/卡关/难度曲线": ["gameplay.challenge", "fam.action.boss_challenge", "game.terraria.boss_difficulty"],
 "内容少/重复/无聊": ["content.scope_variety", "gameplay.core_loop"], "太肝/刷/进度慢": ["progression.effort_reward"],
 "剧情/叙事": ["content.narrative", "fam.rpg.choice_consequence"], "画面/美术/音效": ["content.audiovisual"],
 "操作/手感/键位": ["ux.controls", "gameplay.core_loop"], "UI/界面": ["ux.interface"], "新手引导/教程不足": ["ux.onboarding"],
 "价格/不值": ["business.price_value"], "氪金/开箱/抽卡/付费变强": ["business.paid_randomness", "business.paid_advantage"],
 "DLC/内容拆分": ["business.paywall", "business.ownership"], "更新改坏了/版本变差": ["service.update_regression"],
 "官方态度/不沟通/不修": ["service.communication", "service.maintenance", "service.support"], "客服差": ["service.support"],
 "社区/队友毒性": ["social.conduct"], "地区/语言/锁区/汉化": ["ux.localization", "business.access_requirements"],
 "强制第三方账号/启动器": ["business.access_requirements"], "反作弊软件侵入/强制安装": ["business.privacy_security"],
 "与宣传不符": ["service.marketing_trust", "service.release_readiness"], "多人/联机体验差": ["social.cooperation", "tech.network.session_access"],
}
FREE2V11 = {"地图太大": ["content.world_level"], "NPC太笨": ["gameplay.ai_simulation"], "有木马病毒?": ["business.privacy_security"], "游戏性不足": ["gameplay.core_loop", "content.scope_variety"]}
import sys
SUB = sys.argv[1] if len(sys.argv) > 1 else "final"
lab = pd.read_parquet(f"{ROOT}/results/labels/{SUB}/labels_primary.parquet")
FIDS = [c for c in lab.columns if "." in c]
items = {x["gid"]: x for x in json.load(open(f"{ROOT}/labels/gold/gold_A_items.json"))}
G = json.load(open(f"{ROOT}/labels/gold/gold_A_labels.json"))
rows, per = [], collections.defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})
for gid, g in G.items():
    c = lab[lab.rid == items[gid]["rid"]]
    if not len(c): continue
    c = c.iloc[0]; cset = {f for f in FIDS if c[f] == 1}
    hum_groups = {ch: CHIP2V11[ch] for ch in g["chips"] if ch in CHIP2V11}
    for fr in g.get("free", []): hum_groups[fr] = FREE2V11.get(fr, [])
    covered = set()
    for ch, grp in hum_groups.items():
        hit = bool(cset & set(grp)); per[ch]["tp" if hit else "fn"] += 1; covered |= set(grp)
    extra = cset - covered  # LLM factors not matching any human reason
    for f in extra: per["(llm-only) " + f]["fp"] += 1
    rows.append({"gid": gid, "n_human_reasons": len(hum_groups), "matched": sum(bool(cset & set(v)) for v in hum_groups.values()),
                 "llm_factors": len(cset), "llm_extra": len(extra), "h_pol": g["polarity"], "c_pol": c.polarity,
                 "h_att": (g.get("attributable") or ("yes" if hum_groups else "no")) == "yes", "c_att": bool(c.attributable),
                 "h_quit": g["self_reported_quit"], "c_quit": bool(c.quit), "h_ref": g["refund_intent"], "c_ref": bool(c.refund)})
R = pd.DataFrame(rows)
recall = R.matched.sum() / max(R.n_human_reasons.sum(), 1)
precision_proxy = 1 - R.llm_extra.sum() / max(R.llm_factors.sum(), 1)
summ = {"n": len(R), "reason_recall": recall, "llm_precision_proxy": precision_proxy,
        "polarity_exact": (R.h_pol == R.c_pol).mean(),
        "negative_or_mixed_agreement": (R.h_pol.isin(["negative", "mixed"]) == R.c_pol.isin(["negative", "mixed"])).mean(),
        "attributable_agreement": (R.h_att == R.c_att).mean(), "quit_agreement": (R.h_quit == R.c_quit).mean(), "refund_agreement": (R.h_ref == R.c_ref).mean()}
json.dump({"summary": summ, "per_reason": per, "rows": rows}, open(f"{ROOT}/results/labels/{SUB}/gold_A_eval.json", "w"), ensure_ascii=False, indent=1, default=float)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()})
for k, v in sorted(per.items(), key=lambda kv: -(kv[1]["tp"] + kv[1]["fn"] + kv[1]["fp"]))[:25]: print(k, v)
