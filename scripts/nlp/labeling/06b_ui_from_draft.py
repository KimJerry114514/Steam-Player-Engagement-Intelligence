"""B-page checklist from an architect draft (measurement lens), with English display labels and family-group mapping."""
import json, sys
ROOT = "."
src = sys.argv[1]; d = json.load(open(src))
FAMGROUP = {
 "battle_royale": ["battle_royale"],
 "competitive_multiplayer": ["tactical_fps", "hero_arena_shooter", "battle_royale", "military_large_scale_shooter", "moba", "fighting", "vehicle_combat_sim", "sports_racing", "card_autobattler_board"],
 "rpg_and_branching_narrative": ["singleplayer_rpg_arpg", "open_world_action_rpg", "narrative_puzzle_adventure", "mmorpg", "anime_gacha_visual_novel"],
 "open_world_and_simulation": ["open_world_action_rpg", "simulation_management", "survival_sandbox_multiplayer", "sandbox_crafting_building"],
 "action_survival_and_run_based": ["soulslike_hardcore_action", "roguelike_roguelite", "action_adventure_platformer", "survival_sandbox_multiplayer", "sandbox_crafting_building", "pve_coop_shooter_looter"],
 "sandbox_crafting_building": ["sandbox_crafting_building", "survival_sandbox_multiplayer"],
 "horror": ["horror"]}
out = {"version": "draft-measurement", "source": src,
       "domains": [{"id": x["id"], "label": x["label"].split(" / ")[0], "description": x["description"]} for x in d["domains"]], "factors": []}
for f in d["factors"]:
    fams = FAMGROUP.get(f["family"], [f["family"]]) if f["level"] == "family" else []
    out["factors"].append({"id": f["id"], "label": f["label"].split(" / ")[0], "domain": f["domain"], "level": f["level"], "families": fams,
                           "game_appid": int(f.get("game_appid") or 0), "status": f["status"],
                           "definition": f["definition"] + ("  | Include: " + "; ".join(f["include_rules"][:3]) if f["include_rules"] else "")
                                         + ("  | Exclude: " + "; ".join(f["exclude_rules"][:3]) if f["exclude_rules"] else "")})
json.dump(out, open(f"{ROOT}/labels/gold/codebook_for_ui.json", "w"), ensure_ascii=False, indent=1)
print("factors", len(out["factors"]))
