"""Per-pick context bundle — the raw material for a UNIQUE, sells-the-model rationale.

A Python template repeats itself. The fix: assemble every specific fact about THIS
pick — the player's role and form, the game script from the spread/total, the coach's
tendencies, the matchup, the weather, ADP consensus, role momentum, and (at run time)
beat-writer news — into one structured bundle. The rationale is then WRITTEN by the AI
from this bundle, so it's naturally unique to the player, game, and situation.

The scheduled pick-run does the final step: takes this bundle, adds a live beat-writer
web search for the player, and writes a bespoke 2-3 sentence 'why' — the hook.
"""
import os, json
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def _load(path, default=None):
    p = os.path.join(DATA, path)
    if os.path.exists(p):
        return json.load(open(p)) if path.endswith(".json") else pd.read_parquet(p)
    return default


def game_script(spread_home, total, team_is_home):
    """Turn the number into narrative: favored/dog + shootout/grind + implied team total."""
    if spread_home is None or total is None:
        return {}
    team_spread = spread_home if team_is_home else -spread_home
    implied = total/2 - team_spread/2      # this team's implied points
    role = ("a big favorite" if team_spread <= -7 else "a favorite" if team_spread < 0
            else "a big underdog" if team_spread >= 7 else "an underdog" if team_spread > 0 else "a pick'em")
    pace = ("a shootout" if total >= 49 else "a grind" if total <= 41 else "a normal-tempo game")
    script_note = ""
    if team_spread >= 4:    # this team a dog → likely trailing → pass volume up
        script_note = "trailing script points to pass volume late"
    elif team_spread <= -6:  # favorite → likely leading → run volume / clock
        script_note = "leading script points to run volume and clock control"
    return {"team_role": role, "game_flavor": pace, "implied_team_pts": round(implied, 1),
            "script_note": script_note, "team_spread": team_spread, "total": total}


def build_context(pick, proj_row=None, spread_home=None, total=None, team_is_home=None,
                  weather=None, adp_row=None, trending=None):
    """Assemble the bundle for one pick. `pick` has name/mkt/line/side/proj/edge.
    Everything optional degrades gracefully — the AI writes from whatever is present."""
    coach = _load("coaching_findings.json", {})
    gl = _load("coach_gl_concentration.json", {})
    b = {"player": pick.get("name"), "market": pick.get("mkt"), "line": pick.get("line"),
         "side": pick.get("side"), "our_number": pick.get("desk_proj") or pick.get("proj"),
         "edge_pts": round((pick.get("edge") or 0)*100, 1)}
    # role / form
    if proj_row is not None:
        b["role"] = {"proj_targets_pg": round(float(proj_row.get("proj_targets_pg", 0)), 1),
                     "proj_carries_pg": round(float(proj_row.get("proj_carries_pg", 0)), 1),
                     "position": proj_row.get("position"), "moved_teams": bool(proj_row.get("moved", False)),
                     "rookie": bool(proj_row.get("rookie", False)),
                     "confidence": round(float(proj_row.get("confidence", 0)), 2)}
    # game script
    gs = game_script(spread_home, total, team_is_home)
    if gs:
        b["game_script"] = gs
    # weather (only if it matters for this market)
    if weather and pick.get("mkt") in ("rec_yds", "pass_yds", "receptions", "rush_yds"):
        w = weather
        if (w.get("wind") or 0) >= 12 or (w.get("temp") or 99) <= 35:
            b["weather"] = {"wind": w.get("wind"), "temp": w.get("temp"),
                            "effect": "wind/cold suppresses passing — supports unders on pass-based props, slight rush boost"}
    # matchup (only the validated one)
    if pick.get("mkt") == "rush_yds":
        b["matchup"] = "RB-vs-defense has a small validated signal (opponent-adjusted) — factor it lightly"
    # consensus + momentum
    if adp_row is not None and pd.notna(adp_row.get("adp")):
        b["consensus"] = {"adp": adp_row.get("adp"), "adp_uncertainty": adp_row.get("stdev"),
                          "note": "ADP is the draft market's role read — agreement adds confidence, big disagreement is a flag"}
    if trending:
        b["role_momentum"] = trending   # e.g. "51k adds on Sleeper this week — role is rising"
    # honest confidence framing
    conf = (proj_row.get("confidence", 0.5) if proj_row is not None else 0.5)
    b["confidence_framing"] = ("stable multi-year role — our calibrated zone" if conf >= 0.67
                               else "changed role — lean, not a hammer")
    return b


# The instruction the pick-run gives itself/the model to turn a bundle into the hook:
RATIONALE_PROMPT = """Write a 2-3 sentence handicapper rationale for this pick from the bundle
below. Lead with the single most important factor FOR THIS PLAYER (role, game script, coach
tendency, matchup, weather, consensus, or the beat news) — never the same opener twice. Weave
in one specific number. Sound like a sharp desk analyst who did the work, not a template.
End with the honest confidence. If beat-writer news is provided, use it to support the for/fade.
BUNDLE:\n{bundle}"""


if __name__ == "__main__":
    demo = {"name": "Rashid Shaheed", "mkt": "rec_yds", "line": 31.5, "side": "Over",
            "desk_proj": 41.4, "edge": 0.063}
    b = build_context(demo, spread_home=3.5, total=47.5, team_is_home=False,
                      trending="quietly the Saints' clear deep threat with Olave banged up (beat reports)")
    print(json.dumps(b, indent=2, default=str))
