"""Per-pick factor scorecard — how each model factor pushes FOR or AGAINST the play.

Each factor returns +1 (supports the pick), 0 (no impact), or -1 (works against it) —
ALWAYS relative to the side we're taking, so it's logical for overs AND unders:
  a factor that raises the stat supports an OVER and opposes an UNDER, and vice-versa.
The green/grey/red scorecard is the visual "why" behind every pick — not super
quantitative, just an honest read of which levers help and which fight it.

Factors: Role/Volume, Recent Form, Game Script, Matchup, Weather, Coaching, Consensus,
Regression, Market Value. Missing data = 0 (no impact), never a fake signal.
"""
import numpy as np

FACTORS = ["Role / Volume", "Recent Form", "Game Script", "Matchup", "Health / Injury",
           "Weather", "Coaching", "Consensus", "Regression", "Market Value"]


def _sign_for_side(raises_stat, side):
    """raises_stat: +1 if the factor pushes the stat UP, -1 down, 0 none.
    Return contribution to the PICK given the side (Over/Yes vs Under)."""
    if raises_stat == 0:
        return 0
    over = side in ("Over", "Yes")
    return raises_stat if over else -raises_stat


def scorecard(pick, proj_row=None, script=None, weather=None, matchup=None,
              adp_role=None, regression=None):
    """Return {factor: -1/0/1} relative to the pick's side. Everything optional."""
    side = pick.get("side", "Over")
    mkt = pick.get("mkt", "")
    line = float(pick.get("line", 0) or 0)
    our = float(pick.get("our_num", pick.get("desk_proj", line)) or line)
    sc = {}

    # Role / Volume — does our projected number sit above or below the line?
    sc["Role / Volume"] = _sign_for_side(1 if our > line else -1 if our < line else 0, side)

    # Recent Form — trend vs career (raises stat if trending up)
    if proj_row is not None and proj_row.get("_form_delta") is not None:
        fd = proj_row.get("_form_delta")
        sc["Recent Form"] = _sign_for_side(1 if fd > 1 else -1 if fd < -1 else 0, side)
    else:
        sc["Recent Form"] = 0

    # Game Script — dog throws (raises pass/rec), favorite runs (raises rush)
    sc["Game Script"] = 0
    if script:
        ts = script.get("team_spread", 0)
        if mkt in ("rec_yds", "receptions", "pass_yds"):
            sc["Game Script"] = _sign_for_side(1 if ts >= 3 else -1 if ts <= -6 else 0, side)
        elif mkt in ("rush_yds", "rush_att"):
            sc["Game Script"] = _sign_for_side(1 if ts <= -6 else -1 if ts >= 4 else 0, side)
        elif mkt == "anytime_td":
            # high implied team total raises TD odds
            it = script.get("implied_team_pts", 22)
            sc["Game Script"] = _sign_for_side(1 if it >= 26 else -1 if it <= 18 else 0, side)

    # Matchup — only the validated RB-rush signal (opponent soft vs run raises rush)
    sc["Matchup"] = 0
    if mkt in ("rush_yds", "rush_att") and matchup is not None:
        sc["Matchup"] = _sign_for_side(1 if matchup > 0.05 else -1 if matchup < -0.05 else 0, side)

    # Weather — wind/cold lowers passing (raises rush slightly)
    sc["Weather"] = 0
    if weather:
        bad = (weather.get("wind", 0) or 0) >= 12 or (weather.get("temp", 99) or 99) <= 35
        if bad:
            if mkt in ("rec_yds", "receptions", "pass_yds"):
                sc["Weather"] = _sign_for_side(-1, side)      # lowers stat
            elif mkt in ("rush_yds", "rush_att"):
                sc["Weather"] = _sign_for_side(1, side)

    # Coaching — RZ/goal-line role raises TD; pass-heavy PROE raises pass/rec
    sc["Coaching"] = 0
    if proj_row is not None and proj_row.get("_coach_signal"):
        sc["Coaching"] = _sign_for_side(proj_row["_coach_signal"], side)

    # Consensus — ADP role read: big role raises stat expectation
    sc["Consensus"] = 0
    if adp_role in ("big", "rotational"):
        sc["Consensus"] = _sign_for_side(1, side)
    elif adp_role in ("undrafted", "deep"):
        sc["Consensus"] = _sign_for_side(-1, side)

    # Regression — overperformer regresses down (lowers stat), underperformer up
    sc["Regression"] = 0
    if regression is not None:
        sc["Regression"] = _sign_for_side(-1 if regression > 0.2 else 1 if regression < -0.2 else 0, side)

    # Health / Injury — the player's own status hurts; a teammate's absence opening a
    # role helps. `health` dict: {"status": "Q/Out/None", "opportunity": bool}
    sc["Health / Injury"] = 0
    if pick.get("_health"):
        h = pick["_health"]
        if h.get("status") in ("Out", "Doubtful"):
            sc["Health / Injury"] = _sign_for_side(-1, side)      # can't produce
        elif h.get("status") == "Questionable":
            sc["Health / Injury"] = _sign_for_side(-1, side) if side in ("Over", "Yes") else 0
        elif h.get("opportunity"):
            sc["Health / Injury"] = _sign_for_side(1, side)       # teammate out → more volume

    # Market Value — the edge itself always supports the pick (that's why it's a pick)
    edge = float(pick.get("edge", 0) or 0)
    sc["Market Value"] = 1 if edge >= 0.04 else 0

    return sc


def top_factors(sc, n=6):
    """Return only the factors that actually move the pick (non-zero), most decisive first,
    plus a count of neutral ones — the 'biggest contributing factors' view."""
    active = [(f, v) for f, v in sc.items() if v != 0]
    # market value + role first, then others; keep it readable
    order = {f: i for i, f in enumerate(FACTORS)}
    active.sort(key=lambda x: (x[1] < 0, order.get(x[0], 99)))   # supports first, then against
    neutral = sum(1 for v in sc.values() if v == 0)
    return active[:n], neutral


def summarize(sc):
    pos = [f for f, v in sc.items() if v > 0]
    neg = [f for f, v in sc.items() if v < 0]
    return {"support": pos, "against": neg, "net": len(pos) - len(neg)}


if __name__ == "__main__":
    p = {"side": "Under", "mkt": "receptions", "line": 4.5, "our_num": 2.9, "edge": 0.11}
    sc = scorecard(p, script={"team_spread": -7, "implied_team_pts": 27}, adp_role="undrafted", regression=0.3)
    for f, v in sc.items():
        print(f"  {f:14s}: {'＋ supports' if v>0 else '－ against' if v<0 else '· no impact'}")
    print(summarize(sc))
