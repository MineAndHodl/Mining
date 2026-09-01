"""Per-pick factor scorecard — a plain-language read of which model levers push FOR or
AGAINST the play, on a -1 (against) → 0 (no impact) → +1 (supports) scale, ALWAYS relative
to the side we're taking (so it's logical for overs AND unders: a lever that raises the
stat supports an over and opposes an under).

Plain names, no jargon, and no circular "market value" factor (the edge IS the Desk Score,
not a factor). These reflect real model inputs — an honest window into the model, not
separate alpha. Missing data = 0 (no impact), never a fake signal. Only the levers that
actually move a pick are shown; the rest are counted as "no impact".
"""
FACTORS = ["Projected Usage", "Recent Trend", "Game Flow", "Opponent Defense",
           "Injury Picture", "Weather", "Play-Calling", "Market Read", "Regression"]

DESC = {
    "Projected Usage": "how much volume we expect vs the line",
    "Recent Trend": "which way his recent games are trending",
    "Game Flow": "does the game script (favorite runs, dog throws) help this stat",
    "Opponent Defense": "how the opposing defense holds up vs this",
    "Injury Picture": "his status and any teammate absence that opens a role",
    "Weather": "wind/cold effect (suppresses passing, nudges rushing)",
    "Play-Calling": "the coach's tendencies (pass rate, red-zone/goal-line use)",
    "Market Read": "whether the betting market's role read agrees with ours",
    "Regression": "is he due (underperformed) or overdue to fall off (got lucky)",
}


def _for_side(raises_stat, side):
    if raises_stat == 0:
        return 0
    return raises_stat if side in ("Over", "Yes") else -raises_stat


def scorecard(pick, proj_row=None, script=None, weather=None, matchup=None,
              adp_role=None, regression=None):
    side = pick.get("side", "Over"); mkt = pick.get("mkt", "")
    line = float(pick.get("line", 0) or 0)
    our = float(pick.get("our_num", pick.get("desk_proj", line)) or line)
    sc = {}
    sc["Projected Usage"] = _for_side(1 if our > line else -1 if our < line else 0, side)

    fd = (proj_row or {}).get("_form_delta") if proj_row else None
    sc["Recent Trend"] = _for_side(1 if (fd or 0) > 1 else -1 if (fd or 0) < -1 else 0, side) if fd is not None else 0

    sc["Game Flow"] = 0
    if script:
        ts = script.get("team_spread", 0)
        if mkt in ("rec_yds", "receptions", "pass_yds"):
            sc["Game Flow"] = _for_side(1 if ts >= 3 else -1 if ts <= -6 else 0, side)
        elif mkt in ("rush_yds", "rush_att"):
            sc["Game Flow"] = _for_side(1 if ts <= -6 else -1 if ts >= 4 else 0, side)
        elif mkt == "anytime_td":
            it = script.get("implied_team_pts", 22)
            sc["Game Flow"] = _for_side(1 if it >= 26 else -1 if it <= 18 else 0, side)

    sc["Opponent Defense"] = 0
    if mkt in ("rush_yds", "rush_att") and matchup is not None:
        sc["Opponent Defense"] = _for_side(1 if matchup > 0.05 else -1 if matchup < -0.05 else 0, side)

    sc["Injury Picture"] = 0
    if pick.get("_health"):
        h = pick["_health"]
        if h.get("status") in ("Out", "Doubtful"):
            sc["Injury Picture"] = _for_side(-1, side)
        elif h.get("status") == "Questionable" and side in ("Over", "Yes"):
            sc["Injury Picture"] = _for_side(-1, side)
        elif h.get("opportunity"):
            sc["Injury Picture"] = _for_side(1, side)

    sc["Weather"] = 0
    if weather and ((weather.get("wind", 0) or 0) >= 12 or (weather.get("temp", 99) or 99) <= 35):
        if mkt in ("rec_yds", "receptions", "pass_yds"):
            sc["Weather"] = _for_side(-1, side)
        elif mkt in ("rush_yds", "rush_att"):
            sc["Weather"] = _for_side(1, side)

    sc["Play-Calling"] = _for_side((proj_row or {}).get("_coach_signal", 0), side) if proj_row else 0

    sc["Market Read"] = 0
    if adp_role in ("big", "rotational"):
        sc["Market Read"] = _for_side(1, side)
    elif adp_role in ("undrafted", "deep"):
        sc["Market Read"] = _for_side(-1, side)

    sc["Regression"] = 0
    if regression is not None:
        sc["Regression"] = _for_side(-1 if regression > 0.2 else 1 if regression < -0.2 else 0, side)
    return sc


def top_factors(sc, n=6):
    active = [(f, v) for f, v in sc.items() if v != 0]
    order = {f: i for i, f in enumerate(FACTORS)}
    active.sort(key=lambda x: (x[1] < 0, order.get(x[0], 99)))
    return active[:n], sum(1 for v in sc.values() if v == 0)


if __name__ == "__main__":
    p = {"side": "Under", "mkt": "receptions", "line": 4.5, "our_num": 2.9, "edge": 0.11}
    sc = scorecard(p, script={"team_spread": -7, "implied_team_pts": 27}, adp_role="undrafted", regression=0.3)
    for f, v in sc.items():
        if v: print(f"  {f}: {'supports' if v>0 else 'against'} — {DESC[f]}")
