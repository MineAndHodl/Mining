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


def _clip(x):
    return max(-1.0, min(1.0, x))


def _for_side(raises_stat, side):
    """raises_stat is a lean in [-1,1] (how much it pushes the STAT up). Flip for unders."""
    return raises_stat if side in ("Over", "Yes") else -raises_stat


def scorecard(pick, proj_row=None, script=None, weather=None, matchup=None,
              adp_role=None, adp=None, regression=None):
    """Continuous lean in [-1,1] for EVERY factor (0 = no read). Signs flipped for the
    side so it reads correctly for overs and unders. Each factor is independent."""
    side = pick.get("side", "Over"); mkt = pick.get("mkt", "")
    line = float(pick.get("line", 0) or 0)
    our = float(pick.get("our_num", pick.get("desk_proj", line)) or line)
    pr = proj_row or {}
    sc = {}

    # Projected Usage: how far our number sits from the line, normalized (scales with gap)
    gap = (our - line) / max(abs(line), 1.0)
    sc["Projected Usage"] = _for_side(_clip(gap / 0.25), side)

    # Recent Trend: form delta vs career (yards), continuous
    fd = pr.get("_form_delta")
    sc["Recent Trend"] = _for_side(_clip((fd or 0) / 12.0), side) if fd is not None else 0.0

    # Game Flow: continuous off the spread (dog throws, favorite runs)
    gf = 0.0
    if script:
        ts = script.get("team_spread", 0) or 0
        if mkt in ("rec_yds", "receptions", "pass_yds"):
            gf = _clip(ts / 9.0)              # +ts (dog) raises passing
        elif mkt in ("rush_yds", "rush_att"):
            gf = _clip(-ts / 9.0)            # -ts (favorite) raises rushing
        elif mkt == "anytime_td":
            gf = _clip((script.get("implied_team_pts", 22) - 22) / 8.0)
    sc["Game Flow"] = _for_side(gf, side)

    # Opponent Defense: validated only for rush (RB-rush signal); continuous
    od = 0.0
    if mkt in ("rush_yds", "rush_att") and matchup is not None:
        od = _clip(matchup / 0.15)
    sc["Opponent Defense"] = _for_side(od, side)

    # Injury Picture
    inj = 0.0
    if pick.get("_health"):
        h = pick["_health"]; st = h.get("status")
        if st in ("Out", "Doubtful"): inj = -1.0
        elif st == "Questionable": inj = -0.4
        elif h.get("opportunity"): inj = 0.6
    sc["Injury Picture"] = _for_side(inj, side)

    # Weather (pass markets down, rush up) — magnitude by severity
    wx = 0.0
    if weather:
        wind = weather.get("wind", 0) or 0; temp = weather.get("temp", 70) or 70
        sev = _clip(max(wind - 8, 0) / 15.0 + max(55 - temp, 0) / 40.0)
        if mkt in ("rec_yds", "receptions", "pass_yds"): wx = -sev
        elif mkt in ("rush_yds", "rush_att"): wx = sev * 0.5
    sc["Weather"] = _for_side(wx, side)

    # Play-Calling: coach signal from projection layer (continuous if provided)
    sc["Play-Calling"] = _for_side(_clip(pr.get("_coach_signal", 0)), side)

    # Market Read: ADP rank → role read, continuous (early pick = big role)
    mr = 0.0
    if adp is not None and not (isinstance(adp, float) and adp != adp):
        mr = _clip((80 - float(adp)) / 80.0)     # ADP 1 → +0.99, 80 → 0, 160 → -1
    elif adp_role in ("big", "rotational"): mr = 0.6
    elif adp_role in ("undrafted", "deep"): mr = -0.6
    sc["Market Read"] = _for_side(mr, side)

    # Regression (luck): overperformer → down, underperformer → up, continuous
    rg = 0.0
    if regression is not None:
        rg = _clip(-regression)                  # +regression (overperformed) lowers stat
    sc["Regression"] = _for_side(rg, side)

    return {k: round(float(v), 2) for k, v in sc.items()}


if __name__ == "__main__":
    p = {"side": "Under", "mkt": "receptions", "line": 4.5, "our_num": 2.9, "edge": 0.11}
    sc = scorecard(p, script={"team_spread": -7, "implied_team_pts": 27}, adp_role="undrafted", regression=0.3)
    for f, v in sc.items():
        if v: print(f"  {f}: {'supports' if v>0 else 'against'} — {DESC[f]}")
