"""Per-pick factor scorecard — a plain-language read of which model levers push FOR or
AGAINST the play, on a -1 (against) -> 0 (neutral) -> +1 (supports) scale, ALWAYS relative
to the side we're taking (so it's logical for overs AND unders: a lever that raises the
stat supports an over and opposes an under).

Design principle (Alex's rule): every factor should carry a real lean whenever we have
ANY signal for it. A factor sits at exactly 0 only when it is genuinely neutral for this
pick (a dome kills Weather; an established mid-round starter has no Regression pull) or
truly not applicable (Opponent Defense on a passing prop we haven't validated). We return
None for "not applicable / no data" so the UI can show a flat dash, and a real number
(including a small one) for "checked, this is the lean" — no more wall of silent zeros.

Plain names, no jargon, no circular "market value" factor (the edge IS the Desk Score).
These reflect real model inputs — an honest window into the model, not separate alpha.
"""
FACTORS = ["Projected Usage", "Recent Trend", "Game Flow", "Opponent Defense",
           "Injury Picture", "Weather", "Play-Calling", "Market Read", "Regression"]

DESC = {
    "Projected Usage": "how much volume we expect vs the line",
    "Recent Trend": "which way his recent form is trending vs his role baseline",
    "Game Flow": "does the game script (favorite runs, dog throws) help this stat",
    "Opponent Defense": "how the opposing defense holds up vs this position",
    "Injury Picture": "his status and any teammate absence that opens a role",
    "Weather": "wind/cold effect (suppresses passing, nudges rushing)",
    "Play-Calling": "how central he is to the offense (target/carry load)",
    "Market Read": "whether the draft market's role read agrees with ours",
    "Regression": "is he due (underperformed) or overdue to fall off (got lucky)",
}

PASS_MKTS = ("rec_yds", "receptions", "pass_yds", "pass_att", "pass_completions")
RUSH_MKTS = ("rush_yds", "rush_att", "carries")


def _clip(x):
    return max(-1.0, min(1.0, x))


def _for_side(raises_stat, side):
    """raises_stat is a lean in [-1,1] (how much it pushes the STAT up). Flip for unders/no."""
    if raises_stat is None:
        return None
    return raises_stat if side in ("Over", "Yes") else -raises_stat


def _own_vs_role(pr, mkt):
    """Form/luck signal: how far his OWN recent production sits above/below the role model,
    normalized. + means he's been outproducing his role (hot / lucky); - means underusing it.
    Available directly from the projection for receiving markets."""
    own = pr.get("own_rec_yds"); role = pr.get("role_rec_yds")
    if own is None or role is None or not role:
        return None
    return _clip((own - role) / max(role, 1.0))     # e.g. +30% -> +0.3


def scorecard(pick, proj_row=None, script=None, weather=None, matchup=None,
              adp_role=None, adp=None, regression=None):
    """Continuous lean in [-1,1] for EVERY factor (0 = checked-neutral, None = n/a).
    Signs flipped for the side so it reads correctly for overs and unders."""
    side = pick.get("side", "Over"); mkt = pick.get("mkt", "")
    line = float(pick.get("line", 0) or 0)
    our = float(pick.get("our_num", pick.get("desk_proj", line)) or line)
    pr = proj_row or {}
    hist = float(pr.get("hist_games", 0) or 0)
    sc = {}

    # --- Projected Usage: how far our number sits from the line (always available) ---
    if mkt == "anytime_td":
        # our_num here is a SCORING PROBABILITY, not a yardage number vs a 0.5 line — comparing
        # it to 0.5 wrongly renders every TD pick "against". Use scoring prob vs a league
        # baseline (~0.32) so a likelier scorer supports the Yes.
        sc["Projected Usage"] = _for_side(_clip((our - 0.32) / 0.20), side)
    else:
        gap = (our - line) / max(abs(line), 1.0)
        sc["Projected Usage"] = _for_side(_clip(gap / 0.25), side)

    # --- Recent Trend: explicit form delta if given, else own-vs-role from the projection ---
    fd = pr.get("_form_delta")
    if fd is not None:
        sc["Recent Trend"] = _for_side(_clip(fd / 12.0), side)
    else:
        ovr = _own_vs_role(pr, mkt)
        # form signal only meaningful for receiving markets (that's what own/role covers)
        sc["Recent Trend"] = _for_side(_clip((ovr or 0) * 1.4), side) if (ovr is not None and mkt in PASS_MKTS) else None

    # --- Game Flow: off the spread (dog throws, favorite runs) ---
    gf = None
    if script and script.get("team_spread") is not None:
        ts = script.get("team_spread", 0) or 0
        if mkt in PASS_MKTS:
            gf = _clip(ts / 9.0)              # +ts (dog, trailing) raises passing volume
        elif mkt in RUSH_MKTS:
            gf = _clip(-ts / 9.0)             # -ts (favorite, leading) raises rushing
        elif mkt == "anytime_td":
            gf = _clip((script.get("implied_team_pts", 22) - 22) / 8.0)
    sc["Game Flow"] = _for_side(gf, side)

    # --- Opponent Defense: validated hardest for rush; softer read for receiving ---
    od = None
    if matchup is not None:
        # matchup is (defense allowed / league avg) - 1: + means a soft matchup (raises stat)
        if mkt in RUSH_MKTS:
            od = _clip(matchup / 0.15)
        elif mkt in PASS_MKTS:
            od = _clip(matchup / 0.20) * 0.7   # softer weight for the less-validated pass read
    sc["Opponent Defense"] = _for_side(od, side)

    # --- Injury Picture: his own status + any opportunity opened by a teammate absence ---
    inj = None
    h = pick.get("_health")
    if h:
        st = h.get("status")
        if st in ("Out", "Doubtful"): inj = -1.0
        elif st == "Questionable": inj = -0.4
        elif h.get("opportunity"): inj = 0.6
        else: inj = 0.0                        # checked, healthy, no vacated role
    sc["Injury Picture"] = _for_side(inj, side)

    # --- Weather: pass down, rush up; dome/roof = genuine 0; unknown outdoor = None ---
    wx = None
    if weather is not None:
        roof = weather.get("roof"); roof = (roof if isinstance(roof, str) else "").lower()
        if roof in ("dome", "closed", "indoors"):
            wx = 0.0                           # genuinely neutral, not "no read"
        else:
            wind = weather.get("wind"); temp = weather.get("temp")
            if wind is None and temp is None:
                wx = None                      # outdoor but no forecast yet
            else:
                sev = _clip(max((wind or 0) - 8, 0) / 15.0 + max(55 - (temp if temp is not None else 70), 0) / 40.0)
                if mkt in PASS_MKTS: wx = -sev
                elif mkt in RUSH_MKTS: wx = sev * 0.5
                else: wx = -sev * 0.5
    sc["Weather"] = _for_side(wx, side)

    # --- Play-Calling: how central he is to the offense (target/carry load) ---
    cs = pr.get("_coach_signal")
    if cs is None:
        tgt = pr.get("proj_targets_pg"); car = pr.get("proj_carries_pg")
        if mkt in ("rec_yds", "receptions") and tgt is not None:
            cs = _clip((tgt - 5.5) / 5.0)      # 5.5 tgt/gm ~ neutral, 10+ = clear WR1 load
        elif mkt in ("rush_yds", "rush_att", "carries") and car is not None:
            cs = _clip((car - 10.0) / 8.0)     # 10 carries ~ neutral, 18+ = bell cow
        elif mkt == "pass_yds":
            cs = 0.0
    sc["Play-Calling"] = _for_side(_clip(cs) if cs is not None else None, side)

    # --- Market Read: ADP rank -> role read (early pick = big role the market believes in) ---
    mr = None
    if adp is not None and not (isinstance(adp, float) and adp != adp):
        mr = _clip((80 - float(adp)) / 80.0)   # ADP 1 -> +0.99, 80 -> 0, 160 -> -1
    elif adp_role in ("big", "rotational"): mr = 0.6
    elif adp_role in ("undrafted", "deep"): mr = -0.6
    sc["Market Read"] = _for_side(mr, side)

    # --- Regression: explicit signal, else mean-reversion pull from own-vs-role on small samples ---
    rg = None
    if regression is not None:
        rg = _clip(-regression)                # +regression (overperformed) lowers stat
    else:
        ovr = _own_vs_role(pr, mkt)
        if ovr is not None and mkt in PASS_MKTS:
            # a player far above his role model on a limited sample is overdue to fall back
            sample_pull = 1.0 if hist < 24 else (0.5 if hist < 40 else 0.25)
            rg = _clip(-ovr * 0.8 * sample_pull)
    sc["Regression"] = _for_side(rg, side)

    return {k: (round(float(v), 2) if v is not None else None) for k, v in sc.items()}


if __name__ == "__main__":
    p = {"side": "Under", "mkt": "receptions", "line": 4.5, "our_num": 2.9, "edge": 0.11}
    pr = {"own_rec_yds": 55, "role_rec_yds": 70, "proj_targets_pg": 4.2, "hist_games": 12}
    sc = scorecard(p, proj_row=pr, script={"team_spread": -7, "implied_team_pts": 27},
                   weather={"roof": "outdoors", "wind": 14, "temp": 38},
                   matchup=-0.08, adp=95, regression=None)
    for f, v in sc.items():
        tag = "n/a" if v is None else ("supports" if v > 0 else "against" if v < 0 else "neutral")
        print(f"  {f:18s} {str(v):>6}  {tag:9s} — {DESC[f]}")
