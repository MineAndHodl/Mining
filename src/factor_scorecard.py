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


TD_FACTORS = ["Red-Zone Role", "Vulture Risk", "Coach Tendency", "Finishing Rate",
              "Scoring Environment", "TD Regression", "Market Read"]
TD_DESC = {
    "Red-Zone Role": "his share of the scoring-zone work of his type (goal-line carries for a back, red-zone targets for a pass-catcher)",
    "Vulture Risk": "how much of his team's goal-line carrying goes to someone else (a backup who steals the plunge)",
    "Coach Tendency": "does the coach's goal-line play-calling (run vs pass) favor his way of scoring",
    "Finishing Rate": "how often he turns a scoring-zone chance into a TD, vs the position",
    "Scoring Environment": "his team's implied points — more team TDs, more shots at one",
    "TD Regression": "last season's TDs vs what his role should have produced — due to rise (unlucky) or fall (ran hot)",
    "Market Read": "the market's read on his role",
}

LEAD_BACK_GL_SHARE = 0.55     # measured: the lead back gets ~55% of goal-line carries
TYP_RZ_TGT_SHARE = 0.18       # a real red-zone target earner


def td_scorecard(pick, td=None, script=None, adp=None):
    """TD-SPECIFIC scale built from the DECOMPOSED xTD model (xtd_model.py): the levers that
    actually create touchdowns — goal-line/red-zone share, vulture leakage, coach run-vs-pass at
    the goal line, finishing, environment, and last-season regression. td: a row from the xTD
    table (gl_share, rz_tgt_share, vulture_share, coach_gl_pass_rate, conv_rush_gl, conv_tgt_rz,
    tds_last, xtd, games, position, pos priors). All leans are for a 'Yes' (to-score) pick."""
    td = td or {}
    sc = {}
    pos = td.get("position")
    is_back = pos in ("RB", "FB")
    gl = td.get("gl_share"); rzt = td.get("rz_tgt_share"); rz20 = td.get("rz20_rush_share")
    # --- Red-Zone Role: his cut of the scoring-zone work of HIS type. For a back that is his
    # FULL inside-the-20 carry role (volume creates TD chances across all three zones); the
    # goal-line plunge specifically is judged separately by Vulture Risk — so a high-volume back
    # with a shared goal line reads honestly as "real role, shared plunge", not as a contradiction.
    if is_back and (rz20 is not None or gl is not None):
        role = rz20 if rz20 is not None else gl
        sc["Red-Zone Role"] = _clip((role - 0.40) / 0.30)                  # .40 neutral, .70 -> +1
    elif rzt is not None:
        sc["Red-Zone Role"] = _clip((rzt - TYP_RZ_TGT_SHARE) / 0.18)      # .18 neutral, .36 -> +1
    else:
        sc["Red-Zone Role"] = None
    # --- Vulture Risk (backs only): goal-line carries leaking to someone else works AGAINST him
    v = td.get("vulture_share")
    sc["Vulture Risk"] = _clip(-(v - (1 - LEAD_BACK_GL_SHARE)) / 0.30) if (is_back and v is not None) else None
    # --- Coach Tendency: goal-line pass rate. Backs want a run-it-in coach; catchers the opposite.
    cpr = td.get("coach_gl_pass_rate")
    if cpr is not None:
        lean = (0.42 - cpr) / 0.15                                        # league GL pass ~.42
        sc["Coach Tendency"] = _clip(lean if is_back else -lean)
    else:
        sc["Coach Tendency"] = None
    # --- Finishing Rate: his conversion vs the position prior for that zone/type
    conv = td.get("conv_rush_gl") if is_back else td.get("conv_tgt_rz")
    prior = td.get("pos_conv_rush_gl") if is_back else td.get("pos_conv_tgt_rz")
    sc["Finishing Rate"] = _clip(((conv - prior) / prior) / 0.4) if (conv is not None and prior) else None
    # --- Scoring Environment
    ip = script.get("implied_team_pts") if script else None
    sc["Scoring Environment"] = _clip((ip - 22) / 8) if ip is not None else None
    # --- TD Regression: last season's TDs vs what the role says he should have scored
    tl = td.get("tds_last"); xt = td.get("xtd"); g = td.get("games")
    if tl is not None and xt is not None and g:
        exp = float(xt) * float(g)
        if exp > 0.5:
            sc["TD Regression"] = _clip(-((float(tl) - exp) / exp) / 0.5)  # over-scored -> due to fall
        else:
            sc["TD Regression"] = None
    else:
        sc["TD Regression"] = None
    # --- Market Read
    mr = None
    if adp is not None and not (isinstance(adp, float) and adp != adp):
        mr = _clip((80 - float(adp)) / 80.0)
    elif g and float(g) >= 12:
        mr = 0.3                                                          # established role
    sc["Market Read"] = mr
    return {k: (round(float(v), 2) if v is not None else None) for k, v in sc.items()}


def scorecard(pick, proj_row=None, script=None, weather=None, matchup=None,
              adp_role=None, adp=None, regression=None, contrib=None):
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

    # --- Recent Trend: PRIMARY source is the model's OWN recent-form contribution (form3+form1
    # from the multifactor fit, in stat units) so the slider reflects what actually moved our
    # number and is never blank when the reason says "recent form pushes up". Falls back to an
    # explicit delta, then own-vs-role for receiving. ---
    ct = contrib or {}
    if ct.get("form3") is not None or ct.get("form1") is not None:
        form_c = (ct.get("form3") or 0) + (ct.get("form1") or 0)
        sc["Recent Trend"] = _for_side(_clip(form_c / max(our, 1.0) / 0.15), side)
    elif pr.get("_form_delta") is not None:
        sc["Recent Trend"] = _for_side(_clip(pr["_form_delta"] / 12.0), side)
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

    # --- Injury Picture: his own status (negative) OR opportunity from a teammate's absence
    # (positive, via the measured role cascade). None only when no report exists yet. ---
    inj = pick.get("inj_lean")
    if isinstance(inj, float) and inj != inj:      # NaN from a JSON/parquet round-trip = no report
        inj = None
    if inj is None:
        h = pick.get("_health")
        if h:
            st = h.get("status")
            if st in ("Out", "Doubtful"): inj = -1.0
            elif st == "Questionable": inj = -0.4
            elif h.get("opportunity"): inj = 0.6
            else: inj = 0.0                    # checked, healthy, no vacated role
    # own-status penalties hurt EITHER side (he may not play / play limited): don't flip those
    if inj is not None and inj < 0:
        sc["Injury Picture"] = float(inj)
    else:
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
    else:
        # Not in the fantasy-draft pool (e.g. a veteran starter like a 42-yo QB, who no fantasy
        # market drafts but the NFL market absolutely recognizes as the starter). Fall back to a
        # role read from NFL track record so an established starter carries a real (modest) lean
        # instead of a blank: a long sample IS a recognized role; a thin one is not.
        hg = float(pr.get("hist_games", 0) or 0)
        if hg >= 32: mr = 0.4
        elif hg <= 8 and not pr.get("rookie"): mr = -0.3
    sc["Market Read"] = _for_side(mr, side)

    # --- Regression: explicit signal, else mean-reversion pull from own-vs-role on small samples ---
    rg = None
    if regression is not None:
        rg = _clip(-regression)                # +regression (overperformed) lowers stat
    elif ct.get("base_c") is not None:
        # the model's own mean-reversion term: +base_c means it pushed our number UP (he's been
        # BELOW his level, due to bounce back = raises stat); -base_c means pulled down (ran hot).
        rg = _clip(ct["base_c"] / max(our, 1.0) / 0.10)
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
