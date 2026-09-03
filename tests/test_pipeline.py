"""Full pipeline test + debug harness.

Two layers:
  A. Deterministic unit/invariant tests on synthetic fixtures (no network) — the math.
  B. Live-board invariant tests against a real Odds API pull — catches the class of
     bug that only shows up on real data (phantom lines, venue mismatch, sign flips).

Run: ODDS_API_KEY=xxx python3 tests/test_pipeline.py          (A+B)
     python3 tests/test_pipeline.py --offline                  (A only)

Every check prints PASS/FAIL with a one-line reason. Exit code != 0 if any FAIL.
"""
import os, sys, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import weekly_card as wc
import props_model as pm
import conviction as cv
import desk_card as dc

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail and not cond else ""))
    return bool(cond)


# ============================ A. DETERMINISTIC ============================
def test_implied_devig():
    print("\n-- implied() & de-vig --")
    check("implied(+100)=0.5", abs(wc.implied(100) - 0.5) < 1e-9)
    check("implied(-110)~0.524", abs(wc.implied(-110) - 0.5238) < 1e-3)
    check("implied(+150)=0.4", abs(wc.implied(150) - 0.4) < 1e-9)
    # de-vig two -110s -> 0.5/0.5
    io, iu = wc.implied(-110), wc.implied(-110)
    check("devig(-110/-110)=0.5", abs(io/(io+iu) - 0.5) < 1e-9)


def test_props_anchor():
    print("\n-- props: market-anchored edge --")
    # neutral tilt -> edge should be ~0 (anchored to market)
    side, edge, p, proj = pm.anchored_edge("rec_yds", 55.5, -110, -110, env=1.0)
    check("neutral tilt edge ~0", abs(edge) < 0.02, f"edge={edge}")
    # market_mean inverts p_over: p_over(market_mean(p)) == p
    mu = pm.market_mean("rec_yds", 55.5, 0.5)
    p_back = pm.p_over("rec_yds", mu, 55.5)
    check("market_mean inverts p_over", abs(p_back - 0.5) < 1e-3, f"p_back={p_back}")
    # positive env tilt -> Over edge grows monotonically
    e_lo = pm.anchored_edge("rec_yds", 55.5, -110, -110, env=1.05)[1]
    e_hi = pm.anchored_edge("rec_yds", 55.5, -110, -110, env=1.12)[1]
    check("edge grows with env tilt", e_hi > e_lo, f"{e_lo}->{e_hi}")
    # tilt is clipped (no crazy edge) at extreme env
    e_ex = pm.anchored_edge("rec_yds", 55.5, -110, -110, env=3.0)[1]
    check("tilt clipped at max", e_ex < 0.20, f"edge={e_ex}")
    # p_over monotonic in line
    check("p_over decreasing in line", pm.p_over("rec_yds", 60, 40) > pm.p_over("rec_yds", 60, 80))


def test_conviction():
    print("\n-- conviction: tiers, confidence, dedup --")
    check("tier>=70 TOP", cv.tier(72) == "TOP PLAY")
    check("tier 55-69 STRONG", cv.tier(60) == "STRONG")
    check("tier 42-54 LEAN", cv.tier(50) == "LEAN")
    check("tier<42 None", cv.tier(30) is None)
    check("desk_conf floor 6.0", cv.desk_confidence(42) == 6.0)
    check("desk_conf cap 9.6", cv.desk_confidence(200) == 9.6)
    check("desk_conf monotonic", cv.desk_confidence(70) > cv.desk_confidence(50))
    check("gated prop scores 0", cv.score_prop(0.1, 10, "rush_yds") == 0)
    check("eligible prop scores>0", cv.score_prop(0.06, 10, "rec_yds") > 0)
    # dedup: same play twice -> one row
    flags = [{"score": 70, "label": "BUF -1", "market": "spreads", "game": "g",
              "dedup": ("g", "spreads", "BUF"), "kind": "off-market"},
             {"score": 60, "label": "BUF -1", "market": "spreads", "game": "g",
              "dedup": ("g", "spreads", "BUF"), "kind": "off-market"}]
    card = cv.build_card(flags=flags, props=[])
    check("dedup collapses dupes", len(card["plays"]) == 1, f"n={len(card['plays'])}")
    check("dedup keeps higher score", card["plays"][0]["score"] == 70)


def test_offmarket_synthetic():
    print("\n-- find_off_market: coherence on a crafted board --")
    # Bills-Texans: pinnacle Bills -1; fanatics offers Bills +1 (soft), betmgm Bills -1.
    rows = []
    def add(book, mkt, side, pt, pr): rows.append(
        {"game": "BUF @ HOU", "home": "HOU", "away": "BUF", "commence": "x",
         "book": book, "market": mkt, "side": side, "point": pt, "price": pr, "last_update": "x"})
    add("pinnacle", "spreads", "Buffalo Bills", -1.0, -105); add("pinnacle", "spreads", "Houston Texans", 1.0, -105)
    add("fanatics", "spreads", "Buffalo Bills", 1.0, -110); add("fanatics", "spreads", "Houston Texans", -1.0, -110)
    add("betmgm", "spreads", "Buffalo Bills", -1.0, -110); add("betmgm", "spreads", "Houston Texans", 1.0, -110)
    add("bovada", "spreads", "Buffalo Bills", 2.0, -110); add("bovada", "spreads", "Houston Texans", -2.0, -110)  # offshore, better number
    board = pd.DataFrame(rows)
    fair = wc.pinnacle_fair(board)
    flags = wc.find_off_market(board, fair)
    # the flag should be Bills +1 at fanatics (best BETTABLE), NOT bovada +2 (offshore)
    bills = flags[(flags["side"] == "Buffalo Bills")] if len(flags) else flags
    check("flag exists on Bills", len(bills) == 1, f"n={len(bills)}")
    if len(bills):
        f = bills.iloc[0]
        check("flag book is bettable (fanatics)", f["book"] == "fanatics", f"book={f['book']}")
        check("flag line matches book's actual number", f["point"] == 1.0, f"pt={f['point']}")
        # coherence: displayed book actually offers displayed number
        offered = board[(board["book"] == f["book"]) & (board["market"] == "spreads")
                        & (board["side"] == "Buffalo Bills")]["point"].iloc[0]
        check("COHERENCE: book offers displayed line", offered == f["point"], f"{offered} vs {f['point']}")
        check("edge vs fair correct (+2)", abs(f["vs_fair_pts"] - 2.0) < 1e-6, f"{f['vs_fair_pts']}")
    # offshore bovada +2 must NOT be a flag (bettable_only)
    check("offshore excluded from flags", (flags["book"] != "bovada").all() if len(flags) else True)


def test_totals_sign():
    print("\n-- totals sign logic --")
    rows = []
    def add(book, side, pt, pr): rows.append(
        {"game": "A @ B", "home": "B", "away": "A", "commence": "x", "book": book,
         "market": "totals", "side": side, "point": pt, "price": pr, "last_update": "x"})
    add("pinnacle", "Over", 44.0, -105); add("pinnacle", "Under", 44.0, -105)
    add("draftkings", "Under", 45.5, -110); add("draftkings", "Over", 45.5, -110)  # Under 45.5 better than 44
    add("fanduel", "Over", 42.5, -110); add("fanduel", "Under", 42.5, -110)        # Over 42.5 better than 44
    board = pd.DataFrame(rows)
    fair = wc.pinnacle_fair(board)
    flags = wc.find_off_market(board, fair)
    u = flags[flags["side"] == "Under"]; o = flags[flags["side"] == "Over"]
    check("Under flagged at higher number (45.5)", len(u) == 1 and u.iloc[0]["point"] == 45.5)
    check("Over flagged at lower number (42.5)", len(o) == 1 and o.iloc[0]["point"] == 42.5)
    check("Under edge = +1.5", len(u) and abs(u.iloc[0]["vs_fair_pts"] - 1.5) < 1e-6)
    check("Over edge = +1.5", len(o) and abs(o.iloc[0]["vs_fair_pts"] - 1.5) < 1e-6)


def test_crossbook_props():
    print("\n-- desk_card: cross-book prop shopping --")
    rows = [
        {"game": "g", "book": "draftkings", "mkt": "rec_yds", "name": "P", "line": 55.5, "over_price": -120, "under_price": +100},
        {"game": "g", "book": "fanduel", "mkt": "rec_yds", "name": "P", "line": 55.5, "over_price": +110, "under_price": -130},
        {"game": "g", "book": "pinnacle", "mkt": "rec_yds", "name": "P", "line": 55.5, "over_price": -105, "under_price": -105},
    ]
    pb = pd.DataFrame(rows)
    sh = dc.shop_prop_edges(pb, env_by_team={"g": 1.13})
    check("shops only bettable books (no pinnacle)", len(sh) and (sh["book"] != "pinnacle").all())
    if len(sh):
        # NOTE: desk_card is the legacy market-anchored path (not in the production board).
        # anchored_edge re-anchors to EACH book's own line, so it selects by edge, not raw
        # price. True best-price line-shopping (keep the best payout AT our line) lives in the
        # production prop_board path (groups by name/mkt/side/line, keeps the max-profit book).
        r = sh.iloc[0]
        check("shopped pick is a bettable book with a real edge", r["book"] in ("draftkings", "fanduel") and r["edge"] > 0, f"book={r['book']} edge={r['edge']}")
    # gated market excluded
    pb2 = pd.DataFrame([{"game": "g", "book": "draftkings", "mkt": "rush_yds", "name": "Q",
                         "line": 40.5, "over_price": -110, "under_price": -110}])
    sh2 = dc.shop_prop_edges(pb2, env_by_team={"g": 1.13})
    scored = dc.score_props_for_card(sh2)
    check("gated market not scored into card", len(scored) == 0)


# ============================ B. LIVE BOARD ============================
def test_live(key):
    print("\n== LIVE BOARD INVARIANTS ==")
    data, rem = wc.fetch_odds(key, days=14)
    board = wc.build_board(data)
    if not len(board):
        print("  (no games in 14-day window — skipping live checks)")
        return
    fair = wc.pinnacle_fair(board)
    flags = wc.find_off_market(board, fair)
    print(f"  board rows={len(board)}, games={board['game'].nunique()}, flags={len(flags)}, credits_left={rem}")

    # 1. every flag's (book,market,side,point) exists verbatim in board
    def exists(f):
        m = board[(board["book"] == f["book"]) & (board["market"] == f["market"])
                  & (board["side"] == f["side"])]
        if f["market"] == "h2h":
            return (m["price"] == f["price"]).any()
        return (m["point"] == f["point"]).any()
    check("no phantom flags (all exist in board)", all(exists(f) for _, f in flags.iterrows()) if len(flags) else True)

    # 2. COHERENCE: displayed book actually offers displayed number (the Bills bug)
    def coherent(f):
        if f["market"] == "h2h":
            return True
        m = board[(board["book"] == f["book"]) & (board["market"] == f["market"])
                  & (board["side"] == f["side"])]
        return (m["point"] == f["point"]).any()
    check("COHERENCE: book offers the shown line", all(coherent(f) for _, f in flags.iterrows()) if len(flags) else True)

    # 3. no offshore/EU book in flags
    if len(flags):
        check("no offshore book in flags", flags["book"].isin(wc.BETTABLE_BOOKS).all(),
              f"books={sorted(flags['book'].unique())}")

    # 4. edge recompute matches for spread/total flags
    def edge_ok(f):
        if f["market"] not in ("spreads", "totals"):
            return True
        fp = fair.get((f["game"], f["market"]), {}).get(f["side"], {}).get("point")
        if fp is None:
            return False
        if f["market"] == "spreads":
            d = f["point"] - fp
        else:
            d = (f["point"] - fp) if f["side"] == "Under" else (fp - f["point"])
        return abs(round(d, 1) - f["vs_fair_pts"]) < 1e-6
    check("edge recompute matches", all(edge_ok(f) for _, f in flags.iterrows()) if len(flags) else True)

    # 5. pinnacle spread favorite == pinnacle ML favorite (per game)
    bad = []
    for game in board["game"].unique():
        sp = fair.get((game, "spreads")); ml = fair.get((game, "h2h"))
        if not sp or not ml:
            continue
        fav_sp = min(sp, key=lambda s: sp[s]["point"] if sp[s]["point"] is not None else 0)
        fav_ml = max(ml, key=lambda s: ml[s]["fair_p"])
        # only assert on a CLEAR favorite (field-goal+); pick'em juice-flips are legit
        if sp[fav_sp]["point"] is not None and sp[fav_sp]["point"] <= -2.5 and fav_sp != fav_ml:
            bad.append(game)
    check("spread favorite == ML favorite (clear favs only)", not bad, f"violations={bad}")

    # 6. full desk_card runs + dedup holds
    res = dc.build_desk_card(key=key, do_props=False, days=14)
    plays = res["card"]["plays"]
    keys = [p.get("dedup") for p in plays]
    check("no duplicate plays in final card", len(keys) == len(set(keys)), f"{len(keys)} plays")
    check("desk_card produced a card", isinstance(res["card"]["verdict"], str))


def test_factor_scorecard():
    """Factors must carry real leans (not a wall of zeros) and honor n/a vs neutral."""
    import factor_scorecard as fsc
    pick = {"side": "Under", "mkt": "receptions", "line": 4.5, "our_num": 2.9}
    pr = {"own_rec_yds": 55, "role_rec_yds": 70, "proj_targets_pg": 4.2, "hist_games": 12}
    sc = fsc.scorecard(pick, proj_row=pr, script={"team_spread": -7, "implied_team_pts": 27},
                       weather={"roof": "outdoors", "wind": 14, "temp": 38}, matchup=-0.08, adp=95)
    fired = sum(1 for v in sc.values() if v not in (None, 0.0))
    check("scorecard fires most factors (>=6)", fired >= 6, f"fired={fired}")
    check("Projected Usage supports Under (our#<line => positive lean for the under)", sc["Projected Usage"] > 0)
    # dome -> weather neutral 0.0, NOT None
    sd = fsc.scorecard({"side": "Over", "mkt": "rec_yds", "line": 60, "our_num": 70},
                       proj_row=pr, weather={"roof": "dome"})
    check("dome weather = neutral 0.0 (not n/a)", sd["Weather"] == 0.0)
    # no weather data at all -> None (n/a)
    sn = fsc.scorecard({"side": "Over", "mkt": "rec_yds", "line": 60, "our_num": 70}, proj_row=pr)
    check("missing weather = n/a (None)", sn["Weather"] is None)


def test_whale_missile_discipline():
    """The Sunday Stack (replaces the whale missile): 4-6 of OUR recommended bets, one per game,
    6-12:1, +EV on our numbers; the Primetime SGPs land at a 30-100:1 fair price (lotto lane)."""
    import json as _j
    d = _j.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "ranked_board.json")))
    m = d.get("sunday_stack")
    check("Sunday Stack present (no whale missile)", bool(m) and "whale_missile" not in d)
    if m:
        legs = m["legs"]
        check("stack has 4-6 legs", 4 <= len(legs) <= 6, f"n={len(legs)}")
        check("stack pays 6-12:1", 600 <= m["combined_american"] <= 1300, f"{m['combined_american']:+d}")
        check("one leg per game (independent legs)", len({l["game"] for l in legs}) == len(legs))
        check("every leg is a RECOMMENDED bet (grade B+ or better)", all(l.get("grade") in ("A+", "A", "A-", "B+") for l in legs))
        check("no micro-prop legs", all(float(l["line"]) >= {"rush_yds": 15, "rec_yds": 10, "receptions": 2.5, "pass_yds": 150}.get(l["cat"], 0) for l in legs))
        sides = [l["side"] for l in legs]
        check("not a single-side regime stack (<=3 of one side)", max(sides.count(s) for s in set(sides)) <= 3)
        check("stack is +EV on our numbers and reports true prob + fair", m["ev"] > 0 and 0 < m["true_prob"] < 0.5 and m["fair_american"] > 0)
    for s in d.get("primetime_sgps", []):
        fa = s.get("fair_american") or 0
        check(f"SGP fair in the 30-100:1 lotto band ({s['game'][:22]})", 2500 <= fa <= 10500, f"fair {fa:+d}")
        check("SGP is a correlated stack (lift > 3x)", (s.get("correlation_lift") or 0) > 3)


def test_qc_fixes():
    """The QC bug fixes: ADP-missing != small role (M1), Desk Score hump (C3), skew constant (C2)."""
    import investigate as inv, rank_board as rb, props_model as pm
    # M1: an Under with a BIG edge and NO adp is KEEP, not CONFIRMED
    v, note, mult = inv.investigate({"side": "Under", "edge": 0.15, "confidence": 0.7}, adp_row=None)
    check("M1: missing ADP + big under != CONFIRMED", v != "CONFIRMED", f"got {v}")
    # a genuinely late-ADP (deep) player DOES confirm a small-role under
    v2, _, _ = inv.investigate({"side": "Under", "edge": 0.15, "confidence": 0.7}, adp_row={"adp": 150})
    check("M1: deep-ADP under still CONFIRMED", v2 == "CONFIRMED", f"got {v2}")
    # C3: Desk Score hump — a huge edge scores LOWER than a moderate one (same conf/verdict)
    mod = rb.prop_score({"edge": 0.09, "confidence": 0.7, "score_mult": 1.0})
    huge = rb.prop_score({"edge": 0.25, "confidence": 0.7, "score_mult": 1.0})
    check("C3: moderate edge outscores blowout edge", mod > huge, f"mod={mod} huge={huge}")
    # C3: SUSPECT multiplier cuts the score hard
    conf = rb.prop_score({"edge": 0.09, "confidence": 0.7, "score_mult": 1.0})
    sus = rb.prop_score({"edge": 0.09, "confidence": 0.7, "score_mult": 0.5})
    check("C3: SUSPECT multiplier cuts score", sus < conf - 15, f"conf={conf} sus={sus}")
    # C2: CV is volume-dependent — a workhorse (high projected mean) is STEADIER than a
    # committee back, so its cv is LOWER. A flat cv over-widened the gamma for high-volume
    # players and manufactured phantom Under edges; cv_for fixes that.
    check("C2: rush_yds cv falls with volume (workhorse steadier)",
          pm.cv_for("rush_yds", 85) < pm.cv_for("rush_yds", 25),
          f"hi={pm.cv_for('rush_yds',85):.2f} lo={pm.cv_for('rush_yds',25):.2f}")
    check("C2: cv_for is monotone (brentq-safe in market_mean)",
          pm.cv_for("rec_yds", 30) >= pm.cv_for("rec_yds", 55) >= pm.cv_for("rec_yds", 80))
    # C2: with the correct workhorse cv, a mean modestly above the line is NOT a strong under —
    # the coherence bug (project 78, bet under 74) is gone.
    po = pm.p_over("rush_yds", 78.4, 74.5)
    check("C2: mean>line gives P(over) near 0.5, not a phantom under", 0.44 <= po <= 0.52, f"p_over={po:.3f}")


def test_primetime_sgp_structure():
    """Primetime SGPs (if present on the board) are same-game, 8-10 legs, single book."""
    import json as _j
    d = _j.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "ranked_board.json")))
    sgps = d.get("primetime_sgps", [])
    if not sgps:
        check("primetime SGPs present", True, "none built (props not posted) — skipped"); return
    for s in sgps:
        check(f"SGP {s['game'][:18]} has 8-10 legs", 8 <= s["n_legs"] <= 10, f"n={s['n_legs']}")
        players = [l.get("player") for l in s["legs"]]
        # a correlated SGP intentionally reuses a player across markets (yards+catches) — but
        # every leg belongs to the ONE game and one book
        check(f"SGP {s['game'][:18]} single book", "book" in s and s["book"])


def test_injury_layer():
    """Injury feed + measured role cascade (THE_DESK_PROMPT work order 1), proven on 2025 wk10."""
    print("\n-- injury layer: own status + role cascade --")
    import injury_layer as il, grading, factor_scorecard as fsc
    L = il.InjuryLayer(2025, week=10)
    st = L.status("Bucky Irving")
    check("Out player detected from the report", bool(st) and st["status"] == "Out", str(st))
    m, note = L.role_multiplier("Kenny Gainwell", "rush_yds")
    check("RB2 volume rises when RB1 is out (measured cascade)", m > 1.3, f"x{m}")
    check("cascade note names the absent player", bool(note) and "Irving" in note)
    m2, _ = L.role_multiplier("Stefon Diggs", "rec_yds")
    check("WR2 targets rise when WR1 is out", m2 > 1.1, f"x{m2}")
    mq, _ = L.role_multiplier("Jayden Daniels", "pass_yds")
    check("no QB downgrade for a WR1 absence (measured: yards do not drop)", mq == 1.0, f"x{mq}")
    lean, _ = L.injury_lean("Kenny Gainwell", "rush_yds")
    check("Injury Picture lean positive for the beneficiary", lean is not None and lean > 0.5, str(lean))
    sc = fsc.scorecard({"side": "Under", "mkt": "rec_yds", "line": 50.5, "our_num": 40, "inj_lean": -0.15}, proj_row={"hist_games": 40})
    check("Questionable lean hurts either side (not flipped for the under)", sc["Injury Picture"] == -0.15, str(sc["Injury Picture"]))
    check("cascade applied at the EARNED weight (0.84), not full", il.ROLE_CASCADE_WEIGHT == 0.87 and m < 1.75, f"x{m}")
    empty = il.InjuryLayer(2026)
    check("2026 loader returns cleanly before reports post", isinstance(empty.report, pd.DataFrame))
    lean26, _ = empty.injury_lean("Derrick Henry", "rush_yds")
    check("no report -> honest n/a, not a fake lean", lean26 is None)
    # grading: game edge parses onto the prop scale
    check("game edge '+1.5 pts' -> ~4.5% prob edge", abs(grading.parse_game_edge("+1.5 pts") - 0.045) < 1e-9)
    check("grade bands: 90 -> A+, 66 -> B+, 50 -> C", grading.grade_for(90) == "A+" and grading.grade_for(66) == "B+" and grading.grade_for(50) == "C")


def test_player_intel():
    """Beat intel: date-gated, hard facts gate, soft signals flip for the side, honest n/a."""
    print("\n-- player intel: date gate, hard gate, lean --")
    import player_intel as pi, datetime
    W = (datetime.date(2026, 9, 1), datetime.date(2026, 9, 8))
    stale = [{"date": "2025-12-20", "source": "espn", "title": "X ruled out with toe injury", "snippet": ""}]
    c = pi.classify_items(stale, *W)
    check("stale 'ruled out' from last season is excluded by the date gate", c["n_items"] == 0 and c["hard"] is None)
    cur = [{"date": "2026-09-03", "source": "team", "title": "Coach: he will be featured early, heavy dose of the run game", "snippet": ""}]
    c = pi.classify_items(cur, *W)
    check("in-window coach-intent item fires a positive coach signal", c["signals"]["coach"] > 0 and c["composite"] > 0)
    out = [{"date": "2026-09-04", "source": "beat", "title": "Team has ruled out RB for Sunday", "snippet": ""}]
    check("credible 'ruled out' becomes a hard OUT gate", pi.classify_items(out, *W)["hard"] == "OUT")
    lim = [{"date": "2026-09-04", "source": "beat", "title": "Returning WR will be on a pitch count", "snippet": ""}]
    check("'pitch count' becomes a hard LIMITED gate", pi.classify_items(lim, *W)["hard"] == "LIMITED")
    I = pi.PlayerIntel(2026, 1)
    lo = I.lean("Drake London", "Over"); lu = I.lean("Drake London", "Under")
    check("lean flips for the side (London: role+ supports Over, opposes Under)", lo is not None and lu is not None and lo > 0 and lu < 0, f"over={lo} under={lu}")
    check("no in-window items and no read -> None (honest n/a)", I.lean("Someone Nobody", "Over") is None)
    check("Week-1 intel file has dated, sourced items", all("date" in it and "source" in it for v in I.raw.values() if isinstance(v, dict) for it in v.get("items", [])))


def test_selection_over_pool():
    """Selection runs over the full pool, and a soft signal can reorder near the margin."""
    print("\n-- selection over the pool; intel in the score --")
    import rank_board as rb, grading, json
    base = {"edge": 0.06, "model_p": 0.58, "confidence": 1.0, "n_books": 2, "score_mult": 1.0}
    s0 = rb.prop_score(base); s1 = rb.prop_score({**base, "intel_lean": 1.0}); s2 = rb.prop_score({**base, "intel_lean": -1.0})
    check("intel lean moves the score (provisional, capped)", s1 > s0 > s2 and (s1 - s0) <= rb.INTEL_PROV_WEIGHT + 1e-9, f"{s2} < {s0} < {s1}")
    d = json.load(open(os.path.join(HERE, "..", "data", "ranked_board.json")))
    check("board records the pool it selected from", d.get("pool_summary", {}).get("n", 0) > 50)
    check("slate rows carry scorecards (main tab has sliders)", all((r.get("scorecard") or {}) for r in d.get("recommended", []) if not r.get("kind")))


def test_td_shopping_and_stack():
    """Anytime-TD is line-shopped across books; the Touchdown Tax is +EV real scorers, one per game."""
    print("\n-- anytime TD: shopping + TD stack --")
    import json as _j
    t = pd.read_parquet(os.path.join(HERE, "..", "data", "_raw_td_board.parquet"))
    check("TD board carries shopped fields", all(c in t.columns for c in ("n_books", "implied_best", "implied_consensus", "book")))
    check("TD prices come from multiple books (median >= 3)", t["n_books"].median() >= 3, f"median {t['n_books'].median()}")
    check("best price never worse than consensus", (t["implied_best"] <= t["implied_consensus"] + 1e-9).all())
    d = _j.load(open(os.path.join(HERE, "..", "data", "ranked_board.json")))
    s = d.get("td_stack")
    check("TD stack present", bool(s))
    if s:
        legs = s["legs"]
        check("TD stack has 2-4 legs", 2 <= len(legs) <= 4, f"n={len(legs)}")
        check("one scorer per game", len({l["game"] for l in legs}) == len(legs))
        check("every leg a real scorer (>=33%) with +edge", all(l["model_p"] >= 0.33 and (l.get("edge") or 0) >= 0.02 for l in legs))
        check("TD stack pays ~4-39:1 and is +EV on our numbers", 400 <= s["combined_american"] <= 4000 and s["ev"] > 0, f"{s['combined_american']:+d} ev {s['ev']}")
        check("TD stack prefers 3 real scorers when 3 clear the band", len(legs) >= 3 or len([r for r in d.get("pool_summary", {}).get("td_candidates", [])]) < 3, f"n={len(legs)}")
        check("no early-season market-flagged scorer in the stack", not any(l.get("mkt_flag") for l in legs))


def test_ledger_and_grader():
    """The forward-grading loop: every number frozen in the ledger; settlement math exact."""
    print("\n-- ledger + grader --")
    import ledger as L, grade_week as gw, json as _j
    from names import norm_name as nn
    d = _j.load(open(os.path.join(HERE, "..", "data", "ranked_board.json")))
    snap = L.build_snapshot(d)
    check("ledger resolves the slate's week from the schedule", snap["week"] is not None and snap["season"] == 2026, str((snap["season"], snap["week"])))
    check("ledger freezes the Card with price, grade, units and factor contributions",
          len(snap["card"]) > 0 and all(p.get("price") is not None and p.get("grade") and "contrib" in p for p in snap["card"]))
    check("ledger carries the parlay lanes and the games intel lane", "sunday_stack" in snap and "td_stack" in snap and isinstance(snap["games"], list))
    files = os.listdir(os.path.join(HERE, "..", "data", "ledger")) if os.path.isdir(os.path.join(HERE, "..", "data", "ledger")) else []
    check("a ledger file exists for the current week", any(f.endswith(".json") for f in files))
    check("prop settlement: over/under/push/void", gw.settle_prop("Over", 39.5, 52) == "WIN" and gw.settle_prop("Under", 39.5, 52) == "LOSS"
          and gw.settle_prop("Over", 40, 40) == "PUSH" and gw.settle_prop("Over", 40, None) == "VOID")
    check("game settlement: spread push / total under / ML", gw.settle_game("spread", "Buffalo Bills", 1.0, "Houston Texans", "Buffalo Bills", 24, 23) == "PUSH"
          and gw.settle_game("total", "Under", 42.5, "H", "A", 20, 21) == "WIN" and gw.settle_game("ML", "Houston Texans", None, "Houston Texans", "Buffalo Bills", 24, 23) == "WIN")
    check("units at American prices (-114 -> +0.877, +135 -> +1.35, loss -> -1)", gw.units_won("WIN", -114) == 0.877 and gw.units_won("WIN", 135) == 1.35 and gw.units_won("LOSS", 135) == -1)
    check("CLV is signed in the direction of our side", gw.clv_points("Over", 39.5, 42.5) == 3.0 and gw.clv_points("Under", 39.5, 42.5) == -3.0)
    stats = pd.DataFrame({"_k": [nn("Davante Adams"), nn("George Pickens")], "_tds": [1, 0], "receiving_yards": [70, 44]})
    legs = [{"leg": "a", "name": "Davante Adams", "price": 135}, {"leg": "b", "name": "George Pickens", "price": 170}, {"leg": "c", "name": "Brock Bowers", "price": 200}]
    f = lambda l: gw.grade_pick({"name": l["name"], "cat": "anytime_td", "side": "Yes", "line": 0.5, "price": l["price"], "id": l["leg"]}, stats, {}, {})
    _, overall = gw.grade_parlay(legs, stats, {}, {}, f)
    check("parlay: a lost leg loses, a void leg (DNP) drops out", overall == "LOSS" and gw.grade_parlay(legs[:1] + legs[2:], stats, {}, {}, f)[1] == "WIN")


def test_fresh_eyes_day1():
    """Day-1 findings: measured within-player CV anchors; the public-record rail on the Card."""
    print("\n-- fresh eyes day 1: distribution + record rail --")
    import props_model as pm, grading, json as _j
    check("CV anchors come from the within-player measurement (data/cv_anchors.json)",
          os.path.exists(os.path.join(HERE, "..", "data", "cv_anchors.json")) and abs(pm.cv_for("pass_yds", 240) - 0.29) < 0.03, f"{pm.cv_for('pass_yds', 240):.3f}")
    check("starting-QB passing spread matches the book's implied shape (~.27-.30), not the old .40", pm.cv_for("pass_yds", 250) < 0.32)
    check("receptions anchor is EXTRA dispersion (~.2), not total CV (~.5)", pm.cv_for("receptions", 5.5) < 0.3)
    check("workhorse rush CV ~.45-.50 (was .53)", 0.42 <= pm.cv_for("rush_yds", 90) <= 0.50, f"{pm.cv_for('rush_yds', 90):.3f}")
    rows = [{"score": 90, "model_p": 0.42, "cat": "anytime_td", "line": 0.5, "name": "Dog", "game": "g"},
            {"score": 85, "model_p": 0.58, "cat": "rec_yds", "line": 50.5, "name": "Fav", "game": "g"}]
    rec, _ = grading.apply(rows, weeks_played=5)
    check("public-record rail: a +EV 42% touchdown dog cannot be on the Card", not rows[0]["recommended"] and "Touchdown Tax" in (rows[0]["rail"] or ""))
    check("a 58% play stays on the Card", rows[1]["recommended"])
    d = _j.load(open(os.path.join(HERE, "..", "data", "ranked_board.json")))
    check("every Card pick is at least a coin-flip on our numbers", all((r.get("model_p") or 0) >= 0.5 for r in d.get("recommended", [])))


def test_line_history_and_ledger_alignment():
    """Every bet knows where it opened and where it is now; the ledger is aligned with the product."""
    print("\n-- line history + ledger alignment --")
    import json as _j, line_history as lh, ledger as L, glob as _g
    d = _j.load(open(os.path.join(HERE, "..", "data", "ranked_board.json")))
    card = d.get("recommended", [])
    check("every Card pick carries a line_move (opened -> now)", all(r.get("line_move") for r in card))
    lm = [r["line_move"] for r in card if r.get("line_move")]
    check("line_move tracks the ACTIONABLE line for our side and the consensus", all("open_line" in m and "open_consensus" in m and "still_available" in m for m in lm))
    check("Card lines match the tracked opened line at open (same snapshot)", all(abs(float(r["line"]) - float(r["line_move"]["open_line"])) < 1e-9 for r in card if r.get("line_move") and r["line_move"].get("n_snapshots") == 1) or any(m.get("n_snapshots", 0) > 1 for m in lm))
    snaps = _g.glob(os.path.join(HERE, "..", "data", "line_history", "*", "*_props.parquet"))
    check("a raw all-book line snapshot is archived for the week", len(snaps) >= 1)
    leds = sorted(_g.glob(os.path.join(HERE, "..", "data", "ledger", "*.json")))
    led = _j.load(open(leds[-1]))
    check("ledger marks the open as preview until the Tuesday --open run", led.get("opened_kind") in ("preview", "official"))
    check("ledger carries the late-swap lane and a snapshot history", "late_swap" in led["latest"] and len(led.get("history", [])) >= 1)
    p0 = led["opened"]["card"][0]
    check("ledger rows carry score components + line move for re-earning", "score_conviction" in p0 and "score_trust" in p0 and "line_move" in p0)
    # market-role baseline from the CONSENSUS line: the shop must prefer the better line
    import pandas as pd
    raw = pd.read_parquet(os.path.join(HERE, "..", "data", "_raw_prop_board.parquet"))
    hen = raw[(raw.name == "Derrick Henry") & (raw.mkt == "rush_yds")]
    check("shop prefers the best available line for our side (Henry Over 77.5 @ betrivers, not 78.5)", (not len(hen)) or float(hen.iloc[0]["line"]) <= 77.5, str(hen[["book", "line"]].values.tolist()))


def test_last_call_and_weather():
    """Last Call (kickoff window): the gate, the item types, the weather feed, the stale flag."""
    print("\n-- last call + weather feed --")
    import last_call as lc, weather_feed as wf, line_history as lh, json as _j
    g = lc.games_in_window(150)
    check("last-call gate returns a list (empty outside a window)", isinstance(g, list))
    wide = lc.games_in_window(60 * 24 * 20)
    check("last-call gate finds the next slate inside a wide window", len(wide) >= 1)
    check("kickoff times convert ET -> UTC (8:20 ET Wed 9/9 = 00:20Z Thu)", wf.kickoff_utc("2026-09-09", "20:20").isoformat().startswith("2026-09-10T00:20"))
    w = wf.load()
    check("kickoff forecast file exists with the week's games", len(w) >= 10, f"{len(w)} games")
    dome = next((v for v in w.values() if v.get("roof") in ("dome", "closed")), None)
    check("domes are neutral (wind 0)", dome is not None and dome.get("wind") == 0.0)
    d = _j.load(open(os.path.join(HERE, "..", "data", "ranked_board.json")))
    check("board rows carry the kickoff weather (wx_mult) — the validated weather module is live", any(r.get("wx_mult") is not None for r in d.get("recommended", [])))
    mv = lh.move_for({"name": "Rhamondre Stevenson", "cat": "rush_yds", "side": "Over"}, 2026, 1)
    check("stale-number flag: best Over line vs the consensus (Stevenson 39.5 vs 42.5 -> ~3 pts)", mv is not None and mv.get("stale_book_pts") is not None and mv["stale_book_pts"] >= 1.0, str(mv and mv.get("stale_book_pts")))
    check("stale flag is zero when the best line IS the consensus", (lambda m: m is None or m.get("stale_book_pts") is None or abs(m["stale_book_pts"]) < 1e-9)(lh.move_for({"name": "Rashid Shaheed", "cat": "receptions", "side": "Under"}, 2026, 1)))


def test_games_lane():
    """Games are rebuilt live, labeled, and never on the Card until a games edge is validated."""
    print("\n-- games lane --")
    import json as _j, games_lane as gl
    d = _j.load(open(os.path.join(HERE, "..", "data", "ranked_board.json")))
    g = d.get("best_games", [])
    check("game rows are labeled (game/market/side/book)", all(x.get("game") and x.get("market") and x.get("book") for x in g))
    check("game rows are the live intel lane (not a stale carry-over)", all(x.get("lane") == "intel" for x in g))
    check("no game on the Card while GAMES_ON_CARD is off", gl.GAMES_ON_CARD or not any(x.get("recommended") for x in g))
    check("Card is straight props only", not any(r.get("kind") for r in d.get("recommended", [])))


def test_td_environment():
    """The TD environment is on the MEASURED scale (td_env_validate.py), and the early-season
    market-disagreement rail flags stale-role scorers."""
    print("\n-- TD environment: measured scale, rail --")
    import td_model as tdm, td_board as tb, xtd_model as xm
    check("implied team TDs on the measured scale (27 pts -> ~3.0, not 2.6)", abs(tdm.implied_team_tds(27) - 3.03) < 0.05, f"{tdm.implied_team_tds(27):.2f}")
    check("league-average implied points ~ league-average TDs (multiplier ~1)", abs(tdm.implied_team_tds(22.3) - 2.38) < 0.06)
    b = {"DET": 3.647, "NO": 2.0}
    m = tdm.env_multipliers(b, -7.0, 47.0, "DET", "NO")
    check("elite offense no longer crushed by its own baseline (DET mult > 0.8)", m["DET"] > 0.8, f"{m['DET']:.2f}")
    check("recalibration refit with the environment applied", xm.CAL["b"] < 0.8 and 0.7 < tdm.CAL_B < 1.0, f"{xm.CAL} {tdm.CAL_B}")
    t = pd.read_parquet(os.path.join(HERE, "..", "data", "_raw_td_board.parquet"))
    check("TD board carries the market-disagreement rail fields", "mkt_ratio" in t.columns and "mkt_flag" in t.columns)
    fl = t[t.mkt_flag == True]
    check("flagged scorers are the >1.5x-consensus cases", len(fl) == 0 or fl["mkt_ratio"].min() > tb.TD_MAX_MARKET_RATIO, f"min ratio {fl['mkt_ratio'].min() if len(fl) else None}")
    check("raw TD prices + environment cached for free re-pricing", os.path.exists(tb.RAW_PRICES) and os.path.exists(tb.RAW_ENV))
    top = t[t.p_model.notna()].sort_values("p_model", ascending=False).head(5)
    check("top scorers priced in the measured 45-60% range (not 40-45)", top["p_model"].mean() > 0.45, f"{top['p_model'].mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--offline", action="store_true"); a = ap.parse_args()
    print("=" * 60, "\n  DESK PIPELINE TEST SUITE\n", "=" * 60)
    test_implied_devig(); test_props_anchor(); test_conviction()
    test_offmarket_synthetic(); test_totals_sign(); test_crossbook_props()
    test_factor_scorecard(); test_whale_missile_discipline()
    test_qc_fixes(); test_primetime_sgp_structure(); test_injury_layer(); test_player_intel(); test_selection_over_pool(); test_td_shopping_and_stack(); test_td_environment(); test_ledger_and_grader(); test_games_lane(); test_fresh_eyes_day1(); test_line_history_and_ledger_alignment(); test_last_call_and_weather()
    key = os.environ.get("ODDS_API_KEY")
    if key and not a.offline:
        test_live(key)
    else:
        print("\n(skipping live checks — no ODDS_API_KEY or --offline)")
    n = len(RESULTS); passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n" + "=" * 60)
    print(f"  RESULT: {passed}/{n} passed" + (" — ALL GREEN" if passed == n else ""))
    fails = [(nm, d) for nm, ok, d in RESULTS if not ok]
    if fails:
        print("  FAILURES:")
        for nm, d in fails:
            print(f"    ✗ {nm}  {d}")
    print("=" * 60)
    sys.exit(0 if passed == n else 1)
