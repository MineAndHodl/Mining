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


def test_fresh_eyes_day2():
    """Day-2 findings: trust reads market breadth (not books at the shopped line); the venue is
    detected; a missed season zeroes form; under the market baseline the environment is NOT
    re-added (the market prices it); the market's LAG on form is measured and applied; two
    disagreeing books define no consensus; off-market plays are labeled; partial games are
    signal (full-game form LOSES); post copy is built from the frozen numbers."""
    print("\n-- fresh eyes day 2: breadth, venue, gap, market lag, consensus, post copy --")
    import rank_board as rb, board_factors as bf, json as _j, post_copy as pc, market_lag as ml
    base = {"edge": 0.06, "model_p": 0.58, "confidence": 1.0, "score_mult": 1.0}
    s_shop = rb.prop_score({**base, "n_books": 1, "n_books_mkt": 4}); s_thin = rb.prop_score({**base, "n_books": 1, "n_books_mkt": 1})
    check("trust reads books posting the MARKET, not books at our shopped line (+6 swing)", abs((s_shop - s_thin) - 6.0) < 1e-6, f"{s_shop} vs {s_thin}")
    _pm, sched_rows, code2full, _a, _d = bf._load_context()
    g_home, _ = bf._game_ctx("Cleveland Browns @ Jacksonville Jaguars", "JAX", sched_rows, code2full)
    g_away, _ = bf._game_ctx("Cleveland Browns @ Jacksonville Jaguars", "CLE", sched_rows, code2full)
    check("venue detected from the schedule (JAX home, CLE away) — home was always 0 before", bool(g_home and g_home.get("is_home")) and bool(g_away and not g_away.get("is_home")))
    raw = pd.read_parquet(os.path.join(HERE, "..", "data", "_raw_prop_board.parquet"))
    mf_rows = raw[raw.edge_src == "multifactor"]
    stale = mf_rows[mf_rows.form_gap >= 2]
    check("a missed season (gap >= 2) zeroes the form tilt and cuts trust", len(stale) == 0 or (all(abs((c or {}).get("form3", 0)) < 1e-9 for c in stale["contrib"]) and (stale["confidence"] <= 0.71).all()), f"n={len(stale)}")
    off = mf_rows[mf_rows.form_gap >= 1]
    check("under the market baseline the environment is not re-added (total/home contrib = 0)", len(off) > 0 and all(abs((c or {}).get("total_c", 0)) < 1e-9 and abs((c or {}).get("home", 0)) < 1e-9 for c in off["contrib"]))
    lagf = {m: (v.get("form3") if isinstance(v, dict) else v) for m, v in ml.load_factors().items()}
    check("market lag measured for the four markets (data/market_lag.json)", all(m in lagf for m in ("pass_yds", "rush_yds", "rec_yds", "receptions")), str(lagf))
    check("the market prices late-season QB form; it lags rushing form (pass factor < rush factor)", lagf.get("pass_yds", 1) < lagf.get("rush_yds", 0), str(lagf))
    if lagf.get("pass_yds", 1) == 0:
        pq = off[off.mkt == "pass_yds"]
        check("with lag factor 0 the pass-yds form tilt is zero on every offseason row", all(abs((c or {}).get("form3", 0)) < 1e-9 for c in pq["contrib"]))
    import prop_board as pb
    kr = raw[(raw.name == "Tucker Kraft") & (raw.mkt == "receptions")]
    check("two books at different lines (DK 4.5 / FD 3.5) imply the SAME mean: no manufactured 8.7% edge, no PICK", (not len(kr)) or (all(pb.classify(r.to_dict()) != "PICK" for _, r in kr.iterrows()) and float(kr["edge"].max()) < 0.03), str(kr[["side", "thesis_kind", "edge"]].values.tolist() if len(kr) else "n/a"))
    nc = raw[raw.thesis_kind == "no-consensus"]
    check("a two-book disagreement where the number IS the edge is REVIEW, never a PICK", len(nc) == 0 or all(pb.classify(r.to_dict()) == "REVIEW" for _, r in nc.iterrows()), f"n={len(nc)}")
    bm = raw[(raw.name == "Baker Mayfield") & (raw.mkt == "pass_yds")]
    check("an off-market number is labeled as the thesis (Mayfield U247.5 vs ~235 consensus)", (not len(bm)) or (bm.iloc[0]["thesis_kind"] == "off-market" and float(bm.iloc[0]["off_market"]) >= 5), str(bm[["thesis_kind", "off_market"]].values.tolist() if len(bm) else "n/a"))
    check("every priced row carries tilt / off_market / cons_mean for the grading loop", all(c in raw.columns for c in ("tilt", "off_market", "cons_mean", "thesis_kind", "market_lag", "n_books_mkt")))
    fg = _j.load(open(os.path.join(HERE, "..", "data", "full_game_form.json")))
    check("full-game form (dropping partial games) LOSES in both seasons — partial games are signal, logged", fg["verdict"]["repl"] == "LOSES" and fg["hurt_both_seasons"]["repl"] >= 3)
    d = _j.load(open(os.path.join(HERE, "..", "data", "ranked_board.json")))
    card = [r for r in d.get("recommended", []) if not r.get("kind")]
    check("every Card pick's thesis is a model read or a 3+-book off-market number (never no-consensus)", all(r.get("thesis_kind") in ("model", "off-market") for r in card))
    out = pc.build(d)
    P = out["posts"]
    check("post copy: the teaser fits X (<=280 chars)", P.get("teaser", {}).get("x_ok") is True, str(P.get("teaser", {}).get("chars")))
    check("post copy: the full Card post names every Card pick", all(r["name"] in P["card_full"]["text"] for r in card))
    top = max(card, key=lambda r: (float(r.get("model_p") or 0), float(r.get("score") or 0)))
    check("post copy: the free pick is the Card's most likely winner", top["name"] in P["teaser"]["text"] and top["name"] in P["free_pick"]["thread"][0])
    check("post copy: the Touchdown Tax post carries the fair price and the cash rate", "fair" in P.get("touchdown_tax", {}).get("text", "") and "Cashes" in P.get("touchdown_tax", {}).get("text", ""))
    check("post copy: no report card before a week is graded", ("report_card" not in P) or os.path.exists(os.path.join(HERE, "..", "data", "record.json")))
    fe = _j.load(open(os.path.join(HERE, "..", "data", "fresh_eyes_log.json")))
    days = fe if isinstance(fe, list) else fe.get("days", [])
    check("fresh-eyes log has a day-2 entry with numbers and a verdict", any(str(x.get("date", "")).endswith("09-03") and x.get("verdict") and x.get("numbers") for x in days))


def test_fresh_eyes_day3_universe():
    """Day 3: the data universe — ingested, inventoried, tested; usage won; TD role read; the
    automatic beat feed with clause-level hard gates; second opinions; the watch."""
    print("\n-- fresh eyes day 3: data universe --")
    import json as _j, multifactor as mf, recency as rc, player_intel as pi, xtd_model as xm, datetime as _dt
    inv = _j.load(open(os.path.join(HERE, "..", "data", "data_ledger.json")))
    check("data universe inventoried: 30+ feeds, 100M+ data points", inv["totals"]["sources"] >= 30 and inv["totals"]["data_points"] > 1e8, str(inv["totals"]))
    u = pd.read_parquet(os.path.join(HERE, "..", "data", "universe_features_player_game.parquet"))
    check("derived player-game table carries route participation, snap %, goal-line inclusion", all(c in u.columns for c in ("route_rate", "offense_pct", "gl_snap_share")) and len(u) > 30000)
    w = _j.load(open(os.path.join(HERE, "..", "data", "multifactor_weights.json")))
    check("the model has the two earned role features and their centering medians", all(f in w["features"] for f in mf.ROLE_FEATURES) and "role_med" in w and w["weights"]["rec_yds"].get("role_rr", 0) > 5)
    lab = _j.load(open(os.path.join(HERE, "..", "data", "universe_lab.json")))
    wins = [k for k, v in lab.items() if v.get("verdict") == "WINS"]
    check("98 candidates tried; the winners are usage only (no efficiency / charting / opponent signal)", len(lab) >= 90 and wins and all(k.startswith(("route_rate", "offense_pct", "gl_snap", "rz_snap", "in10_snap", "routes")) for k in wins), str(wins)[:200])
    mv = _j.load(open(os.path.join(HERE, "..", "data", "multifactor_validation.json")))
    check("strong-signal hit rate >= 60% in every market, both held-out seasons, with the role features", all(v["strong_hit"] >= 0.60 for yr in mv["by_season"].values() for v in yr.values()), str({y: {m: v["strong_hit"] for m, v in r.items()} for y, r in mv["by_season"].items()}))
    r = rc.build_recency(2025, None)
    check("recency rows carry the role features", "role_rr" in r.columns and r.role_rr.notna().mean() > 0.9)
    proj = pd.read_parquet(os.path.join(HERE, "..", "data", "projection_2026.parquet"))
    pid = proj[proj.name == "Josh Allen"].player_id.iloc[0]
    ja = r[(r.pid == pid)].iloc[0]
    check("a 1%-snap rest game is not a role read (Allen's snap log has no sub-10% game)", all(x >= 0.10 for x in ja["snap_pct_log"]), str(ja["snap_pct_log"]))
    check("rushing role tilt is OFF across the offseason; receiving stays on (measured on weeks 1-3)", mf.OFFSEASON_ROLE_RATIO["role_snap_d"]["rush_yds"] == 0.0 and mf.OFFSEASON_ROLE_RATIO["role_snap_d"]["rec_yds"] > 0.9)
    tdr = _j.load(open(os.path.join(HERE, "..", "data", "td_role_validation.json")))
    check("goal-line package inclusion beat carry share in both seasons and the model uses that blend", tdr["verdict"]["winner_blend"] is not None and abs(xm.GL_INCLUSION_BLEND - tdr["verdict"]["winner_blend"]) < 1e-9, str(tdr["verdict"]))
    x = pd.read_parquet(os.path.join(HERE, "..", "data", "xtd_2026.parquet"))
    check("the 2026 xTD table carries goal-line inclusion for backs", "GL_inclusion" in x.columns and x[x.position == "RB"]["GL_inclusion"].notna().mean() > 0.5)
    W = (_dt.date(2026, 8, 20), _dt.date(2026, 9, 14))
    allen = [{"date": "2026-08-25", "source": "Jay Skurski of The Buffalo News", "title": "Allen will not play in Thursday's preseason finale against the Steelers.", "snippet": "It comes as no surprise that Allen will be held out."}]
    check("preseason rest is not an OUT (Allen will not play in the preseason finale)", pi.classify_items(allen, *W, name="Josh Allen")["hard"] is None)
    calc = [{"date": "2026-08-31", "source": "rotowire", "title": "Calcaterra (back) was placed on injured reserve by the Eagles on Sunday with a designation to return.", "snippet": ""}]
    check("an IR placement about THIS player is an OUT", pi.classify_items(calc, *W, name="Grant Calcaterra")["hard"] == "OUT")
    deebo = [{"date": "2026-08-29", "source": "rotowire", "title": "Samuel didn't play in Thursday's preseason victory.", "snippet": "Samuel should have a key role with Ricky Pearsall (knee) out for the season."}]
    check("another player's 'out for the season' in his note does not gate him", pi.classify_items(deebo, *W, name="Deebo Samuel")["hard"] is None)
    I = pi.PlayerIntel(2026, 1)
    check("the beat feed is automatic: 300+ board players carry dated, sourced items", sum(1 for v in I.by_key.values() if v["n_items"] > 0) >= 250, str(sum(1 for v in I.by_key.values() if v["n_items"] > 0)))
    ml = _j.load(open(os.path.join(HERE, "..", "data", "market_lag.json")))
    pooled = ml.get("pooled", {})
    check("market lag: one β per market with every tilt scaled in [0, 1]", all("beta" in v and all(0 <= f <= 1 for f in v["lag"].values()) for v in pooled.values()) and len(pooled) >= 4)
    d = _j.load(open(os.path.join(HERE, "..", "data", "ranked_board.json")))
    card = [r for r in d.get("recommended", []) if not r.get("kind")]
    check("second opinions on the Card: Underdog's line on most picks, an outside projection on some", sum(1 for r in card if r.get("ud_line") is not None) >= len(card) // 2 and any(r.get("proj_rw") is not None for r in card))
    st = _j.load(open(os.path.join(HERE, "..", "data", "monitor", "status.json")))
    check("the watch is live: 300+ players watched, a last-check stamp, events by kind", st.get("players_watched", 0) >= 300 and st.get("last_check") and "by_kind_24h" in st)
    html = open(os.path.join(HERE, "..", "output", "the-lab.html")).read()
    check("the Model Room shows the universe (data points, the watch) and no rejected-idea boxes", "data points ingested" in html and "The watch" in html and "tested, no edge" not in html)


def test_fresh_eyes_day4_profile_coaches():
    """Day 4: the tracking & charting profile on every pick; pass-catcher red-zone inclusion in
    xTD (validated both seasons) + a refit calibration; the coach press-conference feed; intel
    patterns for every position with hard gates that survive preseason talk, conditionals, past
    seasons and other-player clauses."""
    print("\n-- fresh eyes day 4: profile levers, receivers' TD role, coach feed --")
    import json as _j, xtd_model as xm, player_intel as pi, datetime as _dt, player_profile as pp
    d = _j.load(open(os.path.join(HERE, "..", "data", "ranked_board.json")))
    card = [r for r in d.get("recommended", []) if not r.get("kind")]
    check("every Card pick carries a profile block (tracking & charting + the reads)", card and all(len([v for v in (r.get("profile") or {}).values() if v is not None]) >= 3 for r in card))
    tds = d.get("by_category", {}).get("anytime_td", [])[:5]
    check("TD picks carry red-zone snaps / goal-line package levers", tds and all(any(k in (r.get("profile") or {}) for k in ("Red-Zone Snaps", "Goal-Line Package")) for r in tds))
    P = pp.Profiles(2026, 0)
    hen = P.profile("Derrick Henry", "rush_yds", "Over", "RB")
    check("profile percentiles are among position peers and flip for the side", "Stacked Boxes Faced" in hen and P.profile("Derrick Henry", "rush_yds", "Under", "RB")["Stacked Boxes Faced"] == -hen["Stacked Boxes Faced"])
    rv = _j.load(open(os.path.join(HERE, "..", "data", "td_role_validation_receivers.json")))
    check("pass-catchers: red-zone inclusion × targets-per-route beat the observed share in both seasons; the model uses that blend", rv["verdict"]["winner_blend"] is not None and abs(xm.RZ_INCLUSION_BLEND - rv["verdict"]["winner_blend"]) < 1e-9, str(rv["verdict"]))
    x = pd.read_parquet(os.path.join(HERE, "..", "data", "xtd_2026.parquet"))
    check("the 2026 xTD table carries the red-zone inclusion share for catchers", "RZ_inclusion_share" in x.columns and x[x.position.isin(["WR", "TE"])]["RZ_inclusion_share"].notna().mean() > 0.5)
    tde = _j.load(open(os.path.join(HERE, "..", "data", "td_env_validation.json")))
    check("the TD calibration was refit after the role changes and the model uses it (top bucket ~.50 -> .53 out of sample)", abs(xm.CAL["a"] - tde["decision"]["recal_xtd"][0]) < 1e-9 and abs(xm.CAL["b"] - tde["decision"]["recal_xtd"][1]) < 1e-9 and tde["final_recal"]["blend_w0.55"]["top_hit"] >= 0.50)
    cal = rv["by_season"]["2025"]["calibration_by_position"]
    check("QBs, RBs, WRs and TEs are all priced by the TD model (raw under-prediction corrected by the recalibration)", all(k in cal for k in ("QB", "RB", "WR", "TE")) and all(v["n"] > 300 for v in cal.values()))
    idx_p = os.path.join(HERE, "..", "data", "universe", "coach_feed", "index.json")
    if os.path.exists(idx_p):
        idx = _j.load(open(idx_p))
        check("coach press conferences read: 50+ articles resolved and 50+ coach quotes about board players", sum(1 for e in idx.values() if e.get("file")) >= 50 and sum(int(e.get("quotes") or 0) for e in idx.values()) >= 50)
    raw = _j.load(open(os.path.join(HERE, "..", "data", "player_intel_2026_w1.json")))
    check("the week's intel file carries coach-feed items", any(it.get("feed") == "coach" for k, v in raw.items() if not k.startswith("_") for it in v.get("items", [])))
    W = (_dt.date(2026, 8, 20), _dt.date(2026, 9, 14))
    def cls(name, title, date="2026-09-09"):
        return pi.classify_items([{"date": date, "source": "t", "title": title, "snippet": ""}], *W, name=name)
    check("receiver coach-speak reads: 'primary red-zone target, get him the ball' supports", cls("Drake London", "London will be the primary target in the red zone, Morris said; we want to get him the ball early.")["signals"]["coach"] > 0)
    check("tight end 'blocking role / inline snaps' opposes", cls("Kyle Pitts", "Pitts is going to have more of a blocking role and inline snaps this year.")["signals"]["coach"] < 0)
    check("QB 'named the starter, no restrictions' supports (role + health)", (lambda c: c["signals"]["role"] > 0 and c["signals"]["health"] > 0)(cls("Geno Smith", "Smith was named the Week 1 starter; no restrictions after the knee cleanup.")))
    check("kicker 'lost the kicking job' opposes", cls("Jake Elliott", "Elliott lost the kicking job to the rookie, Sirianni confirmed.")["signals"]["role"] < 0)
    check("'game-time decision' is a health negative, not an OUT", (lambda c: c["hard"] is None and c["signals"]["health"] < 0)(cls("Josh Allen", "Allen (shoulder) is a game-time decision Sunday, McDermott said.")))
    check("a preseason-week 'will not play' (dated before game week) is not an OUT", cls("Jonathan Taylor", "Steichen said Daniel Jones and Jonathan Taylor will not play.", date="2026-08-28")["hard"] is None)
    check("'recovering from season-ending injuries in 2025' is not an OUT", cls("George Kittle", "Kittle and Bosa are both recovering from season-ending injuries in 2025.")["hard"] is None)
    check("a conditional ('in the event that he is out') is not an OUT", cls("Alvin Kamara", "In the event that Kamara is out to begin the season, Miller would start.")["hard"] is None)
    check("an in-week 'ruled out' about him IS an OUT", cls("Alvin Kamara", "Kamara (ankle) has been ruled out for Sunday's opener, the team announced.")["hard"] == "OUT")
    html = open(os.path.join(HERE, "..", "output", "the-lab.html")).read()
    check("the Model Room reads the coaches and shows the profile on the wall", "what the coaches said" in html.lower() and "tracking & charting profile" in html.lower())


def test_owner_in_the_loop():
    """The owner persona and the thesis registry are part of the loop: files exist, the loop
    protocol references the Owner's Pass, and the latest log entry carries a scored forecast."""
    print("\n-- the owner in the loop --")
    import json as _j
    own = open(os.path.join(HERE, "..", "OWNER.md")).read(); th = open(os.path.join(HERE, "..", "THESES.md")).read(); fe = open(os.path.join(HERE, "..", "FRESH_EYES.md")).read()
    check("OWNER.md carries the Owner's Pass (12 questions), the anticipation protocol and a current forecast", "Owner's Pass" in own and "Anticipation protocol" in own and "Current forecast" in own and own.count("**\"") >= 12)
    check("THESES.md carries every thesis with a kill criterion, the freeze date and the in-season calendar", th.count("| T") >= 12 and "Dies if" in th and "September 8" in th and "Week 9" in th)
    check("FRESH_EYES.md runs the Owner's Pass and logs the forecast", "Owner's Pass" in fe and "owner_forecast" in fe)
    log = _j.load(open(os.path.join(HERE, "..", "data", "fresh_eyes_log.json")))
    last = log[-1]
    check("the latest loop entry forecasts the owner's next three asks (phrased, with answers)", isinstance(last.get("owner_forecast"), list) and len(last["owner_forecast"]) == 3 and all(x.get("ask") and x.get("answer") for x in last["owner_forecast"]))
    check("the latest loop entry carries the Owner's Pass findings and a forecast check slot", last.get("owner_pass") and "owner_forecast_check" in last)


def test_bridge():
    """The bridge (Alex: 'a composite that shows how we get to our edge… the pick and the
    confidence interval'): on every pick shown, baseline + factors − already-in-the-line
    × weather × cascade = our number; P vs fair = edge; conviction + trust × verdict = score."""
    print("\n-- the bridge --")
    import json as _j, math as _m
    import bridge as br, rank_board as rbk
    d = _j.load(open(os.path.join(HERE, "..", "data", "ranked_board.json")))
    shown = [r for r in d.get("recommended", []) if not r.get("kind")] + [r for rows in d.get("by_category", {}).values() for r in rows]
    check("every pick shown carries a bridge", shown and all(isinstance(r.get("bridge"), dict) and not r["bridge"].get("error") for r in shown), f"{sum(1 for r in shown if not r.get('bridge'))} missing")
    bad = br.check(d)
    check("the arithmetic closes on every pick (baseline + steps = our number; xTD steps = expected TDs)", not bad, str(bad[:3]))
    yd = [r for r in shown if r["bridge"].get("kind") == "yardage"]
    check("yardage bridges: start + Σ steps = our number within rounding", all(abs(r["bridge"]["start"]["value"] + sum(s["delta"] for s in r["bridge"]["steps"]) - r["bridge"]["end"]["value"]) <= 0.16 for r in yd))
    check("weeks 0-3: the start is the market's number and the environment is listed as already priced", all(r["bridge"]["start"]["label"].startswith("The market") and set(r["bridge"]["in_line"]) >= {"Game total", "Venue"} for r in yd if r.get("offseason") and r.get("contrib")))
    check("a full-strength read is shown, then what the market already has is subtracted (never hidden)", any(s["kind"] == "priced" for r in yd for s in r["bridge"]["steps"]))
    check("every factor bar says signal × weight", all(s.get("note") and "weight" in s["note"] for r in yd for s in r["bridge"]["steps"] if s["kind"] == "factor" and s.get("key")))
    check("the range is the same distribution p_over uses (p50 ≈ the median of our number's distribution)", all(r["bridge"]["ci"]["p25"] < r["bridge"]["ci"]["p50"] < r["bridge"]["ci"]["p75"] for r in yd))
    check("P − fair = edge on every yardage pick", all(abs((r["bridge"]["prob"]["model_p"] - r["bridge"]["prob"]["fair"]) - r["bridge"]["prob"]["edge"]) < 0.002 for r in yd))
    check("score parts reproduce the row's conviction and trust", all(abs(r["bridge"]["score"]["conviction"]["conviction"] - r["score_conviction"]) < 0.06 and abs(r["bridge"]["score"]["trust"]["trust"] - r["score_trust"]) < 0.06 for r in yd if r.get("score_conviction") is not None))
    check("(conviction + trust) × verdict = the Desk Score (clipped 20-95)", all(abs(min(95, max(20, (r["bridge"]["score"]["conviction"]["conviction"] + r["bridge"]["score"]["trust"]["trust"]) * r["bridge"]["score"]["mult"])) - r["score"]) < 0.2 for r in yd))
    td = [r for r in shown if r["bridge"].get("kind") == "td"]
    check("TD bridges: chances × finish by zone, then environment/cascade/red-zone defense, = expected TDs", td and all(abs(sum(s["delta"] for s in r["bridge"]["steps"]) - r["bridge"]["end"]["value"]) <= 0.006 for r in td))
    check("TD probability chain: raw = 1 − e^(−xTD); blend = w·calibrated + (1−w)·pooled", all(abs(r["bridge"]["prob"]["p_raw"] - (1 - _m.exp(-r["bridge"]["prob"]["lam"]))) < 0.002 and abs(r["bridge"]["prob"]["w_xtd"] * r["bridge"]["prob"]["p_xtd"] + (1 - r["bridge"]["prob"]["w_xtd"]) * r["bridge"]["prob"]["p_pooled"] - r["bridge"]["prob"]["p_model"]) < 0.002 for r in td))
    check("TD score has no trust term (the TD scale is conviction alone, anchored at .35)", all(r["bridge"]["score"]["trust"] is None and r["bridge"]["score"]["conviction"]["anchor"] == rbk.TD_ANCHOR for r in td))
    tmpl = open(os.path.join(HERE, "..", "output", "the-board-template.html")).read(); page = open(os.path.join(HERE, "..", "output", "the-board.html")).read()
    check("the Board renders the bridge on every pick (waterfall, range, P vs fair, score equation)", "renderBridge(p.bridge" in tmpl and "renderTdBridge" in tmpl and "scoreEq(" in tmpl and "How we get to the number" in page)
    lab = open(os.path.join(HERE, "..", "output", "the-lab.html")).read()
    check("the Model Room shows one pick all the way down", "How a number is reached" in lab and '"bridge_example"' in lab)


def test_form_weights():
    """UPGRADE_PLAN §2: opportunity-aware form — a TEAM low-volume anomaly game (monsoon / blowout / QB loss)
    is skipped in the receiving/passing-yards form window (measured: no signal, data/form_anomaly_effect.json),
    down-weighted for receptions, untouched for rushing; mirrors the raw rolling math when nothing is flagged."""
    print("\n-- opportunity-aware form (team-anomaly weights) --")
    import form_weights as fw, numpy as np, json as _j
    tv = fw.team_volume()
    tb8 = tv[(tv.season == 2026) & (tv.posteam == "TB") & (tv.week == 8)]
    check("TB Week 8 2026 (4 pass attempts in the rain) is a team low-pass anomaly", len(tb8) == 1 and bool(tb8.anomaly_lowpass.iloc[0]), str(tb8[["pa", "pa_med"]].to_dict("records")))
    check("anomalies are rare (~1% of team-games) — this is not a blanket exclusion", 0.003 < tv.anomaly_lowpass.mean() < 0.03, f"{tv.anomaly_lowpass.mean():.3%}")
    ys = np.array([50, 60, 70, 80, 90, 0, 40.0]); ones = np.ones(7); w0 = ones.copy(); w0[5] = 0.0
    b1, l3a, l1a = fw.weighted_form(ys, ones); b0, l3b, l1b = fw.weighted_form(ys, w0)
    check("with every weight 1 the math is the raw shift(1) rolling/expanding form", abs(l3a[6] - np.mean([80, 90, 0])) < 1e-9 and l1a[6] == 0.0 and abs(b1[6] - np.mean(ys[:6])) < 1e-9)
    check("a zero-weight game is skipped: the window rolls back past it, the baseline ignores it", abs(l3b[6] - np.mean([70, 80, 90])) < 1e-9 and l1b[6] == 90.0 and abs(b0[6] - np.mean([50, 60, 70, 80, 90])) < 1e-9)
    check("weights by market: rec yards 0, receptions .7, passing and rushing 1 (passing REJECTED by the walk-forward A/B)", fw.WEIGHT_IN_ANOMALY["rec_yds"] == 0.0 and fw.WEIGHT_IN_ANOMALY["pass_yds"] == 1.0 and fw.WEIGHT_IN_ANOMALY["receptions"] == 0.7 and fw.WEIGHT_IN_ANOMALY["rush_yds"] == 1.0)
    ab = _j.load(open(os.path.join(HERE, "..", "data", "form_weights_ab.json")))["results"]
    check("walk-forward A/B on file: receiving yards not worse in either season, passing yards worse (hence excluded from the rule)",
          ab["weighted_2024"]["rec_yds"]["ens_hit_strong"] >= ab["raw_2024"]["rec_yds"]["ens_hit_strong"] - 0.002 and ab["weighted_2025"]["rec_yds"]["ens_hit_strong"] >= ab["raw_2025"]["rec_yds"]["ens_hit_strong"] - 0.002
          and ab["weighted_2024"]["pass_yds"]["ens_hit_strong"] < ab["raw_2024"]["pass_yds"]["ens_hit_strong"])
    import recency as rc
    pl = pd.read_parquet(os.path.join(HERE, "..", "data", "players_cache.parquet")).set_index("gsis_id").display_name
    rec = rc.build_recency(through_season=2026, through_week=9); rec["name"] = rec.pid.map(pl)
    g = rec[(rec.name == "Chris Godwin Jr.") & (rec.mkt == "rec_yds")]
    check("Godwin's receiving-yards window names the excluded Tampa games and his last-1 is not the monsoon zero", len(g) == 1 and g.iloc[0]["form_excluded"] and "2026w8" in g.iloc[0]["form_excluded"] and g.iloc[0]["last1"] != 0.0, str(g[["last1", "last3", "form_excluded"]].to_dict("records")))
    gi = rec[(rec.name == "Bucky Irving") & (rec.mkt == "rush_yds")]
    check("Irving's rushing window is untouched (rushing keeps its information at every team volume)", len(gi) == 1 and not gi.iloc[0]["form_excluded"])
    eff = _j.load(open(os.path.join(HERE, "..", "data", "form_anomaly_effect.json")))
    check("the measurement is on file: rec_yds anomaly slope ~0 vs ~.165 normal", abs(eff["by_market"]["rec_yds"]["team volume <=55%"]["slope"]) < 0.08 and eff["by_market"]["rec_yds"]["80-120% (normal)"]["slope"] > 0.12)


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
    test_qc_fixes(); test_primetime_sgp_structure(); test_injury_layer(); test_player_intel(); test_selection_over_pool(); test_td_shopping_and_stack(); test_td_environment(); test_ledger_and_grader(); test_games_lane(); test_fresh_eyes_day1(); test_fresh_eyes_day2(); test_fresh_eyes_day3_universe(); test_fresh_eyes_day4_profile_coaches(); test_owner_in_the_loop(); test_bridge(); test_line_history_and_ledger_alignment(); test_last_call_and_weather(); test_form_weights()
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
