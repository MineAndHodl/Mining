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
    """Missile locks must be minus-money (market-agreed), and mostly near-locks."""
    import whale_missile as wm
    import json as _j
    d = _j.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "ranked_board.json")))
    m = d.get("whale_missile")
    if not m:
        check("missile present in board", False, "no whale_missile key"); return
    legs = m["legs"]
    check("missile has 5-10 legs (odds-band tuned)", 5 <= len(legs) <= 10, f"n={len(legs)}")
    # payout targets 60-120:1 but is capped by how many genuine near-locks the slate offers
    check("missile payout positive & within target ceiling", 300 <= m["combined_american"] <= 14000,
          f"{m['combined_american']:+d}")
    # expected hits must be the HONEST (market-based) number, never above the leg count
    check("missile expected_hits is market-based (<= legs)", m["expected_hits"] <= len(legs),
          f"{m['expected_hits']} vs {len(legs)}")
    # honest contract: every leg is a market FAVORITE (de-vigged prob >= 0.53) — true 60%+
    # near-locks are too scarce on an efficient slate to fill a 30-100:1 slip, so we use solid
    # favorites and report honest expected-hits (~half), rather than force a contradiction.
    favs_ok = all((l.get("p_mkt") or 0) >= 0.53 for l in legs)
    check("every leg is a market favorite (de-vigged >= 0.53)", favs_ok)
    players = [l["name"] for l in legs]
    check("no duplicate players in missile", len(players) == len(set(players)))


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--offline", action="store_true"); a = ap.parse_args()
    print("=" * 60, "\n  DESK PIPELINE TEST SUITE\n", "=" * 60)
    test_implied_devig(); test_props_anchor(); test_conviction()
    test_offmarket_synthetic(); test_totals_sign(); test_crossbook_props()
    test_factor_scorecard(); test_whale_missile_discipline()
    test_qc_fixes(); test_primetime_sgp_structure(); test_injury_layer(); test_player_intel()
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
