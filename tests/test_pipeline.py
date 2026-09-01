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
        # Over side should be shopped to the book with best Over price (fanduel +110)
        r = sh.iloc[0]
        if r["side"] == "Over":
            check("cross-book picks best Over price (fanduel)", r["book"] == "fanduel", f"book={r['book']}")
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--offline", action="store_true"); a = ap.parse_args()
    print("=" * 60, "\n  DESK PIPELINE TEST SUITE\n", "=" * 60)
    test_implied_devig(); test_props_anchor(); test_conviction()
    test_offmarket_synthetic(); test_totals_sign(); test_crossbook_props()
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
