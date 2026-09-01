"""Desk card orchestrator — the single entry point that builds the What I Like board.

Ties together every priced source into one ranked conviction card:
  1. Off-market spread/total/ML flags   (weekly_card.find_off_market + best_execution)
  2. Market-anchored continuous props    (props_model.anchored_edge)
  3. Cross-book prop line-shopping        (shop_prop_edges, below)
  -> conviction.build_card ranks/tiers everything, adds Desk Confidence.

Design rule: cross-book shopping and edge computation are ONE step for props.
We evaluate the market-anchored edge at each execution book's own line+prices and
keep the book/side that yields the largest honest edge — i.e. we bet where the
number is softest, and the edge is measured against that same book's de-vig. No
phantom "consensus edge" that isn't actually bettable anywhere.

Usage:
  ODDS_API_KEY=xxx python3 desk_card.py            # live
  ODDS_API_KEY=xxx python3 desk_card.py --no-props  # flags only (cheap)

Player props cost 1 credit per event*market region, so props default to the two
calibrated markets only (reception yards + anytime TD feed elsewhere). Uncalibrated
markets are pulled for the full board but NEVER enter conviction (gated in
conviction.ELIGIBLE_PROP_MARKETS).
"""
import os, sys, json, datetime, argparse
import urllib.request, urllib.error
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weekly_card as wc
import props_model as pm
import conviction as cv

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
EVENTS_API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
EVENT_ODDS_API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{eid}/odds"

# Odds API market key -> our internal model market key
PROP_MARKET_MAP = {
    "player_reception_yds": "rec_yds",
    "player_rush_yds": "rush_yds",
    "player_pass_yds": "pass_yds",
    "player_receptions": "receptions",
}
# only pull the calibrated market live by default (credit discipline); the rest are
# available for the full board when explicitly requested.
DEFAULT_LIVE_MARKETS = ["player_reception_yds"]
EXECUTION_BOOKS = wc.EXECUTION_BOOKS   # draftkings, fanduel, betmgm


# ---------------------------------------------------------------- props fetch
def fetch_events(key):
    with urllib.request.urlopen(f"{EVENTS_API}?apiKey={key}", timeout=30) as r:
        return json.load(r), r.headers.get("x-requests-remaining")


def fetch_event_props(key, eid, markets):
    url = (EVENT_ODDS_API.format(eid=eid) + f"?apiKey={key}&regions=us"
           f"&markets={','.join(markets)}&oddsFormat=american")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r), r.headers.get("x-requests-remaining")
    except urllib.error.HTTPError as e:
        if e.code == 422:      # market not offered for this event yet
            return None, None
        raise


def prop_board_rows(event, markets_map=PROP_MARKET_MAP):
    """Flatten one event's player-prop bookmakers into per (player, market, book) rows
    carrying the Over line/price and Under price. Handles the two shapes the API uses
    for over/under props (name=Over/Under, description=player)."""
    gid = f"{event['away_team']} @ {event['home_team']}"
    # keyed by (book, market, player) -> {'line','over','under'}
    acc = {}
    for bk in event.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            mk = markets_map.get(mkt["key"])
            if not mk:
                continue
            for oc in mkt.get("outcomes", []):
                player = oc.get("description") or oc.get("name")
                side = oc.get("name")
                if side not in ("Over", "Under"):
                    # some feeds put player in name and side in description; skip odd shapes
                    continue
                point = oc.get("point")
                price = oc.get("price")
                k = (bk["key"], mk, player)
                acc.setdefault(k, {"line": point})
                if side == "Over":
                    acc[k]["over"] = price
                    acc[k]["line"] = point
                else:
                    acc[k]["under"] = price
    rows = []
    for (book, mk, player), v in acc.items():
        if "over" in v and "under" in v and v.get("line") is not None:
            rows.append({"game": gid, "book": book, "mkt": mk, "name": player,
                         "line": float(v["line"]), "over_price": v["over"],
                         "under_price": v["under"]})
    return rows


def fetch_all_props(key, markets=None, max_events=None):
    markets = markets or DEFAULT_LIVE_MARKETS
    events, rem = fetch_events(key)
    rows, remaining = [], rem
    for i, ev in enumerate(events):
        if max_events and i >= max_events:
            break
        data, rem2 = fetch_event_props(key, ev["id"], markets)
        if rem2:
            remaining = rem2
        if data:
            rows.extend(prop_board_rows(data))
    return pd.DataFrame(rows), remaining


# --------------------------------------------- cross-book prop edge + shopping
def shop_prop_edges(prop_board, rates=None, env_by_team=None, opp_by_pid=None,
                    books=EXECUTION_BOOKS, edge_min=pm.EDGE_MIN):
    """For each (player, market), evaluate the market-anchored edge at EACH execution
    book's own line/prices and keep the single softest number — the book+side where
    our edge is largest. This is cross-book shopping and edge computation in one:
    we only claim an edge we can actually bet, priced against that book's de-vig.

    Returns a DataFrame with one row per player-market: best book, side, edge, line,
    price, model prob and projected mean. rates is optional (used only to attach a
    games-of-data confidence for scoring); without it, confidence defaults mid.
    """
    if not len(prop_board):
        return pd.DataFrame()
    env_by_team = env_by_team or {}
    opp_by_pid = opp_by_pid or {}
    best = {}
    for _, b in prop_board.iterrows():
        if b["book"] not in books:
            continue
        mk = b["mkt"]
        # anchored_edge does its own de-vig from this book's own over/under prices,
        # then tilts by environment/opponent. Tilt inputs default neutral (1.0).
        side, edge, p_over, proj = pm.anchored_edge(
            mk, b["line"], b["over_price"], b["under_price"],
            env=env_by_team.get(b["game"], 1.0),
            opp=opp_by_pid.get(b["name"], 1.0),
        )
        price = b["over_price"] if side == "Over" else b["under_price"]
        key = (b["name"], mk)
        cur = best.get(key)
        if cur is None or edge > cur["edge"]:
            best[key] = {"name": b["name"], "mkt": mk, "side": side,
                         "line": b["line"], "price": price, "book": b["book"],
                         "edge": round(edge, 3), "p_model": p_over, "proj": proj,
                         "game": b["game"]}
    out = pd.DataFrame(best.values())
    if not len(out):
        return out
    # attach games-of-data confidence for the conviction score
    if rates is not None and len(rates):
        gm = rates.set_index(["pid"]) if "pid" in rates.columns else None
        name_games = {}
        if "name" in rates.columns:
            for _, rr in rates.iterrows():
                name_games[(rr.get("name"), rr["mkt"])] = int(rr.get("n", 0))
        out["games"] = out.apply(lambda r: name_games.get((r["name"], r["mkt"]), 8), axis=1)
    else:
        out["games"] = 8      # neutral prior when we can't join usage history
    return out[out["edge"] >= edge_min].sort_values("edge", ascending=False)


# ------------------------------------------------------------------ scoring glue
def score_flags_for_card(flags):
    """Turn find_off_market/best_execution output into conviction-scored dicts."""
    scored = []
    if not len(flags):
        return scored
    for _, f in flags.iterrows():
        if f.get("class") == "intel":      # offshore/EU intel never a pick
            continue
        if f.get("type") == "off-market line":
            crosses = bool(any(min(f.get("point", 0) or 0, f.get("fair_point", 0) or 0) < k
                               <= max(f.get("point", 0) or 0, f.get("fair_point", 0) or 0)
                               for k in wc.KEY_NUMBERS)) if f["market"] == "spreads" else False
            s = cv.score_flag(f.get("vs_fair_pts", 0.0), crosses)
            label = f.get("line_str", wc.fmt_line(f["side"], f.get("point")))
            where = f.get("bet_at") or f.get("book")
            scored.append({"score": s, "label": label, "market": f["market"],
                           "game": f["game"], "edge_pts": f.get("vs_fair_pts"),
                           "where": where, "kind": "off-market",
                           "dedup": (f["game"], f["market"], f["side"])})
        elif f.get("type") == "ML +EV vs fair":
            s = cv.score_flag(0.0, False, ev=f.get("ev"))
            scored.append({"score": s, "label": f"{f['side']} ML", "market": "h2h",
                           "game": f["game"], "ev": f.get("ev"),
                           "where": f.get("book"), "kind": "ml-ev",
                           "dedup": (f["game"], "h2h", f["side"])})
    return scored


def score_props_for_card(shopped):
    """Turn shopped prop edges into conviction-scored dicts (eligible markets only)."""
    scored = []
    if not len(shopped):
        return scored
    for _, p in shopped.iterrows():
        s = cv.score_prop(p["edge"], p.get("games", 8), p["mkt"])
        if s <= 0:                          # gated market
            continue
        scored.append({"score": s, "label": f"{p['name']} {p['side']} {p['line']:g} {p['mkt']}",
                       "market": p["mkt"], "game": p["game"], "edge": p["edge"],
                       "p_model": p["p_model"], "proj": p["proj"], "price": p["price"],
                       "where": p["book"], "games": p.get("games", 8), "kind": "prop",
                       "dedup": (p["game"], p["mkt"], p["name"], p["side"])})
    return scored


def build_desk_card(key=None, do_props=True, prop_markets=None, max_events=None, days=8):
    """Full pipeline -> the ranked What I Like board dict + the raw scored pools."""
    key = key or os.environ.get("ODDS_API_KEY")
    if not key:
        raise SystemExit("Set ODDS_API_KEY")
    data, remaining = wc.fetch_odds(key, days=days)
    board = wc.build_board(data)
    if not len(board) or "book" not in board.columns:
        empty = cv.build_card(flags=[], props=[])
        return {"card": empty, "flags": [], "props": [], "odds_remaining": remaining,
                "props_remaining": remaining, "note": "no games in window"}
    fair = wc.pinnacle_fair(board)
    # find_off_market now returns the best bettable number per side directly, with a
    # coherent (line, book, edge). No separate best_execution pass — that was what
    # stapled a mismatched venue onto a flagged number.
    flags = wc.find_off_market(board, fair)
    flag_dicts = score_flags_for_card(flags)

    prop_dicts, prop_rem = [], remaining
    if do_props:
        try:
            rates = None
            pg_path = os.path.join(DATA_DIR, "prop_player_games.parquet")
            if os.path.exists(pg_path):
                rates = pm.build_rates(pd.read_parquet(pg_path))
            prop_board, prop_rem = fetch_all_props(key, markets=prop_markets, max_events=max_events)
            shopped = shop_prop_edges(prop_board, rates=rates)
            prop_dicts = score_props_for_card(shopped)
        except Exception as e:
            print(f"[props skipped: {e}]", file=sys.stderr)

    card = cv.build_card(flags=flag_dicts, props=prop_dicts)
    return {"card": card, "flags": flag_dicts, "props": prop_dicts,
            "odds_remaining": remaining, "props_remaining": prop_rem}


def render_card(result):
    c = result["card"]
    lines = ["=" * 60, "  THE DESK — WHAT I LIKE", "=" * 60, f"  verdict: {c['verdict']}", ""]
    if not c["plays"]:
        lines.append("  (no plays clear the bar — sitting on hands)")
    for i, p in enumerate(c["plays"], 1):
        tag = p["tier"]
        dc = p["desk_conf"]
        extra = ""
        if p.get("kind") == "prop":
            extra = f"  edge {p['edge']*100:.1f}% | proj {p['proj']} | {p['price']:+d} @ {p['where']}"
        elif p.get("kind") == "off-market":
            extra = f"  +{p.get('edge_pts')}pt vs fair @ {p.get('where')}"
        elif p.get("kind") == "ml-ev":
            extra = f"  {p.get('ev', 0)*100:+.1f}% EV @ {p.get('where')}"
        lines.append(f"  {i}. [{tag} · {dc}] {p['label']}   ({p['game']})")
        if extra:
            lines.append(f"       {extra}")
    lines += ["", f"  odds credits remaining: {result.get('odds_remaining')}",
              f"  prop credits remaining: {result.get('props_remaining')}", "=" * 60]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-props", action="store_true")
    ap.add_argument("--markets", nargs="*", default=None)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--days", type=int, default=8)
    a = ap.parse_args()
    res = build_desk_card(do_props=not a.no_props, prop_markets=a.markets,
                          max_events=a.max_events, days=a.days)
    print(render_card(res))
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M")
    pd.DataFrame(res["card"]["plays"]).to_csv(f"whatilike_{stamp}.csv", index=False)
