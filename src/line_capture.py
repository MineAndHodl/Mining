"""Multi-book line capture — the raw material for CLV.

Pulls every book's number for a market at a point in time, live OR historical, and
normalizes it to one long table: (game_id, commence, home, away, book, market, side,
point, price, snapshot_ts). That table is the panel everything else reads.

Two anchors matter and are labeled here:
  - EXECUTION baseline = DraftKings (where our audience actually bets).
  - SHARP truth        = Pinnacle / lowvig (low-margin, high-limit; the number DK drifts
    toward by kickoff). The DK-minus-sharp gap is the CLV signal.

Cost note (The Odds API): historical games snapshot = 10 credits x markets x regions
(spreads+totals, us+eu = 40). Live games = 1 x markets x regions. Props are per-event.
"""
import os, json, urllib.request, urllib.error, datetime
import pandas as pd

API = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
SHARP_BOOKS = ["pinnacle", "lowvig"]
EXEC_BASELINE = "draftkings"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def _get(url):
    req = urllib.request.urlopen(url, timeout=60)
    return json.load(req), {"cost": req.headers.get("x-requests-last"),
                            "remaining": req.headers.get("x-requests-remaining")}


def _rows_from_games(games, snapshot_ts):
    """Flatten Odds API game objects -> long rows. Spread stored as HOME point; total as Over."""
    rows = []
    for g in games:
        gid = g["id"]; home = g["home_team"]; away = g["away_team"]; comm = g["commence_time"]
        for bk in g.get("bookmakers", []):
            book = bk["key"]
            for m in bk.get("markets", []):
                mk = m["key"]
                for o in m.get("outcomes", []):
                    side = o["name"]; pt = o.get("point"); pr = o.get("price")
                    # normalize side labels
                    if mk == "spreads":
                        s = "home" if side == home else ("away" if side == away else side)
                    elif mk == "totals":
                        s = o["name"].lower()          # over / under
                    else:  # h2h
                        s = "home" if side == home else ("away" if side == away else side)
                    rows.append({"game_id": gid, "commence": comm, "home": home, "away": away,
                                 "book": book, "market": mk, "side": s, "point": pt, "price": pr,
                                 "snapshot_ts": snapshot_ts})
    return rows


def fetch_games(key=None, regions="us,eu", markets="spreads,totals,h2h",
                historical=False, date=None, days=13):
    """One snapshot of game lines across all books. historical=True needs date (RFC3339)."""
    key = key or os.environ.get("ODDS_API_KEY")
    if historical:
        url = (f"{API}/historical/sports/{SPORT}/odds?apiKey={key}&regions={regions}"
               f"&markets={markets}&oddsFormat=american&date={date}")
        raw, meta = _get(url)
        ts = raw.get("timestamp"); games = raw.get("data", [])
    else:
        now = datetime.datetime.now(datetime.timezone.utc)
        frm = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        to = (now + datetime.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = (f"{API}/sports/{SPORT}/odds?apiKey={key}&regions={regions}&markets={markets}"
               f"&oddsFormat=american&commenceTimeFrom={frm}&commenceTimeTo={to}")
        games, meta = _get(url)
        ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    df = pd.DataFrame(_rows_from_games(games, ts))
    return df, meta


def fetch_props_for_events(key=None, event_ids=None, markets=None, regions="us",
                           historical=False, date=None):
    """Player props are per-event. Returns the same long-row shape (side = player name;
    point = line; a 'prop_side' captures Over/Under). 5 credits/event/market-group live."""
    key = key or os.environ.get("ODDS_API_KEY")
    markets = markets or "player_pass_yds,player_rush_yds,player_reception_yds,player_receptions,player_rush_attempts,player_anytime_td"
    rows = []; meta = {}
    for eid in (event_ids or []):
        if historical:
            url = (f"{API}/historical/sports/{SPORT}/events/{eid}/odds?apiKey={key}"
                   f"&regions={regions}&markets={markets}&oddsFormat=american&date={date}")
            raw, meta = _get(url); obj = raw.get("data", {}); ts = raw.get("timestamp")
        else:
            url = (f"{API}/sports/{SPORT}/events/{eid}/odds?apiKey={key}"
                   f"&regions={regions}&markets={markets}&oddsFormat=american")
            obj, meta = _get(url); ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if not obj:
            continue
        home = obj.get("home_team"); away = obj.get("away_team"); comm = obj.get("commence_time")
        for bk in obj.get("bookmakers", []):
            book = bk["key"]
            for m in bk.get("markets", []):
                mk = m["key"]
                for o in m.get("outcomes", []):
                    rows.append({"game_id": eid, "commence": comm, "home": home, "away": away,
                                 "book": book, "market": mk,
                                 "side": o.get("description", o.get("name")),  # player name
                                 "prop_side": o.get("name"),                    # Over/Under/Yes
                                 "point": o.get("point"), "price": o.get("price"),
                                 "snapshot_ts": ts})
    return pd.DataFrame(rows), meta


def consensus_and_gaps(df, market="spreads", side="home"):
    """For each game: DK line, sharp line (mean of pinnacle/lowvig), and the DK-minus-sharp gap.
    Positive gap on a home spread means DK has the home team as a bigger dog than the sharps do."""
    sub = df[(df.market == market) & (df.side == side)]
    out = []
    for gid, g in sub.groupby("game_id"):
        dk = g[g.book == EXEC_BASELINE]["point"]
        sharp = g[g.book.isin(SHARP_BOOKS)]["point"]
        allb = g["point"].dropna()
        if not len(dk) or not len(sharp):
            continue
        dk_v = float(dk.iloc[0]); sh_v = float(sharp.mean())
        row = g.iloc[0]
        out.append({"game_id": gid, "home": row["home"], "away": row["away"],
                    "commence": row["commence"], "market": market, "side": side,
                    "dk": dk_v, "sharp": sh_v, "gap_dk_minus_sharp": round(dk_v - sh_v, 2),
                    "book_spread": round(float(allb.max() - allb.min()), 2), "n_books": len(allb)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    df, meta = fetch_games()
    print("live snapshot:", len(df), "rows |", df.book.nunique(), "books | cost", meta["cost"])
    g = consensus_and_gaps(df, "spreads", "home")
    print(g[["away", "home", "dk", "sharp", "gap_dk_minus_sharp", "book_spread"]].to_string(index=False))
