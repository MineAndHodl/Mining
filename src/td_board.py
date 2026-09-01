"""The TD board — surfaces the calibrated anytime-TD model against live prices.

Pulls posted anytime-TD lines + game spreads/totals, computes each player's model
P(scores a TD) with the live game-environment multiplier, and ranks by edge vs the
posted (vig-included) price. Feeds both the standalone TD board and the Touchdown Tax
parlays. Anytime-TD is a one-sided, fat-margin market, so the edge bar is higher.
"""
import os, sys, json, urllib.request
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import td_model as tdm

API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
ABBR = {"Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF",
"Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE",
"Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
"Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX","Kansas City Chiefs":"KC",
"Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LA","Miami Dolphins":"MIA",
"Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
"New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","San Francisco 49ers":"SF",
"Seattle Seahawks":"SEA","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN","Washington Commanders":"WAS"}


def _get(url):
    r = urllib.request.urlopen(url, timeout=30)
    return json.load(r), r.headers.get("x-requests-remaining")


def fetch_td_and_env(key, days=8, max_games=None):
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    frm = now.strftime("%Y-%m-%dT%H:%M:%SZ"); to = (now+datetime.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    events, rem = _get(f"{API}/events?apiKey={key}&commenceTimeFrom={frm}&commenceTimeTo={to}")
    ids = [e["id"] for e in events][:max_games] if max_games else [e["id"] for e in events]
    td_rows, env = [], {}
    for eid in ids:
        try:
            d, rem = _get(f"{API}/events/{eid}/odds?apiKey={key}&regions=us&markets=player_anytime_td,spreads,totals&oddsFormat=american&bookmakers=draftkings")
        except Exception:
            continue
        home, away = ABBR.get(d.get("home_team")), ABBR.get(d.get("away_team"))
        gid = f"{d.get('away_team')} @ {d.get('home_team')}"
        spread_home = total = None
        for bk in d.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m["key"] == "spreads":
                    for o in m["outcomes"]:
                        if ABBR.get(o["name"]) == home: spread_home = o.get("point")
                elif m["key"] == "totals":
                    total = m["outcomes"][0].get("point")
                elif m["key"] == "player_anytime_td":
                    for o in m["outcomes"]:
                        td_rows.append({"name": o.get("description") or o.get("name"),
                                        "price": o["price"], "game": gid, "team_hint": None})
        if home and total and spread_home is not None:
            env[(gid, home, away)] = {"spread_home": spread_home, "total": total, "home": home, "away": away}
    return pd.DataFrame(td_rows), env, rem


def build_td_board(key=None, max_games=None, days=8):
    key = key or os.environ.get("ODDS_API_KEY")
    rates = pd.read_parquet(f"{DATA}/td_rates.parquet")
    baseline = json.load(open(f"{DATA}/team_td_baseline.json"))
    board, env, rem = fetch_td_and_env(key, days=days, max_games=max_games)
    if not len(board):
        return pd.DataFrame(), rem
    # env multiplier per team from each game's live spread/total
    env_mult = {}
    for (gid, home, away), e in env.items():
        mm = tdm.env_multipliers(baseline, e["spread_home"], e["total"], home, away)
        env_mult.update(mm)
    out = tdm.price_board(rates, board, env_mult)
    return out, rem


if __name__ == "__main__":
    mg = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    b, rem = build_td_board(max_games=mg)
    print(f"credits: {rem} | TD players priced: {len(b)}")
    if len(b):
        good = b[b["edge"].notna() & (b["edge"] >= 0.03)].head(15)
        print("\n=== TD BOARD — best anytime-TD edges ===")
        print(good[["name","game","price","p_model","p_implied","edge","games"]].to_string(index=False))
