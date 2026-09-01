"""Weekly card generator.

Usage: ODDS_API_KEY=xxx python3 weekly_card.py

Pipeline:
1. Refresh nflverse data (current-season pbp), rebuild ratings as-of now.
2. Fetch live multi-book odds from The Odds API (us + eu regions; eu includes
   Pinnacle, the sharp anchor).
3. De-vig Pinnacle to get "fair" prices. Flag:
   a. OFF-MARKET NUMBERS: any retail book whose spread/total differs from
      Pinnacle fair line by >= 1.0 pt (>= 0.5 across key numbers 3/7), or
      whose ML price beats Pinnacle fair probability by >= 2% EV.
      -> These are the primary bets (positive expected CLV).
   b. MODEL DELTAS: model number vs consensus >= 2.5 pts, as context only.
4. Emit the card as CSV + formatted text.
"""
import os, sys, json, datetime
import urllib.request
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import build_team_games, ratings_asof, OFFSEASON_WEEKS

API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
KEY_NUMBERS = {3.0, 7.0}
SHARP_BOOKS = ["pinnacle"]
# US books Alex can actually bet -> real flags; everything else -> market intel only
BETTABLE_BOOKS = ["draftkings", "fanduel", "betmgm", "fanatics", "betrivers", "williamhill_us"]
MAX_QUOTE_AGE_MIN = 30


def fmt_line(side, point):
    """Render a spread/total side unambiguously: 'Buffalo Bills -1.5' / 'Under 42.5'."""
    if point is None:
        return side
    return f"{side} {point:+g}" if side not in ("Over", "Under") else f"{side} {point:g}"

def implied(o):
    return 100/(o+100) if o > 0 else abs(o)/(abs(o)+100)

def fetch_odds(key):
    url = (f"{API}?apiKey={key}&regions=us,eu&markets=h2h,spreads,totals"
           f"&oddsFormat=american")
    with urllib.request.urlopen(url, timeout=30) as r:
        remaining = r.headers.get("x-requests-remaining")
        data = json.load(r)
    return data, remaining

def build_board(data):
    """Long dataframe: game, book, market, side, point, price."""
    rows = []
    for g in data:
        gid = f"{g['away_team']} @ {g['home_team']}"
        for bk in g.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                for oc in mkt.get("outcomes", []):
                    rows.append({
                        "game": gid, "home": g["home_team"], "away": g["away_team"],
                        "commence": g["commence_time"], "book": bk["key"],
                        "market": mkt["key"], "side": oc["name"],
                        "point": oc.get("point"), "price": oc["price"],
                        "last_update": mkt.get("last_update"),
                    })
    return pd.DataFrame(rows)

def pinnacle_fair(board):
    """De-vigged Pinnacle probabilities / lines per game+market."""
    out = {}
    p = board[board["book"].isin(SHARP_BOOKS)]
    for (game, market), grp in p.groupby(["game", "market"]):
        if len(grp) != 2:
            continue
        a, b = grp.iloc[0], grp.iloc[1]
        pa, pb = implied(a["price"]), implied(b["price"])
        s = pa + pb
        out[(game, market)] = {
            a["side"]: {"point": a["point"], "fair_p": pa/s},
            b["side"]: {"point": b["point"], "fair_p": pb/s},
        }
    return out

def find_off_market(board, fair, min_pt=1.0, min_ev=0.02):
    """Retail numbers beating the Pinnacle fair line."""
    flags = []
    retail = board[~board["book"].isin(SHARP_BOOKS)]
    for _, r in retail.iterrows():
        f = fair.get((r["game"], r["market"]), {}).get(r["side"])
        if not f:
            continue
        if r["market"] in ("spreads", "totals") and r["point"] is not None and f["point"] is not None:
            # for spreads: getting MORE points on same side is better.
            diff = (r["point"] - f["point"]) if r["market"] == "spreads" else 0
            if r["market"] == "totals":
                diff = (r["point"] - f["point"]) if r["side"] == "Under" else (f["point"] - r["point"])
            crosses_key = any(min(r["point"] or 0, f["point"] or 0) < k <= max(r["point"] or 0, f["point"] or 0)
                              for k in KEY_NUMBERS) if r["market"] == "spreads" else False
            if diff >= min_pt or (diff >= 0.5 and crosses_key):
                flags.append({**r.to_dict(), "vs_fair_pts": round(diff, 1),
                              "fair_point": f["point"],
                              "line_str": fmt_line(r["side"], r["point"]),
                              "fair_str": fmt_line(r["side"], f["point"]),
                              "class": "FLAG" if r["book"] in BETTABLE_BOOKS else "intel",
                              "type": "off-market line"})
        elif r["market"] == "h2h":
            ev = f["fair_p"] * (r["price"]/100 if r["price"] > 0 else 100/abs(r["price"])) - (1 - f["fair_p"])
            if ev >= min_ev:
                flags.append({**r.to_dict(), "ev": round(ev, 3),
                              "fair_p": round(f["fair_p"], 3), "type": "ML +EV vs fair"})
    return pd.DataFrame(flags)

def model_numbers(data_dir):
    tg = build_team_games(f"{data_dir}/team_game_epa.parquet", None)
    seasons = sorted(tg["season"].unique())
    base = {s: i*(18+OFFSEASON_WEEKS) for i, s in enumerate(seasons)}
    now_week = max(base.values()) + 18 + OFFSEASON_WEEKS + 1  # extend past latest data
    r = ratings_asof(tg, now_week)
    coefs = np.load(f"{data_dir}/../output/coefs.npy", allow_pickle=True).item()
    return r.set_index("team"), coefs

if __name__ == "__main__":
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        sys.exit("Set ODDS_API_KEY")
    data, remaining = fetch_odds(key)
    print(f"Odds API requests remaining this month: {remaining}")
    board = build_board(data)
    fair = pinnacle_fair(board)
    flags = find_off_market(board, fair)
    stamp = datetime.date.today().isoformat()
    board.to_csv(f"board_{stamp}.csv", index=False)
    flags.to_csv(f"card_{stamp}.csv", index=False)
    print(flags.to_string(index=False) if len(flags) else "No off-market numbers right now.")


# ---- Wong teaser flagger -----------------------------------------------
def wong_teaser_legs(board, fair):
    """Basic-strategy 6-pt teaser legs: favorites -7.5 to -8.5 (tease through
    3 and 7) and dogs +1.5 to +2.5 (tease through 3 and 7). Historically the
    only parlay-family bet with documented +EV at standard pricing; grade
    2-leg combos at -120 to -130 and flag when both legs qualify."""
    legs = []
    sp = board[board["market"] == "spreads"]
    for _, r in sp.iterrows():
        pt = r["point"]
        if pt is None: continue
        if -8.5 <= pt <= -7.5 or 1.5 <= pt <= 2.5:
            legs.append({"game": r["game"], "side": r["side"], "point": pt,
                         "book": r["book"], "teased_to": pt + 6,
                         "type": "wong-leg"})
    import pandas as pd
    return pd.DataFrame(legs).drop_duplicates(subset=["game", "side"]) if legs else pd.DataFrame()


# ---- DraftKings execution view ------------------------------------------
DK = "draftkings"
EXECUTION_BOOKS = ["draftkings", "fanduel", "betmgm"]  # Alex's betting set

def best_execution(board, flags):
    """Across the execution book set, attach the best available number per flag
    and which book has it — the card's 'where to bet' column."""
    import pandas as pd
    if not len(flags): return flags
    rows=[]
    for _, f in flags.iterrows():
        same = board[(board["game"]==f["game"])&(board["market"]==f["market"])
                     &(board["side"]==f["side"])&(board["book"].isin(EXECUTION_BOOKS))]
        r = f.to_dict()
        if len(same):
            if f["market"]=="spreads":
                best = same.loc[same["point"].idxmax()]
            elif f["market"]=="totals":
                best = same.loc[same["point"].idxmax() if f["side"]=="Under" else same["point"].idxmin()]
            else:
                best = same.loc[same["price"].idxmax()]
            r["bet_at"], r["bet_point"], r["bet_price"] = best["book"], best["point"], best["price"]
        else:
            r["bet_at"] = None
        rows.append(r)
    out = pd.DataFrame(rows)
    return out.sort_values("bet_at", na_position="last")


def dk_view(board, flags):
    """For each flagged bet, attach what DK offers on the same side right now
    and whether DK is the best available number. Sort DK-bettable first."""
    import pandas as pd
    if not len(flags):
        return flags
    rows = []
    for _, f in flags.iterrows():
        same = board[(board["game"] == f["game"]) & (board["market"] == f["market"])
                     & (board["side"] == f["side"])]
        dk = same[same["book"] == DK]
        best_pt = same["point"].max() if f["market"] == "spreads" else None
        r = f.to_dict()
        if len(dk):
            r["dk_point"], r["dk_price"] = dk.iloc[0]["point"], dk.iloc[0]["price"]
            r["dk_is_best"] = (f["market"] != "spreads") or (dk.iloc[0]["point"] == best_pt)
            r["bettable_on_dk"] = True
        else:
            r["bettable_on_dk"] = False
        rows.append(r)
    out = pd.DataFrame(rows)
    return out.sort_values(["bettable_on_dk"], ascending=False)
