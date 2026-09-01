"""Moneyline board — best value on favorites AND underdogs.

Honest framing consistent with our findings: we do NOT beat the closing line picking
winners, so the ML edge is the same CLV/off-market edge — a retail book pricing a team's
win probability softer than Pinnacle's sharp de-vigged number. We surface the best value
on each side (fav and dog), market-anchored so a model over-rate can't create a fake edge.

For each game: Pinnacle fair win prob (de-vig) = the sharp truth. A bettable book whose
ML implies a lower prob than fair (i.e., better price) by >= min_ev is value. Dogs and
favorites are shown separately because bettors ask for both.
"""
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weekly_card as wc


def american_profit(o):
    return o/100 if o > 0 else 100/abs(o)


def build_ml_board(key=None, days=8, min_ev=0.02):
    key = key or os.environ.get("ODDS_API_KEY")
    data, rem = wc.fetch_odds(key, days=days)
    board = wc.build_board(data)
    if not len(board):
        return pd.DataFrame(), rem
    fair = wc.pinnacle_fair(board)
    rows = []
    ml = board[(board["market"] == "h2h") & (board["book"].isin(wc.BETTABLE_BOOKS))]
    for _, r in ml.iterrows():
        f = fair.get((r["game"], "h2h"), {}).get(r["side"])
        if not f:
            continue
        fp = f["fair_p"]                       # sharp win prob
        ev = fp * american_profit(r["price"]) - (1 - fp)
        if ev >= min_ev:
            # favorite vs dog by the sharp prob
            role = "favorite" if fp >= 0.5 else "underdog"
            rows.append({"game": r["game"], "team": r["side"], "role": role,
                         "price": int(r["price"]), "book": r["book"],
                         "fair_win_prob": round(fp, 3), "ev": round(ev, 3),
                         "fair_price": round(-100*fp/(1-fp)) if fp >= 0.5 else round(100*(1-fp)/fp)})
    df = pd.DataFrame(rows)
    if not len(df):
        return df, rem
    return df.sort_values("ev", ascending=False).drop_duplicates(["game", "team"]), rem


if __name__ == "__main__":
    b, rem = build_ml_board(days=14)
    print(f"credits: {rem} | ML value plays: {len(b)}")
    if len(b):
        for role in ("favorite", "underdog"):
            sub = b[b.role == role].head(6)
            print(f"\n=== BEST {role.upper()} ML VALUE ===")
            print(sub[["team","game","price","fair_price","fair_win_prob","ev","book"]].to_string(index=False) if len(sub) else "  none")
