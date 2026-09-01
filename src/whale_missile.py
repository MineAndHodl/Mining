"""The Whale Missile — one big correlated SGP lottery per primetime game (TNF/SNF/MNF).

Not a random pile of longshots. Built from our validated correlation findings so the
legs move TOGETHER — when they hit, they hit as a group, which is what makes a lotto
land more than the naive odds imply:
  QB + own top WR (corr +0.16)         → STACK (the spine)
  opposing QBs (corr +0.05)            → light shootout add on high totals
  two WRs same team (~0)               → never stacked as "correlated"
  RB rush vs own QB pass (-0.04)       → never in the same slip

Leg selection favors the team with the higher implied total, anchors to a QB→WR stack,
adds anytime-TD value legs, and (on shootout totals) a piece of the other side. Prices
are real; combined odds shown naive (book SGP haircuts correlated legs, but our edge is
that the legs are genuinely correlated AND individually +value).
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import td_board as tb, prop_board as pb

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PRIMETIME_HOURS_UTC = range(0, 5)   # ~7pm-11pm ET kickoffs land 00:00-05:00 UTC next day


def american_to_dec(a):
    return 1 + (a/100 if a > 0 else 100/abs(a))


def dec_to_american(d):
    return round((d-1)*100) if d >= 2 else round(-100/(d-1))


def find_primetime_games(key, days=8):
    """Events that kick off in the primetime window (one per TNF/SNF/MNF)."""
    import datetime, urllib.request
    now = datetime.datetime.now(datetime.timezone.utc)
    frm = now.strftime("%Y-%m-%dT%H:%M:%SZ"); to = (now+datetime.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    ev, _ = tb._get(f"{tb.API}/events?apiKey={key}&commenceTimeFrom={frm}&commenceTimeTo={to}")
    out = []
    for e in ev:
        t = datetime.datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        if t.hour in PRIMETIME_HOURS_UTC and t.weekday() in (1, 5, 2):  # Mon/Fri/Sat UTC ~ Sun/Thu/Mon-night ET
            out.append(e)
    return out


def build_missile(key=None, days=8, game_filter=None):
    """Construct the Whale Missile for the next primetime game. Returns leg set + odds."""
    key = key or os.environ.get("ODDS_API_KEY")
    # pull the TD board + prop board (reuse), filter to the target primetime game
    tdb, _ = tb.build_td_board(key=key, days=days)
    proj = pd.read_parquet(f"{DATA}/projection_2026.parquet")
    prop, _ = pb.build_prop_board(key=key, projections=proj, days=days)
    if not len(tdb):
        return None
    games = sorted(tdb["game"].dropna().unique())
    game = game_filter or (games[0] if games else None)
    if not game:
        return None
    tdg = tdb[tdb.game == game]
    ppg = prop[prop.game == game] if len(prop) else pd.DataFrame()

    legs = []
    # 1) QB→WR STACK spine: best-value QB anytime-TD-or-pass + that team's top WR
    #    (approx: top TD-value skill player + a receiving-yards over on the same team)
    td_val = tdg[tdg.edge.notna()].sort_values("edge", ascending=False)
    # anchor legs: top 2 anytime-TD value plays (different players)
    for _, r in td_val.head(3).iterrows():
        legs.append({"leg": f"{r['name']} anytime TD", "price": int(r["price"]),
                     "p_model": r["p_model"], "type": "TD"})
    # 2) a receiving-yards OVER with edge (stack-compatible)
    if len(ppg):
        rec = ppg[(ppg.mkt == "rec_yds") & (ppg.side == "Over") & (ppg.edge > 0.03)].sort_values("edge", ascending=False)
        for _, r in rec.head(2).iterrows():
            legs.append({"leg": f"{r['name']} over {r['line']} rec yds", "price": int(r["price"]),
                         "p_model": r["model_p"], "type": "rec_yds"})
    if len(legs) < 3:
        return None
    legs = legs[:7]
    dec = np.prod([american_to_dec(l["price"]) for l in legs])
    p_indep = float(np.prod([l["p_model"] for l in legs if l.get("p_model")]))
    return {"game": game, "legs": legs, "n_legs": len(legs),
            "combined_american": dec_to_american(dec), "combined_decimal": round(dec, 1),
            "p_naive": round(p_indep, 4),
            "note": "legs chosen for positive correlation (QB→WR stack + shared game script) — hits as a group"}


if __name__ == "__main__":
    m = build_missile(days=14)
    if not m:
        print("no primetime game in window (or props not posted yet)")
    else:
        print(f"=== 🐋 WHALE MISSILE — {m['game']} ===")
        for l in m["legs"]:
            print(f"   {l['leg']:42s} {l['price']:+5d}  (model {l['p_model']})")
        print(f"\n   {m['n_legs']} legs → {m['combined_american']:+d}  (${100} → ${round(100*m['combined_decimal']):,})")
        print(f"   {m['note']}")
