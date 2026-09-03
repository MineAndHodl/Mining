"""Second opinions on every prop row — logged for the grading loop, never trusted blind:
  ud_line / ud_over / ud_under   Underdog's line and prices for the same player-market (a second market)
  proj_rw                        RotoWire's weekly projection (via Sleeper) for the market's stat
  proj_espn                      ESPN's weekly projection for the market's stat
Each is a number the loop can grade against the market and against us after a few weeks
(does the second market lead the books? do outside projections add to the market baseline?).
"""
import os, glob, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data"); UNI = os.path.join(DATA, "universe")
import sys; sys.path.insert(0, HERE)
from names import norm_name

RW_STAT = {"rec_yds": "rec_yd", "receptions": "rec", "rush_yds": "rush_yd", "rush_att": "rush_att", "pass_yds": "pass_yd", "anytime_td": None}
ESPN_STAT = {"rec_yds": "rec_yds", "receptions": "receptions", "rush_yds": "rush_yds", "rush_att": "rush_att", "pass_yds": "pass_yds"}


def _underdog():
    try:
        import underdog as ud
        d = ud.latest()
        if d.empty: return {}
        d = d[d.mkt.notna() & ~d.live].drop_duplicates(["name", "mkt"])
        return {(norm_name(r.name), r.mkt): (r.line, r.over_price, r.under_price) for r in d.itertuples()}
    except Exception:
        return {}


def _rotowire(week=None):
    fs = sorted(glob.glob(os.path.join(UNI, "sleeper_proj_week", "*.json")))
    if not fs: return {}
    try:
        js = json.load(open(fs[-1]))
        sp = pd.read_parquet(os.path.join(UNI, "sleeper_players.parquet"), columns=["player_id", "full_name"])
        nm = dict(zip(sp.player_id.astype(str), sp.full_name))
    except Exception:
        return {}
    out = {}
    for it in js:
        name = nm.get(str(it.get("player_id")))
        if not name: continue
        st = it.get("stats") or {}
        out[norm_name(name)] = {k: st.get(v) for k, v in RW_STAT.items() if v}
        out[norm_name(name)]["td"] = (st.get("rush_td") or 0) + (st.get("rec_td") or 0)
    return out


def _espn(week=None):
    p = os.path.join(UNI, "espn_fantasy_proj.parquet")
    if not os.path.exists(p): return {}
    try:
        d = pd.read_parquet(p)
    except Exception:
        return {}
    d = d[d.week.notna() & (d.week > 0)]
    if week is not None:
        d = d[d.week == week] if (d.week == week).any() else d
    d = d.sort_values("week").drop_duplicates("name", keep="first")
    out = {}
    for r in d.itertuples():
        out[norm_name(r.name)] = {k: getattr(r, v, None) for k, v in ESPN_STAT.items()}
        out[norm_name(r.name)]["td"] = (getattr(r, "rush_td", 0) or 0) + (getattr(r, "rec_td", 0) or 0)
    return out


def attach(ranked, week=None):
    ud = _underdog(); rw = _rotowire(week); es = _espn(week)
    n = 0
    def one(r):
        nonlocal n
        k = norm_name(r.get("name", "")); mkt = r.get("cat") or r.get("mkt")
        u = ud.get((k, mkt))
        if u:
            r["ud_line"], r["ud_over"], r["ud_under"] = u; n += 1
        p = rw.get(k)
        if p:
            v = p.get("td") if mkt == "anytime_td" else p.get(mkt)
            if v is not None: r["proj_rw"] = round(float(v), 2)
        p = es.get(k)
        if p:
            v = p.get("td") if mkt == "anytime_td" else p.get(mkt)
            if v is not None and v == v: r["proj_espn"] = round(float(v), 2)
        return r
    for key in ("recommended", "best_props", "pool"):
        for r in ranked.get(key, []) or []:
            one(r)
    for rows in (ranked.get("by_category") or {}).values():
        for r in rows: one(r)
    return n


if __name__ == "__main__":
    rb = json.load(open(os.path.join(DATA, "ranked_board.json")))
    print("attached Underdog lines to", attach(rb), "rows")
    for r in rb["recommended"]:
        print(f"  {r['name']:22s} {r['cat']:11s} line {r['line']} | UD {r.get('ud_line')} ({r.get('ud_over')}/{r.get('ud_under')}) | RW {r.get('proj_rw')} | ESPN {r.get('proj_espn')} | ours {r.get('our_num')}")
