"""Underdog Fantasy pick'em lines — a second, public, player-prop market (3,800+ NFL lines a
week: yards, receptions, attempts, anytime/first TD, 1Q/1H splits, longest play). Not a
sportsbook we bet, but a number to compare every line against and a free line-move feed.

    parse(js)  -> tidy DataFrame [name, team, stat, mkt, line, over_price, under_price, game, kickoff, updated]
    latest()   -> the newest snapshot parsed;  history() -> every snapshot stacked (line moves)
"""
import os, glob, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data"); UNI = os.path.join(DATA, "universe", "underdog_lines")
STAT_TO_MKT = {"Receiving Yards": "rec_yds", "Receptions": "receptions", "Rush Yards": "rush_yds", "Pass Yards": "pass_yds",
               "Rush Attempts": "rush_att", "Rush + Rec TDs": "anytime_td", "Pass TDs": "pass_tds", "Pass Attempts": "pass_att",
               "Pass Completions": "pass_cmp", "Rush + Rec Yards": "rush_rec_yds", "Kicking Points": "kicking_points", "FG Made": "field_goals",
               "Targets": "targets", "Longest Reception": "longest_rec", "Longest Rush": "longest_rush", "Interceptions": "interceptions"}


def parse(js):
    ap = {a["id"]: a for a in js.get("appearances", [])}
    pl = {p["id"]: p for p in js.get("players", [])}
    gm = {g["id"]: g for g in js.get("games", [])}
    rows = []
    for o in js.get("over_under_lines", []):
        ou = o.get("over_under") or {}; st = ou.get("appearance_stat") or {}
        a = ap.get(st.get("appearance_id"))
        if not a: continue
        p = pl.get(a.get("player_id"))
        if not p or p.get("sport_id") != "NFL": continue
        g = gm.get(a.get("match_id")) if a.get("match_type") == "Game" else None
        prices = {opt.get("choice"): opt.get("american_price") for opt in (o.get("options") or [])}
        rows.append({"name": f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(), "team_id": p.get("team_id"),
                     "position": p.get("position_name"), "stat": st.get("display_stat"), "mkt": STAT_TO_MKT.get(st.get("display_stat")),
                     "line": float(o.get("stat_value")) if o.get("stat_value") not in (None, "") else None,
                     "over_price": int(prices.get("higher")) if prices.get("higher") not in (None, "") else None,
                     "under_price": int(prices.get("lower")) if prices.get("lower") not in (None, "") else None,
                     "line_type": o.get("line_type"), "live": bool(o.get("live_event")),
                     "game": (g or {}).get("title"), "kickoff": (g or {}).get("scheduled_at"), "ud_id": o.get("id")})
    return pd.DataFrame(rows)


def snapshots():
    return sorted(glob.glob(os.path.join(UNI, "*.json")))


def latest():
    fs = snapshots()
    if not fs: return pd.DataFrame()
    df = parse(json.load(open(fs[-1]))); df["stamp"] = os.path.basename(fs[-1])[:13]
    return df


def history(max_snapshots=40):
    out = []
    for f in snapshots()[-max_snapshots:]:
        try:
            d = parse(json.load(open(f))); d["stamp"] = os.path.basename(f)[:13]; out.append(d)
        except Exception:
            pass
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def moves(min_pts=0.5):
    """Lines that moved between the previous and the latest snapshot (main markets only)."""
    h = history(2)
    if h.empty or h.stamp.nunique() < 2: return pd.DataFrame()
    s0, s1 = sorted(h.stamp.unique())[-2:]
    a = h[(h.stamp == s0) & h.mkt.notna() & ~h.live].drop_duplicates(["name", "stat"])
    b = h[(h.stamp == s1) & h.mkt.notna() & ~h.live].drop_duplicates(["name", "stat"])
    m = a.merge(b, on=["name", "stat", "mkt"], suffixes=("_open", "_now"))
    m["moved"] = m.line_now - m.line_open
    return m[m.moved.abs() >= min_pts][["name", "stat", "mkt", "line_open", "line_now", "moved", "game_now"]].sort_values("moved", key=abs, ascending=False)


if __name__ == "__main__":
    d = latest()
    print(f"{len(d):,} NFL lines, {d.name.nunique()} players, main-market lines: {d.mkt.notna().sum()}")
    print(d[d.mkt.notna()].head(10).to_string())
