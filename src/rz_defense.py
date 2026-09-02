"""Opponent red-zone defense — a MEASURED, small multiplier on expected TDs.

Tested walk-forward on 2023-25 team-games (season-to-date opponent red-zone conversion
allowed, no lookahead): a soft red-zone defense (worst third) lets an offense score ~6%
more TDs than its own average; a stingy one ~3% fewer (slope +0.57 TDs per +100% relative
conversion on a ~2.4-TD base -> ~+24% TDs per +100% rel). Real, small: applied as a capped
multiplier on a player's expected TDs. The player's role matters an order of magnitude more.

Week 1 has no current-season data, so the prior season's rate is shrunk halfway to league.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
SLOPE = 0.24          # TD multiplier = 1 + SLOPE * (opp RZ conversion allowed relative to league)
CAP = (0.94, 1.06)
PRIOR_SEASON_SHRINK = 0.5


def _conv_table(pbp):
    d = pbp[pbp.two_point_attempt == 0].copy()
    d["td"] = ((d.rush_touchdown == 1) | (d.pass_touchdown == 1)).astype(int)
    rz = d[d.yardline_100 <= 20].groupby(["season", "week", "game_id", "posteam", "defteam", "drive"]).agg(td=("td", "max")).reset_index()
    return rz.groupby(["season", "week", "defteam"]).agg(trips=("drive", "size"), tds=("td", "sum")).reset_index()


class RZDefense:
    def __init__(self, target_season=2026, through_week=None):
        pbp = pd.read_parquet(f"{DATA}/pbp_2021_2025.parquet")
        t = _conv_table(pbp)
        cur = t[t.season == target_season]
        if through_week is not None:
            cur = cur[cur.week < through_week]
        prev = t[t.season == target_season - 1]
        lg = (prev.tds.sum() + 2) / (prev.trips.sum() + 4)
        self.rel = {}
        teams = set(prev.defteam) | set(cur.defteam)
        for tm in teams:
            p = prev[prev.defteam == tm]; c = cur[cur.defteam == tm]
            conv_prev = (p.tds.sum() + 2) / (p.trips.sum() + 4) if len(p) else lg
            rel_prev = (conv_prev / lg - 1) * (1 - PRIOR_SEASON_SHRINK)
            if len(c) and c.trips.sum() >= 12:
                conv_cur = (c.tds.sum() + 2) / (c.trips.sum() + 4)
                w = min(1.0, c.trips.sum() / 60.0)               # current season takes over by ~60 trips
                rel = w * (conv_cur / lg - 1) + (1 - w) * rel_prev
            else:
                rel = rel_prev
            self.rel[tm] = float(rel)

    def multiplier(self, opp):
        rel = self.rel.get(opp)
        if rel is None: return 1.0, None
        m = float(np.clip(1 + SLOPE * rel, *CAP))
        note = None
        if m >= 1.03: note = f"a soft red-zone defense (allows ~{int(rel*100):+d}% vs league inside the 20)"
        elif m <= 0.97: note = f"a stingy red-zone defense (allows ~{int(rel*100):+d}% vs league inside the 20)"
        return m, note


if __name__ == "__main__":
    r = RZDefense(2026)
    rows = sorted(r.rel.items(), key=lambda x: x[1])
    print("stingiest RZ defenses (prior season, shrunk):", [(t, round(v, 2)) for t, v in rows[:5]])
    print("softest:", [(t, round(v, 2)) for t, v in rows[-5:]])
