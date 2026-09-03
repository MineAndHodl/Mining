"""Player profile — the tracking & charting levers on every scorecard (Alex, day 4).

Two groups, both on the same -1..+1 player-direction scale the scorecard uses:

  PROFILE (tracking & charting): where the player sits among his position peers on the
  measured-but-unweighted signals — separation, YAC over expected, rush yards over expected,
  yards after contact, stacked boxes faced, catchable-ball and contested-target rates, drops,
  CPOE, pressure faced, air-yards share, targets per route. They describe the player; the
  lab showed they add nothing to form + usage for yardage, so THEY DO NOT MOVE THE NUMBER —
  the page says so — and they are logged with every pick so the grading loop can re-test them.

  READS (real signals, shown as levers): the second market's line vs the books, the outside
  projections vs the line, the crowd (Sleeper trending), the team's pace and pass rate, and the
  goal-line package for backs. Also unweighted today (logged), except that the goal-line
  package is inside xTD.

Percentiles are computed among peers at the same position over the reference season (last
season before Week 4, this season from Week 4), from data/universe_features_player_game.parquet.
"""
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
import sys; sys.path.insert(0, HERE)
from names import norm_name

# lever -> (column, raises_the_stat?, positions, markets)
PROFILE_LEVERS = {
    "Separation":          ("ngs_sep",         True,  ("WR", "TE"),           ("rec_yds", "receptions", "anytime_td")),
    "YAC Over Expected":   ("ngs_yac_oe",      True,  ("WR", "TE", "RB"),     ("rec_yds",)),
    "Air-Yards Share":     ("air_yards_share", True,  ("WR", "TE"),           ("rec_yds",)),
    "Targets per Route":   ("tprr",            True,  ("WR", "TE", "RB"),     ("rec_yds", "receptions")),
    "Catchable Balls":     ("ftn_catchable",   True,  ("WR", "TE", "RB"),     ("rec_yds", "receptions")),
    "Contested Targets":   ("ftn_contested",   False, ("WR", "TE"),           ("rec_yds", "receptions")),
    "Drops":               ("drop_rate",       False, ("WR", "TE", "RB"),     ("rec_yds", "receptions")),
    "Yards Over Expected": ("ngs_ryoe_att",    True,  ("RB", "QB"),           ("rush_yds",)),
    "After Contact":       ("pfr_yac_att",     True,  ("RB",),                ("rush_yds",)),
    "Broken Tackles":      ("bt_rate",         True,  ("RB", "WR", "TE"),     ("rush_yds", "rec_yds")),
    "Stacked Boxes Faced": ("ngs_box8_pct",    False, ("RB",),                ("rush_yds",)),
    "Completion Over Exp.": ("ngs_cpoe",       True,  ("QB",),                ("pass_yds",)),
    "Depth of Target":     ("ngs_iay_qb",      True,  ("QB",),                ("pass_yds",)),
    "Pressure Faced":      ("pfr_pressured_pct", False, ("QB",),              ("pass_yds",)),
    "Bad Throws":          ("pfr_bad_throw_pct", False, ("QB",),              ("pass_yds",)),
    "Red-Zone Snaps":      ("rz_snap_share",   True,  ("WR", "TE", "RB"),     ("anytime_td",)),
    "Goal-Line Package":   ("gl_snap_share",   True,  ("RB", "FB"),           ("anytime_td",)),
}
MIN_GAMES = 4


def _season_ref(season, weeks_played):
    return (season, None) if weeks_played and weeks_played >= 4 else (season - 1, None)


class Profiles:
    def __init__(self, season=2026, weeks_played=0):
        ref, _ = _season_ref(season, weeks_played)
        p = os.path.join(DATA, "universe_features_player_game.parquet")
        self.table = {}; self.ref = ref
        if not os.path.exists(p):
            return
        u = pd.read_parquet(p)
        u = u[u.season == ref].copy()
        u["drop_rate"] = u.pfr_drops / u.tgt.replace(0, np.nan)
        u["bt_rate"] = (u.pfr_bt_rush.fillna(0) + u.pfr_bt_rec.fillna(0)) / (u.carries.fillna(0) + u.tgt.fillna(0)).replace(0, np.nan)
        cols = sorted({v[0] for v in PROFILE_LEVERS.values()} & set(u.columns))
        agg = u.groupby(["pid", "position"]).agg(**{c: (c, "mean") for c in cols}, games=("game_id", "nunique")).reset_index()
        agg = agg[agg.games >= MIN_GAMES]
        # percentile within position, per column
        for c in cols:
            agg[f"{c}_pct"] = agg.groupby("position")[c].rank(pct=True)
        pl = pd.read_parquet(os.path.join(DATA, "universe", "players.parquet"), columns=["gsis_id", "display_name"]) if os.path.exists(os.path.join(DATA, "universe", "players.parquet")) else None
        names = dict(zip(pl.gsis_id, pl.display_name)) if pl is not None else {}
        for r in agg.to_dict("records"):
            nm = names.get(r["pid"])
            if nm:
                self.table[norm_name(nm)] = r

    def profile(self, name, mkt, side, position=None):
        """{lever: lean in [-1,1] or None} for this player-market, flipped for the side."""
        r = self.table.get(norm_name(name))
        if not r:
            return {}
        pos = position or r.get("position")
        out = {}
        for lever, (col, raises, poss, mkts) in PROFILE_LEVERS.items():
            if mkt not in mkts or pos not in poss:
                continue
            pct = r.get(f"{col}_pct")
            if pct is None or (isinstance(pct, float) and pct != pct):
                out[lever] = None; continue
            lean = (float(pct) - 0.5) * 2.0                  # -1 (bottom of peers) .. +1 (top)
            if not raises: lean = -lean                        # a high drop rate LOWERS the stat
            if side in ("Under",): lean = -lean               # flip for the side (helps/hurts our bet)
            out[lever] = round(float(np.clip(lean, -1, 1)), 2)
        return out


def reads(pick, team_game=None):
    """The extra levers from real signals already on the row: the second market, outside
    projections, the crowd; plus pace / pass rate from the team table when present."""
    side = pick.get("side", "Over"); mkt = pick.get("cat") or pick.get("mkt"); line = pick.get("line")
    out = {}
    def flip(x): return None if x is None else (x if side in ("Over", "Yes") else -x)
    try:
        line = float(line)
    except Exception:
        line = None
    # Second Market: Underdog's line vs ours (a higher second-market number raises the stat read)
    ud = pick.get("ud_line")
    if ud is not None and line and mkt != "anytime_td":
        out["Second Market"] = flip(round(float(np.clip((float(ud) - line) / (0.06 * max(abs(line), 1)), -1, 1)), 2))
    elif ud is not None and mkt == "anytime_td" and pick.get("ud_over") is not None and pick.get("price") is not None:
        # a shorter Underdog TD price than the book's = the second market likes him more
        def imp(a):
            a = float(a); return 100 / (a + 100) if a > 0 else -a / (-a + 100)
        out["Second Market"] = round(float(np.clip((imp(pick["ud_over"]) - imp(pick["price"])) / 0.10, -1, 1)), 2)
    else:
        out["Second Market"] = None
    # Outside Projections: RotoWire / ESPN mean vs the line
    pj = [v for v in (pick.get("proj_rw"), pick.get("proj_espn")) if v is not None and not (isinstance(v, float) and v != v)]
    if pj and line and mkt != "anytime_td":
        m = float(np.mean(pj))
        out["Outside Projections"] = flip(round(float(np.clip((m - line) / (0.10 * max(abs(line), 1)), -1, 1)), 2))
    elif pj and mkt == "anytime_td":
        m = float(np.mean(pj))                                # expected TDs
        out["Outside Projections"] = round(float(np.clip((m - 0.35) / 0.35, -1, 1)), 2)
    else:
        out["Outside Projections"] = None
    # The Crowd: Sleeper trending adds / drops (from the intel signals)
    sig = pick.get("intel_signals") or {}
    crowd = sig.get("crowd") if isinstance(sig, dict) else None
    out["The Crowd"] = flip(round(float(crowd), 2)) if (crowd is not None and abs(float(crowd)) > 1e-9) else None
    # Pace & Pass Rate: the team's recent plays per game and pass rate vs league
    if team_game:
        pr = team_game.get("pass_rate_c"); pl = team_game.get("plays_c")
        if pr is not None and pl is not None:
            if mkt in ("rec_yds", "receptions", "pass_yds"):
                lean = float(np.clip(pr / 0.10 + pl / 12.0, -1, 1))
            elif mkt in ("rush_yds", "rush_att"):
                lean = float(np.clip(-pr / 0.10 + pl / 12.0, -1, 1))
            else:
                lean = float(np.clip(pl / 12.0, -1, 1))
            out["Pace & Pass Rate"] = flip(round(lean, 2))
    return out


def team_context(season=2026, weeks_played=0):
    """{team: {pass_rate_c, plays_c}} from the team-game table (last 4 games of the reference season)."""
    ref, _ = _season_ref(season, weeks_played)
    p = os.path.join(DATA, "universe_features_team_game.parquet")
    if not os.path.exists(p):
        return {}
    t = pd.read_parquet(p, columns=["team", "season", "week", "pass_rate", "plays"])
    t = t[t.season == ref].sort_values(["team", "week"])
    last = t.groupby("team").tail(4).groupby("team").agg(pass_rate=("pass_rate", "mean"), plays=("plays", "mean"))
    lg_pr = float(t.pass_rate.mean()); lg_pl = float(t.plays.mean())
    return {tm: {"pass_rate_c": round(float(r.pass_rate - lg_pr), 3), "plays_c": round(float(r.plays - lg_pl), 2)} for tm, r in last.iterrows()}


if __name__ == "__main__":
    P = Profiles(2026, 0)
    print(len(P.table), "players profiled (ref season", P.ref, ")")
    for nm, mkt, side, pos in (("Derrick Henry", "rush_yds", "Over", "RB"), ("Parker Washington", "rec_yds", "Over", "WR"), ("Baker Mayfield", "pass_yds", "Under", "QB")):
        print(nm, P.profile(nm, mkt, side, pos))
    print(list(team_context().items())[:3])
