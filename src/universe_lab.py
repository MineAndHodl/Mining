"""Universe lab — every new signal from the data universe, tested the repo's way (feature_lab:
ridge residual model, two held-out seasons, adopt only if it wins both).

Candidates are built CAUSALLY from a player's prior games (rolling last-3 mean, shifted) or
the opponent's prior games this season, centered so a missing value is neutral (0):

  role       route_rate, snap_pct (offense %), target_share, air_yards_share, wopr, tprr,
             gl_snap_share, rz_snap_share, carries share
  efficiency ngs separation / yac over expected / rush yards over expected / cpoe / time to
             throw / aggressiveness; pfr yards after contact, broken tackles, drops, pressure %
  charting   ftn catchable / contested / drop rates on his targets; team play-action, motion, RPO
  opponent   man-coverage rate, pressure rate, blitz rate, box count, time to throw allowed

    python3 src/universe_lab.py                # all candidates -> data/universe_lab.json
    python3 src/universe_lab.py tprr_l3 opp_man_rate
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multifactor as mf
import feature_lab as fl

DATA = mf.DATA
PLAYER_FEATS = ["route_rate", "offense_pct", "target_share", "air_yards_share", "wopr", "tprr", "gl_snap_share", "rz_snap_share", "in10_snap_share",
                "p11_share", "ngs_sep", "ngs_cushion", "ngs_iay_share", "ngs_yac_oe", "ngs_adot", "ngs_ryoe_att", "ngs_box8_pct", "ngs_efficiency",
                "ngs_ttt", "ngs_cpoe", "ngs_aggr", "ngs_ayd", "pfr_ybc_att", "pfr_yac_att", "pfr_bt_rush", "pfr_bt_rec", "pfr_drops",
                "pfr_pressured_pct", "pfr_bad_throw_pct", "ftn_catchable", "ftn_contested", "ftn_drop", "ftn_screen_share", "epa_per_target",
                "epa_per_carry", "adot", "racr", "carries", "tgt", "routes"]
TEAM_FEATS = ["pa_rate", "motion_rate", "rpo_rate", "screen_rate", "no_huddle_rate", "shotgun_rate", "pass_rate", "plays", "dropbacks", "rz_trips",
              "off_pressure_allowed", "off_ttt"]
OPP_FEATS = ["def_man_rate", "def_pressure_rate", "def_blitz_rate", "def_rushers_avg", "def_box_avg", "def_ttt_allowed"]


def _lagged():
    """Causal per-player (last-3 prior games) and per-team/opponent (prior games this season) tables."""
    u = pd.read_parquet(f"{DATA}/universe_features_player_game.parquet")
    u = u.sort_values(["pid", "season", "week"]).reset_index(drop=True)
    g = u.groupby("pid")
    lag = u[["pid", "game_id", "season", "week"]].copy()
    for f in PLAYER_FEATS:
        if f not in u.columns: continue
        lag[f"{f}_l3"] = g[f].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        lag[f"{f}_d3"] = lag[f"{f}_l3"] - g[f].transform(lambda s: s.shift(1).expanding(min_periods=3).mean())
    t = pd.read_parquet(f"{DATA}/universe_features_team_game.parquet").sort_values(["team", "season", "week"]).reset_index(drop=True)
    gt = t.groupby(["team", "season"])
    tl = t[["team", "game_id", "season", "week"]].copy()
    for f in TEAM_FEATS + OPP_FEATS:
        if f not in t.columns: continue
        tl[f"{f}_l4"] = gt[f].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    return lag, tl


def base_table():
    d = mf.build_features()
    lag, tl = _lagged()
    d = d.merge(lag.drop(columns=["season", "week"]), on=["pid", "game_id"], how="left")
    # own team's prior tendencies and the opponent's
    own = tl.rename(columns={c: f"own_{c}" for c in tl.columns if c.endswith("_l4")}).rename(columns={"team": "posteam"}).drop(columns=["season", "week"])
    d = d.merge(own, on=["posteam", "game_id"], how="left")
    opp = tl.rename(columns={c: f"opp_{c}" for c in tl.columns if c.endswith("_l4")}).rename(columns={"team": "opp"}).drop(columns=["season", "week"])
    d = d.merge(opp, on=["opp", "game_id"], how="left")
    return d


def _center(col):
    def fn(pg):
        x = pg[col].astype(float)
        med = x.groupby(pg["mkt"]).transform("median")
        pg[col + "_c"] = (x - med).fillna(0.0)
        return pg
    return fn


CANDIDATES = {}
for f in PLAYER_FEATS:
    CANDIDATES[f"{f}_l3"] = _center(f"{f}_l3"); CANDIDATES[f"{f}_d3"] = _center(f"{f}_d3")
for f in TEAM_FEATS:
    CANDIDATES[f"own_{f}_l4"] = _center(f"own_{f}_l4")
for f in OPP_FEATS:
    CANDIDATES[f"opp_{f}_l4"] = _center(f"opp_{f}_l4")


def run(names=None, seasons=(2024, 2025)):
    d = base_table()
    names = names or [n for n in CANDIDATES if n in d.columns]
    p = f"{DATA}/universe_lab.json"
    log = json.load(open(p)) if os.path.exists(p) else {}
    rows = []
    for nm in names:
        if nm not in d.columns:
            continue
        fn = CANDIDATES[nm]
        # feature_lab tests a column named `name`; our fn writes name+"_c"
        def wrapped(pg, fn=fn, nm=nm):
            pg = fn(pg); pg[nm] = pg[nm + "_c"]; return pg
        try:
            r = fl.test_feature(nm, wrapped, d.drop(columns=[nm]).assign(**{nm: d[nm]}), seasons=seasons)
        except Exception as e:
            print(f"  {nm}: failed {e}"); continue
        log[nm] = {**r, "by_season": {str(k): v for k, v in r["by_season"].items()}}
        cells = []
        for te in seasons:
            for mkt, v in r["by_season"][te].items():
                cells.append((te, mkt, v["d_hit"], v["d_mae"], v["weight"], v["n"]))
        best = max(cells, key=lambda c: c[2]) if cells else None
        rows.append((nm, r["verdict"], r["wins"], r["cells"], r["hurt_both_seasons"], best))
        print(f"  {nm:28s} {r['verdict']:5s} wins {r['wins']}/{r['cells']} hurt-both {r['hurt_both_seasons']}  best cell: {best[0]} {best[1]} d_hit {best[2]:+.3f} d_mae {best[3]:+.2f} w={best[4]}" if best else f"  {nm}: no cells")
    json.dump(log, open(p, "w"), indent=1)
    return rows


if __name__ == "__main__":
    names = sys.argv[1:] or None
    rows = run(names)
    wins = [r for r in rows if r[1] == "WINS"]
    print(f"\n{len(rows)} candidates tested; WINS: {[r[0] for r in wins]}")
