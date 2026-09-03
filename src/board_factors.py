"""Attach a fully-populated factor scorecard to every pick on The Board.

This is the piece that makes each factor slider carry a REAL lean instead of a wall of
zeros. For each pick it assembles the context the scorecard needs from local data:
  - proj_row      : the player's projection (usage, own-vs-role form, hist sample)
  - script        : team spread + implied points from the 2026 schedule (Game Flow)
  - weather       : roof / wind / temp from the schedule (Weather)
  - matchup       : opponent defense vs position, relative to league (Opponent Defense)
  - adp           : FantasyFootballCalculator draft position (Market Read)
then computes the scorecard and writes it back onto the pick.

Run standalone to re-enrich data/ranked_board.json in place and rebuild the page — cheap,
uses no Odds API credits (all inputs are local).
"""
import os, sys, json, re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import factor_scorecard as fsc
from names import norm_name as _norm_name

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

# projection team codes -> schedule team codes (nflverse abbrevs are consistent, a few aliases)
TEAM_FIX = {"LAR": "LA", "OAK": "LV", "SD": "LAC", "STL": "LA", "WSH": "WAS"}


def _load_context():
    proj = pd.read_parquet(f"{DATA}/projection_2026.parquet")
    proj["_key"] = proj["name"].map(_norm_name)
    proj_map = {r["_key"]: r for r in proj.to_dict("records")}

    sched = pd.read_parquet(f"{DATA}/schedules.parquet")
    sched = sched[sched.season == 2026].copy()
    # map by "Away @ Home" full names AND by team codes
    try:
        import nflreadpy as nfl
        teams = nfl.load_teams().to_pandas()
        code2full = dict(zip(teams["team_abbr"], teams["team_name"]))
    except Exception:
        code2full = {}
    sched_rows = sched.to_dict("records")

    adp = {}
    try:
        for p in json.load(open(f"{DATA}/ffc_adp_2026.json"))["players"]:
            adp[_norm_name(p["name"])] = float(p["adp"])
    except Exception:
        pass

    # opponent defense vs position (latest season), relative to league (vs_lg is ratio)
    def_vs = {}
    for pos, f in (("RB", "def_vs_rb.parquet"), ("WR", "def_vs_wr.parquet")):
        try:
            d = pd.read_parquet(f"{DATA}/{f}")
            d = d[d.season == d.season.max()]
            def_vs[pos] = {row["defteam"]: float(row["vs_lg"]) - 1.0 for _, row in d.iterrows()}
        except Exception:
            def_vs[pos] = {}
    return proj_map, sched_rows, code2full, adp, def_vs


def _game_ctx(game, team_code, sched_rows, code2full):
    """Resolve a pick's game -> (team_spread for team_code, implied points, weather dict)."""
    if not game or "@" not in game:
        return None, None
    away_full, home_full = [s.strip() for s in game.split("@")]
    for r in sched_rows:
        a, h = r.get("away_team"), r.get("home_team")
        af, hf = code2full.get(a, a), code2full.get(h, h)
        if (af in away_full or away_full.endswith(str(a))) and (hf in home_full or home_full.endswith(str(h))):
            spread = float(r.get("spread_line") or 0)   # nflverse: POSITIVE = home favored
            total = float(r.get("total_line") or 44)
            # team_spread in the dog-positive convention the scorecard expects:
            # favorite = negative, underdog = positive. Home is favored when spread_line>0,
            # so the home team's spread is -spread_line and the away team's is +spread_line.
            tc = TEAM_FIX.get(team_code, team_code)
            if tc == h:
                team_spread = -spread
            elif tc == a:
                team_spread = spread
            else:
                team_spread = 0.0
            implied = total / 2 - team_spread / 2
            wx = {"roof": r.get("roof"),
                  "wind": (None if pd.isna(r.get("wind")) else float(r.get("wind"))),
                  "temp": (None if pd.isna(r.get("temp")) else float(r.get("temp")))}
            if wx["wind"] is None or wx["temp"] is None:            # pre-game: the kickoff forecast
                try:
                    import weather_feed as _wf
                    f = _wf.load().get(game) or {}
                    if f and not f.get("error"):
                        wx = {"roof": f.get("roof") or wx["roof"], "wind": f.get("wind"), "temp": f.get("temp"), "forecast": True}
                except Exception:
                    pass
            opp = a if tc == h else (h if tc == a else None)
            return {"team_spread": team_spread, "implied_team_pts": round(implied, 1),
                    "opp": opp, "weather": wx}, opp
    return None, None


_TD_STATS = None
def _load_td_stats():
    """Per-player rows from the DECOMPOSED xTD table (xtd_2026.parquet): goal-line share,
    red-zone target share, vulture leakage, coach goal-line pass rate, conversion vs position,
    last-season TDs vs expected. Position-level conversion priors attached for 'Finishing'."""
    global _TD_STATS
    if _TD_STATS is None:
        try:
            t = pd.read_parquet(f"{DATA}/xtd_2026.parquet")
            backs = t[t.position.isin(["RB", "FB"])]
            catchers = t[t.position.isin(["WR", "TE"])]
            pri_rush = float(backs["conv_rush_gl"].median()) if len(backs) else None
            pri_tgt = float(catchers["conv_tgt_rz"].median()) if len(catchers) else None
            _TD_STATS = {}
            for r in t.to_dict("records"):
                k = _norm_name(r.get("name", ""))
                row = {c: (None if (r.get(c) is None or (isinstance(r.get(c), float) and pd.isna(r.get(c)))) else r.get(c))
                       for c in ("position", "gl_share", "rz_tgt_share", "vulture_share", "coach_gl_pass_rate",
                                 "conv_rush_gl", "conv_tgt_rz", "tds_last", "xtd", "games")}
                row["pos_conv_rush_gl"] = pri_rush; row["pos_conv_tgt_rz"] = pri_tgt
                # a back's full inside-the-20 carry role (volume), distinct from the goal-line plunge
                zs = [r.get(f"{z}_rush_share") for z in ("GL", "IN10", "RZ")]
                zs = [float(v) for v in zs if v is not None and not (isinstance(v, float) and pd.isna(v))]
                row["rz20_rush_share"] = round(sum(zs) / len(zs), 3) if zs else None
                _TD_STATS[k] = row
        except Exception:
            _TD_STATS = {}
    return _TD_STATS


def attach_to_pick(pick, ctx):
    """Compute + attach a scorecard to a single prop pick dict (mutates + returns it).
    Anytime-TD picks get a TD-SPECIFIC scale (red-zone role, finishing, scoring env, regression),
    not the generic yardage factors — the levers have to fit the market."""
    proj_map, sched_rows, code2full, adp, def_vs = ctx
    key = _norm_name(pick.get("name", ""))
    pr = proj_map.get(key, {})
    team_code = TEAM_FIX.get(pr.get("team"), pr.get("team"))
    mkt = pick.get("cat", pick.get("mkt", ""))
    g, opp = _game_ctx(pick.get("game"), team_code, sched_rows, code2full)
    script = {"team_spread": g["team_spread"], "implied_team_pts": g["implied_team_pts"]} if g else None

    if mkt == "anytime_td":
        td = _load_td_stats().get(key)
        pick["scorecard"] = fsc.td_scorecard(pick, td=td, script=script, adp=adp.get(key))
        return pick

    p2 = {"side": pick.get("side", "Over"), "mkt": mkt,
          "line": pick.get("line", 0), "our_num": pick.get("our_num"), "_health": pick.get("_health"),
          "inj_lean": pick.get("inj_lean"), "intel_lean": pick.get("intel_lean")}
    weather = g["weather"] if g else None
    matchup = None
    if g:
        pos = pr.get("position")
        if opp and pos in ("RB",) and mkt in fsc.RUSH_MKTS:
            matchup = def_vs.get("RB", {}).get(opp)
        elif opp and pos in ("WR", "TE") and mkt in fsc.PASS_MKTS:
            matchup = def_vs.get("WR", {}).get(opp)
        elif opp and pos == "QB" and mkt in ("pass_yds", "pass_att", "pass_cmp"):
            # a QB's opponent pass-defense read: WR-receiving-yards-allowed vs league is our
            # available proxy for how leaky a secondary is (weighted down in the scorecard, since
            # matchup is our least-predictive lever).
            matchup = def_vs.get("WR", {}).get(opp)
    sc = fsc.scorecard(p2, proj_row=pr, script=script, weather=weather, matchup=matchup,
                       adp=adp.get(key), contrib=pick.get("contrib"))
    pick["scorecard"] = sc
    return pick


def attach_all(ranked, ctx=None):
    ctx = ctx or _load_context()
    for cat, lst in ranked.get("by_category", {}).items():
        for pk in lst:
            attach_to_pick(pk, ctx)
    for pk in ranked.get("best_props", []):
        attach_to_pick(pk, ctx)
    for pk in ranked.get("recommended", []) or []:
        if not pk.get("scorecard"): attach_to_pick(pk, ctx)
    return ranked


def enrich_file():
    path = os.path.join(DATA, "ranked_board.json")
    ranked = json.load(open(path))
    attach_all(ranked)
    json.dump(ranked, open(path, "w"), indent=1)
    # report coverage
    fired = {}; tot = 0
    for pk in ranked.get("best_props", []):
        tot += 1
        for f, v in (pk.get("scorecard") or {}).items():
            if v not in (None, 0.0):
                fired[f] = fired.get(f, 0) + 1
    print(f"enriched {tot} best_props; factor fire-rate:")
    for f in fsc.FACTORS:
        print(f"  {f:18s} {fired.get(f,0)}/{tot}")
    return ranked


if __name__ == "__main__":
    enrich_file()
