"""Enrichment layers — fantasy expected-usage (regression) + injury availability.

These answer "are we using fantasy / injury / news data?":
  fantasy   ff_opportunity gives EXPECTED stats from actual opportunity. A player far
            ABOVE expectation is a fade (lucky TDs/YAC), far BELOW is a buy (due). We
            fold expected-usage in as a stabilizer and surface the regression signal.
  injuries  official injury reports (Out/Doubtful/Questionable + practice status) →
            an availability multiplier. Reports post during game weeks, so this fires
            in-season (2026 reports drop Week 1); the hook is live and ready now.
  news      beat-writer/depth-chart text is fetched at RUN TIME by the scheduled job
            via WebSearch (not a cached dataset) and passed into the pick rationale.
"""
import os, sys
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

AVAIL = {"Out": 0.0, "Doubtful": 0.15, "Questionable": 0.85}   # game-status → play multiplier


def fantasy_regression(seasons=(2023, 2024, 2025)):
    """Per player: actual vs expected receiving/rushing yards from ff_opportunity.
    Returns a regression signal in [-1,1]: + = overperformed (fade), − = underperformed (buy)."""
    import nflreadpy as nfl
    ff = nfl.load_ff_opportunity(seasons=list(seasons)).to_pandas()
    ff = ff[ff.get("season_type", "REG").eq("REG")] if "season_type" in ff.columns else ff
    g = ff.groupby("player_id").agg(
        rec_act=("rec_yards_gained", "sum"), rec_exp=("rec_yards_gained_exp", "sum"),
        rush_act=("rush_yards_gained", "sum"), rush_exp=("rush_yards_gained_exp", "sum"),
        rec_td=("rec_touchdown", "sum"), rec_td_exp=("rec_touchdown_exp", "sum")).reset_index()
    # normalized over/under-performance
    g["yds_oe"] = ((g.rec_act + g.rush_act) - (g.rec_exp + g.rush_exp)) / \
                  (g.rec_exp + g.rush_exp).clip(lower=20)
    g["td_luck"] = (g.rec_td - g.rec_td_exp)              # positive = lucky TDs (regress down)
    g["regression_signal"] = g["yds_oe"].clip(-0.5, 0.5) * 2   # → roughly [-1,1]
    return g[["player_id", "yds_oe", "td_luck", "regression_signal"]]


def injury_availability(season, week=None):
    """Current injury report → per-player availability multiplier. Empty preseason;
    populated once weekly reports drop. Also returns who's OUT (drop from board)."""
    import nflreadpy as nfl
    try:
        inj = nfl.load_injuries(seasons=[season]).to_pandas()
    except Exception:
        return pd.DataFrame(columns=["gsis_id", "avail", "status", "injury"]), []
    if week is not None and "week" in inj.columns:
        inj = inj[inj["week"] == week]
    if not len(inj):
        return pd.DataFrame(columns=["gsis_id", "avail", "status", "injury"]), []
    inj = inj.sort_values("week").groupby("gsis_id").tail(1)
    inj["avail"] = inj["report_status"].map(AVAIL).fillna(1.0)
    inj["injury"] = inj.get("report_primary_injury", "")
    out_ids = inj.loc[inj["avail"] == 0.0, "gsis_id"].tolist()
    return inj[["gsis_id", "avail", "report_status", "injury"]].rename(
        columns={"report_status": "status"}), out_ids


def apply_enrichment(proj, season, week=None, td_regress=0.15):
    """Fold enrichment into a projection frame: nudge means toward expectation
    (regression), zero out injured-OUT players, attach availability + signals."""
    reg = fantasy_regression()
    proj = proj.merge(reg, left_on="player_id", right_on="player_id", how="left") \
        if "player_id" in proj.columns else proj.assign(regression_signal=np.nan)
    # pull means toward expectation: overperformers shaded down, underperformers up
    if "regression_signal" in proj.columns:
        adj = 1 - td_regress * proj["regression_signal"].fillna(0)
        for c in ("rec_yds", "rush_yds", "receptions"):
            if c in proj.columns:
                proj[c] = proj[c] * adj
    av, out_ids = injury_availability(season, week)
    if len(av):
        proj = proj.merge(av, left_on="player_id", right_on="gsis_id", how="left")
        proj["avail"] = proj["avail"].fillna(1.0)
        for c in ("rec_yds", "rush_yds", "receptions"):
            if c in proj.columns:
                proj[c] = proj[c] * proj["avail"]
    else:
        proj["avail"] = 1.0
    return proj, out_ids


if __name__ == "__main__":
    import pandas as pd
    reg = fantasy_regression()
    print("fantasy regression sample (biggest overperformers = fade):")
    print(reg.sort_values("regression_signal", ascending=False).head(5).to_string(index=False))
    print("\nbiggest underperformers (buy-low / due):")
    print(reg.sort_values("regression_signal").head(5).to_string(index=False))
    av, out_ids = injury_availability(2025, week=1)
    print(f"\ninjury report 2025 wk1: {len(av)} listed, {len(out_ids)} OUT")
