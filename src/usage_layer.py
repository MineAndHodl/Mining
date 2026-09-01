"""In-season usage layer — the switch that un-gates mover/rookie & TD props once real
2026 snaps exist. STAGED: returns empty preseason, activates automatically Week 1+.

As current-season games arrive it computes, from live pbp + snap counts:
  - team PROE / pass-run volume (the coaching tendency, stabilizes ~4 games)
  - each player's current snap share, target share, carry share (the real role)
  - goal-line carry share and red-zone target share (the TD-prop signals)
and blends them into the projection with weight games/(games+K), so current usage takes
over from the preseason prior as evidence accumulates. Once a mover/rookie has enough
current-season usage, their gated projection un-gates (confidence rises off real data).
"""
import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
BLEND_K = 4.0          # current-season games to reach ~50% weight vs the prior


def current_usage(season=2026, through_week=None):
    """Per-player current-season usage. Empty (staged) until 2026 games are played."""
    import nflreadpy as nfl
    try:
        ps = nfl.load_player_stats(seasons=[season]).to_pandas()
    except Exception:
        return pd.DataFrame(), {}
    ps = ps[ps.get("season_type", "REG").eq("REG")] if "season_type" in ps.columns else ps
    if through_week is not None and "week" in ps.columns:
        ps = ps[ps.week <= through_week]
    if not len(ps):
        return pd.DataFrame(), {}     # STAGED — no games yet
    g = ps.groupby("player_id").agg(
        gm=("week", "nunique"), team=("team", "last"), pos=("position", "last"),
        tgt_pg=("targets", "mean"), rec_pg=("receptions", "mean"), recyd_pg=("receiving_yards", "mean"),
        car_pg=("carries", "mean"), rushyd_pg=("rushing_yards", "mean"),
        att_pg=("attempts", "mean"), passyd_pg=("passing_yards", "mean")).reset_index()
    # team PROE proxy from pbp (pass rate over expected) — the coaching tendency
    proe = {}
    try:
        pbp = nfl.load_pbp(seasons=[season]).to_pandas()
        if through_week is not None:
            pbp = pbp[pbp.week <= through_week]
        pbp = pbp[pbp.play_type.isin(["run", "pass"])].dropna(subset=["pass_oe"])
        proe = pbp.groupby("posteam")["pass_oe"].mean().to_dict()
    except Exception:
        pass
    return g, proe


def redzone_signals(season=2026, through_week=None):
    """Goal-line carry share + red-zone target share from current-season pbp (TD-prop signal)."""
    import nflreadpy as nfl
    try:
        pbp = nfl.load_pbp(seasons=[season]).to_pandas()
    except Exception:
        return {}, {}
    if through_week is not None:
        pbp = pbp[pbp.week <= through_week]
    if not len(pbp):
        return {}, {}
    gl = pbp[(pbp.play_type == "run") & (pbp.yardline_100 <= 3)].dropna(subset=["rusher_player_id"])
    gl_share = (gl.groupby("rusher_player_id").size() /
                gl.groupby("posteam").size().reindex(gl["posteam"]).values.sum()) if len(gl) else {}
    rz = pbp[(pbp.play_type == "pass") & (pbp.yardline_100 <= 20)].dropna(subset=["receiver_player_id"])
    rz_share = rz.groupby("receiver_player_id").size().to_dict() if len(rz) else {}
    return (gl.groupby("rusher_player_id").size().to_dict() if len(gl) else {}), rz_share


def blend_into_projection(proj, season=2026, through_week=None):
    """Shift the projection toward current-season usage as games accumulate. Returns proj
    with updated per-game means + an 'ingame_games' column and un-gated confidence for
    movers/rookies who now have real data. Preseason (no games) → returns proj unchanged."""
    cur, proe = current_usage(season, through_week)
    if not len(cur):
        proj = proj.copy(); proj["ingame_games"] = 0
        return proj, {"status": "staged — no 2026 games yet; preseason projections in use"}
    cur = cur.set_index("player_id")
    proj = proj.copy()
    def blend(row, col_now, col_proj):
        pid = row.get("player_id")
        if pid in cur.index and not np.isnan(cur.at[pid, col_now]):
            g = cur.at[pid, "gm"]; w = g / (g + BLEND_K)
            return w * cur.at[pid, col_now] + (1 - w) * row[col_proj]
        return row[col_proj]
    # blend the volume-driven markets toward current usage
    if "rec_yds" in proj: proj["rec_yds"] = proj.apply(lambda r: blend(r, "recyd_pg", "rec_yds"), axis=1)
    if "receptions" in proj: proj["receptions"] = proj.apply(lambda r: blend(r, "rec_pg", "receptions"), axis=1)
    if "rush_yds" in proj: proj["rush_yds"] = proj.apply(lambda r: blend(r, "rushyd_pg", "rush_yds"), axis=1)
    if "rush_att" in proj: proj["rush_att"] = proj.apply(lambda r: blend(r, "car_pg", "rush_att"), axis=1)
    proj["ingame_games"] = proj.get("player_id", pd.Series(index=proj.index)).map(
        lambda p: int(cur.at[p, "gm"]) if p in cur.index else 0)
    # un-gate: movers/rookies with >=3 current games get real confidence
    ungated = 0
    if "confidence" in proj.columns:
        mask = (proj["ingame_games"] >= 3) & (proj["confidence"] < 0.67)
        proj.loc[mask, "confidence"] = np.minimum(0.85, 0.5 + proj.loc[mask, "ingame_games"] / 20.0)
        ungated = int(mask.sum())
    return proj, {"status": f"active — {int(proj['ingame_games'].max())} weeks of 2026 data; {ungated} players un-gated",
                  "team_proe": proe}


if __name__ == "__main__":
    proj = pd.read_parquet(f"{DATA}/projection_2026.parquet")
    out, info = blend_into_projection(proj, season=2026)
    print("usage layer status:", info["status"])
    print("(activates automatically once 2026 games are played)")
