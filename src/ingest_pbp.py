"""Ingest 5 seasons of play-by-play (2021-2025) — the substrate every model learns from.

Produces:
  data/pbp_2021_2025.parquet          run/pass plays with the fields the models AND the EARN
                                      backlog need (zone, type, ids, TD flags, epa, yards,
                                      air yards, YAC, score state, WP, clock, drive)
  data/prop_player_games_5y.parquet   per player-game stat rows (pid, game_id, season, week,
                                      posteam, y, mkt) for rec_yds / rush_yds / pass_yds /
                                      receptions — same schema as the 3-season table, so
                                      every model can be re-fit on 3 vs 5 seasons and compared.
The 3-season files are left untouched until a walk-forward comparison says the deeper history
wins (THE_DESK_PROMPT §2.3: more data is a hypothesis).
"""
import os, sys, time
import pandas as pd
import nflreadpy as nfl

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
SEASONS = [2021, 2022, 2023, 2024, 2025]
PBP_COLS = ["season", "week", "game_id", "posteam", "defteam", "yardline_100", "goal_to_go", "play_type",
            "rush_attempt", "pass_attempt", "rusher_player_id", "receiver_player_id", "passer_player_id",
            "touchdown", "rush_touchdown", "pass_touchdown", "td_player_id", "two_point_attempt", "qb_scramble",
            "down", "ydstogo", "score_differential", "wp", "half_seconds_remaining", "game_seconds_remaining", "qtr",
            "drive", "fixed_drive_result", "epa", "yards_gained", "air_yards", "yards_after_catch", "complete_pass",
            "shotgun", "no_huddle", "xpass", "pass_oe", "season_type", "home_team", "away_team", "roof", "surface"]


def ingest_pbp():
    frames = []
    for s in SEASONS:
        t = time.time()
        d = nfl.load_pbp(seasons=[s]).to_pandas()
        keep = [c for c in PBP_COLS if c in d.columns]
        d = d[keep]
        d = d[d["play_type"].isin(["run", "pass"])]
        frames.append(d); print(f"  {s}: {len(d)} run/pass plays ({time.time()-t:.1f}s)")
    pbp = pd.concat(frames, ignore_index=True)
    pbp.to_parquet(f"{DATA}/pbp_2021_2025.parquet", index=False)
    print("pbp_2021_2025:", len(pbp), "rows,", len(pbp.columns), "cols")
    return pbp


def ingest_player_games():
    sched = pd.read_parquet(f"{DATA}/schedules.parquet")
    gid = {}
    for _, r in sched.iterrows():
        gid[(int(r.season), int(r.week), r.home_team)] = r.game_id; gid[(int(r.season), int(r.week), r.away_team)] = r.game_id
    rows = []
    for s in SEASONS:
        ps = nfl.load_player_stats(seasons=[s]).to_pandas()
        ps = ps[ps["season_type"] == "REG"] if "season_type" in ps.columns else ps
        team_col = "team" if "team" in ps.columns else ("recent_team" if "recent_team" in ps.columns else None)
        pid_col = "player_id" if "player_id" in ps.columns else "gsis_id"
        for _, r in ps.iterrows():
            tm = r[team_col] if team_col else None
            g = gid.get((int(r.season), int(r.week), tm))
            if g is None: continue
            for mkt, col in (("rec_yds", "receiving_yards"), ("rush_yds", "rushing_yards"),
                             ("pass_yds", "passing_yards"), ("receptions", "receptions")):
                v = r.get(col)
                if v is None or pd.isna(v): continue
                # only players who actually participate in that market (targets / carries / attempts)
                part = {"rec_yds": r.get("targets", 0), "receptions": r.get("targets", 0),
                        "rush_yds": r.get("carries", 0), "pass_yds": r.get("attempts", 0)}[mkt]
                if not part or part <= 0: continue
                rows.append({"pid": r[pid_col], "game_id": g, "season": int(r.season), "week": int(r.week),
                             "posteam": tm, "y": float(v), "mkt": mkt})
        print(f"  {s}: player-game rows so far {len(rows)}")
    pg = pd.DataFrame(rows)
    pg.to_parquet(f"{DATA}/prop_player_games_5y.parquet", index=False)
    print("prop_player_games_5y:", len(pg), "rows")
    return pg


if __name__ == "__main__":
    print("== play-by-play =="); ingest_pbp()
    print("== player-games =="); ingest_player_games()
