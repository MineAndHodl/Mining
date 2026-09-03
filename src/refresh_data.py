"""Weekly in-season data refresh — the switch that turns the model from "last season's
prior" into a live one. Without this, NOTHING validated in-season ever activates: the recency
edge never sees a 2026 game, the offseason rail never lifts, the market-role baseline never
reverts, xTD never enters its (validated, +2-3% Brier) in-season mode, and the grader has no
stats to grade against. Run it at the top of every Tuesday / Sunday / Monday job:

    python3 src/refresh_data.py            # season inferred from the calendar

Idempotent: current-season rows are replaced, prior seasons untouched. Before Week 1 the
current season has no games and every step reports "nothing yet" — that is fine.
"""
import os, sys, datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PBP = f"{DATA}/pbp_2021_2025.parquet"            # name kept for every reader; holds ALL seasons through the current one
PG = f"{DATA}/prop_player_games.parquet"
SCHED = f"{DATA}/schedules.parquet"


def current_season(today=None):
    today = today or datetime.date.today()
    return today.year if today.month >= 3 else today.year - 1


RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"
FALLBACK_URLS = {   # nflreadpy refuses a season its calendar hasn't flipped to yet; the release files exist anyway
    "load_pbp": ["{R}/pbp/play_by_play_{s}.parquet"],
    "load_player_stats": ["{R}/stats_player/stats_player_week_{s}.parquet"],
    "load_schedules": ["{R}/schedules/games.csv"],
}


def _load(fn, *a, **k):
    import nflreadpy as nfl
    try:
        d = getattr(nfl, fn)(*a, **k)
        return d.to_pandas() if hasattr(d, "to_pandas") else d
    except Exception as e:
        season = (k.get("seasons") or a[0])[0]
        for u in FALLBACK_URLS.get(fn, []):
            url = u.format(R=RELEASE, s=season)
            try:
                d = pd.read_csv(url, low_memory=False) if url.endswith(".csv") else pd.read_parquet(url)
                if "season" in d.columns: d = d[d.season == season]
                if len(d): return d
            except Exception:
                continue
        raise e


def refresh_pbp(season):
    from ingest_pbp import PBP_COLS
    try:
        d = _load("load_pbp", seasons=[season])
    except Exception as e:
        print(f"pbp {season}: nothing yet ({str(e)[:80]})"); return 0
    if d is None or not len(d):
        print(f"pbp {season}: nothing yet"); return 0
    d = d[[c for c in PBP_COLS if c in d.columns]]
    d = d[d["play_type"].isin(["run", "pass"])]
    base = pd.read_parquet(PBP); base = base[base.season != season]
    out = pd.concat([base, d], ignore_index=True)
    out.to_parquet(PBP, index=False)
    print(f"pbp {season}: {len(d)} run/pass plays through week {int(d.week.max())} (table {len(out)} rows)")
    return int(d.week.max())


def refresh_player_games(season):
    sched = pd.read_parquet(SCHED)
    gid = {}
    for _, r in sched[sched.season == season].iterrows():
        gid[(int(r.week), r.home_team)] = r.game_id; gid[(int(r.week), r.away_team)] = r.game_id
    try:
        ps = _load("load_player_stats", seasons=[season])
    except Exception as e:
        print(f"player stats {season}: nothing yet ({str(e)[:80]})"); return 0
    if ps is None or not len(ps):
        print(f"player stats {season}: nothing yet"); return 0
    ps = ps[ps["season_type"] == "REG"] if "season_type" in ps.columns else ps
    team_col = "team" if "team" in ps.columns else "recent_team"
    pid_col = "player_id" if "player_id" in ps.columns else "gsis_id"
    rows = []
    for _, r in ps.iterrows():
        g = gid.get((int(r.week), r[team_col]))
        if g is None: continue
        for mkt, col, part in (("rec_yds", "receiving_yards", "targets"), ("rush_yds", "rushing_yards", "carries"),
                               ("pass_yds", "passing_yards", "attempts"), ("receptions", "receptions", "targets")):
            v = r.get(col)
            if v is None or pd.isna(v) or not r.get(part) or r.get(part) <= 0: continue
            rows.append({"pid": r[pid_col], "game_id": g, "season": season, "week": int(r.week), "posteam": r[team_col], "y": float(v), "mkt": mkt})
    if not rows:
        print(f"player stats {season}: nothing yet"); return 0
    new = pd.DataFrame(rows)
    base = pd.read_parquet(PG); base = base[base.season != season]
    out = pd.concat([base, new], ignore_index=True)
    out.to_parquet(PG, index=False)
    print(f"player games {season}: {len(new)} rows through week {int(new.week.max())} (table {len(out)} rows)")
    return int(new.week.max())


def refresh_schedules(season):
    try:
        s = _load("load_schedules", seasons=[season])
    except Exception as e:
        print(f"schedules {season}: not refreshed ({str(e)[:80]})"); return
    if s is None or not len(s): return
    base = pd.read_parquet(SCHED); base = base[base.season != season]
    cols = [c for c in base.columns if c in s.columns]
    out = pd.concat([base, s[cols]], ignore_index=True)
    out.to_parquet(SCHED, index=False)
    done = s.home_score.notna().sum() if "home_score" in s.columns else 0
    print(f"schedules {season}: {len(s)} games, {done} with scores")


def season_state():
    """(season, weeks_played) from the player-game table — the mode switch every run reads."""
    season = current_season()
    pg = pd.read_parquet(PG, columns=["season", "week"])
    cur = pg[pg.season == season]
    return season, (int(cur.week.max()) if len(cur) else 0)


def refresh(season=None):
    season = season or current_season()
    print(f"refreshing {season} ...")
    refresh_schedules(season)
    w1 = refresh_pbp(season)
    w2 = refresh_player_games(season)
    # the data universe for the current season (participation, snap counts, pbp) -> the ROLE
    # features (route participation, snap-share trend, goal-line inclusion) the model earned
    try:
        refresh_universe(season)
    except Exception as e:
        print("universe refresh skipped:", e)
    # the xTD table is a cache of last-season shares; in-season it is rebuilt per week by td_board
    print(f"season state: {season}, weeks played {max(w1, w2)}")
    return season, max(w1, w2)


def refresh_universe(season):
    """Pull this season's participation / snap counts / pbp / players from the universe sources and
    rebuild the derived player-game + team-game feature rows for this season (others kept)."""
    import ingest_universe as iu, universe_features as uf
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # per-season fetch (skips fresh files; a 404 means the season is not posted yet)
    for name in ("participation", "snap_counts", "pbp"):
        try:
            spec = iu.SOURCES[name]; url = spec["url"].format(s=season); dest = iu._dest(name, season)
            if not iu._fresh(dest, spec["fresh_hours"]):
                r = iu.requests.get(url, headers=iu.UA, timeout=120)
                if r.status_code == 200:
                    open(dest, "wb").write(r.content); print(f"  universe {name} {season}: refreshed")
        except Exception as e:
            print(f"  universe {name} {season}: {e}")
    if not os.path.exists(os.path.join(iu.UNI, f"pbp_{season}.parquet")) or not os.path.exists(os.path.join(iu.UNI, f"participation_{season}.parquet")):
        print(f"  universe: no {season} participation/pbp yet — role features stay on last season")
        return False
    pg = uf.build_player_game([season], verbose=False); tg = uf.build_team_game([season], verbose=False)
    for fn, new in (("universe_features_player_game.parquet", pg), ("universe_features_team_game.parquet", tg)):
        path = os.path.join(DATA, fn)
        cur = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
        cur = cur[cur.season != season] if len(cur) else cur
        pd.concat([cur, new], ignore_index=True).to_parquet(path, index=False)
    print(f"  universe features rebuilt for {season}: {len(pg):,} player-games")
    return True


if __name__ == "__main__":
    refresh(int(sys.argv[1]) if len(sys.argv) > 1 else None)
