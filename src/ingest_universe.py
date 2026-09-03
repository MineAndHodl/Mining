"""The data universe — every free, reachable NFL dataset, pulled into data/universe/ with one
inventory (data/data_ledger.json) that says exactly what we hold: rows, columns, seasons, size,
when it was refreshed, and what it can feed. The inventory is what the Model Room counts.

    python3 src/ingest_universe.py                 # refresh everything (idempotent, cached)
    python3 src/ingest_universe.py ngs snap_counts # a subset
    python3 src/ingest_universe.py --inventory     # rebuild the ledger only

Sources (all free, all reachable from the session; see DATA_UNIVERSE.md for the full catalog):
  nflverse releases  — play-by-play, player/team weekly stats, Next Gen Stats (tracking-derived),
                       PFR advanced stats, snap counts, participation (personnel + players on
                       the field per play), FTN charting (motion, play action, RPO, screens,
                       pressure, contested/catchable/drops), depth charts, weekly rosters,
                       injuries, officials, contracts, draft picks, combine, ESPN QBR, players,
                       schedules (with lines, refs, stadium, roof)
  Sleeper API        — 11k players (ids across every platform), trending adds/drops, weekly and
                       season projections, weekly stats, schedule
  ESPN               — core API (injuries, depth charts, event odds), fantasy projections, news
  Underdog Fantasy   — public pick'em lines (a vig-free median for hundreds of player props)
  Action Network     — public scoreboard odds (multi-book, sometimes public betting splits)
  RSS                — ESPN, Yahoo, CBS, ProFootballTalk, Google News (per-player queries)
Big files are cached locally (data/universe is gitignored); only derived tables that a live
feature uses get committed.
"""
import os, sys, json, time, glob, datetime, io
import requests
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
UNI = os.path.join(DATA, "universe")
R = "https://github.com/nflverse/nflverse-data/releases/download"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) TheDesk/1.0"}
THIS_SEASON = 2026
SEASONS_5 = list(range(2021, 2026))

# name -> spec. url may contain {s} for per-season assets. fresh_hours: re-download after this
# many hours when the season is the current one (or the asset is a moving snapshot).
SOURCES = {
    # ---- play-level -----------------------------------------------------------------------
    "pbp":            {"url": f"{R}/pbp/play_by_play_{{s}}.parquet", "seasons": SEASONS_5 + [THIS_SEASON], "fresh_hours": 12,
                       "feeds": "everything: EPA, zones, roles, TDs, script, pace", "family": "nflverse"},
    "participation":  {"url": f"{R}/pbp_participation/pbp_participation_{{s}}.parquet", "seasons": list(range(2016, 2026)), "fresh_hours": 24,
                       "feeds": "personnel, players on the field, defenders in box, pressure, time to throw → goal-line packages, routes run", "family": "nflverse"},
    "ftn_charting":   {"url": f"{R}/ftn_charting/ftn_charting_{{s}}.parquet", "seasons": list(range(2022, 2026)) + [THIS_SEASON], "fresh_hours": 24,
                       "feeds": "motion, play action, RPO, screens, blitzers, pass rushers, catchable/contested/drops, QB sneaks", "family": "nflverse"},
    # ---- player / team weekly ---------------------------------------------------------------
    "stats_player":   {"url": f"{R}/stats_player/stats_player_week_{{s}}.parquet", "seasons": list(range(2016, 2026)) + [THIS_SEASON], "fresh_hours": 12,
                       "feeds": "every box-score stat per player-week (targets, air yards, YAC, carries, fantasy points)", "family": "nflverse"},
    "stats_team":     {"url": f"{R}/stats_team/stats_team_week_{{s}}.parquet", "seasons": list(range(2016, 2026)) + [THIS_SEASON], "fresh_hours": 12,
                       "feeds": "team weekly totals: plays, pass/rush splits, red-zone, turnovers", "family": "nflverse"},
    "ngs_passing":    {"url": f"{R}/nextgen_stats/ngs_passing.parquet", "seasons": None, "fresh_hours": 24,
                       "feeds": "tracking-derived: time to throw, air yards, aggressiveness, CPOE, expected completion", "family": "nflverse/NGS"},
    "ngs_receiving":  {"url": f"{R}/nextgen_stats/ngs_receiving.parquet", "seasons": None, "fresh_hours": 24,
                       "feeds": "tracking-derived: separation, cushion, intended air yards share, YAC over expected, catch %", "family": "nflverse/NGS"},
    "ngs_rushing":    {"url": f"{R}/nextgen_stats/ngs_rushing.parquet", "seasons": None, "fresh_hours": 24,
                       "feeds": "tracking-derived: rush yards over expected, efficiency, time to line, 8+ in the box rate", "family": "nflverse/NGS"},
    "pfr_pass":       {"url": f"{R}/pfr_advstats/advstats_week_pass_{{s}}.parquet", "seasons": list(range(2018, 2026)) + [THIS_SEASON], "fresh_hours": 24,
                       "feeds": "pressures, blitzes, hurries, drops by receivers, bad throws, on-target %, pocket time, RPO/play-action splits", "family": "nflverse/PFR"},
    "pfr_rush":       {"url": f"{R}/pfr_advstats/advstats_week_rush_{{s}}.parquet", "seasons": list(range(2018, 2026)) + [THIS_SEASON], "fresh_hours": 24,
                       "feeds": "yards before/after contact, broken tackles", "family": "nflverse/PFR"},
    "pfr_rec":        {"url": f"{R}/pfr_advstats/advstats_week_rec_{{s}}.parquet", "seasons": list(range(2018, 2026)) + [THIS_SEASON], "fresh_hours": 24,
                       "feeds": "targets, ADOT, YAC, broken tackles, drops, interceptions on target, passer rating when targeted", "family": "nflverse/PFR"},
    "pfr_def":        {"url": f"{R}/pfr_advstats/advstats_week_def_{{s}}.parquet", "seasons": list(range(2018, 2026)) + [THIS_SEASON], "fresh_hours": 24,
                       "feeds": "coverage: targets allowed, completion % allowed, yards allowed, pressures, missed tackles", "family": "nflverse/PFR"},
    "snap_counts":    {"url": f"{R}/snap_counts/snap_counts_{{s}}.parquet", "seasons": list(range(2012, 2026)) + [THIS_SEASON], "fresh_hours": 12,
                       "feeds": "offense/defense/ST snaps and snap % per player-game (incl. preseason) → role anchors", "family": "nflverse/PFR"},
    "espn_qbr_week":  {"url": f"{R}/espn_data/qbr_week_level.parquet", "seasons": None, "fresh_hours": 24, "feeds": "ESPN Total QBR per QB-week", "family": "nflverse/ESPN"},
    # ---- rosters / people -------------------------------------------------------------------
    "depth_charts":   {"url": f"{R}/depth_charts/depth_charts_{{s}}.parquet", "seasons": SEASONS_5 + [THIS_SEASON], "fresh_hours": 6,
                       "feeds": "team depth charts by week (role rank, formation slots)", "family": "nflverse"},
    "rosters_weekly": {"url": f"{R}/weekly_rosters/roster_weekly_{{s}}.parquet", "seasons": SEASONS_5 + [THIS_SEASON], "fresh_hours": 6,
                       "feeds": "who is on the roster each week, status, age, experience", "family": "nflverse"},
    "injuries":       {"url": f"{R}/injuries/injuries_{{s}}.parquet", "seasons": SEASONS_5 + [THIS_SEASON], "fresh_hours": 3,
                       "feeds": "official injury reports: practice status, game status", "family": "nflverse"},
    "players":        {"url": f"{R}/players/players.parquet", "seasons": None, "fresh_hours": 48, "feeds": "id crosswalk (gsis/espn/pfr/sleeper/yahoo…), position, draft, birth date", "family": "nflverse"},
    "contracts":      {"url": f"{R}/contracts/historical_contracts.parquet", "seasons": None, "fresh_hours": 168, "feeds": "contract value/guarantees/years (role security)", "family": "nflverse/OTC"},
    "draft_picks":    {"url": f"{R}/draft_picks/draft_picks.parquet", "seasons": None, "fresh_hours": 168, "feeds": "draft capital (rookie baseline prior)", "family": "nflverse"},
    "combine":        {"url": f"{R}/combine/combine.parquet", "seasons": None, "fresh_hours": 168, "feeds": "athleticism (40, vertical, broad, 3-cone, shuttle, size)", "family": "nflverse"},
    "officials":      {"url": f"{R}/officials/officials.parquet", "seasons": None, "fresh_hours": 72, "feeds": "crew assignments per game (penalty/pace tendencies)", "family": "nflverse"},
    "schedules":      {"url": "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv", "seasons": None, "fresh_hours": 6, "kind": "csv",
                       "feeds": "kickoffs, spread/total/moneyline, roof, surface, temp, wind, referee, rest, coaches, QBs", "family": "nflverse"},
    # ---- free APIs --------------------------------------------------------------------------
    "sleeper_players":  {"url": "https://api.sleeper.app/v1/players/nfl", "seasons": None, "fresh_hours": 24, "kind": "json_players",
                         "feeds": "11k players with every platform id, depth-chart order, injury status/body part, practice notes", "family": "Sleeper"},
    "sleeper_trending": {"url": "https://api.sleeper.app/v1/players/nfl/trending/add?lookback_hours=24&limit=100", "seasons": None, "fresh_hours": 1, "kind": "json_snapshot",
                         "feeds": "the crowd's adds in the last 24h (role news leaks here first)", "family": "Sleeper"},
    "sleeper_drops":    {"url": "https://api.sleeper.app/v1/players/nfl/trending/drop?lookback_hours=24&limit=100", "seasons": None, "fresh_hours": 1, "kind": "json_snapshot",
                         "feeds": "the crowd's drops in the last 24h", "family": "Sleeper"},
    "sleeper_state":    {"url": "https://api.sleeper.app/v1/state/nfl", "seasons": None, "fresh_hours": 1, "kind": "json", "feeds": "current season/week", "family": "Sleeper"},
    "sleeper_proj_week": {"url": "https://api.sleeper.app/projections/nfl/{season}/{week}?season_type=regular&position[]=QB&position[]=RB&position[]=WR&position[]=TE&position[]=K", "seasons": None, "fresh_hours": 6, "kind": "json_snapshot",
                          "feeds": "a second projection set per player-week (usage, yards, TDs) — a role read to test against the market", "family": "Sleeper"},
    "sleeper_proj_season": {"url": "https://api.sleeper.app/projections/nfl/{season}?season_type=regular&position[]=QB&position[]=RB&position[]=WR&position[]=TE", "seasons": None, "fresh_hours": 48, "kind": "json_snapshot",
                            "feeds": "season-long projections (role expectations, ADP)", "family": "Sleeper"},
    "sleeper_schedule": {"url": "https://api.sleeper.app/schedule/nfl/regular/{season}", "seasons": None, "fresh_hours": 24, "kind": "json", "feeds": "schedule with status", "family": "Sleeper"},
    "espn_fantasy_proj": {"url": "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leaguedefaults/3?view=kona_player_info", "seasons": None, "fresh_hours": 6, "kind": "json_espn_proj",
                          "feeds": "ESPN's weekly + season projections and ownership % for every player", "family": "ESPN"},
    "espn_injuries":    {"url": "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{team}/injuries?limit=100", "seasons": None, "fresh_hours": 2, "kind": "json_espn_injuries",
                         "feeds": "injury designations with dates and details, per team, between official reports", "family": "ESPN"},
    "espn_news":        {"url": "https://now.core.api.espn.com/v1/sports/news?sport=football&league=nfl&limit=50", "seasons": None, "fresh_hours": 1, "kind": "json_snapshot",
                         "feeds": "headline feed (injuries, roles, transactions)", "family": "ESPN"},
    "underdog_lines":   {"url": "https://api.underdogfantasy.com/beta/v5/over_under_lines", "seasons": None, "fresh_hours": 2, "kind": "json_snapshot",
                         "feeds": "vig-free medians on hundreds of player props (a second market to compare every line against)", "family": "Underdog"},
    "action_scoreboard": {"url": "https://api.actionnetwork.com/web/v1/scoreboard/nfl", "seasons": None, "fresh_hours": 2, "kind": "json_snapshot",
                          "feeds": "multi-book game odds, sometimes public bet/money splits", "family": "Action Network"},
    "rss_news":         {"url": None, "seasons": None, "fresh_hours": 1, "kind": "rss",
                         "feeds": "ESPN / Yahoo / CBS / ProFootballTalk / Google News headlines, dated and sourced (the intel pass reads these)", "family": "RSS"},
}

RSS_FEEDS = {
    "espn": "https://www.espn.com/espn/rss/nfl/news",
    "yahoo": "https://sports.yahoo.com/nfl/rss.xml",
    "cbs": "https://www.cbssports.com/rss/headlines/nfl/",
    "pft": "https://profootballtalk.nbcsports.com/feed/",
    "google_injury": "https://news.google.com/rss/search?q=nfl+injury+report&hl=en-US&gl=US&ceid=US:en",
    "google_inactives": "https://news.google.com/rss/search?q=nfl+inactives+OR+ruled+out+OR+questionable&hl=en-US&gl=US&ceid=US:en",
}
ESPN_TEAM_IDS = list(range(1, 35))   # ESPN team ids 1-34 (some unused)


def _dest(name, season=None, ext="parquet"):
    os.makedirs(UNI, exist_ok=True)
    return os.path.join(UNI, f"{name}_{season}.{ext}" if season is not None else f"{name}.{ext}")


def _fresh(path, hours):
    return os.path.exists(path) and (time.time() - os.path.getmtime(path)) < hours * 3600


def _get(url, timeout=60):
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r


def fetch(name, force=False, season=None, week=None, verbose=True):
    """Download one source (all its seasons) into data/universe. Returns list of (path, rows)."""
    spec = SOURCES[name]; kind = spec.get("kind", "parquet"); out = []
    seasons = spec["seasons"] if spec["seasons"] is not None else [None]
    for s in seasons:
        try:
            if kind == "parquet":
                p = _dest(name, s)
                cur = (s is None) or (s >= THIS_SEASON - (0 if datetime.date.today().month >= 9 else 1))
                if not force and os.path.exists(p) and (not cur or _fresh(p, spec["fresh_hours"])):
                    out.append((p, None)); continue
                url = spec["url"].format(s=s)
                r = _get(url)
                open(p, "wb").write(r.content)
                n = len(pd.read_parquet(p, columns=None).index)
                out.append((p, n))
                if verbose: print(f"  {name} {s or ''}: {n:,} rows")
            elif kind == "csv":
                p = _dest(name, None, "parquet")
                if not force and _fresh(p, spec["fresh_hours"]): out.append((p, None)); continue
                df = pd.read_csv(io.StringIO(_get(spec["url"]).text), low_memory=False)
                df.to_parquet(p, index=False); out.append((p, len(df)))
                if verbose: print(f"  {name}: {len(df):,} rows")
            elif kind == "json_players":
                p = _dest(name, None, "parquet")
                if not force and _fresh(p, spec["fresh_hours"]): out.append((p, None)); continue
                js = _get(spec["url"], timeout=120).json()
                keep = ["player_id", "full_name", "first_name", "last_name", "position", "fantasy_positions", "team", "status", "injury_status",
                        "injury_body_part", "injury_notes", "practice_participation", "practice_description", "depth_chart_order", "depth_chart_position",
                        "age", "years_exp", "number", "height", "weight", "college", "gsis_id", "espn_id", "yahoo_id", "rotowire_id", "rotoworld_id",
                        "sportradar_id", "stats_id", "fantasy_data_id", "pandascore_id", "swish_id", "oddsjam_id", "active", "news_updated", "search_rank"]
                rows = []
                for pid, v in js.items():
                    if not isinstance(v, dict): continue
                    d = {k: v.get(k) for k in keep}; d["player_id"] = pid
                    d["fantasy_positions"] = ",".join(v.get("fantasy_positions") or []) if isinstance(v.get("fantasy_positions"), list) else None
                    rows.append(d)
                df = pd.DataFrame(rows)
                for c in df.columns:
                    if df[c].dtype == object: df[c] = df[c].astype(str).where(df[c].notna(), None)
                df.to_parquet(p, index=False); out.append((p, len(df)))
                if verbose: print(f"  {name}: {len(df):,} players")
            elif kind == "json_snapshot":
                # append-only snapshots: data/universe/<name>/<stamp>.json (a time series of the feed)
                d = os.path.join(UNI, name); os.makedirs(d, exist_ok=True)
                latest = sorted(glob.glob(os.path.join(d, "*.json")))
                if not force and latest and _fresh(latest[-1], spec["fresh_hours"]): out.append((latest[-1], None)); continue
                url = spec["url"]
                if "{season}" in url or "{week}" in url:
                    st = _state(); url = url.format(season=season or st["season"], week=week or st["week"])
                js = _get(url, timeout=60).json()
                stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M")
                p = os.path.join(d, f"{stamp}.json"); json.dump(js, open(p, "w"))
                n = len(js) if isinstance(js, list) else (len(js.get("headlines", js.get("over_under_lines", js.get("games", [])))) if isinstance(js, dict) else 1)
                out.append((p, n))
                if verbose: print(f"  {name}: snapshot {stamp} ({n:,} items)")
            elif kind == "json":
                p = _dest(name, None, "json")
                if not force and _fresh(p, spec["fresh_hours"]): out.append((p, None)); continue
                url = spec["url"]
                if "{season}" in url: url = url.format(season=season or _state()["season"])
                js = _get(url).json(); json.dump(js, open(p, "w")); out.append((p, len(js) if isinstance(js, (list, dict)) else 1))
                if verbose: print(f"  {name}: ok")
            elif kind == "json_espn_proj":
                p = _dest(name, None, "parquet")
                if not force and _fresh(p, spec["fresh_hours"]): out.append((p, None)); continue
                st = _state(); url = spec["url"].format(season=season or st["season"])
                flt = {"players": {"limit": 2000, "sortPercOwned": {"sortAsc": False, "sortPriority": 1},
                                   "filterSlotIds": {"value": [0, 2, 4, 6, 17, 16]}}}
                r = requests.get(url, headers={**UA, "x-fantasy-filter": json.dumps(flt)}, timeout=90); r.raise_for_status()
                rows = []
                for it in r.json().get("players", []):
                    pl = it.get("player", {}); base = {"espn_id": it.get("id"), "name": pl.get("fullName"), "pos_id": pl.get("defaultPositionId"),
                                                      "pro_team_id": pl.get("proTeamId"), "injury_status": pl.get("injuryStatus"),
                                                      "pct_owned": (pl.get("ownership") or {}).get("percentOwned"), "pct_started": (pl.get("ownership") or {}).get("percentStarted")}
                    for stt in pl.get("stats", []) or []:
                        if stt.get("statSourceId") == 1:       # projections
                            row = {**base, "season": stt.get("seasonId"), "week": stt.get("scoringPeriodId"), "split": stt.get("statSplitTypeId"),
                                   "proj_pts": stt.get("appliedTotal")}
                            s_ = stt.get("stats", {}) or {}
                            # ESPN stat ids: 0 pass att, 1 comp, 3 pass yds, 4 pass td, 20 ignore, 23 rush att, 24 rush yds, 25 rush td, 41 rec, 42 rec yds, 43 rec td, 58 targets
                            for sid, nm in (("0", "pass_att"), ("1", "pass_cmp"), ("3", "pass_yds"), ("4", "pass_td"), ("23", "rush_att"), ("24", "rush_yds"), ("25", "rush_td"),
                                            ("41", "receptions"), ("42", "rec_yds"), ("43", "rec_td"), ("58", "targets")):
                                row[nm] = s_.get(sid)
                            rows.append(row)
                df = pd.DataFrame(rows); df.to_parquet(p, index=False); out.append((p, len(df)))
                if verbose: print(f"  {name}: {len(df):,} projection rows")
            elif kind == "json_espn_injuries":
                p = _dest(name, None, "parquet")
                if not force and _fresh(p, spec["fresh_hours"]): out.append((p, None)); continue
                rows = []
                for tid in ESPN_TEAM_IDS:
                    try:
                        js = _get(spec["url"].format(team=tid), timeout=30).json()
                    except Exception:
                        continue
                    for it in js.get("items", []):
                        ref = it.get("$ref")
                        try:
                            d = _get(ref, timeout=30).json()
                        except Exception:
                            continue
                        ath = (d.get("athlete") or {}).get("$ref", "")
                        rows.append({"team_id": tid, "injury_id": d.get("id"), "date": d.get("date"), "status": d.get("status"),
                                     "short": d.get("shortComment"), "long": d.get("longComment"), "type": (d.get("type") or {}).get("description"),
                                     "detail": (d.get("details") or {}).get("detail"), "side": (d.get("details") or {}).get("side"),
                                     "return_date": (d.get("details") or {}).get("returnDate"), "fantasy_status": ((d.get("details") or {}).get("fantasyStatus") or {}).get("description"),
                                     "athlete_ref": ath, "espn_id": ath.rstrip("/").split("/")[-1].split("?")[0] if ath else None})
                df = pd.DataFrame(rows); df.to_parquet(p, index=False); out.append((p, len(df)))
                if verbose: print(f"  {name}: {len(df):,} injury rows")
            elif kind == "rss":
                d = os.path.join(UNI, "rss"); os.makedirs(d, exist_ok=True)
                stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M")
                latest = sorted(glob.glob(os.path.join(d, "*.parquet")))
                if not force and latest and _fresh(latest[-1], spec["fresh_hours"]): out.append((latest[-1], None)); continue
                import xml.etree.ElementTree as ET
                rows = []
                for src, url in RSS_FEEDS.items():
                    try:
                        root = ET.fromstring(_get(url, timeout=30).content)
                        for item in root.iter("item"):
                            g = lambda t: (item.findtext(t) or "").strip()
                            rows.append({"source": src, "title": g("title"), "link": g("link"), "published": g("pubDate"), "summary": g("description")[:500], "pulled": stamp})
                    except Exception as e:
                        rows.append({"source": src, "title": None, "link": None, "published": None, "summary": f"ERROR {e}", "pulled": stamp})
                df = pd.DataFrame(rows); p = os.path.join(d, f"{stamp}.parquet"); df.to_parquet(p, index=False); out.append((p, len(df)))
                if verbose: print(f"  rss: {len(df):,} headlines from {df.source.nunique()} feeds")
        except requests.HTTPError as e:
            if verbose: print(f"  {name} {s or ''}: not available ({e.response.status_code})")
        except Exception as e:
            if verbose: print(f"  {name} {s or ''}: failed ({type(e).__name__}: {str(e)[:80]})")
    return out


_STATE = None
def _state():
    global _STATE
    if _STATE is None:
        try: _STATE = _get("https://api.sleeper.app/v1/state/nfl", timeout=20).json()
        except Exception: _STATE = {"season": str(THIS_SEASON), "week": 1}
    return _STATE


def inventory(write=True):
    """Count what we hold: per source rows, columns, seasons, bytes, refreshed; totals for the Model Room."""
    inv = {"built": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="minutes"), "sources": {}, "totals": {}}
    tot_rows = tot_cells = tot_bytes = 0; n_files = 0
    for name, spec in SOURCES.items():
        kind = spec.get("kind", "parquet")
        files = sorted(glob.glob(os.path.join(UNI, f"{name}_*.parquet")) + glob.glob(os.path.join(UNI, f"{name}.parquet")) +
                       glob.glob(os.path.join(UNI, name, "*.json")) + (glob.glob(os.path.join(UNI, "rss", "*.parquet")) if kind == "rss" else []) +
                       glob.glob(os.path.join(UNI, f"{name}.json")))
        rows = cells = b = 0; cols = None; seasons = set(); latest = None
        for f in files:
            b += os.path.getsize(f); n_files += 1
            mt = datetime.datetime.fromtimestamp(os.path.getmtime(f), datetime.timezone.utc)
            latest = max(latest, mt) if latest else mt
            if f.endswith(".parquet"):
                try:
                    import pyarrow.parquet as pq
                    md = pq.ParquetFile(f); n = md.metadata.num_rows; c = md.metadata.num_columns
                    rows += n; cells += n * c; cols = c
                    if "season" in md.schema.names and n:
                        try:
                            seasons |= set(pd.read_parquet(f, columns=["season"])["season"].dropna().astype(int).unique().tolist())
                        except Exception: pass
                except Exception: pass
            else:
                try:
                    js = json.load(open(f)); n = len(js) if isinstance(js, list) else len(js.get("headlines", js.get("over_under_lines", js.get("games", js))) or [])
                    rows += n; cells += n * 12
                except Exception: pass
        if not files: continue
        inv["sources"][name] = {"family": spec["family"], "feeds": spec["feeds"], "files": len(files), "rows": rows, "columns": cols, "cells": cells,
                                "bytes": b, "seasons": sorted(seasons)[:1] + sorted(seasons)[-1:] if seasons else None, "n_seasons": len(seasons),
                                "refreshed": latest.isoformat(timespec="minutes") if latest else None}
        tot_rows += rows; tot_cells += cells; tot_bytes += b
    # the model's own tables (already in data/)
    own = {}
    for f in ("pbp_2021_2025.parquet", "prop_player_games.parquet", "xtd_2026.parquet", "schedules.parquet"):
        p = os.path.join(DATA, f)
        if os.path.exists(p):
            import pyarrow.parquet as pq
            md = pq.ParquetFile(p); own[f] = {"rows": md.metadata.num_rows, "columns": md.metadata.num_columns}
            tot_rows += md.metadata.num_rows; tot_cells += md.metadata.num_rows * md.metadata.num_columns; tot_bytes += os.path.getsize(p)
    inv["own_tables"] = own
    fam = {}
    for v in inv["sources"].values(): fam[v["family"]] = fam.get(v["family"], 0) + 1
    inv["totals"] = {"sources": len(inv["sources"]), "families": fam, "files": n_files, "rows": tot_rows, "data_points": tot_cells, "bytes": tot_bytes,
                     "seasons_span": [min((s for v in inv["sources"].values() if v["seasons"] for s in v["seasons"]), default=None),
                                      max((s for v in inv["sources"].values() if v["seasons"] for s in v["seasons"]), default=None)]}
    if write:
        json.dump(inv, open(os.path.join(DATA, "data_ledger.json"), "w"), indent=1)
    return inv


def refresh(names=None, force=False):
    names = names or list(SOURCES)
    for n in names:
        print(f"[{n}]")
        fetch(n, force=force)
    inv = inventory()
    t = inv["totals"]
    print(f"\ninventory: {t['sources']} sources · {t['files']} files · {t['rows']:,} rows · {t['data_points']:,} data points · {t['bytes']/1e6:.0f} MB")
    return inv


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--inventory" in sys.argv:
        inv = inventory(); print(json.dumps(inv["totals"], indent=1))
    else:
        refresh(args or None, force="--force" in sys.argv)
