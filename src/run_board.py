"""THE weekly pick-run — the one committed entry point that rebuilds The Board end to end.

    odds pull -> prop board -> TD board -> rank (investigate + Desk Score)
              -> rich reasons + factor scorecards
              -> cross-slate near-lock parlay (non-correlated) + primetime SGPs (correlated)
              -> write ranked_board.json -> rebuild the published page

Run:  ODDS_API_KEY=... python3 src/run_board.py [days]

This replaces the old uncommitted inline script, so the product is reproducible: next week
is `python3 src/run_board.py`. Games (spreads/totals/ML) are preserved from the previous
board unless --games forces a rebuild through the desk_card pipeline (heavier pull).
"""
import os, sys, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "..", "data")

import prop_board as pb
import td_board as tb
import rank_board as rb
import board_factors as bf
import primetime_sgp as ps
import build_board_page as bbp


def load_adp():
    try:
        players = json.load(open(f"{DATA}/ffc_adp_2026.json"))["players"]
        return pd.DataFrame([{"name": p["name"], "adp": float(p["adp"])} for p in players])
    except Exception:
        return pd.DataFrame(columns=["name", "adp"])


def run(key=None, days=14, rebuild_games=False, cached=False, refresh=False, open_week=False, tag=None, window=None):
    """window=<minutes>: LAST CALL mode — price only the games kicking off inside the window
    (props/TDs/games fetched for that window only, ~30-60 credits), never touch the week's
    caches or ranked_board.json, and produce data/last_call.json: voids, late swaps, stale
    numbers, still-live Card picks and kickoff weather for those games."""
    key = key or os.environ.get("ODDS_API_KEY")
    proj = pd.read_parquet(f"{DATA}/projection_2026.parquet")
    window_mode = window is not None
    if window_mode:
        days = float(window) / 1440.0; cached = False; open_week = False; tag = tag or "lastcall"
        cache_p = f"{DATA}/_raw_prop_board_window.parquet"; cache_t = f"{DATA}/_raw_td_board_window.parquet"
        raw_lines_path = f"{DATA}/_raw_prop_lines_window.parquet"
        td_raw_paths = (f"{DATA}/_raw_td_prices_window.parquet", f"{DATA}/_td_env_window.json")
    else:
        cache_p = f"{DATA}/_raw_prop_board.parquet"; cache_t = f"{DATA}/_raw_td_board.parquet"
        raw_lines_path = None; td_raw_paths = None

    # 0) THE MODE SWITCH: which season/week the model believes it is in. Before the first game
    #    of the season the board runs in offseason mode (last season's history, market-role
    #    baseline, half-strength form); once current-season games are in the player-game table
    #    the recency edge, the injury report and xTD's in-season mode take over — automatically.
    import refresh_data as rd
    if refresh:
        rd.refresh()
    season, weeks_played = rd.season_state()
    rec_season, rec_week = (season, weeks_played) if weeks_played >= 1 else (season - 1, None)
    print(f"mode: season {season}, weeks played {weeks_played} -> "
          f"{'IN-SEASON (recency through week %d)' % weeks_played if rec_week else 'OFFSEASON (last-season history, market-role baseline)'}")

    # 0b) KICKOFF WEATHER (Open-Meteo, free): the validated weather module finally has a feed.
    #     Live runs refresh the forecast; cached runs reuse the saved one. Domes are neutral.
    import weather_feed as wf
    try:
        wx = wf.forecast() if not cached else wf.load()
    except Exception as e:
        wx = wf.load(); print("weather forecast not refreshed:", e)
    weather_by_game = {g: {"wind": w.get("wind"), "temp": w.get("temp"), "roof": w.get("roof", "outdoors")}
                       for g, w in (wx or {}).items() if not w.get("error")}
    print(f"weather: forecast for {len(weather_by_game)} games" + (f" (e.g. {next(iter(weather_by_game))}: {wf.weather_note(wx[next(iter(weather_by_game))])})" if weather_by_game else ""))

    # 1) prop board (fixed skew correction), keep PICKs, attach rich reasons
    # cached = RE-PRICE the last fetched raw lines (model changes never cost credits); only if
    # no raw-line cache exists yet does it fall back to the last priced board.
    if cached and os.path.exists(pb.RAW_LINES):
        board, rem = pb.build_prop_board(key=key, projections=proj, days=days, cached=True, weather_by_game=weather_by_game,
                                         recency_season=rec_season, recency_week=rec_week)
    elif cached and os.path.exists(cache_p):
        board = pd.read_parquet(cache_p); rem = "cached (priced)"
    else:
        board, rem = pb.build_prop_board(key=key, projections=proj, days=days, weather_by_game=weather_by_game,
                                         recency_season=rec_season, recency_week=rec_week, raw_path=raw_lines_path)
    if window_mode and (board is None or not len(board)):
        print("no games in window"); return None
        if board is not None and len(board):
            board.to_parquet(cache_p)
    if board is not None and len(board):
        board.to_parquet(cache_p)                 # the priced board always mirrors this run
        board["status"] = board.apply(pb.classify, axis=1)
        board["reason"] = board.apply(lambda r: pb.reason(r), axis=1)
    picks = board[board["status"] == "PICK"] if board is not None and len(board) else pd.DataFrame()
    print(f"props: {0 if board is None else len(board)} priced, {len(picks)} PICKs | credits {rem}")

    # 2) TD board (now calibrated)
    try:
        # cached = re-PRICE the last fetched prices (model changes never cost credits);
        # falls back to the last priced board only if the raw price cache is missing
        if cached and os.path.exists(tb.RAW_PRICES):
            tdb, _ = tb.build_td_board(key=key, days=days, cached=True)
        elif cached and os.path.exists(cache_t):
            tdb = pd.read_parquet(cache_t)
        else:
            tdb, _ = tb.build_td_board(key=key, days=days, raw_paths=td_raw_paths)
        if len(tdb):
            tdb.to_parquet(cache_t)
    except Exception as e:
        tdb = pd.DataFrame(); print("td board skipped:", e)

    # 3) rank: investigate (fixed ADP logic) + Desk Score (hump curve + verdict multiplier)
    adp = load_adp()
    prev = {}
    try:
        prev = json.load(open(f"{DATA}/ranked_board.json"))
    except Exception:
        pass
    # games lane: rebuilt LIVE every run (a cached carry-over shipped stale Sept-1 lines with no
    # labels for a day and a half). --cached reuses the last live build only if it is labeled.
    games = None
    prev_games = prev.get("best_games") or []
    if cached and prev_games and all(g.get("lane") for g in prev_games):
        games = prev_games
    else:
        try:
            import games_lane as gl
            games, _ = gl.build_games(key=key, days=days)
        except Exception as e:
            print("games lane skipped:", e); games = prev_games if all(g.get("lane") for g in prev_games) else []
    ranked = rb.build_ranked(prop_board=picks, td_board=tdb, games=None, top=10, adp=adp)
    ranked["best_games"] = games

    # 4) grades + rails + the official recommended slate, selected over the FULL POOL (every
    #    scored candidate), with intel gates/leans already in the score. See grading.py.
    import grading, td_board as _tb
    weeks = max(_tb.current_season_weeks_played(), weeks_played)
    grading.grade_board(ranked, weeks_played=weeks)
    pool = ranked.pop("pool", [])
    ranked["pool_summary"] = {"n": len(pool), "n_recommended_from_pool": sum(1 for r in pool if r.get("recommended")),
                              "promoted_from_outside_tabs": [f"{r.get('name')} {r.get('cat')} {r.get('side')} {r.get('line')}"
                                                             for r in ranked.get("recommended", [])
                                                             if not any(r.get("name") == x.get("name") and r.get("cat") == x.get("cat")
                                                                        for lst in ranked["by_category"].values() for x in lst)]}

    # 5) factor scorecards on everything shown (tabs + the slate)
    bf.attach_all(ranked)
    gs = ranked.get("grading", {})
    print(f"grades: {gs.get('grade_counts')} | recommended {gs.get('n_recommended')} "
          f"({gs.get('units_recommended')}u) | season cap {gs.get('season_cap')}")

    # 6) parlay lanes. The Sunday Stack is built from the graded slate (offline, every run);
    #    the primetime SGPs need odds (reused in cached mode).
    import sunday_stack as ss, atd_stack as ats
    if window_mode:
        ranked["sunday_stack"] = None; ranked["td_stack"] = None; ranked["primetime_sgps"] = prev.get("primetime_sgps", [])
    else:
        ranked["sunday_stack"] = ss.build_stack(ranked)
        ranked["td_stack"] = ats.build_stack([r for r in pool if r.get("cat") == "anytime_td"])
    ranked.pop("whale_missile", None)
    if window_mode:
        pass
    elif cached:
        ranked["primetime_sgps"] = prev.get("primetime_sgps", [])
    else:
        try:
            ranked["primetime_sgps"] = ps.build_all(key=key, days=days, projections=proj)  # correlated
            print(f"primetime SGPs: {len(ranked['primetime_sgps'])}")
        except Exception as e:
            ranked["primetime_sgps"] = []; print("primetime SGPs skipped:", e)

    # 6) freshness stamp + write + rebuild page
    import datetime
    ranked["generated_at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%b %d, %Y %H:%M UTC")
    def _clean(x):                      # NaN -> null everywhere (DataFrame round-trips re-NaN None)
        if isinstance(x, dict): return {k: _clean(v) for k, v in x.items()}
        if isinstance(x, list): return [_clean(v) for v in x]
        if isinstance(x, float) and x != x: return None
        return x
    ranked = _clean(ranked); pool = _clean(pool)
    # 7) line history: archive this fetch (live runs only — a cached re-price has no new lines)
    #    and annotate every row with where its number OPENED and where it is NOW.
    import ledger, line_history as lh
    wk_season = wk_week = None
    try:
        games_list = sorted({r.get("game") for r in pool if r.get("game")} | {g.get("game") for g in ranked.get("best_games", []) or [] if g.get("game")})
        wk_season, wk_week, _ = ledger.resolve_week(games_list, season)
        if not cached:
            lh.archive(wk_season, wk_week, tag=tag or ("open" if open_week else "run"), props_path=raw_lines_path,
                       td_path=(td_raw_paths[0] if td_raw_paths else None), games=ranked.get("best_games") if window_mode else None)
        ranked = lh.attach({**ranked, "pool": pool}, wk_season, wk_week)
        pool = ranked.pop("pool", pool)
        ranked["line_movers"] = lh.movers(wk_season, wk_week)
    except Exception as e:
        print("line history skipped:", e)
    if window_mode:
        # LAST CALL: the kickoff-window product. The week's board and ledger stay as they are;
        # this writes data/last_call.json (+ a kickoff entry in the ledger) and rebuilds the page.
        import last_call as lc
        out = lc.build(ranked, pool, prev, wk_season, wk_week, window, wx)
        json.dump(ranked, open(f"{DATA}/ranked_window.json", "w"), indent=1)
        bbp.build()
        print(f"LAST CALL: {len(out.get('games', []))} games in the window, {len(out.get('items', []))} items")
        return ranked
    json.dump(ranked, open(f"{DATA}/ranked_board.json", "w"), indent=1)
    # 8) the ledger: freeze every number we put out (the grading loop's input). --open marks
    #    this run as the week's OFFICIAL open (the Tuesday card); any earlier build is a preview.
    try:
        ledger.snapshot({**ranked, "pool": pool}, force_open=open_week)
    except Exception as e:
        print("ledger not written:", e)
    # 9) post copy: the Card / free pick / Touchdown Tax / line moves as ready-to-post text, built
    #    from the frozen numbers (data/post_copy.md) — a post can never say what the model didn't.
    try:
        import post_copy
        post_copy.write(ranked)
    except Exception as e:
        print("post copy skipped:", e)
    bbp.build()
    print(f"board rebuilt: {len(ranked.get('best_props', []))} best props, "
          f"{len(ranked.get('best_games', []))} games, "
          f"{'stack ' + str(ranked['sunday_stack']['combined_american']) if ranked.get('sunday_stack') else 'no stack'}, "
          f"{'Touchdown Tax ' + str(ranked['td_stack']['combined_american']) if ranked.get('td_stack') else 'no Touchdown Tax'}, "
          f"{len(ranked.get('primetime_sgps', []))} SGPs")
    return ranked


if __name__ == "__main__":
    d = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 14
    tag = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--tag=")), None)
    win = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--window=")), None)
    run(days=d, rebuild_games="--games" in sys.argv, cached="--cached" in sys.argv, refresh="--refresh" in sys.argv,
        open_week="--open" in sys.argv, tag=tag, window=win)
