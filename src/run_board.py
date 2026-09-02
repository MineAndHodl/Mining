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
import whale_missile as wm
import primetime_sgp as ps
import build_board_page as bbp


def load_adp():
    try:
        players = json.load(open(f"{DATA}/ffc_adp_2026.json"))["players"]
        return pd.DataFrame([{"name": p["name"], "adp": float(p["adp"])} for p in players])
    except Exception:
        return pd.DataFrame(columns=["name", "adp"])


def run(key=None, days=14, rebuild_games=False, cached=False):
    key = key or os.environ.get("ODDS_API_KEY")
    proj = pd.read_parquet(f"{DATA}/projection_2026.parquet")
    cache_p = f"{DATA}/_raw_prop_board.parquet"; cache_t = f"{DATA}/_raw_td_board.parquet"

    # 1) prop board (fixed skew correction), keep PICKs, attach rich reasons
    if cached and os.path.exists(cache_p):
        board = pd.read_parquet(cache_p); rem = "cached"
    else:
        board, rem = pb.build_prop_board(key=key, projections=proj, days=days)
        if board is not None and len(board):
            board.to_parquet(cache_p)
    if board is not None and len(board):
        board["status"] = board.apply(pb.classify, axis=1)
        board["reason"] = board.apply(lambda r: pb.reason(r), axis=1)
    picks = board[board["status"] == "PICK"] if board is not None and len(board) else pd.DataFrame()
    print(f"props: {0 if board is None else len(board)} priced, {len(picks)} PICKs | credits {rem}")

    # 2) TD board (now calibrated)
    try:
        if cached and os.path.exists(cache_t):
            tdb = pd.read_parquet(cache_t)
        else:
            tdb, _ = tb.build_td_board(key=key, days=days)
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
    games = None if rebuild_games else prev.get("best_games")
    ranked = rb.build_ranked(prop_board=picks, td_board=tdb,
                             games=None, top=10, adp=adp)
    if games is not None:
        ranked["best_games"] = games          # preserve prior games unless rebuilding

    # 4) grades + rails + the official recommended slate, selected over the FULL POOL (every
    #    scored candidate), with intel gates/leans already in the score. See grading.py.
    import grading, td_board as _tb
    weeks = _tb.current_season_weeks_played()
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

    # 6) two parlay lanes (in cached mode, reuse the prior parlays — offline re-rank)
    if cached:
        ranked["whale_missile"] = prev.get("whale_missile")
        ranked["primetime_sgps"] = prev.get("primetime_sgps", [])
    else:
        try:
            ranked["whale_missile"] = wm.build_missile(key=key, days=days, prop=board)  # reuse board (no 2nd API fetch)
        except Exception as e:
            print("whale missile skipped:", e)
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
    ranked = _clean(ranked)
    json.dump(ranked, open(f"{DATA}/ranked_board.json", "w"), indent=1)
    bbp.build()
    print(f"board rebuilt: {len(ranked.get('best_props', []))} best props, "
          f"{len(ranked.get('best_games', []))} games, "
          f"{'missile' if ranked.get('whale_missile') else 'no missile'}, "
          f"{len(ranked.get('primetime_sgps', []))} SGPs")
    return ranked


if __name__ == "__main__":
    d = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 14
    run(days=d, rebuild_games="--games" in sys.argv, cached="--cached" in sys.argv)
