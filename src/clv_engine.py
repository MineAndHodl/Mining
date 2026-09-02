"""CLV engine — does the DK-vs-sharp gap predict where DK's line moves, and does riding
it capture closing-line value?

Given an EARLY snapshot and a CLOSE snapshot (from line_capture), for each game:
  early_gap = DK - sharp        (at the early time)
  dk_move   = DK_close - DK_early
Thesis: DK drifts toward the sharp number, so early_gap should predict -dk_move (if DK is
above sharp, DK should come DOWN toward it). We test:
  1. sign agreement + correlation of early_gap vs the DK move
  2. realized CLV: betting the sharp-implied side at the DK early number, how many points
     better is it than the DK closing number (positive = we beat the close)
  3. profitability (needs results): does the number we locked beat the closing spread ATS
"""
import os, sys, datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import line_capture as lc

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def pair_snapshots(early_df, close_df, market="spreads", side="home", within_days=7,
                   early_ts=None):
    """Join early & close consensus/gaps on game_id; keep games kicking off within `within_days`
    of the early snapshot (i.e. that week's slate). Event ids are stable across snapshots."""
    e = lc.consensus_and_gaps(early_df, market, side).rename(
        columns={"dk": "dk_early", "sharp": "sharp_early", "gap_dk_minus_sharp": "gap_early",
                 "book_spread": "spread_early"})
    c = lc.consensus_and_gaps(close_df, market, side).rename(
        columns={"dk": "dk_close", "sharp": "sharp_close", "gap_dk_minus_sharp": "gap_close"})
    m = e.merge(c[["game_id", "dk_close", "sharp_close"]], on="game_id", how="inner")
    if early_ts and len(m):
        comm = pd.to_datetime(m["commence"], utc=True)
        lo = pd.to_datetime(early_ts, utc=True)
        hi = lo + pd.Timedelta(days=within_days)
        m = m[(comm >= lo) & (comm <= hi)]
    m["dk_move"] = (m["dk_close"] - m["dk_early"]).round(2)
    return m


def analyze(paired):
    """Core stats: does early gap predict the DK move, and what CLV does riding it capture?"""
    d = paired.dropna(subset=["gap_early", "dk_move"]).copy()
    if len(d) < 5:
        return {"n": len(d), "note": "too few games"}
    # DK should move TOWARD sharp: if gap_early>0 (DK above sharp), DK should move DOWN (dk_move<0)
    d["predicted_move"] = -d["gap_early"]          # what the gap says DK should do
    # correlation between the gap-implied move and the actual move
    corr = float(np.corrcoef(d["predicted_move"], d["dk_move"])[0, 1]) if d["predicted_move"].std() else float("nan")
    moved = d[d["gap_early"].abs() >= 0.5]         # games where DK & sharp actually disagreed
    sign_ok = float((np.sign(moved["predicted_move"]) == np.sign(moved["dk_move"])).mean()) if len(moved) else float("nan")
    # realized CLV: bet the side DK will move toward, at DK's early number.
    # if DK is a bigger home dog than sharp (gap_early>0 on home spread), the SHARP side is home;
    # you take home at dk_early (better number), and it should close at dk_close (smaller dog).
    # CLV in points = how much better dk_early is than dk_close for the side you took.
    def clv_points(r):
        if r["gap_early"] > 0:      # take HOME at dk_early (home dog inflated at DK)
            return r["dk_early"] - r["dk_close"]     # +ve if you got more points than close
        elif r["gap_early"] < 0:   # take AWAY: away line = -home; better if home line rose
            return r["dk_close"] - r["dk_early"]
        return 0.0
    moved = moved.copy(); moved["clv_pts"] = moved.apply(clv_points, axis=1)
    return {"n": int(len(d)), "n_disagree": int(len(moved)),
            "corr_gap_vs_move": round(corr, 3),
            "sign_agreement": round(sign_ok, 3) if sign_ok == sign_ok else None,
            "mean_clv_pts": round(float(moved["clv_pts"].mean()), 3) if len(moved) else None,
            "pct_positive_clv": round(float((moved["clv_pts"] > 0).mean()), 3) if len(moved) else None,
            "avg_gap_when_disagree": round(float(moved["gap_early"].abs().mean()), 2) if len(moved) else None}


def run_backtest(key=None, weeks=None, regions="us,eu", markets="spreads,totals", verbose=True):
    """weeks: list of (early_rfc3339, close_rfc3339). Pulls historical snapshots and analyzes."""
    key = key or os.environ.get("ODDS_API_KEY")
    all_sp, all_tot = [], []
    total_cost = 0
    for early_ts, close_ts in weeks:
        ed, em = lc.fetch_games(key, regions, markets, historical=True, date=early_ts)
        cd, cm = lc.fetch_games(key, regions, markets, historical=True, date=close_ts)
        total_cost += int(em["cost"] or 0) + int(cm["cost"] or 0)
        sp = pair_snapshots(ed, cd, "spreads", "home", early_ts=early_ts)
        tot = pair_snapshots(ed, cd, "totals", "over", early_ts=early_ts)
        sp["week_early"] = early_ts; tot["week_early"] = early_ts
        all_sp.append(sp); all_tot.append(tot)
        if verbose:
            print(f"  {early_ts[:10]} -> {close_ts[:10]}: {len(sp)} spread games, {len(tot)} totals")
    SP = pd.concat(all_sp, ignore_index=True); TOT = pd.concat(all_tot, ignore_index=True)
    return {"spreads": analyze(SP), "totals": analyze(TOT),
            "cost": total_cost, "remaining": cm.get("remaining")}, SP, TOT


if __name__ == "__main__":
    # smoke: two snapshots one week apart, 2024
    weeks = [("2024-11-13T18:00:00Z", "2024-11-17T17:00:00Z")]
    res, SP, TOT = run_backtest(weeks=weeks)
    import json
    print(json.dumps(res, indent=1))
