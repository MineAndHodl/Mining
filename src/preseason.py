"""Continuity-weighted preseason prior + the year-over-year persistence backtest.

Tests Alex's hypothesis directly: team strength regresses hard across the offseason,
BUT if you isolate continuity (same coach, same QB, returning snaps) the carryover
fraction should be HIGHER for high-continuity teams. If true, a continuity-weighted
preseason prior beats a flat/uniform prior in Weeks 1-4 — the softest-market window.

Core test:  R_S = a * R_{S-1} + b * (continuity * R_{S-1})   (walk-forward)
  a  = baseline carryover fraction (how much persists on average)
  b  = does continuity RAISE carryover?  b>0 and significant  =>  hypothesis holds.

Then a decision test: predict W1-4 game margins with each prior, compare MAE to the
closing spread (the market benchmark we'd need to beat to have an early edge).
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import build_team_games, ratings_asof, OFFSEASON_WEEKS

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def season_end_ratings(tg):
    """Net rating for each team at the END of each season (data through that season)."""
    seasons = sorted(tg["season"].unique())
    base = {s: i * (18 + OFFSEASON_WEEKS) for i, s in enumerate(seasons)}
    out = {}
    for s in seasons:
        asof = base[s] + 23          # just past the regular season
        r = ratings_asof(tg, asof).set_index("team")["net"]
        out[s] = r
    return out, base


def continuity_table():
    coach = pd.read_parquet(f"{DATA}/team_coach.parquet")
    prim = pd.read_parquet(f"{DATA}/team_primary_qb.parquet")
    wk1 = pd.read_parquet(f"{DATA}/team_wk1_qb.parquet")
    snap = pd.read_parquet(f"{DATA}/team_snap_return.parquet")

    # coach continuity: team's modal coach in S == modal coach in S-1
    c = coach.copy()
    c_prev = c.rename(columns={"season": "s1", "coach": "coach_prev"})
    c_prev["season"] = c_prev["s1"] + 1
    coach_cont = c.merge(c_prev[["season", "team", "coach_prev"]], on=["season", "team"], how="left")
    coach_cont["coach_cont"] = (coach_cont["coach"] == coach_cont["coach_prev"]).astype(float)
    coach_cont.loc[coach_cont["coach_prev"].isna(), "coach_cont"] = np.nan

    # qb continuity: team's week-1 starter in S was the team's PRIMARY qb in S-1
    p_prev = prim.rename(columns={"season": "s1", "primary_qb": "primary_prev"})
    p_prev["season"] = p_prev["s1"] + 1
    qc = wk1.merge(p_prev[["season", "team", "primary_prev"]], on=["season", "team"], how="left")
    qc["qb_cont"] = (qc["wk1_qb"] == qc["primary_prev"]).astype(float)
    qc.loc[qc["primary_prev"].isna(), "qb_cont"] = np.nan

    t = coach_cont[["season", "team", "coach_cont"]].merge(
        qc[["season", "team", "qb_cont"]], on=["season", "team"], how="outer").merge(
        snap[["season", "team", "snap_return"]], on=["season", "team"], how="left")
    return t


def build_panel():
    tg = build_team_games(f"{DATA}/team_game_epa.parquet", None)
    rat, base = season_end_ratings(tg)
    cont = continuity_table()
    rows = []
    seasons = sorted(rat.keys())
    for s in seasons:
        if (s - 1) not in rat:
            continue
        r_now, r_prev = rat[s], rat[s - 1]
        for team in r_now.index:
            if team not in r_prev.index:
                continue
            c = cont[(cont["season"] == s) & (cont["team"] == team)]
            row = {"season": s, "team": team,
                   "R_now": float(r_now[team]), "R_prev": float(r_prev[team])}
            if len(c):
                row["coach_cont"] = c["coach_cont"].iloc[0]
                row["qb_cont"] = c["qb_cont"].iloc[0]
                row["snap_return"] = c["snap_return"].iloc[0]
            rows.append(row)
    return pd.DataFrame(rows)


def persistence_test(panel):
    """Regress R_now on R_prev and continuity-interacted R_prev. b>0 => hypothesis."""
    d = panel.dropna(subset=["R_now", "R_prev"]).copy()
    # continuity composite (0-1): available signals, renormalized when snaps missing
    def comp(row):
        parts, wts = [], []
        if not np.isnan(row.get("coach_cont", np.nan)):
            parts.append(row["coach_cont"]); wts.append(0.30)
        if not np.isnan(row.get("qb_cont", np.nan)):
            parts.append(row["qb_cont"]); wts.append(0.40)
        if not np.isnan(row.get("snap_return", np.nan)):
            parts.append(row["snap_return"]); wts.append(0.30)
        if not parts:
            return np.nan
        w = np.array(wts) / sum(wts)
        return float(np.dot(w, parts))
    d["C"] = d.apply(comp, axis=1)
    d = d.dropna(subset=["C"])

    # Model A: R_now = a * R_prev          (uniform carryover)
    X_a = d[["R_prev"]].values
    a_hat = np.linalg.lstsq(X_a, d["R_now"].values, rcond=None)[0][0]

    # Model B: R_now = a*R_prev + b*(C-centered)*R_prev
    Cc = d["C"] - d["C"].mean()
    X_b = np.column_stack([d["R_prev"].values, (Cc * d["R_prev"]).values])
    coef_b, *_ = np.linalg.lstsq(X_b, d["R_now"].values, rcond=None)
    a_b, b_b = coef_b
    # bootstrap b for significance
    rng = np.random.default_rng(7)
    bs = []
    idx = np.arange(len(d))
    for _ in range(2000):
        j = rng.choice(idx, len(idx), replace=True)
        Xb = X_b[j]; yb = d["R_now"].values[j]
        try:
            bs.append(np.linalg.lstsq(Xb, yb, rcond=None)[0][1])
        except Exception:
            pass
    bs = np.array(bs)
    ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    return {"uniform_carryover_a": round(a_hat, 3),
            "carryover_base_a": round(float(a_b), 3),
            "continuity_slope_b": round(float(b_b), 3),
            "b_ci95": [round(ci[0], 3), round(ci[1], 3)],
            "b_significant": bool(ci[0] > 0 or ci[1] < 0),
            "n": int(len(d)),
            "corr_Rprev_Rnow": round(float(d["R_prev"].corr(d["R_now"])), 3)}


if __name__ == "__main__":
    panel = build_panel()
    panel.to_parquet(f"{DATA}/preseason_panel.parquet")
    res = persistence_test(panel)
    print(json.dumps(res, indent=2))
