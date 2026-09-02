"""Does 5 seasons of history beat 3? Re-fit each model on 2023-24 (the 3-season style) vs
2021-24 (5 seasons), score on 2025 held out. Adopt the deeper history ONLY where it wins.
(Alex: read every play of the last 5 years. Us: yes — and prove it helps before it ships.)

Models compared:
  1. multifactor ridge (yardage residual): sign-hit on strong trends + MAE of the residual
  2. xTD conversion priors (TD scoring): Brier on 2025 player-games (week-1-style fold)
  3. CV-by-volume tiers: stability (do the tiers move?)
  4. offseason form ratio: stability with more opening-week samples
  5. role-cascade coefficients: stability
  6. correlations (QB-WR, WR-WR): stability
Writes data/history_compare.json.
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multifactor as mf
DATA = mf.DATA
PG3 = f"{DATA}/prop_player_games.parquet"; PG5 = f"{DATA}/prop_player_games_5y.parquet"
PBP3 = f"{DATA}/pbp_2023_2025.parquet"; PBP5 = f"{DATA}/pbp_2021_2025.parquet"
MKTS = ["rec_yds", "rush_yds", "pass_yds", "receptions"]
OUT = {}


def _ridge(X, y, l2=10.0):
    return np.linalg.solve(X.T @ X + l2 * np.eye(X.shape[1]), X.T @ y)


def multifactor_compare():
    print("\n=== 1. MULTIFACTOR (yardage residual) — fit through 2024, test 2025 ===")
    res = {}
    for tag, path, first in (("3y", PG3, 2023), ("5y", PG5, 2021)):
        pg = pd.read_parquet(path); pg = pg[(pg.season >= first) & pg.mkt.isin(MKTS)]
        d = mf.build_features(pg)
        tr = d[d.season <= 2024]; te = d[d.season == 2025]
        hits, maes, n = [], [], 0
        for mkt in MKTS:
            s, t = tr[tr.mkt == mkt], te[te.mkt == mkt]
            if len(s) < 300 or len(t) < 100: continue
            w = _ridge(s[mf.RAW_FEATURES].to_numpy(float), s["resid"].to_numpy(float))
            pred = t[mf.RAW_FEATURES].to_numpy(float) @ w
            strong = np.abs(pred) > np.quantile(np.abs(pred), 0.6)
            hits.append((np.sign(pred[strong]) == np.sign(t["resid"].to_numpy()[strong])).mean())
            maes.append(np.abs(pred - t["resid"].to_numpy()).mean()); n += int(strong.sum())
        res[tag] = {"strong_sign_hit": round(float(np.mean(hits)), 4), "resid_mae": round(float(np.mean(maes)), 3), "n_strong": n}
        print(f"  {tag}: strong-trend sign-hit {res[tag]['strong_sign_hit']}  resid MAE {res[tag]['resid_mae']}  (n={n})")
    win = res["5y"]["strong_sign_hit"] > res["3y"]["strong_sign_hit"] and res["5y"]["resid_mae"] <= res["3y"]["resid_mae"] + 0.05
    res["verdict"] = "5y WINS" if win else ("mixed" if res["5y"]["strong_sign_hit"] > res["3y"]["strong_sign_hit"] else "3y holds")
    print("  ->", res["verdict"]); OUT["multifactor"] = res


def xtd_compare():
    print("\n=== 2. xTD conversion priors — learn shares from 2024, priors from 2023-24 vs 2021-24, test 2025 ===")
    import xtd_model as xm, xtd_validate as xv
    pbp5 = pd.read_parquet(PBP5)
    res = {}
    for tag, first in (("3y", 2023), ("5y", 2021)):
        sub = pbp5[pbp5.season >= first]
        xm.load_pbp = (lambda s=sub: s)                      # point the model at this history
        xdf = xm.build_xtd([2024], target_season=2025, week1_mode=True)
        # priors come from build_xtd's data_seasons; widen them explicitly for the 5y case
        if tag == "5y":
            players = xm.load_players()
            conv = xm.conversion_priors(xm._events(sub[sub.season <= 2024]), players)
            # rebuild with wider priors by re-running with all seasons but shares from latest
            xdf = xm.build_xtd([2021, 2022, 2023, 2024], target_season=2025, week1_mode=True)
        x_p = dict(zip(xdf["pid"], xdf["p_atd_raw"]))
        ev = xm._events(sub[sub.season == 2025]); pg = xv.realized_player_games(ev)
        pg = pg[pg.pid.isin(x_p)]; ng = pg.groupby("pid")["game_id"].nunique(); pg = pg[pg.pid.map(ng) >= 4]
        p = np.clip(pg.pid.map(x_p).values, 0.01, 0.99); y = pg.scored.values
        res[tag] = {"brier": round(float(np.mean((p - y) ** 2)), 4), "n": int(len(pg))}
        print(f"  {tag}: Brier {res[tag]['brier']} (n={res[tag]['n']})")
    res["verdict"] = "5y WINS" if res["5y"]["brier"] < res["3y"]["brier"] else "3y holds"
    print("  ->", res["verdict"]); OUT["xtd"] = res


def cv_compare():
    print("\n=== 3. CV by volume tier — 3y vs 5y ===")
    res = {}
    for tag, path, first in (("3y", PG3, 2023), ("5y", PG5, 2021)):
        d = pd.read_parquet(path); d = d[d.season >= first]; r = {}
        for mkt, bins in (("rush_yds", [(15, 45), (45, 70), (70, 999)]), ("rec_yds", [(15, 40), (40, 65), (65, 999)])):
            for lo, hi in bins:
                cvs = []
                for pid, g in d[d.mkt == mkt].groupby("pid"):
                    y = g.y.values
                    if len(y) >= 8 and lo <= y.mean() < hi and y.mean() > 0: cvs.append(y.std() / y.mean())
                r[f"{mkt}[{lo},{hi})"] = round(float(np.median(cvs)), 2) if cvs else None
        res[tag] = r; print(f"  {tag}: {r}")
    OUT["cv_tiers"] = res


def offseason_ratio_compare():
    print("\n=== 4. Offseason form ratio (opening slope / in-season slope) — 3y vs 5y ===")
    res = {}
    for tag, path, first in (("3y", PG3, 2023), ("5y", PG5, 2021)):
        pg = pd.read_parquet(path); pg = pg[(pg.season >= first) & pg.mkt.isin(MKTS)]
        d = mf.build_features(pg); r = {}
        for m in MKTS:
            x = d[d.mkt == m]
            s_in = np.polyfit(x[x.week >= 4].form3, x[x.week >= 4].resid, 1)[0]
            s_op = np.polyfit(x[x.week <= 3].form3, x[x.week <= 3].resid, 1)[0]
            r[m] = round(float(s_op / s_in), 2) if s_in else None
        res[tag] = r; print(f"  {tag}: {r}")
    OUT["offseason_ratio"] = res


def role_compare():
    print("\n=== 5. Role-cascade coefficients — 3y vs 5y ===")
    import role_multiplier as rm
    res = {}
    for tag, path, seasons in (("3y", PBP3, (2023, 2024, 2025)), ("5y", PBP5, (2021, 2022, 2023, 2024, 2025))):
        orig = rm.DATA
        pbp = pd.read_parquet(path)
        # measure() reads the 3y file by name; temporarily swap
        pd_read = pd.read_parquet
        pd.read_parquet = (lambda p, *a, **k: pbp if "pbp_2023_2025" in str(p) else pd_read(p, *a, **k))
        try:
            m = rm.measure(seasons)
        finally:
            pd.read_parquet = pd_read
        r = {k: {"n": v["n_team_seasons"], "RB2": v["pickup"].get("RB2"), "RB3": v["pickup"].get("RB3"),
                 "WR2": v["pickup"].get("WR2"), "WR3": v["pickup"].get("WR3"), "TE1": v["pickup"].get("TE1")} for k, v in m.items()}
        res[tag] = r; print(f"  {tag}: {json.dumps(r)}")
    OUT["role_cascade"] = res


def corr_compare():
    print("\n=== 6. Correlations (same-game) — 3y vs 5y ===")
    res = {}
    for tag, path, first in (("3y", PG3, 2023), ("5y", PG5, 2021)):
        d = pd.read_parquet(path); d = d[d.season >= first]
        # QB pass_yds vs each WR rec_yds on the same team-game, residualized by player mean
        d["z"] = d.groupby(["pid", "mkt"])["y"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
        qb = d[d.mkt == "pass_yds"][["game_id", "posteam", "z"]].rename(columns={"z": "zq"})
        wr = d[d.mkt == "rec_yds"][["game_id", "posteam", "pid", "z"]]
        j = wr.merge(qb, on=["game_id", "posteam"])
        qbwr = float(np.corrcoef(j.z, j.zq)[0, 1])
        # two WRs same game
        w2 = wr.merge(wr, on=["game_id", "posteam"]); w2 = w2[w2.pid_x < w2.pid_y]
        wrwr = float(np.corrcoef(w2.z_x, w2.z_y)[0, 1])
        res[tag] = {"qb_wr": round(qbwr, 3), "wr_wr": round(wrwr, 3), "n_pairs": int(len(j))}
        print(f"  {tag}: QB-WR {res[tag]['qb_wr']}  WR-WR {res[tag]['wr_wr']}  (n={res[tag]['n_pairs']})")
    OUT["correlations"] = res


if __name__ == "__main__":
    multifactor_compare(); xtd_compare(); cv_compare(); offseason_ratio_compare(); role_compare(); corr_compare()
    json.dump(OUT, open(f"{DATA}/history_compare.json", "w"), indent=1)
    print("\nsaved data/history_compare.json")
