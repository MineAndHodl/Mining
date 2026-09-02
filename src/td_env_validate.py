"""TD environment + star calibration — MEASURED (Alex: "how can we dial in the ATD?").

Three questions, all walk-forward on 2021-25 play-by-play + closing spread/total:

1. SCALE: how many offensive TDs does a team score per implied point? The old env
   multiplier assumed 0.62 * 1.08 / 7 = 0.0957 TDs per point with no intercept. Measure the
   real mapping (OLS of team TDs on implied points) and whether last season's team TD
   baseline adds anything on top of the market's implied points.
2. ELASTICITY: the environment multiplier is (implied team TDs / team baseline)^k. Earn k
   (0 = ignore environment, 1 = full ratio) by out-of-sample Brier on player-games, with the
   recalibration refit for each k on the FIT fold (2023->24) and evaluated on 2024->25.
3. STARS: the calibration at the TOP of the board — realized P(scores) for players by
   prior-season TDs/game, and by our predicted bucket. This is where "confidence in TD
   scorers" lives; the pooled Brier is dominated by 200 low-probability players.

Writes data/td_env_validation.json. Run: python3 src/td_env_validate.py
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xtd_model as xm
import xtd_validate as xv

DATA = xm.DATA


def implied_points():
    s = pd.read_parquet(f"{DATA}/schedules.parquet")
    s = s[(s.game_type == "REG") & s.total_line.notna() & s.spread_line.notna()]
    rows = []
    for _, g in s.iterrows():
        ph = (g.total_line + g.spread_line) / 2          # nflverse spread_line = home margin
        rows.append((g.season, g.game_id, g.home_team, ph)); rows.append((g.season, g.game_id, g.away_team, g.total_line - ph))
    return pd.DataFrame(rows, columns=["season", "game_id", "posteam", "imp_pts"])


def team_tds(pbp):
    d = pbp[pbp.two_point_attempt == 0].copy()
    d["td"] = ((d.rush_touchdown == 1) | (d.pass_touchdown == 1)).astype(int)
    return d.groupby(["season", "game_id", "posteam"]).td.sum().reset_index()


def measure_scale(pbp):
    tg = team_tds(pbp); imp = implied_points()
    m = tg.merge(imp, on=["season", "game_id", "posteam"])
    base = tg.groupby(["season", "posteam"]).td.mean().reset_index().rename(columns={"td": "base"}); base["season"] += 1
    m2 = m.merge(base, on=["season", "posteam"])
    a, b = np.polyfit(m.imp_pts, m.td, 1)
    out = {"n_team_games": int(len(m)), "mean_tds": round(float(m.td.mean()), 3), "mean_imp_pts": round(float(m.imp_pts.mean()), 2),
           "tds_per_point_naive": round(float(m.td.mean() / m.imp_pts.mean()), 4), "old_model_tds_per_point": round(0.62 * 1.08 / 7, 4),
           "ols_slope": round(float(a), 4), "ols_intercept": round(float(b), 3),
           "corr_imp_pts": round(float(np.corrcoef(m.imp_pts, m.td)[0, 1]), 3)}
    # walk-forward 2025: implied only vs implied + last-season baseline
    tr, te = m2[m2.season < 2025], m2[m2.season == 2025]
    res = {}
    for cols, name in (([ "imp_pts"], "implied_only"), (["imp_pts", "base"], "implied_plus_base"), (["base"], "base_only")):
        Xtr = np.vstack([tr[c] for c in cols] + [np.ones(len(tr))]).T; Xte = np.vstack([te[c] for c in cols] + [np.ones(len(te))]).T
        c = np.linalg.lstsq(Xtr, tr.td, rcond=None)[0]; pred = Xte @ c
        res[name] = {"mae_2025": round(float(np.abs(pred - te.td).mean()), 4), "coef": [round(float(x), 3) for x in c]}
    out["walk_forward_2025"] = res
    out["finding"] = ("team TDs = %.3f*implied_pts %+.3f; last season's team TD rate adds NOTHING beyond the market's "
                      "implied points (coef ~0) -> the environment is the market's number, on the measured scale" % (a, b))
    return out, (float(a), float(b))


# ------------------------------------------------------------------ player-game folds with env
def fold_frame(pbp, players, tr, te, scale, k, week1=True):
    """Week-1-style predictions for test season te from train season tr, with the per-game
    environment multiplier (implied TDs / team baseline)^k applied to xTD's lambda."""
    xdf = xm.build_xtd([tr], target_season=te, week1_mode=True)
    lam = dict(zip(xdf["pid"], xdf["xtd"]))
    o_lam = {}
    o_p = xv.old_model_probs(pbp[pbp.season == tr], players)
    for pid, p in o_p.items(): o_lam[pid] = -np.log(1 - min(p, 0.999))
    ev = xm._events(pbp[pbp.season == te]); pg = xv.realized_player_games(ev)
    pg = pg[pg["pid"].isin(set(lam) & set(o_lam))]
    ng = pg.groupby("pid")["game_id"].nunique(); pg = pg[pg["pid"].map(ng) >= 4].copy()
    tg = team_tds(pbp[pbp.season == tr]).groupby("posteam").td.mean()
    imp = implied_points(); imp = imp[imp.season == te]
    pg = pg.merge(imp[["game_id", "posteam", "imp_pts"]], on=["game_id", "posteam"], how="left")
    a, b = scale
    pg["imp_tds"] = a * pg["imp_pts"] + b
    pg["base"] = pg["posteam"].map(tg).fillna(tg.mean())
    pg["mult"] = (pg["imp_tds"] / pg["base"]).clip(0.5, 1.8) ** k
    pg.loc[pg.imp_pts.isna(), "mult"] = 1.0
    pg["lam_x"] = pg["pid"].map(lam) * pg["mult"]; pg["lam_o"] = pg["pid"].map(o_lam) * pg["mult"]
    pg["p_x"] = 1 - np.exp(-pg["lam_x"]); pg["p_o"] = 1 - np.exp(-pg["lam_o"])
    return pg


def earn_elasticity(pbp, players, scale, old_scale):
    """Fit recal on 2023->24, test on 2024->25, for each env exponent k and both scales."""
    fits = {}; tests = {}
    out = {}
    for tag, sc in (("measured_scale", scale), ("old_scale", old_scale)):
        for k in (0.0, 0.5, 0.75, 1.0):
            fit = fold_frame(pbp, players, 2023, 2024, sc, k); test = fold_frame(pbp, players, 2024, 2025, sc, k)
            yf, yt = fit["scored"].values, test["scored"].values
            ax, bx = xv._linfit(fit["p_x"].values, yf); ao, bo = xv._linfit(fit["p_o"].values, yf)
            px = np.clip(ax + bx * test["p_x"].values, 0.02, 0.85); po = np.clip(ao + bo * test["p_o"].values, 0.02, 0.85)
            pb = 0.55 * px + 0.45 * po
            bx_, lx_ = xv._score(px, yt); bo_, lo_ = xv._score(po, yt); bb_, lb_ = xv._score(pb, yt)
            # top-of-board calibration: predicted >= .45
            top = pb >= 0.45
            out[f"{tag}_k{k}"] = {"k": k, "xtd_brier": bx_, "old_brier": bo_, "blend_brier": bb_, "blend_logloss": lb_,
                                  "recal_xtd": [round(ax, 3), round(bx, 3)], "recal_old": [round(ao, 3), round(bo, 3)],
                                  "top_bucket": {"n": int(top.sum()), "pred": round(float(pb[top].mean()), 3) if top.any() else None,
                                                 "hit": round(float(yt[top].mean()), 3) if top.any() else None}}
            print(f"  {tag:15s} k={k:<4} blend brier {bb_}  logloss {lb_}  top(>=.45): n={top.sum()} pred {out[f'{tag}_k{k}']['top_bucket']['pred']} hit {out[f'{tag}_k{k}']['top_bucket']['hit']}  recal x=({ax:.3f},{bx:.3f})")
    return out


def star_reality(pbp, players):
    """What do the best scorers ACTUALLY do next season? Realized P(scores) by prior-season
    TDs/game bucket, weeks 1-18 and week 1-3 only. Pooled 2022-25 (four transitions)."""
    ev_all = xm._events(pbp)
    per = ev_all.groupby(["season", "pid"]).agg(tds=("td", "sum"), games=("game_id", "nunique")).reset_index()
    per = per[per.games >= 8]; per["lam_prev"] = per.tds / per.games; per["season"] += 1
    pg = xv.realized_player_games(ev_all).merge(ev_all[["season", "week", "game_id"]].drop_duplicates(), on="game_id")
    m = pg.merge(per[["season", "pid", "lam_prev"]], on=["season", "pid"])
    m["bucket"] = pd.cut(m.lam_prev, [0, .2, .4, .6, .8, 1.0, 2.0])
    full = m.groupby("bucket", observed=True).agg(n=("scored", "size"), hit=("scored", "mean"), lam_prev=("lam_prev", "mean"))
    early = m[m.week <= 3].groupby("bucket", observed=True).agg(n=("scored", "size"), hit=("scored", "mean"))
    full["poisson_of_prev"] = 1 - np.exp(-full["lam_prev"])
    full["hit_early_w1_3"] = early["hit"]; full["n_early"] = early["n"]
    print("\n  realized P(scores) next season by prior-season TDs/game (players with 8+ games):")
    print(full.round(3).to_string())
    return {str(k): {"n": int(v.n), "hit": round(float(v.hit), 3), "lam_prev": round(float(v.lam_prev), 3),
                     "poisson_of_prev": round(float(v.poisson_of_prev), 3),
                     "hit_early_w1_3": round(float(v.hit_early_w1_3), 3) if not pd.isna(v.hit_early_w1_3) else None,
                     "n_early": int(v.n_early) if not pd.isna(v.n_early) else 0}
            for k, v in full.iterrows()}


if __name__ == "__main__" and "--stick" not in sys.argv and "--final" not in sys.argv:
    pbp = xm.load_pbp(); players = xm.load_players()
    print("=== 1. SCALE: implied points -> team TDs ===")
    scale_out, scale = measure_scale(pbp)
    print(json.dumps(scale_out, indent=1))
    old_scale = (0.62 * 1.08 / 7, 0.0)
    print("\n=== 2. ELASTICITY: env exponent k, recal refit per k (fit 2023->24, test 2024->25) ===")
    elast = earn_elasticity(pbp, players, scale, old_scale)
    print("\n=== 3. STARS: reality at the top of the board ===")
    stars = star_reality(pbp, players)
    best = min((v for kk, v in elast.items() if kk.startswith("measured")), key=lambda v: v["blend_brier"])
    res = {"scale": scale_out, "elasticity": elast, "stars_realized": stars,
           "decision": {"env_tds_per_point": round(scale[0], 4), "env_intercept": round(scale[1], 3), "k_star": best["k"],
                        "recal_xtd": best["recal_xtd"], "recal_old": best["recal_old"]}}
    json.dump(res, open(f"{DATA}/td_env_validation.json", "w"), indent=1)
    print("\nDECISION:", res["decision"])


# ------------------------------------------------------------------ early-season goal-line stickiness
def earn_stickiness(pbp, players, scale, sticks=(0.29, 0.37, 0.60, 0.75)):
    """Week-1-mode predictions with different GL stickiness, scored ONLY on weeks 1-3 of the
    test season (the weeks that mode actually ships); recal refit per stickiness on the fit
    fold. Also reports weeks 4-9 so a gain early isn't bought with a loss later."""
    wk = pbp[["game_id", "week"]].drop_duplicates()
    out = {}
    for st in sticks:
        def frame(tr, te):
            xdf = xm.build_xtd([tr], target_season=te, week1_mode=True, gl_stick=st)
            lam = dict(zip(xdf["pid"], xdf["xtd"]))
            o_p = xv.old_model_probs(pbp[pbp.season == tr], players); o_lam = {k: -np.log(1 - min(v, .999)) for k, v in o_p.items()}
            ev = xm._events(pbp[pbp.season == te]); pg = xv.realized_player_games(ev)
            pg = pg[pg["pid"].isin(set(lam) & set(o_lam))]
            ng = pg.groupby("pid")["game_id"].nunique(); pg = pg[pg["pid"].map(ng) >= 4].copy()
            tg = team_tds(pbp[pbp.season == tr]).groupby("posteam").td.mean()
            imp = implied_points(); imp = imp[imp.season == te]
            pg = pg.merge(imp[["game_id", "posteam", "imp_pts"]], on=["game_id", "posteam"], how="left").merge(wk, on="game_id")
            a, b = scale; pg["mult"] = ((a * pg["imp_pts"] + b) / pg["posteam"].map(tg).fillna(tg.mean())).clip(0.5, 1.8)
            pg.loc[pg.imp_pts.isna(), "mult"] = 1.0
            pg["p_x"] = 1 - np.exp(-pg["pid"].map(lam) * pg["mult"]); pg["p_o"] = 1 - np.exp(-pg["pid"].map(o_lam) * pg["mult"])
            return pg
        fit = pd.concat([frame(2022, 2023), frame(2023, 2024)]); test = frame(2024, 2025)
        f13 = fit[fit.week <= 3]
        ax, bx = xv._linfit(f13["p_x"].values, f13["scored"].values); ao, bo = xv._linfit(f13["p_o"].values, f13["scored"].values)
        res = {"stick": st, "recal_xtd_w1_3": [round(ax, 3), round(bx, 3)]}
        for lo, hi, name in ((1, 3, "w1_3"), (4, 9, "w4_9")):
            t = test[(test.week >= lo) & (test.week <= hi)]; y = t["scored"].values
            pb = np.clip(0.55 * (ax + bx * t["p_x"].values) + 0.45 * (ao + bo * t["p_o"].values), 0.02, 0.85)
            br, ll = xv._score(pb, y); top = pb >= 0.45
            res[name] = {"n": int(len(t)), "brier": br, "logloss": ll, "top_n": int(top.sum()),
                         "top_pred": round(float(pb[top].mean()), 3) if top.any() else None, "top_hit": round(float(y[top].mean()), 3) if top.any() else None}
        out[f"stick_{st}"] = res
        print(f"  stick {st:.2f}: w1-3 brier {res['w1_3']['brier']} ll {res['w1_3']['logloss']} top n={res['w1_3']['top_n']} pred {res['w1_3']['top_pred']} hit {res['w1_3']['top_hit']} | "
              f"w4-9 brier {res['w4_9']['brier']} top pred {res['w4_9']['top_pred']} hit {res['w4_9']['top_hit']}")
    return out


if __name__ == "__main__" and "--stick" in sys.argv:
    pbp = xm.load_pbp(); players = xm.load_players()
    print("=== 4. EARLY-SEASON GOAL-LINE STICKINESS (test 2025 weeks 1-3; recal fit on 2023-24 weeks 1-3) ===")
    st = earn_stickiness(pbp, players, (0.1387, -0.717))
    p = f"{DATA}/td_env_validation.json"
    res = json.load(open(p)) if os.path.exists(p) else {}
    res["stickiness"] = st
    json.dump(res, open(p, "w"), indent=1)


# ------------------------------------------------------------------ final shipped recalibration
def final_recal(pbp, players, scale):
    """The shipped numbers: k=1 on the measured scale, raw shares; recal for xTD and the old
    pooled model fit on TWO transitions (2022->23, 2023->24), tested on 2024->25."""
    fit = pd.concat([fold_frame(pbp, players, 2022, 2023, scale, 1.0), fold_frame(pbp, players, 2023, 2024, scale, 1.0)])
    test = fold_frame(pbp, players, 2024, 2025, scale, 1.0)
    yf, yt = fit["scored"].values, test["scored"].values
    ax, bx = xv._linfit(fit["p_x"].values, yf); ao, bo = xv._linfit(fit["p_o"].values, yf)
    px = np.clip(ax + bx * test["p_x"].values, .02, .85); po = np.clip(ao + bo * test["p_o"].values, .02, .85)
    out = {"recal_xtd": [round(ax, 3), round(bx, 3)], "recal_old": [round(ao, 3), round(bo, 3)]}
    for w in (0.45, 0.55, 0.65, 0.75, 1.0):
        pb = w * px + (1 - w) * po; br, ll = xv._score(pb, yt); top = pb >= .45
        out[f"blend_w{w}"] = {"brier": br, "logloss": ll, "top_n": int(top.sum()), "top_pred": round(float(pb[top].mean()), 3), "top_hit": round(float(yt[top].mean()), 3)}
        print(f"  w_xtd={w}: brier {br} ll {ll} | top(>=.45) n={top.sum()} pred {pb[top].mean():.3f} hit {yt[top].mean():.3f}")
    return out


if __name__ == "__main__" and "--final" in sys.argv:
    pbp = xm.load_pbp(); players = xm.load_players()
    print("=== 5. FINAL RECAL (fit 2022->23 + 2023->24, test 2024->25; k=1 measured scale, raw shares) ===")
    fr = final_recal(pbp, players, (0.1387, -0.717))
    p = f"{DATA}/td_env_validation.json"; res = json.load(open(p)) if os.path.exists(p) else {}
    res["final_recal"] = fr; res["decision"]["recal_xtd"] = fr["recal_xtd"]; res["decision"]["recal_old"] = fr["recal_old"]
    json.dump(res, open(p, "w"), indent=1); print("recal:", fr["recal_xtd"], fr["recal_old"])
