"""Walk-forward validation: does the decomposed xTD model predict who scores better than the
old pooled TD model? Learn from season S, predict every player-game in season S+1, score
against reality. Both models get the SAME environment (none) so the test isolates the
quality of the player projection itself. Adopt xTD only if it wins — empirical-first.

Metrics per fold:
  brier      mean (p - y)^2 over player-games (lower = better)
  logloss    (lower = better)
  season_corr corr(predicted expected TDs x games, actual season TDs)
  calibration hit rate by predicted-probability bucket (should track the bucket)
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xtd_model as xm
import td_model as tdm

DATA = xm.DATA


def realized_player_games(ev):
    """Per (pid, game): did he score (any rush/rec TD)?"""
    g = ev.groupby(["pid", "game_id", "posteam"])["td"].sum().reset_index()
    g["scored"] = (g["td"] > 0).astype(int)
    return g


def old_model_probs(pbp_prior, players):
    frames = [pbp_prior[["game_id", "posteam", "yardline_100", "rusher_player_id", "receiver_player_id", "td_player_id"]]]
    rates = tdm.player_rates(frames, players)
    return dict(zip(rates["pid"], 1 - np.exp(-rates["lam_base"])))


def fold(train_season, test_season):
    pbp = xm.load_pbp(); players = xm.load_players()
    # --- xTD from the prior season, week-1 style (no current-season data), env = 1
    xdf = xm.build_xtd([train_season], target_season=test_season, week1_mode=True)
    x_p = dict(zip(xdf["pid"], xdf["p_atd_raw"]))
    x_lam = dict(zip(xdf["pid"], xdf["xtd"]))
    # --- old model from the same prior season
    o_p = old_model_probs(pbp[pbp.season == train_season], players)
    # --- reality in the test season
    ev_test = xm._events(pbp[pbp.season == test_season])
    pg = realized_player_games(ev_test)
    pg = pg[pg["pid"].isin(set(x_p) & set(o_p))]         # players both models can price
    # only players with a real role (>=4 games with a touch) — the ones who get ATD lines
    n_games = pg.groupby("pid")["game_id"].nunique()
    pg = pg[pg["pid"].map(n_games) >= 4]
    pg["p_x"] = pg["pid"].map(x_p); pg["p_o"] = pg["pid"].map(o_p)
    y = pg["scored"].values
    out = {"train": train_season, "test": test_season, "n_player_games": len(pg)}
    for tag, col in (("xtd", "p_x"), ("old", "p_o")):
        p = np.clip(pg[col].values, 0.01, 0.99)
        out[f"{tag}_brier"] = round(float(np.mean((p - y) ** 2)), 4)
        out[f"{tag}_logloss"] = round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4)
    # season-level: expected TDs vs actual
    season_tds = ev_test.groupby("pid")["td"].sum()
    games = ev_test.groupby("pid")["game_id"].nunique()
    common = [p for p in season_tds.index if p in x_lam and p in o_p and games[p] >= 4]
    act = np.array([season_tds[p] for p in common])
    xs = np.array([x_lam[p] * games[p] for p in common])
    os_ = np.array([-np.log(1 - o_p[p]) * games[p] for p in common])
    out["xtd_season_corr"] = round(float(np.corrcoef(xs, act)[0, 1]), 3)
    out["old_season_corr"] = round(float(np.corrcoef(os_, act)[0, 1]), 3)
    # calibration buckets for xTD
    pg["bucket"] = pd.cut(pg["p_x"], [0, .15, .25, .35, .45, .55, .7, 1.0])
    cal = pg.groupby("bucket", observed=True).agg(pred=("p_x", "mean"), hit=("scored", "mean"), n=("scored", "size"))
    return out, cal


if __name__ == "__main__":
    results = []
    for tr, te in ((2023, 2024), (2024, 2025)):
        r, cal = fold(tr, te)
        results.append(r)
        print(f"\n=== train {tr} -> test {te}  ({r['n_player_games']} player-games) ===")
        print(f"  Brier    xTD {r['xtd_brier']}  vs old {r['old_brier']}   {'xTD WINS' if r['xtd_brier'] < r['old_brier'] else 'old wins'}")
        print(f"  LogLoss  xTD {r['xtd_logloss']}  vs old {r['old_logloss']}   {'xTD WINS' if r['xtd_logloss'] < r['old_logloss'] else 'old wins'}")
        print(f"  Season-TD corr  xTD {r['xtd_season_corr']}  vs old {r['old_season_corr']}   {'xTD WINS' if r['xtd_season_corr'] > r['old_season_corr'] else 'old wins'}")
        print("  xTD calibration (pred vs realized hit rate):")
        print(cal.round(3).to_string())
    pd.DataFrame(results).to_json(f"{DATA}/xtd_validation.json", orient="records", indent=1)


# =========================== fair (calibrated) + in-season + blend tests ===========================
def _linfit(p, y):
    """Fit y ~ a + b*p by least squares (the same recalibration form the old model uses)."""
    A = np.vstack([np.ones_like(p), p]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


def _score(p, y):
    p = np.clip(p, 0.01, 0.99)
    return (round(float(np.mean((p - y) ** 2)), 4),
            round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4))


def calibrated_and_blend():
    """Fit recalibration (and blend weight) on 2023->2024, evaluate on 2024->2025. Fair test."""
    pbp = xm.load_pbp(); players = xm.load_players()
    def fold_frame(tr, te):
        xdf = xm.build_xtd([tr], target_season=te, week1_mode=True)
        x_p = dict(zip(xdf["pid"], xdf["p_atd_raw"]))
        o_p = old_model_probs(pbp[pbp.season == tr], players)
        ev = xm._events(pbp[pbp.season == te]); pg = realized_player_games(ev)
        pg = pg[pg["pid"].isin(set(x_p) & set(o_p))]
        ng = pg.groupby("pid")["game_id"].nunique(); pg = pg[pg["pid"].map(ng) >= 4]
        pg["p_x"] = pg["pid"].map(x_p); pg["p_o"] = pg["pid"].map(o_p)
        return pg
    fit = fold_frame(2023, 2024); test = fold_frame(2024, 2025)
    yf = fit["scored"].values; yt = test["scored"].values
    ax, bx = _linfit(fit["p_x"].values, yf); ao, bo = _linfit(fit["p_o"].values, yf)
    # blend weight learned on the fit fold (grid)
    best_w, best_b = 0.5, 9
    for w in np.linspace(0, 1, 21):
        pb = w * (ax + bx * fit["p_x"].values) + (1 - w) * (ao + bo * fit["p_o"].values)
        b, _ = _score(pb, yf)
        if b < best_b: best_b, best_w = b, w
    print("\n=== FAIR TEST: recalibrate on 2023->24, evaluate on 2024->25 (week-1 mode) ===")
    print(f"  recal xTD: a={ax:.3f} b={bx:.3f} | recal old: a={ao:.3f} b={bo:.3f} | learned blend w_xtd={best_w:.2f}")
    px = ax + bx * test["p_x"].values; po = ao + bo * test["p_o"].values
    pbld = best_w * px + (1 - best_w) * po
    for tag, p in (("xTD (calibrated)", px), ("old (calibrated)", po), ("BLEND", pbld)):
        b, l = _score(p, yt); print(f"  {tag:18s} brier={b}  logloss={l}")
    return best_w, (ax, bx), (ao, bo)


def in_season_rolling(season=2025, min_week=4):
    """The real use case from Week 2 on: shares from THIS season's weeks 1..k-1 predict week k.
    Compares the decomposed model (zone shares update fast) vs the old pooled model (same data)."""
    pbp = xm.load_pbp(); players = xm.load_players()
    ev = xm._events(pbp[pbp.season == season])
    rows = []
    for wk in range(min_week, 19):
        past = pbp[(pbp.season == season) & (pbp.week < wk)]
        if past.empty: continue
        # decomposed, in-season mode (no GL yoy shrink), env=1
        xdf = xm.build_xtd([season], target_season=season, week1_mode=False) if False else None
        # build_xtd learns from whole-season pbp by default; restrict via a temporary filter
        ev_past = xm._events(past)
        tzp = xm.team_zone_profile(ev_past, season); lg = xm.league_zone_profile(tzp)
        conv = xm.conversion_priors(xm._events(pbp[pbp.season < season]), players)   # priors from prior seasons only
        sh = xm.player_zone_shares(ev_past).merge(players[["gsis_id", "position"]], left_on="pid", right_on="gsis_id", how="left")
        sh = sh[sh.position.isin(xm.POS_OK)]
        x_p = {}
        for (pid, team), g in sh.groupby(["pid", "posteam"]):
            pos = g["position"].iloc[0]; games = float(g["games"].iloc[0]); lam = 0.0
            for zone in xm.ZONES:
                t = tzp[(tzp.posteam == team) & (tzp.zone == zone)]
                ppg = float(t["plays_pg"].iloc[0]) if len(t) else float(lg.loc[zone, "plays_pg"])
                pr = float(t["pass_rate"].iloc[0]) if len(t) else float(lg.loc[zone, "pass_rate"])
                ppg = 0.7 * ppg + 0.3 * float(lg.loc[zone, "plays_pg"])
                for typ in xm.TYPES:
                    r = g[(g.zone == zone) & (g.type == typ)]
                    n = int(r["n"].iloc[0]) if len(r) else 0; tds = int(r["tds"].iloc[0]) if len(r) else 0
                    so = float(r["share"].iloc[0]) if len(r) else 0.0
                    own = g[g.type == typ]; own_s = float(own["n"].sum() / max(own["team_n"].sum(), 1)) if len(own) else 0.0
                    w = games / (games + xm.SHARE_PSEUDO_GAMES); share = w * so + (1 - w) * own_s
                    prior = conv.get((pos, zone, typ), 0.02)
                    c = (tds + xm.CONV_PSEUDO_OPPS * prior) / (n + xm.CONV_PSEUDO_OPPS)
                    lam += ppg * (pr if typ == "target" else 1 - pr) * share * c
            x_p[pid] = 1 - np.exp(-lam)
        o_p = old_model_probs(past, players)
        this = realized_player_games(ev[ev.week == wk]) if "week" in ev.columns else None
        this = realized_player_games(xm._events(pbp[(pbp.season == season) & (pbp.week == wk)]))
        this = this[this["pid"].isin(set(x_p) & set(o_p))]
        this["p_x"] = this["pid"].map(x_p); this["p_o"] = this["pid"].map(o_p)
        rows.append(this)
    pg = pd.concat(rows, ignore_index=True); y = pg["scored"].values
    print(f"\n=== IN-SEASON ROLLING {season} (weeks {min_week}-18, {len(pg)} player-games): this season's shares predict next week ===")
    for tag, col in (("xTD (decomposed)", "p_x"), ("old (pooled)", "p_o")):
        b, l = _score(pg[col].values, y); print(f"  {tag:18s} brier={b}  logloss={l}")
    bx, lx = _score(pg["p_x"].values, y); bo, lo = _score(pg["p_o"].values, y)
    print(f"  -> {'xTD WINS' if bx < bo else 'old wins'} on Brier in-season")
    return pg


if __name__ == "__main__" and "--deep" in sys.argv:
    calibrated_and_blend()
    in_season_rolling(2025)
    in_season_rolling(2024)
