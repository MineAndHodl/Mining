"""Walk-forward backtest vs closing lines. No leakage: each test season's
margin/total regressions are fit only on prior seasons' features (which were
themselves built from strictly-prior games)."""
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy.stats import norm

SIGMA_M = 13.2   # sd of (margin - prediction)
JUICE = -110

def fit_predict(train, test):
    # margin model: hfa*(1-neutral) + net_diff + rest_diff
    def X_m(d):
        return np.column_stack([(1 - d["neutral"]), d["net_diff"], d["rest_diff"]])
    ym = train["result"].values
    bm, *_ = lstsq(X_m(train), ym, rcond=None)
    test = test.copy()
    test["pred_margin"] = X_m(test) @ bm

    # total model: intercept + off_sum + def_sum + dome + wind
    def X_t(d):
        return np.column_stack([np.ones(len(d)), d["off_sum"], d["def_sum"],
                                d["dome"], d["wind"]])
    yt = train["total_pts"].values
    bt, *_ = lstsq(X_t(train), yt, rcond=None)
    test["pred_total"] = X_t(test) @ bt
    return test, bm, bt


def american_profit(odds, won, stake=1.0):
    if not won:
        return -stake
    return stake * (odds / 100.0 if odds > 0 else 100.0 / abs(odds))


def devig_2way(p1_raw, p2_raw):
    s = p1_raw + p2_raw
    return p1_raw / s, p2_raw / s


def implied(odds):
    return 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)


def run(features_path, test_seasons, min_week=1):
    f = pd.read_parquet(features_path).dropna(subset=["result", "spread_line", "total_line"])
    all_test = []
    for yr in test_seasons:
        train = f[f["season"] < yr]
        test = f[(f["season"] == yr) & (f["week"] >= min_week)]
        if len(train) < 400 or test.empty:
            continue
        t, bm, bt = fit_predict(train, test)
        all_test.append(t)
    t = pd.concat(all_test)

    # ---- spreads: model says home covers if pred_margin > spread_line
    t["edge_sp"] = t["pred_margin"] - t["spread_line"]
    t["home_cover"] = np.sign(t["result"] - t["spread_line"])  # 0 = push
    # ---- totals
    t["edge_tot"] = t["pred_total"] - t["total_line"]
    t["over_hit"] = np.sign(t["total_pts"] - t["total_line"])
    # ---- moneyline
    t["p_home"] = norm.cdf(t["pred_margin"] / SIGMA_M)
    return t


def summarize_threshold(t, edge_col, outcome_col, thresholds):
    rows = []
    for th in thresholds:
        sel = t[np.abs(t[edge_col]) >= th]
        side = np.sign(sel[edge_col])
        res = sel[outcome_col] * side          # +1 win, -1 loss, 0 push
        w, l, p = (res > 0).sum(), (res < 0).sum(), (res == 0).sum()
        n = w + l
        units = w * (100 / 110) - l * 1.0
        rows.append({"thresh": th, "bets": int(w + l + p), "W": int(w), "L": int(l),
                     "push": int(p), "win%": round(100 * w / n, 2) if n else np.nan,
                     "units": round(units, 1),
                     "roi%": round(100 * units / (w + l + p), 2) if n else np.nan})
    return pd.DataFrame(rows)


def summarize_ml(t, edge_thresholds):
    t = t.dropna(subset=["home_ml", "away_ml"]).copy()
    ph_mkt, pa_mkt = devig_2way(t["home_ml"].map(implied), t["away_ml"].map(implied))
    t["ml_edge_home"] = t["p_home"] - ph_mkt
    rows = []
    for th in edge_thresholds:
        profit, n, wins = 0.0, 0, 0
        for _, g in t.iterrows():
            e = g["ml_edge_home"]
            if e >= th:      # bet home
                won = g["result"] > 0
                profit += american_profit(g["home_ml"], won); n += 1; wins += won
            elif e <= -th:   # bet away
                won = g["result"] < 0
                profit += american_profit(g["away_ml"], won); n += 1; wins += won
        rows.append({"edge_min": th, "bets": n, "win%": round(100 * wins / n, 2) if n else np.nan,
                     "units": round(profit, 1), "roi%": round(100 * profit / n, 2) if n else np.nan})
    return pd.DataFrame(rows)
