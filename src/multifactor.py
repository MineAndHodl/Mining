"""The multi-factor prop model — the real alpha is WEIGHTING many signals together and letting
the data decide what's valuable, not betting on any single factor.

We predict the RESIDUAL (actual - the player's own baseline, i.e. what a season-anchored line
would sit at) from a feature set, fit walk-forward so weights are earned out-of-sample:

  form3      recency: recent-3 form vs baseline        (found strongly predictive)
  form1      last-game momentum vs baseline
  def_delta  opponent defense-vs-position, prior year   (found ~0 on its own — let the fit judge)
  total_z    game total vs league (environment)
  home       home/away
  base_z     the level itself (mean-reversion / role scaling)

The fitted coefficients ARE the answer to "what's valuable": recency should earn a big weight,
defense near zero, etc. The projection = baseline + predicted residual, and the edge = that vs
the market line. Re-fit as data arrives — this is meant to keep iterating.
"""
import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
FEATURES = ["form3", "form1", "def_delta", "total_z", "home", "base_z"]


def _defense_maps():
    out = {}
    for pos, f in (("RB", "def_vs_rb.parquet"), ("WR", "def_vs_wr.parquet")):
        try:
            d = pd.read_parquet(f"{DATA}/{f}")
            out[pos] = {(r["defteam"], int(r["season"]) + 1): float(r["vs_lg"]) - 1.0 for _, r in d.iterrows()}
        except Exception:
            out[pos] = {}
    return out


def build_features(pg=None):
    pg = pg if pg is not None else pd.read_parquet(f"{DATA}/prop_player_games.parquet")
    pg = pg.sort_values(["pid", "mkt", "season", "week"]).copy()
    g = pg.groupby(["pid", "mkt"])
    pg["baseline"] = g["y"].transform(lambda s: s.shift(1).expanding(min_periods=4).mean())
    pg["last3"] = g["y"].transform(lambda s: s.shift(1).rolling(3).mean())
    pg["last1"] = g["y"].transform(lambda s: s.shift(1))
    pg["resid"] = pg["y"] - pg["baseline"]
    pg["form3"] = pg["last3"] - pg["baseline"]
    pg["form1"] = pg["last1"] - pg["baseline"]
    # opponent + defense
    def opp(r):
        p = r["game_id"].split("_")
        if len(p) < 4: return None
        a, h = p[2], p[3]
        return h if r["posteam"] == a else a
    pg["opp"] = pg.apply(opp, axis=1)
    pg["home"] = pg.apply(lambda r: 1.0 if r["game_id"].split("_")[-1] == r["posteam"] else 0.0, axis=1)
    dmap = _defense_maps()
    def dd(r):
        pos = "RB" if r["mkt"] in ("rush_yds", "rush_att") else ("WR" if r["mkt"] in ("rec_yds", "receptions") else None)
        return dmap.get(pos, {}).get((r["opp"], r["season"])) if pos else None
    pg["def_delta"] = pg.apply(dd, axis=1).fillna(0.0)
    # game total (environment)
    try:
        sch = pd.read_parquet(f"{DATA}/schedules.parquet")
        tmap = {(int(r["season"]), int(r["week"]), r["home_team"]): float(r["total_line"])
                for _, r in sch.iterrows() if pd.notna(r.get("total_line"))}
        def tot(r):
            p = r["game_id"].split("_"); home = p[3] if len(p) >= 4 else None
            return tmap.get((r["season"], r["week"], home))
        pg["total"] = pg.apply(tot, axis=1)
        pg["total_z"] = ((pg["total"] - pg["total"].mean()) / pg["total"].std()).fillna(0.0)
    except Exception:
        pg["total_z"] = 0.0
    # RAW features (board-computable at predict time) with simple constant centering so no stored
    # std is needed: total centered at 44, baseline centered at its per-market median.
    pg["total_c"] = (pg.get("total", pd.Series(44.0, index=pg.index)).fillna(44.0) - 44.0)
    pg["base_c"] = 0.0
    BASE_MED = {}
    for mkt in pg["mkt"].unique():
        m = pg["mkt"] == mkt
        med = float(pg.loc[m, "baseline"].median())
        BASE_MED[mkt] = med
        pg.loc[m, "base_c"] = pg.loc[m, "baseline"] - med
    pg.attrs["base_med"] = BASE_MED
    return pg.dropna(subset=["resid", "form3", "form1", "baseline"])


# raw feature set used both to fit and to predict live
RAW_FEATURES = ["form3", "form1", "def_delta", "total_c", "home", "base_c"]

# MEASURED (build_features, 2023-25): slope of resid on form3 for opening weeks 1-3 (where the
# form window spans the offseason) as a fraction of the in-season slope. Recent form still
# predicts across the gap, but at roughly half strength — and worst for rush/pass. Applied to
# form3/form1 at predict time whenever the recent log is last season's (recency_week is None).
#   in-season slope: rec .235 rush .353 pass .394 rec'ns .317 | opening: .153 .141 .140 .168
OFFSEASON_FORM_RATIO = {"rec_yds": 0.65, "rush_yds": 0.40, "pass_yds": 0.35, "receptions": 0.53}
OFFSEASON_FORM_DEFAULT = 0.50


def fit_and_save(through_season=2025, l2=10.0, path=None):
    """Fit per-market weights on ALL data through `through_season` (raw features) and save them
    so the live board can predict: residual = w . features, projection = baseline + residual."""
    import json
    path = path or os.path.join(DATA, "multifactor_weights.json")
    d = build_features()
    d = d[d.season <= through_season]
    base_med = d.attrs.get("base_med", {})
    out = {"features": RAW_FEATURES, "base_med": base_med, "weights": {}}
    for mkt in ["rush_yds", "rec_yds", "receptions", "pass_yds", "rush_att"]:
        s = d[d.mkt == mkt]
        if len(s) < 300:
            continue
        X = s[RAW_FEATURES].to_numpy(float); y = s["resid"].to_numpy(float)
        w = np.linalg.solve(X.T @ X + l2 * np.eye(X.shape[1]), X.T @ y)
        out["weights"][mkt] = {f: float(wi) for f, wi in zip(RAW_FEATURES, w)}
    json.dump(out, open(path, "w"), indent=1)
    return out


def predict_residual(mkt, feats, model):
    """feats: dict with form3, form1, def_delta, total_c, home, base_c. Returns predicted residual
    (add to baseline for the projection)."""
    w = model["weights"].get(mkt)
    if not w:
        return 0.0
    return float(sum(w[f] * float(feats.get(f, 0.0) or 0.0) for f in model["features"]))


def fit_walk_forward(pg=None, test_season=2025, l2=1.0):
    """Fit the multi-factor weights on seasons < test, evaluate out-of-sample on test_season.
    Returns earned coefficients and how the ensemble compares to recency-alone and the baseline."""
    d = build_features(pg)
    results = {}
    for mkt in ["rush_yds", "rec_yds", "receptions", "pass_yds"]:
        s = d[d.mkt == mkt]
        tr, te = s[s.season < test_season], s[s.season == test_season]
        if len(tr) < 300 or len(te) < 100:
            continue
        X = tr[FEATURES].to_numpy(float); y = tr["resid"].to_numpy(float)
        # ridge closed form
        A = X.T @ X + l2 * np.eye(X.shape[1]); w = np.linalg.solve(A, X.T @ y)
        Xte = te[FEATURES].to_numpy(float); pred = Xte @ w
        # out-of-sample: does the ensemble prediction's SIGN beat the baseline as a line?
        actual_resid = te["resid"].to_numpy(float)
        strong = np.abs(pred) >= np.quantile(np.abs(pred), 0.6)
        hit_ens = (np.sign(pred[strong]) == np.sign(actual_resid[strong])).mean()
        # recency-alone (form3 only) for comparison
        hit_rec = (np.sign(te["form3"]) == np.sign(actual_resid)).mean()
        results[mkt] = {"weights": {f: round(float(wi), 2) for f, wi in zip(FEATURES, w)},
                        "ens_hit_strong": round(float(hit_ens), 3), "recency_hit": round(float(hit_rec), 3),
                        "n_test": int(len(te))}
    return results


if __name__ == "__main__":
    import json
    r = fit_walk_forward()
    print("=== EARNED multi-factor weights (test season 2025, out-of-sample) ===")
    for mkt, v in r.items():
        print(f"\n{mkt} (n_test={v['n_test']}):")
        print("  weights:", v["weights"])
        print(f"  ensemble strong-signal hit: {v['ens_hit_strong']}  vs recency-alone: {v['recency_hit']}")
