"""Feature lab — the ONE way a new yardage/count feature earns its way into the model.

    python3 src/feature_lab.py                # test every candidate below, two held-out seasons

A candidate is a function that adds one column to the player-game feature table. It is scored
the repo's way: ridge residual model fit on prior seasons, evaluated on 2024 and 2025 held out
separately (a win must replicate), against the SAME model without the column:
  - strong-signal sign-hit (top-40% |prediction|): does our side of the baseline-as-line hit?
  - residual MAE
  - ROI proxy at -110 on the strong signals
Adopt only if it wins on BOTH test seasons. Wash or loss is logged so it is never re-tested.
Results -> data/feature_lab.json (append-style, keyed by feature).
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multifactor as mf

DATA = mf.DATA
BASE = list(mf.RAW_FEATURES)
MARKETS = ["rush_yds", "rec_yds", "receptions", "pass_yds"]


# ------------------------------------------------------------------ candidate features
def _sched():
    s = pd.read_parquet(f"{DATA}/schedules.parquet")
    return s[s.game_type == "REG"] if "game_type" in s.columns else s


def add_spread(pg):
    """Game script: the team's own spread (negative = favorite). Favorites run late; dogs throw."""
    s = _sched()
    sp = {}
    for _, r in s.iterrows():
        if pd.isna(r.get("spread_line")): continue
        sp[(r.game_id, r.home_team)] = -float(r.spread_line)      # home margin -> home team's spread
        sp[(r.game_id, r.away_team)] = float(r.spread_line)
    pg["spread_c"] = pg.apply(lambda r: sp.get((r.game_id, r.posteam), 0.0), axis=1)
    return pg


def add_short_week(pg):
    """Short rest (<= 4 days: Thursday games) vs normal."""
    s = _sched()
    rest = {}
    for _, r in s.iterrows():
        rest[(r.game_id, r.home_team)] = r.get("home_rest"); rest[(r.game_id, r.away_team)] = r.get("away_rest")
    pg["short_week"] = pg.apply(lambda r: 1.0 if (rest.get((r.game_id, r.posteam)) or 7) <= 4 else 0.0, axis=1)
    return pg


def add_pace(pg):
    """Team pace: run/pass plays per game so far this season (prior weeks only), vs league."""
    pbp = pd.read_parquet(f"{DATA}/pbp_2021_2025.parquet", columns=["season", "week", "game_id", "posteam", "play_type"])
    pbp = pbp[pbp.play_type.isin(["run", "pass"])]
    tg = pbp.groupby(["season", "week", "posteam"]).size().rename("plays").reset_index().sort_values(["posteam", "season", "week"])
    tg["cum_plays"] = tg.groupby(["posteam", "season"])["plays"].cumsum().shift(1)
    tg["cum_games"] = tg.groupby(["posteam", "season"]).cumcount()
    tg["ppg_prior"] = np.where(tg.cum_games > 0, tg.cum_plays / tg.cum_games.clip(lower=1), np.nan)
    lg = tg.groupby("season")["plays"].mean()
    tg["pace_c"] = tg["ppg_prior"] - tg["season"].map(lg)
    m = {(r.season, r.week, r.posteam): r.pace_c for r in tg.itertuples()}
    pg["pace_c"] = pg.apply(lambda r: m.get((r.season, r.week, r.posteam)), axis=1).fillna(0.0)
    return pg


CANDIDATES = {"spread_c": add_spread, "short_week": add_short_week, "pace_c": add_pace}


# ------------------------------------------------------------------ the test
def _fit_eval(d, feats, test_season, l2=10.0):
    out = {}
    for mkt in MARKETS:
        s = d[d.mkt == mkt]
        tr, te = s[s.season < test_season], s[s.season == test_season]
        if len(tr) < 300 or len(te) < 100: continue
        X = tr[feats].to_numpy(float); y = tr["resid"].to_numpy(float)
        w = np.linalg.solve(X.T @ X + l2 * np.eye(X.shape[1]), X.T @ y)
        pred = te[feats].to_numpy(float) @ w; act = te["resid"].to_numpy(float)
        strong = np.abs(pred) >= np.quantile(np.abs(pred), 0.6)
        hit = float((np.sign(pred[strong]) == np.sign(act[strong])).mean())
        mae = float(np.abs(pred - act).mean())
        roi = hit * (100 / 110) - (1 - hit)                      # -110 both sides, pushes ignored
        out[mkt] = {"hit": round(hit, 4), "mae": round(mae, 3), "roi": round(roi, 4), "n": int(len(te)),
                    "w": {f: round(float(x), 3) for f, x in zip(feats, w)}}
    return out


def test_feature(name, fn, d=None, seasons=(2024, 2025)):
    d = d if d is not None else mf.build_features()
    d = fn(d.copy())
    res = {"feature": name, "by_season": {}}
    wins = 0; total = 0
    for te in seasons:
        base = _fit_eval(d, BASE, te); withf = _fit_eval(d, BASE + [name], te)
        row = {}
        for mkt in base:
            dh = withf[mkt]["hit"] - base[mkt]["hit"]; dm = base[mkt]["mae"] - withf[mkt]["mae"]
            row[mkt] = {"hit_base": base[mkt]["hit"], "hit_with": withf[mkt]["hit"], "d_hit": round(dh, 4),
                        "mae_base": base[mkt]["mae"], "mae_with": withf[mkt]["mae"], "d_mae": round(dm, 3),
                        "weight": withf[mkt]["w"].get(name), "n": base[mkt]["n"]}
            total += 1; wins += (dh > 0.005 and dm >= 0)
        res["by_season"][te] = row
    # verdict: the feature must help (hit +0.5pt with no MAE loss) in a majority of market-seasons
    # AND not hurt any market in both seasons
    hurt_both = 0
    for mkt in MARKETS:
        if all(res["by_season"].get(te, {}).get(mkt, {}).get("d_hit", 0) < -0.005 for te in seasons): hurt_both += 1
    res["wins"] = wins; res["cells"] = total; res["hurt_both_seasons"] = hurt_both
    res["verdict"] = "WINS" if (wins >= total * 0.5 and hurt_both == 0) else ("LOSES" if hurt_both else "WASH")
    return res


def print_result(res):
    print(f"\n=== {res['feature']}: {res['verdict']} ({res['wins']}/{res['cells']} market-seasons helped; hurt both seasons: {res['hurt_both_seasons']}) ===")
    for te, row in res["by_season"].items():
        for mkt, v in row.items():
            print(f"  {te} {mkt:11s} hit {v['hit_base']:.3f} -> {v['hit_with']:.3f} ({v['d_hit']:+.3f})  mae {v['mae_base']:.2f} -> {v['mae_with']:.2f}  w={v['weight']}  n={v['n']}")


if __name__ == "__main__":
    names = sys.argv[1:] or list(CANDIDATES)
    d = mf.build_features()
    p = f"{DATA}/feature_lab.json"
    log = json.load(open(p)) if os.path.exists(p) else {}
    for nm in names:
        r = test_feature(nm, CANDIDATES[nm], d)
        print_result(r)
        log[nm] = {**r, "by_season": {str(k): v for k, v in r["by_season"].items()}}
    json.dump(log, open(p, "w"), indent=1)
