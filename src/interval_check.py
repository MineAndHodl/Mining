"""Are our ranges honest? The bridge shows a 25-75% "likely range" and a 10-90% "wide range" on
every number. This measures, out of sample, how often the actual stat landed inside them.

Walk-forward: ridge weights fit on seasons < test, our number = baseline + w·features for every
player-game in the test season, the range from the same distribution the pricing uses
(props_model.cv_for / DIST — the volume-tier CVs measured within player-season). A calibrated
25-75 band catches ~50% of outcomes; 10-90 catches ~80%. Reported per market and pooled, for
2024 and 2025 held out, to data/interval_coverage.json — the page states the pooled numbers.

    python3 src/interval_check.py
"""
import os, json, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, HERE)
import multifactor as mf
import bridge as br


def run(test_seasons=(2024, 2025), l2=10.0):
    d = mf.build_features()
    out = {"bands": {"likely": [25, 75], "wide": [10, 90]}, "by_market": {}, "pooled": {}}
    pooled = {"n": 0, "in_likely": 0, "in_wide": 0, "below_p10": 0, "above_p90": 0}
    for mkt in ["rush_yds", "rec_yds", "receptions", "pass_yds"]:
        s = d[d.mkt == mkt]
        agg = {"n": 0, "in_likely": 0, "in_wide": 0, "below_p10": 0, "above_p90": 0}
        for ts in test_seasons:
            tr, te = s[s.season < ts], s[s.season == ts]
            if len(tr) < 300 or len(te) < 100:
                continue
            X = tr[mf.RAW_FEATURES].to_numpy(float); y = tr["resid"].to_numpy(float)
            w = np.linalg.solve(X.T @ X + l2 * np.eye(X.shape[1]), X.T @ y)
            mu = te["baseline"].to_numpy(float) + te[mf.RAW_FEATURES].to_numpy(float) @ w
            act = te["y"].to_numpy(float)
            for m_, a_ in zip(mu, act):
                if not np.isfinite(m_) or not np.isfinite(a_) or m_ <= 0:
                    continue
                q = br.quantiles(mkt, m_)
                agg["n"] += 1
                agg["in_likely"] += int(q["p25"] <= a_ <= q["p75"])
                agg["in_wide"] += int(q["p10"] <= a_ <= q["p90"])
                agg["below_p10"] += int(a_ < q["p10"]); agg["above_p90"] += int(a_ > q["p90"])
        if agg["n"]:
            out["by_market"][mkt] = {"n": agg["n"], "likely": round(agg["in_likely"] / agg["n"], 3), "wide": round(agg["in_wide"] / agg["n"], 3),
                                     "below_p10": round(agg["below_p10"] / agg["n"], 3), "above_p90": round(agg["above_p90"] / agg["n"], 3)}
            for k in pooled: pooled[k] += agg[k]
    if pooled["n"]:
        out["pooled"] = {"n": pooled["n"], "likely": round(pooled["in_likely"] / pooled["n"], 3), "wide": round(pooled["in_wide"] / pooled["n"], 3),
                         "below_p10": round(pooled["below_p10"] / pooled["n"], 3), "above_p90": round(pooled["above_p90"] / pooled["n"], 3),
                         "seasons": list(test_seasons), "note": "walk-forward: weights fit on prior seasons; ranges from the pricing distribution"}
    json.dump(out, open(os.path.join(DATA, "interval_coverage.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=1))
