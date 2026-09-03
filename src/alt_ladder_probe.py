"""Alt-line ladder probe — does the book price the SHAPE of a player's distribution right?

We measured (5 seasons) that game-to-game spread falls with volume: a 75-yard back has CV
~.53, a 25-yard committee back ~.83. A book that prices every alternate line off one generic
shape mis-prices the ladder by tier: too wide on workhorses (their near-the-mean alt Overs
are too cheap), too narrow on low-volume players (their far alt Overs are too cheap).

Method: for each player-market with >= 5 DK alternate Over lines, fit the book's implied
distribution — a gamma (mean mu, CV c) times one free vig factor v (the ladder's hold) — to
the implied Over probabilities by least squares. Compare the fitted c to our measured
cv_for(mkt, mu) by volume tier. If the book's c is flat where ours falls, that is a structural
alt-line edge, and where it lives is the answer.

    ODDS_API_KEY=... python3 src/alt_ladder_probe.py [n_games]    (~4 credits per game)
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import gamma
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prop_board as pb
import props_model as pm
import primetime_sgp as ps

DATA = pb.__dict__.get("DATA") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def implied(o):
    o = float(o); return 100 / (o + 100) if o > 0 else abs(o) / (abs(o) + 100)


def fit_ladder(ladder):
    """ladder: [(line, over_price)] -> (mu, cv, vig, rmse, n)."""
    L = np.array([l for l, _ in ladder], float); P = np.array([implied(p) for _, p in ladder], float)
    if len(L) < 5 or L.std() == 0: return None
    # initial guess: mu at the ~50% rung, cv from the ladder width
    mu0 = float(np.interp(0.5, P[::-1], L[::-1])) if P.min() < 0.5 < P.max() else float(L.mean())
    def loss(x):
        mu, cv, v = x
        if mu <= 0 or cv <= 0.05 or v <= 0.5: return 1e9
        k = 1 / cv ** 2; th = mu / k
        pred = (1 - gamma.cdf(L, a=k, scale=th)) * v
        return float(((pred - P) ** 2).mean())
    best = None
    for cv0 in (0.4, 0.6, 0.9):
        r = minimize(loss, [max(mu0, 1.0), cv0, 1.15], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 2000})
        if best is None or r.fun < best.fun: best = r
    mu, cv, v = best.x
    return {"mu": round(float(mu), 1), "cv_book": round(float(cv), 3), "vig": round(float(v), 3),
            "rmse": round(float(np.sqrt(best.fun)), 4), "n_rungs": int(len(L)), "lo": float(L.min()), "hi": float(L.max())}


def run(n_games=3):
    key = os.environ.get("ODDS_API_KEY")
    events = pb.fetch_events(key, days=14)[:n_games]
    rows = []
    for e in events:
        lad = ps.fetch_alt_ladders(key, e["id"])
        for (k, mkt), ladder in lad.items():
            if mkt not in ("rush_yds", "rec_yds", "pass_yds"): continue     # count markets are discrete; skip
            f = fit_ladder(sorted(ladder))
            if not f or f["rmse"] > 0.03: continue
            f.update({"player": k, "mkt": mkt, "game": f"{e['away_team']} @ {e['home_team']}",
                      "cv_ours": round(pm.cv_for(mkt, f["mu"]), 3)})
            f["cv_ratio_book_over_ours"] = round(f["cv_book"] / f["cv_ours"], 3)
            rows.append(f)
    df = pd.DataFrame(rows)
    if not len(df):
        print("no ladders fit"); return df
    df["tier"] = pd.cut(df["mu"], [0, 30, 55, 85, 400], labels=["<30", "30-55", "55-85", "85+"])
    print(f"{len(df)} player-ladders fit (rmse<=.03) across {n_games} games\n")
    print(df.sort_values(["mkt", "mu"])[["player", "mkt", "mu", "cv_book", "cv_ours", "cv_ratio_book_over_ours", "vig", "n_rungs", "rmse"]].to_string(index=False))
    print("\nBy market x volume tier — the book's fitted CV vs our measured CV:")
    g = df.groupby(["mkt", "tier"], observed=True).agg(n=("player", "size"), cv_book=("cv_book", "mean"), cv_ours=("cv_ours", "mean"), vig=("vig", "mean")).round(3)
    print(g.to_string())
    # the shape test: does the book's CV fall with volume the way ours does?
    out = {"n": int(len(df)), "by_tier": {f"{m}|{t}": {"n": int(r.n), "cv_book": float(r.cv_book), "cv_ours": float(r.cv_ours), "vig": float(r.vig)} for (m, t), r in g.iterrows()}}
    for mkt in df.mkt.unique():
        s = df[df.mkt == mkt]
        if len(s) >= 6:
            slope_book = np.polyfit(s.mu, s.cv_book, 1)[0]; slope_ours = np.polyfit(s.mu, s.cv_ours, 1)[0]
            out[f"slope_{mkt}"] = {"book": round(float(slope_book), 5), "ours": round(float(slope_ours), 5), "n": int(len(s))}
            print(f"  {mkt}: CV slope vs volume — book {slope_book:+.5f}/yd, ours {slope_ours:+.5f}/yd (n={len(s)})")
    json.dump(out, open(f"{DATA}/alt_ladder_probe.json", "w"), indent=1)
    df.to_parquet(f"{DATA}/alt_ladder_probe.parquet", index=False)
    return df


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
