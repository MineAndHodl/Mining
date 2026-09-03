"""Measure the game-to-game spread of every prop market the RIGHT way — within a player's own
season — and write the CV anchors the pricing model uses (data/cv_anchors.json).

Why this exists (Fresh Eyes, day 1): the DK alternate-line ladders imply a distribution ~10-40%
NARROWER than our CV anchors at the high-volume end (pass yds: book .27 vs ours .40). The
book was right. The old anchors were pooled by volume bucket across players, which folds
between-player mean differences into the spread; the honest quantity for pricing a KNOWN
player is the median within-player-season CV (>= 8 games), by volume tier, 2021-25:

  pass_yds  .54 at ~160  .37 at ~205  .28 at ~240  .27 at ~275   (old: .43 -> .37, nearly flat)
  rec_yds   .98 / .84 / .67 / .56 / .52 at 13 / 27 / 43 / 66 / 96 (old top .57 at 80)
  rush_yds 1.06 / .77 / .59 / .52 / .43 at 11 / 29 / 50 / 68 / 97 (old top .53 at 85)
  receptions: the count model adds Poisson variance itself, so its anchor is the EXTRA
  dispersion cv_x with var = mu + (mu*cv_x)^2 -> cv_x ~ .17 (old anchors .58/.46 were the
  TOTAL CV applied as extra dispersion — the variance was double counted).

A 3% predictive inflation is applied (the mean is estimated from ~16 games: sqrt(1 + 1/16)).
"""
import os, sys, json
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
TIERS = {"pass_yds": [100, 180, 220, 260, 320, 500], "rec_yds": [0, 20, 35, 55, 85, 150],
         "rush_yds": [0, 20, 40, 60, 85, 150], "receptions": [0, 2, 3.5, 5, 7, 12]}
PRED_INFLATION = 1.03
MIN_GAMES = 8


def measure(seasons=(2021, 2022, 2023, 2024, 2025)):
    pg = pd.read_parquet(f"{DATA}/prop_player_games.parquet")
    pg = pg[pg.season.isin(seasons)]
    rows = []
    for (pid, mkt, season), d in pg.groupby(["pid", "mkt", "season"]):
        if len(d) < MIN_GAMES: continue
        y = d.y.values; mu = y.mean(); sd = y.std(ddof=1)
        if mu > 0: rows.append((mkt, mu, sd / mu, len(d)))
    r = pd.DataFrame(rows, columns=["mkt", "mu", "cv", "n"])
    anchors, table = {}, {}
    for mkt, bins in TIERS.items():
        s = r[r.mkt == mkt].copy(); s["tier"] = pd.cut(s.mu, bins)
        g = s.groupby("tier", observed=True).agg(n=("cv", "size"), mu=("mu", "mean"), cv=("cv", "median"))
        pts = []
        for _, row in g.iterrows():
            if row.n < 10: continue
            cv = float(row.cv) * PRED_INFLATION
            if mkt == "receptions":                       # extra dispersion beyond Poisson
                cv = float(np.sqrt(max(cv ** 2 - 1.0 / float(row.mu), 0.0)))
            pts.append((round(float(row.mu), 1), round(cv, 3)))
        anchors[mkt] = pts
        table[mkt] = [{"tier": str(t), "n": int(v.n), "mu": round(float(v.mu), 1), "cv_within": round(float(v.cv), 3)} for t, v in g.iterrows()]
    out = {"anchors": anchors, "table": table, "method": "median within-player-season CV (>=8 games), by volume tier, 2021-25, x1.03 predictive inflation; receptions as extra dispersion",
           "n_player_seasons": int(len(r))}
    json.dump(out, open(f"{DATA}/cv_anchors.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    o = measure()
    for mkt, pts in o["anchors"].items():
        print(f"{mkt:11s} {pts}")
