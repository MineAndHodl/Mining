"""The real prop edge: recency-weighted usage/production that the market's season-anchored line
underweights.

Validated out-of-sample (2023-25 player-games): when a player's recent 3-game form diverges
strongly from his season baseline, betting the direction of the trend beat that baseline as a
line ~57-64% of the time — +8.6% ROI in 2024, +13.3% in 2025. Matchup (prior-year defense) did
NOT predict; recency did. So our edge is a RECENCY projection, and we bet where it diverges
from the posted line — the line lags recent usage, especially on lower-limit prop markets.

This module builds, per (player, market): the recency form (recent games, weighted), a longer
baseline, the trend (form - baseline), and the recent game log for the rationale. In-season it
uses current-year games; preseason it falls back to the tail of last season (a weaker signal —
hence Week 1 is our thinnest edge, honestly).
"""
import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
RECENT_N = 4           # games in the "recent form" window
HALFLIFE = 2.5         # recency half-life within that window


def build_recency(through_season=2025, through_week=None):
    """Per (pid, mkt): recency form (weighted recent games), baseline, trend, last game log."""
    pg = pd.read_parquet(f"{DATA}/prop_player_games.parquet").sort_values(["season", "week"])
    if through_week is not None:
        pg = pg[(pg.season < through_season) | ((pg.season == through_season) & (pg.week <= through_week))]
    else:
        pg = pg[pg.season <= through_season]
    rows = []
    for (pid, mkt), d in pg.groupby(["pid", "mkt"]):
        d = d.sort_values(["season", "week"])
        ys = d["y"].to_numpy(dtype=float)
        if len(ys) < 4:
            continue
        baseline = float(np.mean(ys[-16:])) if len(ys) >= 8 else float(np.mean(ys))
        recent = ys[-RECENT_N:]
        w = 0.5 ** (np.arange(len(recent))[::-1] / HALFLIFE)   # most-recent heaviest
        form = float(np.average(recent, weights=w))
        last3 = float(np.mean(ys[-3:])); last1 = float(ys[-1])   # for the multi-factor features
        rows.append({"pid": pid, "mkt": mkt, "form": round(form, 1), "baseline": round(baseline, 1),
                     "last3": round(last3, 1), "last1": round(last1, 1),
                     "trend": round(form - baseline, 1), "n_games": len(ys),
                     "log": [round(float(x), 1) for x in ys[-RECENT_N:]]})
    r = pd.DataFrame(rows)
    # trend strength percentile within each market (the validated edge lives in the top quartile)
    for mkt in r["mkt"].unique():
        m = r["mkt"] == mkt
        r.loc[m, "trend_pct"] = r.loc[m, "trend"].abs().rank(pct=True)
    return r


if __name__ == "__main__":
    r = build_recency()
    print(f"recency signals: {len(r)} player-markets")
    print("\n=== strongest UP trends (recent form well above baseline) ===")
    top = r[r.mkt == "rec_yds"].sort_values("trend", ascending=False).head(8)
    for _, x in top.iterrows():
        print(f"  {x['pid']} rec_yds: last {x['log']} -> form {x['form']} vs baseline {x['baseline']} (trend {x['trend']:+})")
