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


def role_recency(through_season=2025, through_week=None):
    """Per pid: the ROLE features the model earned (multifactor.ROLE_FEATURES) from the data
    universe — route participation over his last 3 games (raw; the board centers it with the
    model's role_med) and the snap-share trend (last-3 offense % minus his prior average)."""
    p = f"{DATA}/universe_features_player_game.parquet"
    if not os.path.exists(p):
        return {}
    u = pd.read_parquet(p, columns=["pid", "season", "week", "route_rate", "offense_pct"])
    if through_week is not None:
        u = u[(u.season < through_season) | ((u.season == through_season) & (u.week <= through_week))]
    else:
        u = u[u.season <= through_season]
    u = u.sort_values(["pid", "season", "week"])
    import multifactor as _mf
    rest = u["offense_pct"] < _mf.REST_SNAP_PCT           # rest days / early exits are not role reads
    u.loc[rest, ["route_rate", "offense_pct"]] = np.nan
    out = {}
    for pid, d in u.groupby("pid"):
        rr = d["route_rate"].to_numpy(float); sp = d["offense_pct"].to_numpy(float)
        rr = rr[~np.isnan(rr)]; sp = sp[~np.isnan(sp)]
        if not len(rr) and not len(sp):
            continue
        # matches multifactor.role_lag_table: last-3 mean minus the expanding mean of everything so far
        out[pid] = {"role_rr": float(np.mean(rr[-3:])) if len(rr) else None,
                    "role_snap_d": (float(np.mean(sp[-3:]) - np.mean(sp)) if len(sp) >= 3 else 0.0),
                    "role_last_season": int(d["season"].iloc[-1]), "role_last_week": int(d["week"].iloc[-1]),
                    "route_rate_log": [round(float(x), 2) for x in rr[-3:]], "snap_pct_log": [round(float(x), 2) for x in sp[-3:]]}
    return out


def build_recency(through_season=2025, through_week=None):
    """Per (pid, mkt): recency form (weighted recent games), baseline, trend, last game log,
    plus the role features (route participation, snap-share trend) from the data universe."""
    pg = pd.read_parquet(f"{DATA}/prop_player_games.parquet").sort_values(["season", "week"])
    if through_week is not None:
        pg = pg[(pg.season < through_season) | ((pg.season == through_season) & (pg.week <= through_week))]
    else:
        pg = pg[pg.season <= through_season]
    roles = role_recency(through_season, through_week)
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
        # when his window ends: the season/week of his most recent game. The board decides PER
        # PLAYER whether the form window crosses the offseason (half-strength, measured) or is
        # older than one offseason (a missed season: unmeasurable, so form is not applied).
        rows.append({"pid": pid, "mkt": mkt, "form": round(form, 1), "baseline": round(baseline, 1),
                     "last3": round(last3, 1), "last1": round(last1, 1),
                     "trend": round(form - baseline, 1), "n_games": len(ys),
                     "last_season": int(d["season"].iloc[-1]), "last_week": int(d["week"].iloc[-1]),
                     "log": [round(float(x), 1) for x in ys[-RECENT_N:]], **(roles.get(pid) or {})})
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
