"""NFL betting model engine.

Core design:
- Team off/def ratings from EPA/play (pass-weighted composite), opponent-adjusted
  iteratively, exponentially recency-weighted, shrunk to league mean.
- Season boundary: extra decay (offseason ~= OFFSEASON_WEEKS weeks of staleness)
  plus shrinkage handles preseason regression.
- Spread: walk-forward linear regression margin ~ net_rating_diff + rest + hfa.
- Total: walk-forward regression total ~ combined ratings + league env + roof.
- Moneyline: margin -> win prob via Normal(margin_pred, SIGMA).
"""
import numpy as np
import pandas as pd

HALFLIFE_WEEKS = 10.0       # recency decay halflife (in schedule weeks)
OFFSEASON_WEEKS = 10.0      # extra staleness added at each season boundary
PRIOR_N = 6.0               # shrinkage pseudo-games toward league average (offense)
PRIOR_N_DEF = 12.0          # defense regresses harder — def EPA is less stable YoY
PASS_W, RUSH_W = 0.62, 0.38 # composite weights (pass EPA more predictive)
N_ITER = 8                  # opponent-adjustment iterations
SIGMA = 13.2                # historical sd of margin vs spread


def build_team_games(epa_path: str, sched_path: str) -> pd.DataFrame:
    """One row per (game, team): offensive composite EPA + meta."""
    epa = pd.read_parquet(epa_path)
    piv = epa.pivot_table(index=["game_id", "season", "week", "posteam", "defteam"],
                          columns="play_type", values="epa_pp").reset_index()
    plays = epa.pivot_table(index=["game_id", "posteam"], columns="play_type",
                            values="plays").reset_index()
    piv = piv.merge(plays, on=["game_id", "posteam"], suffixes=("", "_n"))
    piv["off_epa"] = PASS_W * piv["pass"].fillna(0) + RUSH_W * piv["run"].fillna(0)
    piv = piv.rename(columns={"posteam": "team", "defteam": "opp",
                              "pass": "pass_epa", "run": "rush_epa",
                              "pass_n": "pass_plays", "run_n": "rush_plays"})
    # absolute week index for decay (season boundaries add offseason staleness)
    seasons = sorted(piv["season"].unique())
    base = {s: i * (18 + OFFSEASON_WEEKS) for i, s in enumerate(seasons)}
    piv["abs_week"] = piv["season"].map(base) + piv["week"]
    return piv


def ratings_asof(tg: pd.DataFrame, asof_abs_week: float, window_weeks: float = 80.0):
    """Opponent-adjusted off/def ratings using games strictly before asof_abs_week."""
    d = tg[(tg["abs_week"] < asof_abs_week) &
           (tg["abs_week"] > asof_abs_week - window_weeks)].copy()
    if d.empty:
        return pd.DataFrame(columns=["team", "off", "deff", "n_eff"])
    d["w"] = 0.5 ** ((asof_abs_week - d["abs_week"]) / HALFLIFE_WEEKS)
    lg = np.average(d["off_epa"], weights=d["w"])

    teams = pd.unique(pd.concat([d["team"], d["opp"]]))
    off = {t: 0.0 for t in teams}
    deff = {t: 0.0 for t in teams}  # def rating: EPA/play allowed vs league (neg = good D)

    dt = d[["team", "opp", "off_epa", "w"]].to_records(index=False)
    for _ in range(N_ITER):
        # offense adjusted for opponent defenses faced
        num_o, den_o = {t: 0.0 for t in teams}, {t: 0.0 for t in teams}
        num_d, den_d = {t: 0.0 for t in teams}, {t: 0.0 for t in teams}
        for team, opp, e, w in dt:
            num_o[team] += w * (e - lg - deff[opp]); den_o[team] += w
            num_d[opp] += w * (e - lg - off[team]); den_d[opp] += w
        off = {t: (num_o[t] / (den_o[t] + PRIOR_N)) for t in teams}   # shrunk
        deff = {t: (num_d[t] / (den_d[t] + PRIOR_N_DEF)) for t in teams}

    out = pd.DataFrame({"team": list(teams),
                        "off": [off[t] for t in teams],
                        "deff": [deff[t] for t in teams],
                        "n_eff": [den_o[t] for t in teams]})
    out["net"] = out["off"] - out["deff"]
    return out


def build_game_features(sched: pd.DataFrame, tg: pd.DataFrame,
                        seasons: list[int]) -> pd.DataFrame:
    """Walk-forward: ratings recomputed before each (season, week)."""
    seasons_all = sorted(tg["season"].unique())
    base = {s: i * (18 + OFFSEASON_WEEKS) for i, s in enumerate(seasons_all)}
    rows = []
    for season in seasons:
        weeks = sorted(sched.loc[sched["season"] == season, "week"].unique())
        for week in weeks:
            asof = base.get(season, max(base.values()) + 28) + week
            r = ratings_asof(tg, asof).set_index("team")
            g = sched[(sched["season"] == season) & (sched["week"] == week)]
            for _, gm in g.iterrows():
                h, a = gm["home_team"], gm["away_team"]
                if h not in r.index or a not in r.index:
                    continue
                rows.append({
                    "game_id": gm["game_id"], "season": season, "week": week,
                    "home": h, "away": a,
                    "net_diff": r.at[h, "net"] - r.at[a, "net"],
                    "off_sum": r.at[h, "off"] + r.at[a, "off"],
                    "def_sum": r.at[h, "deff"] + r.at[a, "deff"],
                    "rest_diff": (gm.get("home_rest", 7) or 7) - (gm.get("away_rest", 7) or 7),
                    "dome": 1.0 if str(gm.get("roof")) in ("dome", "closed") else 0.0,
                    "wind": float(gm["wind"]) if pd.notnull(gm.get("wind")) else 0.0,
                    "div_game": float(gm.get("div_game", 0) or 0),
                    "neutral": 1.0 if gm.get("location") == "Neutral" else 0.0,
                    "result": gm.get("result"), "total_pts": gm.get("total"),
                    "spread_line": gm.get("spread_line"), "total_line": gm.get("total_line"),
                    "home_ml": gm.get("home_moneyline"), "away_ml": gm.get("away_moneyline"),
                })
    return pd.DataFrame(rows)
