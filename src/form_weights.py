"""OPPORTUNITY-AWARE FORM WEIGHTS (UPGRADE_PLAN.md §2; measured in data/form_anomaly_effect.json).

A player's raw total is team volume × his share × his efficiency. When the TEAM's volume is forced by the
environment — a monsoon (TB Week 8 2026: 4 pass attempts, 56 carries), a 42-7 blowout (JAX: 17 attempts),
a mid-game QB loss — the raw total collapses for every player on that team and the form window reads it as
a change in HIM. Measured 2021-25 (slope of next-game residual on last-game deviation): a receiving-yards
game played when the team threw <= 55% of its usual attempts predicts NOTHING about next week (-.029 vs
+.165 in normal games, n=63); even target share dies in those games. Receptions keep ~70% of their
information (.142 vs .205). Rushing keeps its information at every team volume (a 56-carry game is a real
read on the backfield), so rushing is untouched.

This is NOT the player-opportunity exclusion that `full_game_form.py` tested and rejected (a player's own
short game — rest, early exit, benching — IS a role signal). The flag here is the TEAM's volume: the cause,
not the symptom. Passing yards were tried under the same rule by analogy (n=10 unmeasurable) and REJECTED by the
walk-forward A/B — see WEIGHT_IN_ANOMALY.
"""
import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
ANOMALY_RATIO = 0.55                       # team attempts <= 55% of its season median
MIN_GAMES_FOR_MEDIAN = 3
# WALK-FORWARD VERDICT (data/form_weights_ab.json, 2024 + 2025): rec_yds .629->.632 / .658->.658 and receptions
# .640->.640 / .668->.669 (neutral-to-better, MAE unchanged) -> ADOPTED. pass_yds LOST (.688->.671, .642->.634;
# MAE up): a QB's own low-attempt game is usually HIS exit (benched / hurt), i.e. a role signal, not weather —
# so passing yards keep the raw window (weight 1). Rushing untouched by measurement.
WEIGHT_IN_ANOMALY = {"rec_yds": 0.0, "pass_yds": 1.0, "receptions": 0.7, "rush_yds": 1.0, "rush_att": 1.0,
                     "pass_att": 1.0, "pass_cmp": 1.0}
_CACHE = {}


def team_volume(pbp=None):
    """Per (season, week, game_id, posteam): pass/rush attempts, the team's season-to-date median pass
    attempts (expanding, >= 3 games, else that season's league median) and the low-pass anomaly flag."""
    if "tv" in _CACHE and pbp is None:
        return _CACHE["tv"]
    p = pbp if pbp is not None else pd.read_parquet(f"{DATA}/pbp_2021_2025.parquet",
                                                     columns=["season", "week", "game_id", "posteam", "play_type", "pass_attempt", "rush_attempt"])
    p = p[p["play_type"].isin(["run", "pass"]) & p["posteam"].notna()]
    tg = p.groupby(["season", "week", "game_id", "posteam"]).agg(pa=("pass_attempt", "sum"), ra=("rush_attempt", "sum")).reset_index()
    tg = tg.sort_values(["posteam", "season", "week"])
    tg["pa_med"] = tg.groupby(["posteam", "season"])["pa"].transform(lambda s: s.expanding(min_periods=MIN_GAMES_FOR_MEDIAN).median())
    lg = tg.groupby("season")["pa"].median()
    tg["pa_med"] = tg["pa_med"].fillna(tg["season"].map(lg))
    tg["pa_ratio"] = tg["pa"] / tg["pa_med"].clip(lower=1)
    tg["anomaly_lowpass"] = tg["pa_ratio"] <= ANOMALY_RATIO
    if pbp is None:
        _CACHE["tv"] = tg
    return tg


def attach_weights(pg, pbp=None):
    """Return pg with a `w` column: the form weight of each player-game for its market."""
    tv = team_volume(pbp)[["game_id", "posteam", "pa", "pa_med", "pa_ratio", "anomaly_lowpass"]]
    out = pg.merge(tv, on=["game_id", "posteam"], how="left")
    anom = out["anomaly_lowpass"].fillna(False).astype(bool)
    w = np.ones(len(out))
    for mkt, wa in WEIGHT_IN_ANOMALY.items():
        m = (out["mkt"] == mkt).to_numpy() & anom.to_numpy()
        w[m] = wa
    out["w"] = w
    out.index = pg.index
    return out


def weighted_form(ys, ws, min_base=4):
    """For one (player, market) series ordered by time: per game, the baseline / last3 / last1 built from
    PRIOR games with weight > 0 (zero-weight games are skipped — the window rolls back; fractional weights
    are weighted means). Mirrors shift(1).expanding(min_periods=4).mean(), shift(1).rolling(3).mean(),
    shift(1) when every weight is 1."""
    n = len(ys); base = np.full(n, np.nan); l3 = np.full(n, np.nan); l1 = np.full(n, np.nan)
    kept_y, kept_w = [], []
    for i in range(n):
        if len(kept_y) >= min_base:
            base[i] = np.average(kept_y, weights=kept_w)
        if len(kept_y) >= 3:
            l3[i] = np.average(kept_y[-3:], weights=kept_w[-3:])
        if len(kept_y) >= 1:
            l1[i] = kept_y[-1]
        if ws[i] > 0:
            kept_y.append(float(ys[i])); kept_w.append(float(ws[i]))
    return base, l3, l1


def summarize_live(ys, ws, seasons, weeks, baseline_n=16):
    """For the live recency row: baseline over the last `baseline_n` counted games, last3/last1 over counted
    games, and the list of excluded games (season, week)."""
    keep = ws > 0
    yk, wk = ys[keep], ws[keep]
    if len(yk) == 0:
        return None
    base = float(np.average(yk[-baseline_n:], weights=wk[-baseline_n:])) if len(yk) >= 8 else float(np.average(yk, weights=wk))
    last3 = float(np.average(yk[-3:], weights=wk[-3:])); last1 = float(yk[-1])
    excluded = [(int(s), int(w)) for s, w, k in zip(seasons, weeks, keep) if not k]
    return base, last3, last1, len(yk), excluded
