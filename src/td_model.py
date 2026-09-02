"""Anytime-TD probability model.

P(player scores a TD) via Poisson: P = 1 - exp(-lambda), where
lambda = player's expected TDs this game
       = regressed_player_TD_rate * team_environment_multiplier

- regressed_player_TD_rate: TDs/game over a recent window, shrunk toward the
  position mean by games played (heavy shrinkage for small samples), blended
  with a red-zone-opportunity-based expectation (RZ opps * position TD/opp rate)
  so usage changes show up faster than results do.
- team_environment_multiplier: (implied team TDs from today's market total &
  spread) / (team's TDs/gm in the sample window) — moves every player on a team
  up/down with the game environment, and handles team changes when the caller
  maps a player to his current team.

Edge vs a book: edge = P_model - implied(price). ATD 'Yes' is a one-sided
market with fat margin, so only flag edges >= EDGE_MIN.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from names import norm_name

RZ_YARD = 10          # "red zone opportunity" = carry/target inside this yardline
BLEND_RESULTS = 0.55  # weight on realized TD rate vs RZ-opportunity expectation
SHRINK_GAMES = 8.0    # pseudo-games toward position mean
EDGE_MIN = 0.05       # min model-vs-implied edge to flag a leg
OFF_TD_POINTS_SHARE = 0.62  # share of team points scored via offensive TDs (hist.)


def implied(o):
    return 100/(o+100) if o > 0 else abs(o)/(abs(o)+100)


def player_rates(pbp_frames: list, players: pd.DataFrame):
    """Per player: games, TDs/gm, RZ opps/gm, team, position, display name.
    pbp_frames: list of polars->pandas pbp subsets with needed columns."""
    ev = pd.concat(pbp_frames, ignore_index=True)
    # TD events credited to rusher/receiver
    td = ev[ev["td_player_id"].notna()]
    tds = td.groupby("td_player_id").size().rename("tds")
    # games played (appearances with any touch)
    touches = pd.concat([
        ev[["game_id", "rusher_player_id"]].rename(columns={"rusher_player_id": "pid"}),
        ev[["game_id", "receiver_player_id"]].rename(columns={"receiver_player_id": "pid"}),
    ]).dropna()
    games = touches.drop_duplicates().groupby("pid").size().rename("games")
    # red-zone opportunities
    rz = ev[ev["yardline_100"] <= RZ_YARD]
    rzo = pd.concat([
        rz["rusher_player_id"], rz["receiver_player_id"]
    ]).dropna().value_counts().rename("rz_opps")
    # latest team seen
    last_team = pd.concat([
        ev[["pid_r" if False else "rusher_player_id", "posteam"]].rename(columns={"rusher_player_id": "pid"}),
        ev[["receiver_player_id", "posteam"]].rename(columns={"receiver_player_id": "pid"}),
    ]).dropna().groupby("pid")["posteam"].last().rename("team")

    df = pd.concat([tds, games, rzo, last_team], axis=1).fillna({"tds": 0, "rz_opps": 0})
    df = df[df["games"] >= 3].reset_index().rename(columns={"index": "pid"})
    df = df.merge(players[["gsis_id", "position", "display_name"]],
                  left_on="pid", right_on="gsis_id", how="left")
    df = df[df["position"].isin(["RB", "WR", "TE", "QB", "FB"])]
    df["td_pg"] = df["tds"] / df["games"]
    df["rz_pg"] = df["rz_opps"] / df["games"]
    # position TD-per-RZ-opp rates and position mean TD/gm (for shrinkage)
    pos_rate = df.groupby("position").apply(
        lambda g: g["tds"].sum() / max(g["rz_opps"].sum(), 1)).rename("pos_td_per_rz")
    pos_mean = df.groupby("position")["td_pg"].median().rename("pos_td_pg")
    df = df.merge(pos_rate, on="position").merge(pos_mean, on="position")
    # blended, shrunk expected TDs/game in the SAMPLE environment
    raw = (BLEND_RESULTS * df["td_pg"]
           + (1-BLEND_RESULTS) * df["rz_pg"] * df["pos_td_per_rz"])
    w = df["games"] / (df["games"] + SHRINK_GAMES)
    df["lam_base"] = w * raw + (1-w) * df["pos_td_pg"] * 0.6
    return df


def team_env(sample_team_tds_pg: dict, market_spread_home, market_total, home, away):
    """Implied offensive TDs per team from today's market, and multipliers.
    team implied points: total/2 -/+ spread/2; TDs ~ points * SHARE / 7."""
    pts_home = market_total/2 + (-market_spread_home)/2 * -1  # spread_home<0 => home favored
    pts_home = market_total/2 - market_spread_home/2 * -1
    # simpler: home points = (total - spread_home_signed)/2 where result=home-away and spread_home = home handicap (neg = favored)
    pts_home = (market_total - market_spread_home) / 2  # spread_home=-3 -> home 3 more than away
    pts_away = market_total - pts_home
    out = {}
    for team, pts in ((home, pts_home), (away, pts_away)):
        implied_tds = pts * OFF_TD_POINTS_SHARE / 7.0 * 1.08  # 1.08: TDs slightly cheaper than 7 w/ 2pt/missed XP
        base = sample_team_tds_pg.get(team, 2.4)
        out[team] = implied_tds / max(base, 0.8)
    return out


CAL_A, CAL_B = 0.055, 0.916  # refit WITH the env multiplier (td_env_validate.py --final: fit 2022->23 + 2023->24, test 2024->25)


def p_calibrated(lam):
    return float(np.clip(CAL_A + CAL_B * (1 - np.exp(-lam)), 0.02, 0.85))


# MEASURED (td_env_validate.py, 2,718 team-games 2021-25): offensive TDs = 0.1387 * implied
# points - 0.717. The old constant (0.62*1.08/7 = 0.0957/pt, no intercept) under-counted every
# offense by ~10% and the best offenses by ~17% (27 implied pts -> 2.58 TDs vs 3.03 real), and
# was the reason every star scorer sat 10-25 pts under his market price. Last season's team TD
# rate adds NOTHING beyond the market's implied points (coef .007), and the full ratio (k=1)
# earned the best out-of-sample Brier (.1650 vs .1665 with no environment).
TD_PER_IMPLIED_PT, TD_INTERCEPT = 0.1387, -0.717
ENV_CLIP = (0.5, 1.8)


def implied_team_tds(pts):
    return max(TD_PER_IMPLIED_PT * float(pts) + TD_INTERCEPT, 0.6)


def env_multipliers(team_td_baseline: dict, spread_home, total, home, away):
    """Environment: the market's implied TDs for each team (measured scale) over the team's
    own baseline TDs/gm that the player's lambda was built on."""
    pts_home = (total - spread_home) / 2
    pts_away = total - pts_home
    return {t: float(np.clip(implied_team_tds(p) / max(team_td_baseline.get(t, 2.4), 0.8), *ENV_CLIP))
            for t, p in ((home, pts_home), (away, pts_away))}


def price_board(rates: pd.DataFrame, board_names: pd.DataFrame,
                env_mult: dict, current_team: dict = None):
    """board_names: DataFrame[name, price, game, team_hint]. Returns model P,
    implied p, edge per board player (matched on display_name)."""
    rows = []
    rates = rates.copy()
    rates["_k"] = rates["display_name"].map(norm_name)
    for _, b in board_names.iterrows():
        m = rates[rates["_k"] == norm_name(b["name"])]     # normalized join (was exact-string, dropped ~21%)
        if not len(m):
            rows.append({**b.to_dict(), "p_model": None, "note": "no data"})
            continue
        r = m.iloc[0]
        team = (current_team or {}).get(b["name"], r["team"])
        mult = env_mult.get(team, 1.0)
        lam = r["lam_base"] * mult
        p = p_calibrated(lam)          # use the fitted recalibration, not the raw Poisson
        rows.append({**b.to_dict(), "p_model": round(p, 3),
                     "p_implied": round(implied(b["price"]), 3),
                     "edge": round(p - implied(b["price"]), 3),
                     "lam": round(lam, 3), "games": int(r["games"])})
    return pd.DataFrame(rows).sort_values("edge", ascending=False, na_position="last")
