"""Continuous / count player-prop model (yards, receptions).

Per player-game we project a distribution of the stat and price P(over line):
  mean = shrunk recent per-game average * opponent_factor * environment_factor
  spread: yards ~ Gamma (right-skewed, non-negative) fit by (mean, cv);
          receptions ~ Negative-Binomial-ish via Gamma-Poisson, approximated Normal
          with variance = mean + mean^2*cv^2 for count.
  P(over L) from the fitted distribution.

Calibrated out-of-sample: predicted P(X>threshold) must match realized rates.
Only edges that beat the vig AND clear a data-confidence floor become picks.
"""
import numpy as np
import pandas as pd
from scipy.stats import gamma, norm

SHRINK_GAMES = 5.0
# per-market game-to-game coefficient of variation (from 2023-25 fit), used for the
# distribution spread; yards are right-skewed (gamma), counts use normal approx.
CV = {"rec_yds": 0.85, "rush_yds": 0.88, "pass_yds": 0.34, "receptions": 0.62, "rush_att": 0.45, "pass_att": 0.20, "pass_cmp": 0.22}
DIST = {"rec_yds": "gamma", "rush_yds": "gamma", "pass_yds": "normal", "receptions": "count", "rush_att": "count", "pass_att": "normal", "pass_cmp": "normal"}
MIN_GAMES = 6           # data-confidence floor to be eligible as a pick
EDGE_MIN = 0.045        # model P(over/under) minus de-vigged implied, to flag


def implied(o):
    return 100/(o+100) if o > 0 else abs(o)/(abs(o)+100)


def build_rates(prop_games: pd.DataFrame):
    """Per (pid, market): shrunk mean, games, team. Recency: weight recent games."""
    rows = []
    for (pid, mkt), d in prop_games.groupby(["pid", "mkt"]):
        d = d.sort_values(["season", "week"])
        n = len(d)
        w = 0.5 ** ((d["season"].max()*18 + d["week"].max() - (d["season"]*18 + d["week"])) / 20.0)
        mean = np.average(d["y"], weights=w)
        # league/position mean for shrinkage = market median as a floor prior
        rows.append({"pid": pid, "mkt": mkt, "n": n, "mean_raw": mean,
                     "team": d["posteam"].iloc[-1]})
    r = pd.DataFrame(rows)
    for mkt in r["mkt"].unique():
        m = r["mkt"] == mkt
        prior = r.loc[m, "mean_raw"].median()
        w = r.loc[m, "n"] / (r.loc[m, "n"] + SHRINK_GAMES)
        r.loc[m, "mean"] = w * r.loc[m, "mean_raw"] + (1 - w) * prior
    return r


def p_over(mkt, mean, line, env=1.0, opp=1.0):
    """P(stat > line) for a player with projected mean, scaled by env & opponent."""
    mu = max(mean * env * opp, 0.1)
    cv = CV[mkt]
    d = DIST[mkt]
    if d == "gamma":
        k = 1.0 / (cv ** 2)                 # shape
        theta = mu / k                       # scale
        return float(1 - gamma.cdf(line, a=k, scale=theta))
    if d == "normal":
        sd = mu * cv
        return float(1 - norm.cdf(line, loc=mu, scale=sd))
    # count (receptions): normal approx with var = mu + (mu*cv)^2, continuity corr
    var = mu + (mu * cv) ** 2
    return float(1 - norm.cdf(line + 0.5, loc=mu, scale=np.sqrt(var)))


def price_props(rates, board, env_by_team=None, opp_by_pid=None):
    """board: DataFrame[pid, name, mkt, line, over_price, under_price, game, team].
    Returns each with model P(over), best side, edge, confidence."""
    env_by_team = env_by_team or {}
    opp_by_pid = opp_by_pid or {}
    out = []
    for _, b in board.iterrows():
        r = rates[(rates["pid"] == b["pid"]) & (rates["mkt"] == b["mkt"])]
        if not len(r):
            continue
        r = r.iloc[0]
        env = env_by_team.get(b.get("team", r["team"]), 1.0)
        po = p_over(b["mkt"], r["mean"], b["line"], env=env, opp=opp_by_pid.get(b["pid"], 1.0))
        imp_o = implied(b["over_price"]); imp_u = implied(b["under_price"])
        fair_o = imp_o / (imp_o + imp_u)     # de-vig
        edge_over = po - fair_o
        edge_under = (1 - po) - (1 - fair_o)
        if edge_over >= edge_under:
            side, edge, price = "Over", edge_over, b["over_price"]
        else:
            side, edge, price = "Under", edge_under, b["under_price"]
        out.append({"name": b.get("name"), "mkt": b["mkt"], "line": b["line"],
                    "side": side, "price": price, "p_model_over": round(po, 3),
                    "edge": round(edge, 3), "games": int(r["n"]),
                    "proj_mean": round(r["mean"] * env, 1), "game": b.get("game")})
    return pd.DataFrame(out).sort_values("edge", ascending=False) if out else pd.DataFrame()


# ---- market-anchored projection (the fix for confident-wrong edges) --------
from scipy.optimize import brentq

def market_mean(mkt, line, p_over_fair):
    """Invert: find the mean whose fitted distribution gives P(over line)=p_over_fair.
    This anchors our projection to what the MARKET implies, so average edge ~= 0.
    Our edge then comes ONLY from tilts we apply on top (environment, matchup)."""
    p_over_fair = min(max(p_over_fair, 0.02), 0.98)
    try:
        return brentq(lambda mu: p_over(mkt, mu, line) - p_over_fair, 0.5, line * 6 + 20)
    except Exception:
        return line  # fallback


def anchored_edge(mkt, line, over_price, under_price, env=1.0, opp=1.0,
                  role_delta=0.0, max_tilt=0.14):
    """Edge = model P(over) at a TILTED mean vs the de-vigged market prob.
    tilt = env * opp * (1+role_delta), clipped so no single prop shows a crazy edge
    unless the tilt genuinely warrants it. Returns (side, edge, p_model, proj_mean)."""
    io, iu = implied(over_price), implied(under_price)
    fair_o = io / (io + iu)
    mu_mkt = market_mean(mkt, line, fair_o)
    tilt = float(np.clip(env * opp * (1 + role_delta), 1 - max_tilt, 1 + max_tilt))
    mu_model = mu_mkt * tilt
    po = p_over(mkt, mu_model, line)
    eo, eu = po - fair_o, (1 - po) - (1 - fair_o)
    if eo >= eu:
        return "Over", round(eo, 3), round(po, 3), round(mu_model, 1)
    return "Under", round(eu, 3), round(po, 3), round(mu_model, 1)
