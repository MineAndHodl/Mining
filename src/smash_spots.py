"""Smash-spot longshot engine — the ceiling outcome, only when the stars align.

Deep out-of-the-money player props (100+ receiving/rushing yards, 8+ catches,
2+ TDs) are priced lazily by books because the tail is hard. We price the tail
from the model's own distribution and only surface a spot when MULTIPLE real
factors stack in its favor — never a lone hunch. Every surfaced spot carries the
reasons, so we can say WHY we like it.

Factors (each a small, capped multiplier on the projected mean; only real ones):
  role        projected volume / target share is elite or ascending          (strong)
  environment player's team implied total is high (more scoring chances)      (strong)
  script      game script favors the market (dog -> pass vol for WR/QB;
              favorite -> rush vol for RB)                                     (medium)
  pace        team runs a high-play, fast offense                             (medium)
  matchup     opponent weak vs the position — WEAK for WR (our data: r=.08),
              modest for RB (0.26); used only as a tiebreak                    (weak)

A spot needs >= MIN_FACTORS aligned AND a model ceiling probability above the bar
(and, when a longshot line is posted, model P > market implied P).
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import props_model as pm

# ceiling thresholds per market (the "pop off" line)
CEILINGS = {
    "rec_yds": [(99.5, "100+ rec yds"), (124.5, "125+ rec yds")],
    "rush_yds": [(99.5, "100+ rush yds"), (124.5, "125+ rush yds")],
    "receptions": [(7.5, "8+ catches"), (9.5, "10+ catches")],
}
MIN_FACTORS = 3
MODEL_P_BAR = 0.16          # don't surface a "longshot" the model itself thinks is <16%
FACTOR_TILT = {"role": 0.08, "environment": 0.07, "script": 0.05, "pace": 0.04, "matchup": 0.03}
MAX_TILT = 0.22


def implied(o):
    return 100/(o+100) if o > 0 else abs(o)/(abs(o)+100)


def factor_stack(row, env_total=None, script=None, pace=None, matchup=None):
    """Return (present_factors, tilt) for one player row. Only appends a factor when
    the signal is genuinely positive; the tilt is the summed, capped multiplier."""
    present, tilt = [], 0.0
    # role: projected volume in the top of its position
    vol = row.get("proj_targets_pg", 0) + row.get("proj_carries_pg", 0)
    if vol >= 7.0 or row.get("_role_pct", 0) >= 0.80:
        present.append("role"); tilt += FACTOR_TILT["role"]
    if env_total is not None and env_total >= 25.0:     # implied team total, high
        present.append("environment"); tilt += FACTOR_TILT["environment"]
    if script:                                          # caller passes a truthy reason
        present.append(f"script:{script}"); tilt += FACTOR_TILT["script"]
    if pace:
        present.append("pace"); tilt += FACTOR_TILT["pace"]
    if matchup:
        present.append("matchup"); tilt += FACTOR_TILT["matchup"]
    return present, float(np.clip(tilt, 0, MAX_TILT))


def find_smash_spots(proj, env_by_team=None, script_by_player=None, pace_by_team=None,
                     matchup_by_player=None, posted=None, min_factors=MIN_FACTORS):
    """proj: DataFrame from props_projection.project (has proj means + volumes).
    posted (optional): DataFrame[name, market, threshold, price] of live longshot lines.
    Returns ranked smash spots with model ceiling P, factors, and edge vs any posted line."""
    env_by_team = env_by_team or {}
    script_by_player = script_by_player or {}
    pace_by_team = pace_by_team or {}
    matchup_by_player = matchup_by_player or {}
    # role percentile within position (for the role factor)
    proj = proj.copy()
    proj["_role_pct"] = proj.groupby("position")["rec_yds"].rank(pct=True)
    rows = []
    for _, r in proj.iterrows():
        present, tilt = factor_stack(
            r, env_total=env_by_team.get(r["team"]),
            script=script_by_player.get(r["name"]), pace=pace_by_team.get(r["team"]),
            matchup=matchup_by_player.get(r["name"]))
        n_fac = len([p for p in present if not p.startswith("script:")]) + \
                (1 if any(p.startswith("script:") for p in present) else 0)
        if n_fac < min_factors:
            continue
        for mkt, ceils in CEILINGS.items():
            mu = r.get(mkt, 0)
            if not mu or mu <= 0:
                continue
            mu_t = mu * (1 + tilt)
            for thr, label in ceils:
                # only the nearest ceiling worth quoting (mu within reach)
                if mu_t < thr * 0.55:
                    continue
                p = pm.p_over(mkt, mu_t, thr)
                if p < MODEL_P_BAR:
                    continue
                row = {"name": r["name"], "pos": r["position"], "team": r["team"],
                       "market": mkt, "ceiling": label, "threshold": thr,
                       "proj_mean": round(mu, 1), "tilt_mean": round(mu_t, 1),
                       "model_p": round(p, 3), "n_factors": n_fac,
                       "factors": ", ".join(present), "confidence": r.get("confidence")}
                if posted is not None:
                    m = posted[(posted["name"] == r["name"]) & (posted["market"] == mkt)
                               & (abs(posted["threshold"] - thr) < 1)]
                    if len(m):
                        price = m.iloc[0]["price"]
                        row["price"] = price
                        row["mkt_p"] = round(implied(price), 3)
                        row["edge"] = round(p - implied(price), 3)
                rows.append(row)
    out = pd.DataFrame(rows)
    if not len(out):
        return out
    sort_key = "edge" if "edge" in out.columns else "model_p"
    return out.sort_values(sort_key, ascending=False)


if __name__ == "__main__":
    import props_projection as pp
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    proj = pd.read_parquet(f"../data/projection_{season}.parquet") \
        if os.path.exists(f"../data/projection_{season}.parquet") else pp.project(season)
    # demo: role-only factors (env/script/pace come from game odds at runtime).
    # relax to min_factors=1 for the preseason ceiling board (role signal only).
    spots = find_smash_spots(proj, min_factors=1)
    cols = ["name", "pos", "team", "ceiling", "proj_mean", "model_p", "factors"]
    print(f"=== {season} ceiling board (role signal only; env/script stack in live) ===")
    print(spots[cols].head(20).to_string(index=False) if len(spots) else "none")
