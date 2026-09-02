"""Correlation lab — measure TRUE leg-to-leg correlation from our own game data, combine
correlated legs the way a book's SGP engine should (Gaussian copula), and price the FAIR
same-game-parlay number.

Why this matters (the real edge in SGPs):
A book's SGP price applies a correlation haircut to the naive product of the singles. If the
book UNDER-adjusts for a correlation we've actually measured, the SGP is +EV. We can't see
DK's SGP quote (no API, no scraping), so instead we compute the FAIR odds from true
correlation and use it as the benchmark: bet the SGP when DK's quote is LONGER than our fair
number. We also answer the line-selection question quantitatively — does taking 30 rec yds
instead of 37 raise EV, and by how much, once correlation is accounted for.

Data: data/prop_player_games.parquet (pid, game_id, season, week, posteam, y, mkt), 2023-25.
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import props_model as pm

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_RNG = np.random.default_rng(7)


def _roles_per_game(pg):
    """Per (game_id, team): the passer's pass_yds, top-2 receivers' rec_yds, lead RB rush_yds.
    Returns a long frame we can pair up to measure correlations."""
    rows = []
    for (gid, team), g in pg.groupby(["game_id", "posteam"]):
        rec = g[g.mkt == "rec_yds"].sort_values("y", ascending=False)
        rush = g[g.mkt == "rush_yds"].sort_values("y", ascending=False)
        passer = g[g.mkt == "pass_yds"].sort_values("y", ascending=False)
        row = {"game_id": gid, "team": team}
        if len(passer): row["qb_pass"] = float(passer.iloc[0]["y"])
        if len(rec) >= 1: row["wr1_rec"] = float(rec.iloc[0]["y"])
        if len(rec) >= 2: row["wr2_rec"] = float(rec.iloc[1]["y"])
        if len(rush): row["rb_rush"] = float(rush.iloc[0]["y"])
        rows.append(row)
    return pd.DataFrame(rows)


def measure_correlations(pg=None):
    """Pearson correlations for the leg archetypes an SGP is built from, plus opposing-team
    (shootout) pairs. These are the true, sample-measured correlations we combine legs with."""
    pg = pg if pg is not None else pd.read_parquet(f"{DATA}/prop_player_games.parquet")
    rpg = _roles_per_game(pg)
    def corr(a, b, frame=None):
        f = (frame if frame is not None else rpg).dropna(subset=[a, b])
        return (round(float(np.corrcoef(f[a], f[b])[0, 1]), 3), int(len(f))) if len(f) > 30 else (None, len(f))
    out = {}
    # same-team pairs
    for label, a, b in [("qb_pass__wr1_rec", "qb_pass", "wr1_rec"),
                        ("qb_pass__wr2_rec", "qb_pass", "wr2_rec"),
                        ("wr1_rec__wr2_rec", "wr1_rec", "wr2_rec"),
                        ("qb_pass__rb_rush", "qb_pass", "rb_rush"),
                        ("wr1_rec__rb_rush", "wr1_rec", "rb_rush")]:
        c, n = corr(a, b); out[label] = {"corr": c, "n": n, "scope": "same_team"}
    # opposing-team (shootout) pairs: join the two teams in each game
    opp = rpg.merge(rpg, on="game_id", suffixes=("_a", "_b"))
    opp = opp[opp["team_a"] < opp["team_b"]]
    for label, a, b in [("opp_qb__opp_qb", "qb_pass_a", "qb_pass_b"),
                        ("qb_pass__opp_wr1", "qb_pass_a", "wr1_rec_b")]:
        c, n = corr(a, b, frame=opp); out[label] = {"corr": c, "n": n, "scope": "opposing"}
    return out


# correlation LOOKUP for building an SGP's matrix from leg relationships. Values are our
# MEASURED co-movements (2023-25, correlation_measured.json), lightly shrunk (x0.85) because
# raw Pearson of counting stats slightly overstates the copula correlation via the shared game
# total. Same player across markets (yards+catches) is strongly correlated (~0.8).
_SHRINK = 0.85
REL_CORR = {"same_player": 0.80,
            "qb_wr_same": round(0.696 * _SHRINK, 2),    # 0.59  (was mis-measured at 0.16)
            "qb_wr2_same": round(0.753 * _SHRINK, 2),   # 0.64
            "wr_wr_same": round(0.507 * _SHRINK, 2),    # 0.43  (NOT independent)
            "qb_rb_same": -0.08, "wr_rb_same": 0.0,
            "opp_qbs": round(0.164 * _SHRINK, 2),       # 0.14
            "qb_opp_wr": round(0.125 * _SHRINK, 2),     # 0.11
            "unrelated": 0.0}


def joint_hit_prob(marg_ps, corr_matrix, n=60000):
    """TRUE joint P(all legs hit) via a Gaussian copula. marg_ps: per-leg hit probs. Each leg
    hits when its latent normal exceeds a threshold; we draw correlated normals and count."""
    k = len(marg_ps)
    if k == 0:
        return 1.0
    R = np.array(corr_matrix, float)
    R = (R + R.T) / 2
    np.fill_diagonal(R, 1.0)
    # nearest-PSD nudge
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-6, None); R = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(R)); R = R / np.outer(d, d)
    L = np.linalg.cholesky(R)
    Z = (_RNG.standard_normal((n, k)) @ L.T)
    from scipy.stats import norm
    thr = norm.ppf(1 - np.clip(np.array(marg_ps), 1e-4, 1 - 1e-4))
    return float(np.mean(np.all(Z > thr, axis=1)))


def fair_sgp(legs):
    """legs: list of {p, price, rel_to[i]=relation-labels vs earlier legs}. Returns naive vs
    true joint prob, correlation lift, and fair American odds to compare DK's SGP quote to."""
    ps = [l["p"] for l in legs]
    k = len(legs)
    R = np.eye(k)
    for i in range(k):
        for j in range(i):
            rel = legs[i].get("rel", {}).get(j, "unrelated")
            R[i, j] = R[j, i] = REL_CORR.get(rel, 0.0)
    naive = float(np.prod(ps))
    joint = joint_hit_prob(ps, R)
    fair_dec = 1.0 / max(joint, 1e-6)
    fair_am = round((fair_dec - 1) * 100) if fair_dec >= 2 else round(-100 / (fair_dec - 1))
    return {"naive_prob": round(naive, 5), "true_joint_prob": round(joint, 5),
            "correlation_lift": round(joint / max(naive, 1e-9), 2),
            "fair_decimal": round(fair_dec, 1), "fair_american": fair_am}


def line_ev(mkt, proj_mean, lines, price_by_line=None):
    """The '30 vs 37' analysis: for a projected mean, P(over) and EV at each candidate line.
    Lower line = higher hit prob but shorter price. Shows where EV and hit-rate actually land."""
    price_by_line = price_by_line or {}
    rows = []
    for ln in lines:
        p = pm.p_over(mkt, proj_mean, ln)
        price = price_by_line.get(ln, -110)
        prof = price / 100 if price > 0 else 100 / abs(price)
        ev = p * prof - (1 - p)
        rows.append({"line": ln, "p_over": round(p, 3), "price": price, "ev_per_$1": round(ev, 3)})
    return pd.DataFrame(rows)


def validate_copula(pg=None, min_games=8):
    """SELF-TRAINING: check whether our copula's predicted joint hit-rate matches what actually
    happened. For real stacks (QB+WR1, QB+WR1+WR2, ...), define each leg as 'over the player's
    own median' (so each leg is ~50% by construction) and measure the REALIZED rate that all
    legs cleared together, vs the naive-independent prediction and our copula prediction. If the
    copula tracks reality, the correlation inputs are right; if not, we shrink/raise them."""
    pg = pg if pg is not None else pd.read_parquet(f"{DATA}/prop_player_games.parquet")
    rpg = _roles_per_game(pg)
    # each player's own median per role-column (the proxy line -> ~50% marginal)
    med = {c: rpg[c].median() for c in ["qb_pass", "wr1_rec", "wr2_rec", "rb_rush"]}
    for c in med:  # over-median indicator
        rpg[c + "_hit"] = (rpg[c] > med[c]).astype(float)
    stacks = {"QB+WR1": ["qb_pass", "wr1_rec"],
              "QB+WR1+WR2": ["qb_pass", "wr1_rec", "wr2_rec"],
              "QB+WR1+WR2+RB": ["qb_pass", "wr1_rec", "wr2_rec", "rb_rush"]}
    rel = {("qb_pass", "wr1_rec"): "qb_wr_same", ("qb_pass", "wr2_rec"): "qb_wr2_same",
           ("wr1_rec", "wr2_rec"): "wr_wr_same", ("qb_pass", "rb_rush"): "qb_rb_same",
           ("wr1_rec", "rb_rush"): "wr_rb_same", ("wr2_rec", "rb_rush"): "wr_rb_same"}
    out = []
    for name, cols in stacks.items():
        f = rpg.dropna(subset=cols)
        if len(f) < 50:
            continue
        hitcols = [c + "_hit" for c in cols]
        realized = float((f[hitcols].sum(axis=1) == len(cols)).mean())
        margins = [float(f[c + "_hit"].mean()) for c in cols]
        naive = float(np.prod(margins))
        legs = [{"p": margins[i], "rel": {j: rel.get((cols[j], cols[i]), rel.get((cols[i], cols[j]), "unrelated"))
                                          for j in range(i)}} for i in range(len(cols))]
        pred = fair_sgp([{**l, "price": -110} for l in legs])["true_joint_prob"]
        out.append({"stack": name, "n_games": len(f), "realized": round(realized, 3),
                    "copula_pred": round(pred, 3), "naive_indep": round(naive, 3),
                    "copula_err": round(pred - realized, 3), "naive_err": round(naive - realized, 3)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    print("=== TRUE leg correlations (measured, 2023-25) ===")
    c = measure_correlations()
    for k, v in c.items():
        print(f"  {k:22s} corr={str(v['corr']):>6}  (n={v['n']}, {v['scope']})")
    json.dump(c, open(f"{DATA}/correlation_measured.json", "w"), indent=1)

    print("\n=== FAIR SGP vs naive: a QB + his 2 WRs + WR1 catches (correlated) ===")
    legs = [{"p": 0.55, "price": -120, "rel": {}},
            {"p": 0.52, "price": -110, "rel": {0: "qb_wr_same"}},
            {"p": 0.50, "price": -110, "rel": {0: "qb_wr2_same", 1: "wr_wr_same"}},
            {"p": 0.58, "price": -140, "rel": {0: "qb_wr_same", 1: "same_player", 2: "wr_wr_same"}}]
    f = fair_sgp(legs)
    print(f"  naive prob {f['naive_prob']} -> true joint {f['true_joint_prob']} "
          f"(correlation lift {f['correlation_lift']}x)")
    print(f"  FAIR odds {f['fair_american']:+d} (dec {f['fair_decimal']}) — bet DK's SGP only if it's LONGER than this")

    print("\n=== line selection: WR projected 41 rec yds, which line? ===")
    print(line_ev("rec_yds", 41.0, [24.5, 30.5, 37.5, 44.5],
                  {24.5: -160, 30.5: -130, 37.5: -110, 44.5: 105}).to_string(index=False))
