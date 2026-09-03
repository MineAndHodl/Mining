"""Market lag — which of our tilts does the market ALREADY price? (Fresh Eyes day 2, generalized day 3)

Under the market-role baseline (a player with no current-season games) our number is the
market's implied mean plus our tilts. A tilt only adds edge if the market does NOT price it.
That is measurable every week from the board itself: regress (market mean − our own season
baseline) on our raw features across the slate, mean-reversion controlled. A coefficient near
our own applied weight means the market's line IS our baseline plus that tilt — adding it again
double-counts (the base_c and total_c lessons); a coefficient near zero means the market lags
the tilt and the edge is real.

Week 1 2026 (n=32/52/84/83, bootstrap 90% CIs): the market prices the game total and home
exactly as our weights do (pass 1.14/pt vs our 1.20; home −4.9 vs −4.1; R² .82), prices
late-season QB passing form (+.144 vs our applied +.057, CI [+.07, +.21]) and receptions form
(+.140 vs +.150, wide CI), and does NOT price rushing (−.116, CI [−.28, +.02]) or receiving
yards form (+.004, CI [−.10, +.10]). So the Week-1 form edge lives in rush/rec yards, not in
QB yards or catch counts.

Applied per market: the market's gap is regressed on OUR total applied tilt (form + role, in
yards); β = the share of our tilt the market already prices; `lag = clip(1 − β, 0, 1)` scales
every tilt in that market in market-baseline mode. Pooled across weeks (n-weighted) as boards
accumulate so a single week's noise fades.
data/market_lag.json holds the pooled table + per-week entries.
"""
import os, json
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PATH = os.path.join(DATA, "market_lag.json")
CONTROLS = ["total_c", "base_c"]                     # the market prices these fully (measured); controls only
TILTS = ["form3", "form1", "role_rr", "role_snap_d"]  # the tilts we apply under the market baseline
MIN_N = 25


def fit(rows, applied, seed=0, boots=800):
    """rows: [{mkt, gap, total_c, base_c, form3, form1, role_rr, role_snap_d}];
    applied: {mkt: {feature: our applied weight}}.
    ONE coefficient per market (day 3): the market's gap is regressed on OUR TOTAL APPLIED TILT
    (Σ weight × feature, in the market's units) with the controls — β is the fraction of what
    we are about to add that the market has already priced; lag = clip(1 − β, 0, 1) applies to
    every tilt in that market. Per-feature coefficients with n = 30-80 rows were unstable
    (collinear role/form terms flipped sign week to week); one β is the estimable quantity.
    Returns {mkt: {n, beta, ci90, r2, coef (controls), lag: {feature: factor}}}."""
    out = {}
    rng = np.random.default_rng(seed)
    by = {}
    for r in rows:
        by.setdefault(r["mkt"], []).append(r)
    for mkt, rs in by.items():
        n = len(rs); ap = applied.get(mkt) or {}
        if n < MIN_N or not any(ap.values()):
            out[mkt] = {"n": n, "beta": None, "lag": {t: 1.0 for t in TILTS}, "note": "too few rows — tilts applied in full"}
            continue
        tilt = np.array([sum(float(ap.get(t) or 0.0) * float(r.get(t) or 0.0) for t in TILTS) for r in rs])
        X = np.column_stack([np.ones(n)] + [np.array([float(r.get(f) or 0.0) for r in rs]) for f in CONTROLS] + [tilt])
        y = np.array([float(r["gap"]) for r in rs])
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        pred = X @ c; r2 = float(1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9))
        bs = []
        for _ in range(boots):
            i = rng.integers(0, n, n)
            try: bs.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0][-1])
            except Exception: pass
        bs = np.array(bs) if bs else np.array([c[-1]])
        beta = float(c[-1])
        lagv = round(float(np.clip(1.0 - max(0.0, beta), 0.0, 1.0)), 3)
        out[mkt] = {"n": n, "beta": round(beta, 4), "ci90": [round(float(np.quantile(bs, .05)), 4), round(float(np.quantile(bs, .95)), 4)],
                    "r2": round(r2, 3), "coef": {f: round(float(v), 4) for f, v in zip(CONTROLS, c[1:1 + len(CONTROLS)])},
                    "tilt_sd": round(float(tilt.std()), 3), "applied": {t: round(float(ap.get(t) or 0.0), 4) for t in TILTS},
                    "lag": {t: lagv for t in TILTS}}
    return out


def pooled(entries):
    """n-weighted pooled β per market over the stored weeks -> lag factors."""
    agg = {}
    for e in entries:
        for mkt, v in e.get("by_market", {}).items():
            if v.get("beta") is None:
                continue
            a = agg.setdefault(mkt, {"n": 0, "wb": 0.0})
            a["n"] += v["n"]; a["wb"] += v["n"] * v["beta"]
    out = {}
    for mkt, a in agg.items():
        beta = a["wb"] / a["n"] if a["n"] else 0.0
        lagv = round(float(np.clip(1.0 - max(0.0, beta), 0.0, 1.0)), 3)
        out[mkt] = {"n": a["n"], "beta": round(beta, 4), "lag": {t: lagv for t in TILTS}}
    return out


def update(season, week, by_market):
    """Store this week's fit and refresh the pooled table. Returns the pooled table."""
    d = json.load(open(PATH)) if os.path.exists(PATH) else {"weeks": []}
    key = f"{season}_w{int(week):02d}" if week is not None else "unknown"
    d["weeks"] = [w for w in d["weeks"] if w.get("key") != key]
    d["weeks"].append({"key": key, "season": season, "week": week, "by_market": by_market})
    d["weeks"].sort(key=lambda w: (w.get("season") or 0, w.get("week") or 0))
    d["pooled"] = pooled(d["weeks"])
    json.dump(d, open(PATH, "w"), indent=1)
    return d["pooled"]


def load_factors():
    """{mkt: {tilt: lag_factor}} from the pooled table (empty when nothing is stored)."""
    if not os.path.exists(PATH):
        return {}
    d = json.load(open(PATH))
    return {m: v.get("lag", {}) for m, v in d.get("pooled", {}).items()}
