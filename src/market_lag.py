"""Market lag — which of our tilts does the market ALREADY price? (Fresh Eyes day 2)

Under the market-role baseline (a player with no current-season games) our number is the
market's implied mean plus our tilts. A tilt only adds edge if the market does NOT price it.
That is measurable every week from the board itself: regress (market mean − our own season
baseline) on our raw features across the slate, mean-reversion controlled. A coefficient near
our own earned weight means the market's line IS our baseline plus that tilt — adding it again
double-counts (the base_c and total_c lessons); a coefficient near zero means the market lags
the tilt and the edge is real.

Week 1 2026 (n=32/52/84/83, bootstrap 90% CIs): the market prices the game total and home
exactly as our weights do (pass 1.14/pt vs our 1.20; home −4.9 vs −4.1; R² .82), prices
late-season QB passing form (+.144 vs our applied +.057, CI [+.07, +.21]) and receptions form
(+.140 vs +.150, wide CI), and does NOT price rushing (−.116, CI [−.28, +.02]) or receiving
yards form (+.004, CI [−.10, +.10]). So the Week-1 form edge lives in rush/rec yards, not in
QB yards or catch counts.

Applied as `lag_factor[mkt] = clip(1 − max(0, market_coef) / our_applied_coef, 0, 1)` on the
form tilt in market-baseline mode; pooled across weeks (n-weighted) as boards accumulate so a
single week's noise fades. data/market_lag.json holds the pooled table + per-week entries.
"""
import os, json
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PATH = os.path.join(DATA, "market_lag.json")
FEATS = ["total_c", "base_c", "form3", "form1"]
MIN_N = 25


def fit(rows, applied, seed=0, boots=1000):
    """rows: [{mkt, gap, total_c, base_c, form3, form1}]; applied: {mkt: our applied form3 weight}.
    Returns {mkt: {n, coef: {...}, ci_form3, lag_factor}}."""
    out = {}
    rng = np.random.default_rng(seed)
    by = {}
    for r in rows:
        by.setdefault(r["mkt"], []).append(r)
    for mkt, rs in by.items():
        n = len(rs)
        if n < MIN_N or not applied.get(mkt):
            out[mkt] = {"n": n, "coef": None, "lag_factor": 1.0, "note": "too few rows — form applied in full"}
            continue
        X = np.column_stack([np.ones(n)] + [np.array([float(r[f] or 0.0) for r in rs]) for f in FEATS])
        y = np.array([float(r["gap"]) for r in rs])
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        pred = X @ c; r2 = float(1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9))
        bs = []
        for _ in range(boots):
            i = rng.integers(0, n, n)
            try: bs.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0][3])
            except Exception: pass
        bs = np.array(bs) if bs else np.array([c[3]])
        coef = {f: round(float(v), 4) for f, v in zip(FEATS, c[1:])}
        a = float(applied[mkt])
        lag = float(np.clip(1.0 - max(0.0, coef["form3"]) / a, 0.0, 1.0)) if a > 0 else 1.0
        out[mkt] = {"n": n, "coef": coef, "r2": round(r2, 3), "ci90_form3": [round(float(np.quantile(bs, .05)), 4), round(float(np.quantile(bs, .95)), 4)],
                    "applied_form3": round(a, 4), "lag_factor": round(lag, 3)}
    return out


def pooled(entries):
    """n-weighted pooled form3 coefficient per market over the stored weeks -> lag factors."""
    agg = {}
    for e in entries:
        for mkt, v in e.get("by_market", {}).items():
            if not v.get("coef"):
                continue
            a = agg.setdefault(mkt, {"n": 0, "wc": 0.0, "applied": v.get("applied_form3")})
            a["n"] += v["n"]; a["wc"] += v["n"] * v["coef"]["form3"]
    out = {}
    for mkt, a in agg.items():
        coef = a["wc"] / a["n"] if a["n"] else 0.0
        ap = a.get("applied") or 0.0
        out[mkt] = {"n": a["n"], "form3_coef": round(coef, 4), "applied_form3": ap,
                    "lag_factor": round(float(np.clip(1.0 - max(0.0, coef) / ap, 0.0, 1.0)) if ap > 0 else 1.0, 3)}
    return out


def update(season, week, by_market):
    """Store this week's fit and refresh the pooled table. Returns the pooled lag factors."""
    d = json.load(open(PATH)) if os.path.exists(PATH) else {"weeks": []}
    key = f"{season}_w{int(week):02d}" if week is not None else "unknown"
    d["weeks"] = [w for w in d["weeks"] if w.get("key") != key]
    d["weeks"].append({"key": key, "season": season, "week": week, "by_market": by_market})
    d["weeks"].sort(key=lambda w: (w.get("season") or 0, w.get("week") or 0))
    d["pooled"] = pooled(d["weeks"])
    json.dump(d, open(PATH, "w"), indent=1)
    return d["pooled"]


def load_factors():
    """{mkt: lag_factor} from the pooled table (1.0 when nothing is stored)."""
    if not os.path.exists(PATH):
        return {}
    d = json.load(open(PATH))
    return {m: v["lag_factor"] for m, v in d.get("pooled", {}).items()}
