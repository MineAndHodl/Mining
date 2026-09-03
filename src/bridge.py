"""The bridge — how we get to the number, the probability, the edge and the grade, on every pick.

Alex: "All the different ranking things should have a composite that shows how we get to our
% over/under-perform / edge… how do the factors interact and get weighted and come up with what
really matters — the pick and the confidence interval."

The model is additive, then distributional, then a score — so it can be shown as four honest steps:

  1. THE NUMBER (a waterfall, in the stat's units)
       start   = the baseline we build on (the market's implied mean before Week 4; his own
                 season baseline once he has current-season games; the role projection for a
                 player with no game log)
       + each factor's full read = weight × signal (recent form, last game, route participation,
         snap-share trend; in season also matchup, game total, venue, mean reversion)
       − "already in the line": the part of those reads the market has — the measured offseason
         discount on last season's form and the measured share of our tilt the market already
         moved. Shown as a bar, never hidden.
       × kickoff weather, × the injury cascade
       = our number, with the distribution's 25-75% and 10-90% ranges (the confidence interval)
  2. THE PROBABILITY: P(over the line) from that distribution vs the market's de-vigged fair
       → the edge in probability points
  3. THE SCORE: Desk Score = [45 + value(edge) + hittable(p)]  (conviction)
                              + [sample depth + market breadth + beat intel]  (trust)
                              × the investigate verdict
  4. THE GRADE: bands → rails (public-record floor, early-season cap, per-thesis caps) → units

Touchdowns get the same four steps from the decomposed xTD model: expected TDs by zone and
type (chances per game × finishing rate), × the scoring environment (the team's implied points),
× the injury cascade, × red-zone defense → P(score) raw → calibrated → blended with the pooled
model → vs the best price → edge → the TD-scale score.

Every number here is already on the row; this module only arranges it, and it checks that the
arithmetic closes (the residual is written as `unexplained`; the board run reports any > 1%).
"""
import os, math
import numpy as np
from scipy.stats import gamma, norm

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
import sys; sys.path.insert(0, HERE)
import props_model as pm
import rank_board as rbk
import multifactor as mf

LABEL = {"form3": "Recent form (last 3)", "form1": "Last game", "role_rr": "Route participation", "role_snap_d": "Snap-share trend",
         "def_delta": "Matchup", "total_c": "Game total", "home": "Venue", "base_c": "Mean reversion"}
TILTS = ("form3", "form1", "role_rr", "role_snap_d")          # what the market can lag
ENV = ("def_delta", "total_c", "home", "base_c")             # what the market's number already prices (weeks 0-3)
UNIT = {"rec_yds": "yds", "rush_yds": "yds", "pass_yds": "yds", "receptions": "catches", "rush_att": "carries"}
MIN_BAR = 0.05

_W = None
def _weights():
    """The earned ridge weights (data/multifactor_weights.json) — so every bar can say
    'signal × weight'."""
    global _W
    if _W is None:
        try:
            import json
            _W = json.load(open(os.path.join(DATA, "multifactor_weights.json"))).get("weights", {})
        except Exception:
            _W = {}
    return _W


def factor_note(mkt, f, contrib_value):
    """'his last 3 averaged +15.6 yds vs his baseline × weight 0.27' — the signal in its own
    units and the weight it earned, recovered from the contribution (weight × signal)."""
    w = (_weights().get(mkt) or {}).get(f)
    if not w:
        return None
    sig = contrib_value / w; u = UNIT.get(mkt, "")
    if f == "form3": return f"his last 3 averaged {sig:+.1f} {u} vs his baseline × weight {w:.2f}"
    if f == "form1": return f"his last game ran {sig:+.1f} {u} vs his baseline × weight {w:.2f}"
    if f == "role_rr": return f"route rate {sig*100:+.0f} pts vs the market's typical player × weight {w:.1f}"
    if f == "role_snap_d": return f"snap share {sig*100:+.0f} pts, last 3 vs season × weight {w:.1f}"
    if f == "def_delta": return f"the matchup {sig:+.2f} vs an average defense × weight {w:.1f}"
    if f == "total_c": return f"game total {sig:+.1f} pts vs 44 × weight {w:.2f}"
    if f == "home": return f"home game × weight {w:+.2f}"
    if f == "base_c": return f"his baseline sits {sig:+.1f} {u} vs the market's typical player × weight {w:.2f}"
    return None


def _f(x, default=None):
    try:
        v = float(x); return default if v != v else v
    except Exception:
        return default


def quantiles(mkt, mu, qs=(0.10, 0.25, 0.50, 0.75, 0.90)):
    """Our distribution's quantiles at mean mu (the same shape p_over uses)."""
    mu = max(float(mu), 0.1); cv = pm.cv_for(mkt, mu); d = pm.DIST.get(mkt, "gamma")
    out = {}
    for q in qs:
        if d == "gamma":
            k = 1.0 / (cv ** 2); v = gamma.ppf(q, a=k, scale=mu / k)
        elif d == "normal":
            v = norm.ppf(q, loc=mu, scale=mu * cv)
        else:
            var = mu + (mu * cv) ** 2; v = norm.ppf(q, loc=mu, scale=math.sqrt(var))
        out[f"p{int(q * 100)}"] = round(float(max(v, 0.0)), 1)
    return out


def score_block(pick, anchor=0.52, use_trust=True):
    """Desk Score = (conviction + trust) × verdict → grade → rails → units, in pieces."""
    edge = _f(pick.get("edge")); p = _f(pick.get("model_p"))
    parts = rbk.desk_score_parts(edge, p, anchor=anchor)
    tr = rbk.trust_parts(pick) if use_trust else None
    mult = _f(pick.get("score_mult"), 1.0) or 1.0; score = _f(pick.get("score"))
    txt = f"{parts['base']:g} + value {parts['value']:+g} + hittable {parts['hittable']:+g} = conviction {parts['conviction']:g}"
    if tr is not None:
        txt += f"; trust {tr['trust']:+g} (sample {tr['sample']:+g}, breadth {tr['breadth']:+g}, intel {tr['intel']:+g})"
    if abs(mult - 1) > 1e-6:
        txt += f"; × {mult:g} verdict"
    if score is not None:
        txt += f" → Desk Score {score:g}"
    return {"conviction": parts, "trust": tr, "mult": mult, "verdict": pick.get("verdict"), "score": score,
            "grade": pick.get("grade"), "grade_sized": pick.get("grade_sized"), "units": pick.get("units"),
            "rail": pick.get("rail"), "cap_note": pick.get("cap_note"), "recommended": bool(pick.get("recommended")), "text": txt}


def yardage_bridge(pick):
    mkt = pick.get("cat") or pick.get("mkt"); side = pick.get("side", "Over")
    line = _f(pick.get("line")); our = _f(pick.get("our_num")); start = _f(pick.get("bridge_start"))
    contrib = pick.get("contrib") or {}; raw = pick.get("contrib_raw") or {}
    if our is None or start is None or mkt in (None, "anytime_td"):
        return None
    offseason = bool(pick.get("offseason")); gap = _f(pick.get("form_gap")); lag = _f(pick.get("market_lag"))
    steps = []; in_line = []; off = []
    if contrib:
        # the tilts at their FULL read (weight × signal), then what the market already has
        priced = 0.0
        missed = gap is not None and gap >= 2
        for f in TILTS:
            r_ = _f(raw.get(f), _f(contrib.get(f), 0.0)); a_ = _f(contrib.get(f), 0.0)
            # a role tilt switched OFF for this market before Week 4 (measured: no early-season
            # signal) is not "in the line" — it is simply not applied yet; say so, no bar
            if offseason and not missed and f in ("role_rr", "role_snap_d") and mf.OFFSEASON_ROLE_RATIO.get(f, {}).get(mkt, 0.5) == 0:
                off.append(LABEL[f]); continue
            if abs(r_) >= MIN_BAR:
                steps.append({"label": LABEL[f], "delta": round(r_, 2), "kind": "factor", "key": f, "note": factor_note(mkt, f, r_)})
            priced += (r_ - a_)
        if abs(priced) >= MIN_BAR:
            if missed:
                steps.append({"label": "Set aside — missed last season", "delta": round(-priced, 2), "kind": "priced",
                              "note": "his last games were two seasons ago: the form read is unmeasurable, so it is not applied and trust is cut"})
            else:
                ratio = mf.OFFSEASON_FORM_RATIO.get(mkt, mf.OFFSEASON_FORM_DEFAULT)
                bits = []
                if offseason: bits.append(f"form keeps {ratio:.0%} of its weight across an offseason (measured)")
                if lag is not None and lag < 0.999: bits.append(f"the market has already moved {1-lag:.0%} of the rest (measured this week)")
                steps.append({"label": ("Faded or already priced" if offseason else "Already in the line"), "delta": round(-priced, 2), "kind": "priced",
                              "note": "; ".join(bits) if bits else "the part of our read the market already prices"})
        # the environment: bars in season; under the market's number it is already priced (chips)
        for f in ENV:
            a_ = _f(contrib.get(f), 0.0)
            if abs(a_) >= MIN_BAR:
                steps.append({"label": LABEL[f], "delta": round(a_, 2), "kind": "factor", "key": f, "note": factor_note(mkt, f, a_)})
            elif offseason:
                in_line.append(LABEL[f])
        # tiny leftovers (below the bar threshold) so the arithmetic still closes
        shown = sum(s_["delta"] for s_ in steps)
        leftover = sum(_f(contrib.get(f), 0.0) for f in contrib) - shown
        if abs(leftover) >= MIN_BAR:
            steps.append({"label": "Smaller factors", "delta": round(leftover, 2), "kind": "factor"})
    after_factors = start + sum(_f(x.get("delta"), 0.0) for x in steps)
    cur = after_factors
    wx = _f(pick.get("wx_mult"), 1.0); rm = _f(pick.get("role_mult"), 1.0)
    if abs(wx - 1.0) >= 0.0005:
        d = cur * (wx - 1.0)
        note = f"×{wx:.3f}"
        if pick.get("wx_wind") is not None: note += f" — {int(_f(pick.get('wx_wind'), 0))} mph wind"
        if pick.get("wx_temp") is not None: note += f", {int(_f(pick.get('wx_temp'), 0))}°F"
        steps.append({"label": "Kickoff weather", "delta": round(d, 2), "kind": "mult", "note": note}); cur += d
    if abs(rm - 1.0) >= 0.0005:
        d = cur * (rm - 1.0)
        steps.append({"label": "Injury cascade", "delta": round(d, 2), "kind": "mult", "note": f"×{rm:.3f}" + (f" — {pick.get('role_note')}" if pick.get("role_note") else "")}); cur += d
    unexplained = round(our - cur, 2)
    if abs(unexplained) >= 0.05:
        steps.append({"label": "Rounding", "delta": unexplained, "kind": "residual"})
    qs = quantiles(mkt, our)
    fair = _f(pick.get("fair")); p = _f(pick.get("model_p")); edge = _f(pick.get("edge"))
    if fair is None and p is not None and edge is not None:
        fair = p - edge
    mnum = _f(pick.get("market_num"))
    if not contrib:
        start_label = "Role projection (no game log yet)"
    elif offseason:
        start_label = "The market's number" if _f(pick.get("cons_mean")) is not None else "The book's implied number"
    else:
        start_label = "His season baseline"
    return {"kind": "yardage", "unit": UNIT.get(mkt, ""), "side": side, "line": line,
            "start": {"label": start_label, "value": round(start, 1),
                      "note": ("median of every book's implied mean — the market's role read; our alpha is the tilt on top" if (offseason and contrib) else
                               ("his current-season average, on which every factor applies at its earned weight" if contrib else
                                "a role-model projection for a player without recent games"))},
            "in_line": in_line, "off": off, "steps": steps, "end": {"label": "Our number", "value": round(our, 1)},
            "ci": qs, "market_ci": (quantiles(mkt, mnum) if mnum else None), "market_num": mnum, "coverage": coverage() or None,
            "prob": {"model_p": p, "fair": (round(fair, 3) if fair is not None else None), "edge": edge, "price": pick.get("price"), "book": pick.get("book"),
                     "text": (f"P({side.lower()} {line:g}) = {p*100:.0f}% on our distribution vs {fair*100:.0f}% fair at the price → {edge*100:+.1f} pts"
                              if (p is not None and fair is not None and edge is not None and line is not None) else None)},
            "score": score_block(pick, anchor=0.52), "unexplained": unexplained}


_COV = None
def coverage():
    """Measured out of sample (interval_check.py): how often the actual stat landed inside the
    likely (25-75%) and wide (10-90%) ranges, pooled over the yardage/count markets."""
    global _COV
    if _COV is None:
        try:
            import json
            _COV = json.load(open(os.path.join(DATA, "interval_coverage.json"))).get("pooled") or {}
        except Exception:
            _COV = {}
    return _COV


_XTD = None
def _xtd_rows():
    global _XTD
    if _XTD is None:
        try:
            import pandas as pd
            from names import norm_name
            t = pd.read_parquet(os.path.join(DATA, "xtd_2026.parquet"))
            _XTD = {norm_name(r["name"]): r for r in t.drop_duplicates("name").to_dict("records")}
        except Exception:
            _XTD = {}
    return _XTD


ZONES = {"GL": "Inside the 5", "IN10": "6-10 yard line", "RZ": "Red zone (11-20)", "OPEN": "Open field"}


def td_bridge(pick):
    from names import norm_name
    r = _xtd_rows().get(norm_name(pick.get("name", "")))
    p_model = _f(pick.get("model_p")) or _f(pick.get("our_num")); edge = _f(pick.get("edge")); imp = _f(pick.get("p_implied"))
    if imp is None and p_model is not None and edge is not None:
        imp = p_model - edge
    if r is None or p_model is None:
        return None
    env = _f(pick.get("env_mult"), 1.0) or 1.0; rm = _f(pick.get("role_mult"), 1.0) or 1.0; rzm = _f(pick.get("rz_mult"), 1.0) or 1.0
    steps = []; total = 0.0
    for z in ("GL", "IN10", "RZ", "OPEN"):
        for t in ("rush", "target"):
            o = _f(r.get(f"{z}_{t}_opps"), 0.0); c = _f(r.get(f"{z}_{t}_conv"), 0.0); lam = o * c
            if lam >= 0.005:
                steps.append({"label": f"{ZONES[z]} — {'carries' if t == 'rush' else 'targets'}", "delta": round(lam, 3), "kind": "factor",
                              "note": f"{o:.2f} chances/game × {c*100:.0f}% finish"})
            total += lam
    xtd = _f(r.get("xtd"), total)
    leftover = xtd - sum(s_["delta"] for s_ in steps)
    if abs(leftover) >= 0.003:
        steps.append({"label": "Smaller chances", "delta": round(leftover, 3), "kind": "factor"})
    cur = xtd
    if abs(env - 1) >= 0.0005:
        d = cur * (env - 1); steps.append({"label": "Scoring environment", "delta": round(d, 3), "kind": "mult", "note": f"×{env:.2f} — the team's implied points today vs its baseline"}); cur += d
    if abs(rm - 1) >= 0.0005:
        d = cur * (rm - 1); steps.append({"label": "Injury cascade", "delta": round(d, 3), "kind": "mult", "note": f"×{rm:.2f}" + (f" — {pick.get('role_note')}" if pick.get("role_note") else "")}); cur += d
    if abs(rzm - 1) >= 0.0005:
        d = cur * (rzm - 1); steps.append({"label": "Red-zone defense", "delta": round(d, 3), "kind": "mult", "note": f"×{rzm:.2f}" + (f" — {pick.get('rz_note')}" if pick.get("rz_note") else "")}); cur += d
    lam_x = _f(pick.get("lam_xtd"), cur)
    unexplained = round(lam_x - cur, 3)
    if abs(unexplained) >= 0.003:
        steps.append({"label": "Rounding", "delta": unexplained, "kind": "residual"})
    p_raw = 1 - math.exp(-lam_x); p_x = _f(pick.get("p_xtd")); p_old = _f(pick.get("p_old")); w = _f(pick.get("xtd_weight"), 0.55)
    lam_pooled = _f(pick.get("lam_pooled"))
    prob = {"lam": round(lam_x, 3), "p_raw": round(p_raw, 3), "p_xtd": p_x, "p_pooled": p_old, "lam_pooled": lam_pooled, "w_xtd": w, "p_model": p_model,
            "implied": (round(imp, 3) if imp is not None else None), "edge": edge, "price": pick.get("price"), "book": pick.get("book"),
            "text": (f"{lam_x:.2f} expected TDs → {p_raw*100:.0f}% raw → {p_x*100:.0f}% calibrated → blended {w:.0%}/{1-w:.0%} with the pooled model ({p_old*100:.0f}%) → {p_model*100:.0f}% vs {imp*100:.0f}% implied at the best price → {edge*100:+.1f} pts"
                     if (p_x is not None and p_old is not None and imp is not None and edge is not None) else None)}
    dist = {"p0": round(math.exp(-lam_x), 3), "p1": round(lam_x * math.exp(-lam_x), 3), "p2plus": round(1 - math.exp(-lam_x) * (1 + lam_x), 3)}
    sb = score_block(pick, anchor=rbk.TD_ANCHOR, use_trust=False)
    if pick.get("mkt_flag"):
        sb["mult"] = 0.8; sb["text"] += " (×0.8: above the whole market on a last-season role read)"
    return {"kind": "td", "unit": "xTD", "side": "Yes", "line": 0.5,
            "start": {"label": "Expected touchdowns, by where his chances come from", "value": 0.0,
                      "note": "last season's zone shares on this season's team tendencies; chances per game × his finishing rate in that zone"},
            "in_line": [], "steps": steps, "end": {"label": "Expected TDs this game", "value": round(lam_x, 2)}, "dist": dist, "prob": prob,
            "score": sb, "unexplained": unexplained}


def build(pick):
    mkt = pick.get("cat") or pick.get("mkt")
    try:
        return td_bridge(pick) if mkt == "anytime_td" else yardage_bridge(pick)
    except Exception as e:
        return {"error": str(e)[:120]}


def attach_all(ranked):
    n = 0
    seen = set()
    for key in ("recommended", "best_props"):
        for r in ranked.get(key, []) or []:
            if r.get("kind"): continue
            b = build(r)
            if b: r["bridge"] = b; n += 1; seen.add(id(r))
    for rows in (ranked.get("by_category") or {}).values():
        for r in rows:
            if id(r) in seen: continue
            b = build(r)
            if b: r["bridge"] = b; n += 1
    return n


def check(ranked, tol_frac=0.01, tol_abs=0.15):
    """Rows whose bridge does not close (or errored)."""
    bad = []
    for key in ("recommended", "best_props"):
        for r in ranked.get(key, []) or []:
            b = r.get("bridge")
            if isinstance(b, dict) and (b.get("error") or abs(_f(b.get("unexplained"), 0.0)) > max(tol_frac * abs(_f(r.get("our_num"), 1.0)), (0.003 if b.get("kind") == "td" else tol_abs))):
                bad.append((key, r.get("name"), r.get("cat"), b.get("error"), b.get("unexplained")))
    for cat, rows in (ranked.get("by_category") or {}).items():
        for r in rows:
            b = r.get("bridge")
            if isinstance(b, dict) and (b.get("error") or abs(_f(b.get("unexplained"), 0.0)) > max(tol_frac * abs(_f(r.get("our_num"), 1.0)), (0.003 if b.get("kind") == "td" else tol_abs))):
                bad.append((cat, r.get("name"), r.get("cat"), b.get("error"), b.get("unexplained")))
    return bad


if __name__ == "__main__":
    import json
    rb = json.load(open(os.path.join(DATA, "ranked_board.json")))
    for r in rb["recommended"][:3] + rb["by_category"]["anytime_td"][:1]:
        b = build(r); print(r["name"], r.get("cat"), r.get("side"), r.get("line"))
        print("  start", b["start"]["label"], b["start"]["value"], "→ end", b["end"]["value"], "| unexplained", b.get("unexplained"), "| in line:", b.get("in_line"))
        for s_ in b["steps"]: print("   ", f"{s_['delta']:+.2f}" if b["kind"] == "yardage" else f"{s_['delta']:+.3f}", s_["label"], "—", s_.get("note", ""))
        print("  ci", b.get("ci") or b.get("dist")); print("  prob", b["prob"]["text"]); print("  score", b["score"]["text"], "→", b["score"]["grade"], b["score"]["units"], "u", b["score"].get("rail") or "")
    print("bad:", check(rb))
