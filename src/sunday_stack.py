"""The Sunday Stack — 4-6 of our own recommended bets, compounded to 6-12:1 (Alex's spec).

Not a lotto. This is the parlay a sharp would build from the slate: our highest-conviction
straight bets, one per game (so the legs are independent and the book prices them as a plain
product — no correlation haircut), distinct theses, combined to a sensible 6-12:1.

Why it can carry alpha: a parlay COMPOUNDS edge. Four independent -110 legs that each truly
hit 57% pay ~+1230 and cash ~11% of the time — about +40% expected ROI, far more than any
single leg (variance compounds too, so it is sized as a parlay, not a straight bet). The
honest caveat: if a leg's edge is overstated, the parlay overstates it more; the forward-
grading loop is the check. We choose the combination with the highest EV inside the band,
report true hit probability and fair odds, and track it as its own record.
"""
import itertools
import numpy as np

BAND = (7.0, 13.0)        # decimal payout: 6:1 .. 12:1
LEG_RANGE = (4, 6)        # Alex: 4-6 good bets
MAX_PER_THESIS = 2
MAX_SAME_SIDE = 3         # an all-Unders stack is one regime bet even across games
POOL_TOP = 12             # combos from the top-N recommended straight bets by Desk Score
MIN_LINE = {"rush_yds": 15.0, "rec_yds": 10.0, "receptions": 2.5, "pass_yds": 150.0}   # no micro-props as parlay legs


def american_to_dec(a):
    a = float(a); return 1 + (a / 100 if a > 0 else 100 / abs(a))


def dec_to_american(d):
    return int(round((d - 1) * 100)) if d >= 2 else int(round(-100 / (d - 1)))


def _leg(p):
    return {"leg": f"{p.get('name')} {str(p.get('side', '')).lower()} {p.get('line')} {p.get('cat', '').replace('_', ' ')}",
            "name": p.get("name"), "cat": p.get("cat"), "side": p.get("side"), "line": p.get("line"),
            "price": int(p.get("price")), "model_p": float(p.get("model_p")), "grade": p.get("grade"),
            "score": p.get("score"), "game": p.get("game"), "book": p.get("book"),
            "why": (p.get("reason") or "")[:160].rsplit(".", 1)[0] + "."}


def build_stack(ranked):
    """Pick the best 4-6-leg combination of RECOMMENDED straight props: one per game, <=2 per
    thesis, payout in BAND, maximum EV (true hit prob x payout - 1). Returns dict or None."""
    cands = [p for p in ranked.get("recommended", []) if not p.get("kind") and p.get("model_p") and p.get("price") is not None
             and float(p.get("line") or 0) >= MIN_LINE.get(p.get("cat"), 0)]
    cands = sorted(cands, key=lambda x: float(x.get("score") or 0), reverse=True)[:POOL_TOP]
    if len(cands) < LEG_RANGE[0]:
        return None
    best = None
    for n in range(LEG_RANGE[0], LEG_RANGE[1] + 1):
        for combo in itertools.combinations(cands, n):
            games = [c.get("game") for c in combo]
            if len(set(games)) < n:                                   # one leg per game (independence)
                continue
            theses = [f"{c.get('cat')}:{c.get('side')}" for c in combo]
            if max(theses.count(t) for t in theses) > MAX_PER_THESIS:
                continue
            sides = [c.get("side") for c in combo]
            if max(sides.count(x) for x in set(sides)) > MAX_SAME_SIDE:
                continue
            dec = float(np.prod([american_to_dec(c["price"]) for c in combo]))
            if not (BAND[0] <= dec <= BAND[1]):
                continue
            p = float(np.prod([float(c["model_p"]) for c in combo]))
            ev = p * dec - 1
            key = (ev, sum(float(c.get("score") or 0) for c in combo))
            if best is None or key > best[0]:
                best = (key, combo, dec, p, ev)
    if best is None:
        return None
    _, combo, dec, p, ev = best
    legs = [_leg(c) for c in combo]
    fair_dec = 1 / p
    return {
        "legs": legs, "n_legs": len(legs), "games": sorted({l["game"] for l in legs}),
        "combined_american": dec_to_american(dec), "combined_decimal": round(dec, 2),
        "true_prob": round(p, 4), "fair_american": dec_to_american(fair_dec), "ev": round(ev, 3),
        "expected_hits": round(sum(l["model_p"] for l in legs), 1),
        "note": (f"{len(legs)} of our own recommended bets, one per game so the legs are independent and the book "
                 f"prices them as a plain product. Pays about {dec_to_american(dec)//100}:1. On our numbers it cashes "
                 f"{p*100:.0f}% of the time (fair odds {dec_to_american(fair_dec):+d}) — a parlay compounds the edge, so the "
                 f"expected return is {ev*100:+.0f}% per dollar IF the legs' edges are real. Sized as a parlay, tracked as "
                 f"its own record. Most weeks it loses; the wins are big and it's positive-expectation, which is the point.")}


if __name__ == "__main__":
    import json, os
    r = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "ranked_board.json")))
    s = build_stack(r)
    if not s:
        print("no stack"); raise SystemExit
    print(f"SUNDAY STACK — {s['n_legs']} legs → {s['combined_american']:+d}  true {s['true_prob']*100:.1f}%  fair {s['fair_american']:+d}  EV {s['ev']*100:+.0f}%")
    for l in s["legs"]:
        print(f"  {l['grade']:3s} {l['leg']:45s} {l['price']:+5d}  p={l['model_p']:.2f}  {l['game']}")
