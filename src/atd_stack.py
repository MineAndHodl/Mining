"""The Touchdown Tax (Alex's name) — 3-4 of our best anytime-TD scorers, cross-game, compounded (Alex's spec).

Same logic as the Sunday Stack, on touchdown scorers: legs from different games (independent,
priced by the book as a plain product), each a REAL scorer (model >= 38% to score — no lottos)
with a positive edge at the shopped best price, 3-4 legs, payout in ~5-12:1, the max-EV
combination. Three ~55% scorers at -120 pay ~+500 and cash ~17% of the time; four pay ~+1000
and cash ~9%. TD edges are the noisiest we carry (one-sided market, fat margins), so the
caveat is louder here: this is +EV on our numbers, and the grading loop is the judge.
"""
import itertools
import numpy as np

# STRUCTURAL TRUTH (measured, td_env_validation.json): players who scored 1+ TD/game last
# season find the end zone in only ~51% of next season's games (62% in weeks 1-3, small n),
# yet the market prices them at 64-76%. Our calibrated top bucket (pred .50) HITS .54 out of
# sample — so when we say 50% it is 50%, and the stars' Yes prices are 10-25 pts too rich.
# The +EV touchdowns therefore live at plus money (36-48% scorers at +135..+410). A stack of
# three of those pays ~15-30:1 and cashes ~8%; the 6-12:1 shape of the Sunday Stack would need
# 55% scorers at -120, which are exactly the ones the market overprices. So the Touchdown Tax is
# 3 legs (4 only if 3 can't make the band) of the MOST LIKELY +EV scorers, one per game — the
# objective is P(cash), not payout: it is the "real scorers" stack, honestly a lotto-lite.
BAND = (12.0, 40.0)       # decimal payout ~11:1 .. 39:1 — where three plus-money scorers land
LEG_RANGE = (3, 4)
MIN_P = 0.36              # a real scorer, not a lottery ticket
MIN_EDGE = 0.03           # vs the best shopped price
MIN_STACK_EV = 0.30       # the parlay itself must carry real edge on our numbers
MAX_PER_POS = 2
POOL_TOP = 14
FALLBACK_LEGS = 2         # if no 3-leg clears, fall back to the best 2-leg (band 5-30)


def american_to_dec(a):
    a = float(a); return 1 + (a / 100 if a > 0 else 100 / abs(a))


def dec_to_american(d):
    return int(round((d - 1) * 100)) if d >= 2 else int(round(-100 / (d - 1)))


def _search(cands, leg_range, band):
    """Max P(cash) over in-band combos (one per game, position cap, parlay EV floor)."""
    best = None
    for n in range(leg_range[0], leg_range[1] + 1):
        for combo in itertools.combinations(cands, n):
            if len({c.get("game") for c in combo}) < n:
                continue
            pos = [c.get("position") or "?" for c in combo]
            if max(pos.count(x) for x in set(pos)) > MAX_PER_POS:
                continue
            dec = float(np.prod([american_to_dec(c["price"]) for c in combo]))
            if not (band[0] <= dec <= band[1]):
                continue
            p = float(np.prod([float(c["model_p"]) for c in combo]))
            ev = p * dec - 1
            if ev < MIN_STACK_EV:
                continue
            key = (round(p, 4), ev)
            if best is None or key > best[0]:
                best = (key, combo, dec, p, ev)
    return best


def build_stack(td_rows):
    """td_rows: scored anytime-TD candidate dicts (name, game, price, book, model_p, edge, score,
    position, reason). Returns the best in-band combination or None."""
    cands = [r for r in td_rows if r.get("cat") == "anytime_td" and r.get("model_p") and r.get("price") is not None
             and float(r["model_p"]) >= MIN_P and float(r.get("edge") or 0) >= MIN_EDGE and not r.get("mkt_flag")]
    cands = sorted(cands, key=lambda x: float(x.get("model_p") or 0), reverse=True)[:POOL_TOP]
    best = _search(cands, LEG_RANGE, BAND)
    if best is None and len(cands) >= FALLBACK_LEGS:
        best = _search(cands, (FALLBACK_LEGS, FALLBACK_LEGS), (5.0, 30.0))
    if best is None:
        return None
    _, combo, dec, p, ev = best
    legs = [{"leg": f"{c['name']} anytime TD", "name": c["name"], "game": c.get("game"), "price": int(c["price"]),
             "book": c.get("book"), "model_p": round(float(c["model_p"]), 3), "edge": c.get("edge"),
             "grade": c.get("grade"), "position": c.get("position"),
             "why": (c.get("reason") or "")[:170].rsplit(".", 1)[0] + "."} for c in combo]
    key = (ev, sum(float(c.get("score") or 0) for c in combo))
    return {"legs": legs, "n_legs": len(legs), "games": sorted({l["game"] for l in legs}),
            "combined_american": dec_to_american(dec), "combined_decimal": round(dec, 2),
            "true_prob": round(p, 4), "fair_american": dec_to_american(1 / p), "ev": round(ev, 3),
            "expected_hits": round(sum(l["model_p"] for l in legs), 1),
            "note": (f"{len(legs)} real scorers (each {int(min(l['model_p'] for l in legs)*100)}%+ to find the end zone on our numbers), "
                     f"one per game so they're independent, each at the best price across {'/'.join(sorted({l['book'] for l in legs if l.get('book')}))}; "
                     f"built for the highest chance to cash among +EV scorers, not the biggest number. "
                     f"Pays ~{dec_to_american(dec)//100}:1; cashes {p*100:.0f}% of the time on our numbers (fair {dec_to_american(1/p):+d}), "
                     f"{ev*100:+.0f}% expected return per dollar IF the edges are real. Why no stars: last season's 1-TD-a-game backs "
                     f"score in ~51% of the next season's games (measured), and the market prices them at 64-76% — so the +EV "
                     f"touchdowns live at plus money, and three of them pay 15-30:1. A real-edge lotto-lite, sized small, not a lock.")}
