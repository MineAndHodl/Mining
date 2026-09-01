"""Conviction engine — the 'What I Like' ranker.

Takes every priced edge from every source (off-market spread/total flags, moneyline
+EV, Wong teasers, TD props, calibrated yardage props) and ranks them into tiers.
Only calibrated model sources are allowed in. Hard cutoff: nothing below LEAN ships.

Output target: 5-7 real plays a week. 2-4 elite (TOP PLAY / STRONG), the rest LEAN
fillers we still like. Some weeks fewer. Empty TOP PLAY is allowed and stated.

Tiers by conviction score (0-100), blending edge size and data confidence:
  >=70  TOP PLAY   (rare — the bet we'd size biggest)
  55-69 STRONG
  42-54 LEAN
  <42   not shown
Entertainment plays (missiles, TD parlays) are a SEPARATE lane, always labeled
lottery, never counted against the conviction bar.
"""
import pandas as pd

# calibrated, pick-eligible model sources only. add markets here as they pass calibration.
ELIGIBLE_PROP_MARKETS = {"rec_yds", "anytime_td"}   # rush_yds/pass_yds/receptions gated until usage layer


def score_flag(vs_fair_pts, crosses_key, ev=None):
    """Off-market spread/total flag: CLV edge in points is the signal."""
    s = 40 + 18 * vs_fair_pts            # 1pt off fair ~ 58, 2pt ~ 76
    if crosses_key:
        s += 8
    if ev:
        s += min(ev * 200, 12)
    return min(s, 95)


def score_prop(edge, games, market):
    """Prop edge: model P minus de-vigged implied, scaled by data confidence."""
    if market not in ELIGIBLE_PROP_MARKETS:
        return 0
    conf = min(games / 14.0, 1.0)        # ~full confidence by 14 games
    s = 38 + edge * 260 * (0.6 + 0.4 * conf)   # 5% edge ~ 51-64 depending on data
    return min(s, 92)


def tier(score):
    if score >= 70:
        return "TOP PLAY"
    if score >= 55:
        return "STRONG"
    if score >= 42:
        return "LEAN"
    return None


def desk_confidence(score):
    """Proprietary 0-10 DESK CONFIDENCE rating (NOT a win probability).
    Curved so every play that clears the bar reads 6.0+ and elite plays hit 8-9.x;
    caps at 9.6 — nothing is ever a 10. This is our conviction rating, displayed;
    the honest internal edge lives underneath it.
      score 42 (LEAN cutoff) -> 6.0
      score 70 (TOP PLAY)    -> 7.9
      score 95 (max)         -> 9.6
    """
    d = 6.0 + (max(score, 42) - 42) / (95 - 42) * 3.6
    return round(min(d, 9.6), 1)


def build_card(flags=None, props=None, max_plays=7):
    """Assemble the ranked What-I-Like board from scored candidates.
    flags/props: lists of dicts already scored (key 'score', plus display fields).
    Returns the top plays above the LEAN cutoff, capped at max_plays."""
    cands = []
    for f in (flags or []):
        t = tier(f["score"])
        if t:
            cands.append({**f, "tier": t, "lane": "core"})
    for p in (props or []):
        t = tier(p["score"])
        if t:
            cands.append({**p, "tier": t, "lane": "core"})
    for c in cands:
        c["desk_conf"] = desk_confidence(c["score"])
    cands.sort(key=lambda x: x["score"], reverse=True)
    top = cands[:max_plays]
    n_elite = sum(1 for c in top if c["tier"] in ("TOP PLAY", "STRONG"))
    return {
        "plays": top,
        "n_elite": n_elite,
        "verdict": ("quiet week — sitting on hands" if not top else
                    f"{n_elite} elite, {len(top)-n_elite} lean" if n_elite else
                    f"{len(top)} leans, nothing elite — small or pass"),
    }
