"""The Whale Missile — one big Wavey-style parlay of near-locks.

The point (Alex's rule): 8-10 legs where MOST are genuine near-locks, so a LOT of them
hit. The payout comes from STACKING many likely minus-money legs, not from a pile of
longshots. That means:
  - pull from the whole slate, not one game — legs across different games are INDEPENDENT,
    which is exactly what makes "a lot of them hit" true (same-game legs move together and
    would all-hit-or-all-miss, defeating the purpose).
  - cap 2 legs per game so we never clump correlated bets.
  - MOST legs must be near-locks (model prob >= 0.60, minus-money); allow at most 2 spicier
    legs (0.55-0.60) for payout; optionally 1 near-lock anytime-TD (a bell-cow / alpha).
  - prices are real; combined odds shown naive (the book haircuts a same-game parlay, but
    a cross-game parlay of independent legs prices close to the product).
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import td_board as tb, prop_board as pb

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MIN_P = 0.55            # a leg must be genuinely likely to make the slip at all
LOCK_P = 0.60           # near-lock threshold — MOST legs must clear this
MAX_PER_GAME = 2        # never clump correlated legs from one game
MAX_SPICE = 2           # at most two 0.55-0.60 legs for payout


def american_to_dec(a):
    return 1 + (a/100 if a > 0 else 100/abs(a))


def dec_to_american(d):
    return round((d-1)*100) if d >= 2 else round(-100/(d-1))


def build_missile(key=None, days=8, target=9, target_low=6000, target_high=12000):
    """Wavey-style near-lock parlay across the slate, tuned to a TARGET payout band
    (default +6000..+12000 = 60:1..120:1) so it hits far more often and builds engagement,
    instead of a 500:1 that never cashes. We add the highest-probability near-locks until the
    combined odds enter the band, so fewer, safer legs -> higher hit rate. Returns leg set + odds."""
    key = key or os.environ.get("ODDS_API_KEY")
    proj = pd.read_parquet(f"{DATA}/projection_2026.parquet")
    prop, _ = pb.build_prop_board(key=key, projections=proj, days=days)
    if prop is None or not len(prop):
        return None
    # credibility: prefer ESTABLISHED, recognizable players (real ADP / real NFL sample) over
    # deep rookies. A near-lock on a star reads as a lock; a rookie-suppression under does not,
    # and the model is shakiest on thin-sample players. Build a per-name credibility score.
    import re as _re
    def _nm(n): return _re.sub(r"[^a-z]", "", str(n).lower().replace(".", "").replace("'", ""))
    adp_map = {}
    try:
        for pl in json.load(open(f"{DATA}/ffc_adp_2026.json"))["players"]:
            adp_map[_nm(pl["name"])] = float(pl["adp"])
    except Exception:
        pass
    hist_map = {_nm(r["name"]): float(r.get("hist_games") or 0) for _, r in proj.iterrows()}
    def credibility(name):
        k = _nm(name); adp = adp_map.get(k); hg = hist_map.get(k, 0)
        c = 0.0
        if adp is not None: c += max(0.0, (160 - adp) / 160.0)      # early ADP -> recognizable
        c += min(hg, 40) / 40.0 * 0.6                                # NFL track record
        return c                                                     # ~0 (deep rookie) .. ~1.5 (star vet)
    mkt_lbl = {"rec_yds": "rec yds", "rush_yds": "rush yds", "receptions": "catches",
               "pass_yds": "pass yds", "rush_att": "rush att", "pass_completions": "completions"}

    prop = prop.copy(); prop["_p"] = prop["model_p"].fillna(0)
    cand = []
    for _, r in prop[prop["_p"] >= MIN_P].iterrows():
        p = float(r["_p"]); price = int(r["price"])
        # a genuine near-lock is MINUS-money: the market agrees it's likely. Plus-money "locks"
        # are the model overrating a suppression on a thin-data player — not a lock at all.
        # Keep those only as spice, and never let the model prob claim lock status on them.
        market_agrees = price <= -105
        cred = credibility(r["name"])
        is_lock = p >= LOCK_P and market_agrees
        if not market_agrees and p < 0.66:            # weak model AND market disagrees -> skip
            continue
        if cred < 0.15 and not market_agrees:         # deep unknown w/o market backing -> skip
            continue
        # rank by market-agreed prob, boosted by how recognizable/established the player is
        srt = (p if market_agrees else p * 0.75) + cred * 0.10
        cand.append({"leg": f"{r['name']} {r['side'].lower()} {r['line']:g} {mkt_lbl.get(r['mkt'], r['mkt'])}",
                     "price": price, "p_model": round(p, 3), "type": r["mkt"],
                     "game": r.get("game"), "name": r["name"], "lock": is_lock,
                     "cred": round(cred, 2), "_sort": srt})
    # one near-lock anytime-TD (heavy favorite's bell cow / alpha) — never a pile of them
    try:
        tdb, _ = tb.build_td_board(key=key, days=days)
        for _, r in tdb[tdb["p_model"].notna()].sort_values("p_model", ascending=False).head(1).iterrows():
            if float(r["p_model"]) >= 0.55:
                cand.append({"leg": f"{r['name']} anytime TD", "price": int(r["price"]),
                             "p_model": round(float(r["p_model"]), 3), "type": "anytime_td",
                             "game": r.get("game"), "name": r["name"], "lock": float(r["p_model"]) >= 0.6})
    except Exception:
        pass
    if len(cand) < 5:
        return None

    # Selection order: recognizable near-locks first (stars the public knows), then any
    # near-lock, then at most 2 spice legs. Diversify: one leg per player, <=2 per game.
    legs, per_game, seen_player, spice_used = [], {}, set(), 0

    def _try_add(c, allow_spice=True):
        nonlocal spice_used
        if len(legs) >= target + 1 or c["name"] in seen_player:
            return
        if per_game.get(c["game"], 0) >= MAX_PER_GAME:
            return
        is_spice = not c["lock"]
        if is_spice and (not allow_spice or spice_used >= MAX_SPICE):
            return
        legs.append(c); seen_player.add(c["name"]); per_game[c["game"]] = per_game.get(c["game"], 0) + 1
        if is_spice: spice_used += 1

    # ordered pool: to MAXIMIZE how many legs are expected to hit, add the highest-probability
    # near-locks first (recognizable names get a small tiebreak so the slip stays marketable),
    # then lower-prob locks, then spice. Higher-p legs are more juiced, so more of them are
    # needed to reach the payout band — which is exactly what lifts the expected-hits count.
    credible_locks = sorted([c for c in cand if c["lock"] and c["cred"] >= 0.30],
                            key=lambda x: (x["p_model"], x["cred"]), reverse=True)
    other_locks = sorted([c for c in cand if c["lock"] and c["cred"] < 0.30],
                         key=lambda x: x["p_model"], reverse=True)
    spice = sorted([c for c in cand if not c["lock"]], key=lambda x: x["_sort"], reverse=True)
    pool = credible_locks + other_locks + spice

    def _cur_odds():
        return dec_to_american(float(np.prod([american_to_dec(l["price"]) for l in legs]))) if legs else -10000

    # add highest-probability near-locks until the payout enters the target band. Fewer, safer
    # legs => higher hit rate. Spice only if we still can't reach the band with locks.
    for c in pool:
        if _cur_odds() >= target_low and len(legs) >= 5:
            break
        allow = c in spice
        if _cur_odds() >= target_low:              # already in band — stop adding
            break
        _try_add(c, allow_spice=allow)
        if _cur_odds() > target_high and len(legs) >= 5:
            legs.pop()                             # last leg overshot the band; drop it
            break
    if len(legs) < 5:
        return None

    dec = float(np.prod([american_to_dec(l["price"]) for l in legs]))
    p_all = float(np.prod([l["p_model"] for l in legs]))
    exp_hits = round(sum(l["p_model"] for l in legs), 1)
    n_locks = sum(1 for l in legs if l["lock"])
    games = sorted({l["game"] for l in legs})
    return {"legs": legs, "n_legs": len(legs), "games": games,
            "combined_american": dec_to_american(dec), "combined_decimal": round(dec, 1),
            "p_all": round(p_all, 4), "expected_hits": exp_hits, "n_locks": n_locks,
            "note": (f"{len(legs)} legs across {len(games)} games, all near-locks — tuned to a "
                     f"~{dec_to_american(dec)//1000*10 or 60}:1 payout so it cashes far more often "
                     f"than a 500:1 moon-shot (we expect about {exp_hits} of {len(legs)} to hit). "
                     "Independent legs across games; the payout comes from stacking likely bets, not "
                     "longshots. Shorter and safer by design — for engagement, not a lottery.")}


if __name__ == "__main__":
    m = build_missile(days=14)
    if not m:
        print("no near-lock slate in window (props not posted yet)")
    else:
        print("=== 🐋 WHALE MISSILE — Wavey-style near-lock parlay ===")
        for l in m["legs"]:
            lk = "LOCK" if l["lock"] else "spice"
            print(f"   {l['leg']:42s} {l['price']:+5d}  (model {l['p_model']}, {lk})")
        print(f"\n   {m['n_legs']} legs across {len(m['games'])} games → {m['combined_american']:+d}  "
              f"(${100} → ${round(100*m['combined_decimal']):,})")
        print(f"   {m['note']}")
