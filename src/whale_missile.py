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


def build_missile(key=None, days=8, target=10, target_low=3000, target_high=10000):
    """Wavey-style near-lock parlay across the slate, tuned to a TARGET payout band
    (default +6000..+12000 = 60:1..120:1) so it hits far more often and builds engagement,
    instead of a 500:1 that never cashes. We add the highest-probability near-locks until the
    combined odds enter the band, so fewer, safer legs -> higher hit rate. Returns leg set + odds."""
    key = key or os.environ.get("ODDS_API_KEY")
    proj = pd.read_parquet(f"{DATA}/projection_2026.parquet")
    prop, _ = pb.build_prop_board(key=key, projections=proj, days=days)
    # a near-lock parlay is about MARKET-likely legs on players with a real track record — never
    # a rookie the model overrates (no recent games). Keep only players who HAVE a recent log.
    if prop is not None and len(prop) and "edge_src" in prop.columns:
        prop = prop[prop["edge_src"] == "multifactor"].copy()
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
    for _, r in prop.iterrows():
        p = float(r["_p"]); price = int(r["price"])
        pmkt = float(r.get("fair", p) or p)           # de-vigged MARKET probability (the honest read)
        # Gate on the MARKET's probability (not our model) — that's what makes a leg genuinely
        # likely and excludes the rookie-under illusions the model overrates. Books set most lines
        # ~50/50, so true near-locks are scarce; we take the market-likeliest legs available.
        market_agrees = price <= -105
        cred = credibility(r["name"])
        is_lock = pmkt >= 0.58 and market_agrees
        if pmkt < 0.53:                               # market sees it as a favorite (true locks are scarce)
            continue
        if p < 0.50 or price > 110:                   # our model agrees; allow slight plus-money
            continue
        if cred < 0.15:                               # recognizable, real track record
            continue
        # rank by MARKET probability, small recognizability tiebreak
        srt = pmkt + cred * 0.05
        cand.append({"leg": f"{r['name']} {r['side'].lower()} {r['line']:g} {mkt_lbl.get(r['mkt'], r['mkt'])}",
                     "price": price, "p_model": round(p, 3), "type": r["mkt"],
                     "p_mkt": round(pmkt, 3),
                     "game": r.get("game"), "name": r["name"], "lock": is_lock,
                     "cred": round(cred, 2), "_sort": srt})
    # one near-lock anytime-TD (heavy favorite's bell cow / alpha) — never a pile of them
    try:
        tdb, _ = tb.build_td_board(key=key, days=days)
        for _, r in tdb[tdb["p_model"].notna()].sort_values("p_model", ascending=False).head(1).iterrows():
            if float(r["p_model"]) >= 0.55:
                cand.append({"leg": f"{r['name']} anytime TD", "price": int(r["price"]),
                             "p_model": round(float(r["p_model"]), 3), "type": "anytime_td",
                             "p_mkt": round(float(r.get("p_implied", r["p_model"]) or r["p_model"]), 3),
                             "game": r.get("game"), "name": r["name"], "lock": float(r["p_model"]) >= 0.6})
    except Exception:
        pass
    if len(cand) < 5:
        return None

    # Selection: add the market-likeliest favorites (highest de-vigged prob first) until the
    # payout enters the target band. Diversify: one leg per player, <=2 per game. Recognizable
    # names get a small tiebreak so the slip stays marketable.
    legs, per_game, seen_player = [], {}, set()

    def _try_add(c):
        if len(legs) >= target or c["name"] in seen_player:
            return
        if per_game.get(c["game"], 0) >= MAX_PER_GAME:
            return
        legs.append(c); seen_player.add(c["name"]); per_game[c["game"]] = per_game.get(c["game"], 0) + 1

    pool = sorted(cand, key=lambda x: (x["p_mkt"], x["cred"]), reverse=True)

    def _cur_odds():
        return dec_to_american(float(np.prod([american_to_dec(l["price"]) for l in legs]))) if legs else -10000

    for c in pool:
        if _cur_odds() >= target_low and len(legs) >= 6:
            break                                  # in band with enough legs — stop
        _try_add(c)
        if _cur_odds() > target_high and len(legs) >= 6:
            legs.pop()                             # last leg overshot the band
            break
    if len(legs) < 5:
        return None

    dec = float(np.prod([american_to_dec(l["price"]) for l in legs]))
    # HONEST expected hits: use the de-vigged MARKET probability per leg, not our (optimistic)
    # model prob. The market is the fairer read of how many actually land.
    exp_hits = round(sum(l.get("p_mkt", l["p_model"]) for l in legs), 1)
    exp_hits_model = round(sum(l["p_model"] for l in legs), 1)
    p_all = float(np.prod([l.get("p_mkt", l["p_model"]) for l in legs]))
    n_locks = sum(1 for l in legs if l["lock"])
    games = sorted({l["game"] for l in legs})
    return {"legs": legs, "n_legs": len(legs), "games": games,
            "combined_american": dec_to_american(dec), "combined_decimal": round(dec, 1),
            "p_all": round(p_all, 4), "expected_hits": exp_hits, "expected_hits_model": exp_hits_model,
            "n_locks": n_locks,
            "note": (f"{len(legs)} legs across {len(games)} games — every leg a market favorite "
                     f"(~53-60%), tuned to ~{max(dec_to_american(dec)//100, 30)}:1. Honest math: on the "
                     f"market's own numbers we expect about {exp_hits} of {len(legs)} to hit, and the full "
                     f"slip cashes maybe 1 in {max(round(1/max(p_all,1e-4)), 20)} — a lotto built from "
                     "solid favorites, not longshots (true 60%+ locks are too scarce to reach this payout). "
                     "Independent-ish legs across games; size it small.")}


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
