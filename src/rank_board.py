"""Unified ranking — one comparable score across every prop category and every game bet,
so the desk can be viewed three ways: best-in-each-category, best-across-everything, and
best games. A weak category simply won't appear in the cross-board if nothing ranks.

Score design (comparable across markets):
  props: edge and EV, scaled by data confidence (market-trust already inside confidence).
  games: the Desk Confidence (off-market points-of-value + key numbers).
Everything lands on a 0-100 'Desk Score' so props and games each rank cleanly, and the
cross-prop board is an apples-to-apples merge of the per-category scores.
"""
import numpy as np
import pandas as pd

PROP_CATEGORIES = ["rec_yds", "rush_yds", "receptions", "pass_yds", "rush_att", "anytime_td"]
CAT_LABEL = {"rec_yds": "Receiving Yards", "rush_yds": "Rushing Yards", "receptions": "Receptions",
             "pass_yds": "Passing Yards", "rush_att": "Rush Attempts", "anytime_td": "Anytime TD"}


EDGE_PEAK = 0.11        # edge beyond which a bigger gap is a stale-model red flag, not free money
MIN_HITTABLE = 0.45     # a "best bet" must have at least this chance to win — no lottery tickets
CAT_CAP = 3             # max picks from any single market on the best-bets board (diversity)

def _desk_score(edge, model_p, mult=1.0):
    """Unified Desk Score = CONVICTION, not raw edge. Two components:
      value    — the edge, humped so a blowout gap can't top the board
      hittable — reward likely plays and PENALIZE longshots, so a +1000 anytime-TD lottery with a
                 big % edge can never outrank a -140 near-lock (the bug Alex caught).
    A best bet is something you'd put real money on: probable AND +value AND a trusted signal."""
    edge = float(edge or 0); p = float(model_p if model_p is not None else 0.5)
    if edge <= EDGE_PEAK:
        val = edge * 260                             # 4%→10, 8%→21, 11%→29
    else:
        val = EDGE_PEAK * 260 - (edge - EDGE_PEAK) * 260   # decay past the peak
    hittable = (p - 0.52) * 55                        # p .70→+10, .52→0, .20 (longshot)→-18
    score = (45 + val + hittable) * float(mult or 1.0)
    return round(float(np.clip(score, 20, 95)), 1)


def prop_score(row):
    """Desk Score for a prop pick: conviction (edge + hit-probability), times the investigate
    verdict multiplier (SUSPECT cut hard, CONFIRMED boosted)."""
    return _desk_score(row.get("edge"), row.get("model_p"), row.get("score_mult", 1.0))


def td_score(row):
    """TD board on the SAME conviction scale — so a longshot anytime-TD (low model_p) scores low
    and can't masquerade as a best bet, no matter how big its % edge vs a fat-margin market."""
    return _desk_score(row.get("edge"), row.get("p_model"))


def game_score(desk_conf):
    """Games use Desk Confidence (6.0-9.6) mapped onto the same 0-100 scale."""
    return round((float(desk_conf) - 6.0) / 3.6 * 60 + 40, 1)   # 6.0->40, 9.6->100


def build_ranked(prop_board=None, td_board=None, games=None, top=10, adp=None):
    """Returns the toggleable structure: per-category top N, best-across-props top N, and
    best games top N. Each big edge is investigated individually (investigate.py) using ADP
    consensus — CONFIRMED alpha is boosted, SUSPECT (stale) is cut, the rest kept."""
    out = {"by_category": {}, "best_props": [], "best_games": []}
    rows = []
    if prop_board is not None and len(prop_board):
        pb = prop_board[prop_board.get("status", "PICK").eq("PICK")] if "status" in prop_board else prop_board
        recs = pb.to_dict("records")
        if adp is not None and len(adp):
            import investigate as inv
            recs = inv.adjudicate_board(recs, adp)   # attaches verdict/note/score_mult
        for r in recs:
            rows.append({"cat": r["mkt"], "score": prop_score(r), "name": r["name"],
                         "game": r.get("game"), "line": r.get("line"), "side": r.get("side"),
                         "our_num": r.get("desk_proj"), "edge": r.get("edge"), "ev": r.get("ev"),
                         "model_p": r.get("model_p"), "price": r.get("price"), "book": r.get("book"),
                         "n_books": r.get("n_books"), "shop_adv": r.get("shop_adv"),
                         "verdict": r.get("verdict", "KEEP"), "verdict_note": r.get("verdict_note", ""),
                         "score_mult": r.get("score_mult", 1.0), "reason": r.get("reason", "")})
    if td_board is not None and len(td_board):
        for _, r in td_board[td_board["edge"].notna()].iterrows():
            mp = int((r.get("p_model") or 0) * 100); imp = int((r.get("p_implied") or 0) * 100)
            edge = float(r.get("edge") or 0)
            if edge >= 0.03:
                rsn = (f"Model {mp}% to score vs {imp}% implied — we're higher, a genuine +value TD, "
                       f"but a one-sided market, so a small add.")
            elif mp >= 50:
                rsn = (f"A likely scorer ({mp}%), but the market's already there ({imp}% implied) — "
                       f"no Yes value; shown as a red-zone reference / SGP building block, not a bet.")
            else:
                rsn = (f"Model {mp}% vs {imp}% implied — no edge; reference only.")
            rows.append({"cat": "anytime_td", "score": td_score(r), "name": r["name"],
                         "game": r.get("game"), "line": 0.5, "side": "Yes",
                         "our_num": r.get("p_model"), "edge": r.get("edge"), "ev": None,
                         "model_p": r.get("p_model"), "price": r.get("price"), "book": "draftkings",
                         "reason": rsn})
    allp = pd.DataFrame(rows)
    if len(allp):
        # per-category top N. Anytime-TD is a "who's most likely to score" view — sort it by
        # probability so the near-lock scorers (favorites) lead, not just +1000 lotteries.
        for cat in PROP_CATEGORIES:
            sub = allp[allp.cat == cat]
            sort_col = "model_p" if cat == "anytime_td" else "score"
            sub = sub.sort_values(sort_col, ascending=False, na_position="last").head(top)
            if len(sub):
                out["by_category"][cat] = sub.to_dict("records")
        # BEST BETS = best-across, but only genuinely hittable plays (no lottery longshots).
        # A +1000 anytime-TD is not a "best bet" — it lives in its own tab, not here.
        # DIVERSITY: cap any one market at CAT_CAP so the board is a genuine best-of-the-slate
        # across bet types, not a monotone stack of (say) passing unders. Same-thesis clustering
        # is both a weak product AND correlated risk — if "Week-1 passing unders" is wrong, a
        # stacked board fails together. We still never force a bad pick: everything here cleared
        # MIN_HITTABLE and is ranked by conviction; the cap just spreads the slots.
        hittable = allp[allp["model_p"].fillna(0.5) >= MIN_HITTABLE].sort_values(
            "score", ascending=False)
        picked, per_cat = [], {}
        for r in hittable.to_dict("records"):
            c = r["cat"]
            if per_cat.get(c, 0) >= CAT_CAP:
                continue
            picked.append(r); per_cat[c] = per_cat.get(c, 0) + 1
            if len(picked) >= top:
                break
        # if the cap left us short of `top` (thin slate), backfill with the next-best regardless
        if len(picked) < top:
            have = {(r["name"], r["cat"], r.get("line")) for r in picked}
            for r in hittable.to_dict("records"):
                if (r["name"], r["cat"], r.get("line")) in have:
                    continue
                picked.append(r)
                if len(picked) >= top:
                    break
        out["best_props"] = picked

    # games (spreads/totals/ML) — already scored dicts with desk_conf
    if games:
        g = [{**x, "score": game_score(x.get("desk_conf", 6.0))} for x in games]
        out["best_games"] = sorted(g, key=lambda x: x["score"], reverse=True)[:top]
    return out


if __name__ == "__main__":
    # smoke test with synthetic
    pb = pd.DataFrame([
        {"mkt": "rec_yds", "status": "PICK", "name": "A", "game": "X@Y", "line": 60, "side": "Over",
         "desk_proj": 68, "edge": 0.09, "ev": 0.12, "confidence": 0.7, "price": -110, "book": "dk", "reason": "..."},
        {"mkt": "rush_yds", "status": "PICK", "name": "B", "game": "X@Y", "line": 55, "side": "Under",
         "desk_proj": 48, "edge": 0.06, "ev": 0.07, "confidence": 0.66, "price": -108, "book": "fd", "reason": "..."},
    ])
    r = build_ranked(prop_board=pb, games=[{"label": "BUF -1", "desk_conf": 8.3, "seg": "side"}])
    print("categories:", list(r["by_category"].keys()))
    print("best_props top:", [(p["name"], p["cat"], p["score"]) for p in r["best_props"]])
    print("best_games:", [(g["label"], g["score"]) for g in r["best_games"]])
