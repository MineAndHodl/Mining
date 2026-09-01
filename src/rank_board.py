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


def prop_score(row):
    """0-100 Desk Score for a prop PICK. The trick: a MODERATE edge on an ESTABLISHED
    role is our best bet — a huge edge is usually the model being stale on a low-data
    player, not free money. So the edge curve PEAKS around 8% and PENALIZES beyond ~11%,
    and confidence (role certainty) is weighted heavily so unknowns can't top the board."""
    edge = float(row.get("edge", 0))
    ev = float(row.get("ev", 0))
    conf = float(row.get("confidence", 0.5))
    # edge is monotonic (bigger IS better) — we DON'T blanket-penalize large edges; the
    # per-play investigation (investigate.py) decides if a big edge is alpha or stale.
    base = 40 + edge * 280 + max(ev, 0) * 18
    conf_factor = 0.4 + 0.6 * min(conf / 0.7, 1.0)
    s = base * conf_factor
    # apply the investigation's verdict multiplier if present (SUSPECT gets cut, not the size)
    s *= float(row.get("score_mult", 1.0))
    return round(min(s, 94), 1)


def td_score(row):
    """TD board: edge vs implied on the same skewed scale."""
    edge = float(row.get("edge", 0) or 0)
    return round(min(45 + edge * 300, 97), 1)


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
                         "price": r.get("price"), "book": r.get("book"),
                         "verdict": r.get("verdict", "KEEP"), "verdict_note": r.get("verdict_note", ""),
                         "reason": r.get("reason", "")})
    if td_board is not None and len(td_board):
        for _, r in td_board[td_board["edge"].notna()].iterrows():
            rows.append({"cat": "anytime_td", "score": td_score(r), "name": r["name"],
                         "game": r.get("game"), "line": 0.5, "side": "Yes",
                         "our_num": r.get("p_model"), "edge": r.get("edge"), "ev": None,
                         "price": r.get("price"), "book": "draftkings",
                         "reason": f"Model {int((r.get('p_model') or 0)*100)}% to score vs {int((r.get('p_implied') or 0)*100)}% implied."})
    allp = pd.DataFrame(rows)
    if len(allp):
        # per-category top N
        for cat in PROP_CATEGORIES:
            sub = allp[allp.cat == cat].sort_values("score", ascending=False).head(top)
            if len(sub):
                out["by_category"][cat] = sub.to_dict("records")
        # best across ALL props — some categories may not appear
        out["best_props"] = allp.sort_values("score", ascending=False).head(top).to_dict("records")

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
