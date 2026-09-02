"""The Games lane — rebuilt LIVE every run (was a stale Sept-1 carry-over with no labels).

What it honestly is: MARKET INTEL, not a model edge. Everything we have measured on game
sides says the same thing (output/qc-report.html, clv_engine.py, macro_themes.json):
  - our EPA power ratings show NO ATS edge (49-51%, worse at higher "edges");
  - the book-gap thesis (a retail book off Pinnacle predicts the move) is DISPROVEN
    (sign agreement .42-.58, corr ~.1) — although CLV itself is real (+0.46 pt, p<1e-22);
  - road-dog / HFA / overs macro angles are priced out.
So a game row is "a bettable book is N points off the sharpest number in the world" —
a price fact a follower can act on, graded at the games scale (1 pt ~ 3% win prob), and
NEVER on the Card until a games edge is validated forward (GAMES_ON_CARD).

Cost: one odds call (h2h+spreads+totals, us+eu) per run.
"""
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weekly_card as wc
import conviction as cv

GAMES_ON_CARD = False      # no validated games edge yet -> intel lane only (never recommended)
TOP = 10


def _reason(f):
    fair = f.get("fair_str"); ln = f.get("line_str"); bk = f.get("book")
    if f.get("type") == "ML +EV vs fair":
        return (f"Pinnacle's sharp number makes this side a {f.get('fair_p', 0)*100:.0f}% shot; {bk} is paying "
                f"{int(f['price']):+d} — {f.get('ev', 0)*100:+.1f}% expected value against the sharpest book in the world. "
                f"Market intel, not a model call: books are rarely this far off Pinnacle for long.")
    key = " through a key number" if f.get("crosses_key") else ""
    return (f"Pinnacle makes it {fair}; {bk} is offering {ln} — {f.get('vs_fair_pts')} point(s) better{key}. "
            f"That's closing-line value if the retail number catches up (it usually does), which is the whole "
            f"'beat the close' idea. We have NO validated model edge on game sides, so this is priced-line intel: "
            f"take the number if you like the side, and expect it to move.")


def build_games(key=None, days=14):
    key = key or os.environ.get("ODDS_API_KEY")
    data, rem = wc.fetch_odds(key, days=days)
    board = wc.build_board(data)
    if not len(board) or "book" not in board.columns:
        return [], rem
    fair = wc.pinnacle_fair(board)
    flags = wc.find_off_market(board, fair)
    rows = []
    for _, f in (flags.iterrows() if len(flags) else []):
        if f.get("type") == "ML +EV vs fair":
            score = cv.score_flag(0.0, False, ev=f.get("ev"))
            pick = f"{f['side']} ML"; kind = "ML"; edge = f"{f.get('ev', 0)*100:+.1f}% EV"
            our = f"fair {int(round(100 / max(f.get('fair_p', .5), 1e-6) - 100)):+d}" if f.get("fair_p", 0) < 0.5 else f"fair {f.get('fair_p', 0)*100:.0f}%"
            line = f"{int(f['price']):+d}"
        else:
            score = cv.score_flag(f.get("vs_fair_pts", 0.0), bool(f.get("crosses_key")))
            pick = f.get("line_str"); kind = "spread" if f["market"] == "spreads" else "total"
            edge = f"{f.get('vs_fair_pts'):+.1f} pts"; our = f.get("fair_str"); line = f.get("line_str")
        rows.append({"pick": pick, "kind": kind, "game": f["game"], "market": f["market"], "side": f["side"],
                     "line": line, "point": f.get("point"), "price": int(f["price"]) if f.get("price") == f.get("price") else None,
                     "book": (f.get("book") or "").replace("williamhill_us", "caesars").title(),
                     "desk_num": our, "edge": edge, "desk_conf": cv.desk_confidence(score), "score": score,
                     "lane": "intel", "reason": _reason(f)})
    rows = sorted(rows, key=lambda r: r["score"], reverse=True)[:TOP]
    return rows, rem


if __name__ == "__main__":
    rows, rem = build_games()
    print(f"credits {rem} | {len(rows)} game rows")
    for r in rows:
        print(f"  {r['score']:5.1f} {r['kind']:6s} {r['pick']:22s} @ {r['book']:12s} {r['edge']:>10s}  ({r['game']})")
