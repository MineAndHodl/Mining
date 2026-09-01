"""Deterministic verification layer. Every card must pass ALL checks before
anything is emailed, posted, or shown. Any violation blocks the send.

Checks are independent re-derivations from the raw board — not re-reads of
the pipeline's own intermediate output.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timezone


def verify_card(board: pd.DataFrame, flags: pd.DataFrame, max_age_min=30):
    """Return list of violations (empty list = card is clean)."""
    v = []
    b = board.copy()

    # 0) board internal sanity: spread points at each book must be symmetric
    sp = b[b["market"] == "spreads"].dropna(subset=["point"])
    sym = sp.groupby(["game", "book"])["point"].sum()
    bad = sym[np.abs(sym) > 1e-9]
    for (game, book), s in bad.items():
        v.append(f"ASYMMETRIC SPREAD: {book} {game} points sum to {s}")

    # pinnacle favored-side sanity: more-negative point should carry higher prob
    def implied(o):
        return 100/(o+100) if o > 0 else abs(o)/(abs(o)+100)
    # cross-market invariant: pinnacle's spread favorite must be its ML favorite
    p = sp[sp["book"] == "pinnacle"]
    ml = b[(b["market"] == "h2h") & (b["book"] == "pinnacle")]
    for game, grp in p.groupby("game"):
        if len(grp) != 2:
            continue
        g = grp.sort_values("point")
        fav = g.iloc[0]
        # Only assert when there's a CLEAR favorite: a field-goal+ spread, OR a
        # smaller spread whose juice also sits on the favorite. At pick'em (e.g.
        # -1 priced at +102) the ML can legitimately favor the other side — not a bug.
        clear_fav = fav["point"] <= -2.5 or (fav["point"] <= -1.5 and fav["price"] <= -112)
        if clear_fav:
            m = ml[ml["game"] == game]
            if len(m) == 2:
                probs = {r["side"]: implied(r["price"]) for _, r in m.iterrows()}
                if probs.get(fav["side"], 1.0) < 0.5 - 0.02:
                    v.append(f"SIGN SANITY: pinnacle {game}: {fav['side']} "
                             f"{fav['point']:+g} by spread but ML underdog — "
                             f"side/point mapping is wrong")

    if flags is None or not len(flags):
        return v

    for i, f in flags.iterrows():
        # 1) the flagged quote must exist verbatim in the raw board
        m = b[(b["game"] == f["game"]) & (b["book"] == f["book"]) &
              (b["market"] == f["market"]) & (b["side"] == f["side"])]
        def _eq(a, c):
            return (pd.isnull(a) and pd.isnull(c)) or a == c
        m2 = m[m.apply(lambda r: _eq(r["point"], f.get("point")) and r["price"] == f["price"], axis=1)] if len(m) else m
        if not len(m2):
            v.append(f"PHANTOM QUOTE: flag {f['side']} {f['point']} @ {f['book']} "
                     f"({f['game']}) not found verbatim in board")
            continue
        # 2) fair point must match pinnacle's actual point for that side
        pin = b[(b["game"] == f["game"]) & (b["book"] == "pinnacle") &
                (b["market"] == f["market"]) & (b["side"] == f["side"])]
        if len(pin) and "fair_point" in f and pd.notnull(f.get("fair_point")):
            if pin.iloc[0]["point"] != f["fair_point"]:
                v.append(f"FAIR MISMATCH: {f['game']} {f['side']}: fair_point "
                         f"{f['fair_point']} vs pinnacle actual {pin.iloc[0]['point']}")
        # 3) recompute the edge independently
        if f["market"] == "spreads" and pd.notnull(f.get("fair_point")):
            d = f["point"] - f["fair_point"]
            if abs(d - f.get("vs_fair_pts", d)) > 1e-9:
                v.append(f"EDGE MISMATCH: {f['game']} {f['side']}: recomputed {d} "
                         f"vs reported {f.get('vs_fair_pts')}")
        if f["market"] == "totals" and pd.notnull(f.get("fair_point")):
            d = (f["point"] - f["fair_point"]) if f["side"] == "Under" else (f["fair_point"] - f["point"])
            if abs(d - f.get("vs_fair_pts", d)) > 1e-9:
                v.append(f"EDGE MISMATCH (total): {f['game']} {f['side']}")
        # 4) line_str must contain the exact signed number
        if "line_str" in f and pd.notnull(f.get("line_str")) and pd.notnull(f.get("point")):
            want = f"{f['point']:+g}" if f["side"] not in ("Over", "Under") else f"{f['point']:g}"
            if want not in str(f["line_str"]):
                v.append(f"RENDER MISMATCH: '{f['line_str']}' does not contain '{want}'")
        # 5) quote freshness
        if "last_update" in f and pd.notnull(f.get("last_update")):
            lu = pd.Timestamp(f["last_update"])
            age = (pd.Timestamp.now(tz="UTC") - lu).total_seconds()/60
            if age > max_age_min:
                v.append(f"STALE QUOTE: {f['side']} @ {f['book']} is {age:.0f} min old")
    return v


if __name__ == "__main__":
    import sys, glob
    boards = sorted(glob.glob("/home/claude/nfl-model/archive/board_*.csv"))
    board = pd.read_csv(boards[-1])
    flagfiles = sorted(glob.glob("/home/claude/nfl-model/archive/flags_*.csv"))
    flags = pd.read_csv(flagfiles[-1]) if flagfiles else pd.DataFrame()
    out = verify_card(board, flags)
    print(f"checked latest snapshot: {len(out)} violation(s)")
    for x in out:
        print(" -", x)
