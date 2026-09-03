"""Line history — where every number OPENED and where it is NOW (Alex: "for every bet it should
be useful to know where it opened and where it is now").

Every fetch archives the whole raw market (every book's line and price for every prop, the
shopped anytime-TD prices, the games intel rows) to data/line_history/<season>_w<week>/
<UTC timestamp>_<tag>.parquet. From those snapshots any pick gets a series and a move:

  open   — the first snapshot of the week in which the player-market was posted (consensus
           line = median across books; best price at our side)
  now    — the latest snapshot
  moved  — signed in OUR direction: positive = the market moved toward our side (we are
           beating the close so far, and the opened number is gone); negative = it moved
           against us (a better number is available now, and the market disagrees)
  cents  — price move at the best book for our side, in cents of American odds

This is the "beat the close" product made visible, and the ledger's closing line comes from
the last snapshot before kickoff instead of a separate capture.
"""
import os, sys, glob, json, datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from names import norm_name

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
HIST = os.path.join(DATA, "line_history")


def _dir(season, week):
    d = os.path.join(HIST, f"{season}_w{int(week):02d}"); os.makedirs(d, exist_ok=True); return d


def archive(season, week, tag="run", props_path=None, td_path=None, games=None):
    """Archive the raw caches of the run that just happened (props: every book; TD: best +
    consensus per player; games: intel rows). Returns the snapshot stamp."""
    if week is None: return None
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M")
    d = _dir(season, week)
    n = {}
    try:
        p = pd.read_parquet(props_path or f"{DATA}/_raw_prop_lines.parquet")
        p["_k"] = p["name"].map(norm_name); p["stamp"] = stamp
        p.to_parquet(os.path.join(d, f"{stamp}_{tag}_props.parquet"), index=False); n["props"] = len(p)
    except Exception as e:
        print("line history: props not archived:", e)
    try:
        t = pd.read_parquet(td_path or f"{DATA}/_raw_td_prices.parquet")
        t["_k"] = t["name"].map(norm_name); t["stamp"] = stamp
        t.to_parquet(os.path.join(d, f"{stamp}_{tag}_td.parquet"), index=False); n["td"] = len(t)
    except Exception as e:
        print("line history: td not archived:", e)
    try:
        g = games if games is not None else json.load(open(f"{DATA}/ranked_board.json")).get("best_games", [])
        if g:
            gd = pd.DataFrame([{k: r.get(k) for k in ("game", "market", "side", "point", "price", "book", "desk_num")} for r in g]); gd["stamp"] = stamp
            gd.to_parquet(os.path.join(d, f"{stamp}_{tag}_games.parquet"), index=False); n["games"] = len(gd)
    except Exception as e:
        print("line history: games not archived:", e)
    print(f"line history: snapshot {stamp} ({tag}) — {n}")
    return stamp


def _snapshots(season, week, kind="props"):
    files = sorted(glob.glob(os.path.join(HIST, f"{season}_w{int(week):02d}", f"*_{kind}.parquet")))
    return files


def prop_series(season, week, name, mkt):
    """[(stamp, consensus_line, n_books, best_over, best_under, lines_by_book)] across snapshots."""
    k = norm_name(name); out = []
    for f in _snapshots(season, week, "props"):
        try:
            p = pd.read_parquet(f)
        except Exception:
            continue
        s = p[(p["_k"] == k) & (p["mkt"] == mkt)]
        if not len(s): continue
        so = s.dropna(subset=["over_price"]); su = s.dropna(subset=["under_price"])
        # the number a bettor can ACT on: the lowest posted line for an Over (best price among
        # books at that line), the highest for an Under — what the Card shows. Consensus = median.
        o_line = float(so["line"].min()) if len(so) else None
        u_line = float(su["line"].max()) if len(su) else None
        o_price = int(so[so["line"] == o_line]["over_price"].max()) if o_line is not None else None
        u_price = int(su[su["line"] == u_line]["under_price"].max()) if u_line is not None else None
        out.append({"stamp": str(s["stamp"].iloc[0]), "line": float(s["line"].median()), "n_books": int(s["book"].nunique()),
                    "over_line": o_line, "over_price": o_price, "under_line": u_line, "under_price": u_price,
                    "books": {b: float(l) for b, l in zip(s["book"], s["line"])}})
    return out


def td_series(season, week, name):
    k = norm_name(name); out = []
    for f in _snapshots(season, week, "td"):
        try:
            t = pd.read_parquet(f)
        except Exception:
            continue
        s = t[t["_k"] == k]
        if not len(s): continue
        r = s.iloc[0]
        out.append({"stamp": str(r["stamp"]), "best_price": int(r["price"]), "book": r.get("book"),
                    "implied_consensus": float(r["implied_consensus"]) if r.get("implied_consensus") == r.get("implied_consensus") else None})
    return out


def _fmt_stamp(st):
    try:
        return datetime.datetime.strptime(st, "%Y%m%dT%H%M").strftime("%a %H:%M")
    except Exception:
        return st


def move_for(row, season, week):
    """The open→now read for one board row (prop or anytime TD). Returns a dict or None."""
    cat = row.get("cat"); side = row.get("side")
    if cat == "anytime_td":
        ser = td_series(season, week, row.get("name"))
        if len(ser) < 1: return None
        o, n = ser[0], ser[-1]
        ic_o, ic_n = o.get("implied_consensus"), n.get("implied_consensus")
        moved = round((ic_n - ic_o) * 100, 1) if (ic_o is not None and ic_n is not None) else None   # consensus implied % moved (+ = market likes him more now)
        return {"open_price": o["best_price"], "open_book": o.get("book"), "opened_at": _fmt_stamp(o["stamp"]),
                "now_price": n["best_price"], "now_book": n.get("book"), "now_at": _fmt_stamp(n["stamp"]),
                "moved_pts": moved, "moved_unit": "% implied", "n_snapshots": len(ser),
                "with_us": (moved is not None and moved > 0) if moved is not None else None}
    ser = prop_series(season, week, row.get("name"), cat)
    if len(ser) < 1: return None
    o, n = ser[0], ser[-1]
    if side == "Over":
        ol, nl, po, pn = o["over_line"], n["over_line"], o["over_price"], n["over_price"]
        moved = round(nl - ol, 1) if (ol is not None and nl is not None) else None
    elif side == "Under":
        ol, nl, po, pn = o["under_line"], n["under_line"], o["under_price"], n["under_price"]
        moved = round(ol - nl, 1) if (ol is not None and nl is not None) else None
    else:
        ol, nl, po, pn, moved = o["line"], n["line"], None, None, None
    cents = (pn - po) if (po is not None and pn is not None) else None
    cons_moved = round((n["line"] - o["line"]) * (1 if side == "Over" else -1), 1) if side in ("Over", "Under") else None
    # STALE NUMBER: the best book for our side sits behind the consensus RIGHT NOW (Over below the
    # median line, Under above it) — the last-call edge when the market has moved and one book lags
    stale = round((n["line"] - nl) if side == "Over" else (nl - n["line"]), 1) if (nl is not None and side in ("Over", "Under")) else None
    return {"open_line": ol, "open_price": po, "opened_at": _fmt_stamp(o["stamp"]), "open_books": o["n_books"], "open_consensus": o["line"],
            "now_line": nl, "now_price": pn, "now_at": _fmt_stamp(n["stamp"]), "now_books": n["n_books"], "now_consensus": n["line"],
            "moved_pts": moved, "moved_unit": "pts", "moved_cents": cents, "consensus_moved": cons_moved, "n_snapshots": len(ser),
            "stale_book_pts": stale, "with_us": (moved > 0) if moved is not None else None,
            "still_available": (moved is not None and moved <= 0)}


def attach(ranked, season, week):
    """Annotate every prop/TD row on every tab (and the parlay legs) with its line move."""
    if week is None: return ranked
    seen = {}
    def ann(r):
        key = (r.get("name"), r.get("cat"), r.get("side"))
        if key not in seen:
            try: seen[key] = move_for(r, season, week)
            except Exception: seen[key] = None
        if seen[key]: r["line_move"] = seen[key]
    for r in ranked.get("recommended", []) or []:
        if not r.get("kind"): ann(r)
    for r in ranked.get("best_props", []) or []: ann(r)
    for lst in (ranked.get("by_category") or {}).values():
        for r in lst: ann(r)
    for r in ranked.get("pool", []) or []: ann(r)
    for lane in ("sunday_stack", "td_stack"):
        s = ranked.get(lane)
        if s:
            for l in s.get("legs", []):
                cat = l.get("cat") or ("anytime_td" if lane == "td_stack" else None)
                ann({"name": l.get("name"), "cat": cat, "side": l.get("side") or ("Yes" if cat == "anytime_td" else None), "_leg": l})
                mv = seen.get((l.get("name"), cat, l.get("side") or ("Yes" if cat == "anytime_td" else None)))
                if mv: l["line_move"] = mv
    n = sum(1 for v in seen.values() if v)
    print(f"line history: moves attached to {n} rows")
    return ranked


def movers(season, week, top=12):
    """Biggest consensus-line moves since the first snapshot (steam) across the whole market."""
    files = _snapshots(season, week, "props")
    if len(files) < 2: return []
    a = pd.read_parquet(files[0]); b = pd.read_parquet(files[-1])
    ca = a.groupby(["_k", "name", "mkt"])["line"].median().rename("open"); cb = b.groupby(["_k", "name", "mkt"])["line"].median().rename("now")
    m = pd.concat([ca, cb], axis=1).dropna().reset_index()
    m["move"] = (m["now"] - m["open"]).round(1); m["abs"] = m["move"].abs()
    m = m[m["abs"] > 0].sort_values("abs", ascending=False).head(top)
    return [{"name": r["name"], "mkt": r["mkt"], "open": float(r["open"]), "now": float(r["now"]), "move": float(r["move"])} for _, r in m.iterrows()]


if __name__ == "__main__":
    import refresh_data as rd, ledger as L
    season = rd.current_season()
    week = L.resolve_week([], season)[1] if False else None
    led = sorted(glob.glob(os.path.join(DATA, "ledger", "*.json")))
    if led:
        d = json.load(open(led[-1])); season, week = d["season"], d["week"]
        for f in _snapshots(season, week, "props"): print(" ", os.path.basename(f))
        for m in movers(season, week): print(f"  {m['name']:22s} {m['mkt']:10s} {m['open']} -> {m['now']} ({m['move']:+.1f})")
