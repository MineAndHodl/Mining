"""Post copy — ready-to-post lines for X / Discord, generated from the FROZEN numbers every run.

Businessman hat (Fresh Eyes day 2): the Card is only a product when it leaves the page. Every
post here is built from ranked_board.json / the ledger / record.json — nothing is typed by hand,
so a post can never say something the model didn't. Posts:
  card_full      the whole Card (paid channel)             — every pick, price, book, grade, size
  teaser         X, <=280 chars                            — how many plays, the rail, ONE free pick
  free_pick      X thread (2-3 posts)                      — the free pick's handicap, honest caveat
  touchdown_tax  Thursday                                  — the legs, the payout, the fair, the EV
  line_moves     the market moved since we opened          — with us / against us
  report_card    Monday (only when the week is graded)     — W-L, units, close beaten

    python3 src/post_copy.py           -> data/post_copy.md + data/post_copy.json
"""
import os, sys, json, glob, datetime

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
X_LIMIT = 280
BOOK = {"fanduel": "FanDuel", "draftkings": "DraftKings", "betrivers": "BetRivers", "espnbet": "ESPN BET", "fanatics": "Fanatics",
        "betmgm": "BetMGM", "williamhill_us": "Caesars", "caesars": "Caesars", "bovada": "Bovada", "pointsbetus": "PointsBet",
        "betonlineag": "BetOnline", "lowvig": "LowVig", "hardrockbet": "Hard Rock", "ballybet": "Bally Bet", "betparx": "betPARX"}
MKT = {"rec_yds": "rec yds", "rush_yds": "rush yds", "pass_yds": "pass yds", "receptions": "receptions", "rush_att": "carries",
       "anytime_td": "anytime TD", "kicking_points": "kicking pts", "field_goals": "FGs"}


def _book(b): return BOOK.get(str(b or "").lower(), str(b or "").replace("_us", "").title())
def _mkt(m): return MKT.get(m, str(m).replace("_", " "))
def _price(p):
    try: return f"{int(p):+d}"
    except Exception: return "?"
def _pct(p):
    try: return f"{float(p) * 100:.0f}%"
    except Exception: return "?"
def _line(x):
    try: return f"{float(x):g}"
    except Exception: return str(x)


def _sentences(text, n=2):
    """First n sentences of a paragraph (the handicap without the money paragraph)."""
    import re
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return " ".join(parts[:n]).strip()


def _week():
    leds = sorted(glob.glob(f"{DATA}/ledger/*.json"))
    if not leds:
        return None, None, None
    led = json.load(open(leds[-1]))
    return led.get("season"), led.get("week"), led


def build(board=None, record=None, now=None):
    board = board if board is not None else json.load(open(f"{DATA}/ranked_board.json"))
    season, week, led = _week()
    wk = f"Week {week}" if week else "This week"
    card = [r for r in board.get("recommended", []) if r.get("recommended") and not r.get("kind")]
    out = {"season": season, "week": week, "generated_at": (now or datetime.datetime.now(datetime.timezone.utc)).isoformat(timespec="minutes"), "posts": {}}
    if not card:
        out["posts"]["card_full"] = {"text": f"THE DESK — {wk}: no Card yet (the board has no recommended plays).", "chars": 0}
        return out
    # ---- card_full (paid channel) ------------------------------------------------------------
    lines = [f"THE DESK — {wk} Card ({len(card)} plays)"]
    for r in card:
        lines.append(f"▸ {r['name']} {str(r['side']).upper()} {_line(r['line'])} {_mkt(r['cat'])} ({_price(r['price'])} {_book(r['book'])}) · "
                     f"{r.get('grade_sized') or r.get('grade')} · {r.get('units', 0):g}u — we make it {_pct(r.get('model_p'))}, the market {_pct(_fair(r))}")
    min_p = min(float(r.get("model_p") or 0) for r in card)
    avg_edge = sum(float(r.get("edge") or 0) for r in card) / len(card)
    lines.append(f"Rail: every Card play is ≥{min_p * 100:.0f}% to hit on our numbers (avg edge {avg_edge * 100:+.1f}%). "
                 f"Sizes are units, not confidence theater. Graded Monday at the price posted.")
    txt = "\n".join(lines); out["posts"]["card_full"] = {"text": txt, "chars": len(txt)}
    # ---- the free pick = the Card's most likely winner (the public record is a product) ---------
    free = max(card, key=lambda r: (float(r.get("model_p") or 0), float(r.get("score") or 0)))
    fp = (f"{free['name']} {str(free['side']).upper()} {_line(free['line'])} {_mkt(free['cat'])} ({_price(free['price'])} {_book(free['book'])})")
    teaser = (f"{wk} Card is live: {len(card)} plays, every one ≥{min_p * 100:.0f}% to hit on our numbers.\n\n"
              f"Free one: {fp} — we make it {_pct(free.get('model_p'))}.\n\n"
              f"The other {len(card) - 1} are in the Desk. Graded Monday, at the posted price.")
    if len(teaser) > X_LIMIT:
        teaser = (f"{wk} Card: {len(card)} plays, all ≥{min_p * 100:.0f}% on our numbers.\n\nFree one: {fp} — {_pct(free.get('model_p'))} for us.\n\n"
                  f"The rest are in the Desk. Graded Monday.")
    out["posts"]["teaser"] = {"text": teaser, "chars": len(teaser), "x_ok": len(teaser) <= X_LIMIT}
    why = _sentences(free.get("reason", ""), 3)
    caveat = ("Honest caveat: this is late-2025 form across the offseason — an early-season read, sized as a lean."
              if free.get("offseason") else "Multi-factor signal, sized to conviction.")
    thread = [f"Free pick, {wk}: {fp}.\n\nWe make it {_pct(free.get('model_p'))}; the price says {_pct(_fair(free))}. Thread 🧵",
              why, f"{caveat} Posted at the number above; we grade it Monday whether it wins or loses."]
    thread = [_fit(t) for t in thread]
    out["posts"]["free_pick"] = {"thread": thread, "chars": [len(t) for t in thread], "x_ok": all(len(t) <= X_LIMIT for t in thread)}
    # ---- Touchdown Tax (Thursday) -------------------------------------------------------------
    s = board.get("td_stack")
    if s and s.get("legs"):
        legs = s["legs"]
        ll = "\n".join(f"▸ {l['name']} anytime TD ({_price(l['price'])} {_book(l.get('book'))}) · {_pct(l.get('model_p'))} for us" for l in legs)
        tt = (f"TOUCHDOWN TAX — {wk}\n\n{ll}\n\nPays ~{s['combined_decimal'] - 1:.0f}:1 ({_price(s['combined_american'])}). "
              f"Cashes {s['true_prob'] * 100:.0f}% of the time on our numbers (fair {_price(s['fair_american'])}) — {s['ev'] * 100:+.0f}% expected return if the edges are real.\n\n"
              f"No stars on purpose: last year's 1-TD-a-game guys score in ~51% of next season's games and get priced at 64-76%. The value is the plus-money scorers. Small stake, real edge.")
        out["posts"]["touchdown_tax"] = {"text": tt, "chars": len(tt), "x_ok": len(tt) <= X_LIMIT, "n_legs": len(legs)}
    # ---- line moves since we opened -------------------------------------------------------------
    moves = []
    for r in card:
        m = r.get("line_move") or {}
        if m and (m.get("moved_pts") or 0) != 0 and m.get("n_snapshots", 1) > 1:
            moves.append((r, m))
    if moves:
        ml = "\n".join(f"▸ {r['name']} {str(r['side']).upper()} {_mkt(r['cat'])}: opened {_line(m['open_line'])} → now {_line(m['now_line'])} "
                       f"({'with us ✅' if m.get('with_us') else 'against us'})" for r, m in moves)
        with_us = sum(1 for _, m in moves if m.get("with_us"))
        lm = f"Line check, {wk}: {len(moves)} of our {len(card)} Card numbers have moved since we posted — {with_us} toward us.\n\n{ml}\n\nWe grade at the price we posted, not the price you can get now."
        out["posts"]["line_moves"] = {"text": lm, "chars": len(lm), "x_ok": len(lm) <= X_LIMIT}
    # ---- Monday report card (only from a graded week) ----------------------------------------
    rec = record if record is not None else (json.load(open(f"{DATA}/record.json")) if os.path.exists(f"{DATA}/record.json") else None)
    if rec and rec.get("weeks"):
        w = rec["weeks"][-1]; c = w.get("card") or {}
        if (c.get("w", 0) + c.get("l", 0)) > 0:
            tot = rec.get("card_season") or {}
            clv = f" · beat the close on avg by {c['clv_avg']:+.2f}" if c.get("clv_avg") is not None else ""
            rc = (f"REPORT CARD — Week {w['week']}\n\nThe Card: {c['w']}-{c['l']}" + (f"-{c['push']}" if c.get("push") else "")
                  + f" · {c['units']:+.2f}u{clv}\nSeason: {tot.get('w', 0)}-{tot.get('l', 0)} · {tot.get('units', 0):+.2f}u\n\n"
                  f"Every play graded at the posted price. Losses stay on the record.")
            out["posts"]["report_card"] = {"text": rc, "chars": len(rc), "x_ok": len(rc) <= X_LIMIT}
    return out


def _fair(r):
    """Market's de-vigged probability for our side (the honest 'the market says' number):
    the row's `fair` when present, else model_p - edge (edge is defined as model_p - fair)."""
    try:
        f = r.get("fair")
        if f is not None and f == f:
            return float(f)
        return float(r.get("model_p")) - float(r.get("edge"))
    except Exception:
        return None


def _fit(t, limit=X_LIMIT):
    if len(t) <= limit:
        return t
    cut = t[:limit - 1]
    for sep in (". ", "; ", ", ", " "):
        i = cut.rfind(sep)
        if i > limit * 0.6:
            return (cut[:i + (1 if sep == ". " else 0)]).rstrip() + ("" if sep == ". " else "…")
    return cut.rstrip() + "…"


def to_markdown(out):
    L = [f"# Post copy — {out.get('season')} week {out.get('week')} · generated {out.get('generated_at')}", ""]
    for k, v in out["posts"].items():
        L.append(f"## {k}" + ("" if v.get("x_ok", True) else f"  (over {X_LIMIT} chars — Discord / thread it)"))
        L.append("")
        if "thread" in v:
            for i, t in enumerate(v["thread"], 1):
                L.append(f"**{i}/{len(v['thread'])}** ({len(t)} chars)\n\n{t}\n")
        else:
            L.append(v["text"] + f"\n\n_({v.get('chars', len(v['text']))} chars)_\n")
    return "\n".join(L)


def write(board=None):
    out = build(board)
    json.dump(out, open(f"{DATA}/post_copy.json", "w"), indent=1)
    open(f"{DATA}/post_copy.md", "w").write(to_markdown(out))
    return out


if __name__ == "__main__":
    o = write()
    print(to_markdown(o))
