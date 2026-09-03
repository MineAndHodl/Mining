"""Grade a week from the ledger — the forward-grading loop (THE_DESK_PROMPT work order 2).

  python3 src/grade_week.py 2026 1            grade week 1 (needs the week's stats/scores posted)
  python3 src/grade_week.py 2026 1 --close    capture closing lines for the opened picks (run Sunday AM)

Results per pick: WIN / LOSS / PUSH / VOID (player did not play -> no action, as the books
settle it), units won at the ledger price, and CLV (opened line vs closing line, in the
direction of our side). Lanes are graded SEPARATELY and never mixed (the record rule):
  card     — the straight bets we sent (the only record that counts)
  stack    — Sunday Stack (parlay record)
  tax      — Touchdown Tax (parlay record)
  sgp      — primetime SGPs (lotto record, graded leg by leg + all-hit)
  games    — intel lane: CLV only (did the retail number move toward Pinnacle?)
Also: calibration by grade and by model probability across the whole graded POOL (not only
the card) — that is the sample the weights get re-earned on. Appends to data/record.json.
"""
import os, sys, json, datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from names import norm_name
import ledger as L

DATA = L.DATA
STAT_COL = {"rec_yds": "receiving_yards", "rush_yds": "rushing_yards", "pass_yds": "passing_yards",
            "receptions": "receptions", "rush_att": "carries", "pass_att": "attempts", "pass_cmp": "completions"}
TEAM_FIX = {"LAR": "LA", "OAK": "LV", "SD": "LAC", "STL": "LA", "WSH": "WAS"}


# ------------------------------------------------------------------ settlement math
def units_won(result, price, stake=1.0):
    """Units returned net of stake for a 1u (or `stake`) bet at an American price."""
    if result == "WIN":
        p = float(price); return round(stake * (p / 100 if p > 0 else 100 / abs(p)), 3)
    if result == "LOSS": return -stake
    return 0.0


def settle_prop(side, line, actual):
    if actual is None or (isinstance(actual, float) and actual != actual): return "VOID"
    a, ln = float(actual), float(line)
    if a == ln: return "PUSH"
    return "WIN" if ((a > ln) == (side == "Over")) else "LOSS"


def settle_td(actual_tds, played):
    if not played: return "VOID"
    return "WIN" if (actual_tds or 0) > 0 else "LOSS"


def settle_game(kind, side, point, home, away, home_score, away_score):
    """kind: spread / total / ML. side: team name (spread/ML) or Over/Under (total)."""
    if home_score is None or away_score is None: return "VOID"
    hs, as_ = float(home_score), float(away_score)
    if kind == "total":
        tot = hs + as_
        if tot == float(point): return "PUSH"
        return "WIN" if ((tot > float(point)) == (side == "Over")) else "LOSS"
    is_home = side == home
    margin = (hs - as_) if is_home else (as_ - hs)
    if kind == "ML":
        if margin == 0: return "PUSH"
        return "WIN" if margin > 0 else "LOSS"
    adj = margin + float(point)
    if adj == 0: return "PUSH"
    return "WIN" if adj > 0 else "LOSS"


def clv_points(side, opened_line, close_line, cat=None):
    """Closing-line value in the direction of our side (positive = we beat the close)."""
    if opened_line is None or close_line is None: return None
    o, c = float(opened_line), float(close_line)
    return round((c - o) if side in ("Over",) else (o - c), 2) if side in ("Over", "Under") else round(c - o, 2)


def parse_pick_line(pick):
    """'Buffalo Bills +1' -> ('Buffalo Bills', 1.0); 'Under 42.5' -> ('Under', 42.5); 'X ML' -> ('X', None)."""
    s = str(pick).strip()
    if s.endswith(" ML"): return s[:-3], None
    parts = s.rsplit(" ", 1)
    try: return parts[0], float(parts[1])
    except Exception: return s, None


# ------------------------------------------------------------------ actuals
def load_week_stats(season, week):
    """Per-player stats for the week (nflverse). Returns DataFrame keyed by normalized name."""
    try:
        import refresh_data as rd
        st = rd._load("load_player_stats", seasons=[season])
    except Exception as e:
        raise RuntimeError(f"player stats unavailable: {e}")
    st = st[(st["week"] == week) & (st.get("season_type", "REG") == "REG")] if "season_type" in st.columns else st[st["week"] == week]
    name_col = "player_display_name" if "player_display_name" in st.columns else "player_name"
    st = st.copy(); st["_k"] = st[name_col].map(norm_name)
    st["_tds"] = st.get("rushing_tds", 0).fillna(0) + st.get("receiving_tds", 0).fillna(0)
    return st


def load_week_scores(season, week):
    try:
        import refresh_data as rd
        s = rd._load("load_schedules", seasons=[season])
    except Exception:
        s = pd.read_parquet(f"{DATA}/schedules.parquet")
    s = s[(s.season == season) & (s.week == week)]
    out = {}
    for _, g in s.iterrows():
        out[(g.away_team, g.home_team)] = {"home_score": g.home_score if g.home_score == g.home_score else None,
                                           "away_score": g.away_score if g.away_score == g.away_score else None,
                                           "spread_line": g.spread_line, "total_line": g.total_line}
    return out


# ------------------------------------------------------------------ closing lines
def capture_close(season, week):
    """Closing lines for the OPENED picks: the last line-history snapshot (consensus line across
    books + best price at each side), falling back to the last raw board. Run it right after
    the Sunday-morning run (or the late-swap run) — that fetch IS the closing snapshot."""
    led = L.load(season, week)
    if not led: raise SystemExit("no ledger for that week")
    close = {"taken_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="minutes"), "props": {}, "td": {}, "games": {}}
    try:
        import line_history as lh
        files = lh._snapshots(season, week, "props")
        if files:
            p = pd.read_parquet(files[-1]); close["source"] = os.path.basename(files[-1])
            g = p.groupby(["_k", "mkt"]).agg(line=("line", "median"), over_price=("over_price", "max"), under_price=("under_price", "max"), n_books=("book", "nunique"))
            for (k, mkt), r in g.iterrows():
                close["props"][f"{k}|{mkt}"] = {"line": float(r["line"]), "over_price": int(r["over_price"]) if r["over_price"] == r["over_price"] else None,
                                               "under_price": int(r["under_price"]) if r["under_price"] == r["under_price"] else None, "n_books": int(r["n_books"])}
    except Exception as e:
        print("line-history close not captured:", e)
    if not close["props"]:
        try:
            pb = pd.read_parquet(f"{DATA}/_raw_prop_board.parquet"); pb["_k"] = pb["name"].map(norm_name)
            for _, r in pb.iterrows():
                close["props"][f"{r['_k']}|{r['mkt']}"] = {"line": float(r["line"]), "over_price": int(r.get("over_price_kept") or 0) or None,
                                                           "under_price": int(r.get("under_price_kept") or 0) or None, "book": r.get("book")}
        except Exception as e:
            print("prop close not captured:", e)
    try:
        td = pd.read_parquet(f"{DATA}/_raw_td_prices.parquet"); td["_k"] = td["name"].map(norm_name)
        for _, r in td.iterrows():
            close["td"][r["_k"]] = {"price": int(r["price"]), "book": r.get("book"), "implied_consensus": r.get("implied_consensus")}
    except Exception as e:
        print("td close not captured:", e)
    for g in (led.get("latest") or {}).get("games", []):
        close["games"][f"{g['game']}|{g['market']}|{g['side']}"] = {"point": g.get("point"), "price": g.get("price"), "book": g.get("book")}
    led["close"] = close
    json.dump(led, open(L.ledger_path(season, week), "w"), indent=1)
    print(f"close captured: {len(close['props'])} props, {len(close['td'])} td, {len(close['games'])} games")
    return led


# ------------------------------------------------------------------ grading
def grade_pick(p, stats, scores, close):
    k = norm_name(p.get("name") or ""); cat = p.get("cat"); side = p.get("side")
    row = stats[stats["_k"] == k] if stats is not None else pd.DataFrame()
    played = len(row) > 0
    res = {"id": p.get("id"), "name": p.get("name"), "cat": cat, "side": side, "line": p.get("line"), "price": p.get("price"),
           "grade": p.get("grade"), "units": p.get("units"), "model_p": p.get("model_p"), "edge": p.get("edge"), "played": played}
    if cat == "anytime_td":
        tds = float(row["_tds"].iloc[0]) if played else None
        res["actual"] = tds; res["result"] = settle_td(tds, played)
        c = (close or {}).get("td", {}).get(k)
        res["close_price"] = c.get("price") if c else None
    else:
        col = STAT_COL.get(cat)
        actual = float(row[col].iloc[0]) if (played and col in row.columns) else None
        res["actual"] = actual; res["result"] = settle_prop(side, p.get("line"), actual) if played else "VOID"
        c = (close or {}).get("props", {}).get(f"{k}|{cat}")
        res["close_line"] = c.get("line") if c else None
        res["clv_pts"] = clv_points(side, p.get("line"), res["close_line"])
    res["units_won"] = units_won(res["result"], p.get("price") or -110, stake=float(p.get("units") or 1.0))
    res["units_won_flat"] = units_won(res["result"], p.get("price") or -110, stake=1.0)
    return res


def grade_parlay(legs, stats, scores, close, leg_fn):
    out = [leg_fn(l) for l in legs]
    results = [o["result"] for o in out]
    if any(r == "LOSS" for r in results): overall = "LOSS"
    elif all(r in ("WIN", "VOID", "PUSH") for r in results) and any(r == "WIN" for r in results): overall = "WIN"   # voids drop out
    else: overall = "VOID"
    return out, overall


def grade(season, week, write=True):
    led = L.load(season, week)
    if not led: raise SystemExit(f"no ledger for {season} week {week}")
    opened = led["opened"]; close = led.get("close")
    stats = load_week_stats(season, week)
    scores = load_week_scores(season, week)
    # --- the Card
    card = [grade_pick(p, stats, scores, close) for p in opened["card"]]
    # --- the LATE SWAP lane (from the latest snapshot — it is built Sunday, not Tuesday)
    late_rows = list((led.get("late_swap_all") or {}).values()) or (led.get("latest") or {}).get("late_swap", [])
    late = [grade_pick(p, stats, scores, close) for p in late_rows]
    # --- the pool (calibration sample): every graded candidate, flat 1u
    pool = [grade_pick(p, stats, scores, close) for p in opened.get("pool", [])]
    # --- Sunday Stack
    stack = opened.get("sunday_stack"); stack_res = None
    if stack:
        legs, overall = grade_parlay(stack["legs"], stats, scores, close,
                                     lambda l: grade_pick({"name": l["name"], "cat": l["cat"], "side": l["side"], "line": l["line"], "price": l["price"], "id": l["leg"]}, stats, scores, close))
        stack_res = {"legs": legs, "result": overall, "price": stack["combined_american"], "units_won": units_won(overall, stack["combined_american"], 0.25)}
    # --- Touchdown Tax
    tax = opened.get("td_stack"); tax_res = None
    if tax:
        legs, overall = grade_parlay(tax["legs"], stats, scores, close,
                                     lambda l: grade_pick({"name": l["name"], "cat": "anytime_td", "side": "Yes", "line": 0.5, "price": l["price"], "id": l["leg"]}, stats, scores, close))
        tax_res = {"legs": legs, "result": overall, "price": tax["combined_american"], "units_won": units_won(overall, tax["combined_american"], 0.25)}
    # --- SGPs (leg by leg; 'all hit' is the ticket)
    sgp_res = []
    for s in opened.get("sgps", []):
        def _leg(l):
            mk = l.get("mkt"); nm = l.get("player")
            if mk == "anytime_td":
                return grade_pick({"name": nm, "cat": "anytime_td", "side": "Yes", "line": 0.5, "price": l.get("price"), "id": l["leg"]}, stats, scores, close)
            side = "Over" if " over " in l["leg"] else "Under"
            return grade_pick({"name": nm, "cat": mk, "side": side, "line": l.get("line"), "price": l.get("price"), "id": l["leg"]}, stats, scores, close)
        legs, overall = grade_parlay(s["legs"], stats, scores, close, _leg)
        sgp_res.append({"game": s["game"], "legs": legs, "result": overall, "hits": sum(1 for l in legs if l["result"] == "WIN"), "n": len(legs)})
    # --- games intel lane: result + CLV vs the closing (latest) number
    games_res = []
    for g in opened.get("games", []):
        away, home = L.game_teams(g["game"]); sc = scores.get((away, home), {})
        side_name, pt = parse_pick_line(g["pick"])
        side = side_name if g["kind"] != "total" else g["side"]
        r = settle_game(g["kind"], side, pt, g["game"].split(" @ ")[1], g["game"].split(" @ ")[0], sc.get("home_score"), sc.get("away_score"))
        c = (close or {}).get("games", {}).get(f"{g['game']}|{g['market']}|{g['side']}")
        games_res.append({"pick": g["pick"], "kind": g["kind"], "book": g["book"], "result": r, "opened_point": pt,
                          "close_point": c.get("point") if c else None,
                          "clv_pts": (round((c["point"] - pt) * (1 if g["kind"] != "total" or g["side"] == "Under" else -1), 2) if (c and c.get("point") is not None and pt is not None) else None)})

    def _rec(rows):
        w = sum(1 for r in rows if r["result"] == "WIN"); l = sum(1 for r in rows if r["result"] == "LOSS"); p = sum(1 for r in rows if r["result"] == "PUSH")
        v = sum(1 for r in rows if r["result"] == "VOID")
        return {"w": w, "l": l, "push": p, "void": v, "units": round(sum(r.get("units_won", 0) for r in rows), 2),
                "units_flat": round(sum(r.get("units_won_flat", 0) for r in rows), 2),
                "hit": round(w / (w + l), 3) if (w + l) else None,
                "clv_avg": round(float(np.mean([r["clv_pts"] for r in rows if r.get("clv_pts") is not None])), 2) if any(r.get("clv_pts") is not None for r in rows) else None}

    # calibration by grade and by model_p over the pool
    def _cal(rows, key):
        out = {}
        for r in rows:
            if r["result"] not in ("WIN", "LOSS"): continue
            b = r.get(key)
            if key == "model_p":
                if b is None: continue
                b = f"{int(float(b)*10)/10:.1f}"
            out.setdefault(b, {"n": 0, "w": 0, "p_sum": 0.0})
            out[b]["n"] += 1; out[b]["w"] += (r["result"] == "WIN"); out[b]["p_sum"] += float(r.get("model_p") or 0.5)
        return {k: {"n": v["n"], "hit": round(v["w"] / v["n"], 3), "pred": round(v["p_sum"] / v["n"], 3)} for k, v in sorted(out.items())}

    results = {"graded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="minutes"),
               "card": {"picks": card, "record": _rec(card)},
               "late_swap": {"picks": late, "record": _rec(late)},
               "pool": {"record": _rec(pool), "by_grade": _cal(pool, "grade"), "by_model_p": _cal(pool, "model_p")},
               "stack": stack_res, "tax": tax_res, "sgps": sgp_res,
               "games": {"rows": games_res, "record": _rec(games_res)}}
    led["results"] = results; led["status"] = "graded"
    if write:
        json.dump(led, open(L.ledger_path(season, week), "w"), indent=1)
        _append_record(season, week, results)
    return results


def _append_record(season, week, results):
    p = f"{DATA}/record.json"
    rec = json.load(open(p)) if os.path.exists(p) else {"weeks": []}
    rec["weeks"] = [w for w in rec["weeks"] if not (w["season"] == season and w["week"] == week)]
    rec["weeks"].append({"season": season, "week": week, "card": results["card"]["record"], "pool": results["pool"]["record"],
                         "late_swap": results.get("late_swap", {}).get("record"),
                         "stack": (results["stack"] or {}).get("result"), "stack_units": (results["stack"] or {}).get("units_won"),
                         "tax": (results["tax"] or {}).get("result"), "tax_units": (results["tax"] or {}).get("units_won"),
                         "sgps": [(s["result"], s["hits"], s["n"]) for s in results["sgps"]],
                         "games_clv": results["games"]["record"].get("clv_avg")})
    rec["weeks"].sort(key=lambda w: (w["season"], w["week"]))
    tot = {"w": 0, "l": 0, "push": 0, "units": 0.0}
    for w in rec["weeks"]:
        for k in ("w", "l", "push"): tot[k] += w["card"][k]
        tot["units"] = round(tot["units"] + w["card"]["units"], 2)
    rec["card_season"] = tot
    json.dump(rec, open(p, "w"), indent=1)


def summary_text(results):
    c = results["card"]["record"]
    lines = [f"THE CARD: {c['w']}-{c['l']}" + (f"-{c['push']}" if c['push'] else "") + f" ({c['units']:+.2f}u sized, {c['units_flat']:+.2f}u flat)"
             + (f", avg CLV {c['clv_avg']:+.2f}" if c.get("clv_avg") is not None else "")]
    for pk in results["card"]["picks"]:
        lines.append(f"  {pk['result']:5s} {pk['name']} {pk['side']} {pk['line']} ({pk['price']:+d}) actual {pk['actual']} grade {pk['grade']}")
    ls = results.get("late_swap", {}).get("record") or {}
    if ls.get("w", 0) + ls.get("l", 0): lines.append(f"LATE SWAP: {ls['w']}-{ls['l']} ({ls['units_flat']:+.2f}u flat)")
    if results["stack"]: lines.append(f"SUNDAY STACK: {results['stack']['result']} ({results['stack']['price']:+d})")
    if results["tax"]: lines.append(f"TOUCHDOWN TAX: {results['tax']['result']} ({results['tax']['price']:+d})")
    for s in results["sgps"]: lines.append(f"SGP {s['game']}: {s['hits']}/{s['n']} legs hit — {s['result']}")
    p = results["pool"]
    lines.append(f"POOL (calibration, flat 1u, n={p['record']['w']+p['record']['l']}): hit {p['record']['hit']}, {p['record']['units_flat']:+.2f}u; by grade: " +
                 ", ".join(f"{g} {v['hit']:.0%} (n={v['n']})" for g, v in p["by_grade"].items()))
    g = results["games"]["record"]
    lines.append(f"GAMES INTEL: {g['w']}-{g['l']}" + (f", avg CLV {g['clv_avg']:+.2f} pts" if g.get("clv_avg") is not None else ""))
    return "\n".join(lines)


if __name__ == "__main__":
    season, week = int(sys.argv[1]), int(sys.argv[2])
    if "--close" in sys.argv:
        capture_close(season, week)
    else:
        r = grade(season, week)
        print(summary_text(r))
