"""LAST CALL — the kickoff-window product (Alex: "another hour to pregame last-minute bets
that change from stuff right before kickoff — last-minute edge could be a category and a
theme"). Runs ~75-90 minutes before each slate (run_board.py --window=150): only the games
about to kick, with fresh prices, the game-day inactives, the kickoff forecast, and every
opened number's move. It never touches the week's board or the opened ledger.

What "last-minute edge" honestly is, and what each item type means:
  VOID        an opened pick whose player is OUT (inactive) — do not bet; a parlay leg drops.
  LATE SWAP   a player lifted by a teammate's absence (measured cascade) who clears the Card
              bar — priced before the books finish adjusting. Own record.
  STALE       the best book for our side sits >= 1 pt behind the consensus RIGHT NOW on a
              Card-quality play — one book lagging a market that already moved. Own record.
  STILL LIVE  an opened Card pick whose number is still available (or better) — last chance.
  CLOSED      an opened Card pick whose number moved toward us — CLV banked, do not chase.
  WEATHER     the kickoff forecast that moved a number (wind / cold, outdoors).
"""
import os, sys, json, datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ledger as L
import weather_feed as wf

DATA = L.DATA
OUT = f"{DATA}/last_call.json"
GRADE_OK = ("A+", "A", "A-", "B+")
STALE_MIN_PTS = 1.0


def _kick(game, season):
    away, home = L.game_teams(game)
    try:
        s = pd.read_parquet(f"{DATA}/schedules.parquet")
        r = s[(s.season == season) & (s.away_team == away) & (s.home_team == home)]
        if len(r):
            r = r.iloc[0]; return wf.kickoff_utc(r["gameday"], r["gametime"]).isoformat(timespec="minutes")
    except Exception:
        pass
    return None


def _row(r, kind, note, extra=None):
    d = {"type": kind, "name": r.get("name"), "game": r.get("game"), "cat": r.get("cat"), "side": r.get("side"),
         "line": r.get("line"), "price": r.get("price"), "book": r.get("book"), "grade": r.get("grade"), "model_p": r.get("model_p"),
         "edge": r.get("edge"), "our_num": r.get("our_num"), "market_num": r.get("market_num"), "line_move": r.get("line_move"),
         "role_note": r.get("role_note"), "inj_status": r.get("inj_status"), "note": note}
    if extra: d.update(extra)
    return d


def build(ranked, pool, prev_full, season, week, window_min, wx):
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="minutes")
    games = sorted({r.get("game") for r in pool if r.get("game")} | {g.get("game") for g in ranked.get("best_games", []) or [] if g.get("game")})
    game_rows = [{"game": g, "kickoff_utc": _kick(g, season), "weather": wf.weather_note((wx or {}).get(g)), "wx": (wx or {}).get(g)} for g in games]
    led = L.load(season, week) if week else None
    opened = (led or {}).get("opened") or {}
    now_by_key = {}
    for r in pool:
        now_by_key[(r.get("name"), r.get("cat"), r.get("side"))] = r
    items = []
    # 1) opened Card picks in the window: void / still live / closed
    for p in opened.get("card", []):
        if p.get("game") not in games: continue
        cur = now_by_key.get((p.get("name"), p.get("cat"), p.get("side")))
        if cur is None:
            items.append(_row(p, "VOID?", "line is off the board right now — check the inactives before betting", {"opened": True})); continue
        if (cur.get("inj_status") or "") == "Out":
            items.append(_row(cur, "VOID", f"{p['name']} is OUT — do not bet; a parlay leg with him drops out", {"opened": True})); continue
        mv = cur.get("line_move") or {}
        if mv.get("still_available"):
            items.append(_row(cur, "STILL LIVE", f"opened {mv.get('open_line')} ({mv.get('open_price')}) — still there at {cur.get('book')}; last chance", {"opened": True}))
        elif mv.get("with_us"):
            items.append(_row(cur, "CLOSED", f"opened {mv.get('open_line')} → now {mv.get('now_line')}: moved with us (+{mv.get('moved_pts')}) — CLV banked, do not chase", {"opened": True}))
        else:
            items.append(_row(cur, "MOVED AGAINST", f"opened {mv.get('open_line')} → now {mv.get('now_line')}: a better number exists; still graded {cur.get('grade')}", {"opened": True}))
    # 2) parlay legs voided
    for lane in ("sunday_stack", "td_stack"):
        s_ = opened.get(lane) or {}
        for l in s_.get("legs", []) or []:
            if l.get("game") not in games: continue
            cat = l.get("cat") or ("anytime_td" if lane == "td_stack" else None)
            cur = now_by_key.get((l.get("name"), cat, l.get("side") or ("Yes" if cat == "anytime_td" else None)))
            if cur is not None and (cur.get("inj_status") or "") == "Out":
                items.append(_row(cur, "VOID", f"{l['name']} is OUT — the {'Sunday Stack' if lane == 'sunday_stack' else 'Touchdown Tax'} leg drops", {"lane": lane}))
    # 3) late swaps + stale numbers from the fresh window pool
    for r in pool:
        if r.get("kind") or r.get("grade") not in GRADE_OK or (r.get("model_p") or 0) < 0.5: continue
        if r.get("role_note") and "is out" in str(r.get("role_note")):
            items.append(_row(r, "LATE SWAP", r.get("role_note")))
            continue
        mv = r.get("line_move") or {}
        st = mv.get("stale_book_pts")
        if st is not None and st >= STALE_MIN_PTS:
            items.append(_row(r, "STALE", f"{r.get('book')} is {st:g} pt behind the consensus ({mv.get('now_consensus')}) on our side — a lagging number"))
    # 4) weather that moved a number
    for g in game_rows:
        w = g.get("wx") or {}
        if w and not w.get("error") and w.get("roof") not in ("dome", "closed", "indoors") and ((w.get("wind") or 0) >= 12 or (w.get("temp") is not None and w["temp"] < 45)):
            items.append({"type": "WEATHER", "game": g["game"], "note": f"kickoff forecast {g['weather']} — passing numbers trimmed, rushing lifted"})
    order = {"VOID": 0, "VOID?": 1, "LATE SWAP": 2, "STALE": 3, "STILL LIVE": 4, "WEATHER": 5, "MOVED AGAINST": 6, "CLOSED": 7}
    items.sort(key=lambda x: (order.get(x["type"], 9), -(x.get("model_p") or 0)))
    out = {"stamp": stamp, "window_min": window_min, "season": season, "week": week, "games": game_rows, "items": items,
           "counts": {k: sum(1 for i in items if i["type"] == k) for k in order}}
    json.dump(out, open(OUT, "w"), indent=1)
    # the ledger: kickoff entry + late swaps accumulated for Monday's grading
    if led is not None:
        led.setdefault("kickoff", {})[stamp] = {"games": games, "counts": out["counts"],
                                                "items": [{k: v for k, v in i.items() if k != "line_move"} for i in items]}
        ls = led.setdefault("late_swap_all", {})
        for i in items:
            if i["type"] in ("LATE SWAP", "STALE"):
                key = f"{i['name']}|{i['cat']}|{i['side']}|{i['line']}"
                if key not in ls:
                    ls[key] = {**{k: v for k, v in i.items() if k != "line_move"}, "id": key, "lane": i["type"].lower().replace(" ", "_"), "stamp": stamp,
                               "units": 0.5, "recommended": True}
        json.dump(led, open(L.ledger_path(season, week), "w"), indent=1)
    return out


def load_recent(max_age_hours=8):
    try:
        d = json.load(open(OUT))
        age = datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(d["stamp"])
        return d if age.total_seconds() <= max_age_hours * 3600 else None
    except Exception:
        return None


def text(out):
    if not out: return "no last call"
    lines = [f"LAST CALL {out['stamp']} — {len(out['games'])} games kicking within {out['window_min']} min"]
    for g in out["games"]:
        lines.append(f"  {g['game']}  kick {str(g.get('kickoff_utc') or '')[11:16]}Z  {g.get('weather') or ''}")
    for i in out["items"]:
        if i["type"] == "WEATHER": lines.append(f"  [WEATHER] {i['game']}: {i['note']}")
        else: lines.append(f"  [{i['type']}] {i.get('name')} {i.get('side') or ''} {i.get('line') or ''} ({i.get('price')}) @ {i.get('book')} {i.get('grade') or ''} — {i.get('note')}")
    return "\n".join(lines)


def games_in_window(minutes, season=None):
    """Games kicking off within `minutes` from now (from the schedule) — the prompt's gate."""
    now = datetime.datetime.now(datetime.timezone.utc)
    season = season or (now.year if now.month >= 3 else now.year - 1)
    s = pd.read_parquet(f"{DATA}/schedules.parquet"); s = s[(s.season == season) & (s.game_type == "REG")]
    out = []
    for _, r in s.iterrows():
        try: ko = wf.kickoff_utc(r["gameday"], r["gametime"])
        except Exception: continue
        if now - datetime.timedelta(minutes=5) <= ko <= now + datetime.timedelta(minutes=minutes):
            out.append({"game": f"{wf.FULL.get(r['away_team'], r['away_team'])} @ {wf.FULL.get(r['home_team'], r['home_team'])}",
                        "week": int(r["week"]), "kickoff_utc": ko.isoformat(timespec="minutes"), "minutes_to_kick": int((ko - now).total_seconds() // 60)})
    return out


if __name__ == "__main__":
    arg = next((a for a in sys.argv if a.startswith("--games-in=")), None)
    if arg:
        g = games_in_window(int(arg.split("=")[1]))
        if not g: print("none")
        for x in g: print(f"{x['game']} | week {x['week']} | kick {x['kickoff_utc']} | in {x['minutes_to_kick']} min")
    else:
        print(text(load_recent(48)))
