"""The Watch — constant monitoring of the free feeds, with a memory of what it has already seen.

Every poll: refresh the fast feeds (ESPN/RotoWire player notes, RSS headlines, Sleeper injury
status + trending, Underdog lines, Action Network game odds), diff them against the last poll,
and write only what CHANGED as events:
  note        a new dated player note about a player on our board (with the beat source)
  headline    a new headline naming a board player
  status      a Sleeper injury-status / practice change for a board player
  trending    a board player entering the top-25 adds or drops
  line        an Underdog player line that moved >= 0.5 (or a TD price move)
  game        a spread / total move on the Action Network consensus
State: data/monitor/state.json (what we've seen), data/monitor/feed.jsonl (events, append-only),
data/monitor/status.json (for the Model Room: last check, counts, what is being watched).

    python3 src/monitor.py            # one poll
    python3 src/monitor.py --dry      # show what would be logged
"""
import os, sys, json, glob, datetime
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data"); UNI = os.path.join(DATA, "universe")
MON = os.path.join(DATA, "monitor"); STATE = os.path.join(MON, "state.json"); FEED = os.path.join(MON, "feed.jsonl"); STATUS = os.path.join(MON, "status.json")
sys.path.insert(0, HERE)
from names import norm_name

FAST = ["espn_injuries", "rss_news", "sleeper_players", "sleeper_trending", "sleeper_drops", "underdog_lines", "action_scoreboard", "espn_news"]
COACHES = True            # also poll the coaches' press conferences (Google News -> team sites / papers)
MAX_AGE_DAYS = 7          # an item older than this is history, not an update


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def _board():
    names = set(); card = set()
    try:
        rb = json.load(open(os.path.join(DATA, "ranked_board.json")))
        for r in rb.get("recommended", []): card.add(r["name"])
        for k, rows in rb.get("by_category", {}).items():
            for r in rows: names.add(r["name"])
        for r in rb.get("best_props", []): names.add(r["name"])
    except Exception:
        pass
    for f in ("_raw_prop_lines.parquet", "_raw_td_board.parquet"):
        p = os.path.join(DATA, f)
        if os.path.exists(p):
            try: names |= set(pd.read_parquet(p, columns=["name"])["name"].dropna().unique().tolist())
            except Exception: pass
    return names, card


def poll(dry=False, refresh=True):
    os.makedirs(MON, exist_ok=True)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {"seen_notes": [], "seen_links": [], "status": {}, "trending": [], "drops": [], "ud_lines": {}, "games": {}}
    if refresh:
        try:
            import ingest_universe as iu
            for n in FAST:
                try: iu.fetch(n, verbose=False)
                except Exception as e: print(f"  {n}: {e}")
        except Exception as e:
            print("refresh skipped:", e)
    names, card = _board(); keys = {norm_name(n): n for n in names}
    now = _now(); ts = now.isoformat(timespec="minutes")
    events = []
    # --- coaches' press conferences: new quotes about board players ----------------------------
    try:
        import coach_feed as cfd
        if refresh and COACHES:
            cfd.refresh(verbose=False)
        seen_c = set(state.get("seen_coach", []))
        for k, rec in cfd.items(players=names).items():
            for it in rec["items"]:
                sig = f"{it.get('url')}|{(it.get('title') or '')[:60]}"
                if sig in seen_c: continue
                seen_c.add(sig)
                if not it.get("quote") and not it.get("coach_speak"): continue
                events.append({"ts": ts, "kind": "coach", "who": rec["name"], "on_card": rec["name"] in card, "text": (it.get("title") or "")[:280],
                               "source": it.get("source"), "url": it.get("url"), "date": it.get("date"), "quote": bool(it.get("quote"))})
        state["seen_coach"] = sorted(seen_c)[-20000:]
    except Exception as e:
        print("coach feed:", e)
    def ev(kind, who, text, source=None, url=None, extra=None):
        events.append({"ts": ts, "kind": kind, "who": who, "on_card": who in card, "text": text[:280], "source": source, "url": url, **(extra or {})})
    # --- player notes (ESPN core injuries feed) ------------------------------------------------
    try:
        import intel_feed as itf
        espn, sleeper = itf._crosswalk()
        p = os.path.join(UNI, "espn_injuries.parquet")
        if os.path.exists(p):
            inj = pd.read_parquet(p); seen = set(state["seen_notes"])
            for r in inj.itertuples():
                iid = str(r.injury_id)
                if iid in seen: continue
                nm = espn.get(str(r.espn_id)) if r.espn_id else None
                seen.add(iid)
                if not nm or norm_name(nm) not in keys: continue
                d_ = itf._iso(r.date)
                if d_ and d_ < (now - datetime.timedelta(days=MAX_AGE_DAYS)).strftime("%Y-%m-%d"): continue   # history, not an update
                short = r.short if isinstance(r.short, str) else ""
                ev("note", keys[norm_name(nm)], short, source=itf._source(short) or "rotowire/espn", url=r.athlete_ref if isinstance(r.athlete_ref, str) else None,
                   extra={"status": r.status, "date": itf._iso(r.date)})
            state["seen_notes"] = sorted(seen)[-20000:]
        # --- headlines ---------------------------------------------------------------------------
        rss = sorted(glob.glob(os.path.join(UNI, "rss", "*.parquet")))
        if rss:
            h = pd.read_parquet(rss[-1]); seen_l = set(state["seen_links"]); lower = {k: v.lower() for k, v in keys.items()}
            for r in h.itertuples():
                if not r.link or r.link in seen_l: continue
                seen_l.add(r.link); tl = f"{r.title or ''} {r.summary or ''}".lower()
                d_ = itf._iso(r.published)
                if d_ and d_ < (now - datetime.timedelta(days=MAX_AGE_DAYS)).strftime("%Y-%m-%d"): continue
                hits = [keys[k] for k in keys if lower[k] and lower[k] in tl]
                for nm in hits[:3]:
                    ev("headline", nm, r.title or "", source=f"rss:{r.source}", url=r.link, extra={"date": itf._iso(r.published)})
            state["seen_links"] = sorted(seen_l)[-20000:]
        # --- Sleeper status changes ----------------------------------------------------------------
        st_prev = state.get("status", {}); st_now = {}
        for sid, r in sleeper.items():
            nm = r.full_name
            k = norm_name(nm)
            if k not in keys: continue
            cur = f"{getattr(r, 'injury_status', None) or ''}|{getattr(r, 'practice_participation', None) or ''}|{getattr(r, 'injury_body_part', None) or ''}".replace("None", "")
            st_now[k] = cur
            if k in st_prev and st_prev[k] != cur and cur.strip("|"):
                ev("status", keys[k], f"Sleeper status: {cur.replace('|', ' · ').strip(' ·')} (was {st_prev[k].replace('|', ' · ').strip(' ·') or 'clear'})", source="sleeper")
        state["status"] = st_now
        # --- trending -----------------------------------------------------------------------------
        for kind, label in (("sleeper_trending", "adds"), ("sleeper_drops", "drops")):
            fs = sorted(glob.glob(os.path.join(UNI, kind, "*.json")))
            if not fs: continue
            js = json.load(open(fs[-1])); top = [str(x.get("player_id")) for x in js[:25]]
            prev = set(state.get(label, []))
            for rank, pid in enumerate(top, 1):
                r = sleeper.get(pid)
                if r is None or pid in prev: continue
                if norm_name(r.full_name) in keys:
                    ev("trending", keys[norm_name(r.full_name)], f"entered the top-25 Sleeper {label} (#{rank}, {int(js[rank-1].get('count', 0)):,} leagues in 24h)", source="sleeper")
            state[label] = top
    except Exception as e:
        print("player feeds:", e)
    # --- Underdog line moves -------------------------------------------------------------------
    try:
        import underdog as ud
        d = ud.latest()
        if len(d):
            d = d[d.mkt.notna() & ~d.live].drop_duplicates(["name", "stat"])
            prev = state.get("ud_lines", {}); cur = {}
            for r in d.itertuples():
                key = f"{r.name}|{r.stat}"; cur[key] = [r.line, r.over_price, r.under_price]
                if key in prev and norm_name(r.name) in keys:
                    pl, po, pu = prev[key]
                    if r.line is not None and pl is not None and abs(r.line - pl) >= 0.5:
                        ev("line", keys[norm_name(r.name)], f"Underdog {r.stat}: {pl:g} → {r.line:g}", source="underdog", extra={"mkt": r.mkt, "from": pl, "to": r.line})
                    elif r.mkt == "anytime_td" and po is not None and r.over_price is not None and abs(r.over_price - po) >= 20:
                        ev("line", keys[norm_name(r.name)], f"Underdog anytime TD price: {po:+d} → {r.over_price:+d}", source="underdog", extra={"mkt": r.mkt, "from": po, "to": r.over_price})
            state["ud_lines"] = cur
    except Exception as e:
        print("underdog:", e)
    # --- game odds moves (Action Network consensus) ---------------------------------------------
    try:
        fs = sorted(glob.glob(os.path.join(UNI, "action_scoreboard", "*.json")))
        if fs:
            js = json.load(open(fs[-1])); prev = state.get("games", {}); cur = {}
            for g in js.get("games", []):
                teams = {t["id"]: t.get("abbr") or t.get("display_name") for t in g.get("teams", [])}
                title = f"{teams.get(g.get('away_team_id'))} @ {teams.get(g.get('home_team_id'))}"
                cons = next((o for o in g.get("odds", []) if o.get("book_id") == 15 and o.get("type") == "game"), None) or next((o for o in g.get("odds", []) if o.get("type") == "game"), None)
                if not cons: continue
                cur[title] = [cons.get("spread_home"), cons.get("total")]
                if title in prev:
                    ps, pt = prev[title]
                    if ps is not None and cons.get("spread_home") is not None and abs(cons["spread_home"] - ps) >= 0.5:
                        ev("game", title, f"spread {ps:+g} → {cons['spread_home']:+g} (home)", source="action network")
                    if pt is not None and cons.get("total") is not None and abs(cons["total"] - pt) >= 0.5:
                        ev("game", title, f"total {pt:g} → {cons['total']:g}", source="action network")
            state["games"] = cur
    except Exception as e:
        print("games:", e)
    # --- write ----------------------------------------------------------------------------------
    if dry:
        for e in events[:40]: print(f"  {e['kind']:9s} {e['who']:26s} {e['text'][:120]}  [{e.get('source')}]")
        print(f"{len(events)} events (dry run)")
        return events
    json.dump(state, open(STATE, "w"))
    with open(FEED, "a") as f:
        for e in events: f.write(json.dumps(e) + "\n")
    # status for the Model Room
    day_ago = (now - datetime.timedelta(hours=24)).isoformat(timespec="minutes")
    recent = []
    if os.path.exists(FEED):
        with open(FEED) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if e.get("ts", "") >= day_ago: recent.append(e)
                except Exception: pass
    polls = (json.load(open(STATUS)).get("polls_24h", []) if os.path.exists(STATUS) else [])
    polls = [p for p in polls if p >= day_ago] + [ts]
    feeds = {}
    try:
        inv = json.load(open(os.path.join(DATA, "data_ledger.json")))
        for n in FAST:
            v = inv.get("sources", {}).get(n)
            if v: feeds[n] = {"refreshed": v.get("refreshed"), "rows": v.get("rows")}
    except Exception: pass
    status = {"last_check": ts, "polls_24h": polls, "events_24h": len(recent), "events_total": sum(1 for _ in open(FEED)) if os.path.exists(FEED) else 0,
              "players_watched": len(names), "card_players": len(card), "feeds": feeds,
              "by_kind_24h": {k: sum(1 for e in recent if e["kind"] == k) for k in ("note", "headline", "coach", "status", "trending", "line", "game")},
              "latest": sorted(recent, key=lambda e: (e.get("date") or e["ts"][:10], e["ts"]), reverse=True)[:14]}
    json.dump(status, open(STATUS, "w"), indent=1)
    print(f"watch: {len(events)} new events · {len(recent)} in 24h · watching {len(names)} players ({len(card)} on the Card) · {len(polls)} checks in 24h")
    return events


if __name__ == "__main__":
    poll(dry="--dry" in sys.argv, refresh="--no-refresh" not in sys.argv)
