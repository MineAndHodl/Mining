"""Intel feed — the beat-writer layer, fed AUTOMATICALLY from free sources every run.

player_intel.py used to need an in-session search pass (RSS "not reachable"). It is reachable
now, and better sources exist for free:
  ESPN core injuries   dated player notes for every team (RotoWire-style: "Hand (leg) participated
                       at practice Tuesday, Daniel Flick of The AJC reports" + a long comment),
                       one item per note, the beat source they cite parsed out
  RSS                  ESPN / Yahoo / CBS / ProFootballTalk / Google News, matched to board players
  Sleeper players      injury status + body part + practice participation + notes, with a timestamp
  Sleeper trending     the crowd's adds and drops in the last 24h (role news leaks here first)
Every item carries date, source, title, snippet, url — the same shape player_intel.classify_items
reads — and is merged into data/player_intel_<season>_w<week>.json without touching any human
'read' already there. Hard gates (ruled out / pitch count) and soft signals then flow into the
board exactly as before.

    python3 src/intel_feed.py [--season 2026 --week 1] [--dry]
"""
import os, sys, re, json, glob, datetime
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data"); UNI = os.path.join(DATA, "universe")
sys.path.insert(0, HERE)
from names import norm_name

CITE = re.compile(r"(?:,|\.)?\s*(?:according to|per|as reported by|reports?|report)\s+([A-Z][\w.'\-]+(?:\s+[A-Z][\w.'\-]+){0,3}(?:\s+of\s+(?:The\s+)?[A-Z][\w.'\-]+(?:\s+[A-Z][\w.'\-]+){0,4})?)", re.I)
CITE2 = re.compile(r"([A-Z][\w.'\-]+(?:\s+[A-Z][\w.'\-]+){0,2}\s+of\s+(?:the\s+team's\s+official\s+site|(?:The\s+)?[A-Z][\w.'\-]+(?:\s+[A-Z][\w.'\-]+){0,4}))\s+reports?", re.I)
CITE3 = re.compile(r"(The Athletic's|ESPN's|NFL Network's|The Boston Globe's|[A-Z][\w.'\-]+(?:\s+[A-Z][\w.'\-]+)*'s)\s+([A-Z][\w.'\-]+\s+[A-Z][\w.'\-]+)")


def _source(text):
    for rx in (CITE2, CITE):
        m = rx.search(text or "")
        if m:
            return m.group(1).strip()
    m = CITE3.search(text or "")
    if m:
        return f"{m.group(2)} ({m.group(1).rstrip(chr(39) + 's')})"
    return None


def _board_players():
    names = set()
    for f in ("_raw_prop_lines.parquet", "_raw_td_board.parquet", "_raw_prop_board.parquet"):
        p = os.path.join(DATA, f)
        if os.path.exists(p):
            try: names |= set(pd.read_parquet(p, columns=["name"])["name"].dropna().unique().tolist())
            except Exception: pass
    return names


def _crosswalk():
    """espn_id -> display name, sleeper_id -> name, for the players we care about."""
    espn = {}; sleeper = {}
    try:
        pl = pd.read_parquet(os.path.join(UNI, "players.parquet"), columns=["display_name", "espn_id", "gsis_id"])
        for r in pl.dropna(subset=["espn_id"]).itertuples():
            try: espn[str(int(float(r.espn_id)))] = r.display_name
            except Exception: pass
    except Exception: pass
    try:
        sp = pd.read_parquet(os.path.join(UNI, "sleeper_players.parquet"))
        for r in sp.itertuples():
            nm = r.full_name if isinstance(r.full_name, str) and r.full_name != "None" else None
            if not nm: continue
            sleeper[str(r.player_id)] = r
            e = getattr(r, "espn_id", None)
            if e and str(e) not in ("None", "nan"):
                try: espn.setdefault(str(int(float(e))), nm)
                except Exception: pass
    except Exception: pass
    return espn, sleeper


def _iso(d):
    if d is None or (isinstance(d, float) and d != d) or str(d).strip() == "":
        return None
    try:
        return pd.to_datetime(d, utc=True).strftime("%Y-%m-%d")
    except Exception:
        pass
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(str(d).replace(" EST", " -0500").replace(" EDT", " -0400").replace(" PST", " -0800").replace(" PDT", " -0700")).strftime("%Y-%m-%d")
    except Exception:
        return None


def build_items(board_names=None):
    """{norm_name: {"name": display, "items": [...]}} from every feed."""
    board_names = board_names or _board_players()
    keys = {norm_name(n): n for n in board_names}
    espn, sleeper = _crosswalk()
    out = {}
    def add(name, item):
        k = norm_name(name)
        if k not in keys: return
        rec = out.setdefault(k, {"name": keys[k], "items": []})
        rec["items"].append(item)
    # ---- ESPN core injuries (RotoWire-style notes) ------------------------------------------
    p = os.path.join(UNI, "espn_injuries.parquet")
    if os.path.exists(p):
        inj = pd.read_parquet(p)
        for r in inj.itertuples():
            nm = espn.get(str(r.espn_id)) if r.espn_id else None
            if not nm: continue
            short = r.short if isinstance(r.short, str) else ""; long = r.long if isinstance(r.long, str) else ""
            src = _source(short) or _source(long) or "rotowire/espn"
            add(nm, {"date": _iso(r.date), "source": src, "title": short[:300], "snippet": long[:500], "url": r.athlete_ref if isinstance(r.athlete_ref, str) else None,
                     "status": r.status, "feed": "espn_injuries"})
    # ---- RSS headlines matched to board players ---------------------------------------------
    rss = sorted(glob.glob(os.path.join(UNI, "rss", "*.parquet")))
    if rss:
        h = pd.read_parquet(rss[-1])
        lower = {k: v.lower() for k, v in keys.items()}
        for r in h.itertuples():
            text = f"{r.title or ''} {r.summary or ''}"; tl = text.lower()
            for k, nm in keys.items():
                if lower[k] and lower[k] in tl:
                    add(nm, {"date": _iso(r.published), "source": f"rss:{r.source}", "title": (r.title or "")[:300], "snippet": re.sub(r"<[^>]+>", "", r.summary or "")[:400], "url": r.link, "feed": "rss"})
    # ---- Sleeper player status ---------------------------------------------------------------
    for sid, r in sleeper.items():
        nm = r.full_name
        if norm_name(nm) not in keys: continue
        st = getattr(r, "injury_status", None); st = st if isinstance(st, str) and st not in ("None", "nan") else None
        nu = getattr(r, "news_updated", None)
        try: nd = datetime.datetime.fromtimestamp(float(nu) / 1000, datetime.timezone.utc).strftime("%Y-%m-%d") if nu and str(nu) not in ("None", "nan") else None
        except Exception: nd = None
        if st:
            body = getattr(r, "injury_body_part", None); prac = getattr(r, "practice_participation", None); notes = getattr(r, "injury_notes", None)
            bits = [f"injury status {st}"] + ([f"({body})"] if isinstance(body, str) and body not in ("None", "nan") else []) + \
                   ([f"practice: {prac}"] if isinstance(prac, str) and prac not in ("None", "nan") else [])
            add(nm, {"date": nd or datetime.date.today().isoformat(), "source": "sleeper", "title": "Sleeper: " + ", ".join(bits),
                     "snippet": notes if isinstance(notes, str) and notes not in ("None", "nan") else "", "url": None, "feed": "sleeper_status"})
    # ---- Coach press conferences (coach_feed): quotes and coach-speak about board players ---------
    try:
        import coach_feed as cfd
        for k, rec in cfd.items(players=set(keys.values())).items():
            for it in rec["items"]:
                add(rec["name"], it)
    except Exception as _e:
        print("coach feed items skipped:", _e)
    # ---- Sleeper trending adds / drops --------------------------------------------------------
    for kind, label in (("sleeper_trending", "trending add"), ("sleeper_drops", "trending drop")):
        fs = sorted(glob.glob(os.path.join(UNI, kind, "*.json")))
        if not fs: continue
        try: js = json.load(open(fs[-1]))
        except Exception: continue
        stamp = os.path.basename(fs[-1])[:8]; d = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
        for rank, it in enumerate(js[:60], 1):
            r = sleeper.get(str(it.get("player_id")))
            if r is None: continue
            add(r.full_name, {"date": d, "source": "sleeper", "title": f"Sleeper {label}: #{rank} in the last 24h ({int(it.get('count', 0)):,} leagues)",
                              "snippet": "", "url": None, "feed": kind})
    return out


def merge_into_week(season, week, items, window=None, write=True):
    path = os.path.join(DATA, f"player_intel_{season}_w{week}.json")
    cur = json.load(open(path)) if os.path.exists(path) else {}
    if "_window" not in cur:
        if window is None:
            window = (datetime.date.today() - datetime.timedelta(days=12), datetime.date.today() + datetime.timedelta(days=7))
        cur["_window"] = {"start": window[0].isoformat(), "end": window[1].isoformat()}
    cur["_method"] = (cur.get("_method", "") + " | auto-feed: ESPN injuries notes, RSS (ESPN/Yahoo/CBS/PFT/Google News), Sleeper status + trending").strip(" |")
    n_new = 0
    for k, rec in items.items():
        name = rec["name"]
        # keep the human key if it exists under a different display form
        key = next((kk for kk in cur if not kk.startswith("_") and norm_name(kk) == k), name)
        slot = cur.setdefault(key, {"items": []})
        have = {(it.get("source"), (it.get("title") or "")[:80]) for it in slot.get("items", [])}
        for it in rec["items"]:
            sig = (it.get("source"), (it.get("title") or "")[:80])
            if sig in have or not it.get("date"): continue
            slot.setdefault("items", []).append(it); have.add(sig); n_new += 1
    cur["_auto_feed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="minutes")
    if write:
        json.dump(cur, open(path, "w"), indent=1)
    return path, n_new, cur


def refresh_sources(coaches=True):
    try:
        import ingest_universe as iu
        for n in ("espn_injuries", "rss_news", "sleeper_players", "sleeper_trending", "sleeper_drops"):
            iu.fetch(n, verbose=False)
    except Exception as e:
        print("intel feed: source refresh skipped:", e)
    if coaches:
        try:
            import coach_feed as cfd
            cfd.refresh(verbose=True)
        except Exception as e:
            print("intel feed: coach feed skipped:", e)


def run(season, week, dry=False, refresh=True):
    if refresh:
        refresh_sources()
    items = build_items()
    n_items = sum(len(v["items"]) for v in items.values())
    if dry:
        print(f"intel feed: {len(items)} board players with {n_items} items")
        return items
    path, n_new, cur = merge_into_week(season, week, items)
    print(f"intel feed: {len(items)} players, {n_items} items, {n_new} new -> {os.path.basename(path)}")
    return cur


if __name__ == "__main__":
    a = sys.argv
    season = int(a[a.index("--season") + 1]) if "--season" in a else 2026
    week = int(a[a.index("--week") + 1]) if "--week" in a else 1
    run(season, week, dry="--dry" in a, refresh="--no-refresh" not in a)
