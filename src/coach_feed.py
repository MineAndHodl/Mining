"""Coach feed — what the head coaches actually said this week, from their press conferences.

Alex: "Can we analyze coaches' pre/post game pressers during the week — the transcripts?"
Yes, and for free: team sites publish full transcripts ("Transcript: Head Coach Mike Vrabel
Press Conference 9/1"), local papers and ESPN quote the pressers, and Google News RSS indexes
all of it within hours. This module, per head coach (from the schedule's coach columns):
  1. queries Google News RSS for "<coach>" press conference / transcript / said, last 7 days
  2. resolves the Google redirect links to the real article (Google's own decode endpoint)
  3. fetches the article, keeps the body text (team-site CMS or generic paragraphs), caches it
  4. pulls every sentence that names a player on our board, plus the coach-speak around it
     ("game-time decision", "no restrictions", "get him the ball", "committee", "pitch count")
  5. emits intel items {date, source: "Coach <name> · <outlet>", title: the sentence, ...} that
     player_intel.classify_items reads like any beat note (hard gates + soft signals)
Transcripts are stored in data/universe/coach_feed/ (text) and counted in the inventory.

    python3 src/coach_feed.py            # refresh + print a summary
    python3 src/coach_feed.py --dry
"""
import os, sys, re, json, glob, time, datetime, hashlib
import requests
import pandas as pd
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data"); UNI = os.path.join(DATA, "universe")
CF = os.path.join(UNI, "coach_feed"); RESOLVED = os.path.join(CF, "resolved.json")
sys.path.insert(0, HERE)
from names import norm_name

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) TheDesk/1.0"}
MAX_LINKS_PER_COACH = 6
COACH_SPEAK = [r"game-?time decision", r"day-?to-?day", r"we'?ll see", r"no restrictions", r"full go", r"ready to roll", r"good to go", r"on track",
               r"held out", r"limited", r"didn'?t practice", r"pitch count", r"snap count", r"ramp", r"ease", r"get him the ball", r"more involved",
               r"featured", r"lean on", r"big part", r"workhorse", r"our guy", r"starter", r"starting", r"expanded", r"package", r"committee",
               r"rotat", r"hot hand", r"share", r"spell", r"role", r"touches", r"targets", r"carries", r"snaps", r"reps", r"first[- ]team",
               r"questionable", r"doubtful", r"out\b", r"ruled", r"injur", r"sore", r"tight", r"progress", r"cleared", r"practice"]


def coaches(season=2026):
    """{team: head coach} from the schedule (home_coach / away_coach), latest season."""
    sch = pd.read_parquet(os.path.join(DATA, "schedules.parquet"))
    s = sch[sch.season == season]
    out = {}
    for r in s.itertuples():
        if isinstance(getattr(r, "home_coach", None), str): out[r.home_team] = r.home_coach
        if isinstance(getattr(r, "away_coach", None), str): out[r.away_team] = r.away_coach
    return out


def _rss(query):
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    r = requests.get(url, headers=UA, timeout=25); r.raise_for_status()
    root = ET.fromstring(r.content)
    items = []
    for it in root.iter("item"):
        items.append({"title": (it.findtext("title") or "").strip(), "link": (it.findtext("link") or "").strip(),
                      "published": (it.findtext("pubDate") or "").strip(), "source": (it.find("source").text if it.find("source") is not None else "")})
    return items


def resolve(link, cache):
    """Google News RSS link -> real article URL (Google's batchexecute decode), cached."""
    if link in cache:
        return cache[link]
    url = None
    try:
        tok = link.split("/articles/")[1].split("?")[0]
        page = requests.get(link, headers=UA, timeout=25).text
        sig = re.search(r'data-n-a-sg="([^"]+)"', page); ts = re.search(r'data-n-a-ts="([^"]+)"', page)
        if sig and ts:
            req = [["Fbv4je", f'["garturlreq",[["X","X",["X","X"],null,null,1,1,"US:en",null,1,null,null,null,null,null,0,1],"X","X",1,[1,1,1],1,1,null,0,0,null,0],"{tok}",{ts.group(1)},"{sig.group(1)}"]', None, "generic"]]
            rb = requests.post("https://news.google.com/_/DotsSplashUi/data/batchexecute",
                               headers={**UA, "content-type": "application/x-www-form-urlencoded;charset=UTF-8"},
                               data={"f.req": json.dumps([req])}, timeout=25)
            m = re.search(r'garturlres\\",\\"(https?://[^\\"]+)', rb.text) or re.search(r'"garturlres","(https?://[^"]+)"', rb.text.replace("\\", ""))
            if m: url = m.group(1)
    except Exception:
        url = None
    cache[link] = url
    return url


def _clean(txt):
    import html as _html
    txt = _html.unescape(txt)
    if "â€" in txt or "Ã" in txt:
        try: txt = txt.encode("latin-1", "ignore").decode("utf-8", "ignore")
        except Exception: pass
    txt = txt.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"').replace("\u2014", " — ")
    return re.sub(r"[ \t]+", " ", txt).strip()


def article_text(url):
    """Body text of an article: the team-site CMS body, else the paragraphs of <article>/<main>,
    with scripts, nav, headers, footers and asides stripped and short fragments dropped."""
    try:
        r = requests.get(url, headers=UA, timeout=25); r.raise_for_status(); html = r.text
    except Exception:
        return None
    html = re.sub(r"<(script|style|nav|header|footer|aside|form|svg|figure)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    body = None
    m = re.search(r'<div class="nfl-c-article__body">(.*?)(?:<div class="nfl-c-article__footer|</article>)', html, re.S)
    if m: body = m.group(1)
    if not body:
        m = re.search(r"<article[^>]*>(.*?)</article>", html, re.S) or re.search(r"<main[^>]*>(.*?)</main>", html, re.S)
        body = m.group(1) if m else html
    paras = [re.sub(r"<[^>]+>", " ", p_) for p_ in re.findall(r"<p[^>]*>(.*?)</p>", body, re.S)]
    paras = [_clean(p_) for p_ in paras]
    paras = [p_ for p_ in paras if len(p_) >= 40 and not re.match(r"(skip|search|submit|sign up|subscribe|follow|©|advertisement)", p_, re.I)]
    txt = "\n".join(paras)
    return txt if len(txt) > 400 else None


def _board_players():
    names = set()
    for f in ("_raw_prop_lines.parquet", "_raw_td_board.parquet"):
        p = os.path.join(DATA, f)
        if os.path.exists(p):
            try: names |= set(pd.read_parquet(p, columns=["name"])["name"].dropna().unique().tolist())
            except Exception: pass
    return names


SPEECH = re.compile(r"\b(said|says|told|noted|added|explained|asked about|was asked|on whether|when asked|described|called|expects?|wants?|acknowledged|confirmed|announced|ruled)\b", re.I)


def mentions(text, players, coach=None, transcript=False):
    """Sentences naming a board player. A full-name match anywhere; a last-name match only when the
    full name appears somewhere in the article (journalists name in full first, then by surname).
    Each hit is tagged quote=True when the coach is speaking (a transcript, or the sentence / its
    neighbour carries the coach's surname with a speech verb)."""
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", text) if x.strip()]
    low = text.lower()
    coach_last = coach.split()[-1].lower() if coach else None
    out = []
    for i, s_ in enumerate(sents):
        sl = s_.lower()
        for n in players:
            full = n.lower(); last = n.split()[-1].lower()
            hit = full in sl or (len(last) > 3 and full in low and re.search(r"\b" + re.escape(last) + r"\b", sl))
            if not hit: continue
            prev = sents[i - 1] if i else ""; nxt = sents[i + 1] if i + 1 < len(sents) else ""
            ctx = f"{prev} {s_} {nxt}".lower()
            quote = bool(transcript or (coach_last and coach_last in ctx and SPEECH.search(ctx)))
            out.append((n, s_, nxt, quote))
    return out


def refresh(season=2026, days=7, dry=False, verbose=True, max_coaches=None):
    os.makedirs(CF, exist_ok=True)
    cache = json.load(open(RESOLVED)) if os.path.exists(RESOLVED) else {}
    idx_p = os.path.join(CF, "index.json"); index = json.load(open(idx_p)) if os.path.exists(idx_p) else {}
    players = _board_players()
    cs = coaches(season)
    new_items = 0; new_docs = 0; n_rss = 0
    for team, coach in list(cs.items())[:max_coaches]:
        try:
            items = _rss(f'"{coach}" (press conference OR transcript OR said OR says) when:{days}d')
        except Exception as e:
            if verbose: print(f"  {team} {coach}: rss failed {e}")
            continue
        n_rss += len(items)
        for it in items[:MAX_LINKS_PER_COACH]:
            link = it["link"]
            if link in index:
                continue
            url = resolve(link, cache)
            time.sleep(0.4)
            entry = {"team": team, "coach": coach, "title": it["title"], "published": it["published"], "outlet": it["source"], "url": url, "chars": 0, "mentions": 0}
            if url:
                txt = article_text(url)
                if txt:
                    h = hashlib.md5(url.encode()).hexdigest()[:10]
                    fn = os.path.join(CF, f"{team}_{h}.txt"); open(fn, "w").write(f"{it['title']}\n{it['published']}\n{url}\n\n{txt}")
                    entry["file"] = os.path.basename(fn); entry["chars"] = len(txt)
                    tr = bool(re.search(r"transcript|press conference|media availability", it["title"], re.I))
                    entry["transcript"] = tr
                    ms = mentions(txt, players, coach=coach, transcript=tr); entry["mentions"] = len(ms); entry["quotes"] = sum(1 for m_ in ms if m_[3])
                    new_items += len(ms); new_docs += 1
            index[link] = entry
    if not dry:
        json.dump(cache, open(RESOLVED, "w")); json.dump(index, open(idx_p, "w"), indent=1)
    if verbose:
        print(f"coach feed: {len(cs)} coaches · {n_rss} rss items · {new_docs} new articles read · {new_items} new player mentions · {len(index)} articles indexed")
    return index


def items(players=None, days=10):
    """Intel items from every stored article: one per (player, sentence)."""
    players = players or _board_players()
    idx_p = os.path.join(CF, "index.json")
    if not os.path.exists(idx_p): return {}
    index = json.load(open(idx_p))
    out = {}
    cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days))
    for link, e in index.items():
        if not e.get("file"): continue
        try:
            from email.utils import parsedate_to_datetime
            d = parsedate_to_datetime(e["published"])
        except Exception:
            d = None
        if d and d < cutoff: continue
        p = os.path.join(CF, e["file"])
        if not os.path.exists(p): continue
        txt = open(p).read().split("\n\n", 1)[-1]
        tr = bool(e.get("transcript"))
        for name, sent, nxt, quote in mentions(txt, players, coach=e.get("coach"), transcript=tr):
            flag = any(re.search(cs, sent.lower()) for cs in COACH_SPEAK)
            if not quote and not flag:
                continue                                   # a passing mention with no coach-speak: noise
            src = (f"Coach {e['coach']} · {e['outlet'] or 'press conference'}" if quote else f"{e['outlet'] or 'news'} (re: {e['coach']})")
            out.setdefault(norm_name(name), {"name": name, "items": []})["items"].append({
                "date": d.strftime("%Y-%m-%d") if d else None, "source": src, "title": sent[:300], "snippet": nxt[:300],
                "url": e.get("url"), "feed": "coach", "quote": quote, "coach_speak": flag})
    return out


if __name__ == "__main__":
    dry = "--dry" in sys.argv
    refresh(dry=dry)
    it = items()
    n = sum(len(v["items"]) for v in it.values())
    print(f"{len(it)} board players mentioned by a coach this week, {n} sentences")
    for k, v in list(it.items())[:6]:
        for x in v["items"][:2]:
            print(f"  {v['name']:22s} {x['date']} | {x['source'][:44]:44s} | {x['title'][:120]}")
