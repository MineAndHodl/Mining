"""Player intel — what the beat writers and pundits say, woven into every prop pick.

THE PRINCIPLE (Alex): the model generalizes on purpose — position priors, league-wide
coefficients, role tiers — and that gives a BASE RATE. But a prop is a bet on one player's
Sunday, and what decides it is often not in any table: he's more hurt than the report says,
the coach just said he's featuring him, the OC's plan this week, a contract year. Beat
writers are where that lives. So: THE MODEL IS THE PRIOR, THE PLAYER IS THE BET, AND BEAT
INTEL IS THE UPDATE.

What this layer does with an intel file (data/player_intel_<season>_w<week>.json):
  - HARD GATES (facts, not opinions): a credible "ruled out / will not play / placed on IR"
    within the week window is treated like an official Out; "pitch count / snap limit / eased
    back" cuts trust like a Doubtful.
  - SOFT SIGNALS (-1..+1 each, from the items' text): health nuance beyond the report,
    coach intent (featured / expanded role vs committee / ease in), role news (depth-chart
    change), situational (contract year, return, revenge). These drive the "Beat Intel"
    scorecard lever and the "Beat note" in the paragraph.
  - The projection is NOT moved by soft intel yet. Their weight is EARNED the only honest way:
    every signal is logged with the pick, the forward-grading loop grades the pick, and the
    signals that predict winners get a weight. Until then they inform the reader and the
    trust, not the number. (Hard gates move the board now because they are facts.)

How the file is produced: AUTOMATICALLY every run by intel_feed.py (ESPN/RotoWire dated player
notes with the beat source they cite, RSS from ESPN/Yahoo/CBS/PFT/Google News, Sleeper injury
status + practice + trending adds/drops), merged with any in-session human read (which wins).
Every item carries source, date and url so nothing is unattributed.
"""
import os, sys, json, re, datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from names import norm_name

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")

# --- classification rules: phrase -> (signal, value). Conservative; a human read can override.
HARD_OUT = [r"ruled out", r"will not play", r"placed on (injured reserve|ir)\b", r"out for (the )?(week|game|season)",
            r"won't play", r"has been ruled out", r"downgraded to out"]
HARD_LIMIT = [r"pitch count", r"on a snap count", r"snap limit", r"limited to \d+ snaps", r"eas(e|ed|ing) (him )?(back )?in", r"limited role", r"on a (snap )?count"]
HEALTH_NEG = [r"limited (in|at) practice", r"did not practice", r"dnp\b", r"still (bothered|dealing|nursing)", r"setback",
              r"re-?aggravat", r"not 100", r"playing through", r"managing (a|the)", r"aggravat"]
HEALTH_POS = [r"full (participant|practice)", r"no (injury )?designation", r"removed from (the )?injury report", r"fully healthy",
              r"100 ?%", r"no restrictions", r"good to go", r"cleared"]
COACH_POS = [r"featured", r"feature (him|back)", r"loves (him|what)", r"expanded role", r"bigger role", r"more (touches|carries|targets|work)",
             r"bell ?cow", r"workhorse", r"lead back", r"go-?to", r"focal point", r"emphasi[sz]", r"get him the ball", r"heavy dose",
             r"a lot of (him|work)", r"increased (role|usage|workload)", r"starting", r"named the starter", r"wr1\b", r"no\.? ?1 (receiver|option)"]
COACH_NEG = [r"committee", r"split (carries|reps|time)", r"timeshare", r"rotation", r"share (the )?(load|backfield)", r"reduced role",
             r"fewer (touches|targets|snaps)", r"backup", r"demoted", r"behind ", r"third-?down back", r"change-?of-?pace", r"decoy"]
ROLE_POS = [r"promoted", r"moved up", r"first-?team", r"starter", r"depth chart .* (1|first)", r"took first-team reps"]
ROLE_NEG = [r"second-?team", r"lost (his|the) (job|starting)", r"benched", r"depth chart .* (2|3|second|third)"]
SITUATIONAL = [(r"contract year", 0.2), (r"revenge", 0.15), (r"return(s|ing) (from|to)", 0.1), (r"suspension", -0.3), (r"holdout", -0.4),
               (r"trade (request|rumor)", -0.2),
               # the crowd (Sleeper trending adds/drops, auto-fed): role news leaks here first — small, provisional
               (r"sleeper trending add", 0.2), (r"sleeper trending drop", -0.25)]

WEIGHTS = {"health": 0.35, "coach": 0.35, "role": 0.2, "situational": 0.1}


def _hits(text, pats):
    t = text.lower()
    return [p for p in pats if re.search(p, t)]


PRESEASON = re.compile(r"preseason|exhibition|training camp|joint practice", re.I)


def _about_him(text, name):
    """Sentences of `text` that are about THIS player: contain his last name (RotoWire notes open
    with it: 'Allgeier could open...'), or the whole text when no name is known."""
    if not name:
        return [text]
    last = name.split()[-1].lower()
    first = name.split()[0].lower()
    sents = re.split(r"(?<=[.!?])\s+", text)
    mine = []
    for x in sents:
        if last not in x.lower() and first not in x.lower():
            continue
        # clause level: "Samuel has a key role with Ricky Pearsall (knee) out for the season" —
        # the clause carrying a hard phrase must itself name him
        clauses = re.split(r",|;|\bwith\b|\bwhile\b|\bas\b|\bafter\b|\bbecause\b", x)
        mine.extend([c for c in clauses if last in c.lower() or first in c.lower()])
    return mine


def classify_items(items, week_start, week_end, name=None):
    """items: [{date 'YYYY-MM-DD', source, title, snippet, url}]. Returns the structured intel.
    HARD gates fire only from a sentence about THIS player (his name in it) that is not about the
    preseason — "Allen will not play in Thursday's preseason finale" is rest, and "Schefter
    reports Love won't play" inside Allgeier's note is about Love."""
    in_window = []
    for it in items:
        try:
            d = datetime.date.fromisoformat(str(it.get("date", ""))[:10])
        except Exception:
            continue
        if week_start <= d <= week_end:
            in_window.append(it)
    sig = {"health": 0.0, "coach": 0.0, "role": 0.0, "situational": 0.0}
    hard = None; evidence = []
    for it in in_window:
        text = f"{it.get('title','')} {it.get('snippet','')}"
        src = it.get("source", "?"); date = str(it.get("date", ""))[:10]
        own = [x for x in _about_him(text, name) if not PRESEASON.search(x)]
        own_text = " ".join(own)
        if own_text and _hits(own_text, HARD_OUT):
            hard = "OUT"; evidence.append(f"[{src} {date}] {it.get('title','')}")
        elif own_text and _hits(own_text, HARD_LIMIT) and hard != "OUT":
            hard = "LIMITED"; evidence.append(f"[{src} {date}] {it.get('title','')}")
        if PRESEASON.search(text) and not own_text:
            continue                      # a preseason note with nothing about him: no soft signal either
        for pats, key, val in ((HEALTH_NEG, "health", -0.5), (HEALTH_POS, "health", 0.4), (COACH_POS, "coach", 0.5),
                               (COACH_NEG, "coach", -0.5), (ROLE_POS, "role", 0.5), (ROLE_NEG, "role", -0.6)):
            h = _hits(text, pats)
            if h:
                sig[key] += val; evidence.append(f"[{src} {date}] {it.get('title','')}")
        for pat, val in SITUATIONAL:
            if re.search(pat, text.lower()):
                sig["situational"] += val
    sig = {k: float(np.clip(v, -1, 1)) for k, v in sig.items()}
    composite = sum(WEIGHTS[k] * sig[k] for k in sig)
    # de-dupe evidence, keep order
    seen = set(); ev = [e for e in evidence if not (e in seen or seen.add(e))]
    return {"n_items": len(in_window), "hard": hard, "signals": sig, "composite": round(float(composite), 3), "evidence": ev[:6]}


class PlayerIntel:
    def __init__(self, season=2026, week=1, path=None):
        self.season, self.week = season, week
        path = path or f"{DATA}/player_intel_{season}_w{week}.json"
        self.raw = json.load(open(path)) if os.path.exists(path) else {}
        self.window = (datetime.date.fromisoformat(self.raw.get("_window", {}).get("start", "2000-01-01")),
                       datetime.date.fromisoformat(self.raw.get("_window", {}).get("end", "2100-01-01")))
        self.by_key = {}
        for name, rec in self.raw.items():
            if name.startswith("_"): continue
            # a rec may carry a human 'read' (override) plus raw items
            cls = classify_items(rec.get("items", []), *self.window, name=name)
            if rec.get("read"):                       # human-quality read overrides rule hits
                rd = rec["read"]
                for k in cls["signals"]:
                    if k in rd: cls["signals"][k] = float(np.clip(rd[k], -1, 1))
                if rd.get("hard"): cls["hard"] = rd["hard"]
                cls["composite"] = round(float(sum(WEIGHTS[k] * cls["signals"][k] for k in cls["signals"])), 3)
                if rd.get("note"): cls["note"] = rd["note"]
            self.by_key[norm_name(name)] = cls

    def get(self, name):
        return self.by_key.get(norm_name(name))

    def lean(self, name, side):
        """Beat Intel scorecard lever, flipped for the side (coach/role/health raise volume -> support over).
        Health hard-negatives hurt either side. None if no intel."""
        c = self.get(name)
        if not c or c["n_items"] == 0 and not c.get("note"): return None
        v = c["composite"]
        if c["hard"] in ("OUT", "LIMITED"): return -1.0 if c["hard"] == "OUT" else -0.6
        return float(np.clip(v if side in ("Over", "Yes") else -v, -1, 1))

    def note(self, name):
        c = self.get(name)
        if not c: return None
        if c.get("note"): return c["note"]
        if not c["evidence"]: return None
        return "; ".join(c["evidence"][:2])


if __name__ == "__main__":
    # demo classification on synthetic items inside the window
    demo = [{"date": "2026-09-03", "source": "patriots.com", "title": "Vrabel says Stevenson will be featured early", "snippet": "expects a heavy dose of the run game"},
            {"date": "2025-12-20", "source": "espn", "title": "Stevenson ruled out with toe injury", "snippet": ""}]
    print(classify_items(demo, datetime.date(2026, 9, 1), datetime.date(2026, 9, 8)))
