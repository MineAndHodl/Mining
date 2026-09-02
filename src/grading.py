"""Grades, rails, and the record rule — what a pick MEANS and what we stand behind.

The Desk Score (0-100) is a CONVICTION RANK — edge x hittability x trust — not a win
probability (the "model %" on the card is the win probability). Nobody can read "73". So
every straight bet gets a letter grade, a unit size, and a recommended flag, and only
recommended bets are sent and counted in the record.

Grade (from Desk Score)      units   meaning
  A+   >= 82                 2.0     our strongest conviction: big, trusted edge on a likely play
  A    76-81                 1.5     clear edge, likely play — a core play
  A-   70-75                 1.0     solid edge
  B+   64-69                 0.5     a real but modest edge — the FLOOR for a recommendation
  B    56-63                 0       a lean: we like it, not enough to bet; shown, not counted
  C    < 56                  0       reference only (no edge, negative edge, or a lottery)

Rails (Alex: "I'm fine riding a macro edge... but to start we need some rails"):
  EARLY-SEASON CAP  weeks 0-1 max grade A- (1u); weeks 2-3 max A; week 4+ uncapped. Week-1
                    reads lean on offseason form and the model is still calibrating — the cap
                    lifts as current-season games confirm the reads. It is a rail, not a belief.
  THESIS CAP        at most THESIS_CAP recommended picks sharing one thesis (market + side,
                    e.g. "pass_yds:Under"). A same-thesis stack is correlated risk. A VALIDATED
                    macro thesis (walk-forward-backtested, listed in MACRO_THESES) may carry a
                    higher cap — that is how a macro edge gets ridden, with evidence first.
  GAME CAP          at most GAME_CAP recommended straight bets from one game.
  WEEKLY UNIT CAP   total recommended exposure <= WEEKLY_UNIT_CAP units.
Higher-scored picks claim rail slots first; a pick that loses its slot is shown as a lean
with the rail named, so nothing is hidden.

Record rule: every RECOMMENDED straight bet (grade >= B+ after rails) counts, at its unit
size, from the moment it's published. Leans and reference plays never count. The Whale
Missile and Primetime SGPs are a separate LOTTO LANE with its own record (honest hit math),
never mixed into the straight-bet record.
"""
import re

GRADE_BANDS = [("A+", 82), ("A", 76), ("A-", 70), ("B+", 64), ("B", 56), ("C", -1e9)]
UNITS = {"A+": 2.0, "A": 1.5, "A-": 1.0, "B+": 0.5, "B": 0.0, "C": 0.0}
ORDER = ["C", "B", "B+", "A-", "A", "A+"]
REC_FLOOR = "B+"

THESIS_CAP = 3
GAME_CAP = 2
WEEKLY_UNIT_CAP = 15.0
# validated macro theses -> cap override. Empty until a thesis has a walk-forward backtest.
# e.g. {"pass_yds:Under": 5}
MACRO_THESES = {}

GRADE_MEANING = {
    "A+": "our strongest conviction — a big, trusted edge on a likely play (2u)",
    "A": "a clear edge on a likely play — a core play (1.5u)",
    "A-": "a solid edge (1u)",
    "B+": "a real but modest edge — the floor for a recommendation (0.5u)",
    "B": "a lean: the model likes it, not enough to bet — shown, not counted",
    "C": "reference only — no edge, negative edge, or a lottery",
}


def grade_for(score):
    s = float(score if score is not None else 0)
    for g, lo in GRADE_BANDS:
        if s >= lo:
            return g
    return "C"


def season_cap(weeks_played):
    w = int(weeks_played or 0)
    if w <= 1: return "A-"
    if w <= 3: return "A"
    return None


def cap_grade(grade, cap):
    if cap is None: return grade
    return grade if ORDER.index(grade) <= ORDER.index(cap) else cap


def parse_game_edge(edge_str):
    """Games carry edge as text ('+1.5 pts' / '+4.9% EV'). Convert to a probability-edge so
    games grade on the SAME scale as props: ~1 point of spread value ~= 3% win probability
    (key numbers are already rewarded inside Desk Confidence); an EV% is used directly."""
    if edge_str is None: return 0.0
    if isinstance(edge_str, (int, float)): return float(edge_str)
    m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*pt", str(edge_str))
    if m: return float(m.group(1)) * 0.03
    m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", str(edge_str))
    if m: return float(m.group(1)) / 100.0
    return 0.0


def thesis_of(row):
    if row.get("kind"):                       # game bet
        return f"{row.get('kind')}:{row.get('pick', '')[:3]}"
    return f"{row.get('cat')}:{row.get('side')}"


def apply(rows, weeks_played=0):
    """rows: straight-bet dicts (props + games) with 'score'. Mutates each with grade, units,
    recommended, rail, thesis. Returns (recommended_list, summary)."""
    cap = season_cap(weeks_played)
    for r in rows:
        g = grade_for(r.get("score"))                      # the QUALITY grade — always shown
        gc = cap_grade(g, cap)                             # the SIZING grade under the season cap
        r["grade"] = g; r["grade_sized"] = gc; r["units"] = UNITS[gc]; r["thesis"] = thesis_of(r)
        r["recommended"] = ORDER.index(g) >= ORDER.index(REC_FLOOR)
        r["rail"] = None
        r["cap_note"] = (f"early-season cap: sized as {gc} ({UNITS[gc]:g}u) until current-season games confirm"
                         if cap and gc != g else None)
    # rails claim slots in score order
    per_thesis, per_game, units_used = {}, {}, 0.0
    rec = []
    for r in sorted(rows, key=lambda x: float(x.get("score") or 0), reverse=True):
        if not r["recommended"]:
            continue
        t = r["thesis"]; gm = r.get("game") or r.get("pick", "")
        tcap = MACRO_THESES.get(t, THESIS_CAP)
        if per_thesis.get(t, 0) >= tcap:
            r["recommended"] = False; r["rail"] = f"thesis cap ({t}, {tcap} max)"; r["units"] = 0.0; continue
        if per_game.get(gm, 0) >= GAME_CAP:
            r["recommended"] = False; r["rail"] = f"game cap ({GAME_CAP} per game)"; r["units"] = 0.0; continue
        if units_used + r["units"] > WEEKLY_UNIT_CAP:
            r["recommended"] = False; r["rail"] = f"weekly unit cap ({WEEKLY_UNIT_CAP}u)"; r["units"] = 0.0; continue
        per_thesis[t] = per_thesis.get(t, 0) + 1; per_game[gm] = per_game.get(gm, 0) + 1
        units_used += r["units"]; rec.append(r)
    counts = {}
    for r in rows:
        counts[r["grade"]] = counts.get(r["grade"], 0) + 1
    summary = {"weeks_played": int(weeks_played or 0), "season_cap": cap, "units_recommended": round(units_used, 1),
               "n_recommended": len(rec), "grade_counts": counts,
               "rails": {"thesis_cap": THESIS_CAP, "game_cap": GAME_CAP, "weekly_unit_cap": WEEKLY_UNIT_CAP,
                         "macro_theses": MACRO_THESES}}
    return rec, summary


RECORD_RULE = ("Only RECOMMENDED straight bets (grade B+ or better after rails) count toward the record, at "
               "their unit size, from the moment they're published. Leans (B) and reference plays (C) never "
               "count. The Whale Missile and Primetime SGPs are a separate lotto lane with its own record.")


# --------------------------------------------------------------------------- board pass
def _key(r):
    if r.get("kind"): return ("game", r.get("pick"))
    return ("prop", r.get("name"), r.get("cat"), r.get("line"), r.get("side"))


def grade_board(ranked, weeks_played=0):
    """One pass over the assembled board: put games on the SAME conviction scale as props,
    grade every unique straight bet, apply the rails, propagate to every tab, and attach the
    official recommended slate + the record rule + a plain-language grade key."""
    from rank_board import _desk_score
    # games -> unified scale (their Desk-Confidence score was on its own inflated scale)
    for g in ranked.get("best_games", []) or []:
        e = parse_game_edge(g.get("edge"))
        g["edge_pct"] = round(e, 3); g["model_p"] = round(0.5 + e, 3)
        g["score_legacy"] = g.get("score")
        g["score"] = _desk_score(e, 0.5 + e)
    # unique straight bets across tabs
    uniq = {}
    tabs = [ranked.get("best_props", [])] + list((ranked.get("by_category") or {}).values()) + [ranked.get("best_games", []) or []]
    for lst in tabs:
        for r in lst:
            k = _key(r)
            if k not in uniq: uniq[k] = dict(r)
    rows = list(uniq.values())
    rec, summary = apply(rows, weeks_played)
    graded = {_key(r): r for r in rows}
    for lst in tabs:
        for r in lst:
            g = graded[_key(r)]
            for f in ("grade", "grade_sized", "units", "recommended", "rail", "thesis", "cap_note"):
                r[f] = g.get(f)
            if r.get("kind"):
                for f in ("edge_pct", "model_p", "score", "score_legacy"): r[f] = g.get(f)
    ranked["best_games"] = sorted(ranked.get("best_games", []) or [], key=lambda x: float(x.get("score") or 0), reverse=True)
    # official slate: recommended straight bets, best first, with a one-line "why" and units
    slate = sorted(rec, key=lambda x: float(x.get("score") or 0), reverse=True)
    ranked["recommended"] = [dict(r) for r in slate]          # keep the scorecard: the slate IS the main tab
    ranked["grading"] = summary
    ranked["record_rule"] = RECORD_RULE
    # plain-language key with THIS week's realized averages per grade (so a grade is interpretable)
    key = []
    for g in ["A+", "A", "A-", "B+", "B", "C"]:
        rs = [r for r in rows if r.get("grade") == g and r.get("model_p") is not None]
        avg_p = round(100 * sum(float(r["model_p"]) for r in rs) / len(rs)) if rs else None
        avg_e = round(100 * sum(float(r.get("edge_pct") if r.get("kind") else (r.get("edge") or 0)) for r in rs) / len(rs), 1) if rs else None
        key.append({"grade": g, "units": UNITS[g], "meaning": GRADE_MEANING[g], "n": len(rs),
                    "avg_model_p": avg_p, "avg_edge_pct": avg_e})
    ranked["grade_key"] = key
    return ranked
