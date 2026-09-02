"""Unified ranking — one comparable score across every prop category and every game bet,
so the desk can be viewed three ways: best-in-each-category, best-across-everything, and
best games. A weak category simply won't appear in the cross-board if nothing ranks.

Score design (comparable across markets):
  props: edge and EV, scaled by data confidence (market-trust already inside confidence).
  games: the Desk Confidence (off-market points-of-value + key numbers).
Everything lands on a 0-100 'Desk Score' so props and games each rank cleanly, and the
cross-prop board is an apples-to-apples merge of the per-category scores.
"""
import numpy as np
import pandas as pd

PROP_CATEGORIES = ["rec_yds", "rush_yds", "receptions", "pass_yds", "rush_att", "anytime_td"]
CAT_LABEL = {"rec_yds": "Receiving Yards", "rush_yds": "Rushing Yards", "receptions": "Receptions",
             "pass_yds": "Passing Yards", "rush_att": "Rush Attempts", "anytime_td": "Anytime TD"}


EDGE_PEAK = 0.11        # edge beyond which a bigger gap is a stale-model red flag, not free money
MIN_HITTABLE = 0.45     # a "best bet" must have at least this chance to win — no lottery tickets
CAT_CAP = 3             # max picks from any single market on the best-bets board (diversity)
INTEL_PROV_WEIGHT = 4.0 # provisional Desk-Score points for a full-strength beat-intel lean (to be earned)

def _desk_score(edge, model_p, mult=1.0, anchor=0.52):
    """Unified Desk Score = CONVICTION, not raw edge. Two components:
      value    — the edge, humped so a blowout gap can't top the board, and DISCOUNTED for
                 low-probability plays (a tail edge is noisier and carries favorite-longshot
                 bias, so a +1000 lotto's big % edge counts far less than a coin-flip's).
      hittable — reward likely plays, penalize longshots, measured against `anchor` — the
                 neutral hit-probability for THIS market. Yardage/counts are ~50/50 bets
                 (anchor 0.52); a TD Yes is always a dog to score, so its neutral point is
                 much lower (~0.35). Using one flat anchor let a +1000 TD lotto outrank a +135
                 value TD; a market-appropriate anchor fixes it.
    A best bet is something you'd put real money on: probable AND +value AND a trusted signal."""
    edge = float(edge or 0); p = float(model_p if model_p is not None else 0.5)
    if edge <= EDGE_PEAK:
        val = edge * 260                             # 4%→10, 8%→21, 11%→29
    else:
        val = EDGE_PEAK * 260 - (edge - EDGE_PEAK) * 260   # decay past the peak
    val *= min(1.0, p / 0.40)                          # tail-reliability (harmless for p>=.40)
    hittable = (p - anchor) * 55
    score = (45 + val + hittable) * float(mult or 1.0)
    return round(float(np.clip(score, 20, 95)), 1)


def _trust(row):
    """Robustness of the edge — what separates an A+ from an A- when edges look alike:
      sample depth : projection confidence (games of real history behind the number)
      market breadth: how many books post this line (a one-book line is thinner evidence the
                      number is real; 3+ books agreeing the line exists is firmer footing)
    Range about -7 .. +9 Desk-Score points."""
    conf = row.get("confidence"); nb = row.get("n_books")
    t = 0.0
    if conf is not None and not (isinstance(conf, float) and conf != conf):
        t += (float(conf) - 0.7) * 22.0                  # 1.0 -> +6.6, 0.7 -> 0, 0.46 -> -5, 0.33 -> -8
    if nb is not None and not (isinstance(nb, float) and nb != nb):
        nb = int(nb); t += (-3.0 if nb <= 1 else (0.0 if nb == 2 else 3.0))
    # Beat intel: PROVISIONAL weight. A prior, not an earned weight — capped small so it can
    # reorder picks near the margin (the point of gathering it) without pretending we know its
    # value. The forward-grading loop raises it, lowers it, or kills it. Hard gates (OUT/LIMITED)
    # already act upstream at pricing.
    bi = row.get("intel_lean")
    if bi is not None and not (isinstance(bi, float) and bi != bi):
        t += float(np.clip(bi, -1, 1)) * INTEL_PROV_WEIGHT
    return t


def prop_score(row):
    """Desk Score for a prop pick: conviction (edge + hit-probability) + trust (sample depth,
    market breadth), times the investigate verdict multiplier (SUSPECT cut, CONFIRMED boosted)."""
    base = _desk_score(row.get("edge"), row.get("model_p"), 1.0)
    s = (base + _trust(row)) * float(row.get("score_mult", 1.0) or 1.0)
    return round(float(np.clip(s, 20, 95)), 1)


TD_ANCHOR = 0.35   # a typical anytime-TD Yes is a ~35% dog to score — the neutral hit point here

def td_score(row):
    """TD board on the SAME conviction scale, but anchored to TD reality (a Yes is a dog to
    score, so 0.35 is neutral, not 0.52). A +1000 lotto still can't top it (tail discount +
    hittable penalty), and a juiced -195 favorite whose Yes is NEGATIVE-edge still floors at 20."""
    return _desk_score(row.get("edge"), row.get("p_model"), anchor=TD_ANCHOR)


def _nn(x):
    """NaN -> None so the page gets null, not a NaN literal."""
    return None if (isinstance(x, float) and x != x) else x


def _fnum(x):
    try:
        v = float(x); return None if v != v else v
    except Exception:
        return None


def _td_reason(r, mp, imp, edge):
    """Handicapper paragraph for an anytime-TD pick, spoken from the DECOMPOSED xTD model:
    the role (goal-line / red-zone share), the vulture, the coach's goal-line tendency, last
    season's TDs vs what the role should have produced, and the environment — then the price."""
    name = r.get("name", "He"); pos = r.get("position") or ""
    is_back = pos in ("RB", "FB")
    gl = _fnum(r.get("gl_share")); rzt = _fnum(r.get("rz_tgt_share")); vult = _fnum(r.get("vulture_share"))
    cpr = _fnum(r.get("coach_gl_pass_rate")); tl = _fnum(r.get("tds_last")); xt = _fnum(r.get("xtd"))
    g = _fnum(r.get("games")); lam = _fnum(r.get("lam_xtd"))
    bits = []
    # role
    if is_back and gl is not None:
        if gl >= 0.70: bits.append(f"he owns the goal line — about {int(gl*100)}% of his team's carries inside the 5 last season")
        elif gl >= 0.50: bits.append(f"he's the lead goal-line back (~{int(gl*100)}% of the inside-the-5 carries)")
        else: bits.append(f"his goal-line role is shared (~{int(gl*100)}% of the inside-the-5 carries)")
    elif rzt is not None:
        if rzt >= 0.28: bits.append(f"he's the red-zone target — roughly {int(rzt*100)}% of his team's throws inside the 20 went his way")
        elif rzt >= 0.15: bits.append(f"he earns a real red-zone target share (~{int(rzt*100)}%)")
        else: bits.append(f"his red-zone target share is modest (~{int(rzt*100)}%)")
    # vulture — and, for a back whose goal line is shared, the volume that still carries him
    if is_back and vult is not None and vult >= 0.40:
        bits.append(f"the caveat: ~{int(vult*100)}% of the goal-line work leaks to a backup, which is what caps a back's TD equity")
        zs = [_fnum(r.get(k)) for k in ("IN10_rush_share", "RZ_rush_share")]
        zs = [z for z in zs if z is not None]
        opn = _fnum(r.get("OPEN_rush_opps"))
        if zs and np.mean(zs) >= 0.35:
            vol = f"he still handles ~{int(np.mean(zs)*100)}% of the carries inside the 20"
            if opn and opn >= 6: vol += f" on top of ~{opn:.0f} open-field carries a game"
            bits.append(vol + " — volume creates chances even when the plunge is shared")
    # coach tendency
    if cpr is not None:
        if is_back and cpr <= 0.35: bits.append(f"the coach runs it in at the goal line (only ~{int(cpr*100)}% pass inside the 5) — exactly what a back wants")
        elif is_back and cpr >= 0.50: bits.append(f"one drag: this coach throws at the goal line (~{int(cpr*100)}% pass inside the 5), which siphons plunges")
        elif (not is_back) and cpr >= 0.48: bits.append(f"the coach throws at the goal line (~{int(cpr*100)}% pass inside the 5) — that feeds the pass-catchers")
        elif (not is_back) and cpr <= 0.33: bits.append(f"one drag: this coach runs it in at the goal line (~{int(cpr*100)}% pass), which takes scoring chances away from the receivers")
    # regression vs role
    if tl is not None and xt is not None and g:
        exp = xt * g
        if exp > 0.5:
            diff = (tl - exp) / exp
            if diff >= 0.30: bits.append(f"he scored {int(tl)} last season against a role that should produce about {exp:.0f} — he ran hot, so we lean on the role, not the box score")
            elif diff <= -0.30: bits.append(f"he scored only {int(tl)} last season on a role that should produce about {exp:.0f} — that's a bounce-back profile, not a broken one")
    rn = r.get("role_note")
    if isinstance(rn, str) and rn:
        bits.append(f"injury angle: {rn}")
    zn = r.get("rz_note")
    if isinstance(zn, str) and zn:
        bits.append(f"the opponent is {zn} — a measured but small effect (~5%)")
    body = "; ".join(bits)
    lead = f"{name} to score. "
    if lam is not None:
        lead += f"Our expected-TD number for this game is {lam:.2f}, which makes him ~{mp}% to find the end zone. "
    else:
        lead += f"We make him ~{mp}% to find the end zone. "
    case = (body[0].upper() + body[1:] + ". ") if body else ""
    bk = r.get("book"); nb = _fnum(r.get("n_books")); pr = _fnum(r.get("price"))
    shop = (f" Best price {int(pr):+d} at {str(bk).replace('_us','').title()}" + (f" (shopped across {int(nb)} books)" if nb and nb > 1 else "") + ".") if (pr is not None and bk) else ""
    if edge >= 0.03:
        price = (f"The market has him at {imp}% implied; we're higher, so there's real Yes value here — "
                 f"but anytime-TD is a one-sided, fat-margin market, so it's sized as a small add.{shop}")
    elif mp >= 45:
        price = (f"He's one of the likelier scorers on the slate, but the market prices him even higher ({imp}% implied at the best "
                 f"book) — star touchdowns get overbet, so there's no Yes value; a red-zone reference and an SGP building block, not a bet.")
    else:
        price = f"Against {imp}% implied there's no edge — reference only."
    return lead + case + price


def game_score(desk_conf):
    """Games use Desk Confidence (6.0-9.6) mapped onto the same 0-100 scale."""
    return round((float(desk_conf) - 6.0) / 3.6 * 60 + 40, 1)   # 6.0->40, 9.6->100


def build_ranked(prop_board=None, td_board=None, games=None, top=10, adp=None):
    """Returns the toggleable structure: per-category top N, best-across-props top N, and
    best games top N. Each big edge is investigated individually (investigate.py) using ADP
    consensus — CONFIRMED alpha is boosted, SUSPECT (stale) is cut, the rest kept."""
    out = {"by_category": {}, "best_props": [], "best_games": []}
    rows = []
    if prop_board is not None and len(prop_board):
        pb = prop_board[prop_board.get("status", "PICK").eq("PICK")] if "status" in prop_board else prop_board
        recs = pb.to_dict("records")
        if adp is not None and len(adp):
            import investigate as inv
            recs = inv.adjudicate_board(recs, adp)   # attaches verdict/note/score_mult
        for r in recs:
            rows.append({"cat": r["mkt"], "score": prop_score(r), "name": r["name"],
                         "game": r.get("game"), "line": r.get("line"), "side": r.get("side"),
                         "our_num": r.get("desk_proj"), "market_num": r.get("market_num"), "edge": r.get("edge"), "ev": r.get("ev"),
                         "model_p": r.get("model_p"), "price": r.get("price"), "book": r.get("book"),
                         "n_books": r.get("n_books"), "shop_adv": r.get("shop_adv"),
                         "verdict": r.get("verdict", "KEEP"), "verdict_note": r.get("verdict_note", ""),
                         "score_mult": r.get("score_mult", 1.0), "reason": r.get("reason", ""),
                         "contrib": r.get("contrib"), "confidence": r.get("confidence"),
                         "offseason": r.get("offseason"),
                         "inj_status": _nn(r.get("inj_status")), "inj_detail": _nn(r.get("inj_detail")),
                         "role_mult": _nn(r.get("role_mult")), "role_note": _nn(r.get("role_note")),
                         "inj_lean": _nn(r.get("inj_lean")), "inj_note": _nn(r.get("inj_note")),
                         "intel_lean": _nn(r.get("intel_lean")), "intel_note": _nn(r.get("intel_note")),
                         "intel_signals": r.get("intel_signals"), "intel_hard": _nn(r.get("intel_hard"))})
    if td_board is not None and len(td_board):
        for _, r in td_board[td_board["edge"].notna()].iterrows():
            mp = int((r.get("p_model") or 0) * 100); imp = int((r.get("p_implied") or 0) * 100)
            edge = float(r.get("edge") or 0)
            rsn = _td_reason(r, mp, imp, edge)
            sc = td_score(r)
            flagged = bool(r.get("mkt_flag")) if r.get("mkt_flag") is not None and r.get("mkt_flag") == r.get("mkt_flag") else False
            if flagged:
                sc = round(sc * 0.8, 1)
                rsn += (f" Trust cut: we sit {r.get('mkt_ratio')}x above the whole market on a last-season role read with no "
                        f"current-season games — the market's role read wins until he proves the share (off the TD Stack).")
            rows.append({"cat": "anytime_td", "score": sc, "name": r["name"], "mkt_flag": flagged, "mkt_ratio": _nn(r.get("mkt_ratio")),
                         "game": r.get("game"), "line": 0.5, "side": "Yes",
                         "our_num": r.get("p_model"), "edge": r.get("edge"), "ev": None,
                         "model_p": r.get("p_model"), "price": r.get("price"), "book": _nn(r.get("book")) or "draftkings",
                         "n_books": _nn(r.get("n_books")), "position": _nn(r.get("position")),
                         "reason": rsn})
    allp = pd.DataFrame(rows)
    if len(allp):
        # per-category top N, sorted by Desk Score EVERYWHERE (including anytime-TD). Sorting TD
        # by raw probability was wrong: it floated juiced favorites (Jonathan Taylor -195, our
        # model 47% -> a NEGATIVE-edge bet) to the top of the tab with a Desk Score of 20, above
        # the actual +value plays (Adams +135, Rice +150). Desk Score already balances both
        # failure modes — it sinks +1000 lottery longshots (low hit prob) AND -195 negative-edge
        # favorites (no Yes value) — so it is the honest sort for a picks board, and it makes the
        # "ranked by Desk Score" header true.
        for cat in PROP_CATEGORIES:
            sub = allp[allp.cat == cat]
            sub = sub.sort_values("score", ascending=False, na_position="last").head(top)
            if len(sub):
                out["by_category"][cat] = sub.to_dict("records")
        # BEST BETS = best-across, but only genuinely hittable plays (no lottery longshots).
        # A +1000 anytime-TD is not a "best bet" — it lives in its own tab, not here.
        # DIVERSITY: cap any one market at CAT_CAP so the board is a genuine best-of-the-slate
        # across bet types, not a monotone stack of (say) passing unders. Same-thesis clustering
        # is both a weak product AND correlated risk — if "Week-1 passing unders" is wrong, a
        # stacked board fails together. We still never force a bad pick: everything here cleared
        # MIN_HITTABLE and is ranked by conviction; the cap just spreads the slots.
        hittable = allp[allp["model_p"].fillna(0.5) >= MIN_HITTABLE].sort_values(
            "score", ascending=False)
        picked, per_cat = [], {}
        for r in hittable.to_dict("records"):
            c = r["cat"]
            if per_cat.get(c, 0) >= CAT_CAP:
                continue
            picked.append(r); per_cat[c] = per_cat.get(c, 0) + 1
            if len(picked) >= top:
                break
        # if the cap left us short of `top` (thin slate), backfill with the next-best regardless
        if len(picked) < top:
            have = {(r["name"], r["cat"], r.get("line")) for r in picked}
            for r in hittable.to_dict("records"):
                if (r["name"], r["cat"], r.get("line")) in have:
                    continue
                picked.append(r)
                if len(picked) >= top:
                    break
        out["best_props"] = picked
        # THE POOL: every scored candidate (not the tab-truncated top-10s). Selection — grades,
        # rails, intel gates, the slate — must run over this, or a signal applied downstream
        # can only veto survivors, never surface a pick the ranking missed.
        out["pool"] = allp.sort_values("score", ascending=False).to_dict("records")

    # games (spreads/totals/ML) — already scored dicts with desk_conf
    if games:
        g = [{**x, "score": game_score(x.get("desk_conf", 6.0))} for x in games]
        out["best_games"] = sorted(g, key=lambda x: x["score"], reverse=True)[:top]
    return out


if __name__ == "__main__":
    # smoke test with synthetic
    pb = pd.DataFrame([
        {"mkt": "rec_yds", "status": "PICK", "name": "A", "game": "X@Y", "line": 60, "side": "Over",
         "desk_proj": 68, "edge": 0.09, "ev": 0.12, "confidence": 0.7, "price": -110, "book": "dk", "reason": "..."},
        {"mkt": "rush_yds", "status": "PICK", "name": "B", "game": "X@Y", "line": 55, "side": "Under",
         "desk_proj": 48, "edge": 0.06, "ev": 0.07, "confidence": 0.66, "price": -108, "book": "fd", "reason": "..."},
    ])
    r = build_ranked(prop_board=pb, games=[{"label": "BUF -1", "desk_conf": 8.3, "seg": "side"}])
    print("categories:", list(r["by_category"].keys()))
    print("best_props top:", [(p["name"], p["cat"], p["score"]) for p in r["best_props"]])
    print("best_games:", [(g["label"], g["score"]) for g in r["best_games"]])
