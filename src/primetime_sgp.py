"""Correlated primetime SGP — one Wavey-style same-game parlay for each primetime game
(TNF / SNF / MNF), 8-10 legs, built from our validated correlation findings.

This is the OPPOSITE product from the cross-slate near-lock parlay (whale_missile):
  - whale_missile: legs across DIFFERENT games, INDEPENDENT -> a lot of them hit (safe-ish).
  - primetime_sgp: legs in the SAME game, POSITIVELY CORRELATED -> when it hits it hits as a
    group, so the true joint probability beats the naive product and can beat the book's SGP
    haircut. A high-payout correlated lotto with real, individually-reasoned legs.

The correlation spine (from data/correlation_findings.json):
  QB + own top WR      corr +0.164  -> STACK (the core)
  opposing QBs         corr +0.054  -> light shootout add on the other side
  two WRs same team    ~0           -> never "stacked as correlated" (but ok as extra overs)
  RB rush vs own QB    -0.04        -> AVOID rush legs that fight the pass script
So the theme is a SHOOTOUT: everything ties to more passing and more points. Built at ONE
book (SGPs are single-book) using that book's real lines/prices.
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prop_board as pb
import td_board as tb
import correlation_lab as cl
from names import norm_name


def implied(o):
    return 100 / (o + 100) if o > 0 else abs(o) / (abs(o) + 100)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PRIMETIME_HOURS_UTC = set(range(0, 5))    # ~7-11pm ET kickoffs land 00:00-05:00 UTC
SGP_BOOK = "draftkings"                     # SGPs are placed at one book; DK = our baseline


def american_to_dec(a):
    return 1 + (a / 100 if a > 0 else 100 / abs(a))


def dec_to_american(d):
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def find_primetime_events(key, days=14):
    """The primetime games in the window (TNF/SNF/MNF): a night kickoff (00:00-05:00 UTC),
    one game per calendar day (the standalone primetime game is the latest that day)."""
    import datetime
    events = pb.fetch_events(key, days=days)
    by_day = {}
    for e in events:
        t = datetime.datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        by_day.setdefault(t.date(), []).append((t, e))
    out = []
    for day in sorted(by_day):
        lst = sorted(by_day[day], key=lambda x: x[0])
        t_last, e_last = lst[-1]
        # a standalone primetime game: night kickoff AND the only/last game that day
        if t_last.hour in PRIMETIME_HOURS_UTC and (len(lst) == 1 or lst[-1][0] > lst[-2][0]):
            out.append(e_last)
    return out


def _best_over(rows):
    """From a set of book rows for one (player, market), the softest OVER: lowest line,
    then best (highest) over price. Returns (line, price) or None."""
    rows = [r for r in rows if r.get("over_price") is not None and r.get("line") is not None]
    if not rows:
        return None
    rows.sort(key=lambda r: (r["line"], -r["over_price"]))    # lowest line, best price
    return rows[0]["line"], int(rows[0]["over_price"])


def build_sgp_for_game(game_props, td_rows, proj, game_label, min_legs=8, max_legs=10):
    """Construct one correlated shootout SGP from a single game's props (one book)."""
    df = game_props[game_props["book"] == SGP_BOOK]
    if not len(df):
        df = game_props  # some books missing; fall back to whatever we have
    proj = proj.copy(); proj["_k"] = proj["name"].map(norm_name)
    pj = proj.set_index("_k")

    def role(nm):
        k = norm_name(nm)
        if k in pj.index:
            r = pj.loc[k]; r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
            return r.get("position"), r.get("team"), float(r.get("rec_yds", 0) or 0)
        return None, None, 0.0

    # identify QBs (have a pass_yds prop) and receivers (rec_yds prop), by team
    legs, used = [], set()
    def add(nm, mkt, line, price, why):
        key = (nm, mkt)
        if key in used or line is None or price is None:
            return False
        pos, tm, _ = role(nm)
        legs.append({"leg": f"{nm} over {line:g} {_MLBL.get(mkt, mkt)}", "player": nm, "mkt": mkt,
                     "line": float(line), "price": int(price), "why": why, "team": tm, "pos": pos})
        used.add(key); return True

    passers = sorted(df[df.mkt == "pass_yds"]["name"].unique(),
                     key=lambda n: _best_over(df[(df.name == n) & (df.mkt == "pass_yds")].to_dict("records"))[0]
                     if _best_over(df[(df.name == n) & (df.mkt == "pass_yds")].to_dict("records")) else 0,
                     reverse=True)
    # rank receivers within each team's passer by projection/line
    def team_receivers(team):
        recs = []
        for nm in df[df.mkt == "rec_yds"]["name"].unique():
            pos, tm, ry = role(nm)
            bo = _best_over(df[(df.name == nm) & (df.mkt == "rec_yds")].to_dict("records"))
            if bo:
                recs.append((nm, ry or bo[0], tm))
        recs.sort(key=lambda x: x[1], reverse=True)
        return recs

    # SPINE: each side's QB pass over + that QB's top-2 receivers rec over (the +0.16 stack)
    for qb in passers[:2]:
        _, qteam, _ = role(qb)
        bo = _best_over(df[(df.name == qb) & (df.mkt == "pass_yds")].to_dict("records"))
        if bo:
            add(qb, "pass_yds", bo[0], bo[1], "QB spine — a passing/shootout game lifts every over here")
        # this QB's receivers (same team preferred; fall back to any if team unknown)
        recs = [r for r in team_receivers(qteam) if r[2] == qteam] or team_receivers(qteam)
        for nm, _, _ in recs[:2]:
            bo = _best_over(df[(df.name == nm) & (df.mkt == "rec_yds")].to_dict("records"))
            if bo and add(nm, "rec_yds", bo[0], bo[1], "stacks with his QB (corr +0.16) — they hit together"):
                # add his receptions over too (same player, tightly correlated)
                bor = _best_over(df[(df.name == nm) & (df.mkt == "receptions")].to_dict("records"))
                if bor and len(legs) < max_legs:
                    add(nm, "receptions", bor[0], bor[1], "same player as his yards leg — moves with it")

    # anytime-TD of the top pass-catchers (correlated with a high-scoring game)
    if td_rows is not None and len(td_rows):
        for _, r in td_rows.sort_values("p_model", ascending=False).head(3).iterrows():
            if len(legs) >= max_legs:
                break
            add(r["name"], "anytime_td", 0.5, int(r["price"]),
                "anytime TD — ties to the same shootout script as the passing legs")

    legs = legs[:max_legs]
    if len(legs) < min_legs:
        return None
    dec = float(np.prod([american_to_dec(l["price"]) for l in legs]))

    # --- true joint probability + FAIR odds (the benchmark to beat DK's SGP quote) ----------
    def relation(a, b):
        if a["player"] == b["player"]:
            return "same_player"
        same = a["team"] is not None and a["team"] == b["team"]
        aqb, bqb = a["mkt"] == "pass_yds", b["mkt"] == "pass_yds"
        arec = a["mkt"] in ("rec_yds", "receptions") or (a["mkt"] == "anytime_td" and a["pos"] in ("WR", "TE", "RB"))
        brec = b["mkt"] in ("rec_yds", "receptions") or (b["mkt"] == "anytime_td" and b["pos"] in ("WR", "TE", "RB"))
        arush, brush = a["mkt"] == "rush_yds", b["mkt"] == "rush_yds"
        if same:
            if (aqb and brec) or (bqb and arec): return "qb_wr_same"
            if arec and brec: return "wr_wr_same"
            if (aqb and brush) or (bqb and arush): return "qb_rb_same"
            return "wr_wr_same"
        if aqb and bqb: return "opp_qbs"
        if (aqb and brec) or (bqb and arec): return "qb_opp_wr"
        return "unrelated"

    fl = [{"p": implied(l["price"]), "price": l["price"],
           "rel": {j: relation(l, legs[j]) for j in range(i)}} for i, l in enumerate(legs)]
    fair = cl.fair_sgp(fl)

    return {"game": game_label, "book": SGP_BOOK, "theme": "shootout stack (positively correlated)",
            "legs": legs, "n_legs": len(legs), "combined_american": dec_to_american(dec),
            "combined_decimal": round(dec, 1),
            "fair_american": fair["fair_american"], "correlation_lift": fair["correlation_lift"],
            "true_joint_prob": fair["true_joint_prob"], "naive_prob": fair["naive_prob"],
            "note": (f"{len(legs)}-leg same-game shootout stack. The legs are positively correlated "
                     f"(QB + his receivers + TDs rise together), so the true chance of it hitting is "
                     f"{fair['correlation_lift']}x the naive parlay math. Our FAIR price is "
                     f"{fair['fair_american']:+d} — bet it at {SGP_BOOK.title()} only if their SGP quote "
                     f"is LONGER than that (that's when the book under-prices the correlation). Size it small.")}


_MLBL = {"rec_yds": "rec yds", "rush_yds": "rush yds", "receptions": "catches",
         "pass_yds": "pass yds", "rush_att": "rush att", "anytime_td": "anytime TD"}


def build_all(key=None, days=14, projections=None, max_games=3):
    """Build one correlated SGP for each primetime game (up to 3)."""
    key = key or os.environ.get("ODDS_API_KEY")
    proj = projections if projections is not None else pd.read_parquet(f"{DATA}/projection_2026.parquet")
    events = find_primetime_events(key, days=days)[:max_games]
    if not events:
        return []
    out = []
    for e in events:
        gl = f"{e['away_team']} @ {e['home_team']}"
        props, _ = pb.fetch_props(key, [e["id"]])
        if not len(props):
            continue
        td_rows = None
        try:
            tdb, _ = tb.build_td_board(key=key, days=days)
            td_rows = tdb[tdb.game == gl] if len(tdb) else None
        except Exception:
            pass
        sgp = build_sgp_for_game(props, td_rows, proj, gl)
        if sgp:
            out.append(sgp)
    return out


if __name__ == "__main__":
    sgps = build_all(days=14)
    print(f"built {len(sgps)} primetime SGPs")
    for s in sgps:
        print(f"\n=== 🏟️  {s['game']} — {s['n_legs']} legs → {s['combined_american']:+d} "
              f"(${100}→${round(100*s['combined_decimal']):,}) ===")
        for l in s["legs"]:
            print(f"   {l['leg']:38s} {l['price']:+5d}  · {l['why']}")
