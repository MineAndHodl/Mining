"""Edge investigation — adjudicate every large model-vs-line gap individually.

A big edge is NOT automatically wrong. It's either (a) our projection is stale/missing a
role change — bury it — or (b) real alpha the market hasn't caught — keep it. The only way
to tell is to check independent corroborators. We use ADP consensus (the draft market's
role read), the player's data reliability, and role/rookie flags:

  our UNDER + consensus also sees a SMALL role  -> CONFIRMED (market prop line too high = alpha)
  our UNDER + consensus sees a BIG role         -> SUSPECT  (we're missing his usage = our error)
  our OVER  + consensus sees a BIG role         -> CONFIRMED
  our OVER  + consensus sees a SMALL role       -> SUSPECT  (we're over-projecting a nobody)
  thin data (rookie / few games) + no corroboration -> SUSPECT (can't trust our number)
  no contradiction found                        -> KEEP as potential alpha (don't bury)
"""
import numpy as np
import pandas as pd


def _adp_role(adp):
    """ADP → rough role read. Lower ADP = bigger expected role."""
    if adp is None or (isinstance(adp, float) and np.isnan(adp)):
        return "undrafted"          # consensus sees ~no fantasy role
    if adp <= 60:
        return "big"                # every-week starter
    if adp <= 130:
        return "rotational"
    return "deep"                   # late flier / minimal role


def investigate(play, adp_row=None):
    """Return (verdict, note, score_multiplier). verdict in
    CONFIRMED / KEEP / SUSPECT. Only SUSPECT is penalized."""
    edge = abs(float(play.get("edge", 0) or 0))
    side = play.get("side", "")
    conf = float(play.get("confidence", play.get("_conf", 0.5)) or 0.5)
    rookie = bool(play.get("rookie", False))
    # small edges never need investigation
    if edge < 0.10:
        return "KEEP", "moderate edge — no investigation needed", 1.0

    adp = adp_row.get("adp") if adp_row is not None else None
    role = _adp_role(adp)
    big_role = role in ("big", "rotational")
    small_role = role in ("undrafted", "deep")

    # thin data with no consensus support = can't trust our number
    if (rookie or conf < 0.4) and adp is None:
        return "SUSPECT", "big edge but thin data and no draft-market role — likely our number is stale", 0.45

    if side == "Under":
        if small_role:
            return "CONFIRMED", f"we're low AND the draft market sees a small role ({role}) — the prop line looks too high; real under value", 1.08
        if big_role:
            return "SUSPECT", f"we're low but the draft market sees a {role} role — we may be missing his usage", 0.5
    if side == "Over":
        if big_role:
            return "CONFIRMED", f"we're high AND consensus sees a {role} role — corroborated over", 1.08
        if small_role:
            return "SUSPECT", f"we're high but consensus sees a {role} role — we may be over-projecting", 0.5
    # gap exists but nothing contradicts us → keep as potential alpha
    return "KEEP", "big edge with no contradicting signal — treated as potential alpha, flagged for eyes", 1.0


def adjudicate_board(picks, adp):
    """Attach verdict/note to each pick and a score multiplier. adp: DataFrame[name,adp]."""
    adp = adp.copy()
    adp["_k"] = adp["name"].str.lower().str.replace(".", "", regex=False).str.replace("'", "", regex=False).str.strip()
    amap = adp.set_index("_k").to_dict("index")
    out = []
    for p in (picks if isinstance(picks, list) else picks.to_dict("records")):
        k = str(p.get("name", "")).lower().replace(".", "").replace("'", "").strip()
        v, note, mult = investigate(p, amap.get(k))
        out.append({**p, "verdict": v, "verdict_note": note, "score_mult": mult})
    return out
