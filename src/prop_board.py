"""The Prop Board — projection vs posted line vs price vs vig vs edge, all markets.

For every posted player prop we show what actually makes it a bet:
  line        the market number
  desk_proj   our calibrated projection (mean)
  side/model_p our probability of the side we like
  price       the odds you get
  vig         the juice on that market (both-sides implied minus 1)
  fair        the vig-free market probability
  edge        model_p - fair  (true edge, after removing the juice)
  ev          expected value per $1 at the posted price
Plus a data-rooted reason. Markets: rec yds, receptions, rush yds, pass yds, anytime TD.

Confidence gates what becomes a PICK: stable-tier projections (calibrated, Brier .051)
are pickable; movers/rookies are shown as lower-confidence context.
"""
import os, sys, json, urllib.request
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import props_model as pm

API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
MARKET_MAP = {"player_reception_yds": "rec_yds", "player_receptions": "receptions",
              "player_rush_yds": "rush_yds", "player_pass_yds": "pass_yds"}
BOOKS = "draftkings,fanduel,betmgm"
EDGE_MIN = 0.04


def implied(o):
    return 100/(o+100) if o > 0 else abs(o)/(abs(o)+100)

def american_profit(price):
    return price/100 if price > 0 else 100/abs(price)


def fetch_events(key, days=9):
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    to = (now + datetime.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    frm = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    url = f"{API}/events?apiKey={key}&commenceTimeFrom={frm}&commenceTimeTo={to}"
    return json.load(urllib.request.urlopen(url, timeout=30))


def fetch_props(key, event_ids, books=BOOKS):
    mkts = ",".join(list(MARKET_MAP) + ["player_anytime_td"])
    rows, rem = [], None
    for eid in event_ids:
        url = (f"{API}/events/{eid}/odds?apiKey={key}&regions=us&markets={mkts}"
               f"&oddsFormat=american&bookmakers={books}")
        try:
            r = urllib.request.urlopen(url, timeout=30)
        except Exception:
            continue
        rem = r.headers.get("x-requests-remaining")
        d = json.load(r)
        gid = f"{d.get('away_team')} @ {d.get('home_team')}"
        for bk in d.get("bookmakers", []):
            for m in bk.get("markets", []):
                mk = m["key"]
                if mk == "player_anytime_td":
                    for o in m["outcomes"]:
                        rows.append({"game": gid, "book": bk["key"], "mkt": "anytime_td",
                                     "name": o.get("description") or o.get("name"),
                                     "side": "Yes", "line": 0.5, "price": o["price"]})
                elif mk in MARKET_MAP:
                    # collapse over/under into one row per (player, line)
                    acc = {}
                    for o in m["outcomes"]:
                        key2 = (o.get("description"), o.get("point"))
                        acc.setdefault(key2, {})[o["name"]] = o["price"]
                    for (player, pt), pr in acc.items():
                        if "Over" in pr and "Under" in pr and pt is not None:
                            rows.append({"game": gid, "book": bk["key"], "mkt": MARKET_MAP[mk],
                                         "name": player, "line": float(pt),
                                         "over_price": pr["Over"], "under_price": pr["Under"]})
    return pd.DataFrame(rows), rem


def _norm(n):
    return str(n).lower().replace(".", "").replace("'", "").replace(" jr", "").replace(" iii", "").strip()


def price_ou(mkt, mu, line, over_price, under_price):
    """Model P(over) at OUR projection mu vs the de-vigged market. Returns dict."""
    p_over = pm.p_over(mkt, mu, line)
    io, iu = implied(over_price), implied(under_price)
    vig = io + iu - 1
    fair_o = io / (io + iu)
    e_over = p_over - fair_o
    e_under = (1 - p_over) - (1 - fair_o)
    if e_over >= e_under:
        side, edge, price, p = "Over", e_over, over_price, p_over
    else:
        side, edge, price, p = "Under", e_under, under_price, 1 - p_over
    ev = p * american_profit(price) - (1 - p)
    return {"side": side, "model_p": round(p, 3), "edge": round(edge, 3),
            "price": price, "vig": round(vig, 3), "fair": round(fair_o if side == "Over" else 1 - fair_o, 3),
            "ev": round(ev, 3)}


def build_prop_board(key=None, projections=None, td_rates=None, max_games=None):
    """Returns the full priced prop board joined to our projections, best book per prop."""
    key = key or os.environ.get("ODDS_API_KEY")
    if projections is None:
        raise ValueError("pass projections (from props_projection.project)")
    events = fetch_events(key)
    ids = [e["id"] for e in events][:max_games] if max_games else [e["id"] for e in events]
    board, rem = fetch_props(key, ids)
    if not len(board):
        return pd.DataFrame(), rem
    proj = projections.copy()
    proj["_k"] = proj["name"].map(_norm)
    pj = proj.set_index("_k")
    out = []
    for _, b in board.iterrows():
        k = _norm(b["name"])
        pr = pj.loc[k] if k in pj.index else None
        if pr is not None and isinstance(pr, pd.DataFrame):
            pr = pr.iloc[0]
        if b["mkt"] == "anytime_td":
            continue  # TD handled by td_model elsewhere; kept out of the yards board
        mu = float(pr[b["mkt"]]) if (pr is not None and b["mkt"] in pr and pd.notna(pr[b["mkt"]])) else None
        if mu is None or mu <= 0:
            continue
        p = price_ou(b["mkt"], mu, b["line"], b["over_price"], b["under_price"])
        conf = float(pr["confidence"]) if pr is not None and "confidence" in pr else 0.2
        out.append({"name": b["name"], "game": b["game"], "book": b["book"], "mkt": b["mkt"],
                    "line": b["line"], "desk_proj": round(mu, 1),
                    "position": pr["position"] if pr is not None else "?",
                    "team": pr["team"] if pr is not None else "?",
                    "moved": bool(pr["moved"]) if pr is not None and "moved" in pr else False,
                    "rookie": bool(pr["rookie"]) if pr is not None and "rookie" in pr else False,
                    "confidence": conf, **p})
    df = pd.DataFrame(out)
    if not len(df):
        return df, rem
    # best book per (player, market, side): max edge
    df = df.sort_values("edge", ascending=False).drop_duplicates(["name", "mkt", "side"])
    return df, rem


# trust band: how far our projection may sit from the line before we assume WE'RE stale
# (the market repriced a role) rather than that we've found an edge.
TRUST_LO, TRUST_HI = 0.62, 1.6

def classify(row):
    """PICK (calibrated disagreement), REVIEW (disagree too hard — market likely knows a
    role change), or PASS (edge too small)."""
    ratio = row["desk_proj"] / row["line"] if row["line"] else 1.0
    if not (TRUST_LO <= ratio <= TRUST_HI):
        return "REVIEW"
    if row["confidence"] < 0.5:
        return "REVIEW"
    if row["edge"] < EDGE_MIN:
        return "PASS"
    return "PICK"


def reason(row, news=None):
    """Handicapper-voiced rationale that LEADS with the factor doing the most work for this
    specific pick — soft line, our projection, a role/usage note, or a hard-disagree flag.
    Quantitative facts are woven in but it reads like a desk, not a regression table."""
    mkt_lbl = {"rec_yds": "receiving yards", "rush_yds": "rushing yards",
               "receptions": "catches", "pass_yds": "passing yards"}[row["mkt"]]
    name, side, line, proj = row["name"], row["side"], row["line"], row["desk_proj"]
    edge_pts, ev, vig = row["edge"]*100, row["ev"]*100, row["vig"]*100
    status = row.get("status", classify(row))
    price = int(row["price"])

    if status == "REVIEW":
        lead = (f"We've got {name} projected well off the {line} {mkt_lbl} number, which usually "
                f"means the room repriced his role before we did — a committee, a target-share "
                f"squeeze, or a snap change. We're flagging it, not firing it, until the depth "
                f"chart and beat reports confirm the workload.")
        if news:
            lead += f" Latest word: {news}"
        return lead

    # PICK — pick the dominant angle
    diff = proj - line
    if abs(diff) >= 0.15 * max(line, 1):        # our number is the story
        lead = (f"Our model sees {name} at {proj} {mkt_lbl} — a real gap on the {side.lower()} "
                f"of {line}. ")
    else:                                        # the price/number is the story
        lead = (f"This is a number and a price play: {name}'s {line} is a hair soft and we're "
                f"buying the {side.lower()} while it's there. ")
    body = (f"Fair on the {side.lower()} after stripping the {vig:.0f}% juice is "
            f"{int(row['fair']*100)}%; we make it {int(row['model_p']*100)}% — a {edge_pts:+.0f}-point "
            f"edge, {ev:+.0f}% on the ticket at {price:+d}. ")
    trust = ("This is a stable, multi-year role, which is exactly where our projections grade out "
             "calibrated." if row["confidence"] >= 0.67 else
             "Role's in flux, so we're sizing this as a lean, not a hammer.")
    tail = f" {news}" if news else ""
    return lead + body + trust + tail


if __name__ == "__main__":
    import props_projection as pp
    key = os.environ.get("ODDS_API_KEY")
    proj = pd.read_parquet("../data/projection_2026.parquet")
    mg = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    board, rem = build_prop_board(key=key, projections=proj, max_games=mg)
    print(f"credits left: {rem} | priced props: {len(board)}")
    if len(board):
        top = board[board["confidence"] >= 0.67].sort_values("edge", ascending=False).head(12)
        cols = ["name", "mkt", "line", "desk_proj", "side", "model_p", "price", "vig", "edge", "ev", "book"]
        print("\n=== TOP STABLE-TIER PROP EDGES ===")
        print(top[cols].to_string(index=False))
        print("\n=== sample reasons ===")
        for _, r in top.head(3).iterrows():
            print("•", reason(r), "\n")
