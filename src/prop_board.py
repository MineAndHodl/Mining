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

Confidence gates what becomes a PICK: stable-tier projections are pickable; movers/rookies
are shown as lower-confidence context. (Prop calibration vs real book lines is still being
measured — see the CLV/backtest work; no headline calibration number is claimed here.)
"""
import os, sys, json, urllib.request
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import props_model as pm
import weather as wx

API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
# Only markets we actually PROJECT — no point fetching (and paying credits for) markets the
# projection can't price. pass_att/pass_cmp were credit burn with zero coverage.
MARKET_MAP = {"player_reception_yds": "rec_yds", "player_receptions": "receptions",
              "player_rush_yds": "rush_yds", "player_pass_yds": "pass_yds",
              "player_rush_attempts": "rush_att",
              "player_kicking_points": "kicking_points", "player_field_goals": "field_goals"}
# shop across all major bettable US books, not just three — line-shopping is guaranteed
# better-than-field value, and cross-book dispersion is where the real prop edge lives.
BOOKS = "draftkings,fanduel,betmgm,fanatics,betrivers,williamhill_us,espnbet,hardrockbet"
# Residual-edge world: after removing the double-counted game total / prior-year defense, real
# prop edges are modest. A meaningful matchup+environment lean clears ~2%; we surface those as
# honest LEANS rather than pretending they're 10% locks.
EDGE_MIN = 0.02


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


from names import norm_name as _norm


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
            "ev": round(ev, 3), "model_p_over": round(p_over, 3), "fair_over": round(fair_o, 3),
            "over_price_kept": over_price, "under_price_kept": under_price}


def build_prop_board(key=None, projections=None, td_rates=None, max_games=None, weather_by_game=None, days=14):
    """Returns the full priced prop board joined to our projections, best book per prop."""
    key = key or os.environ.get("ODDS_API_KEY")
    if projections is None:
        raise ValueError("pass projections (from props_projection.project)")
    events = fetch_events(key, days=days)
    ids = [e["id"] for e in events][:max_games] if max_games else [e["id"] for e in events]
    board, rem = fetch_props(key, ids)
    if not len(board):
        return pd.DataFrame(), rem
    proj = projections.copy()
    proj["_k"] = proj["name"].map(_norm)
    pj = proj.set_index("_k")
    # matchup/environment context: schedule (game totals) + defense-vs-position
    import board_factors as bf
    _pmap, sched_rows, code2full, _adp, def_vs = bf._load_context()
    LEAGUE_TEAM_PTS = 22.5
    out = []
    for _, b in board.iterrows():
        k = _norm(b["name"])
        pr = pj.loc[k] if k in pj.index else None
        if pr is not None and isinstance(pr, pd.DataFrame):
            pr = pr.iloc[0]
        if b["mkt"] == "anytime_td":
            continue  # TD handled by td_model elsewhere; kept out of the yards board
        our_mu = float(pr[b["mkt"]]) if (pr is not None and b["mkt"] in pr and pd.notna(pr[b["mkt"]])) else None
        if our_mu is None or our_mu <= 0:
            continue
        pos = pr["position"] if pr is not None else "?"
        team = pr["team"] if pr is not None else None
        # --- MARKET-ANCHORED pricing: baseline = the (sharp) market line; edge comes ONLY from
        # named factors the market may underweight (game environment + opponent defense). No
        # factor -> no edge -> no pick. This kills the skew-artifact edges that floated a low-
        # target TE like Goedert to #1, and makes every edge explainable. ---------------------
        # CRITICAL: the book's prop line ALREADY prices the game total and prior-year defense.
        # So we only tilt by the small UNPRICED RESIDUAL (props are softer than game lines, but
        # not by much) — otherwise we double-count what's already in the number and manufacture a
        # fake systematic edge (the trap that floated Goedert / every-QB-over to the top).
        ENV_RESID, DEF_RESID_RUSH, DEF_RESID_REC = 0.22, 0.30, 0.18
        env = 1.0; opp = 1.0; def_delta = None; implied_pts = None; opp_team = None
        g, opp_team = bf._game_ctx(b["game"], bf.TEAM_FIX.get(team, team), sched_rows, code2full)
        if g:
            implied_pts = g["implied_team_pts"]
            env = float(np.clip(1 + (implied_pts / LEAGUE_TEAM_PTS - 1) * ENV_RESID, 0.96, 1.05))
            if opp_team:
                if b["mkt"] in ("rush_yds", "rush_att") and pos == "RB":
                    d = def_vs.get("RB", {}).get(opp_team)
                    if d is not None: def_delta = d; opp = float(np.clip(1 + d * DEF_RESID_RUSH, 0.94, 1.06))
                elif b["mkt"] in ("rec_yds", "receptions") and pos in ("WR", "TE"):
                    d = def_vs.get("WR", {}).get(opp_team)
                    if d is not None: def_delta = d; opp = float(np.clip(1 + d * DEF_RESID_REC, 0.96, 1.04))
        # weather tilt (gameday) folds into env
        wg = (weather_by_game or {}).get(b["game"])
        if wg:
            env *= wx.market_weather_mult(b["mkt"], wind=wg.get("wind"), temp=wg.get("temp"),
                                          roof=wg.get("roof", "outdoors"))
        side, edge, p_over_model, mu_model = pm.anchored_edge(
            b["mkt"], b["line"], b["over_price"], b["under_price"], env=env, opp=opp)
        model_p = p_over_model if side == "Over" else round(1 - p_over_model, 3)
        # line-comparable "our number": the line nudged by our tilt (env*opp), NOT the raw mean
        # (a right-skewed mean sits far above the line and would look like a huge fake gap).
        our_num = round(b["line"] * env * opp, 1)
        io, iu = implied(b["over_price"]), implied(b["under_price"])
        fair_o = io / (io + iu)
        fair = round(fair_o if side == "Over" else 1 - fair_o, 3)
        price = b["over_price"] if side == "Over" else b["under_price"]
        ev = round(model_p * american_profit(price) - (1 - model_p), 3)
        conf = float(pr["confidence"]) if pr is not None and "confidence" in pr else 0.2
        out.append({"name": b["name"], "game": b["game"], "book": b["book"], "mkt": b["mkt"],
                    "line": b["line"], "desk_proj": our_num, "our_raw_proj": round(our_mu, 1),
                    "position": pos, "team": team if team else "?",
                    "moved": bool(pr["moved"]) if pr is not None and "moved" in pr else False,
                    "rookie": bool(pr["rookie"]) if pr is not None and "rookie" in pr else False,
                    "confidence": conf, "side": side, "edge": round(edge, 3), "model_p": model_p,
                    "fair": fair, "price": price, "ev": ev,
                    "env_tilt": round(env, 3), "opp_tilt": round(opp, 3),
                    "def_delta": (round(def_delta, 3) if def_delta is not None else None),
                    "implied_pts": (round(implied_pts, 1) if implied_pts is not None else None),
                    "opp_team": opp_team,
                    "over_price_kept": b["over_price"], "under_price_kept": b["under_price"]})
    df = pd.DataFrame(out)
    if not len(df):
        return df, rem
    # --- line shopping (at the SAME line): how many books offer our pick, and how much our
    # best price beats the field. The max-edge row is already the best line+price combo, so we
    # keep it and attach the shopping context at that exact line (never trade a better line for
    # a juicier price on a worse one). ------------------------------------------------------
    shop = []
    for (nm, mk, sd, ln), g in df.groupby(["name", "mkt", "side", "line"]):
        profits = [american_profit(float(p)) for p in g["price"]]
        best_i = int(np.argmax(profits))
        adv = max(profits) - float(np.median(profits))       # our best payout vs field median, cents/$
        shop.append({"name": nm, "mkt": mk, "side": sd, "line": ln,
                     "n_books": int(g["book"].nunique()), "shop_adv": round(adv, 3),
                     "best_price": int(g["price"].iloc[best_i]), "book_best": g["book"].iloc[best_i]})
    shopdf = pd.DataFrame(shop)
    df = df.sort_values("edge", ascending=False).drop_duplicates(["name", "mkt", "side"])
    if len(shopdf):
        df = df.merge(shopdf, on=["name", "mkt", "side", "line"], how="left")
        df["price"] = df["best_price"].fillna(df["price"]).astype(int)   # best price AT our line
        df["book"] = df["book_best"].fillna(df["book"])
        df["ev"] = (df["model_p"] * df["price"].map(american_profit) - (1 - df["model_p"])).round(3)
    return df, rem


# trust band: how far our projection may sit from the line before we assume WE'RE stale
# (the market repriced a role) rather than that we've found an edge.
TRUST_LO, TRUST_HI = 0.62, 1.6

FLAT_PRIOR_MARKETS = {"kicking_points", "field_goals"}   # league-prior projections, not per-player

def classify(row):
    """PICK (calibrated disagreement), REVIEW (disagree too hard — market likely knows a
    role change), or PASS (edge too small)."""
    # kicker markets run on a flat league prior (no per-player signal yet) — never a PICK
    if row.get("mkt") in FLAT_PRIOR_MARKETS:
        return "REVIEW"
    ratio = row["desk_proj"] / row["line"] if row["line"] else 1.0
    if not (TRUST_LO <= ratio <= TRUST_HI):
        return "REVIEW"
    if row["confidence"] < 0.45:
        return "REVIEW"
    # low-trust markets (pass_yds) must clear a higher edge bar to become a pick
    need = EDGE_MIN / wx.MARKET_TRUST.get(row.get("mkt", ""), 0.7)
    if row["edge"] < need:
        return "PASS"
    return "PICK"


POS_UNIT = {"rush_yds": "opposing backfields", "rush_att": "opposing backfields",
            "rec_yds": "opposing wide receivers", "receptions": "opposing pass-catchers",
            "pass_yds": "opposing quarterbacks"}


def _defense_clause(row, side):
    """A real matchup read from the opponent's defense-vs-position allowance."""
    d = row.get("def_delta"); opp = row.get("opp_team")
    if d is None or opp is None:
        return None
    unit = POS_UNIT.get(row["mkt"], "this position")
    pct = abs(d) * 100
    if d >= 0.08:
        return (f"{opp} were a leak here last year — they gave up about {pct:.0f}% more than league "
                f"average to {unit}, one of the softer spots on the board, which is the engine of this over")
    if d >= 0.03:
        return f"{opp}'s defense graded a touch generous to {unit} (~{pct:.0f}% above average), a small tailwind"
    if d <= -0.08:
        return (f"{opp} were stingy against {unit} last year — roughly {pct:.0f}% below league average, "
                f"and that's a big part of why we're on the under")
    if d <= -0.03:
        return f"{opp} held up a bit better than average against {unit} (~{pct:.0f}% below), which caps the ceiling"
    return f"{opp} were roughly league-average against {unit}"


def _env_clause(row, side):
    """A game-environment read from the implied team total."""
    ip = row.get("implied_pts")
    if ip is None:
        return None
    if ip >= 26:
        return f"the market has this team implied for ~{ip:.0f}, a top-shelf scoring environment that lifts volume"
    if ip >= 23:
        return f"a healthy ~{ip:.0f}-point implied team total keeps the volume flowing"
    if ip <= 18:
        return f"a soft ~{ip:.0f}-point implied total points to a grind-it-out, low-volume game — fuel for the under"
    if ip <= 20.5:
        return f"a modest ~{ip:.0f}-point implied total tempers the ceiling"
    return None


def reason(row, news=None):
    """Handicapper-voiced rationale. We anchor to the (sharp) market number and build the case
    from the NAMED factors that actually move this pick — the opponent's defense-vs-position, the
    game's implied total, and where our adjusted number lands — then the edge in cents. Every
    pick earns a real, specific paragraph, not a template."""
    mkt_lbl = {"rec_yds": "receiving yards", "rush_yds": "rushing yards", "receptions": "catches",
               "pass_yds": "passing yards", "rush_att": "rush attempts",
               "kicking_points": "kicking points", "field_goals": "field goals"}.get(row["mkt"], row["mkt"])
    name, side, line, proj = row["name"], row["side"], row["line"], row["desk_proj"]
    s = side.lower()
    price = int(row["price"])
    io, iu = implied(row.get("over_price_kept", price)), implied(row.get("under_price_kept", price))
    vig = (io + iu - 1) * 100 if (io and iu) else 4.5
    ev = row.get("ev", 0) * 100

    status = row.get("status", classify(row))
    if status == "REVIEW":
        return (f"{name}'s {line} {mkt_lbl} is on our radar but we're not firing yet — the matchup and "
                f"role signals don't line up cleanly enough to bet into the market's number. We'll "
                f"revisit once the snap counts and injury report post."
                + (f" Latest: {news}" if news else ""))

    # Build the specific case: lead with the strongest real factor, then support it.
    dfc = _defense_clause(row, side)
    envc = _env_clause(row, side)
    # which factor is doing more work? (bigger deviation from neutral)
    opp_dev = abs(row.get("opp_tilt", 1) - 1); env_dev = abs(row.get("env_tilt", 1) - 1)
    clauses = []
    if dfc and (opp_dev >= env_dev): clauses.append(dfc)
    if envc: clauses.append(envc)
    if dfc and (opp_dev < env_dev): clauses.append(dfc)
    if not clauses:
        clauses.append(f"the number's a touch soft at the price and our read on the workload favors the {s}")

    open_lines = {
        "Over": f"We're on {name}'s over {line} {mkt_lbl}. ",
        "Under": f"Give us {name} under {line} {mkt_lbl}. ",
    }
    lead = open_lines.get(side, f"We like {name} {s} {line} {mkt_lbl}. ")
    case = clauses[0][0].upper() + clauses[0][1:] + ". "
    if len(clauses) > 1:
        case += "On top of that, " + clauses[1] + ". "
    # where our adjusted number sits + the money
    num = (f"Roll the matchup and the environment into the base line and we land at {proj} — on the "
           f"{s} side of {line}. ")
    money = (f"Strip the {vig:.0f}% juice and the fair {s} is {int(row['fair']*100)}%; we make it "
             f"{int(row['model_p']*100)}%, worth {ev:+.0f} cents on the dollar at {price:+d}"
             + (f" ({row.get('book')})" if row.get("book") else "") + ". ")
    trust = ("This is the tier our numbers grade out sharpest." if row["confidence"] >= 0.67
             else "Role's still settling, so we're sizing it as a lean." if row["confidence"] >= 0.5
             else "Thin sample — a small, speculative sprinkle.")
    tail = f" Beat-writer note: {news}" if news else ""
    return lead + case + num + money + trust + tail


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
