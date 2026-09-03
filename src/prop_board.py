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


def pi_week(recency_week):
    """Upcoming NFL week for the intel file: the first week whose games are still ahead."""
    try:
        import datetime
        sch = pd.read_parquet(os.path.join(os.path.dirname(__file__), "..", "data", "schedules.parquet"))
        sch = sch[sch.season == sch.season.max()]
        today = datetime.date.today().isoformat()
        ahead = sch[sch.gameday >= today] if "gameday" in sch.columns else sch
        return int(ahead.week.min()) if len(ahead) else 1
    except Exception:
        return 1


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


RAW_LINES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "_raw_prop_lines.parquet")


def fetch_prop_board(key=None, days=14, max_games=None, raw_path=None):
    """Fetch every posted prop line from every book (raw, unpriced) and cache it so the board
    can be RE-PRICED for free after any model change (mirrors td_board's fetch/price split).
    `days` may be fractional (a kickoff window of 150 minutes = 150/1440)."""
    key = key or os.environ.get("ODDS_API_KEY")
    events = fetch_events(key, days=days)
    ids = [e["id"] for e in events][:max_games] if max_games else [e["id"] for e in events]
    if not ids:
        return pd.DataFrame(), None
    board, rem = fetch_props(key, ids)
    try:
        if len(board): board.to_parquet(raw_path or RAW_LINES, index=False)
    except Exception as e:
        print("raw prop lines not cached:", e)
    return board, rem


def build_prop_board(key=None, projections=None, td_rates=None, max_games=None, weather_by_game=None,
                     days=14, recency_season=2025, recency_week=None, cached=False, raw_path=None):
    """Returns the full priced prop board. The EDGE is recency: our recent-form number vs the
    market line (validated +8-13% ROI out-of-sample; matchup did NOT predict, recency did).
    cached=True re-prices the last fetched raw lines (no credits)."""
    if projections is None:
        raise ValueError("pass projections (from props_projection.project)")
    rp = raw_path or RAW_LINES
    if cached and os.path.exists(rp):
        board, rem = pd.read_parquet(rp), "cached"
    else:
        board, rem = fetch_prop_board(key=key, days=days, max_games=max_games, raw_path=raw_path)
    if not len(board):
        return pd.DataFrame(), rem
    return price_prop_board(board, projections, weather_by_game=weather_by_game,
                            recency_season=recency_season, recency_week=recency_week), rem


def price_prop_board(board, projections, weather_by_game=None, recency_season=2025, recency_week=None):
    """Price a raw multi-book line table: the multi-factor projection, the distribution, the
    side, the edge, the injury/intel layers, and the line shop. Pure function of its inputs."""
    rem = None
    proj = projections.copy()
    proj["_k"] = proj["name"].map(_norm)
    pj = proj.set_index("_k")
    # THE MODEL: a MULTI-FACTOR projection = baseline + weighted(recency, mean-reversion, opponent
    # defense, game total, home). Weights are earned out-of-sample (multifactor.py) — the alpha is
    # the ensemble, not any single factor. Recency is keyed by player_id -> mapped to names.
    import recency as rc, multifactor as mf, board_factors as bf, json as _json
    rec = rc.build_recency(through_season=recency_season, through_week=recency_week)
    id2name = dict(zip(proj["player_id"], proj["name"])) if "player_id" in proj.columns else {}
    recmap = {}
    for r in rec.to_dict("records"):
        nm = id2name.get(r["pid"])
        if nm:
            recmap[(_norm(nm), r["mkt"])] = r
    try:
        model = _json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "multifactor_weights.json")))
    except Exception:
        model = {"features": [], "base_med": {}, "weights": {}}
    _pmap, sched_rows, code2full, _adp, def_vs = bf._load_context()
    # INJURY LAYER: own status (Out -> never a pick) + the measured role cascade (a teammate's
    # absence vacates opportunity that redistributes). Loads cleanly even before the week's
    # report is posted.
    try:
        import injury_layer as il
        INJ = il.InjuryLayer(season=int(recency_season) + (1 if recency_week is None else 0))
    except Exception as _e:
        INJ = None; print("injury layer unavailable:", _e)
    # PLAYER INTEL: beat/pundit read per player (the model is the prior, the player is the bet).
    # Hard facts gate now; soft signals inform the reader/trust and are logged to earn a weight.
    try:
        import player_intel as pi
        INTEL = pi.PlayerIntel(season=int(recency_season) + (1 if recency_week is None else 0),
                               week=pi_week(recency_week))
    except Exception as _e:
        INTEL = None; print("player intel unavailable:", _e)
    # CONSENSUS per player-market (median line across books; de-vigged fair Over at that line).
    # In market-baseline mode the role baseline must come from the CONSENSUS, not from each
    # book's own line — otherwise every book's line implies its own baseline, every line looks
    # equally +EV, and the shop can't prefer BetRivers 77.5 over FanDuel 78.5 (it took 78.5).
    cons = {}; cons_mean = {}
    try:
        bb = board.copy(); bb["_k"] = bb["name"].map(_norm)
        for (k_, m_), g_ in bb.groupby(["_k", "mkt"]):
            med = float(g_["line"].median())
            at = g_[g_["line"] == med]
            if not len(at): at = g_.iloc[(g_["line"] - med).abs().argsort()[:1]]
            fo = []
            for _, r_ in at.iterrows():
                io2, iu2 = implied(r_["over_price"]), implied(r_["under_price"])
                if io2 + iu2: fo.append(io2 / (io2 + iu2))
            cons[(k_, m_)] = (med, float(np.mean(fo)) if fo else 0.5, int(g_["book"].nunique()))
            # CONSENSUS MEAN (Fresh Eyes day 2): the median of each book's IMPLIED MEAN (its line +
            # its de-vigged price through our distribution). With two books at different lines
            # (DK 4.5 / FD 3.5) the median LINE is 4.0 — a number nobody posts — and the old code
            # took the nearest book's opinion as the truth and then "found" an 8.7% edge betting
            # the other book (Kraft Under 4.5). Two disagreeing books define no consensus; their
            # average does not manufacture one either, but it stops the edge being invented.
            means = []
            for _, r_ in g_.drop_duplicates("book").iterrows():
                io2, iu2 = implied(r_["over_price"]), implied(r_["under_price"])
                if io2 + iu2 and m_ != "anytime_td":
                    try: means.append(float(pm.market_mean(m_, float(r_["line"]), io2 / (io2 + iu2))))
                    except Exception: pass
            if means: cons_mean[(k_, m_)] = float(np.median(means))
    except Exception as _e:
        cons = {}; cons_mean = {}
    # MARKET LAG (Fresh Eyes day 2): before pricing, measure which tilts the market already
    # prices this week — regress (market mean at the consensus − our season baseline) on our
    # raw features across the slate — and pool it with prior weeks (market_lag.py). The form
    # tilt under a market baseline is scaled by the market's LAG on it (1 = the market ignores
    # late-season form in this market; 0 = it prices it fully, so we add nothing).
    lag_factors = {}
    try:
        import market_lag as ml
        lag_rows = []
        seen = set()
        for _, b0 in board.iterrows():
            k0 = _norm(b0["name"]); m0 = b0["mkt"]
            if (k0, m0) in seen or m0 == "anytime_td":
                continue
            seen.add((k0, m0))
            rr0 = recmap.get((k0, m0)); pr0 = pj.loc[k0] if k0 in pj.index else None
            if rr0 is None or pr0 is None:
                continue
            if isinstance(pr0, pd.DataFrame): pr0 = pr0.iloc[0]
            cl0, cf0, _ = cons.get((k0, m0), (None, None, 0))
            if cl0 is None:
                continue
            g0, _o = bf._game_ctx(b0["game"], bf.TEAM_FIX.get(pr0["team"], pr0["team"]), sched_rows, code2full)
            tot0 = (g0.get("implied_team_pts", 22) * 2) if g0 else 44.0
            bl0 = float(rr0["baseline"])
            rmed0 = (model.get("role_med", {}).get(m0) or {}).get("role_rr", 0.0)
            rrr0 = rr0.get("role_rr"); rsd0 = rr0.get("role_snap_d")
            lag_rows.append({"mkt": m0, "gap": float(pm.market_mean(m0, cl0, cf0)) - bl0, "total_c": tot0 - 44.0,
                             "base_c": bl0 - model.get("base_med", {}).get(m0, bl0),
                             "form3": float(rr0["last3"]) - bl0, "form1": float(rr0["last1"]) - bl0,
                             "role_rr": (float(rrr0) - rmed0) if rrr0 is not None and rrr0 == rrr0 else 0.0,
                             "role_snap_d": float(rsd0) if rsd0 is not None and rsd0 == rsd0 else 0.0})
        applied = {}
        for m_ in {r_["mkt"] for r_ in lag_rows}:
            w_ = model.get("weights", {}).get(m_, {}); fr_ = mf.OFFSEASON_FORM_RATIO.get(m_, mf.OFFSEASON_FORM_DEFAULT)
            applied[m_] = {"form3": w_.get("form3", 0.0) * fr_, "form1": w_.get("form1", 0.0) * fr_,
                           "role_rr": w_.get("role_rr", 0.0) * mf.OFFSEASON_ROLE_RATIO["role_rr"].get(m_, 0.5),
                           "role_snap_d": w_.get("role_snap_d", 0.0) * mf.OFFSEASON_ROLE_RATIO["role_snap_d"].get(m_, 0.5)}
        fit_ = ml.fit(lag_rows, applied)
        wk_ = (int(recency_week) + 1) if recency_week is not None else 1
        pooled_ = ml.update(int(recency_season) + (1 if recency_week is None else 0), wk_, fit_)
        lag_factors = {m_: v_.get("lag", {}) for m_, v_ in pooled_.items()}
        print("market lag β (share of our tilt the market already prices):", {m_: (fit_[m_].get("beta"), fit_[m_].get("ci90")) for m_ in fit_})
        print("market lag factors:", {m_: (v_.get("form3") if v_ else None) for m_, v_ in lag_factors.items()})
    except Exception as _e:
        print("market lag skipped:", _e)
    out = []
    for _, b in board.iterrows():
        k = _norm(b["name"]); mkt = b["mkt"]
        pr = pj.loc[k] if k in pj.index else None
        if pr is not None and isinstance(pr, pd.DataFrame):
            pr = pr.iloc[0]
        if mkt == "anytime_td":
            continue
        own_inj = INJ.status(b["name"]) if INJ else None
        if own_inj and own_inj.get("status") == "Out":
            continue                                       # listed Out: never on the board
        intel = INTEL.get(b["name"]) if INTEL else None
        if intel and intel.get("hard") == "OUT":
            continue                                       # beat-reported out before the official report
        rr = recmap.get((k, mkt))
        pos = pr["position"] if pr is not None else "?"
        team = pr["team"] if pr is not None else None
        contrib = {}
        if rr is not None:              # we have a recent log -> full multi-factor projection
            baseline = float(rr["baseline"]); log = rr["log"]; src = "multifactor"
            # game context for the non-recency factors
            g, opp_team = bf._game_ctx(b["game"], bf.TEAM_FIX.get(team, team), sched_rows, code2full)
            total = (g.get("implied_team_pts", 22) * 2) if g else 44.0    # game total ~ 2x team implied
            # venue from the schedule row (Fresh Eyes day 2: the old string test compared a team code
            # to the last three letters of the home team's full name — home was ALWAYS 0)
            home = 1.0 if (g and g.get("is_home")) else 0.0
            ddelta = 0.0
            if g and opp_team:
                if mkt in ("rush_yds", "rush_att"): ddelta = def_vs.get("RB", {}).get(opp_team, 0.0)
                elif mkt in ("rec_yds", "receptions"): ddelta = def_vs.get("WR", {}).get(opp_team, 0.0)
            rmed = (model.get("role_med", {}).get(mkt) or {}).get("role_rr", 0.0)
            _rrr = rr.get("role_rr"); _rsd = rr.get("role_snap_d")
            feats = {"form3": float(rr["last3"]) - baseline, "form1": float(rr["last1"]) - baseline,
                     "def_delta": ddelta, "total_c": total - 44.0, "home": home,
                     "base_c": baseline - model.get("base_med", {}).get(mkt, baseline),
                     # ROLE (earned, universe_lab): route participation vs the market's typical player, snap-share trend
                     "role_rr": (float(_rrr) - rmed) if _rrr is not None and _rrr == _rrr else 0.0,
                     "role_snap_d": float(_rsd) if _rsd is not None and _rsd == _rsd else 0.0}
            # OFFSEASON RAIL (measured, multifactor.OFFSEASON_FORM_RATIO): when the "recent" games
            # are last season's finale (weeks 17-18 rest/backup noise), form predicts the residual
            # at only ~35-65% of its in-season strength. The model's single form weight is fit on
            # in-season rows, so applying it unscaled across the offseason roughly DOUBLED every
            # Week-1 edge and manufactured a tail of absurd ones (a +45% "edge" on a rushing under).
            role_base = baseline
            # PER-PLAYER gap (Fresh Eyes day 2): the offseason treatment keys off HIS window, not
            # the global mode. gap 0 = his last game is this season (full form, own baseline);
            # gap 1 = his window crosses one offseason (the measured half-strength ratio + the
            # market-role baseline — also for a player returning in Week 6 with no 2026 games);
            # gap >= 2 = he missed a whole season (Watson: "closed last season with 176, 125..."
            # were 2024 games). The ratio was measured on gap-1 rows only; gap >= 2 has n = 4-12
            # in five seasons — unmeasurable — so form is NOT applied and trust is cut.
            cur_season = int(recency_season) + (1 if recency_week is None else 0)
            last_season = rr.get("last_season")
            gap = (cur_season - int(last_season)) if last_season is not None and last_season == last_season else (1 if recency_week is None else 0)
            if gap >= 2:
                feats["form3"] = 0.0; feats["form1"] = 0.0; feats["role_rr"] = 0.0; feats["role_snap_d"] = 0.0
            lag = 1.0; lagd = {}
            if gap >= 1:
                ratio = mf.OFFSEASON_FORM_RATIO.get(mkt, mf.OFFSEASON_FORM_DEFAULT)
                lagd = lag_factors.get(mkt, {}) or {}       # the market's lag on each tilt (measured weekly)
                lag = float(lagd.get("form3", 1.0))
                feats["form3"] *= ratio * lag; feats["form1"] *= ratio * float(lagd.get("form1", 1.0))
                feats["role_rr"] *= mf.OFFSEASON_ROLE_RATIO["role_rr"].get(mkt, 0.5) * float(lagd.get("role_rr", 1.0))
                feats["role_snap_d"] *= mf.OFFSEASON_ROLE_RATIO["role_snap_d"].get(mkt, 0.5) * float(lagd.get("role_snap_d", 1.0))
                # OFFSEASON ROLE BASELINE: with no current-season games, our baseline is last
                # season's average — blind to the role changes the market has already priced (a
                # backup now starting, a rookie's full role, a star's new QB). Every absurd Week-1
                # edge traced to that gap (a 21% "edge" on Justin Jefferson under). So in Week 1
                # the MARKET-IMPLIED MEAN is the role baseline and our edge is the discounted
                # tilts on top of it — our alpha, the market's role read. Not anchoring: once
                # current-season games exist we revert to our own (validated) baseline.
                c_line, c_fair, _nb = cons.get((k, mkt), (None, None, 0))
                if c_line is None:
                    io_, iu_ = implied(b["over_price"]), implied(b["under_price"])
                    c_fair = io_ / (io_ + iu_) if (io_ + iu_) else 0.5; c_line = b["line"]
                role_base = float(cons_mean.get((k, mkt), pm.market_mean(mkt, c_line, c_fair)))
                # The mean-reversion weight was EARNED on our own baseline — a noisy sample mean
                # that overstates high values and deserves shrinking. The market's number is
                # already a shrunk estimate; regressing it again double-shrinks and pushed 73% of
                # the Week-1 board to Unders (every star "due to fall"). With a market baseline,
                # mean-reversion is zero — the market has already done it.
                feats["base_c"] = 0.0
                # ...and so has the ENVIRONMENT (Fresh Eyes day 2, measured on this week's board,
                # mean-reversion controlled): the market's number minus our season baseline moves
                # with the game total at 1.14/pt for passing (our earned weight 1.20), 0.21 for
                # receiving (earned 0.21), and with home at -4.9 for passing (earned -4.1); R² .82
                # for passing. The market line IS our baseline plus our own environment tilts, so
                # adding total_c/home on top of it double-counts — Herbert O233.5 (+15.6 from the
                # total) and Goff O261.5 (+14.9) were phantom edges. Under the market baseline only
                # the tilt the market is known to LAG applies: recent form (half-strength).
                feats["total_c"] = 0.0; feats["home"] = 0.0; feats["def_delta"] = 0.0
            resid = mf.predict_residual(mkt, feats, model)
            our_mu = max(role_base + resid, 0.1)
            trend = round(float(rr["last3"]) - baseline, 1)
            # per-factor contribution (weight x feature) for the honest scorecard/rationale
            w = model.get("weights", {}).get(mkt, {})
            contrib = {f: round(w.get(f, 0.0) * feats[f], 2) for f in feats} if w else {}
        else:                          # no recent games (rookie / new role) -> role-model projection
            our_mu = float(pr[mkt]) if (pr is not None and mkt in pr and pd.notna(pr[mkt])) else None
            trend = 0.0; baseline = our_mu; log = None; src = "role"; gap = None; last_season = None; lag = None
        if our_mu is None or our_mu <= 0:
            continue
        wg = (weather_by_game or {}).get(b["game"])
        wx_mult = 1.0
        if wg:
            wx_mult = wx.market_weather_mult(mkt, wind=wg.get("wind"), temp=wg.get("temp"), roof=wg.get("roof", "outdoors"))
            our_mu = our_mu * wx_mult
        # role cascade: a teammate's absence redistributes volume to him (measured coefficients)
        role_mult, role_note = (INJ.role_multiplier(b["name"], mkt) if INJ else (1.0, None))
        if role_mult != 1.0:
            our_mu = our_mu * role_mult
        p = price_ou(mkt, our_mu, b["line"], b["over_price"], b["under_price"])
        conf = float(pr["confidence"]) if pr is not None and "confidence" in pr else 0.2
        if gap is not None and gap >= 2: conf *= 0.7    # missed a season: the form read is unmeasurable, trust cut
        # own status shy of Out: Doubtful/Questionable erode trust (the scorecard shows it too)
        if own_inj and own_inj.get("status") == "Doubtful": conf *= 0.5       # mostly don't play (scratch rail)
        elif own_inj and own_inj.get("status") == "Questionable": conf *= 0.92 # measured: no performance drop; scratch risk only
        inj_lean, inj_note = (INJ.injury_lean(b["name"], mkt) if INJ else (None, None))
        if intel and intel.get("hard") == "LIMITED": conf *= 0.6   # pitch count / eased in: a fact, cuts trust
        out.append({"name": b["name"], "game": b["game"], "book": b["book"], "mkt": mkt,
                    "line": b["line"], "desk_proj": round(our_mu, 1),
                    "position": pos, "team": team if team else "?",
                    "moved": bool(pr["moved"]) if pr is not None and "moved" in pr else False,
                    "rookie": bool(pr["rookie"]) if pr is not None and "rookie" in pr else False,
                    "confidence": conf, "edge_src": src, "trend": round(trend, 1),
                    "baseline": (round(baseline, 1) if baseline else None), "recent_log": log,
                    "contrib": contrib, "offseason": (gap is None and recency_week is None) or (gap is not None and gap >= 1),
                    "form_gap": gap, "last_season": (int(last_season) if last_season is not None and last_season == last_season else None),
                    "market_lag": (round(float(lag), 3) if lag is not None else None),
                    "market_num": round(pm.market_mean(mkt, b["line"], p["fair_over"]), 1),
                    "cons_mean": (round(cons_mean[(k, mkt)], 1) if (k, mkt) in cons_mean else None),
                    "side": p["side"], "edge": p["edge"], "model_p": p["model_p"], "fair": p["fair"],
                    "price": p["price"], "ev": p["ev"],
                    "over_price_kept": b["over_price"], "under_price_kept": b["under_price"],
                    "inj_status": (own_inj or {}).get("status"), "inj_detail": (own_inj or {}).get("injury"),
                    "role_mult": role_mult, "role_note": role_note,
                    "inj_lean": inj_lean, "inj_note": inj_note,
                    "intel_lean": (INTEL.lean(b["name"], p["side"]) if INTEL else None),
                    "intel_note": (INTEL.note(b["name"]) if INTEL else None),
                    "intel_signals": (intel or {}).get("signals"), "intel_hard": (intel or {}).get("hard"),
                    "wx_mult": round(float(wx_mult), 3), "wx_wind": (wg or {}).get("wind"), "wx_temp": (wg or {}).get("temp"),
                    "wx_roof": (wg or {}).get("roof")})
    df = pd.DataFrame(out)
    if not len(df):
        return df
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
    # MARKET BREADTH (Fresh Eyes day 2): how many books post this player-market AT ALL, and the
    # consensus line. `n_books` above counts books at OUR exact line — and the row we keep is the
    # best line for our side, which is usually a one-book outlier. So every shopped Over sat at
    # n_books=1 (trust -3) while a same-line Under scored +3: a 6-point Desk-Score swing that was
    # an artifact of shopping, not evidence. Trust reads `n_books_mkt`; the shopping context stays
    # for display and `cons_line` lets the grading loop earn "is the outlier book soft or sharp?"
    breadth = (df.groupby(["name", "mkt"]).agg(n_books_mkt=("book", "nunique"),
                                               cons_line=("line", "median")).reset_index())
    df = df.sort_values("edge", ascending=False).drop_duplicates(["name", "mkt", "side"])
    if len(shopdf):
        df = df.merge(shopdf, on=["name", "mkt", "side", "line"], how="left")
        df["price"] = df["best_price"].fillna(df["price"]).astype(int)   # best price AT our line
        df["book"] = df["book_best"].fillna(df["book"])
        df["ev"] = (df["model_p"] * df["price"].map(american_profit) - (1 - df["model_p"])).round(3)
    df = df.merge(breadth, on=["name", "mkt"], how="left")
    df["line_vs_cons"] = (df["line"] - df["cons_line"]).round(1)
    # WHERE THE EDGE COMES FROM (Fresh Eyes day 2): our tilt vs the consensus mean, and this
    # book's NUMBER vs the consensus — both signed for our side. A pick whose edge is mostly the
    # number is an OFF-MARKET play (a sharp's line-shop: BetRivers 247.5 when three books imply
    # ~234), honest only when 3+ books define the consensus; with two disagreeing books there
    # is no consensus to trust, so it is REVIEW, never a PICK.
    sgn = np.where(df["side"].eq("Over"), 1.0, -1.0)
    cm = df["cons_mean"].astype(float)
    df["tilt"] = ((df["desk_proj"].astype(float) - cm) * sgn).round(2)
    df["off_market"] = ((cm - df["market_num"].astype(float)) * sgn).round(2)
    def _kind(r):
        if r["cons_mean"] is None or r["cons_mean"] != r["cons_mean"]:
            return "model"
        om = max(float(r["off_market"] or 0), 0.0); tl = max(float(r["tilt"] or 0), 0.0)
        if om <= 0 or om < tl:
            return "model"
        return "off-market" if int(r.get("n_books_mkt") or 0) >= 3 else "no-consensus"
    df["thesis_kind"] = df.apply(_kind, axis=1)
    return df


# trust band: how far our projection may sit from the line before we assume WE'RE stale
# (the market repriced a role) rather than that we've found an edge.
TRUST_LO, TRUST_HI = 0.62, 1.6

FLAT_PRIOR_MARKETS = {"kicking_points", "field_goals"}   # league-prior projections, not per-player

def classify(row):
    """A PICK requires the multi-factor edge: a real recent-data-backed projection that diverges
    from the line by more than the market-trust threshold. Rookies/no-recent-data -> REVIEW."""
    if row.get("mkt") in FLAT_PRIOR_MARKETS:
        return "REVIEW"                                   # kicker flat prior — never a pick
    if row.get("edge_src") != "multifactor":
        return "REVIEW"                                   # no recent data to model — untrusted
    if row.get("thesis_kind") == "no-consensus":
        return "REVIEW"                                   # two books disagree, no consensus to bet against
    ratio = row["desk_proj"] / row["line"] if row["line"] else 1.0
    if not (TRUST_LO <= ratio <= TRUST_HI):
        return "REVIEW"
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
    """Handicapper voice rooted in the REAL edge: the player's recent game log vs a market line
    that's still anchored to his season number. We show the last games, the form, the baseline the
    book is pricing, and the gap — then the money. Every pick earns a specific, data-rooted case."""
    mkt_lbl = {"rec_yds": "receiving yards", "rush_yds": "rushing yards", "receptions": "catches",
               "pass_yds": "passing yards", "rush_att": "rush attempts",
               "kicking_points": "kicking points", "field_goals": "field goals"}.get(row["mkt"], row["mkt"])
    unit = {"rec_yds": "yards", "rush_yds": "yards", "pass_yds": "yards",
            "receptions": "catches", "rush_att": "carries"}.get(row["mkt"], "")
    name, side, line = row["name"], row["side"], row["line"]
    s = side.lower(); price = int(row["price"])
    io, iu = implied(row.get("over_price_kept", price)), implied(row.get("under_price_kept", price))
    vig = (io + iu - 1) * 100 if (io and iu) else 4.5
    ev = row.get("ev", 0) * 100
    log = row.get("recent_log"); baseline = row.get("baseline"); trend = row.get("trend", 0)
    form = row.get("desk_proj")

    status = row.get("status", classify(row))
    if status == "REVIEW":
        if row.get("edge_src") != "multifactor":
            return (f"{name}'s {line} {mkt_lbl} — we're holding. No stable recent game log to lean on "
                    f"(new role or rookie), and our edge is a recency read, so we won't bet into the "
                    f"market's number blind. Revisit once he's banked a few games."
                    + (f" Latest: {news}" if news else ""))
        return (f"{name}'s {line} {mkt_lbl} is close to our number — no strong recent trend to attack "
                f"the line with, so we pass rather than force it." + (f" Latest: {news}" if news else ""))

    # the case: the recent log PLUS the multi-factor drivers (recency, matchup, environment,
    # mean-reversion) that actually moved our number — the ensemble is the alpha.
    log = list(log) if log is not None and len(log) else None
    loglist = ", ".join(f"{v:g}" for v in log) if log else None
    offseason = row.get("offseason")
    gap = row.get("form_gap"); ls_ = row.get("last_season")
    stale_form = gap is not None and gap == gap and int(gap) >= 2
    if stale_form:
        when = f"last played in {int(ls_)}, closing that year" if ls_ is not None and ls_ == ls_ else "last played two seasons ago, closing"
    else:
        when = "closed last season" if offseason else "over his last four"
    open_lines = {"Over": f"We're on {name} over {line} {mkt_lbl}. ",
                  "Under": f"Give us {name} under {line} {mkt_lbl}. "}
    lead = open_lines.get(side, f"We like {name} {s} {line} {mkt_lbl}. ")
    case = ""
    if loglist and baseline:
        case = f"He {when} with games of {loglist} {unit} (season line ~{baseline:g}). "
    # name the factors doing the work, from their actual signed contributions to our number
    contrib = row.get("contrib") or {}
    LABEL = {"form3": "recent form", "form1": "last game", "def_delta": "the matchup",
             "total_c": "the game total", "home": "the venue", "base_c": "mean-reversion",
             "role_rr": "his route participation", "role_snap_d": "his snap-share trend"}
    drivers = sorted([(f, v) for f, v in contrib.items() if abs(v) >= 0.4], key=lambda x: -abs(x[1]))[:3]
    mkt_num = row.get("market_num")
    kind = row.get("thesis_kind"); cmn = row.get("cons_mean"); nbm_ = row.get("n_books_mkt"); om_ = row.get("off_market"); tl_ = row.get("tilt")
    if kind == "off-market" and cmn is not None and cmn == cmn and mkt_num:
        # the NUMBER is the play: this book sits off the consensus of the other books
        case += (f"{int(nbm_)} books post this market; their consensus implies about {float(cmn):g} {unit or mkt_lbl} and "
                 f"{str(row.get('book')).replace('_us', '')} hangs {float(line):g}, which implies ~{float(mkt_num):g} — "
                 f"{abs(float(om_)):g} {unit or 'units'} off the market for our side. That number, not our model, is the play"
                 + (f"; our own read adds {float(tl_):+g} on top. " if tl_ is not None and tl_ == tl_ and abs(float(tl_)) >= 0.3 else "; our own read is flat on him. "))
    elif drivers:
        parts = []
        for f, v in drivers:
            d = ("pushes down" if v < 0 else "pushes up")
            parts.append(f"{LABEL.get(f, f)} {d}")
        # compare our expected number to the MARKET's implied number (both means) — the honest,
        # non-contradictory read (comparing our mean to the line itself misleads on skewed stats).
        vs = (f"we land at {form:g} where the market's number implies about {mkt_num:g}"
              if mkt_num else f"our number sits at {form:g}")
        case += "Our model weighs it out (" + ", ".join(parts) + f") and {vs} — that gap is the {s}. "
    elif mkt_num:
        case += f"We project {form:g}; the market's line implies about {mkt_num:g} — the gap is the {s}. "
    money = (f"Strip the {vig:.0f}% juice and the fair {s} is {int(row['fair']*100)}%; we make it "
             f"{int(row['model_p']*100)}% — {ev:+.0f} cents on the dollar at {price:+d}"
             + (f" ({row.get('book')})" if row.get("book") else "")
             + (f", best of {int(row['n_books'])} books" if row.get("n_books", 0) and row['n_books'] > 1 else "") + ". ")
    # the shop, in words: our line vs the consensus across the books that post this market
    nbm = row.get("n_books_mkt"); cl = row.get("cons_line"); lvc = row.get("line_vs_cons")
    if nbm is not None and nbm == nbm and cl is not None and cl == cl and lvc is not None and lvc == lvc and abs(float(lvc)) >= 0.5 and int(nbm) > 1:
        better = (side == "Over" and lvc < 0) or (side == "Under" and lvc > 0)
        pts = abs(float(lvc)); ptxt = f"{pts:g} point{'s' if pts != 1 else ''}"
        money += (f"The consensus line across {int(nbm)} books is {float(cl):g}; {str(row.get('book')).replace('_us', '')} posts {float(line):g} — "
                  + (f"the softer number for our side, so the shop is worth {ptxt} of line. " if better
                     else "a worse number than the field, so the price has to carry the play. "))
    # kickoff weather, only when it moved the number (wind above ~8 mph or cold, outdoors)
    wxm = row.get("wx_mult")
    if wxm is not None and wxm == wxm and abs(float(wxm) - 1.0) >= 0.02 and row.get("wx_roof") not in ("dome", "closed", "indoors"):
        wnd = row.get("wx_wind"); tmp = row.get("wx_temp")
        wbits = []
        if wnd is not None and wnd == wnd and float(wnd) >= 8: wbits.append(f"{float(wnd):.0f} mph wind")
        if tmp is not None and tmp == tmp and float(tmp) < 55: wbits.append(f"{float(tmp):.0f}°F")
        if wbits:
            case += (f"Kickoff forecast: {', '.join(wbits)} — that {'trims' if wxm < 1 else 'lifts'} our number "
                     f"{abs(float(wxm) - 1) * 100:.0f}% ({'passing suffers in wind and cold' if wxm < 1 else 'offenses lean on the run'}). ")
    if stale_form:
        conf_note = (f"Honest caveat: his last games were in {int(ls_) if ls_ is not None and ls_ == ls_ else 'a prior season'} — a missed season, so "
                     "recent form is NOT applied (unmeasurable at that gap); this number is the market's role read plus the "
                     "environment (game total, venue, weather), with trust cut.")
    else:
        conf_note = ("Honest caveat: this leans on late-2025 form across the offseason — an early-season "
                     "read, sized as a lean." if offseason else "Multi-factor signal, sized to conviction.")
        mlag = row.get("market_lag")
        if offseason and mlag is not None and mlag == mlag and float(mlag) < 0.75:
            conf_note += (f" Measured this week: the market already prices late-season {mkt_lbl} form"
                          + (" in full" if float(mlag) <= 0.05 else f" at about {(1 - float(mlag)) * 100:.0f}% of our weight")
                          + ", so our form tilt is scaled down to what it leaves on the table.")
    # injury context: the role cascade (a teammate out feeds him) or his own status
    inj_line = ""
    rn = row.get("role_note"); st = row.get("inj_status")
    if rn and isinstance(rn, str):
        inj_line = f" Injury angle: {rn}."
    elif st in ("Doubtful", "Questionable"):
        det = row.get("inj_detail") or "injury"
        inj_line = (f" Injury note: he's listed {st} ({det}). Questionable players who play produce like healthy ones "
                    f"(measured), so this is a scratch risk, not a performance discount — confirm he's active before betting."
                    if st == "Questionable" else
                    f" Injury caveat: he's listed {st} ({det}) — Doubtful players mostly sit; treat as a scratch risk and confirm active.")
    news = news or row.get("intel_note")
    tail = f" Beat note: {news}" if (news and isinstance(news, str)) else ""
    return lead + case + money + conf_note + inj_line + tail


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
