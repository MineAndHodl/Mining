"""The TD board — surfaces the calibrated anytime-TD model against live prices.

Pulls posted anytime-TD lines + game spreads/totals, computes each player's model
P(scores a TD) with the live game-environment multiplier, and ranks by edge vs the
posted (vig-included) price. Feeds both the standalone TD board and the Touchdown Tax
parlays. Anytime-TD is a one-sided, fat-margin market, so the edge bar is higher.
"""
import os, sys, json, urllib.request
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import td_model as tdm
from names import norm_name

API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
ABBR = {"Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF",
"Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE",
"Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
"Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX","Kansas City Chiefs":"KC",
"Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LA","Miami Dolphins":"MIA",
"Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
"New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","San Francisco 49ers":"SF",
"Seattle Seahawks":"SEA","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN","Washington Commanders":"WAS"}


def _get(url):
    r = urllib.request.urlopen(url, timeout=30)
    return json.load(r), r.headers.get("x-requests-remaining")


TD_BOOKS = "draftkings,fanduel,betmgm,fanatics,betrivers,williamhill_us,espnbet,hardrockbet"
TD_MAX_MARKET_RATIO = 1.5     # early season: our p more than 1.5x the consensus -> stale-role suspicion
EARLY_WEEKS = 3


def fetch_td_and_env(key, days=8, max_games=None):
    """Anytime-TD prices from ALL major books (line-shopped: best price per player, with the
    consensus for trust), plus each game's spread/total (DK, else any). TD prices vary a lot
    across books — shopping is pure price edge with zero model risk."""
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    frm = now.strftime("%Y-%m-%dT%H:%M:%SZ"); to = (now+datetime.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    events, rem = _get(f"{API}/events?apiKey={key}&commenceTimeFrom={frm}&commenceTimeTo={to}")
    ids = [e["id"] for e in events][:max_games] if max_games else [e["id"] for e in events]
    td_rows, env = [], {}
    for eid in ids:
        try:
            d, rem = _get(f"{API}/events/{eid}/odds?apiKey={key}&regions=us&markets=player_anytime_td,spreads,totals&oddsFormat=american&bookmakers={TD_BOOKS}")
        except Exception:
            continue
        home, away = ABBR.get(d.get("home_team")), ABBR.get(d.get("away_team"))
        gid = f"{d.get('away_team')} @ {d.get('home_team')}"
        spread_home = total = None
        prices = {}                                    # name -> {book: price}
        for bk in d.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m["key"] == "spreads":
                    if spread_home is None or bk["key"] == "draftkings":
                        for o in m["outcomes"]:
                            if ABBR.get(o["name"]) == home: spread_home = o.get("point")
                elif m["key"] == "totals":
                    if total is None or bk["key"] == "draftkings":
                        total = m["outcomes"][0].get("point")
                elif m["key"] == "player_anytime_td":
                    for o in m["outcomes"]:
                        nm = o.get("description") or o.get("name")
                        prices.setdefault(nm, {})[bk["key"]] = int(o["price"])
        for nm, bp in prices.items():
            imps = {b: implied(pr) for b, pr in bp.items()}
            best_book = min(imps, key=imps.get)        # lowest implied = best price for us
            td_rows.append({"name": nm, "price": bp[best_book], "book": best_book, "game": gid, "team_hint": None,
                            "n_books": len(bp), "implied_best": round(imps[best_book], 4),
                            "implied_consensus": round(float(np.median(list(imps.values()))), 4),
                            "price_dk": bp.get("draftkings")})
        if home and total and spread_home is not None:
            env[(gid, home, away)] = {"spread_home": spread_home, "total": total, "home": home, "away": away}
    board = pd.DataFrame(td_rows)
    save_raw_prices(board, env)
    return board, env, rem


RAW_PRICES = f"{DATA}/_raw_td_prices.parquet"; RAW_ENV = f"{DATA}/_td_env.json"


def save_raw_prices(board, env):
    """Cache the fetched prices + game environment so the board can be RE-PRICED for free
    (model changes must never cost API credits)."""
    try:
        if len(board): board.to_parquet(RAW_PRICES, index=False)
        json.dump([{"gid": k[0], "home": k[1], "away": k[2], **v} for k, v in env.items()], open(RAW_ENV, "w"))
    except Exception as e:
        print("raw TD cache not saved:", e)


def load_raw_prices():
    board = pd.read_parquet(RAW_PRICES)
    env = {(e["gid"], e["home"], e["away"]): {"spread_home": e["spread_home"], "total": e["total"], "home": e["home"], "away": e["away"]}
           for e in json.load(open(RAW_ENV))}
    return board, env


def implied(o):
    o = float(o); return 100/(o+100) if o > 0 else abs(o)/(abs(o)+100)


TEAM_FIX = {"LAR": "LA", "OAK": "LV", "SD": "LAC", "STL": "LA", "WSH": "WAS"}
XTD_FACTOR_COLS = ["xtd", "xtd_rush", "xtd_rec", "gl_share", "rz_tgt_share", "vulture_share",
                   "coach_gl_pass_rate", "conv_rush_gl", "conv_tgt_rz", "tds_last", "games", "position",
                   "GL_rush_share", "IN10_rush_share", "RZ_rush_share", "OPEN_rush_opps"]


def build_xtd_table(target_season=2026, refresh=False):
    """The decomposed expected-TD table for the current season, cached. Uses the latest two
    seasons of play-by-play (latest weighted), 2026 team overrides for offseason moves, and
    week-1 shrinkage until current-season plays exist. See xtd_model.py / xtd_validation.json."""
    import xtd_model as xm
    cache = f"{DATA}/xtd_{target_season}.parquet"
    if os.path.exists(cache) and not refresh:
        return pd.read_parquet(cache)
    proj = pd.read_parquet(f"{DATA}/projection_2026.parquet")
    current_team = {norm_name(r["name"]): TEAM_FIX.get(r["team"], r["team"]) for _, r in proj.iterrows()}
    # coaching changes flip RZ tendency: compare this season's coach to last season's
    try:
        c_now, c_prev = xm.coach_map(target_season), xm.coach_map(target_season - 1)
        new_coach = {t for t, c in c_now.items() if c and c_prev.get(t) and c != c_prev.get(t)}
    except Exception:
        new_coach = set()
    xdf = xm.build_xtd([target_season - 2, target_season - 1], target_season=target_season,
                       current_team=current_team, new_coach_teams=new_coach)
    xdf["_k"] = xdf["name"].map(norm_name)
    xdf.to_parquet(cache, index=False)
    return xdf


def current_season_weeks_played(target_season=2026):
    """How many weeks of THIS season's play-by-play we have (drives the xTD blend weight)."""
    try:
        pbp = pd.read_parquet(f"{DATA}/pbp_2021_2025.parquet", columns=["season", "week"])
        cur = pbp[pbp.season == target_season]
        return int(cur["week"].max()) if len(cur) else 0
    except Exception:
        return 0


def build_td_board(key=None, max_games=None, days=8, cached=False):
    """cached=True re-prices the last fetched prices/environment (no credits)."""
    key = key or os.environ.get("ODDS_API_KEY")
    rates = pd.read_parquet(f"{DATA}/td_rates.parquet")
    baseline = json.load(open(f"{DATA}/team_td_baseline.json"))
    if cached and os.path.exists(RAW_PRICES) and os.path.exists(RAW_ENV):
        board, env = load_raw_prices(); rem = "cached"
    else:
        board, env, rem = fetch_td_and_env(key, days=days, max_games=max_games)
    if not len(board):
        return pd.DataFrame(), rem
    # env multiplier per team from each game's live spread/total
    env_mult = {}
    for (gid, home, away), e in env.items():
        mm = tdm.env_multipliers(baseline, e["spread_home"], e["total"], home, away)
        env_mult.update(mm)
    out = tdm.price_board(rates, board, env_mult)
    # ---- blend in the decomposed xTD model (validated: equal to old at week 1, +2-3% Brier
    # in-season). p = w*xTD + (1-w)*old, w ramps .55 -> .85 as current-season weeks accrue.
    try:
        import xtd_model as xm
        xdf = build_xtd_table()
        w = xm.blend_weight(current_season_weeks_played())
        xmap = {r["_k"]: r for r in xdf.to_dict("records")}
        # injury layer: Out -> off the board; a teammate's absence lifts his scoring-zone share
        try:
            import injury_layer as il
            INJ = il.InjuryLayer(season=2026)
        except Exception:
            INJ = None
        import rz_defense as rzd
        RZD = rzd.RZDefense(2026)
        p_x_list, lam_x_list, fac = [], [], {c: [] for c in XTD_FACTOR_COLS}
        keep_mask, role_notes, rz_notes, rz_mults = [], [], [], []
        for _, b in out.iterrows():
            st = INJ.status(b["name"]) if INJ else None
            if st and st.get("status") == "Out":
                keep_mask.append(False)
            else:
                keep_mask.append(True)
            r = xmap.get(norm_name(b["name"]))
            if r is None or b.get("p_model") is None:
                p_x_list.append(None); lam_x_list.append(None); role_notes.append(None); rz_notes.append(None); rz_mults.append(None)
                for c in XTD_FACTOR_COLS: fac[c].append(None)
                continue
            team = r["team"]; mult = env_mult.get(team, 1.0)
            rmult, rnote = (INJ.role_multiplier(b["name"], "anytime_td") if INJ else (1.0, None))
            role_notes.append(rnote)
            # opponent red-zone defense (measured, small): the other team in this game
            g = str(b.get("game", "")); away_full, home_full = (g.split(" @ ") + [""])[:2]
            home_c, away_c = ABBR.get(home_full), ABBR.get(away_full)
            opp = away_c if team == home_c else (home_c if team == away_c else None)
            zmult, znote = RZD.multiplier(opp) if opp else (1.0, None)
            rz_notes.append(znote); rz_mults.append(zmult)
            lam_x = float(r["xtd"]) * mult * rmult * zmult
            p_x = float(xm.p_atd(lam_x))
            p_x_list.append(round(p_x, 3)); lam_x_list.append(round(lam_x, 3))
            for c in XTD_FACTOR_COLS: fac[c].append(r.get(c))
        out["p_xtd"] = p_x_list; out["lam_xtd"] = lam_x_list; out["role_note"] = role_notes
        out["rz_note"] = rz_notes; out["rz_mult"] = rz_mults
        for c in XTD_FACTOR_COLS: out[c] = fac[c]
        out = out[pd.Series(keep_mask, index=out.index)].copy()   # drop players listed Out
        out["p_old"] = out["p_model"]
        blended = []
        for _, b in out.iterrows():
            if b.get("p_xtd") is None or pd.isna(b.get("p_xtd")):
                blended.append(b.get("p_model"))
            else:
                blended.append(round(w * float(b["p_xtd"]) + (1 - w) * float(b["p_old"]), 3))
        out["p_model"] = blended
        out["xtd_weight"] = w
        if "implied_best" in out.columns:
            out["p_implied"] = out["implied_best"].fillna(out["p_implied"])
        out["edge"] = [round(p - i, 3) if p is not None and not pd.isna(p) and i is not None else None
                       for p, i in zip(out["p_model"], out["p_implied"])]
        # EARLY-SEASON MARKET-DISAGREEMENT RAIL (same finding as the Week-1 market-role baseline
        # for yardage props): with no current-season games, a last-season share can be a stale
        # role fact — Vidal's 2025 goal-line work came in Hampton's absence — and when we sit
        # >1.5x above EVERY book on a scorer, the likeliest explanation is a role read we're
        # missing, not 13 points of edge. Trust is cut (score x0.8) and the TD Stack skips him
        # until current-season shares exist. Logged (mkt_ratio) for the grading loop to earn.
        wk = current_season_weeks_played()
        cons = out["implied_consensus"] if "implied_consensus" in out.columns else out["p_implied"]
        out["mkt_ratio"] = [round(float(p) / float(c), 2) if p is not None and not pd.isna(p) and c and not pd.isna(c) else None
                            for p, c in zip(out["p_model"], cons)]
        out["mkt_flag"] = [(wk <= EARLY_WEEKS and r is not None and r > TD_MAX_MARKET_RATIO) for r in out["mkt_ratio"]]
        out = out.sort_values("edge", ascending=False, na_position="last")
    except Exception as e:
        print("xTD blend skipped:", e)
    return out, rem


if __name__ == "__main__":
    mg = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    b, rem = build_td_board(max_games=mg)
    print(f"credits: {rem} | TD players priced: {len(b)}")
    if len(b):
        good = b[b["edge"].notna() & (b["edge"] >= 0.03)].head(15)
        print("\n=== TD BOARD — best anytime-TD edges ===")
        print(good[["name","game","price","p_model","p_implied","edge","games"]].to_string(index=False))
