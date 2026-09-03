"""The pick ledger — every number we put out, frozen at the moment we put it out.

This is the foundation of the forward-grading loop (THE_DESK_PROMPT work order 2): a model
that can't be graded can't improve. Each weekly run snapshots the whole product into
data/ledger/<season>_w<week>.json:

  opened  — the week's OFFICIAL open: the Tuesday card, run with `run_board.py --open`. This
            is what "beat the close" and the record are measured against. Before an official
            open exists, the first build stands in as a "preview" open (opened_kind) and the
            official run replaces it; an official open is never replaced.
  latest  — the most recent snapshot (Thursday refresh, Sunday final, late swap overwrite it).
  history — one line per snapshot taken (when, how many Card / late-swap rows, kind).
  close   — closing lines for the opened picks (grade_week.capture_close: the last
            line-history snapshot before kickoff — consensus line + best price at our side).
  results — filled by grade_week.py after the games: W/L/push/void, units, CLV.
  late_swap — the Sunday 11:40 ET lane: players lifted by a teammate's game-day absence who
            clear the Card's bar; its own record, never retro-added to the opened Card.
Every row also carries its line_move (opened → now, from line_history) and its Desk Score
components (score_conviction / score_trust) so the loop can re-earn the assumed constants.

Each snapshot carries: The Card (every recommended straight bet with its model number, price,
grade, units and the factor contributions that produced it), the graded pool (every scored
candidate with grade — the calibration-by-grade sample), the Sunday Stack, the Touchdown Tax,
the primetime SGPs and the games intel lane.
"""
import os, sys, json, datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
LEDGER_DIR = os.path.join(DATA, "ledger")

ABBR = {"Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF",
"Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE",
"Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
"Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX","Kansas City Chiefs":"KC",
"Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LA","Miami Dolphins":"MIA",
"Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
"New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","San Francisco 49ers":"SF",
"Seattle Seahawks":"SEA","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN","Washington Commanders":"WAS"}

PICK_FIELDS = ["name", "game", "cat", "side", "line", "price", "book", "our_num", "market_num", "model_p", "edge",
               "score", "grade", "grade_sized", "units", "recommended", "rail", "thesis", "confidence", "n_books",
               "shop_adv", "contrib", "inj_lean", "inj_status", "role_mult", "role_note", "intel_lean", "intel_signals",
               "intel_hard", "mkt_ratio", "mkt_flag", "position", "offseason", "verdict", "score_mult", "line_move",
               "score_conviction", "score_trust", "wx_mult", "wx_wind", "wx_temp", "wx_roof",
               "n_books_mkt", "cons_line", "line_vs_cons", "form_gap", "last_season", "fair",
               "market_lag", "thesis_kind", "cons_mean", "tilt", "off_market", "ud_line", "ud_over", "ud_under", "proj_rw", "proj_espn"]


def game_teams(game):
    away, home = (str(game).split(" @ ") + [""])[:2]
    return ABBR.get(away), ABBR.get(home)


def resolve_week(games, season=None, today=None):
    """Map the board's games onto the schedule -> (season, week, kickoff of the first game)."""
    sched = pd.read_parquet(f"{DATA}/schedules.parquet")
    today = today or datetime.date.today()
    season = season or (today.year if today.month >= 3 else today.year - 1)
    s = sched[(sched.season == season) & (sched.game_type == "REG")].copy()
    s["gameday"] = pd.to_datetime(s["gameday"]).dt.date
    weeks = []
    for g in games:
        away, home = game_teams(g)
        m = s[(s.away_team == away) & (s.home_team == home) & (s.gameday >= today - datetime.timedelta(days=1))]
        if len(m):
            weeks.append((int(m.iloc[0]["week"]), m.iloc[0]["gameday"]))
    if not weeks:
        return season, None, None
    wk = min(w for w, _ in weeks)
    first = min(d for w, d in weeks if w == wk)
    return season, wk, first.isoformat()


def _pick(r):
    d = {k: r.get(k) for k in PICK_FIELDS}
    d["id"] = f"{r.get('name')}|{r.get('cat')}|{r.get('side')}|{r.get('line')}"
    return d


def build_snapshot(ranked):
    games = sorted({r.get("game") for r in (ranked.get("pool") or ranked.get("recommended") or []) if r.get("game")}
                   | {g.get("game") for g in ranked.get("best_games", []) or [] if g.get("game")})
    season, week, kickoff = resolve_week(games)
    pool = ranked.get("pool") or []
    # LATE SWAP lane: a teammate's game-day absence lifted this player's number (role cascade),
    # and he clears the Card's own bar — its own record, never retro-added to the opened Card
    late = [_pick(r) for r in pool if not r.get("kind") and r.get("role_note") and "is out" in str(r.get("role_note"))
            and r.get("grade") in ("A+", "A", "A-", "B+") and (r.get("model_p") or 0) >= 0.5]
    snap = {"season": season, "week": week, "first_kickoff": kickoff,
            "generated_at": ranked.get("generated_at"), "taken_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="minutes"),
            "card": [_pick(r) for r in ranked.get("recommended", []) if not r.get("kind")],
            "late_swap": late,
            "pool": [_pick(r) for r in pool if r.get("grade") in ("A+", "A", "A-", "B+", "B")],
            "sunday_stack": ranked.get("sunday_stack"),
            "td_stack": ranked.get("td_stack"),
            "sgps": [{k: v for k, v in s.items() if k != "note"} for s in ranked.get("primetime_sgps", []) or []],
            "games": [{k: g.get(k) for k in ("pick", "kind", "game", "market", "side", "point", "price", "book", "desk_num", "edge", "score", "grade")}
                      for g in ranked.get("best_games", []) or []],
            "grading": ranked.get("grading")}
    return snap


def ledger_path(season, week):
    return os.path.join(LEDGER_DIR, f"{season}_w{int(week):02d}.json")


def snapshot(ranked, force_open=False):
    """Write the week's ledger: `latest` always. `opened` = the week's OFFICIAL open (the Tuesday
    card, run with --open); until an official open exists the first build stands in as a
    'preview' open and the next official run replaces it. An official open is never replaced."""
    snap = build_snapshot(ranked)
    if snap["week"] is None:
        print("ledger: could not resolve the week from the schedule — not written"); return None
    os.makedirs(LEDGER_DIR, exist_ok=True)
    path = ledger_path(snap["season"], snap["week"])
    led = json.load(open(path)) if os.path.exists(path) else {"season": snap["season"], "week": snap["week"], "status": "open"}
    if "opened" not in led:
        led["opened"] = snap; led["opened_kind"] = "official" if force_open else "preview"
    elif force_open and led.get("opened_kind") != "official":
        led["opened"] = snap; led["opened_kind"] = "official"
    led["latest"] = snap
    led.setdefault("history", []).append({"taken_at": snap["taken_at"], "n_card": len(snap["card"]), "n_late_swap": len(snap["late_swap"]),
                                          "kind": "official" if force_open else "refresh"})
    json.dump(led, open(path, "w"), indent=1)
    print(f"ledger: {os.path.basename(path)} [{led.get('opened_kind')} open] — card {len(snap['card'])}, late-swap {len(snap['late_swap'])}, pool {len(snap['pool'])}, "
          f"stack {'yes' if snap['sunday_stack'] else 'no'}, tax {'yes' if snap['td_stack'] else 'no'}, sgps {len(snap['sgps'])}, games {len(snap['games'])}")
    return path


def load(season, week):
    p = ledger_path(season, week)
    return json.load(open(p)) if os.path.exists(p) else None


if __name__ == "__main__":
    ranked = json.load(open(f"{DATA}/ranked_board.json"))
    snapshot(ranked, force_open="--reopen" in sys.argv)
