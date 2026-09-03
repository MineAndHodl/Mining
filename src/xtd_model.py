"""xTD — expected touchdowns built from DECOMPOSED red-zone data (Alex's spec).

The old TD model was one pooled number (RZ touches x a position-wide rate). Real TD
equity is structured, and every piece of that structure is measurable in play-by-play:

    xTD(player) = SUM over zone z, type t of
        E[team plays in z]            (how often the offense gets there — environment)
      x P(type t | zone z, team)      (coach's RZ / goal-line pass-vs-run tendency)
      x share(player, z, t)           (his cut of those plays — the RZ / goal-line ROLE;
                                       a backup who steals goal-line carries shows up here)
      x P(TD | z, t, player)          (conversion — finishing ability, shrunk to position)

Zones (yardline_100):  GL 1-5 (goal line)  |  IN10 6-10  |  RZ 11-20  |  OPEN 21+
Types:                 rush  |  target

Then P(anytime TD) = 1 - exp(-xTD)  (Poisson), recalibrated on holdout.

Empirical anchors already measured (data/coaching_findings.json):
  goal-line carries -> rush TDs corr 0.85 (total carries only 0.75)
  RZ targets        -> rec  TDs corr 0.77 (overall targets only 0.73)
  lead back gets only ~55% of goal-line carries (vulture is real)
  goal-line role is only 29% sticky year-over-year -> shrink last-season GL share HARD
  coach pass tendency persists if coach stays (0.44) and FLIPS on a change (-0.67)

Honest Week-1 caveat: with no current-season plays, shares come from last season, shrunk
toward a role prior (depth chart x coach concentration). As games accrue, observed
current-season shares take over — the model is built to update, not to sit on priors.
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from names import norm_name

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

ZONES = {"GL": (1, 5), "IN10": (6, 10), "RZ": (11, 20), "OPEN": (21, 100)}
TYPES = ("rush", "target")
POS_OK = ("RB", "WR", "TE", "QB", "FB")

# shrinkage strengths (pseudo-observations)
SHARE_PSEUDO_GAMES = 4.0      # player zone-share -> prior, weighted by games of data
CONV_PSEUDO_OPPS = 25.0       # player conversion -> position x zone prior
GL_YOY_STICKINESS = 0.37      # measured (5 seasons): slope of next-season GL share on last season's
# GOAL-LINE PACKAGE INCLUSION (td_role_validate.py, in-season rolling 2024 + 2025): a back's share
# of the team's goal-line SNAPS (participation data: who was on the field inside the 5) blended
# half-and-half with his goal-line CARRY share beats carry share alone on backs' Brier in both
# seasons (.18138 -> .18118, .18157 -> .18011). Carries inside the 5 are a handful of plays; being
# on the field for them is observed every play. Inclusion × the team's RB share of GL carries
# (QB sneaks take the rest) is the carry-share-equivalent.
GL_INCLUSION_BLEND = 0.5      # weight on the carry share; 1 - this on the snap-inclusion share
# Tested and NOT adopted (td_env_validate.py --stick): the goal-line role erodes through a season
# (slope .65 in weeks 1-3, .48 weeks 4-9, .36 weeks 10-18), so a higher early-season stickiness
# was tried — out-of-sample it was a wash (weeks 1-3 Brier .1715 vs .1714 across .29/.37/.60/.75).
# Open finding for the grading loop: week-1-mode predictions of .45-.55 HIT ~62% in weeks 1-3
# (n=63, 3 seasons) vs ~50% mid-season — early-season stars may be under-priced by the shrink,
# but the sample is too small to hand-apply a bump. build_xtd(gl_stick=...) is there to retest.
COACH_CHANGE_SHRINK = 0.6     # new coach: pull RZ pass tendency this far toward league mean


# --------------------------------------------------------------------------- data
def load_pbp():
    return pd.read_parquet(f"{DATA}/pbp_2021_2025.parquet")


_PLAYERS = None
def load_players():
    global _PLAYERS
    if _PLAYERS is None:
        cache = f"{DATA}/players_cache.parquet"
        if os.path.exists(cache):
            _PLAYERS = pd.read_parquet(cache)
        else:
            import nflreadpy as nfl
            p = nfl.load_players().to_pandas()
            _PLAYERS = p[["gsis_id", "display_name", "position"]].dropna(subset=["gsis_id"])
            _PLAYERS.to_parquet(cache, index=False)
    return _PLAYERS


def _zone(yl):
    yl = np.asarray(yl, dtype=float)
    out = np.full(len(yl), "OPEN", dtype=object)
    out[yl <= 20] = "RZ"; out[yl <= 10] = "IN10"; out[yl <= 5] = "GL"
    return out


def _events(pbp, season=None):
    """Long table: one row per rush-or-target opportunity with zone, type, player, td."""
    d = pbp if season is None else pbp[pbp.season == season]
    d = d[(d.two_point_attempt == 0) & d.play_type.isin(["run", "pass"])].copy()
    d["zone"] = _zone(d["yardline_100"])
    rush = d[d.rush_attempt == 1][["season", "week", "game_id", "posteam", "zone", "rusher_player_id", "rush_touchdown"]]
    rush = rush.rename(columns={"rusher_player_id": "pid", "rush_touchdown": "td"}); rush["type"] = "rush"
    tgt = d[(d.pass_attempt == 1) & d.receiver_player_id.notna()][
        ["season", "week", "game_id", "posteam", "zone", "receiver_player_id", "pass_touchdown"]]
    tgt = tgt.rename(columns={"receiver_player_id": "pid", "pass_touchdown": "td"}); tgt["type"] = "target"
    ev = pd.concat([rush, tgt], ignore_index=True).dropna(subset=["pid"])
    ev["td"] = ev["td"].fillna(0).astype(int)
    return ev


# --------------------------------------------------------------------------- team / coach layer
def team_zone_profile(ev, pbp_season):
    """Per team: plays per game in each zone, and pass (target) rate within each zone.
    This is the coach's red-zone / goal-line play-calling signature."""
    games = ev.groupby("posteam")["game_id"].nunique().rename("games")
    z = ev.groupby(["posteam", "zone"]).agg(plays=("pid", "size"), targets=("type", lambda s: (s == "target").sum()))
    z = z.reset_index().merge(games, on="posteam")
    z["plays_pg"] = z["plays"] / z["games"]
    z["pass_rate"] = z["targets"] / z["plays"].clip(lower=1)
    return z


def league_zone_profile(tzp):
    lg = tzp.groupby("zone").apply(lambda g: pd.Series({
        "plays_pg": g["plays"].sum() / g["games"].sum(),
        "pass_rate": g["targets"].sum() / max(g["plays"].sum(), 1)})).reset_index()
    return lg.set_index("zone")


def coach_map(season):
    s = pd.read_parquet(f"{DATA}/schedules.parquet")
    s = s[s.season == season]
    m = {}
    for _, r in s.iterrows():
        m[r["home_team"]] = r.get("home_coach"); m[r["away_team"]] = r.get("away_coach")
    return m


# --------------------------------------------------------------------------- player layer
HEALTHY_LEAD_MIN = {"rush": 3, "target": 2}   # a core player "played" if he had this many opps
HEALTHY_CORE = 2                                # the group's core = top-2 by opps per game (>= 4 games)


def healthy_mask(ev):
    """Boolean per event row: were the team's CORE players of that type (top-2 by opportunities
    per game, min 4 games) all active in that game? A backup's share measured while the starter
    was hurt is stale the moment the starter returns (Vidal 2025: 150 carries in LAC weeks 6-13
    with Hampton out; 5-14/game beside him), and the season TOTAL can even make the fill-in
    look like the lead. Shares are measured on the healthy roster; the injury layer supplies the
    cascade forward when a starter IS out — so the two never double count."""
    pg = ev.groupby(["season", "posteam", "type", "pid", "game_id"]).size().reset_index(name="n")
    rate = pg.groupby(["season", "posteam", "type", "pid"]).agg(g=("game_id", "nunique"), n=("n", "sum")).reset_index()
    rate = rate[rate.g >= 4]; rate["pg"] = rate.n / rate.g
    core = (rate.sort_values("pg", ascending=False).groupby(["season", "posteam", "type"]).head(HEALTHY_CORE)
                .rename(columns={"pid": "core_pid"})[["season", "posteam", "type", "core_pid"]])
    # per game: did every core player reach the threshold?
    thr = pg["type"].map(HEALTHY_LEAD_MIN)
    played = pg[pg.n >= thr][["season", "posteam", "type", "pid", "game_id"]].rename(columns={"pid": "core_pid"})
    cg = core.merge(played, on=["season", "posteam", "type", "core_pid"], how="left")
    ncore = core.groupby(["season", "posteam", "type"]).size().rename("n_core")
    present = cg.dropna(subset=["game_id"]).groupby(["season", "posteam", "type", "game_id"]).size().rename("n_present").reset_index()
    present = present.merge(ncore.reset_index(), on=["season", "posteam", "type"])
    ok_games = present[present.n_present >= present.n_core][["season", "posteam", "type", "game_id"]]
    ok_games["_ok"] = True
    e = ev.merge(ok_games, on=["season", "posteam", "type", "game_id"], how="left")
    return e["_ok"].fillna(False).values


def gl_inclusion(season, max_week=None):
    """{(pid, team): goal-line snap inclusion} and {team: RB share of GL carries} for a season (to a week)."""
    p = os.path.join(DATA, "universe_features_player_game.parquet")
    if not os.path.exists(p):
        return {}, {}
    u = pd.read_parquet(p, columns=["pid", "season", "week", "posteam", "position", "gl_snaps", "team_gl", "gl_carries", "team_gl_carries"])
    u = u[u.season == season]
    if max_week is not None:
        u = u[u.week <= max_week]
    if u.empty:
        return {}, {}
    g = u.groupby(["pid", "posteam"]).agg(gl_snaps=("gl_snaps", "sum"), team_gl=("team_gl", "sum")).reset_index()
    g["incl"] = g.gl_snaps / g.team_gl.replace(0, np.nan)
    incl = {(r.pid, r.posteam): float(r.incl) for r in g.itertuples() if r.incl == r.incl}
    rb = u[u.position.isin(["RB", "FB"])].groupby("posteam")["gl_carries"].sum()
    tot = u.drop_duplicates(["posteam", "week"]).groupby("posteam")["team_gl_carries"].sum()
    rb_frac = {t: float(rb.get(t, 0) / tot[t]) for t in tot.index if tot[t] and tot[t] == tot[t]}
    return incl, rb_frac


def player_zone_shares(ev, healthy_only=False, unhealthy_weight=1.0):
    """Per (player, team): share of the team's plays of each type in each zone, plus games.
    A goal-line vulture shows up as a non-lead back holding GL rush share. healthy_only drops
    (or, with unhealthy_weight<1, down-weights) games where the group's core wasn't all active.
    TESTED, NOT ADOPTED (td_env_validation.json healthy_roster_shares): dropping them loses
    clearly out of sample (Brier .1666 vs .1624) — backups who got work keep getting some, and
    starters get hurt again; the raw season share is the better predictor on average."""
    if healthy_only and "season" in ev.columns:
        ok = healthy_mask(ev)
        ev = ev.assign(_w=np.where(ok, 1.0, float(unhealthy_weight)))
        ev = ev[ev._w > 0]
    else:
        ev = ev.assign(_w=1.0)
    tz = ev.groupby(["posteam", "zone", "type"])["_w"].sum().rename("team_n")
    pz = ev.groupby(["pid", "posteam", "zone", "type"]).agg(n=("_w", "sum"), tds=("td", "sum")).reset_index()
    pz = pz.merge(tz.reset_index(), on=["posteam", "zone", "type"])
    pz["share"] = pz["n"] / pz["team_n"].clip(lower=1e-9)
    # games of evidence behind the share = healthy games he appeared in for that type
    games = ev.groupby(["pid", "posteam", "type"])["game_id"].nunique().rename("games").reset_index()
    pz = pz.merge(games, on=["pid", "posteam", "type"])
    return pz


def conversion_priors(ev, players):
    """P(TD | opportunity) by position x zone x type — the finishing baseline."""
    e = ev.merge(players[["gsis_id", "position"]], left_on="pid", right_on="gsis_id", how="left")
    e = e[e.position.isin(POS_OK)]
    pr = e.groupby(["position", "zone", "type"]).agg(n=("td", "size"), tds=("td", "sum")).reset_index()
    pr["rate"] = (pr["tds"] + 1) / (pr["n"] + 2)      # light Laplace
    return pr.set_index(["position", "zone", "type"])["rate"].to_dict()


# --------------------------------------------------------------------------- the model
def build_xtd(data_seasons, target_season=None, current_team=None, env_mult=None,
              new_coach_teams=None, week1_mode=None, gl_stick=None, healthy_only=False, unhealthy_weight=1.0):
    """Return DataFrame[pid, name, position, team, xtd, xtd_rush, xtd_rec, gl_share, rz_tgt_share,
    coach_gl_pass_rate, vulture_share, conv_rush_gl, conv_tgt_rz, tds_last, xtd_last].

    data_seasons : seasons of play-by-play to learn shares/conversion from (latest weighted most)
    target_season: season we're predicting (for coach lookup / week-1 shrink)
    current_team : {norm_name -> team} overrides (offseason moves)
    env_mult     : {team -> multiplier} from today's market (implied points vs baseline)
    new_coach_teams: set of teams whose head coach changed (PROE flips)
    week1_mode   : True -> shrink GL shares by measured YoY stickiness (no current-season data)
    """
    pbp = load_pbp(); players = load_players()
    ev_all = _events(pbp[pbp.season.isin(data_seasons)])
    latest = max(data_seasons)
    ev_latest = ev_all[ev_all.season == latest]
    if week1_mode is None:
        week1_mode = (target_season is not None and target_season > latest)

    tzp = team_zone_profile(ev_latest, latest)
    lg = league_zone_profile(tzp)
    conv = conversion_priors(ev_all, players)
    shares = player_zone_shares(ev_latest, healthy_only=healthy_only, unhealthy_weight=unhealthy_weight)
    gl_incl, rb_frac = gl_inclusion(latest)
    shares = shares.merge(players[["gsis_id", "position", "display_name"]], left_on="pid", right_on="gsis_id", how="left")
    shares = shares[shares.position.isin(POS_OK)]

    # player-level conversion (shrunk to position x zone x type prior)
    def conv_rate(pos, zone, typ, n, tds):
        prior = conv.get((pos, zone, typ), 0.02)
        return (tds + CONV_PSEUDO_OPPS * prior) / (n + CONV_PSEUDO_OPPS)

    env_mult = env_mult or {}
    new_coach_teams = set(new_coach_teams or [])
    current_team = current_team or {}

    rows = []
    for (pid, team), g in shares.groupby(["pid", "posteam"]):
        pos = g["position"].iloc[0]; name = g["display_name"].iloc[0]
        games = float(g["games"].max())
        tm = current_team.get(norm_name(name), team)       # play for whoever he plays for NOW
        mult = float(env_mult.get(tm, 1.0))
        xtd_rush = xtd_rec = 0.0
        feats = {}
        for zone in ZONES:
            trow = tzp[(tzp.posteam == tm) & (tzp.zone == zone)]
            plays_pg = float(trow["plays_pg"].iloc[0]) if len(trow) else float(lg.loc[zone, "plays_pg"])
            pass_rate = float(trow["pass_rate"].iloc[0]) if len(trow) else float(lg.loc[zone, "pass_rate"])
            # coach change -> tendency flips; pull toward league
            if tm in new_coach_teams:
                pass_rate = (1 - COACH_CHANGE_SHRINK) * pass_rate + COACH_CHANGE_SHRINK * float(lg.loc[zone, "pass_rate"])
            # a little shrink of team zone volume toward league (small samples, roster churn)
            plays_pg = 0.7 * plays_pg + 0.3 * float(lg.loc[zone, "plays_pg"])
            plays_pg *= mult                                   # today's environment
            for typ in TYPES:
                r = g[(g.zone == zone) & (g.type == typ)]
                n = float(r["n"].iloc[0]) if len(r) else 0.0
                tds = int(r["tds"].iloc[0]) if len(r) else 0
                share_obs = float(r["share"].iloc[0]) if len(r) else 0.0
                # share shrinkage: toward his own overall share of that type (role), by games
                own_type = g[g.type == typ]
                own_share = float(own_type["n"].sum() / max(own_type["team_n"].sum(), 1)) if len(own_type) else 0.0
                g_typ = float(r["games"].iloc[0]) if len(r) else games
                w = g_typ / (g_typ + SHARE_PSEUDO_GAMES)
                share = w * share_obs + (1 - w) * own_share
                # goal-line PACKAGE INCLUSION (measured, see GL_INCLUSION_BLEND): blend the sparse
                # carry share with "was he on the field for the goal-line snaps"
                if zone == "GL" and typ == "rush" and pos in ("RB", "FB"):
                    gi = gl_incl.get((pid, team))
                    if gi is not None:
                        share = GL_INCLUSION_BLEND * share + (1 - GL_INCLUSION_BLEND) * gi * rb_frac.get(team, 0.75)
                        feats["GL_inclusion"] = round(gi, 3)
                # goal-line role is only ~29% sticky year over year: in week-1 mode, regress the
                # GL share hard toward his general rush role (the honest prior for a new season)
                if week1_mode and zone == "GL" and typ == "rush":
                    st = GL_YOY_STICKINESS if gl_stick is None else float(gl_stick)
                    share = st * share + (1 - st) * own_share
                p_type = pass_rate if typ == "target" else (1 - pass_rate)
                opps = plays_pg * p_type * share
                c = conv_rate(pos, zone, typ, n, tds)
                contrib = opps * c
                if typ == "rush": xtd_rush += contrib
                else: xtd_rec += contrib
                feats[f"{zone}_{typ}_share"] = round(share, 3)
                feats[f"{zone}_{typ}_conv"] = round(c, 3)
                feats[f"{zone}_{typ}_opps"] = round(opps, 3)
        xtd = xtd_rush + xtd_rec
        # vulture: how much GL rush share on THIS team sits with someone else
        team_gl = shares[(shares.posteam == team) & (shares.zone == "GL") & (shares.type == "rush")]
        lead_gl = float(team_gl["share"].max()) if len(team_gl) else 0.0
        my_gl = feats.get("GL_rush_share", 0.0)
        vult = (lead_gl - my_gl) if pos in ("RB", "FB") and my_gl >= lead_gl - 1e-9 else \
               (1.0 - my_gl if pos in ("RB", "FB") else 0.0)
        tds_last = int(g["tds"].sum())
        rows.append({"pid": pid, "name": name, "position": pos, "team": tm, "games": games,
                     "xtd": round(xtd, 3), "xtd_rush": round(xtd_rush, 3), "xtd_rec": round(xtd_rec, 3),
                     "gl_share": feats.get("GL_rush_share", 0.0),
                     "rz_tgt_share": round(np.mean([feats.get("GL_target_share", 0), feats.get("IN10_target_share", 0),
                                                    feats.get("RZ_target_share", 0)]), 3),
                     "vulture_share": round(1.0 - my_gl, 3) if pos in ("RB", "FB") else None,
                     "conv_rush_gl": feats.get("GL_rush_conv"), "conv_tgt_rz": feats.get("RZ_target_conv"),
                     "coach_gl_pass_rate": round(float(tzp[(tzp.posteam == tm) & (tzp.zone == "GL")]["pass_rate"].iloc[0])
                                                 if len(tzp[(tzp.posteam == tm) & (tzp.zone == "GL")]) else float(lg.loc["GL", "pass_rate"]), 3),
                     "tds_last": tds_last, "xtd_last": round(xtd / max(mult, 1e-6), 3) * 1.0,
                     "env_mult": round(mult, 3), **feats})
    df = pd.DataFrame(rows)
    df["p_atd_raw"] = 1 - np.exp(-df["xtd"])
    return df.sort_values("xtd", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- calibration
# Fitted on the 2023->2024 walk-forward fold (see data/xtd_validation.json). Raw Poisson is
# over-dispersed (the low bucket under-predicts, the top over-predicts); this compresses it.
# linear recalibration, refit WITH the environment multiplier applied (td_env_validate.py:
# --final: fit 2022->23 + 2023->24, test 2024->25; top bucket pred .50 / hit .54). The earlier (0.084, 0.812)
# was fit at env=1 and then applied under a mis-scaled multiplier -> stars under-priced.
CAL = {"a": 0.085, "b": 0.741}
def p_atd(xtd):
    raw = 1 - np.exp(-np.asarray(xtd, dtype=float))
    return np.clip(CAL["a"] + CAL["b"] * raw, 0.02, 0.85)


# Week-1 blend with the old pooled model (they're equal with prior-season data only), ramping
# toward xTD as current-season games accrue — where it wins by 2-3% Brier, replicated.
def blend_weight(current_season_weeks_played):
    w = float(current_season_weeks_played or 0)
    return float(np.clip(0.55 + 0.075 * w, 0.55, 0.85))   # wk0 .55 -> wk4 .85


if __name__ == "__main__":
    df = build_xtd([2024, 2025], target_season=2026)
    cols = ["name", "position", "team", "games", "xtd", "xtd_rush", "xtd_rec", "gl_share", "rz_tgt_share",
            "vulture_share", "coach_gl_pass_rate", "tds_last", "p_atd_raw"]
    print(df[cols].head(25).to_string())
