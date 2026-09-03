"""TD confidence — is goal-line PACKAGE INCLUSION (participation: who is on the field inside
the 5) a better read of a back's goal-line role than his goal-line CARRY share?

Carries inside the 5 are sparse (2-3 team plays a game); a back's carry share swings with a
handful of plays, while "was he on the field for the goal-line snaps" is observed every play.
Test (in-season rolling, the way xTD was validated): weeks 4-18 of 2024 and 2025, this season's
prior weeks predict whether he scores THIS week; the only change is the GL rush share for
backs: carry share (current) vs snap-inclusion share vs a blend. Brier / log-loss, lower wins.

    python3 src/td_role_validate.py        -> data/td_role_validation.json
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xtd_model as xm
import xtd_validate as xv

DATA = xm.DATA
BLENDS = [1.0, 0.75, 0.5, 0.25, 0.0]      # weight on the carry share (1.0 = today's model)


def gl_snap_table():
    u = pd.read_parquet(f"{DATA}/universe_features_player_game.parquet",
                        columns=["pid", "game_id", "season", "week", "posteam", "position", "gl_snaps", "team_gl", "gl_carries", "team_gl_carries"])
    return u


def rolling(season, min_week=4, blends=BLENDS, sneak_adj=True, pbp=None, players=None):
    """One pass per week; every blend priced from the same share tables. Returns {blend: DataFrame}."""
    pbp = pbp if pbp is not None else xm.load_pbp(); players = players if players is not None else xm.load_players()
    u = gl_snap_table(); u = u[u.season == season]
    conv = xm.conversion_priors(xm._events(pbp[pbp.season < season]), players)
    ev_season = xm._events(pbp[pbp.season == season])
    rows = {b: [] for b in blends}
    for wk in range(min_week, 19):
        ev_past = ev_season[ev_season.week < wk] if "week" in ev_season.columns else xm._events(pbp[(pbp.season == season) & (pbp.week < wk)])
        if ev_past.empty: continue
        tzp = xm.team_zone_profile(ev_past, season); lg = xm.league_zone_profile(tzp)
        sh = xm.player_zone_shares(ev_past).merge(players[["gsis_id", "position"]], left_on="pid", right_on="gsis_id", how="left")
        sh = sh[sh.position.isin(xm.POS_OK)]
        up = u[u.week < wk].groupby(["pid", "posteam"]).agg(gl_snaps=("gl_snaps", "sum"), team_gl=("team_gl", "sum")).reset_index()
        up["gl_incl"] = up.gl_snaps / up.team_gl.replace(0, np.nan)
        incl = {(r.pid, r.posteam): r.gl_incl for r in up.itertuples()}
        gl_ev = ev_past[(ev_past.zone == "GL") & (ev_past.type == "rush")].merge(players[["gsis_id", "position"]], left_on="pid", right_on="gsis_id", how="left")
        rb_frac = gl_ev.groupby("posteam")["position"].apply(lambda s: s.isin(["RB", "FB"]).mean()).to_dict()
        tz = {(r.posteam, r.zone): (float(r.plays_pg), float(r.pass_rate)) for r in tzp.itertuples()}
        x_p = {b: {} for b in blends}
        for (pid, team), g in sh.groupby(["pid", "posteam"]):
            pos = g["position"].iloc[0]; games = float(g["games"].iloc[0])
            lam = {b: 0.0 for b in blends}
            gi = incl.get((pid, team))
            for zone in xm.ZONES:
                ppg, pr = tz.get((team, zone), (float(lg.loc[zone, "plays_pg"]), float(lg.loc[zone, "pass_rate"])))
                ppg = 0.7 * ppg + 0.3 * float(lg.loc[zone, "plays_pg"])
                for typ in xm.TYPES:
                    r = g[(g.zone == zone) & (g.type == typ)]
                    n = int(r["n"].iloc[0]) if len(r) else 0; tds = int(r["tds"].iloc[0]) if len(r) else 0
                    so = float(r["share"].iloc[0]) if len(r) else 0.0
                    own = g[g.type == typ]; own_s = float(own["n"].sum() / max(own["team_n"].sum(), 1)) if len(own) else 0.0
                    w = games / (games + xm.SHARE_PSEUDO_GAMES); share = w * so + (1 - w) * own_s
                    prior = conv.get((pos, zone, typ), 0.02)
                    c = (tds + xm.CONV_PSEUDO_OPPS * prior) / (n + xm.CONV_PSEUDO_OPPS)
                    base = ppg * (pr if typ == "target" else 1 - pr) * c
                    for b in blends:
                        sh_b = share
                        if zone == "GL" and typ == "rush" and pos in ("RB", "FB") and gi is not None and gi == gi:
                            snap_share = gi * (rb_frac.get(team, 0.75) if sneak_adj else 1.0)
                            sh_b = b * share + (1 - b) * snap_share
                        lam[b] += base * sh_b
            for b in blends:
                x_p[b][pid] = 1 - np.exp(-lam[b])
        this0 = xv.realized_player_games(ev_season[ev_season.week == wk])
        this0 = this0[this0["pid"].isin(set(x_p[blends[0]]))].merge(players[["gsis_id", "position"]], left_on="pid", right_on="gsis_id", how="left")
        for b in blends:
            t = this0.copy(); t["p_x"] = t["pid"].map(x_p[b]); t["week"] = wk; rows[b].append(t)
    return {b: pd.concat(v, ignore_index=True) for b, v in rows.items()}


def run():
    out = {"design": "in-season rolling weeks 4-18; GL rush share for backs = blend × carry share + (1−blend) × goal-line snap inclusion × team RB share of GL carries",
           "by_season": {}}
    pbp = xm.load_pbp(); players = xm.load_players()
    for season in (2024, 2025):
        res = {}
        allb = rolling(season, pbp=pbp, players=players)
        for b in BLENDS:
            pg = allb[b]
            y = pg["scored"].values; p = np.clip(pg["p_x"].values, 0.01, 0.99)
            backs = pg.position.isin(["RB", "FB"]).values
            res[str(b)] = {"n": int(len(pg)), "brier": round(float(np.mean((p - y) ** 2)), 5), "logloss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 5),
                           "brier_backs": round(float(np.mean((p[backs] - y[backs]) ** 2)), 5), "n_backs": int(backs.sum())}
            print(f"  {season} blend {b:.2f}: brier {res[str(b)]['brier']:.5f} (backs {res[str(b)]['brier_backs']:.5f}, n={res[str(b)]['n_backs']})  logloss {res[str(b)]['logloss']:.5f}")
        out["by_season"][str(season)] = res
    # verdict: does any blend < 1 beat the current model on BOTH seasons (backs' Brier)?
    best = None
    for b in BLENDS[1:]:
        wins = all(out["by_season"][s][str(b)]["brier_backs"] < out["by_season"][s]["1.0"]["brier_backs"] - 1e-6 for s in ("2024", "2025"))
        if wins:
            gain = sum(out["by_season"][s]["1.0"]["brier_backs"] - out["by_season"][s][str(b)]["brier_backs"] for s in ("2024", "2025"))
            if best is None or gain > best[1]: best = (b, gain)
    out["verdict"] = {"winner_blend": best[0] if best else None, "gain_brier_backs_sum": round(best[1], 5) if best else None,
                      "result": ("WINS: snap inclusion improves the goal-line role read" if best else "WASH/LOSES: carry share stays")}
    json.dump(out, open(f"{DATA}/td_role_validation.json", "w"), indent=1)
    print("verdict:", out["verdict"])
    return out


if __name__ == "__main__":
    run()
