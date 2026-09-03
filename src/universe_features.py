"""Derived tables from the data universe — the raw feeds turned into per-player-game and
per-team-game signals the labs can test (data/universe_features_<kind>.parquet).

  player_game  (pid, game_id, season, week, team, opp):
     snap_pct, offense_snaps                          <- PFR snap counts
     routes, dropbacks, route_rate                    <- participation × pbp (on the field on a dropback)
     gl_snaps, team_gl_plays, gl_snap_share           <- participation × pbp (on the field inside the 5)
     in10_snap_share, rz_snap_share                   <- same, inside the 10 / 20
     p11_share                                         <- share of his snaps in 11 personnel
     tgt, routes -> tprr                               <- targets per route run
     ngs_*  separation, cushion, iay_share, yac_oe, ryoe_att, box8_pct, efficiency, ttt, cpoe, aggr  <- Next Gen Stats
     pfr_*  ybc_att, yac_att, broken_tackles, drops, pressures, pressured_pct                          <- PFR advanced
     ftn_*  catchable_rate, contested_rate, drop_rate (on his targets)                                <- FTN charting
     target_share, air_yards_share, wopr, racr, epa_per_target, epa_per_carry                          <- nflverse stats
  team_game    (team, game_id, season, week):
     pa_rate, motion_rate, rpo_rate, screen_rate, no_huddle_rate, shotgun_rate                        <- FTN (offense)
     def_man_rate, def_zone_rate, def_box_avg, def_pressure_rate, def_blitz_rate, def_rushers_avg      <- participation/FTN (defense)
     rz_trips, gl_plays, plays, dropbacks, pass_rate                                                   <- pbp

    python3 src/universe_features.py            # builds both tables for 2021-2025
"""
import os, sys, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data"); UNI = os.path.join(DATA, "universe")
SEASONS = list(range(2021, 2026))
PBP_COLS = ["game_id", "play_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "qb_dropback", "yardline_100",
            "rush_attempt", "pass_attempt", "rusher_player_id", "receiver_player_id", "passer_player_id", "epa", "touchdown", "drive",
            "fixed_drive", "shotgun", "no_huddle", "qb_scramble", "sack"]


def _pbp(season):
    p = os.path.join(UNI, f"pbp_{season}.parquet")
    if not os.path.exists(p):
        return None
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(p).schema.names)
    d = pd.read_parquet(p, columns=[c for c in PBP_COLS if c in have])
    d = d[d.season_type == "REG"] if "season_type" in d.columns else d
    return d


def _players():
    pl = pd.read_parquet(os.path.join(UNI, "players.parquet"))
    cols = {c: c for c in pl.columns}
    gsis = "gsis_id" if "gsis_id" in pl.columns else "player_id"
    return pl, gsis


def build_player_game(seasons=SEASONS, verbose=True):
    pl, gsis = _players()
    pfr2gsis = dict(zip(pl["pfr_id"], pl[gsis])) if "pfr_id" in pl.columns else {}
    pos_of = dict(zip(pl[gsis], pl["position"])) if "position" in pl.columns else {}
    out = []
    for s in seasons:
        pbp = _pbp(s)
        if pbp is None:
            continue
        pbp = pbp[pbp.play_type.isin(["run", "pass"]) | (pbp.qb_dropback == 1)]
        # --- participation: who was on the field for each play -------------------------------
        part_p = os.path.join(UNI, f"participation_{s}.parquet")
        rows = {}
        if os.path.exists(part_p):
            part = pd.read_parquet(part_p, columns=["nflverse_game_id", "play_id", "possession_team", "offense_personnel", "offense_players"])
            m = pbp.merge(part, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "play_id"], how="inner")
            m = m[m.offense_players.notna() & (m.offense_players != "")]
            m["is_db"] = (m.qb_dropback == 1).astype(int); m["is_gl"] = (m.yardline_100 <= 5).astype(int)
            m["is_in10"] = (m.yardline_100 <= 10).astype(int); m["is_rz"] = (m.yardline_100 <= 20).astype(int)
            m["is_11"] = m.offense_personnel.fillna("").str.contains("1 RB").astype(int) & m.offense_personnel.fillna("").str.contains("1 TE").astype(int)
            team_tot = m.groupby(["game_id", "posteam"]).agg(team_plays=("play_id", "size"), team_db=("is_db", "sum"), team_gl=("is_gl", "sum"),
                                                              team_in10=("is_in10", "sum"), team_rz=("is_rz", "sum")).reset_index()
            ex = m[["game_id", "season", "week", "posteam", "defteam", "offense_players", "is_db", "is_gl", "is_in10", "is_rz", "is_11"]].copy()
            ex["pid"] = ex.offense_players.str.split(";"); ex = ex.explode("pid")
            ex = ex[ex.pid.map(lambda x: pos_of.get(x) in ("RB", "WR", "TE", "FB", "QB"))]
            pg = ex.groupby(["pid", "game_id", "season", "week", "posteam", "defteam"]).agg(
                snaps_part=("is_db", "size"), routes=("is_db", "sum"), gl_snaps=("is_gl", "sum"), in10_snaps=("is_in10", "sum"),
                rz_snaps=("is_rz", "sum"), p11=("is_11", "sum")).reset_index()
            pg = pg.merge(team_tot, on=["game_id", "posteam"], how="left")
            pg["route_rate"] = pg.routes / pg.team_db.replace(0, np.nan)
            pg["gl_snap_share"] = pg.gl_snaps / pg.team_gl.replace(0, np.nan)
            pg["in10_snap_share"] = pg.in10_snaps / pg.team_in10.replace(0, np.nan)
            pg["rz_snap_share"] = pg.rz_snaps / pg.team_rz.replace(0, np.nan)
            pg["p11_share"] = pg.p11 / pg.snaps_part.replace(0, np.nan)
            pg["snap_share_part"] = pg.snaps_part / pg.team_plays.replace(0, np.nan)
            for r in pg.to_dict("records"):
                rows[(r["pid"], r["game_id"])] = r
        # --- targets / carries from pbp (for TPRR and shares) -----------------------------------
        tg = pbp[pbp.pass_attempt == 1].groupby(["receiver_player_id", "game_id"]).size().rename("tgt")
        ca = pbp[pbp.rush_attempt == 1].groupby(["rusher_player_id", "game_id"]).size().rename("carries")
        gl_c = pbp[(pbp.rush_attempt == 1) & (pbp.yardline_100 <= 5)].groupby(["rusher_player_id", "game_id"]).size().rename("gl_carries")
        gl_t = pbp[(pbp.pass_attempt == 1) & (pbp.yardline_100 <= 10)].groupby(["receiver_player_id", "game_id"]).size().rename("in10_targets")
        team_gl_c = pbp[(pbp.rush_attempt == 1) & (pbp.yardline_100 <= 5)].groupby(["posteam", "game_id"]).size().rename("team_gl_carries")
        team_in10_t = pbp[(pbp.pass_attempt == 1) & (pbp.yardline_100 <= 10)].groupby(["posteam", "game_id"]).size().rename("team_in10_targets")
        base = pd.concat([tg.rename_axis(["pid", "game_id"]), ca.rename_axis(["pid", "game_id"]), gl_c.rename_axis(["pid", "game_id"]),
                          gl_t.rename_axis(["pid", "game_id"])], axis=1).reset_index()
        meta = pbp.drop_duplicates("game_id")[["game_id", "season", "week"]]
        base = base.merge(meta, on="game_id", how="left")
        # team for the player-game: from rows (participation) or from the play
        pteam = pd.concat([pbp[pbp.pass_attempt == 1][["receiver_player_id", "game_id", "posteam", "defteam"]].rename(columns={"receiver_player_id": "pid"}),
                           pbp[pbp.rush_attempt == 1][["rusher_player_id", "game_id", "posteam", "defteam"]].rename(columns={"rusher_player_id": "pid"})]).dropna().drop_duplicates(["pid", "game_id"])
        base = base.merge(pteam, on=["pid", "game_id"], how="left")
        base = base.merge(team_gl_c.reset_index(), on=["posteam", "game_id"], how="left").merge(team_in10_t.reset_index(), on=["posteam", "game_id"], how="left")
        pgdf = pd.DataFrame(list(rows.values())) if rows else pd.DataFrame(columns=["pid", "game_id"])
        df = base.merge(pgdf.drop(columns=[c for c in ("season", "week", "posteam", "defteam") if c in pgdf.columns]), on=["pid", "game_id"], how="outer")
        for c in ("season", "week"):
            if c in df.columns and df[c].isna().any():
                df[c] = df[c].fillna(df["game_id"].map(dict(zip(meta.game_id, meta[c]))))
        df["tprr"] = df.tgt / df.routes.replace(0, np.nan)
        df["gl_carry_share"] = df.gl_carries / df.team_gl_carries.replace(0, np.nan)
        df["in10_target_share"] = df.in10_targets / df.team_in10_targets.replace(0, np.nan)
        # --- snap counts (PFR) ------------------------------------------------------------------
        sp = os.path.join(UNI, f"snap_counts_{s}.parquet")
        if os.path.exists(sp):
            sc = pd.read_parquet(sp, columns=["game_id", "pfr_player_id", "offense_snaps", "offense_pct", "game_type"])
            sc = sc[sc.game_type == "REG"].copy(); sc["pid"] = sc.pfr_player_id.map(pfr2gsis)
            sc = sc.dropna(subset=["pid"]).drop_duplicates(["pid", "game_id"])[["pid", "game_id", "offense_snaps", "offense_pct"]]
            df = df.merge(sc, on=["pid", "game_id"], how="left")
        # --- Next Gen Stats (player-week) --------------------------------------------------------
        for kind, cols in (("receiving", {"avg_separation": "ngs_sep", "avg_cushion": "ngs_cushion", "percent_share_of_intended_air_yards": "ngs_iay_share",
                                          "avg_yac_above_expectation": "ngs_yac_oe", "avg_intended_air_yards": "ngs_adot", "catch_percentage": "ngs_catch_pct"}),
                           ("rushing", {"rush_yards_over_expected_per_att": "ngs_ryoe_att", "percent_attempts_gte_eight_defenders": "ngs_box8_pct",
                                        "efficiency": "ngs_efficiency", "avg_time_to_los": "ngs_ttlos"}),
                           ("passing", {"avg_time_to_throw": "ngs_ttt", "completion_percentage_above_expectation": "ngs_cpoe", "aggressiveness": "ngs_aggr",
                                        "avg_air_yards_differential": "ngs_ayd", "avg_intended_air_yards": "ngs_iay_qb"})):
            p = os.path.join(UNI, f"ngs_{kind}.parquet")
            if not os.path.exists(p): continue
            ng = pd.read_parquet(p); ng = ng[(ng.season == s) & (ng.season_type == "REG") & (ng.week > 0)]
            ng = ng.rename(columns=cols)[["player_gsis_id", "week"] + list(cols.values())].rename(columns={"player_gsis_id": "pid"})
            df = df.merge(ng, on=["pid", "week"], how="left")
        # --- PFR advanced (player-game) ----------------------------------------------------------
        for kind, cols in (("rush", {"rushing_yards_before_contact_avg": "pfr_ybc_att", "rushing_yards_after_contact_avg": "pfr_yac_att", "rushing_broken_tackles": "pfr_bt_rush"}),
                           ("rec", {"receiving_drop": "pfr_drops", "receiving_broken_tackles": "pfr_bt_rec", "receiving_rat": "pfr_rat_when_targeted"}),
                           ("pass", {"times_pressured_pct": "pfr_pressured_pct", "passing_bad_throw_pct": "pfr_bad_throw_pct", "times_blitzed": "pfr_blitzed", "times_pressured": "pfr_pressured"})):
            p = os.path.join(UNI, f"pfr_{kind}_{s}.parquet")
            if not os.path.exists(p): continue
            pf = pd.read_parquet(p); pf = pf[pf.game_type == "REG"].copy(); pf["pid"] = pf.pfr_player_id.map(pfr2gsis)
            pf = pf.dropna(subset=["pid"]).rename(columns=cols)[["pid", "game_id"] + list(cols.values())].drop_duplicates(["pid", "game_id"])
            df = df.merge(pf, on=["pid", "game_id"], how="left")
        # --- FTN charting on his targets -----------------------------------------------------------
        fp = os.path.join(UNI, f"ftn_charting_{s}.parquet")
        if os.path.exists(fp):
            ft = pd.read_parquet(fp, columns=["nflverse_game_id", "nflverse_play_id", "is_catchable_ball", "is_contested_ball", "is_drop", "is_play_action", "is_screen_pass", "is_rpo", "is_motion"])
            tt = pbp[pbp.pass_attempt == 1][["game_id", "play_id", "receiver_player_id"]].merge(ft, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"], how="inner")
            agg = tt.groupby(["receiver_player_id", "game_id"]).agg(ftn_targets=("play_id", "size"), ftn_catchable=("is_catchable_ball", "mean"),
                                                                    ftn_contested=("is_contested_ball", "mean"), ftn_drop=("is_drop", "mean"),
                                                                    ftn_screen_share=("is_screen_pass", "mean")).reset_index().rename(columns={"receiver_player_id": "pid"})
            df = df.merge(agg, on=["pid", "game_id"], how="left")
        # --- nflverse weekly stats shares ------------------------------------------------------------
        stp = os.path.join(UNI, f"stats_player_{s}.parquet")
        if os.path.exists(stp):
            st = pd.read_parquet(stp, columns=["player_id", "week", "season_type", "target_share", "air_yards_share", "wopr", "racr", "receiving_epa", "rushing_epa", "targets", "carries", "receiving_air_yards"])
            st = st[st.season_type == "REG"].rename(columns={"player_id": "pid"})
            st["epa_per_target"] = st.receiving_epa / st.targets.replace(0, np.nan); st["epa_per_carry"] = st.rushing_epa / st.carries.replace(0, np.nan)
            st["adot"] = st.receiving_air_yards / st.targets.replace(0, np.nan)
            st = st[["pid", "week", "target_share", "air_yards_share", "wopr", "racr", "epa_per_target", "epa_per_carry", "adot"]].drop_duplicates(["pid", "week"])
            df = df.merge(st, on=["pid", "week"], how="left")
        df["position"] = df.pid.map(pos_of)
        out.append(df)
        if verbose: print(f"  player-game {s}: {len(df):,} rows, {df.shape[1]} cols")
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(os.path.join(DATA, "universe_features_player_game.parquet"), index=False)
    return res


def build_team_game(seasons=SEASONS, verbose=True):
    out = []
    for s in seasons:
        pbp = _pbp(s)
        if pbp is None: continue
        pl = pbp[pbp.play_type.isin(["run", "pass"]) | (pbp.qb_dropback == 1)].copy()
        pl["is_db"] = (pl.qb_dropback == 1).astype(int); pl["is_gl"] = (pl.yardline_100 <= 5).astype(int); pl["is_rz"] = (pl.yardline_100 <= 20).astype(int)
        off = pl.groupby(["posteam", "game_id", "season", "week"]).agg(plays=("play_id", "size"), dropbacks=("is_db", "sum"), gl_plays=("is_gl", "sum"),
                                                                       shotgun_rate=("shotgun", "mean"), no_huddle_rate=("no_huddle", "mean")).reset_index().rename(columns={"posteam": "team"})
        off["pass_rate"] = off.dropbacks / off.plays.replace(0, np.nan)
        rz = pl[pl.is_rz == 1].drop_duplicates(["game_id", "posteam", "fixed_drive"]).groupby(["posteam", "game_id"]).size().rename("rz_trips").reset_index().rename(columns={"posteam": "team"})
        off = off.merge(rz, on=["team", "game_id"], how="left"); off["rz_trips"] = off.rz_trips.fillna(0)
        # FTN offense + defense rates
        fp = os.path.join(UNI, f"ftn_charting_{s}.parquet")
        if os.path.exists(fp):
            ft = pd.read_parquet(fp, columns=["nflverse_game_id", "nflverse_play_id", "is_play_action", "is_motion", "is_rpo", "is_screen_pass", "n_blitzers", "n_pass_rushers", "n_defense_box"])
            m = pl[["game_id", "play_id", "posteam", "defteam", "is_db", "rush_attempt"]].merge(ft, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"], how="inner")
            fo = m.groupby(["posteam", "game_id"]).agg(pa_rate=("is_play_action", "mean"), motion_rate=("is_motion", "mean"), rpo_rate=("is_rpo", "mean"),
                                                        screen_rate=("is_screen_pass", "mean")).reset_index().rename(columns={"posteam": "team"})
            db = m[m.is_db == 1]
            fd = db.groupby(["defteam", "game_id"]).agg(def_blitz_rate=("n_blitzers", lambda x: (x > 0).mean()), def_rushers_avg=("n_pass_rushers", "mean")).reset_index().rename(columns={"defteam": "team"})
            ru = m[m.rush_attempt == 1].groupby(["defteam", "game_id"]).agg(def_box_avg_ftn=("n_defense_box", "mean")).reset_index().rename(columns={"defteam": "team"})
            off = off.merge(fo, on=["team", "game_id"], how="left").merge(fd, on=["team", "game_id"], how="left").merge(ru, on=["team", "game_id"], how="left")
        # participation defense: man/zone, box, pressure
        pp = os.path.join(UNI, f"participation_{s}.parquet")
        if os.path.exists(pp):
            part = pd.read_parquet(pp, columns=["nflverse_game_id", "play_id", "defenders_in_box", "was_pressure", "defense_man_zone_type", "defense_coverage_type", "time_to_throw"])
            m = pl[["game_id", "play_id", "posteam", "defteam", "is_db", "rush_attempt"]].merge(part, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "play_id"], how="inner")
            db = m[m.is_db == 1].copy()
            db["man"] = db.defense_man_zone_type.fillna("").str.upper().str.contains("MAN").astype(float)
            db.loc[db.defense_man_zone_type.fillna("") == "", "man"] = np.nan
            dd = db.groupby(["defteam", "game_id"]).agg(def_man_rate=("man", "mean"), def_pressure_rate=("was_pressure", "mean"), def_ttt_allowed=("time_to_throw", "mean")).reset_index().rename(columns={"defteam": "team"})
            rb = m[m.rush_attempt == 1].groupby(["defteam", "game_id"]).agg(def_box_avg=("defenders_in_box", "mean")).reset_index().rename(columns={"defteam": "team"})
            oo = db.groupby(["posteam", "game_id"]).agg(off_pressure_allowed=("was_pressure", "mean"), off_ttt=("time_to_throw", "mean")).reset_index().rename(columns={"posteam": "team"})
            off = off.merge(dd, on=["team", "game_id"], how="left").merge(rb, on=["team", "game_id"], how="left").merge(oo, on=["team", "game_id"], how="left")
        out.append(off)
        if verbose: print(f"  team-game {s}: {len(off):,} rows, {off.shape[1]} cols")
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(os.path.join(DATA, "universe_features_team_game.parquet"), index=False)
    return res


if __name__ == "__main__":
    print("player-game features"); pgf = build_player_game()
    print("team-game features"); tgf = build_team_game()
    print(f"done: {len(pgf):,} player-games × {pgf.shape[1]} cols; {len(tgf):,} team-games × {tgf.shape[1]} cols")
