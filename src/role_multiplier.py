"""Role multiplier — the injury CASCADE, measured from play-by-play (Alex's idea).

"Injury" is more than "is he hurt coming in." When a top player sits, his opportunity is
vacated and REDISTRIBUTES to teammates: the WR1 out means WR2/WR3/TE targets rise and the
QB's volume shifts; the RB1 out means the RB2 carries (and goal-line work) rise. We do not
assume how it redistributes — we measure it, from every 2023-25 team-game where a team's
top-share player was out on the injury report.

Measured quantities (per absent-role, per beneficiary-role):
    redistribution coefficient = (beneficiary share in absent games - normal share)
                                 / vacated share of the absent player
i.e. what fraction of the vacated opportunity each teammate role picks up. Plus the effect on
team volume itself (do pass attempts drop when the WR1 is out?).

Output: data/role_multiplier.json -> applied by injury_layer at projection time.
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")


def load_injuries(seasons):
    frames = []
    for s in seasons:
        cache = f"{DATA}/injuries_{s}.parquet"
        if os.path.exists(cache):
            frames.append(pd.read_parquet(cache)); continue
        try:
            import nflreadpy as nfl
            d = nfl.load_injuries(seasons=[s]).to_pandas()
        except Exception:
            # nflreadpy validates the season range; fall back to the nflverse release file
            try:
                d = pd.read_csv(f"https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{s}.csv")
            except Exception:
                d = pd.DataFrame()
        if len(d):
            d.to_parquet(cache, index=False)
        frames.append(d)
    return pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else pd.DataFrame()


def player_game_usage(pbp):
    """Per (season, week, team, pid): targets, carries; per (season, week, team): team totals."""
    d = pbp[(pbp.two_point_attempt == 0)]
    tg = d[(d.pass_attempt == 1) & d.receiver_player_id.notna()].groupby(
        ["season", "week", "posteam", "receiver_player_id"]).size().rename("targets").reset_index().rename(columns={"receiver_player_id": "pid"})
    ca = d[d.rush_attempt == 1].groupby(["season", "week", "posteam", "rusher_player_id"]).size().rename("carries").reset_index().rename(columns={"rusher_player_id": "pid"})
    u = tg.merge(ca, on=["season", "week", "posteam", "pid"], how="outer").fillna(0)
    team = d.groupby(["season", "week", "posteam"]).agg(
        team_targets=("pass_attempt", lambda s: int((s == 1).sum())),
        team_carries=("rush_attempt", lambda s: int((s == 1).sum()))).reset_index()
    # pass yards per team-game (for the QB-volume effect)
    py = d[d.pass_attempt == 1].groupby(["season", "week", "posteam"])["yards_gained"].sum().rename("team_pass_yds").reset_index() \
        if "yards_gained" in d.columns else None
    return u, team, py


def measure(seasons=(2023, 2024, 2025)):
    pbp = pd.read_parquet(f"{DATA}/pbp_2023_2025.parquet")
    players = pd.read_parquet(f"{DATA}/players_cache.parquet")
    pos = dict(zip(players.gsis_id, players.position))
    inj = load_injuries(seasons)
    out_set = set()
    if len(inj):
        o = inj[inj.report_status.isin(["Out", "Doubtful"]) & inj.gsis_id.notna()]
        out_set = set(zip(o.season.astype(int), o.week.astype(int), o.gsis_id))
    u, team, _ = player_game_usage(pbp)
    u["position"] = u["pid"].map(pos)
    u = u.merge(team, on=["season", "week", "posteam"])
    u["tgt_share"] = u["targets"] / u["team_targets"].clip(lower=1)
    u["car_share"] = u["carries"] / u["team_carries"].clip(lower=1)

    results = {}
    for absent_role, share_col, kind in (("WR1", "tgt_share", "targets"), ("TE1", "tgt_share", "targets"), ("RB1", "car_share", "carries")):
        pos_filter = {"WR1": ["WR"], "TE1": ["TE"], "RB1": ["RB"]}[absent_role]
        rows = []
        for (season, tm), g in u.groupby(["season", "posteam"]):
            # the team's top-share player of that position over the season
            season_share = g[g.position.isin(pos_filter)].groupby("pid")[share_col].mean()
            if season_share.empty: continue
            star = season_share.idxmax(); star_share = float(season_share.max())
            if star_share < (0.18 if kind == "targets" else 0.35): continue     # must be a real #1
            weeks = sorted(g.week.unique())
            played = set(g[g.pid == star].week)
            absent_weeks = [w for w in weeks if w not in played and (int(season), int(w), star) in out_set]
            if not absent_weeks: continue
            present = g[g.week.isin(played)]; absent = g[g.week.isin(absent_weeks)]
            # team volume effect
            tv_p = present.drop_duplicates(["week"])[f"team_{kind}"].mean()
            tv_a = absent.drop_duplicates(["week"])[f"team_{kind}"].mean()
            # beneficiaries: each teammate's share when star present vs absent, by role
            for pid, gg in g[g.pid != star].groupby("pid"):
                sp = gg[gg.week.isin(played)][share_col].mean(); sa = gg[gg.week.isin(absent_weeks)][share_col].mean()
                if np.isnan(sp) or np.isnan(sa): continue
                # role of the beneficiary: rank among same-position teammates by season share
                p = gg.position.iloc[0]
                same = g[g.position == p].groupby("pid")[share_col].mean().sort_values(ascending=False)
                rank = int(list(same.index).index(pid)) + 1 if pid in same.index else 9
                rows.append({"season": season, "team": tm, "absent": absent_role, "vacated": star_share,
                             "ben_pos": p, "ben_rank": rank, "share_present": sp, "share_absent": sa,
                             "pickup_frac": (sa - sp) / star_share, "n_absent": len(absent_weeks),
                             "team_vol_ratio": tv_a / tv_p if tv_p else np.nan})
        df = pd.DataFrame(rows)
        if df.empty: continue
        agg = df.groupby(["ben_pos", "ben_rank"]).agg(pickup=("pickup_frac", "median"), n=("pickup_frac", "size")).reset_index()
        agg = agg[agg.n >= 8]
        results[absent_role] = {
            "n_team_seasons": int(df.drop_duplicates(["season", "team"]).shape[0]),
            "team_volume_ratio": round(float(df.drop_duplicates(["season", "team"])["team_vol_ratio"].median()), 3),
            "pickup": {f"{r.ben_pos}{int(r.ben_rank)}": round(float(r.pickup), 3) for r in agg.itertuples()},
        }
    return results


if __name__ == "__main__":
    res = measure()
    json.dump(res, open(f"{DATA}/role_multiplier.json", "w"), indent=1)
    for k, v in res.items():
        print(f"\n{k} OUT  (n={v['n_team_seasons']} team-seasons)  team volume x{v['team_volume_ratio']}")
        for ben, f in sorted(v["pickup"].items(), key=lambda x: -x[1]):
            print(f"   {ben:5s} picks up {f:+.2f} of the vacated share")
