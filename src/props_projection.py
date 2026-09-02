"""Preseason prop projection engine — un-gates the prop board before 2026 data.

The market is softest on props in the preseason and early weeks, because books
anchor lines to stale prior-year usage. Our edge comes from projecting each
player's 2026 ROLE from a weighted hierarchy of signals we already have, turning
role -> volume -> yards/receptions, and flagging where our number diverges from a
market line that hasn't repriced a role change yet.

Hierarchy (each layer weighted, weights shift as real data arrives):
  1. Own recency-weighted per-game usage (2023-25), the base.
  2. 2026 role: team + within-position rank (depth chart / roster), which sets a
     rank-prior share estimated EMPIRICALLY from 2019-25 (what a WR1/WR2/RB1 gets).
  3. Vacated opportunity: shares are normalized within each 2026 team, so a departed
     player's targets/carries flow to whoever remains, weighted by role. (The market
     is slowest on exactly this.)
  4. Rookies / no history: draft capital + depth rank -> role prior only.
  5. Efficiency: yards-per-target / yards-per-carry, shrunk to a positional prior.

Output: per player, a projected per-game mean for each market (rec_yds, receptions,
rush_yds, pass_yds) that feeds props_model.p_over to price ANY posted line.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

SKILL = {"WR", "TE", "RB", "FB", "QB"}
REC_POS = {"WR", "TE", "RB", "FB"}
# how much to trust own history vs role prior, by games of history
def w_own(games, moved):
    # Established players are hard to beat with a role prior — trust their own number.
    # Only genuinely reset trust when a player changed teams (role uncertainty).
    base = games / (games + 5.0)           # ~full trust by ~15-20 games
    if moved:                              # new team -> role reset, lean on prior
        base *= 0.6
    return float(np.clip(base, 0.0, 0.95))

# efficiency shrink toward positional prior (light — don't flatten stars)
EFF_SHRINK = 4.0


def load_usage():
    return pd.read_parquet(f"{DATA}/player_season_usage.parquet")


def recency_usage(usage, upto_season, halflife=1.2):
    """Per player: recency-weighted per-game usage & efficiency using seasons <= upto."""
    d = usage[usage["season"] <= upto_season].copy()
    d["age"] = upto_season - d["season"]           # 0 = most recent completed season
    d["w"] = 0.5 ** (d["age"] / halflife) * d["games"]  # weight by recency AND sample
    rows = []
    for pid, grp in d.groupby("player_id"):
        wsum = grp["w"].sum()
        if wsum <= 0:
            continue
        def wm(col):
            return float(np.average(grp[col], weights=grp["w"]))
        rows.append({
            "player_id": pid, "name": grp["player_display_name"].iloc[-1],
            "position": grp.sort_values("season")["position"].iloc[-1],
            "last_team": grp.sort_values("season")["team"].iloc[-1],
            "hist_games": float(grp["games"].sum()),
            "tgt_pg": wm("targets_pg"), "rec_pg": wm("receptions_pg"),
            "recyd_pg": wm("rec_yds_pg"), "car_pg": wm("carries_pg"),
            "rushyd_pg": wm("rush_yds_pg"), "att_pg": wm("att_pg"), "passyd_pg": wm("pass_yds_pg"),
            "tgt_share": wm("tgt_share") if grp["tgt_share"].notna().any() else np.nan,
        })
    r = pd.DataFrame(rows)
    # efficiency (shrunk later): yards per target / per carry
    r["ypt"] = r["recyd_pg"] / r["tgt_pg"].replace(0, np.nan)
    r["catch"] = r["rec_pg"] / r["tgt_pg"].replace(0, np.nan)
    r["ypc"] = r["rushyd_pg"] / r["car_pg"].replace(0, np.nan)
    r["ypa"] = r["passyd_pg"] / r["att_pg"].replace(0, np.nan)
    return r


def rank_prior_shares(usage, upto_season):
    """Empirical role priors: a player's share of the TEAM's total targets (or carries),
    by within-position rank. Denominator is the whole team so shares across all positions
    sum to ~1 — WR1~0.22, WR2~0.15, TE1~0.13, RB1(rec)~0.10, etc. Uses seasons <= upto."""
    d = usage[(usage["season"] <= upto_season)].copy()
    out = {}
    for mkt, val, positions in [("tgt", "targets", ["WR", "TE", "RB"]),
                                ("car", "carries", ["RB"])]:
        ranks = {}
        for (tm, s), tgrp in d.groupby(["team", "season"]):
            team_total = tgrp[val].sum()               # WHOLE team denominator
            if team_total <= 0:
                continue
            for pos in positions:
                gp = tgrp[tgrp["position"] == pos].sort_values(val, ascending=False).reset_index()
                for i, row in gp.iterrows():
                    ranks.setdefault((pos, i + 1), []).append(row[val] / team_total)
        for (pos, rk), shares in ranks.items():
            out[(mkt, pos, rk)] = float(np.mean(shares))
    return out


def latest_depth_rank(season):
    """Within-position 2026 depth rank per player (gsis_id) from rosters+depth charts."""
    import nflreadpy as nfl
    try:
        dc = nfl.load_depth_charts(seasons=[season]).to_pandas()
    except Exception:
        return {}
    if not len(dc):
        return {}
    if "gsis_id" not in dc.columns or "pos_rank" not in dc.columns:
        return {}
    dc = dc.dropna(subset=["gsis_id"])
    # latest snapshot per player-position (time column varies by season)
    tcol = next((c for c in ["dt", "week", "season"] if c in dc.columns), None)
    if tcol:
        dc = dc.sort_values(tcol)
    posc = "pos_abb" if "pos_abb" in dc.columns else ("pos_name" if "pos_name" in dc.columns else None)
    grpcols = ["gsis_id"] + ([posc] if posc else [])
    dc = dc.groupby(grpcols).tail(1)
    # collapse alignment codes (LWR/RWR/SWR->WR etc.) via pos_grp when present
    rank = {}
    for _, r in dc.iterrows():
        rank.setdefault(r["gsis_id"], r["pos_rank"])
        rank[r["gsis_id"]] = min(rank[r["gsis_id"]], r["pos_rank"])
    return rank


def roster_2026(season):
    import nflreadpy as nfl
    r = nfl.load_rosters(seasons=[season]).to_pandas()
    r = r[r["position"].isin(SKILL)]
    return r[["gsis_id", "full_name", "position", "team", "years_exp"]].rename(
        columns={"gsis_id": "player_id", "full_name": "name"})


def draft_priors(season):
    """Rookie draft capital -> a role prior strength (1st round WR/RB = starter)."""
    import nflreadpy as nfl
    dp = nfl.load_draft_picks(seasons=[season]).to_pandas()
    dp = dp[dp["position"].isin(SKILL)]
    return {r["gsis_id"]: {"round": r["round"], "pick": r["pick"], "pos": r["position"]}
            for _, r in dp.iterrows() if pd.notna(r["gsis_id"])}


def project(season, team_vol_season=None):
    """Full preseason projection for `season`. Returns per-player projected per-game
    means for each market. team_vol_season = season to source team pace from (default
    season-1)."""
    usage = load_usage()
    upto = season - 1
    hist = recency_usage(usage, upto)
    priors = rank_prior_shares(usage, upto)
    ros = roster_2026(season)
    depth = latest_depth_rank(season)
    rookies = draft_priors(season)
    tv = pd.read_parquet(f"{DATA}/team_season_volume.parquet")
    tvs = team_vol_season or (season - 1)
    tv = tv[tv["season"] == tvs].set_index("team")
    league_tgt = tv["team_targets_pg"].mean() if len(tv) else 34.0
    league_car = tv["team_carries_pg"].mean() if len(tv) else 26.0

    # positional efficiency priors (shrink targets)
    effp = {p: {"ypt": hist[hist.position == p]["ypt"].median(),
                "catch": hist[hist.position == p]["catch"].median(),
                "ypc": hist[hist.position == p]["ypc"].median(),
                "ypa": hist[hist.position == p]["ypa"].median()}
            for p in REC_POS | {"QB"}}

    m = ros.merge(hist.drop(columns=["name", "position"], errors="ignore"),
                  on="player_id", how="left")
    m["moved"] = (m["last_team"].notna()) & (m["last_team"] != m["team"])
    m["hist_games"] = m["hist_games"].fillna(0.0)
    m["depth_rank"] = m["player_id"].map(depth)

    # ---- role rank within team-position (for share prior) ----
    def role_score(row):
        # returning production, else rookie capital, else depth rank
        prod = (row["tgt_pg"] if row["position"] in REC_POS else 0) + \
               (row["car_pg"] if row["position"] == "RB" else 0)
        if pd.notna(prod) and prod > 0:
            return 1000 + prod                      # producers ranked first, by production
        if row["player_id"] in rookies:
            return 500 - rookies[row["player_id"]]["pick"] / 10.0   # earlier pick = higher
        dr = row["depth_rank"]
        return 100 - (dr if pd.notna(dr) else 9)     # depth chart fallback
    m["role_score"] = m.apply(role_score, axis=1)

    proj_rows = []
    for team, tg in m.groupby("team"):
        for mkt, pos_list, val_pg, league_v, tvcol in [
            ("tgt", ["WR", "TE", "RB", "FB"], "tgt_pg", league_tgt, "team_targets_pg"),
            ("car", ["RB", "FB"], "car_pg", league_car, "team_carries_pg")]:
            sub = tg[tg["position"].isin(pos_list)].copy()
            if not len(sub):
                continue
            team_v = tv.at[team, tvcol] if team in tv.index else league_v
            # rank within position for prior share
            sub["rk"] = sub.groupby("position")["role_score"].rank(ascending=False, method="first")
            def prior_share(row):
                pos = row["position"] if (mkt, row["position"], 1) in priors else \
                      ("RB" if mkt == "car" else "WR")
                return priors.get((mkt, pos, int(row["rk"])),
                                  priors.get((mkt, pos, 6), 0.02))
            sub["prior_sh"] = sub.apply(prior_share, axis=1)
            # own share
            if mkt == "tgt":
                sub["own_sh"] = (sub["tgt_pg"] / team_v)
            else:
                sub["own_sh"] = (sub["car_pg"] / team_v)
            sub["own_sh"] = sub["own_sh"].fillna(0.0)
            sub["w"] = [w_own(g, mv) for g, mv in zip(sub["hist_games"], sub["moved"])]
            sub["blend_sh"] = sub["w"] * sub["own_sh"] + (1 - sub["w"]) * sub["prior_sh"]
            # normalize so the group's shares sum to a realistic team total: receivers
            # take ~100% of targets; RBs take ~90% of carries (QB/WR scrambles the rest).
            group_total = 0.90 if mkt == "car" else 1.0
            ssum = sub["blend_sh"].sum()
            if ssum > 0:
                sub["norm_sh"] = sub["blend_sh"] / ssum * group_total
            else:
                sub["norm_sh"] = 0.0
            sub["vol_pg"] = sub["norm_sh"] * team_v
            for _, row in sub.iterrows():
                proj_rows.append({"player_id": row["player_id"], "name": row["name"],
                                  "position": row["position"], "team": team,
                                  "market_group": mkt, "vol_pg": row["vol_pg"],
                                  "moved": row["moved"], "hist_games": row["hist_games"],
                                  "rookie": row["player_id"] in rookies})
    pv = pd.DataFrame(proj_rows)
    # pivot volume: targets & carries per player
    tgt = pv[pv.market_group == "tgt"].set_index("player_id")["vol_pg"]
    car = pv[pv.market_group == "car"].set_index("player_id")["vol_pg"]
    base = pv.drop_duplicates("player_id").set_index("player_id")
    out = base[["name", "position", "team", "moved", "hist_games", "rookie"]].copy()
    out["proj_targets_pg"] = tgt.reindex(out.index).fillna(0.0)
    out["proj_carries_pg"] = car.reindex(out.index).fillna(0.0)

    # efficiency (shrink player to positional prior)
    heff = hist.set_index("player_id")
    def eff(pid, pos, col):
        pr = effp.get(pos, effp["WR"])[col]
        if pid in heff.index and pd.notna(heff.at[pid, col]):
            g = heff.at[pid, "hist_games"]
            wgt = g / (g + EFF_SHRINK)
            return wgt * heff.at[pid, col] + (1 - wgt) * pr
        return pr
    out["ypt"] = [eff(i, p, "ypt") for i, p in zip(out.index, out["position"])]
    out["catch"] = [eff(i, p, "catch") for i, p in zip(out.index, out["position"])]
    out["ypc"] = [eff(i, p, "ypc") for i, p in zip(out.index, out["position"])]

    # role-model per-market means (share-based — for CHANGED situations)
    role_rec_yds = out["proj_targets_pg"] * out["ypt"]
    role_receptions = out["proj_targets_pg"] * out["catch"]
    role_rush_yds = out["proj_carries_pg"] * out["ypc"]
    # own recency-weighted per-game (for STABLE players — beats naive, ~unbiased)
    own_rec_yds = heff["recyd_pg"].reindex(out.index)
    own_receptions = heff["rec_pg"].reindex(out.index)
    own_rush_yds = heff["rushyd_pg"].reindex(out.index)
    # HYBRID: trust own for established/stable, role model for movers/rookies/no-history
    wt = pd.Series([w_own(g, mv) for g, mv in zip(out["hist_games"], out["moved"])], index=out.index)
    out["w_own"] = wt
    def blend(own, role):
        own = own.fillna(0.0)
        return (wt * own + (1 - wt) * role).where(own.notna() | (role > 0), role)
    out["rec_yds"] = wt * own_rec_yds.fillna(role_rec_yds) + (1 - wt) * role_rec_yds
    out["receptions"] = wt * own_receptions.fillna(role_receptions) + (1 - wt) * role_receptions
    out["rush_yds"] = wt * own_rush_yds.fillna(role_rush_yds) + (1 - wt) * role_rush_yds
    out["rush_att"] = out["proj_carries_pg"]   # rush attempts projection = projected carries/game
    out["role_rec_yds"] = role_rec_yds; out["own_rec_yds"] = own_rec_yds
    out["pass_yds"] = 0.0                        # non-QBs don't pass; QBs appended below

    # confidence: high for established stable players (projection beats naive, unbiased);
    # low for movers/rookies (high variance — pick only on a large line divergence).
    def conf(row):
        if row["rookie"] or row["hist_games"] < 3:
            return 0.20
        c = min(row["hist_games"] / 24.0, 1.0)
        if row["moved"]:
            c *= 0.6
        return round(float(c), 2)
    out["confidence"] = out.apply(conf, axis=1)
    out["tier"] = pd.cut(out["confidence"], [-1, 0.34, 0.66, 1.01],
                         labels=["role-est (high var)", "medium", "stable"])
    out = out.reset_index()

    # ---- QBs: append passing-yards projections (they're not in the share-based volume loop) --
    qbros = ros[ros["position"] == "QB"].copy()
    heff = hist.set_index("player_id")
    league_qb = hist[(hist.position == "QB") & (hist.passyd_pg > 150)]["passyd_pg"].median()
    league_qb = float(league_qb) if pd.notna(league_qb) else 230.0
    qrows = []
    for _, r in qbros.iterrows():
        pid = r["player_id"]
        own = heff.at[pid, "passyd_pg"] if pid in heff.index else np.nan
        hg = float(heff.at[pid, "hist_games"]) if pid in heff.index else 0.0
        moved = bool(pid in heff.index and pd.notna(heff.at[pid, "last_team"]) and heff.at[pid, "last_team"] != r["team"])
        rookie = pid in rookies
        w = w_own(hg, moved)
        py = w * (own if pd.notna(own) else league_qb) + (1 - w) * league_qb
        qrows.append({"player_id": pid, "name": r["name"], "position": "QB", "team": r["team"],
                      "moved": moved, "hist_games": hg, "rookie": rookie,
                      "proj_targets_pg": 0.0, "proj_carries_pg": 0.0, "pass_yds": max(py, 0.0),
                      "rec_yds": 0.0, "receptions": 0.0, "rush_yds": 0.0, "rush_att": 0.0})
    qdf = pd.DataFrame(qrows)
    # keep only each team's QB1 (highest projected passing) — backups have ~no props
    if len(qdf):
        qdf = qdf.sort_values("pass_yds", ascending=False).drop_duplicates("team")
        qdf["confidence"] = [0.20 if (rk or hg < 3) else round(min(hg / 24.0, 1.0) * (0.6 if mv else 1.0), 2)
                             for rk, hg, mv in zip(qdf["rookie"], qdf["hist_games"], qdf["moved"])]
        qdf["tier"] = pd.cut(qdf["confidence"], [-1, 0.34, 0.66, 1.01],
                             labels=["role-est (high var)", "medium", "stable"])
        out = pd.concat([out, qdf], ignore_index=True)
    return out.sort_values("rec_yds", ascending=False)


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    p = project(season)
    p.to_parquet(f"{DATA}/projection_{season}.parquet")
    cols = ["name", "position", "team", "proj_targets_pg", "rec_yds", "receptions",
            "proj_carries_pg", "rush_yds", "moved", "rookie"]
    print(f"=== {season} preseason projections (top 20 rec_yds) ===")
    print(p[cols].head(20).to_string(index=False))
