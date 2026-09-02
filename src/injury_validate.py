"""Earn the injury weights — do NOT assume them (Alex: "I give ideas, you test them").

Two questions, answered walk-forward on 2024-25 player-games:

1. ROLE CASCADE WEIGHT. For a beneficiary (RB2/RB3 when the RB1 is out; WR2/WR3/TE1 when the
   WR1 is out), the cascade predicts a volume bump m (from measured share redistribution).
   Does the realized stat bump match it?  realized_bump = actual - baseline;
   predicted_bump = baseline * (m - 1). Regress realized on predicted: the slope k* IS the
   earned weight (1.0 = apply the full multiplier; 0.5 = halve it; ~0 = it doesn't translate).
   Also report MAE at k in {0, .25, .5, .75, 1} so the choice is visible.

2. OWN-STATUS PENALTY. Among players listed Questionable/Doubtful who PLAYED, how does their
   actual/baseline ratio compare to healthy players'? That's the measured volume discount (and
   whether the trust cut is warranted at all).

Baseline = the player's season-to-date per-game average (weeks 1..k-1) — the no-injury-info
projection. Shares for role ranking are season-to-date too (no lookahead).
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import role_multiplier as rm

DATA = rm.DATA
STAT_FOR = {"RB": "rush_yds", "WR": "rec_yds", "TE": "rec_yds"}


def build_rows(seasons=(2024, 2025), min_week=4):
    pbp = pd.read_parquet(f"{DATA}/pbp_2021_2025.parquet")
    players = pd.read_parquet(f"{DATA}/players_cache.parquet")
    pos = dict(zip(players.gsis_id, players.position))
    pg = pd.read_parquet(f"{DATA}/prop_player_games.parquet")
    inj = rm.load_injuries(list(seasons))
    coef = json.load(open(f"{DATA}/role_multiplier.json"))
    u, team, _ = rm.player_game_usage(pbp)
    u["position"] = u.pid.map(pos); u = u.merge(team, on=["season", "week", "posteam"])
    rows, status_rows = [], []
    for season in seasons:
        us = u[u.season == season]; ps = pg[pg.season == season]
        rep = inj[inj.season == season] if len(inj) else pd.DataFrame()
        for wk in range(min_week, 19):
            past = us[us.week < wk]
            if past.empty: continue
            agg = past.groupby(["posteam", "pid"]).agg(t=("targets", "sum"), c=("carries", "sum"),
                                                        tt=("team_targets", "sum"), tc=("team_carries", "sum")).reset_index()
            agg["tgt_share"] = agg.t / agg.tt.clip(lower=1); agg["car_share"] = agg.c / agg.tc.clip(lower=1)
            agg["position"] = agg.pid.map(pos)
            # roles season-to-date
            role = {}; share = {}
            for tm, g in agg.groupby("posteam"):
                for p, col in (("WR", "tgt_share"), ("TE", "tgt_share"), ("RB", "car_share")):
                    sub = g[g.position == p].sort_values(col, ascending=False)
                    for i, r in enumerate(sub.itertuples(), 1):
                        role[r.pid] = (tm, f"{p}{i}"); share[r.pid] = {"tgt_share": r.tgt_share, "car_share": r.car_share}
            # this week's Out/Doubtful top-role players
            wrep = rep[(rep.week == wk) & rep.report_status.isin(["Out", "Doubtful"])] if len(rep) else pd.DataFrame()
            absences = {}
            for _, r in wrep.iterrows():
                tr = role.get(r.gsis_id)
                if tr and tr[1] in ("WR1", "TE1", "RB1"):
                    sh = share[r.gsis_id]
                    absences.setdefault(tr[0], []).append({"role": tr[1], "vac": sh["car_share"] if tr[1] == "RB1" else sh["tgt_share"]})
            # baseline per player: season-to-date mean of the stat
            base = ps[ps.week < wk].groupby(["pid", "mkt"])["y"].agg(["mean", "count"]).reset_index()
            base = base[base["count"] >= 3]
            bmap = {(r.pid, r.mkt): r["mean"] for _, r in base.iterrows()}
            actual = ps[ps.week == wk]
            amap = {(r.pid, r.mkt): r.y for _, r in actual.iterrows()}
            # beneficiaries
            for pid, (tm, my_role) in role.items():
                p = my_role[:2]; stat = STAT_FOR.get(p)
                if not stat or (pid, stat) not in bmap or (pid, stat) not in amap: continue
                m = 1.0
                for a in absences.get(tm, []):
                    c = coef.get(a["role"], {}); pick = c.get("pickup", {}); vol = float(c.get("team_volume_ratio", 1.0))
                    typ = "carries" if a["role"] == "RB1" else "targets"
                    if (typ == "carries") != (p == "RB"): continue
                    frac = pick.get(my_role, 0.10 if p == "WR" else 0.0)
                    if a["role"] == "TE1": frac *= 0.5
                    my_share = share[pid]["car_share" if typ == "carries" else "tgt_share"]
                    if my_share <= 0.02: continue
                    m *= vol * (my_share + frac * a["vac"]) / my_share
                m = float(np.clip(m, 0.6, 1.8))
                if m != 1.0:
                    b = bmap[(pid, stat)]
                    rows.append({"season": season, "week": wk, "pid": pid, "role": my_role, "stat": stat,
                                 "baseline": b, "actual": amap[(pid, stat)], "mult": m,
                                 "pred_bump": b * (m - 1), "real_bump": amap[(pid, stat)] - b})
            # own status among players who played
            if len(rep):
                wr_all = rep[rep.week == wk]
                st = dict(zip(wr_all.gsis_id, wr_all.report_status))
                for (pid, mkt), a in amap.items():
                    if (pid, mkt) not in bmap or mkt not in ("rush_yds", "rec_yds"): continue
                    b = bmap[(pid, mkt)]
                    if b < 15: continue
                    status_rows.append({"status": st.get(pid, "Healthy"), "ratio": a / b, "mkt": mkt})
    return pd.DataFrame(rows), pd.DataFrame(status_rows)


if __name__ == "__main__":
    df, sdf = build_rows()
    print(f"\n=== ROLE CASCADE: {len(df)} beneficiary player-games (2024-25, wk4+) ===")
    print(f"  mean predicted bump {df.pred_bump.mean():+.1f} yds | mean realized bump {df.real_bump.mean():+.1f} yds")
    # earned weight = slope of realized on predicted (through origin)
    k_star = float((df.pred_bump * df.real_bump).sum() / (df.pred_bump ** 2).sum())
    print(f"  EARNED WEIGHT k* (slope realized~predicted) = {k_star:.2f}")
    for k in (0, 0.25, 0.5, 0.75, 1.0):
        pred = df.baseline * (1 + k * (df.mult - 1))
        mae = (df.actual - pred).abs().mean(); bias = (df.actual - pred).mean()
        print(f"    k={k:<4} MAE={mae:6.2f}  bias={bias:+6.2f}")
    for role, g in df.groupby("role"):
        if len(g) >= 15:
            ks = float((g.pred_bump * g.real_bump).sum() / (g.pred_bump ** 2).sum())
            print(f"    {role}: n={len(g)} k*={ks:.2f}  pred {g.pred_bump.mean():+.1f} / real {g.real_bump.mean():+.1f}")
    print(f"\n=== OWN STATUS (played anyway): actual/baseline ratio ===")
    for s, g in sdf.groupby("status"):
        print(f"  {s:12s} n={len(g):5d}  median ratio {g.ratio.median():.3f}  mean {g.ratio.mean():.3f}")
    out = {"role_cascade": {"n": int(len(df)), "k_star": round(k_star, 3),
                            "pred_bump_mean": round(float(df.pred_bump.mean()), 2), "real_bump_mean": round(float(df.real_bump.mean()), 2)},
           "own_status": {s: {"n": int(len(g)), "median_ratio": round(float(g.ratio.median()), 3), "mean_ratio": round(float(g.ratio.mean()), 3)}
                          for s, g in sdf.groupby("status")}}
    json.dump(out, open(f"{DATA}/injury_validation.json", "w"), indent=1)
