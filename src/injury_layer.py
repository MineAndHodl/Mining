"""Injury layer — own status AND the role cascade, applied to every pick.

Two things, both first-class (THE_DESK_PROMPT §2.5):
  1. OWN STATUS from the weekly injury report: Out -> never a pick; Doubtful -> strong penalty;
     Questionable -> trust penalty; healthy -> checked, neutral.
  2. ROLE MULTIPLIER: a teammate's absence vacates opportunity that redistributes — measured in
     role_multiplier.py from 2023-25 play-by-play (RB1 out: RB2 +.55 / RB3 +.32 of the vacated
     carry share; WR1 out: targets spread — WR2 +.20, WR3 +.13, TE1 +.07, backs +.08; the QB's
     volume does NOT drop, so no QB downgrade). Applied to a pick's projected volume and
     surfaced on the Injury Picture slider and in the paragraph.

Roles are ranked from last season's shares (WR1/WR2/WR3, TE1, RB1/RB2/RB3) with the 2026 team
overrides, so a traded RB1 is that team's RB1 here. The report loader tries nflreadpy, then the
nflverse release CSV, and returns empty cleanly if the week's report isn't posted yet — the
plumbing is live before the data is.
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from names import norm_name
import role_multiplier as rm

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
TEAM_FIX = {"LAR": "LA", "OAK": "LV", "SD": "LAC", "STL": "LA", "WSH": "WAS"}

# EARNED (injury_validate.py, 2024-25 walk-forward; data/injury_validation.json):
#   ROLE_CASCADE_WEIGHT: realized yardage bump vs predicted bump slope k* = 0.84 over 583
#     beneficiary player-games (RB2 .98, RB3 1.08, WR2 .73, WR3 .91, TE1 1.28). The cascade is
#     real and translates nearly 1:1 into yards; applied at the earned 0.84, not assumed 1.0.
#   Questionable: players who PLAY post a median actual/baseline of .845 vs healthy .849 — no
#     performance discount. It is a scratch-risk flag ("confirm active"), so a mild lean and a
#     small trust cut only. Doubtful mostly don't play -> hard rail (not a performance claim).
ROLE_CASCADE_WEIGHT = 0.87   # re-earned on the 5-season coefficients (was .84 on 3y)
STATUS_PENALTY = {"Out": None, "Doubtful": -1.0, "Questionable": -0.15}
MKT_TYPE = {"rec_yds": "targets", "receptions": "targets", "rush_yds": "carries", "rush_att": "carries",
            "pass_yds": "pass", "anytime_td": "mixed"}


class InjuryLayer:
    def __init__(self, season=2026, week=None):
        self.season = season; self.week = week
        self.coef = json.load(open(f"{DATA}/role_multiplier.json")) if os.path.exists(f"{DATA}/role_multiplier.json") else {}
        self.report = self._load_report()
        self.roles, self.shares = self._roles_from_last_season()

    # ------------------------------------------------------------------ data
    def _load_report(self):
        try:
            inj = rm.load_injuries([self.season])
        except Exception:
            inj = pd.DataFrame()
        if not len(inj):
            return pd.DataFrame(columns=["week", "team", "gsis_id", "full_name", "position", "report_status", "practice_status", "report_primary_injury"])
        wk = self.week if self.week is not None else int(inj["week"].max())
        cur = inj[inj["week"] == wk].copy()
        cur["_k"] = cur["full_name"].map(norm_name)
        return cur

    def _roles_from_last_season(self):
        """Rank each team's WRs/TEs/RBs by last-season share; apply 2026 team overrides."""
        pbp = pd.read_parquet(f"{DATA}/pbp_2021_2025.parquet")
        pbp = pbp[pbp.season == pbp.season.max()]
        players = pd.read_parquet(f"{DATA}/players_cache.parquet")
        pos = dict(zip(players.gsis_id, players.position)); nm = dict(zip(players.gsis_id, players.display_name))
        u, team, _ = rm.player_game_usage(pbp)
        u = u.merge(team, on=["season", "week", "posteam"])
        agg = u.groupby(["posteam", "pid"]).agg(targets=("targets", "sum"), carries=("carries", "sum"),
                                                tt=("team_targets", "sum"), tc=("team_carries", "sum")).reset_index()
        agg["tgt_share"] = agg.targets / agg.tt.clip(lower=1); agg["car_share"] = agg.carries / agg.tc.clip(lower=1)
        agg["position"] = agg.pid.map(pos); agg["name"] = agg.pid.map(nm); agg["key"] = agg.name.map(norm_name)
        # 2026 team overrides
        try:
            proj = pd.read_parquet(f"{DATA}/projection_2026.parquet")
            cur = {norm_name(r["name"]): TEAM_FIX.get(r["team"], r["team"]) for _, r in proj.iterrows()}
            agg["team"] = [cur.get(k, t) for k, t in zip(agg.key, agg.posteam)]
        except Exception:
            agg["team"] = agg.posteam
        roles, shares = {}, {}
        for tm, g in agg.groupby("team"):
            for p, col in (("WR", "tgt_share"), ("TE", "tgt_share"), ("RB", "car_share")):
                sub = g[g.position == p].sort_values(col, ascending=False)
                for i, r in enumerate(sub.itertuples(), 1):
                    roles[r.key] = (tm, f"{p}{i}"); shares[r.key] = {"tgt_share": float(r.tgt_share), "car_share": float(r.car_share), "team": tm, "pos": p}
        return roles, shares

    # ------------------------------------------------------------------ own status
    def status(self, name):
        if not len(self.report): return None
        r = self.report[self.report["_k"] == norm_name(name)]
        if not len(r): return None
        r = r.iloc[0]
        return {"status": r.get("report_status"), "practice": r.get("practice_status"), "injury": r.get("report_primary_injury")}

    # ------------------------------------------------------------------ cascade
    def team_absences(self, team):
        """Top-role players on this team listed Out/Doubtful this week -> [(role, name, vacated_share)]."""
        out = []
        if not len(self.report): return out
        rep = self.report[self.report.report_status.isin(["Out", "Doubtful"])]
        for _, r in rep.iterrows():
            k = r["_k"]; tm_role = self.roles.get(k)
            if not tm_role or tm_role[0] != team: continue
            role = tm_role[1]
            if role in ("WR1", "TE1", "RB1"):
                sh = self.shares[k]
                vac = sh["car_share"] if role == "RB1" else sh["tgt_share"]
                out.append({"role": role, "name": r["full_name"], "vacated": vac, "status": r["report_status"]})
        return out

    def role_multiplier(self, name, mkt):
        """(multiplier on projected volume, note) from teammates' absences. 1.0 if none."""
        k = norm_name(name); tm_role = self.roles.get(k)
        if not tm_role: return 1.0, None
        team, my_role = tm_role; sh = self.shares.get(k, {})
        typ = MKT_TYPE.get(mkt, "targets")
        if typ == "mixed":                                       # anytime TD: a back scores on carries, a catcher on targets
            typ = "carries" if my_role.startswith("RB") else "targets"
        if typ == "pass": return 1.0, None                      # measured: no QB downgrade for a WR1 absence
        mult = 1.0; notes = []
        for a in self.team_absences(team):
            src = a["role"]; c = self.coef.get(src, {})
            pick = c.get("pickup", {}); vol = float(c.get("team_volume_ratio", 1.0))
            if src == "RB1" and typ != "carries": continue
            if src in ("WR1", "TE1") and typ != "targets": continue
            frac = pick.get(my_role)
            if frac is None:
                frac = 0.10 if my_role[:2] in ("WR",) else 0.0     # pooled deep-WR pickup
            if src == "TE1": frac *= 0.5                           # thin sample, half weight
            my_share = sh.get("car_share" if typ == "carries" else "tgt_share", 0.0)
            if my_share <= 0.02: continue
            gained = frac * a["vacated"]
            mult *= vol * (my_share + gained) / my_share
            if gained > 0.005:
                notes.append(f"{a['name']} ({src}) is {a['status'].lower()} — we credit {name} with ~{int(round(frac*100))}% of his vacated "
                             f"{'carries' if typ=='carries' else 'targets'} (measured redistribution), about +{int(round(gained/my_share*100))}% volume")
        mult = 1.0 + ROLE_CASCADE_WEIGHT * (mult - 1.0)          # earned weight, not assumed
        return round(float(np.clip(mult, 0.6, 1.7)), 3), ("; ".join(notes) if notes else None)

    def injury_lean(self, name, mkt):
        """The Injury Picture slider: own status (negative) or opportunity from an absence (positive).
        Returns None only when no report exists at all (honest n/a)."""
        if not len(self.report): return None, None
        st = self.status(name)
        if st and st["status"] in STATUS_PENALTY:
            pen = STATUS_PENALTY[st["status"]]
            return (pen if pen is not None else -1.0), f"listed {st['status']} ({st.get('injury') or 'injury'}; {st.get('practice') or 'no practice data'})"
        mult, note = self.role_multiplier(name, mkt)
        if mult > 1.02:
            return float(np.clip((mult - 1.0) / 0.3, 0, 1)), note
        return 0.0, "checked — healthy, no vacated role on his team"


if __name__ == "__main__":
    il = InjuryLayer(2026)
    print("report rows this week:", len(il.report), "| roles ranked:", len(il.roles))
    # dry-run against last season's Week 10 to prove the cascade fires
    il25 = InjuryLayer(2025, week=10)
    print("2025 wk10 report rows:", len(il25.report))
    fired = 0
    for k, (tm, role) in list(il25.roles.items()):
        if role in ("RB2", "WR2", "TE1"):
            m, n = il25.role_multiplier(k, "rush_yds" if role.startswith("RB") else "rec_yds")
            if m != 1.0:
                fired += 1
                if fired <= 6: print(f"  {k:22s} {tm} {role}: x{m}  {n}")
    print("cascade fired for", fired, "players in 2025 wk10")
