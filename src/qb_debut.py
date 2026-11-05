"""QB DEBUT LEVER — what a first-time starting quarterback does to his skill players (measured).

Measured 2026-11-05 (`data/qb_debut_effect.json`): when a team's starting QB has <= 1 prior
career start and the start comes mid-season (week 5+, i.e. an emergency starter, not a planned
rookie opener), 2021-25:
  * the team's top-2 pass catchers' RECEIVING YARDS come in at 0.776x their last-3 average vs
    0.882x in an ordinary week (n=135, se .059)  -> x0.879 on rec_yds
  * their RECEPTIONS do not move (0.904 vs 0.896; n=140)  -> catch counts stay, yards per catch
    collapse: the debut-QB offense is check-downs
  * the lead back's RUSHING YARDS come in at 0.817x vs 0.993x (n=63, se .079) -> x0.823 on rush_yds
The multifactor model is blind to this by construction (its baseline and form are the player's
own games with the normal QB), so the lever is applied as a multiplier AFTER the projection,
like weather and the injury cascade, and it shows up as its own bar in the bridge.

Who starts: the schedule's posted starter when nflverse has it, else the week's override file
`data/qb_starters_<season>_w<week>.json` ({team: "QB name"}) written from the beat (Last Call /
the intel pass). Prior starts are counted from every schedule row before this game (1999+).
"""
import os, json
import pandas as pd
from names import norm_name

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
EFFECT_PATH = os.path.join(DATA, "qb_debut_effect.json")
MAX_PRIOR_STARTS = 1
TEAM_FIX = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA"}


class QbDebut:
    def __init__(self, season, week, schedules=None):
        self.season, self.week = int(season), int(week) if week is not None else 1
        try:
            eff = json.load(open(EFFECT_PATH))
        except Exception:
            eff = {}
        self.effect = eff
        self.mult = {k: float(v) for k, v in (eff.get("multiplier") or {}).items()}
        self.starters, self.prior = {}, {}
        try:
            s = schedules if schedules is not None else pd.read_parquet(os.path.join(DATA, "schedules.parquet"))
            wk = s[(s["season"] == self.season) & (s["week"] == self.week)]
            for _, r in wk.iterrows():
                for side in ("home", "away"):
                    nm = r.get(f"{side}_qb_name")
                    if isinstance(nm, str) and nm:
                        self.starters[str(r[f"{side}_team"])] = nm
            past = s[s["home_score"].notna() & ((s["season"] < self.season) | ((s["season"] == self.season) & (s["week"] < self.week)))]
            names = pd.concat([past["home_qb_name"], past["away_qb_name"]]).dropna().map(norm_name)
            self.prior = names.value_counts().to_dict()
        except Exception:
            pass
        ov = os.path.join(DATA, f"qb_starters_{self.season}_w{self.week}.json")
        if os.path.exists(ov):
            try:
                self.starters.update({str(k): str(v) for k, v in json.load(open(ov)).items()})
            except Exception:
                pass

    def debut(self, team):
        """(name, prior_starts) when this team's starter has <= MAX_PRIOR_STARTS career starts, else None."""
        t = TEAM_FIX.get(str(team), str(team))
        nm = self.starters.get(t)
        if not nm:
            return None
        p = int(self.prior.get(norm_name(nm), 0))
        return (nm, p) if p <= MAX_PRIOR_STARTS else None

    def multiplier(self, team, mkt, position, top_target=False, lead_back=False):
        """Measured multiplier on the projected mean for a skill player whose QB is a debut starter.
        SCOPE = what was measured: the team's top-2 pass catchers (rec_yds) and its lead back
        (rush_yds). Targets 3-5 (rel .94, n.s.) and the RB2 (rel .99) showed no effect — no lever."""
        d = self.debut(team)
        if not d:
            return 1.0, None
        nm, p = d
        pos = str(position or "").upper()
        if mkt == "rec_yds" and top_target and pos in ("WR", "TE", "RB") and "rec_yds" in self.mult:
            m, who = self.mult["rec_yds"], "top targets"
        elif mkt == "rush_yds" and lead_back and pos == "RB" and "rush_yds" in self.mult:
            m, who = self.mult["rush_yds"], "lead back"
        else:
            return 1.0, None
        deb = (self.effect.get("debut") or {}).get(mkt, {}); ref = (self.effect.get("reference") or {}).get(mkt, {})
        note = (f"{nm} makes his {'first' if p == 0 else 'second'} career start — measured on emergency starters 2021-25 "
                f"(n={deb.get('n', '?')}), the {who} came in at {float(deb.get('mean_ratio', m)) * 100:.0f}% of their last-3 average "
                f"vs {float(ref.get('mean_ratio', 1.0)) * 100:.0f}% in an ordinary week; applied ×{m:.3f}")
        return float(m), note
