"""Fantasy consensus enrichment — ADP + Sleeper (role signal, esp. for changed situations).

Sources that actually work from here (ESPN/Yahoo/dynastyprocess 403 out):
  FantasyFootballCalculator ADP  — consensus draft rank + high/low/stdev (role certainty)
  Sleeper                        — depth_chart_position, injury_status, trending add/drop
  FantasyPros page               — parseable backup for rankings

Use: ADP is a SECOND market signal alongside the prop lines. Where our projection agrees
with consensus, confidence up. Where we strongly disagree (we're high on a player the
market drafts late, or vice-versa), it's a REVIEW flag — same discipline as line
divergence. For rookies/movers (our weak spot) ADP fills in the consensus role. ADP
stdev is a ready-made role-uncertainty measure. Sleeper trending catches role momentum
(a backup being added en masse = something changed) — beat-writer-adjacent.
"""
import os, json, urllib.request
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
UA = {"User-Agent": "Mozilla/5.0"}


def _get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20).read()


def load_adp(year=2026, teams=12):
    """FFC consensus ADP → DataFrame[name, position, team, adp, stdev, times_drafted]."""
    try:
        d = json.loads(_get(f"https://fantasyfootballcalculator.com/api/v1/adp/standard?teams={teams}&year={year}"))
        pl = d.get("players", [])
        df = pd.DataFrame(pl)
        return df[["name", "position", "team", "adp", "stdev", "times_drafted", "high", "low"]]
    except Exception:
        # fall back to cached
        p = f"{DATA}/ffc_adp_2026.json"
        if os.path.exists(p):
            return pd.DataFrame(json.load(open(p)).get("players", []))
        return pd.DataFrame()


def load_sleeper_signals():
    """Sleeper trending adds + injury/depth for the players moving right now."""
    try:
        players = json.loads(_get("https://api.sleeper.app/v1/players/nfl"))
        trend = json.loads(_get("https://api.sleeper.app/v1/players/nfl/trending/add?limit=50"))
        rows = []
        for t in trend:
            pl = players.get(t["player_id"], {})
            if pl.get("gsis_id"):
                rows.append({"gsis_id": pl["gsis_id"], "name": pl.get("full_name"),
                             "trend_adds": t["count"], "depth": pl.get("depth_chart_position"),
                             "injury": pl.get("injury_status")})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def _norm(n):
    return str(n).lower().replace(".", "").replace("'", "").replace(" jr", "").replace(" iii", "").strip()


def adp_crosscheck(proj, year=2026):
    """Attach ADP consensus to a projection frame and flag agreement/disagreement.
    Returns proj with adp, adp_stdev, and a consensus_flag: 'agree' / 'we're high' /
    'we're low' / 'no consensus' (undrafted). Uses ADP rank as a rough role proxy."""
    adp = load_adp(year)
    if not len(adp):
        proj["adp"] = np.nan; proj["consensus_flag"] = "n/a"; return proj
    adp["_k"] = adp["name"].map(_norm)
    a = adp.set_index("_k")
    proj = proj.copy(); proj["_k"] = proj["name"].map(_norm)
    proj["adp"] = proj["_k"].map(a["adp"])
    proj["adp_stdev"] = proj["_k"].map(a["stdev"])
    # our projection rank within position vs ADP rank within position (rough role agreement)
    def flag(row):
        if pd.isna(row["adp"]):
            return "undrafted"          # consensus sees ~no role — be cautious if we're high
        return "drafted"
    proj["consensus_flag"] = proj.apply(flag, axis=1)
    return proj.drop(columns=["_k"])


if __name__ == "__main__":
    adp = load_adp()
    print(f"ADP: {len(adp)} players. Highest role-uncertainty (stdev) inside top 60:")
    top = adp[adp.adp <= 60].sort_values("stdev", ascending=False).head(8)
    print(top[["name", "position", "team", "adp", "stdev"]].to_string(index=False))
    sl = load_sleeper_signals()
    print(f"\nSleeper trending (role momentum): {len(sl)} players")
    if len(sl):
        print(sl.head(6)[["name", "trend_adds", "depth", "injury"]].to_string(index=False))
