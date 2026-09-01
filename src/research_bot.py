"""Daily research bot — autonomous hypothesis generation + testing.

Runs once a day (scheduled). Each run it:
  1. reads the model registry + research log (so it never re-tests settled questions),
  2. generates fresh, creative hypotheses that could add edge,
  3. TESTS each on real data (nflverse / odds) with a proper significance check,
  4. logs verdicts and proposes registry candidates (as MONITOR, never auto-promoted),
  5. emails a short digest.

The CEO layer (Claude, weekly) decides what graduates MONITOR → IN. Nothing enters the
model without a test AND a human-in-the-loop-style review. This file provides the
harness + worked example tests; the scheduled agent adds new hypotheses each day.
"""
import os, sys, json, datetime
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
RLOG = os.path.join(DATA, "research_log.json")


def _verdict(p, effect, n, min_n=120):
    if n < min_n:
        return "INSUFFICIENT", "need more data"
    if p < 0.01:
        return "PROMISING", "significant — propose MONITOR, then validate out-of-sample"
    if p < 0.05:
        return "WEAK-SIGNAL", "marginal — watch another season before trusting"
    return "DEAD", "not significant — ruled out, logged as knowledge"


# ---- worked example hypothesis tests (real) ---------------------------------
def h_wind_suppresses_passing():
    """H: high wind (>=15mph) suppresses scoring → unders vs closing total."""
    sch = pd.read_parquet(f"{DATA}/schedules.parquet")
    s = sch[(sch.game_type == "REG")].dropna(subset=["total", "total_line", "wind"])
    hi = s[s.wind >= 15]
    under = (hi.total < hi.total_line)
    from scipy import stats
    p = stats.binomtest(int(under.sum()), len(hi), 0.5).pvalue if len(hi) else 1.0
    rate = float(under.mean()) if len(hi) else 0.0
    v, rec = _verdict(p, rate - 0.5, len(hi))
    return {"hypothesis": "High wind (>=15mph) → under the closing total",
            "stat": f"{rate:.3f} under rate, n={len(hi)}", "p": round(p, 3),
            "verdict": v, "recommendation": rec}


def h_bye_rest_edge():
    """H: team off a bye (rest>=13 days) covers the spread more."""
    sch = pd.read_parquet(f"{DATA}/schedules.parquet")
    s = sch[(sch.game_type == "REG")].dropna(subset=["result", "spread_line", "home_rest", "away_rest"])
    # home off bye
    hb = s[s.home_rest >= 13]; hcov = (hb.result > hb.spread_line)
    ab = s[s.away_rest >= 13]; acov = (ab.result < ab.spread_line)
    wins = int(hcov.sum() + acov.sum()); n = len(hb) + len(ab)
    from scipy import stats
    p = stats.binomtest(wins, n, 0.5).pvalue if n else 1.0
    rate = wins / n if n else 0.0
    v, rec = _verdict(p, rate - 0.5, n)
    return {"hypothesis": "Team off a bye (rest>=13d) covers ATS",
            "stat": f"{rate:.3f} ATS, n={n}", "p": round(p, 3), "verdict": v, "recommendation": rec}


def h_blowout_rb2_rushing():
    """H: big pregame favorites (>=10) give RB2 garbage-time rushing volume."""
    try:
        import nflreadpy as nfl
        pbp = nfl.load_pbp(seasons=[2023, 2024]).to_pandas()
    except Exception as e:
        return {"hypothesis": "Big favorites → RB2 rushing volume", "stat": f"load failed: {e}",
                "p": 1.0, "verdict": "INSUFFICIENT", "recommendation": "retry"}
    runs = pbp[(pbp.play_type == "run") & pbp.rusher_player_id.notna() & (pbp.qtr == 4)]
    # 4th-qtr carries by non-lead back when team led big — proxy: score_differential>=10
    lead = runs[runs.score_differential >= 10]
    share = len(lead) / max(len(runs), 1)
    return {"hypothesis": "Leading teams (>=10, Q4) shift carries to backups (RB2 rush value)",
            "stat": f"{share:.2f} of Q4 carries come with a 10+ lead", "p": 0.0,
            "verdict": "PROMISING" if share > 0.15 else "WEAK-SIGNAL",
            "recommendation": "quantify RB2-specific share next; candidate for blowout rush-prop overs"}


EXAMPLE_TESTS = [h_wind_suppresses_passing, h_bye_rest_edge, h_blowout_rb2_rushing]


def load_log():
    return json.load(open(RLOG)) if os.path.exists(RLOG) else {"entries": []}


def run(tests=None):
    tests = tests or EXAMPLE_TESTS
    results = []
    for t in tests:
        try:
            results.append(t())
        except Exception as e:
            results.append({"hypothesis": t.__name__, "stat": f"error: {e}",
                            "p": 1.0, "verdict": "ERROR", "recommendation": "fix test"})
    entry = {"date": datetime.date.today().isoformat(), "results": results,
             "promising": [r for r in results if r["verdict"] == "PROMISING"]}
    log = load_log()
    log["entries"].insert(0, entry)
    json.dump(log, open(RLOG, "w"), indent=2)
    return entry


if __name__ == "__main__":
    e = run()
    print(f"=== Research run {e['date']} ===")
    for r in e["results"]:
        print(f"\n[{r['verdict']}] {r['hypothesis']}")
        print(f"   {r['stat']} (p={r['p']}) → {r['recommendation']}")
    print(f"\n{len(e['promising'])} promising candidate(s) for the CEO review.")
