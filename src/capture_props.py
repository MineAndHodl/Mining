"""Full multi-book prop capture — snapshot EVERY book's full prop menu for the whole slate,
timestamped, so we build our own open->close panel and can measure real prop CLV.

DraftKings (and the field) post the core prop menu ~9+ days out, so we capture from the
opener onward — the earliest snapshot is the most valuable (max time-to-close for CLV).
Each run appends one snapshot to data/prop_snapshots/; pair an early one with a pre-kick one
to grade CLV per book and per prop, and to catch reverse line movement (the free sharp-money
proxy). Cross-book dispersion is where the real prop edge lives (9.6-pt avg range vs 0.3 on
game sides), so we keep ALL books, not just DK.
"""
import os, sys, json, datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import line_capture as lc

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
SNAP_DIR = os.path.join(DATA, "prop_snapshots")
# the full core menu books post this far out (rush/pass attempts & completions post closer in)
MARKETS = ("player_pass_yds,player_pass_tds,player_rush_yds,player_reception_yds,"
           "player_receptions,player_anytime_td,player_1st_td,player_rush_attempts,"
           "player_pass_attempts,player_pass_completions")


def capture(key=None, days=14, regions="us,us2"):
    """One snapshot of the whole slate's props across all books. Saves a parquet, returns it."""
    key = key or os.environ.get("ODDS_API_KEY")
    os.makedirs(SNAP_DIR, exist_ok=True)
    events = lc._get(f"{lc.API}/sports/{lc.SPORT}/events?apiKey={key}"
                     f"&commenceTimeFrom={datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}"
                     f"&commenceTimeTo={(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(days=days)):%Y-%m-%dT%H:%M:%SZ}")[0]
    ids = [e["id"] for e in events]
    df, meta = lc.fetch_props_for_events(key=key, event_ids=ids, markets=MARKETS, regions=regions)
    if not len(df):
        print("no props posted in window"); return df
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(SNAP_DIR, f"nfl_props_{ts}.parquet")
    df.to_parquet(path)
    # summary + cross-book dispersion among the OVER lines (the shopping signal)
    o = df[df.get("prop_side", "").eq("Over")] if "prop_side" in df.columns else df
    disp = []
    for (pl, mk), g in o.groupby(["side", "market"]):
        pts = g["point"].dropna()
        if g["book"].nunique() >= 4 and len(pts) >= 4:
            disp.append(float(pts.max() - pts.min()))
    print(f"captured {len(df)} rows | {df['game_id'].nunique()} games | {df['book'].nunique()} books | "
          f"{df['market'].nunique()} markets | credits {meta.get('remaining')}")
    print(f"saved {os.path.relpath(path, DATA)}")
    if disp:
        print(f"cross-book OVER-line dispersion: median {np.median(disp):.1f} pts, "
              f">=1pt in {np.mean([d>=1 for d in disp])*100:.0f}% of player-markets (n={len(disp)})")
    return df


def list_snapshots():
    if not os.path.isdir(SNAP_DIR):
        return []
    return sorted(f for f in os.listdir(SNAP_DIR) if f.endswith(".parquet"))


if __name__ == "__main__":
    capture(days=int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 14)
    print("snapshots on disk:", list_snapshots())
