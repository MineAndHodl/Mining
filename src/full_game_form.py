"""Full-game form (Fresh Eyes day 2) — does the recent-form window get poisoned by PARTIAL games?

The hypothesis (handicapper hat): the market knows a star left in the first quarter, or sat the
second half of a Week-18 rest game; our `last3` treats those as ordinary bad games and drags
form3 down for the next three weeks — a phantom Under every time. Across the offseason it is
worse: weeks 17-18 rest/backup games are the LAST games in the window for every Week-1 number.

The test (the repo's way, feature_lab-style, two held-out seasons):
  opportunities per player-game from play-by-play (carries / targets / pass attempts by market);
  a game is PARTIAL when its opportunities are below `PARTIAL_FRAC` of the player's median over
  his previous `LOOKBACK` games; form3_full / form1_full are the same features computed over the
  last 3 / last 1 FULL games. Compare the ridge residual model with the standard features vs the
  full-game features (replacement) and vs both (addition), on 2024 and 2025 separately, for all
  weeks and for weeks 1-3 (the offseason gap) — strong-signal sign-hit, MAE, ROI proxy.
Adopt only if it wins both seasons. Result -> data/full_game_form.json (never re-test a wash).

    python3 src/full_game_form.py
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multifactor as mf
import feature_lab as fl

DATA = mf.DATA
PARTIAL_FRAC = 0.5      # a game with < 50% of his usual opportunities is a partial game
LOOKBACK = 8            # "usual" = median opportunities over his previous 8 games
MARKETS = ["rush_yds", "rec_yds", "receptions", "pass_yds"]
OPP_KIND = {"rush_yds": "carries", "rush_att": "carries", "rec_yds": "targets", "receptions": "targets", "pass_yds": "attempts"}


def opportunities():
    """(pid, game_id) -> carries, targets, attempts from play-by-play."""
    cols = ["season", "week", "game_id", "rush_attempt", "pass_attempt", "rusher_player_id", "receiver_player_id", "passer_player_id", "season_type"]
    p = pd.read_parquet(f"{DATA}/pbp_2021_2025.parquet", columns=cols)
    if "season_type" in p.columns:
        p = p[p.season_type == "REG"]
    c = p[p.rush_attempt == 1].groupby(["rusher_player_id", "game_id"]).size().rename("carries")
    t = p[p.pass_attempt == 1].groupby(["receiver_player_id", "game_id"]).size().rename("targets")
    a = p[p.pass_attempt == 1].groupby(["passer_player_id", "game_id"]).size().rename("attempts")
    out = pd.concat([c.rename_axis(["pid", "game_id"]), t.rename_axis(["pid", "game_id"]), a.rename_axis(["pid", "game_id"])], axis=1).fillna(0.0)
    return out.reset_index()


def add_full_game_form(pg, frac=PARTIAL_FRAC, lookback=LOOKBACK):
    """Adds opp, partial, last3_full, last1_full, form3_full, form1_full, n_partial_in_window."""
    opp = opportunities()
    pg = pg.merge(opp, on=["pid", "game_id"], how="left")
    pg["opp"] = [r[OPP_KIND.get(m, "carries")] for r, m in zip(pg[["carries", "targets", "attempts"]].to_dict("records"), pg["mkt"])]
    pg["opp"] = pg["opp"].fillna(0.0)
    pg = pg.sort_values(["pid", "mkt", "season", "week"]).reset_index(drop=True)
    l3f = np.full(len(pg), np.nan); l1f = np.full(len(pg), np.nan); npart = np.zeros(len(pg)); is_part = np.zeros(len(pg), dtype=bool)
    for _, idx in pg.groupby(["pid", "mkt"]).indices.items():
        y = pg["y"].to_numpy(float)[idx]; o = pg["opp"].to_numpy(float)[idx]
        for j in range(len(idx)):
            if j == 0:
                continue
            lo = max(0, j - lookback)
            prev_o = o[lo:j]; prev_y = y[lo:j]
            med = float(np.median(prev_o)) if len(prev_o) else 0.0
            full = prev_o >= frac * med if med > 0 else np.ones(len(prev_o), dtype=bool)
            # was THIS game (j-1, the most recent one) partial vs the games before it?
            if j >= 2:
                m2 = float(np.median(o[max(0, j - 1 - lookback):j - 1]))
                is_part[idx[j - 1]] = (m2 > 0 and o[j - 1] < frac * m2)
            keep_y = prev_y[full]
            last3_std = y[max(0, j - 3):j]
            npart[idx[j]] = int((~full[-3:]).sum())
            if len(keep_y) >= 2:
                l3f[idx[j]] = float(np.mean(keep_y[-3:])); l1f[idx[j]] = float(keep_y[-1])
            else:
                l3f[idx[j]] = float(np.mean(last3_std)); l1f[idx[j]] = float(y[j - 1])
    pg["last3_full"] = l3f; pg["last1_full"] = l1f; pg["n_partial_in_window"] = npart; pg["partial"] = is_part
    pg["form3_full"] = pg["last3_full"] - pg["baseline"]; pg["form1_full"] = pg["last1_full"] - pg["baseline"]
    return pg


def _eval(d, feats, test_season, weeks=None, l2=10.0):
    out = {}
    for mkt in MARKETS:
        s = d[d.mkt == mkt]
        tr = s[s.season < test_season]; te = s[s.season == test_season]
        if weeks is not None:
            te = te[te.week.isin(weeks)]
        if len(tr) < 300 or len(te) < 60:
            continue
        X = tr[feats].to_numpy(float); yv = tr["resid"].to_numpy(float)
        w = np.linalg.solve(X.T @ X + l2 * np.eye(X.shape[1]), X.T @ yv)
        pred = te[feats].to_numpy(float) @ w; act = te["resid"].to_numpy(float)
        strong = np.abs(pred) >= np.quantile(np.abs(pred), 0.6)
        hit = float((np.sign(pred[strong]) == np.sign(act[strong])).mean())
        mae = float(np.abs(pred - act).mean())
        out[mkt] = {"hit": round(hit, 4), "mae": round(mae, 3), "roi": round(hit * (100 / 110) - (1 - hit), 4), "n": int(len(te)),
                    "n_strong": int(strong.sum()), "w": {f: round(float(x), 3) for f, x in zip(feats, w)}}
    return out


def run(seasons=(2024, 2025)):
    d = mf.build_features()
    d = add_full_game_form(d)
    d = d.dropna(subset=["form3_full", "form1_full"])
    BASE = list(mf.RAW_FEATURES)
    REPL = ["form3_full", "form1_full", "def_delta", "total_c", "home", "base_c"]
    ADD = BASE + ["form3_full"]
    res = {"design": {"partial_frac": PARTIAL_FRAC, "lookback": LOOKBACK, "definition": "partial = opportunities (carries/targets/attempts by market) < 50% of his median over the previous 8 games; full-game form = last 3 / last 1 FULL games"},
           "descriptive": {}, "by_season": {}}
    # how common is the contamination we fear?
    desc = {}
    for mkt in MARKETS:
        s = d[d.mkt == mkt]
        wk1 = s[s.week <= 3]; ins = s[s.week >= 4]
        desc[mkt] = {"partial_rate_all": round(float(s["partial"].mean()), 4),
                     "partial_rate_wk17_18": round(float(s[s.week >= 17]["partial"].mean()), 4),
                     "wk1_3_windows_with_partial": round(float((wk1["n_partial_in_window"] > 0).mean()), 4),
                     "wk4plus_windows_with_partial": round(float((ins["n_partial_in_window"] > 0).mean()), 4),
                     "mean_form3_shift_when_contaminated_wk1_3": round(float((wk1.loc[wk1.n_partial_in_window > 0, "form3_full"] - wk1.loc[wk1.n_partial_in_window > 0, "form3"]).mean()), 2) if (wk1.n_partial_in_window > 0).any() else None}
    res["descriptive"] = desc
    wins = {"repl": 0, "add": 0}; cells = 0; hurt_both = {"repl": 0, "add": 0}
    for te in seasons:
        row = {}
        for label, weeks in (("all", None), ("wk1_3", [1, 2, 3]), ("wk4plus", list(range(4, 19)))):
            base = _eval(d, BASE, te, weeks); repl = _eval(d, REPL, te, weeks); add = _eval(d, ADD, te, weeks)
            row[label] = {}
            for mkt in base:
                row[label][mkt] = {"n": base[mkt]["n"], "n_strong": base[mkt]["n_strong"],
                                   "hit_base": base[mkt]["hit"], "hit_repl": repl.get(mkt, {}).get("hit"), "hit_add": add.get(mkt, {}).get("hit"),
                                   "mae_base": base[mkt]["mae"], "mae_repl": repl.get(mkt, {}).get("mae"), "mae_add": add.get(mkt, {}).get("mae"),
                                   "w_add_form3_full": add.get(mkt, {}).get("w", {}).get("form3_full")}
                if label == "all":
                    cells += 1
                    for k in ("repl", "add"):
                        v = row[label][mkt]
                        wins[k] += (v[f"hit_{k}"] is not None and v[f"hit_{k}"] - v["hit_base"] > 0.005 and v[f"mae_{k}"] <= v["mae_base"])
        res["by_season"][str(te)] = row
    for k in ("repl", "add"):
        for mkt in MARKETS:
            if all(((res["by_season"][str(te)]["all"].get(mkt) or {}).get(f"hit_{k}") or 0) - ((res["by_season"][str(te)]["all"].get(mkt) or {}).get("hit_base") or 0) < -0.005 for te in seasons):
                hurt_both[k] += 1
    res["wins"] = wins; res["cells"] = cells; res["hurt_both_seasons"] = hurt_both
    res["verdict"] = {k: ("WINS" if (wins[k] >= cells * 0.5 and hurt_both[k] == 0) else ("LOSES" if hurt_both[k] else "WASH")) for k in wins}
    return res


def print_result(res):
    print("\n=== full-game form: partial-game contamination of the form window ===")
    for mkt, v in res["descriptive"].items():
        print(f"  {mkt:11s} partial games {v['partial_rate_all']:.1%} of all, {v['partial_rate_wk17_18']:.1%} in wks 17-18; "
              f"wk1-3 windows touched {v['wk1_3_windows_with_partial']:.1%} (wk4+ {v['wk4plus_windows_with_partial']:.1%}); form3 shift when touched {v['mean_form3_shift_when_contaminated_wk1_3']}")
    for te, row in res["by_season"].items():
        for label, mk in row.items():
            for mkt, v in mk.items():
                print(f"  {te} {label:7s} {mkt:11s} n={v['n']:5d} hit base {v['hit_base']:.3f} repl {v['hit_repl']} add {v['hit_add']} | mae {v['mae_base']:.2f} / {v['mae_repl']} / {v['mae_add']} | w_add={v['w_add_form3_full']}")
    print(f"  verdict: {res['verdict']} (wins {res['wins']} of {res['cells']} market-seasons; hurt both {res['hurt_both_seasons']})")


if __name__ == "__main__":
    r = run()
    print_result(r)
    json.dump(r, open(f"{DATA}/full_game_form.json", "w"), indent=1)
