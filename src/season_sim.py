"""Monte Carlo simulation of the 2026 NFL season from the ratings engine.

Margin ~ Normal(hfa + b*net_diff + rest, SIGMA). Simulates all 272 regular
season games, seeds playoffs (top-4 division winners by record, 3 wildcards),
plays single-elim brackets with better-seed home field.
Tiebreaks: win pct, then random (full NFL tiebreakers omitted — noise at 20k sims).
"""
import numpy as np
import pandas as pd
import polars as pl

SIGMA = 13.2

DIVISIONS = {
    'AFC East': ['BUF','MIA','NE','NYJ'], 'AFC North': ['BAL','CIN','CLE','PIT'],
    'AFC South': ['HOU','IND','JAX','TEN'], 'AFC West': ['DEN','KC','LAC','LV'],
    'NFC East': ['DAL','NYG','PHI','WAS'], 'NFC North': ['CHI','DET','GB','MIN'],
    'NFC South': ['ATL','CAR','NO','TB'], 'NFC West': ['ARI','LA','SEA','SF'],
}
TEAM_DIV = {t: d for d, ts in DIVISIONS.items() for t in ts}
TEAM_CONF = {t: d.split()[0] for d, ts in DIVISIONS.items() for t in ts}


def simulate(ratings_csv, sched_parquet, coefs_npy, n_sims=20000, seed=7):
    rng = np.random.default_rng(seed)
    r = pd.read_csv(ratings_csv).set_index('team')['net'] if 'net' in pd.read_csv(ratings_csv).columns else None
    rr = pd.read_csv(ratings_csv)
    net = dict(zip(rr['team'], rr['off'] - rr['deff'])) if 'net' not in rr.columns else dict(zip(rr['team'], rr['net']))
    coefs = np.load(coefs_npy, allow_pickle=True).item()
    hfa, b_net, b_rest = coefs['bm']

    sched = pl.read_parquet(sched_parquet).to_pandas()
    g = sched[(sched['season'] == 2026) & (sched['game_type'] == 'REG')].copy()
    teams = sorted(set(g['home_team']) | set(g['away_team']))
    tidx = {t: i for i, t in enumerate(teams)}

    mu = np.array([
        (hfa if row['location'] != 'Neutral' else 0)
        + b_net * (net[row['home_team']] - net[row['away_team']])
        + b_rest * ((row['home_rest'] or 7) - (row['away_rest'] or 7))
        for _, row in g.iterrows()])
    hi = g['home_team'].map(tidx).values
    ai = g['away_team'].map(tidx).values
    n_g = len(g)

    wins = np.zeros((n_sims, len(teams)), dtype=np.int16)
    div_champ = np.zeros(len(teams)); made_po = np.zeros(len(teams))
    conf_champ = np.zeros(len(teams)); sb_champ = np.zeros(len(teams))

    def game_wp(a, b, home_edge=True):
        m = (hfa if home_edge else 0) + b_net * (net[a] - net[b])
        return 1 - _phi(-m / SIGMA)

    from math import erf, sqrt
    def _phi(x): return 0.5 * (1 + erf(x / sqrt(2)))

    for s in range(n_sims):
        margins = mu + rng.normal(0, SIGMA, n_g)
        hw = margins > 0
        w = np.zeros(len(teams), dtype=np.int16)
        np.add.at(w, hi, hw.astype(np.int16))
        np.add.at(w, ai, (~hw).astype(np.int16))
        wins[s] = w

        # seeding
        for conf in ('AFC', 'NFC'):
            champs, rest = [], []
            for d, ts in DIVISIONS.items():
                if not d.startswith(conf): continue
                order = sorted(ts, key=lambda t: (w[tidx[t]], rng.random()), reverse=True)
                champs.append(order[0]); rest += order[1:]
            champs = sorted(champs, key=lambda t: (w[tidx[t]], rng.random()), reverse=True)
            wc = sorted(rest, key=lambda t: (w[tidx[t]], rng.random()), reverse=True)[:3]
            seeds = champs + wc  # 1..7
            for t in champs: div_champ[tidx[t]] += 1
            for t in seeds: made_po[tidx[t]] += 1
            # playoff: 1 bye; (2v7)(3v6)(4v5); reseed; conf champ
            alive = seeds
            r1 = [alive[0]]
            for a_i, b_i in ((1,6),(2,5),(3,4)):
                a,bt = alive[a_i], alive[b_i]
                r1.append(a if rng.random() < game_wp(a,bt) else bt)
            r1 = sorted(r1, key=lambda t: seeds.index(t))
            r2 = []
            for a,bt in ((r1[0],r1[3]),(r1[1],r1[2])):
                r2.append(a if rng.random() < game_wp(a,bt) else bt)
            r2 = sorted(r2, key=lambda t: seeds.index(t))
            cc = r2[0] if rng.random() < game_wp(r2[0], r2[1]) else r2[1]
            conf_champ[tidx[cc]] += 1
            if conf == 'AFC': afc = cc
            else: nfc = cc
        sb = afc if rng.random() < game_wp(afc, nfc, home_edge=False) else nfc
        sb_champ[tidx[sb]] += 1

    out = pd.DataFrame({
        'team': teams,
        'mean_wins': wins.mean(0).round(2),
        'win_total_median': np.median(wins, 0),
        'p_div': (div_champ / n_sims).round(4),
        'p_playoffs': (made_po / n_sims).round(4),
        'p_conf': (conf_champ / n_sims).round(4),
        'p_sb': (sb_champ / n_sims).round(4),
    }).sort_values('p_sb', ascending=False)
    out['division'] = out['team'].map(TEAM_DIV)
    return out, wins, teams
