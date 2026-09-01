# beat-the-close

NFL betting model desk + content engine. Private.

- `src/engine.py` — EPA power ratings (opponent-adjusted, recency-weighted, asymmetric off/def regression)
- `src/backtest.py` — walk-forward backtest vs closing lines
- `src/season_sim.py` — 20k-season Monte Carlo (futures, explorer data)
- `src/weekly_card.py` — live board: Pinnacle de-vig, off-market flags, Wong teasers, DK execution view
- `src/cards.py` — branded social graphics (Playwright HTML->PNG)
- `output/` — ratings, futures, overlays, published pages
- `archive/` — odds snapshots (opening/line-path/closing), one CSV per pull, committed by scheduled runs
