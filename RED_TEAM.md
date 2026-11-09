# RED TEAM — the ten holes a betting quant would poke, the rebuttal, the plan (Nov 8 2026)

Numbers measured for this review: Week 1 pool re-graded row by row (118 yardage/count props at the
opener): hit .551; actual landed on our side of the market-implied mean .559; by claimed probability
(0-.53 → .500 n=22, .53-.56 → .510 n=49, .56-.60 → .552 n=29, .60-.65 → .750 n=16, >.65 → .500 n=2),
slope of realized on claimed 1.4. Interval coverage (20.8k held-out outcomes): below our p10 16.5% (rush)
/ 17.8% (rec) vs 10% designed; above p90 10.7% / 11.0%. Copula validation (2,978 team-games, legs = over
the player's own median): QB+WR1 realized .370 vs copula .326 vs independent .245; QB+WR1+WR2 .295 vs .240
vs .121; +RB .130 vs .112 vs .059.

1. **The backtest beats our own baseline, not the market.** The 57-64% strong-signal hit and the ROI
   figures use the player's expanding mean as the line — a line nobody posts, and the market already prices
   form (measured: QB passing form fully, catch counts 64%). *Rebuttal:* partial. The market-lag module
   discounts what is priced, and the one true market test (Week 1 pool, above) is 55% at the opener with
   claimed probabilities tracking realized — one week, break-even to slightly positive. *Plan:* make
   "vs the closing line" THE acceptance metric: Odds API historical endpoint (line_capture has it) for a
   2024-25 sample; archive every line from Week 9; report CLV and ROI by claimed-edge decile.
2. **Winner's curse.** We price ~300 props and show the 8 largest edges; the max of noisy estimates is
   biased up. *Rebuttal:* partial — rails cap it, A- hit 73% (n=15) but B+ 51%, the >.65 bucket went 1-1.
   *Plan:* empirical-Bayes shrinkage of claimed edge fitted on the graded pool (realized on claimed, slope
   <1 → shrink); show and size on the shrunk edge.
3. **Tails too thin on the left.** 16-18% of outcomes fall below our p10 (designed 10%); the right tail is
   fine. So P(under) at low lines is understated and extreme probabilities (Williams .71, Flowers .67) are
   suspect. *Rebuttal:* partial — measured, known (interval_check), and the open item on the list. *Plan:*
   early-exit / zero-inflated mixture (T4 challenger) by position and injury status; refit CV; re-measure
   coverage; calibrate p>.65 as its own bucket.
4. **SGP correlations are hand-set and shrunk arbitrarily** (Pearson × .85; cross-team pairs typed in).
   *Rebuttal:* partial, direction known — the copula UNDER-predicts realized joint hits (above), so our
   fair prices are too LONG: conservative (we skip good slips rather than take bad ones). *Plan:* fit the
   correlations to realized joint rates (drop the .85 shrink), add cross-team and script-conditional pairs,
   collect DK SGP quotes (owner pastes slips) to calibrate the book, rank copula instead of Gaussian.
5. **Trend hunting is p-hacking.** Thirty cuts of the schedule will find a 65% cover; the big-road-favorite
   fade ran 67% → 40% → 62% by era. *Rebuttal:* none as edges — labeled leans, never Card, era instability
   shown — but still quoted with sizes. *Plan:* pre-registered hypotheses; estimate 2006-19, test 2020-25,
   then live; situational reads enter a penalized games model (team EPA, QB status, rest, line move) that
   must show CLV before a games unit is risked.
6. **Garbage in — this week's boards were hand-fed.** Lines typed from one book's preview hours old, no DK
   prices, 2026 route participation unpublished so the earned role feature ran on 2025 routes, injuries by
   regex over beat notes, no inactives, a starter's name typed into JSON. *Rebuttal:* none. *Plan:* Odds API
   key (DK execution + 8-book shop, line history, Last Call with inactives); role fallback to 2026 snap
   counts with a measured substitution; a coverage score on every pick showing which inputs were live.
7. **Form on raw totals confuses the environment with the player** (the monsoon; blowouts; a QB's exit).
   *Rebuttal:* partial — team-anomaly weighting shipped tonight (receiving yards 0, receptions .7; walk-forward
   not worse; passing tried and rejected). *Plan:* UPGRADE_PLAN §3 — team volume × share × efficiency
   projected separately, precision-weighted share form, adopt only if it beats the raw model walk-forward.
8. **A light-switch baseline at Week 4 and a refusal to shrink toward the market.** The market line is the
   best single estimator of a mean; the optimal in-season projection is a fitted blend. "Anchoring is
   circular" conflates copying with shrinkage. *Rebuttal:* partial — the principle was a reaction to a real
   artifact, and the market mean IS the baseline before Week 4 — but the in-season blend argument stands.
   *Plan:* fit projection = w·model + (1−w)·market walk-forward on the line history; publish w per market
   weekly as the edge meter; if w → 0 the edge is not there and we want to know.
9. **The record keeps summaries, sizing is by assumed bands, and this week's picks are not in the ledger.**
   Desk Score constants are assumed; Week 1 pool results were stored as a summary (re-graded by hand for this
   review); Thursday/SNF picks bypassed the ledger. *Rebuttal:* none. *Plan:* every pick and lean
   pre-registered with a timestamp before kickoff (manual-board path for hand-fed weeks); row-level results
   persisted; fractional Kelly on shrunk p and price with a cap replaces grade bands after 3 graded weeks;
   the calibration table published weekly.
10. **Small n everywhere; "measured" levers at 1.5-2 SE** (debut-QB n=63-135, anomaly n=63, cascade n=37).
    *Rebuttal:* partial — SEs reported, the core features earn weights on 60k rows and two held-out
    seasons, levers are multipliers with trust cuts; Flowers shows the risk anyway. *Plan:* partial-pooling
    estimates that shrink each lever toward zero by its SE; a sunset rule (re-measured each season or it
    drops to n/a); convert multipliers into ridge features so the penalty does the shrinking.

What changes first: (1) acceptance metric = closing line / CLV; (2) pre-registration of every pick into
the ledger starting with this week's grading; (3) re-fit the copula correlations; (4) the Odds API key is
the single biggest unblock and sits with the owner.
