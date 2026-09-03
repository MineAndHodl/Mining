# The Desk — Theses, the pre-season lock, and the in-season plan

Alex, 2026-09-03: *"Before this season we have the thesis(es) and the model structure figured out;
in season we are just calibrating, testing, hypothesizing, capturing new proprietary data, figuring
out how it all interacts, finding edge."*

This is that document. Everything the model believes, with its evidence and its kill criteria;
what freezes on **Tuesday, September 8 (the Week 1 official open)**; what is allowed to move in
season, on what calendar; and the dataset we accumulate that nobody else has.

Status: **VALIDATED** (out of sample, replicated) · **MEASURED** (one measurement, applied) ·
**PROVISIONAL** (applied with a small assumed weight, owed to the loop) · **LOGGED** (captured on
every row, not applied) · **STRUCTURAL** (a product/lane rule, not a prediction).

## 1. The theses

| # | Thesis | Status | Evidence | In-season test | Dies if |
|---|---|---|---|---|---|
| T1 | **The line lags recent usage.** Last-3-game form vs the season baseline predicts the residual; strong signals hit 57-67%. | VALIDATED | multifactor walk-forward 2024/2025; +8.6% / +13.3% ROI proxy | CLV and hit rate of form-driven picks by market, every Monday; market-lag β weekly | strong-signal CLV ≤ 0 over 8 weeks, or hit < 52.4% over 120 picks |
| T2 | **Usage beats efficiency.** Route participation and the snap-share trend add to form; separation, RYOE, CPOE, contact, charting and opponent tendencies do not. | VALIDATED | universe_lab: 98 candidates, two seasons; role features up in every market | market-lag β on the role tilt from week 4; profile levers graded by CLV | role-feature picks under-perform form-only picks over 8 weeks |
| T3 | **Weeks 0-3: the market's number is the role baseline, and it already prices the environment and QB form.** Our tilt is only what the market lags (rushing and receiving-yards form). | MEASURED | market_lag.py cross-section, bootstrap CIs; the 2024/2025 weeks-1-3 evaluation | re-measured every board; retired automatically once players have current-season games | Week 1-3 Card hit < 45% — the lag factors were wrong |
| T4 | **The distribution is half the model**: within-player-season CV by volume tier, extra dispersion for counts. | VALIDATED | cv_measure.py (3,700 player-seasons); DK ladders matched; interval_check.py: the 25-75 range catches 47.5% and the 10-90 range 76% of 20.8k held-out outcomes (misses lean low); widening the CV does not improve P(over) log-loss — the width stands, the left-tail shape is the open item | calibration by model_p bucket over the graded pool; the left tail (early exits) as a challenger | a bucket misses by > 8 pts over 200 picks |
| T5 | **Touchdowns are structure**: zone plays × coach tendency × role share × finishing, on the measured environment scale; stars are overpriced; "who is on the field" reads the role better than a handful of carries/targets. | VALIDATED | xtd/td_env/td_role validations, both seasons; top bucket .50 → .53 | calibration by position and price band monthly; Touchdown Tax record; early-season top bucket | top bucket (≥.45) hits < 45% over 150 picks |
| T6 | **The number is often the edge**: consensus = median of book-implied means; an off-market book is a play when 3+ books define the consensus; the shop is worth points. | MEASURED | day-2 consensus fix; line-history stale flags | off-market lane graded separately; which book is soft (log on every row); Underdog leads or lags (week 6) | off-market picks hit < 50% over 60 |
| T7 | **Injuries cascade measurably** (RB2 +.55, WR2 +.20, QB +13% on a WR1 out); Questionable players who play produce normally. | VALIDATED | injury_validate.py (583 beneficiary games), k*=.87 | late-swap lane record; Last Call inactives | late-swap hit < 50% over 40 |
| T8 | **Beat writers and coaches know first.** Hard facts gate now; soft signals (coach intent, role, health nuance, the crowd) are shown and logged, not weighted. | PROVISIONAL | intel_feed + coach_feed live; 4-pt provisional score weight | earn the weight from 3+ graded weeks (signals vs outcomes); kill or raise | signals show no predictive value after 6 weeks → shown, weight 0 |
| T9 | **Weather moves passing** (validated multiplier; domes neutral). | VALIDATED | weather.py backtest; Open-Meteo forecast live | CLV on wind ≥ 15 games | — |
| T10 | **Games (sides/totals) are not an edge for us yet**; the games lane is market intel. | STRUCTURAL | ratings ~50% ATS; book-gap thesis disproven | CLV log only; officials/pace tested on a branch | promoted only if a games feature wins two seasons |
| T11 | **Parlays are lanes, not the record**: Sunday Stack compounds real edge; Touchdown Tax = 3 most likely +EV scorers; Primetime SGP = the lotto. | STRUCTURAL | correlation_lab (measured copula); band tuning | each lane its own record | — |
| T12 | **The product is the public record**: every Card pick ≥ 50% to hit on our numbers, frozen at open, graded at the posted price; the free pick is the most likely winner; late swap is the paid hook; the Model Room shows what the model consumes. | STRUCTURAL | CARD_MIN_P rail; ledger; post_copy | hit rate and units by week; follower/subscriber counts once posting starts | — |

## 2. The pre-season lock (Tuesday, September 8 — the Week 1 official open)

**Frozen from the official open**: the feature set (8 raw features + the offseason and role ratios),
the CV anchors, the xTD blends and calibration, the market-lag machinery, the grade bands and
rails (CARD_MIN_P .50, 8 picks, A- cap), the lanes and their bands, the pipeline order. A model
change after the open is a branch (`fresh/<date>`) that runs as a **challenger**, never a swap.

**Before the lock (this week):**
- [ ] Alex: pull the bundle and push — every scheduled job (Tuesday card, Thursday Tax, Sunday final,
      Last Call, Monday grading, Fresh Eyes, the Watch) runs from GitHub. Nothing live until then.
- [ ] Tuesday 14:30 UTC: `run_board.py --open --tag=tuesday` — the official Week 1 open, ledger
      `opened_kind = official`, post copy generated.
- [ ] Thursday: Touchdown Tax run. Sunday 11:40 ET: Last Call with inactives. Monday: first grading —
      the record starts.
- [ ] Alex: the persona decision (AI-run brand or a plain brand) and the posting keys — the Card is
      ready to post; nothing posts without him.

## 3. The in-season plan — calibrate, test, capture, connect, find

**Weekly (automatic):** Tuesday open → Thursday Tax → Sunday final + Last Call → Monday grading
with CLV, records by lane, calibration by grade and by model_p → the Monday report card. The Watch
every two hours. Fresh Eyes every morning: one measured improvement, the Owner's Pass, the forecast.

**The calendar of re-earning (parameters move on schedule, structure moves on branches):**
- **Every board**: market-lag β per market (weeks 0-3); line history; second-market lines; intel.
- **Week 4** (3 graded weeks): earn or kill the intel weight (T8); market-lag on the ROLE tilt
  in-season; profile levers vs CLV (which tracking percentile predicts beating the close?); the
  early-season TD top bucket check; xTD switches to in-season shares (already coded).
- **Week 6**: Underdog vs the books — does the second market lead the move? Which book is soft
  (n ≥ 150 off-market rows)? Rookie baseline from draft capital + projections vs the market.
- **Week 9 (mid-season refit, on a branch)**: re-fit the ridge weights with 2026 rows, re-measure CV
  anchors, refit the TD calibration, re-earn the cascade. Runs as a challenger for two weeks;
  promoted only if it beats the incumbent on CLV and hit rate.
- **Week 14**: playoff/rest-week handling (the rest-game rule for role features gets its first
  live test); Week 18 posture (starters resting) written into the rails.
- **After the season**: the retro — every thesis re-scored on the full labeled year.

**Challenger boards (the in-season change mechanism):** any structural idea runs in shadow — the
same slate priced two ways, both frozen in the ledger, both graded — for two weeks before it can
replace the incumbent. Nobody moves a live weight because it "makes sense".

## 4. The proprietary dataset (the moat we are building)
Every week the ledger freezes, for every pick and every pool row: our number, the consensus mean,
the book's number, the second market's number, two outside projections, the tilt and the
off-market gap, the market-lag factor, the intel signals and the coach quotes, the profile
percentiles, the weather, the injury context, the open → close move, and the outcome. Line history
keeps every book's every line at every fetch. The Watch keeps every note, headline, status change,
trending move and line move with a timestamp. The coach feed keeps the transcripts.

By Week 9 that is a labeled set no public source has: *what we thought, what every market thought,
what the beat writers and coaches said, what happened* — the raw material for every in-season
re-earn above, and for next year's model.

## 5. The standing "needs you" list (Alex)
1. Pull the bundle and push (unblocks every job).
2. The brand persona decision and the posting keys (X, Discord bot, Whop).
3. Whether to spend: X API pay-per-use for a curated beat list (~$60/mo) — not needed yet.
4. Declare the official first model launched (the loop keeps running either way).
