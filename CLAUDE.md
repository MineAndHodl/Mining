# The Desk — project brain

**The full operating prompt is `THE_DESK_PROMPT.md` — read it first; this file is the short form.**

Read this first, every session. It is the accumulated clarity of a long iterative build:
what we're doing, how we work, what's been proven, what every variable is for, and what's
still owed. Treat the *findings* as settled (don't rederive them) and the *principles* as
non-negotiable. The owner (Alex) plays owner; the assistant plays head quant.

## Mission
A professional NFL betting model + a monetized public picks brand (X + paid Discord,
Wavey-style). Maximize betting return, hit some big slips, drive people to a paid product.
The board is the product: every pick on it must be something a sharp would nod at.

## Operating principles (the alpha is mostly HOW we work)
1. **Measure, never assert.** "The market is efficient" is a hypothesis to backtest, not a
   conclusion to state. If a projection is anchoring to the line, stop — that guarantees
   zero edge. When pushed, the escape from both caving and false confidence is: go measure.
2. **Calibration is the product.** A 59% is worthless unless ~59% of them hit. Validate
   out-of-sample (walk-forward) before trusting any number, and adopt a model only if it wins.
   More structure is not automatically more accuracy (xTD v1 was a wash at Week 1).
3. **The distribution is half the model.** The spread/shape decides the bet as much as the
   mean. Measure variance *conditionally* (by volume, role), never pooled — a flat CV
   manufactured a whole board of phantom Unders.
4. **Exhaustive variable inventory, per bet type — sewn into the fundamentals (Alex's rule).**
   For every market, enumerate every stat or variable we can access and find a way to apply
   it to *that* bet. Every lever on a pick's scorecard must fit the market (TDs get goal-line
   share and vultures, not yardage sliders). A lever sits at n/a only when the data genuinely
   isn't there — never because we didn't wire it. The inventory below is the living list;
   new data sources get added to it, and new features earn their weights (walk-forward),
   they are not assumed.
5. **Coherence check on every pick.** Would a sharp nod or squint? "Project 78, bet under
   74.5" fails even when the math is right; a card whose sliders contradict its paragraph
   fails. Fix the model or fix the framing — never ship the contradiction.
6. **Think downstream.** A change to the distribution rebalances best bets, the missile, the
   SGPs, and the credit bill at once. Trace every fix through every tab before shipping.
7. **Own the QC as an adversary, every run.** Don't wait for the owner to catch it. What's
   the worst pick on this board and why is it here? Which slider is blank and is that honest?
8. **Every pick gets a handicapper paragraph** rooted in its own data (recent games, the
   role, the coach, the opponent, the environment, the price) and a scorecard whose sliders
   agree with the words. Honest caveats stay in (early-season reads, one-sided markets).

## Validated findings (settled — carry forward)
- **Recency is the core yardage edge.** Last-3-game form vs season baseline predicts the
  residual: +8.6% ROI 2024, +13.3% 2025, out-of-sample. Strong-trend picks hit 57-64%.
- **Prior-year matchup (defense vs position) does NOT predict** (corr ~0). It stays as a
  down-weighted lever, not a driver.
- **Multi-factor ensemble beats single-factor recency** (ridge on [form3, form1, def_delta,
  total_c, home, base_c]; walk-forward weights; stable across L2 3-30). `multifactor.py`.
- **Market-anchoring is circular.** Copying the market guarantees no edge; "market efficient"
  was a modeling artifact.
- **CV is volume-dependent.** rush_yds CV 0.83 low-usage vs ~0.53 for 70+-yd backs; rec_yds
  0.89 vs 0.57; a workhorse is steadier. `props_model.cv_for`.
- **Mean ≠ median on skewed yardage**, but the correction is modest for high-volume players;
  compare our mean to the market-implied mean, never to the line itself.
- **Correlations are measured, not assumed:** QB-WR 0.696, two-WR 0.507, same-player
  different-skill (pass+rush) ~0.10. Gaussian copula; de-vig 0.96/leg. Our fair SGP matched
  DK within 1% on a 3-leg and converged on a 6-leg. `correlation_lab.py`.
- **TD structure (coaching_findings.json):** goal-line carries → rush TDs corr 0.85 (total
  carries 0.75); RZ targets → rec TDs 0.77; lead back gets only ~55% of goal-line carries
  (vultures are real); goal-line role only 29% sticky year-over-year; coach pass tendency
  persists if the coach stays (0.44) and flips on a change (-0.67).
- **xTD (decomposed expected TDs) — `xtd_model.py`, `xtd_validation.json`:** equal to the
  pooled model at Week 1 (both draw the same stale prior); **wins in-season by 2-3% Brier,
  replicated in 2024 and 2025** — its edge is responsiveness to current-season role and
  tendency changes. Adopted with linear recalibration (a=.084, b=.812); blended with the
  pooled model at w=.55 in Week 1, ramping to .85 by Week 5.
- **Offseason form is half-strength (measured):** slope of residual on last-3 form is 0.334
  in-season but 0.164 in weeks 1-3 (0.146 in Week 1); by market rush 40%, pass 35%, rec 65%,
  rec'ns 53% of in-season. The single fitted form weight over-applied across the gap roughly
  doubled every Week-1 edge. `multifactor.OFFSEASON_FORM_RATIO`, applied in `prop_board`.
- **Week-1 baseline must be the market's role read:** every absurd Week-1 edge (+45% on a
  rushing under, +21% on Jefferson under) traced to a stale last-season baseline blind to a
  role change the market had priced (backup now starting, rookie's full role). With no
  current-season games, the market-implied mean is the role baseline and our edge is the
  discounted tilts on top — our alpha, the market's role read; NOT anchoring, since we revert
  to our own validated baseline once current-season games exist. Result: PICK edges max +15%,
  median +5.6% (was +44% / +11%).
- **Grades & rails (`grading.py`):** Desk Score is a conviction rank, not a win probability.
  A+ >=82 (2u), A 76-81 (1.5u), A- 70-75 (1u), B+ 64-69 (0.5u, the recommendation floor),
  B lean (shown, not counted), C reference. Trust term (sample depth, book breadth) separates
  the top. Rails: early-season sizing cap (wk 0-1 A-, wk 2-3 A), max 3 per thesis, 2 per
  game, 15u/week; a validated macro thesis (backtested, in MACRO_THESES) may raise its cap.
  Record rule: only recommended straight bets count; parlays are a separate lotto-lane record.
  Games grade on the same scale (~1 pt spread value ~= 3% win prob) — the most efficient
  market, so few games clear B+.
- **Injury layer (`injury_layer.py`, `role_multiplier.py`, `role_multiplier.json`):** own status
  from the weekly report (Out → never a pick; Doubtful/Questionable → trust cut, slider negative
  on either side) PLUS the measured role cascade from 2023-25 pbp: **RB1 out → RB2 +.55 / RB3
  +.32 of the vacated carry share** (n=37); **WR1 out → targets spread** (WR2 +.20, WR3 +.13,
  TE1 +.07, backs +.08; n=36) and **the QB's passing yards RISE ~13%** (game-script confounding)
  — so NO QB downgrade for a WR1 absence. Applied to projected volume, xTD zone shares, the
  Injury Picture slider and the paragraph. 2026 loader falls back to the nflverse release CSV
  and returns empty cleanly until reports post. **Weights EARNED, not assumed
  (`injury_validate.py`, 583 beneficiary player-games 2024-25):** realized-vs-predicted yardage
  bump slope k*=0.84 (RB2 .98, RB3 1.08, WR2 .73, WR3 .91, TE1 1.28) — the cascade translates
  nearly 1:1 into yards, applied at 0.84; without it beneficiaries are under-projected by ~8
  yds. Questionable players who play produce like healthy ones (median ratio .845 vs .849) —
  NO performance discount, only a scratch-risk flag and a small trust cut.
- **Five seasons of play-by-play (2021-25) — `ingest_pbp.py`, `history_compare.py`:** 176k
  plays with epa/air-yards/YAC/score-state/WP (the substrate for the EARN backlog). Tested 3 vs
  5 seasons walk-forward on 2025: **multifactor WINS with 5** (strong-trend sign-hit .594 →
  .619, residual MAE down) → adopted, weights re-fit; **xTD a dead heat** (Brier .1662 both);
  CV tiers, offseason form ratio and the role cascade **replicate almost exactly** (RB2 .545 →
  .567 on 63 vs 37 team-seasons; WR2 .199 both) and the deeper sample fixed a 9-sample TE1
  fluke. Cascade weight re-earned at k*=.87. `prop_player_games.parquet` is now the 5-season
  table (3-season kept as `_3y`).
- **Mean-reversion must be ZERO under the Week-1 market baseline.** The base_c weight was
  earned on OUR noisy expanding-mean baseline (high sample means overstate, so shrink). The
  market's number is already shrunk; applying the term to it double-shrank every star to the
  Under (board 194 U / 71 O). Zeroed in market-baseline mode → 133/124, median edge 2.7%.
- **Beat intel layer (`player_intel.py`, `data/player_intel_<season>_w<week>.json`):** per-slate
  beat/pundit pass, every item dated + sourced and gated to the week (a 2023 injury story and a
  2025 'ruled out' both surfaced in search and were correctly discarded). Hard facts (ruled
  out → OUT; pitch count → LIMITED) gate the board; soft signals (health nuance, coach intent,
  role news, situational) drive the Beat Intel lever + "Beat note" and are logged on the row
  (intel_signals) for the grading loop to earn a weight — they do NOT move the projection yet.
  Week-1 finding: intel leaned against 6 of 9 unders — partly real (Rice ramp-up → Worthy;
  London's 56.6% share with Penix), partly the known optimism bias of camp coverage. Many 2026
  staff changes surfaced (BAL Minter/Doyle, CLE Monken, ATL Stefanski, DET OC Petzing, LAC
  McDaniel) — the coach-change flip matters this year. Automation blocked: Google News RSS
  and ESPN RSS not reachable from the session; needs a news API key or the weekly in-session pass.
- **Structural truths:** near-locks are minus-money and scarce (books balance ~50/50), so
  "8-10 near-locks at 100:1" is self-contradictory — the missile is an honest lotto of
  53-60% favorites. Week 1 is the weakest week (recency needs games across the offseason
  gap). Heavy TD favorites (-195) are usually NEGATIVE-edge Yes bets; TD value lives on
  moderate dogs. Anytime-TD needs its own Desk Score anchor (0.35 neutral, not 0.52).

## Variable inventory — every accessible input, mapped to each bet (principle #4)
Status: **WIRED** = in the model or scorecard now · **EARN** = data is here, feature not yet
in the model — add it and let walk-forward earn its weight · **NEED** = data source to add.

**Data we hold:** nflverse play-by-play 2023-25 (`data/pbp_2023_2025.parquet`: zone,
play type, rusher/receiver/passer, TD flags, epa, yards, air yards, YAC, down/distance, score
diff, WP, clock, drive, xpass/PROE inputs), 5-season player-game table, player
stats, schedules (spread, total, coach, roof, wind, temp, rest), players (position), 2026
depth snapshot, FFC ADP, Odds API (8 books: props, alt lines, game lines), def_vs_rb/wr,
team_game_epa, game_crew_stats (officials), hist_open_close (line history), CLV backtests,
correlation measurements, DK SGP calibration quotes.

### Yardage / count props (rec_yds, rush_yds, pass_yds, receptions, attempts)
- WIRED (earned weights): recent form (last-3, last-1 vs baseline) · prior-year matchup
  (weak) · game total · home/away · mean-reversion (baseline vs market median).
- WIRED (scorecard/rationale): **injury own-status gate + measured role cascade** · game script from spread (dog throws, favorite runs) ·
  weather (roof/wind/temp) · play-calling load (targets, carries) · market role read
  (ADP; NFL-track-record fallback) · opponent defense incl. QB pass-defense proxy.
- EARN: team pace / plays per game (pbp) · coach pass rate / PROE (stable in-season) · QB-change effect (depth) ·
  opponent EPA allowed by pass/rush (team_game_epa) · target share, air yards, aDOT (pbp) ·
  red-zone vs open-field split of a player's volume (pbp) · officials' crew pace/penalty
  tendencies (game_crew_stats) · line movement & book disagreement (hist_open_close, 8-book
  dispersion, CLV engine) · rest / bye / travel (schedule) · QB change (depth snapshot) ·
  division / primetime situational flags · score-differential-conditional usage (garbage
  time, from WP).
- NEED: snap counts (nflverse snap_counts) · targets per route run · practice reports beyond
  the injury report · game-day inactives (the report is Wed-Fri; inactives drop 90 min pre-kick).

### Anytime TD
- WIRED (xTD): expected team plays per zone (GL 1-5 / IN10 / RZ 11-20 / OPEN) · coach
  goal-line & red-zone pass-vs-run rate (with coach-change flip) · player carry share and
  target share per zone (the red-zone role; the vulture shows up as leaked goal-line share) ·
  per-zone conversion vs position prior (finishing) · environment multiplier from implied
  team points · last-season TDs vs role-expected (regression) · Week-1 goal-line shrink
  (29% stickiness) · blend with pooled model (ramps in-season) · linear recalibration.
- EARN: opponent red-zone defense (TDs allowed per RZ trip, pbp) · QB-sneak share of
  goal-line carries (already in pbp as a rusher; make it explicit as a vulture) · 2-point
  and short-field (turnover/field-position) drive starts · red-zone route share ·
  score-state (trailing teams throw at the goal line).
- NEED: goal-line personnel/formation · injuries (a starter out reshuffles every share).

### Games (spreads, totals, moneylines)
- WIRED: Desk Confidence (off-market points vs consensus, key numbers) · CLV engine · multi-
  book line capture · book-difference read (DK baseline).
- EARN: team offensive/defensive EPA (team_game_epa) · PROE and pace · rest / travel /
  bye · officials (crew stats → totals) · weather → totals · QB status (depth) · home/away
  splits · situational (division, primetime, lookahead) · line-movement direction.

### Same-game parlays (primetime SGP) and the Whale Missile
- WIRED: measured correlations + copula fair price · DK calibration (2 quotes) · alt-line
  ladders tuned to a hit-probability band · one market per player · at most one near-lock
  TD leg · missile = market-favorite legs across different games, honest expected-hits.
- EARN: game-script-conditional correlations (blowout vs close) · more DK quotes for
  calibration (owner can paste slips).

## Pipeline (src/)
`run_board.py` is the weekly run: odds → `prop_board` (multifactor projection, volume-CV
pricing, 8-book line shop) → `td_board` (pooled + xTD blend) → `rank_board` (conviction
Desk Score, diversity cap, TD anchor) → `board_factors`/`factor_scorecard` (per-market
scorecards) → `whale_missile` + `primetime_sgp` → `build_board_page` → `output/the-board.html`.
`--cached` re-ranks offline from `_raw_*` parquets (no credits; won't recompute pricing).
Research/validation: `multifactor.py`, `recency.py`, `xtd_model.py`, `xtd_validate.py`,
`correlation_lab.py`, `clv_engine.py`. Tests: `python3 tests/test_pipeline.py` (must be green).

## Still owed
- **Forward grading loop** (the biggest missing foundation — next up): log every pick's number and
  factor contributions, grade vs close and outcome, re-learn weights. Turns a static model
  into one that improves. Offered repeatedly; not yet greenlit.
- Injuries/inactives feed; snap counts. Then the EARN features above, one at a time, each
  proven walk-forward before it ships.
- SGP payout/EV balance tuning (30-100:1 target) and stronger SGP selling copy.
- Blocked on owner: X/Discord/Whop keys; DK SGP quotes for more calibration points.

## Workflow notes
Direct GitHub push is blocked from this session (repo not in the authorized set); commit
locally and refresh `/home/claude/beat-the-close.bundle` (`git bundle create ... main`),
which the owner pulls. Commit messages end with the session's attribution footer.
