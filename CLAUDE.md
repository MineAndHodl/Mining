# The Desk — project brain

**The full operating prompt is `THE_DESK_PROMPT.md` — read it first; this file is the short form.
`IDEAS_LEDGER.md` is the audit of every idea Alex ever raised and its honest status — read it
before proposing anything "new". `FRESH_EYES.md` is the daily improvement loop (three hats,
one shipped improvement a day, logged in `data/fresh_eyes_log.json`) — "improve" means run it.
`OWNER.md` is Alex as a persona in that loop (the Owner's Pass runs on everything before it ships;
his next asks are forecast and scored daily). `THESES.md` is what we believe, what freezes at the
Week 1 open, and the in-season calendar of re-earning. `DATA_UNIVERSE.md` is every data source.**

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
8. **Run the architecture review every session, unprompted** (THE_DESK_PROMPT §2b): survivors
   vs selection; upstream vs downstream; truncated pools; earned vs assumed; card/score
   agreement; cost vs coverage; where the model is blind by construction. Selection runs over
   the FULL pool with every signal in the score; expensive signals get a written coverage rule.
9. **Every pick gets a handicapper paragraph** rooted in its own data (recent games, the
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
- **CV is volume-dependent — measured WITHIN a player's season (`cv_measure.py`, Fresh Eyes
  day 1).** pass_yds .55/.39/.29/.28 at 160/205/240/275; rec_yds 1.01 → .53 (13 → 96 yds);
  rush_yds 1.09 → .45 (11 → 97); receptions' EXTRA dispersion beyond Poisson ~.20 (the old
  anchor applied total CV as extra dispersion — variance double-counted). The earlier pooled
  anchors were 10-40% too wide at the high-volume end; DK's alternate-line ladders
  (`alt_ladder_probe.py`) implied the within-player shape, not ours. `data/cv_anchors.json`
  overrides `props_model.CV_ANCHORS`; the Card re-priced 5U/3O → 4/4.
- **Market-role baseline comes from the CONSENSUS line, not each book's line** (Fresh Eyes day
  1): with a per-book baseline every posted line implied its own number, every line looked
  equally +EV and the shop could not prefer BetRivers 77.5 over FanDuel 78.5. Consensus line =
  median across books (display); the role baseline itself is the median of the books' implied
  means (day 2, below); the Card takes the best available number for our side.
- **Feature lab (`feature_lab.py`, two held-out seasons):** game-script spread, short week and
  team pace add NOTHING to the residual model (all WASH) — the obvious handicapper angles are
  already absorbed by form, total, home and baseline. Logged in `data/feature_lab.json`.
  **Full-game form (`full_game_form.py`, day 2) LOSES both seasons:** dropping partial games
  (rest, early exits; opportunities < 50% of usual) from the form window makes every market
  worse — a short game is a real signal about the role. Never re-test.
- **What the market already prices into a weeks-0-3 line (`market_lag.py`, `data/market_lag.json`,
  measured on the board itself, mean-reversion controlled, re-measured every run and pooled):**
  the market's number minus our season baseline moves with the game total at 1.14/pt for passing
  (our earned weight 1.20), home −4.9 (earned −4.1), R² .82 — the market line IS our baseline plus
  our environment tilts, so under the market-role baseline total/home/def_delta are ZERO (like
  base_c); adding them double-counted (Herbert O233.5 +15.6 from the total was a phantom). On
  FORM the market is selective: it prices late-season QB passing form fully (+.148 vs our applied
  +.057, CI [+.07, +.21]) and catch counts (+.168 vs +.150), leaves 75% of receiving-yards form
  and ALL of rushing form (−.08, CI [−.28, +.02]) on the table → `lag_factor` per market on the
  form tilt (pass 0, receptions 0, rec .75, rush 1.0 in Week 1 2026). The Week-1 edge is rushing
  and receiving yards; QB yards and catch counts are reference until current-season games.
- **Per-player form gap (day 2):** the offseason treatment keys off HIS window, not the global
  mode — gap 1 (one offseason) = half-strength form + market-role baseline (also a returner in
  Week 6 with no current-season games); gap ≥ 2 (a missed season) = form OFF, trust ×.7, the
  paragraph names the year (n = 4-12 such rows in five seasons: unmeasurable, so not applied).
- **Consensus = the median of each book's IMPLIED MEAN (day 2)**, not the median line: two books
  at different lines (DK 4.5 / FD 3.5) implied the same mean, and the old median-line code
  manufactured an 8.7% edge between them. Rows carry `tilt` (our read vs consensus) and
  `off_market` (this book's number vs consensus) signed for our side; `thesis_kind` = model /
  off-market (3+ books, the number is the play — told that way) / no-consensus (2 books → REVIEW).
- **Trust reads market breadth (`n_books_mkt`, day 2)**, not books at our shopped line (always 1
  for a best line — a 6-point artifact against every shopped pick). Home/away is read from the
  schedule row (the old string test never matched — home was 0 on every live row).
- **THE DATA UNIVERSE (day 3, `DATA_UNIVERSE.md`, `ingest_universe.py`, `data/data_ledger.json`):** 36
  free feeds reachable from the session — nflverse play-by-play, **participation (who was on the
  field for every play, 2016-25)**, FTN charting, Next Gen Stats, PFR advanced, snap counts, depth
  charts, rosters, injuries, contracts, draft/combine, officials, QBR, schedules; Sleeper (players,
  trending, RotoWire projections), ESPN (core injuries = dated beat notes, fantasy projections,
  news), Underdog lines, Action Network odds, RSS. 3.5M rows, 183M data points. Pull generously,
  adopt stingily: `universe_features.py` → `universe_lab.py` (98 candidates, two held-out seasons).
- **USAGE IS THE EDGE, EFFICIENCY IS NOT (day 3, `data/universe_lab.json`):** of 98 signals, only
  role/usage won — route participation (share of team dropbacks on the field), offense snap %,
  goal-line / red-zone / inside-10 snap inclusion. NGS separation, RYOE, CPOE, YAC-over-expected,
  PFR contact/pressure, FTN catchable/contested/drops/play-action/motion/RPO, opponent man/blitz/
  box/pressure: all WASH beyond form. Adopted `role_rr` (route participation, last 3 games, centered
  per market) + `role_snap_d` (snap-share trend) as raw features (weights re-fit, 8 features):
  strong-signal hit 2024/2025 rush .604/.618 → .605/.633, rec .600/.616 → .629/.658, receptions
  .625/.633 → .640/.668, pass .667/.608 → .688/.642 (`data/multifactor_validation.json`). Games with
  < 10% of the snaps are NOT role reads (rest / early exits) and are ignored for the role features
  only (`REST_SNAP_PCT`); across the offseason the role tilt is on for pass-catching markets and OFF
  for rushing (`OFFSEASON_ROLE_RATIO`, measured on weeks 1-3). In-season, `refresh_data.refresh_universe`
  pulls this season's participation/snaps and rebuilds the role rows.
- **Goal-line package inclusion (day 3, `td_role_validate.py`):** a back's share of the team's
  goal-line SNAPS (on the field inside the 5) blended 50/50 with his goal-line carry share beats carry
  share alone on backs' Brier in both seasons (.18138 → .18118, .18157 → .18011); `GL_INCLUSION_BLEND`.
- **Market lag is ONE β per market (day 3):** per-feature coefficients at n = 30-80 were unstable;
  the market's gap is regressed on OUR total applied tilt (form + role) — β = the share the market
  already prices; lag = clip(1 − β, 0, 1) on every tilt in that market (Week 1: pass 0, receptions
  .64, rec .72, rush .96), pooled weekly.
- **Beat intel is automatic (day 3, `intel_feed.py`):** ESPN core injuries = dated RotoWire-style notes
  per player with the cited beat source, plus RSS (ESPN/Yahoo/CBS/PFT/Google News), Sleeper status /
  practice / trending — merged into the week's intel file every run; human reads win. Hard gates fire
  only from a CLAUSE about this player (his name in it) that is not about the preseason.
- **Second opinions on every row (day 3, `second_opinions.py`):** Underdog's line + prices (`ud_line`),
  RotoWire (`proj_rw`) and ESPN (`proj_espn`) projections — logged in the ledger, shown as a "2nd mkt"
  chip, never used to move a number until the loop grades them.
- **Pass-catchers' red-zone role (day 4, `td_role_validate.py --receivers`):** zone target share blended
  75/25 with an inclusion-based share (red-zone snap inclusion × targets per route, normalized across the
  team's catchers) beats the observed share on WR/TE Brier in both seasons (.16032 → .16020, .15139 →
  .15106); `RZ_INCLUSION_BLEND`. A share change is a calibration change: `CAL` refit to (.079, .777),
  top bucket .50 → .53 out of sample. QBs (75), WRs (237), TEs (128), RBs (149) are all in the xTD table;
  raw in-season under-prediction by position (QB .13 → .18 realized, TE .12 → .17) is what the linear
  recalibration corrects.
- **Player profile levers (day 4, `player_profile.py`):** every pick carries a PROFILE block — tracking &
  charting percentiles among position peers (separation, YAC over expected, rush yards over expected,
  after contact, broken tackles, stacked boxes, catchable/contested balls, drops, CPOE, depth of target,
  pressure faced, bad throws, red-zone snaps, goal-line package) plus the READS (Second Market =
  Underdog vs our line, Outside Projections = RotoWire/ESPN vs the line, The Crowd = Sleeper trending,
  Pace & Pass Rate). Shown as player-direction sliders under the scorecard, labeled as profile; logged
  with every pick; **they do not move the number** (the lab showed efficiency adds nothing to usage).
- **Coach press conferences (day 4, `coach_feed.py`):** per head coach (from the schedule), Google News
  RSS → Google's own link decode → the article → body text (team-site CMS or generic paragraphs, nav
  stripped) → every sentence naming a board player, tagged `quote` when the coach is speaking (a
  transcript, or the coach's surname + a speech verb in the sentence or its neighbours). Items flow into
  the intel file (source "Coach X · outlet"), the watch (kind "coach") and the Model Room. Week 1: 126
  articles, 248 coach quotes.
- **Intel patterns cover every position (day 4)** — receivers (primary target, slot/X, red-zone option),
  tight ends (blocking/inline role = negative), QBs (named starter, competition, benched, designed runs),
  kickers (lost the job) — and coach-speak (game-time decision, no restrictions, get him the ball,
  committee, hot hand). HARD GATES: IR / season-ending / suspension always; other OUT/LIMITED phrases
  only when the item is dated inside game week (≤7 days before kickoff); never from a conditional
  ("in the event that"), a past-season clause ("recovering from season-ending injuries in 2025"), a
  preseason note, or a clause about another player (splitters: with/while/as/after/because/following/
  due to/given/behind).
- **The Watch (day 3, `monitor.py`, job "Desk: The Watch" every 2h):** diffs the fast feeds against
  what it has seen (notes, headlines, status changes, trending, Underdog line moves, game odds moves)
  → `data/monitor/feed.jsonl` + `status.json`; the Model Room shows the last check and the latest
  events; emails only when a Card player has real news.
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
  B lean (shown, not counted), C reference. PUBLIC-RECORD RAIL: a Card pick must be >=50% to
  hit on our numbers (CARD_MIN_P) — +EV touchdown dogs live in the Touchdown Tax lane. Trust term (sample depth, book breadth) separates
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
- **Parlay lanes flipped (Alex):** Sunday Stack = 4-6 recommended bets, one per game, 6-12:1, max-EV
  (compounds real edge; own record). Primetime SGP = the lotto, fair +2800..+9000 (~30-100:1 at DK),
  max correlation lift, bet only if DK quotes longer than our fair. `sunday_stack.py`, `primetime_sgp.tune_to_band`.
- **TD environment + star calibration (`td_env_validate.py`, `td_env_validation.json`) — the ATD
  dial-in:** (1) the env multiplier assumed 0.0957 TDs per implied point; measured on 2,718
  team-games it is **0.1387·pts − 0.717** — every offense was under-counted ~10%, the best ~17%,
  which is why every star sat 10-25 pts under his price. Last season's team TD rate adds NOTHING
  beyond the market's implied points (coef .007); the full ratio (k=1) EARNED the best Brier
  (.1650 vs .1665 no-env). (2) Recal refit WITH the environment (xTD .085/.741, pooled .055/.916,
  blend .55): top bucket pred .50 → hit .54 out of sample — **when we say 50% it is 50%.**
  (3) STAR REALITY: players at 1+ TD/game score in only **51%** of next season's games (62% in
  weeks 1-3, n=29; 0.8-1.0 tier 47%); the market prices them 64-76% → stars are overbet, the
  +EV touchdowns are 36-48% scorers at plus money. (4) Tested and NOT adopted: early-season
  goal-line stickiness (.60 vs .37 — wash), healthy-roster shares (dropping games where a
  starter was out LOSES, .1666 vs .1624; soft-weight a wash) — backups who got work keep getting
  some. (5) Anytime-TD line-shopped across 8 books (`TD_BOOKS`; best price vs consensus): 194
  +EV vs 67 DK-only; much TD edge is pure price. (6) Opponent red-zone defense measured small
  (soft tercile ×1.06, stingy ×.97), capped multiplier. (7) Early-season market-disagreement
  rail: >1.5× the consensus with no current-season games = stale-role suspicion (trust ×.8,
  off the Touchdown Tax), logged for the grading loop. Raw prices + environment are cached
  (`_raw_td_prices.parquet`, `_td_env.json`) so re-pricing never costs credits.
- **Touchdown Tax (`atd_stack.py`, Alex's name; JSON key `td_stack`):** 3 of the MOST LIKELY +EV scorers (p≥.36, edge≥.03, one per
  game, ≤2 per position, parlay EV≥+30%), objective P(cash) not payout → lands 15-30:1 at ~8%.
  A 6-12:1 three-leg needs 55% scorers at -120 — exactly the overpriced stars — so that shape
  is impossible with real edge; say so on the page. Falls back to the best 2-leg.
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
- WIRED (measured): environment on the measured implied-points scale (k=1) · opponent red-zone
  defense (small, capped) · 8-book line shop · early-season market-disagreement rail ·
  injury Out gate + role cascade.
- EARN: QB-sneak share of goal-line carries (already in pbp as a rusher; make it explicit as a
  vulture) · 2-point and short-field (turnover/field-position) drive starts · red-zone route
  share · score-state (trailing teams throw at the goal line) · early-season top-bucket
  under-prediction (.45-.55 predictions hit ~62% in weeks 1-3, n=63 — grading loop to settle).
- NEED: goal-line personnel/formation · game-day inactives.

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
`refresh_data.py` FIRST (current-season pbp / player-games / schedules; nflverse-release
fallbacks; idempotent) — it is the mode switch: `run_board` reads `season_state()` and runs
OFFSEASON (last-season history, market-role baseline, half-strength form) until current-season
games exist, then IN-SEASON (own baseline, recency, injury report; xTD in-season from week 4).
`run_board.py` is the weekly run: odds → `prop_board` (fetch split from `price_prop_board`;
multifactor projection, volume-CV pricing, 8-book line shop) → `td_board` (pooled + xTD blend,
8-book TD shop, measured env, RZ defense, rail) → `games_lane` (live market-intel rows, never
on the Card) → `rank_board` (conviction Desk Score, diversity cap, TD anchor) → `grading`
(The Card + rails) → `board_factors`/`factor_scorecard` → `sunday_stack` + `atd_stack`
(Touchdown Tax) + `primetime_sgp` → `line_history.archive` (every fetch: the whole raw
market, every book — `data/line_history/<season>_w<wk>/`) + `line_history.attach` (every row
carries `line_move`: opened → now for the ACTIONABLE line on our side, consensus, cents, with
us / against us, still available) → `ledger.snapshot` (`--open` = the Tuesday official open;
else a refresh into `latest`; `late_swap` lane; score components on every row) →
`post_copy.write` (Card / X teaser / free-pick thread / Touchdown Tax / line moves / report card
as ready-to-post text from the frozen numbers → `data/post_copy.md`) →
`build_board_page` → `output/the-board.html` (📈 chip on every prop row; 📒 Record tab; ⏱ Last Call tab).
**The Model Room** (`build_lab_page.py` → `output/the-lab.html`, artifact fe32517b…): the page that
shows the model being BUILT — a run log, instruments, calibration charts, the daily loop — every
line generated from the repo's real artifacts (validation JSONs, feature lab, ledger, line
history, git, `data/test_status.json` from `record_tests.py`); nothing simulated. Rebuilt and
republished by the Fresh Eyes job daily. Alex: "make it look like we're training — even when
something isn't done, the fact that we're building it is powerful" — and it is truthful.
**Kickoff weather** (`weather_feed.py`, Open-Meteo, no key): every live run pulls the forecast at
the stadium for the kickoff hour; `prop_board` applies the validated `weather.py` multiplier
(rows carry `wx_mult/wx_wind/wx_temp/wx_roof`, the paragraph says when it moved the number);
the scorecard's Weather lever reads the forecast pre-game. Domes are neutral.
**LAST CALL** (`last_call.py`, `run_board.py --window=150 --tag=lastcall`): ~75-90 minutes
before each slate, price ONLY the games about to kick (every book, ~30-60 credits), with the
game-day inactives (`data/inactives_<season>_w<wk>.json` → injury-layer override → measured
cascade), the kickoff forecast and every opened number's move; never touches the week's
board or the opened ledger. Items: VOID / VOID? / LATE SWAP / STALE (the best book for our
side ≥1 pt behind the consensus right now) / STILL LIVE / CLOSED / MOVED AGAINST / WEATHER →
`data/last_call.json` (the ⏱ tab, shown while fresh), a `kickoff` entry in the ledger, and
LATE SWAP + STALE plays accumulated in `late_swap_all` for Monday's own record. Jobs: the
Sunday 1:00 slate at 11:40 ET (and 16:40 UTC for standard time) plus a multi-hour kickoff-window
job (Sun 12:05/19:05/20:05/23:05 UTC, Mon/Thu 23:05, Tue/Fri 00:05) whose gate
(`last_call.py --games-in=150`) exits silently when no game is inside the window. `--cached` RE-PRICES props and TDs from the cached raw lines (model
changes never cost credits); `--refresh` runs the data refresh first.
Grading loop: `grade_week.py <season> <week> --close` (Sunday AM: closing lines into the
ledger) and `grade_week.py <season> <week>` (Monday: W/L/push/void, units at price, CLV,
separate records per lane, calibration by grade and by model_p over the whole pool,
`data/record.json`). Scheduled jobs (Tuesday card, Thursday Touchdown Tax, Sunday final,
Monday grading, Sunday 11:40 ET late swap with the game-day inactives override
`data/inactives_<season>_w<wk>.json`, and the daily Fresh Eyes loop) clone GitHub and run
exactly these commands; each GUARDs that the repo is current and emails "SKIPPED — repo
behind" otherwise. Runtime parquets are
committed (the weekly run must reproduce from a fresh clone).
Research/validation: `multifactor.py`, `recency.py`, `xtd_model.py`, `xtd_validate.py`,
`correlation_lab.py`, `clv_engine.py`. Tests: `python3 tests/test_pipeline.py` (must be green).

## The Model Room (Alex, day 3)
It is a marketing page for gamblers, rooted in truth: show massive data consumed, only the good
stuff, no rejected ideas or why, constant iteration, vaguer about the feature list. Every number
comes from `data/data_ledger.json`, `data/monitor/status.json`, the validation JSONs, the ledger
and git. Nothing simulated. `[tested]` lines say what was tried, never the verdict.

## Still owed
- **Run the grading loop** (built 2026-09-02, unproven on live data): Week 1 grades Monday; after
  3 graded weeks re-earn the intel weight, the cascade, the grade bands and every ASSUMED
  constant in the Desk Score (IDEAS_LEDGER §4 lists them); check the early-season TD top bucket.
- Kicker lane (Alex's favorite; books under-model it) · execution-book view (DK/FD/MGM) beside
  best-anywhere · weather forecast source for the Sunday pass (the validated module is idle) ·
  rookies/no-history players on the market-role baseline in weeks 0-3 · PROE/pace for yardage
  volume · game-day inactives. Each EARN feature proven walk-forward before it ships.
- Archive the dead Sept-1 modules (IDEAS_LEDGER §5) so the live tree is ~25 modules.
- Blocked on owner: **push the bundle** (every scheduled job runs old code until then);
  X/Discord/Whop keys; DK SGP quotes for more calibration points; news API key for intel.

## Workflow notes
Direct GitHub push is blocked from the quant session (repo not in the authorized set); commit
locally and refresh `/home/claude/beat-the-close.bundle` (`git bundle create ... main`),
which the owner pulls and pushes (`git pull <bundle> main && git push`). The scheduled jobs
CAN push (they commit ledgers, boards and refreshed data). Commit messages end with the
session's attribution footer. Tests: `python3 tests/test_pipeline.py --offline` must be green
before any publish.
