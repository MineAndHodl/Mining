# The Desk — Ideas Ledger (audit of everything Alex raised, and what became of it)

Written 2026-09-02 after a full review of the build: the memory of the whole collaboration,
all 60 commits, the scheduled jobs, and the live code. Every idea Alex raised is listed with
its honest status. Status words: **LIVE** (in the weekly run), **EARNED** (measured
walk-forward and adopted), **ASSUMED** (in the run, but the number was set by hand and is
owed to the grading loop), **PARTIAL**, **NOT BUILT**, **DEAD** (built early, no longer in
the run), **REJECTED** (tested, lost), **BLOCKED** (needs something only Alex can provide).

## 0. Found during this audit and fixed the same day
1. **The GitHub repo is ~60 commits behind the real code.** Push is blocked from the quant
   session; every scheduled job (Tuesday card, Thursday Touchdown Tax, Sunday final, Monday
   grading, Research Bot) clones GitHub — so until Alex pulls the bundle and pushes, every
   job would run the Sept-1 model. All five job prompts were rewritten today and now GUARD:
   if the repo lacks the current files they email "SKIPPED — repo behind" and stop.
2. **No in-season data path existed.** `SEASONS` was hard-coded to 2021-25 and the board was
   called with `recency_season=2025, recency_week=None` unconditionally — the model would have
   stayed in offseason mode (last-season history, market-role baseline, half-strength form)
   ALL SEASON, and none of the validated in-season edges (recency, xTD's +2-3% Brier, the
   injury report) would ever activate. Built `refresh_data.py` (pbp / player-game / schedule
   refresh with nflverse-release fallbacks), the mode switch in `run_board` (prints
   OFFSEASON vs IN-SEASON), and in-season xTD from week 4 (where it was validated).
3. **No grading loop** (work order 2, "the biggest missing foundation"). Built `ledger.py`
   (every number frozen at open + latest + close) and `grade_week.py` (W/L/push/void, units
   at price, CLV in points, separate records per lane, calibration by grade and by model
   probability over the whole graded pool, `data/record.json`). Settlement math unit-tested;
   stats path verified against nflverse 2025 data. Week 1 is the first live test.
4. **The Games tab was stale**: rows from Sept 1 carried forward on every run with no game
   labels (Arizona +440 at FanDuel was long gone). Rebuilt live every run (`games_lane.py`)
   and relabeled honestly as MARKET INTEL — we have no validated edge on sides (ratings
   ~50% ATS; the book-gap thesis was disproven) — never on the Card (`GAMES_ON_CARD=False`).
5. **Runtime data was gitignored** (`data/*.parquet`) — a fresh clone could not build the
   board. The 20 parquets the run reads (~10 MB) are now committed.
6. **Props re-pricing cost credits** — fetch is now split from pricing (`fetch_prop_board`
   / `price_prop_board`, raw lines cached), like the TD board; `--cached` re-prices free.
7. **Alex's name won**: "TD Stack" → **Touchdown Tax**.

### 0b. Found by the Fresh Eyes loop, day 2 (2026-09-03) — all fixed, every prop re-priced
8. **The trust term punished the line shop.** `n_books` counted books at OUR exact line; the
   kept row is the best line for our side, usually a one-book outlier, so every shopped Over
   sat at n_books=1 (−3) while same-line Unders sat at +3 — a 6-point Desk-Score artifact.
   Trust now reads `n_books_mkt` (books posting the market at all); the shopping context stays.
9. **Home/away was never applied.** The live board compared a team code to the last three
   letters of the home team's full name ("ers") — `home` was 0 on every row since the venue
   weight was earned. Now read from the schedule row (`is_home`).
10. **The environment was being double-counted in weeks 0-3.** Measured on the board itself
    (`market_lag.py`, mean-reversion controlled): the market's number minus our season
    baseline moves with the game total at 1.14/pt for passing (our earned weight 1.20), home
    at −4.9 (earned −4.1), R² .82 — the market line IS our baseline plus our own environment
    tilts. Herbert O233.5 (+15.6 from the total) and Goff O261.5 (+14.9) were phantom Overs.
    Under the market-role baseline total/home/def_delta are now zero, like base_c.
11. **The market prices some late-season form and not others (`data/market_lag.json`).** Same
    regression on form3: QB passing +.148 (ours applied +.057; 90% CI [+.07, +.21]) → the
    market prices it fully; receptions +.168 (ours +.150) → priced; receiving yards +.029
    (ours +.116) → 75% left on the table; rushing −.083 (CI [−.28, +.02]) → the market ignores
    late-season rushing form entirely. Applied as a per-market factor on the form tilt in
    market-baseline mode, re-measured every board and pooled across weeks. The Week-1 edge is
    rushing and receiving yards — QB yards and catch counts are reference only until games.
12. **A missed season is not "last season".** Watson U178.5 read "closed last season with 176,
    125, 168, 128" — 2024 games behind an Achilles. Per-player gap: gap 1 = half-strength form
    + market baseline (also for a returner in Week 6), gap ≥ 2 = form off, trust ×.7,
    paragraph says the year (n = 4-12 such rows in five seasons: unmeasurable, so not applied).
13. **Two books at different lines are not a consensus.** DK 4.5 / FD 3.5 made the median
    "4.0", the code took the nearest book's opinion as truth and "found" 8.7% betting the other
    (Kraft U4.5). Consensus mean = median of every book's implied mean (they agreed at ~4.6;
    the edge vanished). Rows carry `tilt` (our read vs consensus) and `off_market` (this book's
    number vs consensus), signed for our side; a pick whose edge is mostly the number is labeled
    **off-market** (3+ books; Mayfield U247.5 vs ~235 → 11.9 yds off) and told that way, or
    **no-consensus** (2 books) → REVIEW, never a PICK.
14. **Score components never reached the ledger** (written on the record, not copied to the
    ranked row) — nulls in `score_conviction/score_trust`. Fixed.
15. **REJECTED: full-game form.** Dropping partial games (opportunities < 50% of his usual;
    rest games, early exits) from the form window LOSES on every market in both held-out
    seasons (`full_game_form.py`); partial games are signal — a cut role stays cut.
16. **Post copy** (`post_copy.py` → `data/post_copy.md` every run): the Card, an X teaser
    (≤280), the free-pick thread, the Touchdown Tax, line moves, the Monday report card — all
    generated from the frozen numbers, so a post can never say what the model didn't.

### 0c. Alex's data brief, day 3 (2026-09-03) — "pull in a ton more data… constant monitoring… PFF?"
17. **The data universe** (`DATA_UNIVERSE.md`): 36 free feeds reachable from the session, ingested
    (`ingest_universe.py`), inventoried (`data/data_ledger.json`: 3.5M rows, 183M data points),
    turned into per-player-game / per-team-game signals (`universe_features.py`) and put through
    the lab (`universe_lab.py`, 98 candidates). Participation data (who was on the field for every
    play) is the crown jewel and it is free.
18. **Usage won, efficiency lost**: route participation + snap-share trend EARNED (every market,
    both seasons); goal-line package inclusion EARNED in xTD (both seasons). Next Gen Stats, PFR
    contact/pressure, FTN charting and opponent tendencies: tested, no edge beyond form + usage.
19. **Tweets / live updates**: the X API is pay-per-use ($0.005/read; ~$60/mo for a curated beat
    list) — optional. The free path is live now: ESPN's core API carries dated RotoWire-style
    notes per player that cite the beat source, plus RSS (ESPN/Yahoo/CBS/PFT/Google News) and
    Sleeper status/trending → the beat-intel layer is automatic (`intel_feed.py`) and The Watch
    (`monitor.py`, every 2h) logs what changed and republishes the Model Room.
20. **Different fantasy projections**: RotoWire (via Sleeper) and ESPN weekly projections captured
    and logged on every row (`proj_rw`, `proj_espn`); graded by the loop before they can move a
    number. FantasyPros needs a key; not needed.
21. **PFF grades**: $200/yr (PFF Pro) for API access; grades are opinions on top of charting we
    already hold free (FTN + participation). Not bought. If ever bought, it goes through the lab.
22. **A second market**: Underdog's 3,800 NFL lines pulled every 2h — `ud_line` on every row, the
    "2nd mkt" chip on the board, Underdog line moves in the watch. Graded after 4 weeks.
23. **The Model Room**: counts what the model consumes (data points, plays broken down, beat
    reports, prices, signals on trial, checks), shows The Watch, no rejected boxes, vaguer wall.

### 0d. Alex's follow-ups, day 4 (2026-09-03)
24. **Receivers and QBs in the TD model** — they always were (WR 237, TE 128, QB 75, RB 149 in the
    xTD table: zone target shares, QB rush shares, position conversion priors). Added for catchers:
    red-zone inclusion × targets per route (EARNED, both seasons); calibration refit; per-position
    calibration checked (all corrected by the recal).
25. **Efficiency on the sliders** — the PROFILE block (`player_profile.py`): tracking & charting
    percentiles among position peers + Second Market, Outside Projections, The Crowd, Pace & Pass
    Rate. Labeled profile, logged on every pick, does not move the number. Marketable and true.
26. **Beat feed for every position** — patterns broadened (WR/TE/QB/K + coach-speak); hard gates
    hardened after the broadening manufactured false OUTs (preseason, conditionals, past seasons,
    other players' clauses).
27. **Coach press conferences** — `coach_feed.py`: Google News → team-site transcripts / local papers
    / ESPN → quotes about board players → intel, the watch, the Model Room. Free. OCs next (no
    coordinator table in the schedule; needs a coach map).

### 0e. Alex's ask, day 5 (2026-09-03) — "a composite that shows how we get to our edge… the pick and the confidence interval"
28. **The bridge** (`bridge.py`, on every pick shown, Board + Model Room) — a waterfall in the
    stat's units: the baseline we build on (the market's number before Week 4; his own baseline in
    season) + each factor's FULL read (signal × the weight it earned) − what has faded across the
    offseason or is already in the line (the measured form ratio × the measured market lag, one
    striped bar, never hidden) × weather × the injury cascade = our number, with the pricing
    distribution's 25-75 and 10-90 ranges; then P(over) vs the vig-free fair → edge; then
    45 + value + hittable = conviction, + sample/breadth/intel = trust, × verdict = Desk Score →
    grade → rails → units. Touchdowns: chances × finish by zone × environment × cascade × red-zone
    defense = expected TDs → raw → calibrated → blended → vs implied → edge. The arithmetic must
    close on every pick (`run_board` prints the residual count; tests enforce it). Inputs frozen in
    the ledger (`bridge_start`, `contrib_raw`, `mu_pre`, the TD multipliers).
29. **Range coverage measured** (`interval_check.py`, 20,828 held-out player-games, 2024+2025):
    the likely range catches 47.5% (target 50%), the wide range 76% (target 80%); receptions are
    right (51/82); yardage misses lean LOW (13-18% below p10 vs 10%) — early exits and blowouts.
    **Tested, not adopted:** scaling the CV up (×1.1-1.3) fixes band coverage but does NOT improve
    out-of-sample P(over) log-loss at lines around our number (flat for rushing, worse for
    receiving/passing) — the width is right for pricing; the miss is the SHAPE of the left tail.
    Queued: a two-part distribution (an early-exit component) as a challenger; stated on the page
    as measured.
30. Queued from the bridge: the **breakeven line** on every pick (the number at which our edge is
    zero — "edge gone at 44.5"); the bridge in one line in the post copy; grading the bars — which
    factor's contribution predicts CLV in season (the attribution version of T1/T2).

## 1. Product lanes
| Idea (Alex) | Status | Where / note |
|---|---|---|
| Weekly automated picks, iterative | LIVE | `run_board.py`; scheduled jobs re-pointed (blocked on the push) |
| Tuesday card / Sunday final / Monday grading | LIVE (prompts rewritten today) | grades last week first, refreshes data, publishes, commits |
| "What I Like" 5-7 plays, Desk Confidence 6-10 | SUPERSEDED | became The Card + A+..B+ grades + Desk Score (his later ask for interpretable grades) |
| Picks only, no sizing (early) vs units by grade (later) | LIVE | units are a suggested scale (A+ 2u … B+ 0.5u); he sizes himself |
| Unders first-class; surface great straight bets even if they parlay badly | LIVE | Card is straight bets; 5 of 8 this week are Unders |
| All props listed, best surfaced | LIVE | category tabs over the full pool |
| One massive primetime lotto SGP, 30-100:1 | LIVE | `primetime_sgp.tune_to_band` → fair +2800..+9000; **bet only if DK quotes longer than fair** (DK quotes can't be fetched; calibrated on 2 slips) |
| Nuclear whale missile → flipped to a 4-6 leg 6-12:1 cross-slate parlay | LIVE | Sunday Stack (max-EV, one per game, ≤2/thesis); whale missile retired |
| 3-4 leg anytime-TD parlays ~2x/week ("Touchdown Tax") | LIVE | 3 most-likely +EV scorers, 15-30:1 (6-12:1 with real edge is impossible — the 55% scorers are the overpriced stars); Thursday refresh re-pointed |
| Futures content (SB, divisions, win totals, yardage leaders) | DEAD | built Sept 1 (`futures_expanded.json`, `ml_board.py`); not on the board, never validated. Recommend: retire until a futures lane earns a place |
| Kicker props (Alex LOVES them) | NOT BUILT | only a flat league prior, gated to REVIEW. A real kicker model is the top new-lane candidate (see §4) |
| Deep "smash spot" longshots when stars align | DEAD | `smash_spots.py` from Sept 1 is not wired; revisit only once the grading loop shows our distribution tails are calibrated |
| Referee overlay, exotic visible inputs | NOT BUILT | crew data exists (`game_crew_stats`), never earned; on the Research Bot's list |
| Tap-to-load DK slips, DK referral | BLOCKED | DK site policy-blocked for automation; affiliate is Alex's |
| "Watch the game" live commiseration | BLOCKED | needs X keys + a live feed |
| Wordle/Poeltl player-guess game unlocking a free pick | NOT STARTED | Alex: model first, games later |
| Degen/meme naming | PARTIAL | The Card, Sunday Stack, Touchdown Tax (his), Primetime Lotto; "whale missile" retired by his own flip |
| Board readable on mobile | LIVE | hosted artifact, slimmed payload |
| Last-minute edge as a category ("Last Call": inactives, stale numbers, still-live picks, kickoff weather, ~75-90 min before each slate) | LIVE (2026-09-03) | `last_call.py` + kickoff-window jobs; own record (late swap / stale) |
| Bet across DK + FanDuel + BetMGM only | GAP | the board's "best price" spans 8 books (Fanatics/BetRivers/Caesars/ESPN/Hard Rock). Small to-do: show best-at-your-books beside best-anywhere |

## 2. Model and data
| Idea (Alex) | Status | Where / note |
|---|---|---|
| Market efficiency is a hypothesis; find a real edge | EARNED | recency: +8.6% / +13.3% ROI out-of-sample; `recency.py` |
| Weight every signal together, not one factor | EARNED | ridge on [form3, form1, def_delta, total_c, home, base_c]; 5-season fit; `multifactor.py` |
| Read 5 years of play-by-play | EARNED | adopted where it won (multifactor .594→.619), dead heat elsewhere |
| Coaching/play-calling drives props; goal-line & RZ roles; play-caller consistency | EARNED (TDs) / EARN (yardage) | xTD coach tendency + coach-change flip; PROE/pace for yardage volume never wired (`usage_layer.py` staged) |
| Expected-TD from our own data (RZ share, coach RZ mix, vulture, last-season TDs) | EARNED | xTD; +2-3% Brier in-season; today: measured environment scale, refit calibration, star reality (1-TD/game backs score in 51% next season; market prices 64-76%) |
| Injury feed before data flows; role multiplier | EARNED | cascade k*=.87; Questionable = scratch risk only; 2026 report loads via fallback (empty until posted — first real test Week 1 Wed/Thu) |
| "Don't over-weight injuries — test it" | EARNED | `injury_validate.py` |
| Beat writers / pundits woven into every pick | PARTIAL | layer LIVE (date-gated JSON, hard gates OUT/LIMITED, Beat Intel lever, provisional 4-pt score weight); automation blocked (no news API) → the Tuesday job does a per-player web pass; weight to be EARNED by the grading loop |
| Intel must influence selection, not annotate survivors | LIVE | selection over the full pool; coverage rule top-40 |
| Continuity across the offseason (coach, returning players, college stats/scheme fit) | PARTIAL | coach-change voids offense (used), QB continuity null, 2-yr prior; rookies = draft capital only — their projections are junk (1.2 yds) and railed to REVIEW; college/scheme fit NOT built |
| Early numbers that move our way = the real "beat the close" product | PARTIAL → LIVE from Week 1 | Tuesday is the early number; the ledger opened/close capture + Monday CLV grading now measure it |
| Fantasy ADP as an input | LIVE | FFC ADP → Market Read (ESPN blocked) |
| ff_opportunity expected stats / lucky-vs-due flags | DEAD | `enrich.py` superseded by recency + mean-reversion; TD regression lives in xTD |
| Weather on passing props | LIVE (2026-09-03) | `weather_feed.py` (Open-Meteo, free) feeds the validated `weather.py` multiplier on every live run; the paragraph and the Weather lever show it |
| Autonomous daily research bot | LIVE (re-pointed today) | one hypothesis/day through the repo's own walk-forward harnesses; proposes a branch, never promotes |
| Weekly self-update engine + model registry (74 params) | DEAD / STALE | registry contradicts the current model (e.g. the "skew Under" it monitors was a bug we removed). Archive it; CLAUDE.md + validation JSONs are the truth |
| "Improve" alone should yield real gains | PARTIAL | §2b architecture review + the grading loop (built today). Weights get re-earned after 3+ graded weeks |
| Reprompt from scratch / restart? | DONE | THE_DESK_PROMPT.md; no rewrite |
| Sliders in player direction; Desk Score consistent across bets; edge in probability points; fewer/higher-conviction Card; on-brand name | LIVE | all shipped |
| ATD dial-in + Touchdown Tax | LIVE | today's commit |

## 3. Brand and operations
| Item | Status |
|---|---|
| X: @TheDeskNFL rename (blocked by X's review period), bio saved | BLOCKED (X) |
| Discord "The Desk" server built; Whop paywall + bot token | BLOCKED (Alex) |
| Engagement Lab weekly | scheduled; needs X API keys |
| Automated tweets / recaps / bet slips | NOT BUILT (keys) |
| GitHub push from the quant session | BLOCKED — the bundle is the transport; **pull + push is action item #1** |

## 4. What actually makes it into this week's board (the live model, honestly labeled)
- **Yardage/count props** — baseline: market-implied mean (Week 1 rule, EARNED) → own 16-game
  history once games exist; residual = ridge(form3, form1, def_delta, total_c, home, base_c)
  (EARNED, 5y) × offseason form ratio (EARNED); distribution: gamma with volume-dependent CV
  (EARNED); injury Out gate + role cascade (EARNED); weather (NOT applied); beat intel hard
  gates (LIVE) + provisional score weight (ASSUMED, 4 pts); trust term (ASSUMED); Desk Score
  hump/tail/anchor (ASSUMED); grade bands + rails (ASSUMED, sensible); micro-prop rail.
- **Anytime TD** — xTD (EARNED) on the measured environment scale (EARNED today) blended
  .55 with the pooled model (EARNED weight); RZ defense (EARNED, small); 8-book shop; early-
  season market-disagreement rail (ASSUMED, logged); TD anchor .35 (ASSUMED).
- **Games** — intel only. **Parlays** — products of the above; SGPs via the measured copula.
- **What is owed to the grading loop:** every ASSUMED item above, the intel weight, and the
  grade thresholds. Three graded weeks is the first honest read; six is a verdict.

## 5. The quant's critical review — deepen, simplify, remove
**Deepen (in this order).**
1. Run the loop: Week 1 is graded Monday; after 3 weeks re-earn the intel weight, the cascade,
   the grade bands, and check the TD top bucket (early-season under-prediction is an open finding).
2. Kicker lane (Alex's favorite, books under-model it): team FG-range drive rate × distance
   make-rate × kicker; price kicking points / made FGs. About a day; earn it before shipping.
3. Execution-book view (DK/FD/MGM) next to best-anywhere; the ledger records both.
4. Weather forecast source (Open-Meteo, free) for the Sunday pass → the validated weather
   module finally applies.
5. Rookies / no-history players: give them the market-role baseline in weeks 0-3 (zero tilt)
   so beat intel and injuries can still surface one — today they are silently REVIEW.
6. PROE / pace for yardage volume (usage_layer's idea, done the multifactor way).

**Simplify / remove (dead weight that confuses future sessions and the research bot).**
- Archive to `archive/legacy_src/`: `whale_missile`, `ml_board`, `smash_spots`, `usage_layer`,
  `enrich`, `model_update`, `research_sim`, `season_sim`, `engine`, `backtest`, `preseason`,
  `cards`, `context_bundle`, `desk_card`, `verify`, `fantasy`, the `whatilike_*.csv` files, the
  Sept-1 dashboards. The live tree should be ~25 modules, not 50. (Not done today — it is a
  mechanical move; do it in a session where Alex can eyeball the list.)
- `data/model_registry.json` and `research_log.json` v1: keep as archive, stop treating as truth.
- The pooled TD model: retire after Week 5 if the blend weight stays ≥.85 in-season.
- The games lane: keep only as CLV content for the "beat the close" brand; if it clutters, drop it.
- The Primetime SGP: honest limits — no DK quote feed, 2 calibration slips; keep it as the
  lotto content it is, never as edge.

**The risks I would name to the owner.**
1. The push. Nothing scheduled works on the new model until the bundle is on GitHub.
2. Week 1 is the weakest week by construction (offseason mode; median edge ~3%). Expect the
   Card to run 52-56% early, not 60%; the point of Weeks 1-3 is the record and the CLV read.
3. The in-season paths (refresh → mode switch → in-season xTD) were smoke-tested on 2025 data,
   never on live 2026 data; the Week 2 Tuesday email's "mode:" line is the first real check.
4. The 2026 injury report loader has never seen a real 2026 report.
5. Beat intel is a web search per player run by a scheduled agent — variable quality; hard
   gates are the only part I would trust blind.
