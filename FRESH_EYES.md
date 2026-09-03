# Fresh Eyes — the daily loop until the official first model launches

Alex, 2026-09-02: *"Every new day, until we launch the official first model, look at this with
complete fresh eyes. Think of new ideas, weave them in, build off the knowledge base, make it a
statistical masterpiece, find non-obvious alpha, be critical of the work so far, then improvise
and get slightly better every single day. Think like a maniacal handicapper (edge), a quant
(how it all fits), and a businessman (marketable, fun, makes money)."*

This file IS that logic. The daily job (`Desk: Fresh Eyes`, 07:00 ET) follows it verbatim; a
human session that says "improve" follows it too. Log every day in `data/fresh_eyes_log.json`.

## The loop (one day = one pass, ~2-3 hours of work, ONE shipped improvement)

**0. Read, don't remember.** CLAUDE.md, THE_DESK_PROMPT.md §2b, IDEAS_LEDGER.md, the last three
entries of `data/fresh_eyes_log.json`, the latest `data/ledger/*.json` (and its `results` if
graded), `data/feature_lab.json`, `data/record.json`. Then open the live board
(`data/ranked_board.json`) and read the Card like a stranger who just paid for it.

**1. Three hats, written down before touching code (15 minutes each).**
- *Maniacal handicapper* — where is the edge NOBODY is pricing? Look at what the market
  prices lazily (shape of distributions, stale roles, late news, cross-market consistency,
  book-specific shading, timing of postings). Name 3 non-obvious angles with a way to test each.
- *Quant* — does everything fit? Run THE_DESK_PROMPT §2b (survivors vs selection, upstream vs
  downstream, truncated pools, earned vs assumed, card/score agreement, cost vs coverage,
  blind-by-construction, the sharp's squint). Name the single weakest assumption in the live
  model today and how to measure it.
- *Businessman* — is it marketable and fun, and does it make money? What would a follower
  screenshot? What would make them pay? What looks like noise to a newcomer? Name one product
  change (copy, lane, cadence, rail) that improves W-L optics, shareability, or conversion
  without lying.

**2. Critique the work product, adversarially.** Pick the worst pick on today's Card and say
why it is there. Pick one number the model shows and re-derive it by hand. Find one place
where the page and the model disagree. Write the three findings down even if none is a bug.

**2b. The Owner's Pass (OWNER.md §6).** Before choosing, run Alex's twelve questions on today's
board and today's candidate: coverage ("are we including ___?"), truth, would it impress a gambler,
confidence, the edge nobody else has, money/followers, runs without him, what else, self-critique,
dumbed down, ugly stuff off the page, decide-don't-ask. Write the two that bite hardest. Then score
yesterday's `owner_forecast` (hits/misses) and write today's — the three asks he is most likely to
make next, phrased as he would, with what answers each. Two misses on the same kind of ask → edit
OWNER.md.

**3. Choose ONE thing and finish it.** The best candidate from steps 1-2 that can be measured
today. Test it the repo's way:
- yardage/count feature → `src/feature_lab.py` (two held-out seasons; adopt only if it wins both);
- distribution/shape → `src/cv_measure.py` and a book comparison (`src/alt_ladder_probe.py`);
- touchdowns → `src/xtd_validate.py` / `src/td_env_validate.py`;
- injuries/roles → `src/injury_validate.py`;
- anything about grades, trust, intel weight → the ledger's graded pool (`grade_week.py`),
  and if fewer than 3 graded weeks exist, log it as "owed to the loop" rather than guessing.
A wash or a loss is a result: log it so it is never re-tested. A product change ships with a
test in `tests/test_pipeline.py`.

**4. Ship or shelve, honestly.** Tests green → commit with the day's log entry → bundle (or
branch `fresh/<date>` from the scheduled job, never main) → email/report: what was tried,
the numbers with vs without, the verdict, what changed on the board, tomorrow's candidate.
Never change a weight because it "makes sense"; never present an untested idea as a finding.

**5. Leave the trail.** Append to `data/fresh_eyes_log.json`:
`{date, hats: {handicapper, quant, business}, critiques, owner_pass, owner_forecast_check, owner_forecast, tested, numbers, verdict, shipped, tomorrow}`.
Update IDEAS_LEDGER.md when an idea changes status. Update CLAUDE.md only for settled findings.

## The owner in the loop
OWNER.md is Alex as a persona: how he thinks (businessman → handicapper → quant, exhaustive,
recursive), how he prompts (stacked asks, coverage and confidence questions, competitor questions,
presentation asks with a truth clause, delegated how, free rein), what the loop gets from him, where
his input needs translation (volume vs rigor, profile ≠ weight, underspecified asks, he never sees
the rejected list, ops follow-through is his bottleneck), and how to write to him. THESES.md is what
we believe, what freezes at the Week 1 open, and the in-season calendar of re-earning.

## Standing rules for the loop
- Measure, never assert. The data sets the weight; a feature ships only if it wins out of sample.
- The public record is a product: hit rate matters as much as units (CARD_MIN_P = .50).
- The Card is frozen at open; the loop improves NEXT week's numbers, never this week's ledger.
- Credits are a cost: re-price from cached lines; a live fetch only when the day needs prices.
- One improvement a day beats six ideas. Small, replicated, shipped.

## Launch criteria for "the official first model" (proposed — Alex decides)
1. The bundle is on GitHub and one full scheduled cycle (Tue → Thu → Sun → Sun late swap → Mon)
   has completed without a "SKIPPED" email.
2. Week 1 is graded in the ledger with CLV captured (the loop has run once end to end).
3. Tests green; the live tree archived down to the modules the run uses.
4. The record page shows real numbers. From then on, "official" = every number is frozen and
   graded in public, and the loop keeps running.

## Backlog of non-obvious angles (each with its test) — pick from here on quiet days
- **Cross-market consistency**: market receptions × player's yards-per-catch vs the market's
  receiving-yards line; when they disagree, one line is stale. Test prospectively via the
  ledger (log implied YPR at open).
- **Book dispersion as signal**: when 8 books disagree by ≥2 pts, is the outlier soft or sharp?
  Every row now carries `cons_mean`, `off_market`, `tilt`, `thesis_kind`, `n_books_mkt`
  (day 2); grade the off-market lane vs the model lane after 4 weeks. One-snapshot read:
  BetRivers is off the consensus 61% of the time it posts, FanDuel 12%; DK's outliers skew high.
- **Star Overs shaded**: the TD market overprices stars 10-25 pts; do yardage Overs on
  top-ADP players carry the same public shade? Log ADP rank + side + result.
- **Early-exit mixture**: P(under) on high lines gets help from injury exits / blowouts; a
  zero-inflated component for the left tail. Measure the exit rate by position; test on
  weeks 1-18 with `feature_lab`-style held-out seasons.
- **Rookie baseline**: no-history players onto the market-role baseline in weeks 0-3 so intel
  and injuries can surface one (today they are silently REVIEW). Day 3 adds the inputs: draft
  capital (`draft_picks`), combine, RotoWire/ESPN projections, preseason notes.
- **Market lag on the ROLE tilt in-season**: the weekly β now covers form + role together; once
  games exist, check whether the market prices route participation as fast as it prices yards.
- **Second market vs the books**: `ud_line` on every row from day 3 — after 4 weeks, does
  Underdog's number lead the sportsbooks' move (a free sharp read) or follow it?
- **Outside projections vs the market**: `proj_rw` / `proj_espn` logged; grade with CLV.
- **FTN charting in the red zone**: contested/catchable rates on RZ targets, QB-sneak share as the
  explicit vulture (the universe has it; the yardage lab found nothing, the TD model may).
- **Kicker lane**: FG-range drive rate × distance make-rate × kicker; earn before shipping.
- ~~Weather feed~~ DONE 2026-09-03 (`weather_feed.py`).
- ~~Rest-game contamination of the form window~~ TESTED day 2: LOSES both seasons (partial
  games are signal). Never re-test.
- **Market lag, pooled**: `data/market_lag.json` re-measures every board which tilts the market
  prices; after 3+ weeks compare the pooled factors to CLV by market (does the rushing edge
  actually beat the close?). If the in-season cross-section says the market lags in-season
  form too, the same factor belongs in in-season mode (today it is weeks 0-3 only).
- **Receptions in Week 1**: catch-count form is priced by the market (day 2) — so the Week-1
  receptions edge, if any, is pure shape (our .20 extra dispersion vs the book's). Probe DK's
  receptions ladders the way `alt_ladder_probe.py` did for yards before trusting a shape edge.
- **Posting timing**: when do DK/FD post next week's props? If Tuesday 14:30 UTC misses half
  the menu, add a Wednesday capture. Read it off the `latest` ledger snapshots.
- **Execution-book view**: DK/FD/MGM price beside best-anywhere on every row.
- **Alt-line edge, properly**: DK's ladder shape matched reality better than ours did (day 1).
  Re-probe after the CV fix; the edge, if any, is in the vig curve per rung, not the shape.
- **TD 2+ / first TD**: over-dispersion by goal-line hogs vs Poisson; needs those markets in
  the feed.
- **Content mechanics that don't lie**: the Monday "beat the close" report card, the Thursday
  Touchdown Tax post, the late-swap ping as the paid-tier hook, the free pick = the Sunday
  "window closed" leftovers.

## Day log summary (details in data/fresh_eyes_log.json)
- **2026-09-02 (day 1)** — Handicapper: probed DK alternate ladders for a shape edge; none —
  the book's implied spread matched reality and OURS was 10-40% too wide at the high-volume
  end (pass yds .40 vs .28 real). Quant: re-measured within-player-season CV by tier on 5
  seasons (`cv_measure.py`, `data/cv_anchors.json`), fixed receptions' double-counted
  variance; every prop re-priced (Card went 5U/3O → 4/4; Goff/Henry Overs surfaced, Taylor
  Under fell). Feature lab: game-script spread, short week, pace — all WASH, logged.
  Businessman: public-record rail (Card ≥50% to hit; +EV TD dogs live in the Touchdown Tax),
  a Record tab on the board, and a Sunday 11:40 ET late-swap job with a game-day inactives
  override that prices the backup off the measured cascade before the books do.
- **2026-09-03 (day 2)** — Quant: the ridge weights were earned against OUR baseline but applied
  on top of the MARKET's number in weeks 0-3. Measured on the board (`market_lag.py`): the
  market's line is our baseline plus our own total/home tilts (1.14/pt vs 1.20; R² .82) → the
  environment is no longer re-added under the market baseline (Herbert/Goff Overs were phantom).
  The same regression on form: the market prices QB passing form (+.148 vs our +.057) and
  catch counts, ignores rushing form (−.08) and leaves 75% of receiving-yards form → applied as
  a per-market factor, pooled weekly. Bugs: trust penalized the shop (books at our line, always
  1); home/away never applied (string bug); a missed season read as "last season" (Watson);
  two books at different lines "disagreeing" into an 8.7% edge (Kraft) → consensus = median
  of book-implied means, off-market plays labeled (Mayfield U247.5, 11.9 off), two-book
  disagreements REVIEW. Handicapper: rest/partial-game contamination tested — LOSES, partial
  games are signal. Businessman: `post_copy.py` — Card, X teaser, free-pick thread, Touchdown
  Tax, line moves, report card, generated from the frozen numbers. Card 5O/3U → 3O/5U, rushing
  form is the Week-1 edge. Tests 136 → 155.
- **2026-09-03 (day 3, Alex's data brief)** — Built the data universe (`DATA_UNIVERSE.md`): 36 free
  feeds ingested and inventoried (183M data points), derived player-game / team-game tables, and a
  98-candidate lab. Usage won, efficiency lost: route participation + snap-share trend adopted
  (every market up, both seasons; rest games ignored for role; rushing role off across the
  offseason); goal-line package inclusion adopted in xTD (both seasons). Market lag simplified to
  one β per market. Beat intel automated from ESPN/RotoWire notes + RSS + Sleeper with clause-level
  hard gates. Underdog's lines + RotoWire/ESPN projections logged on every row. The Watch (feeds
  diffed every 2h, its own job). Model Room revamped: what the model consumes, the watch, no
  rejected items. Tests 155 → 173.
- **2026-09-03 (day 4, Alex's follow-ups)** — Receivers/QBs were already in the TD model; what was
  missing was a less noisy catcher role: red-zone inclusion × targets-per-route, blend .75, wins both
  seasons → adopted, calibration refit. The tracking & charting signals that lost in the lab now live on
  every pick as PROFILE levers (percentiles among peers) with the reads (second market, outside
  projections, the crowd, pace) — honest label, logged, unweighted. Coach press conferences: 32 coaches
  → Google News → transcripts/quotes → intel + the watch (126 articles, 248 quotes in week 1). Intel
  patterns for every position; hard gates hardened (game-week dating, IR-always, no conditionals/past/
  other-player clauses). Tests 173 → 192.

## The bridge must close
Every pick shown carries `bridge` (src/bridge.py): baseline + factors − faded/priced × weather ×
cascade = our number; P − fair = edge; (conviction + trust) × verdict = Desk Score. `run_board`
prints `bridge: N picks, K with a residual` — K must be 0 before anything ships; a model change
that breaks the arithmetic (a new multiplicative step, a new feature) is not done until the bridge
shows it. `tests/test_pipeline.py::test_bridge` enforces it.
