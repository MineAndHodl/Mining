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
`{date, hats: {handicapper, quant, business}, critiques, tested, numbers, verdict, shipped, tomorrow}`.
Update IDEAS_LEDGER.md when an idea changes status. Update CLAUDE.md only for settled findings.

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
  Log `n_books`, dispersion, outlier book on every pick; grade after 4 weeks.
- **Star Overs shaded**: the TD market overprices stars 10-25 pts; do yardage Overs on
  top-ADP players carry the same public shade? Log ADP rank + side + result.
- **Early-exit mixture**: P(under) on high lines gets help from injury exits / blowouts; a
  zero-inflated component for the left tail. Measure the exit rate by position; test on
  weeks 1-18 with `feature_lab`-style held-out seasons.
- **Rookie baseline**: no-history players onto the market-role baseline in weeks 0-3 so intel
  and injuries can surface one (today they are silently REVIEW).
- **Kicker lane**: FG-range drive rate × distance make-rate × kicker; earn before shipping.
- ~~Weather feed~~ DONE 2026-09-03 (`weather_feed.py`).
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
