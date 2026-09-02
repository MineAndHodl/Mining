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
| Weather on passing props | PARTIAL | `weather.py` validated, but the weekly run never passes weather (no forecast source; schedule wind/temp are post-hoc). NEED: a forecast API for the Sunday pass |
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
