# The Data Universe — everything we can get on the NFL, what it costs, and what it's for

Written 2026-09-03 (Fresh Eyes day 3) after a deep dive on what exists, what's tracked, where it
lives, what it costs and whether it reaches the model from our environment. Alex's brief: *"go out
and pull in a ton more data… constant monitoring… different fantasy projections… PFF grades?…
as much free shit as possible… organized, accessible, testable, aligned with the existing model."*

The rule that governs all of it: **data is pulled generously, features are adopted stingily.**
Everything below is ingested, inventoried and available to the labs; only what wins two held-out
seasons (or a replicated in-season Brier) goes into a pick.

## 1. What we hold right now (`data/data_ledger.json`, rebuilt by `src/ingest_universe.py`)

**36 feeds · 125 files · 3.47M rows · 182.7M data points · 230 MB · 1980–2026.** Pulled by
`python3 src/ingest_universe.py` (idempotent, cached, per-source freshness windows), stored in
`data/universe/` (gitignored; re-downloadable in ~3 minutes), inventoried by `--inventory`.
The Model Room's "by the numbers" reads this file.

## 2. The catalog

Status: **LIVE** in a pick today · **EARNED** measured and adopted · **LOGGED** captured on every
row for the grading loop · **TESTED** put on trial, not adopted (kept in the universe) · **READY**
ingested, no test yet · **PAID** not pulled.

### Free, structured, reachable (nflverse releases — the spine)
| Feed | What it is | Coverage · cadence | Rows | Feeds the model how | Status |
|---|---|---|---|---|---|
| Play-by-play (`pbp`) | every play, 372 columns: EPA, WP, air yards, YAC, zones, down/distance, personnel-adjacent fields | 2021-25 (5 seasons) + 2026 weekly | 247k | form, zones, TDs, script, pace, the whole player-game table | LIVE |
| Participation (`participation`) | **who was on the field for every play**: offense/defense players, formation, personnel, defenders in box, pass rushers, pressure, time to throw, coverage type (man/zone) | 2016-25, weekly in-season | 479k | **route participation** (on the field on a dropback), **snap-share trend**, **goal-line package inclusion** (inside the 5), red-zone snaps, personnel shares, opponent man/zone + box + pressure | EARNED (routes, snap trend, GL inclusion) · TESTED (opponent tendencies: no edge) |
| FTN charting (`ftn_charting`) | per-play charting: motion, play action, RPO, screen, no-huddle, blitzers, pass rushers, catchable / contested / created / drop, QB out of pocket, sneak, throwaway | 2022-25 + 2026 weekly | 185k | team play-calling rates; a receiver's catchable/contested/drop rates | TESTED (all WASH for yardage) · READY for TD/RZ work |
| Player weekly stats (`stats_player`) | 150 box-score columns per player-week incl. target share, air-yards share, WOPR, RACR, EPA, fantasy points | 2016-25 + 2026 | 182k | shares, efficiency | TESTED (target share / WOPR / aDOT: WASH beyond form + routes) |
| Team weekly stats (`stats_team`) | team totals per week | 2016-25 | 5.5k | pace, pass rate, red zone | READY |
| Next Gen Stats (`ngs_*`) | tracking-derived weekly: separation, cushion, intended-air-yards share, YAC over expected; rush yards over expected, 8+ box rate, efficiency, time to LOS; time to throw, CPOE, aggressiveness | 2016-25 | 27k | efficiency beyond yards | TESTED (all WASH: usage beats efficiency) |
| PFR advanced (`pfr_*`) | pressures, hurries, blitzes, bad throws, drops, yards before/after contact, broken tackles, coverage stats | 2018-25 | 122k | efficiency, QB pressure, coverage | TESTED (WASH) |
| Snap counts (`snap_counts`) | offense/defense/ST snaps and % per player-game | 2013-25 + 2026 | 325k | **snap-share trend** (earned); rest-game detection (<10% snaps = not a role read) | EARNED |
| Depth charts (`depth_charts`) | team depth charts, per week/day | 2021-26 | 1.2M | role rank, formation slots | LIVE (role read) |
| Weekly rosters (`rosters_weekly`) | roster status, age, experience | 2021-26 | 235k | id crosswalk, availability | LIVE |
| Injuries (`injuries`) | official reports: practice + game status | 2021-25 + 2026 when posted | 29k | own-status gate, cascade | LIVE |
| Players (`players`) | id crosswalk (gsis/espn/pfr/sleeper/yahoo…), position, draft, birth date | all | 25k | joins everything | LIVE |
| Contracts (`contracts`, OTC) | value, guarantees, years | all | 53k | role security, contract-year situational | READY |
| Draft picks / combine | draft capital; 40, vertical, broad, 3-cone, size | 1980-2026 / 2000-26 | 22k | rookie priors | READY (rookie baseline is a backlog item) |
| Officials (`officials`) | crew per game | 2015-26 | 22k | penalty/pace tendencies → totals | READY (games lane) |
| ESPN QBR (`espn_qbr_week`) | Total QBR per QB-week | 2006-25 | 11k | QB quality | READY |
| Schedules (`schedules`, nfldata) | kickoffs, spread/total/ML, roof, surface, temp, wind, referee, rest, coaches, QBs | 1999-2026 | 7.5k | environment, venue, rest, weather history | LIVE |

### Free APIs, reachable
| Feed | What it is | Cadence | Feeds the model how | Status |
|---|---|---|---|---|
| **Sleeper** players | 12k players with every platform id, depth-chart order, injury status/body part, practice notes, news timestamp | daily | crosswalk; injury/practice status between official reports | LIVE (intel feed) |
| Sleeper trending adds / drops | the crowd's adds and drops in the last 24h | hourly | role news leaks here first — a small provisional situational signal, logged | LIVE (intel, watch) |
| Sleeper weekly projections (**RotoWire**) | targets, carries, yards, TDs per player-week | 6h | a second projection set vs the market | LOGGED (`proj_rw` on every row) |
| Sleeper season projections + ADP | season-long role expectations | 2 days | rookie/role priors | READY |
| **ESPN** core API — team injuries | **dated player notes for every team** (RotoWire-style, cites the beat source: "Daniel Flick of The AJC reports") | 2h | **the beat-intel layer, automated** (hard gates + soft signals) | LIVE |
| ESPN core API — depth charts, event odds, athlete stats | depth charts by season; ESPN BET + others' game odds; per-athlete stat logs | on demand | role read; a game-odds cross-check | READY |
| ESPN fantasy projections | weekly + season projections and ownership % for 2,000 players | 6h | second projection set | LOGGED (`proj_espn`) |
| ESPN news API + RSS (ESPN, Yahoo, CBS, ProFootballTalk, Google News queries) | headlines, dated | hourly | intel feed (matched to board players), the watch | LIVE |
| **Underdog Fantasy** pick'em lines | 3,800+ NFL player lines a week: yards, receptions, attempts, anytime/first TD, 1Q/1H splits, longest play — with prices | 2h | **a second market**: `ud_line` on every row, line-move feed in the watch, the "2nd mkt" chip on the board | LOGGED (grade after 4 weeks: does it lead the books?) |
| Action Network public scoreboard | multi-book game odds (10 book ids), public bet/money split fields (populate in-week) | 2h | games lane; a free public-money read when it fills | LOGGED (watch) |
| **Coach press conferences** (`coach_feed.py`) | the 32 head coaches' pressers — team-site transcripts, local papers, ESPN — found via Google News RSS, links decoded, articles read, every sentence about a board player kept and tagged as a quote when the coach is speaking | 2h (the watch) | the beat-intel layer (coach intent, health nuance, hard gates), the watch | LIVE |
| The Odds API (key) | 8 books' props, alt lines, TD prices, game lines | per run (credits) | the whole board | LIVE |
| Open-Meteo | kickoff-hour forecast at the stadium | per run | weather multiplier | LIVE |

### Not reachable / not worth it from here
| Source | Why not | Alternative |
|---|---|---|
| ESPN `site.api` scoreboard/news/injuries | 403 from this environment | `sports.core.api` + `now.core.api` + RSS work (used) |
| Reddit JSON | 403 | RSS + Sleeper trending cover the crowd |
| Pro-Football-Reference site | 403 + ToS; scraping not needed | nflverse ships PFR's advanced stats and snap counts |
| NGS site API, NFL.com API | 401 (auth) | nflverse ships NGS weekly |
| FantasyPros API | needs a key (free tier limited) | Sleeper (RotoWire) + ESPN projections already give two sets |
| PrizePicks | 403 | Underdog's lines serve the same purpose |
| Pinnacle | 401 (guest API closed) | Pinnacle odds arrive through The Odds API |

### Paid — what they cost and my read
| Source | Price (Sept 2026) | What you'd get | Verdict |
|---|---|---|---|
| **PFF+ / PFF Pro** | $9.99/mo · $99.99/yr (Premium Stats) · **$199.99/yr PFF Pro with API/CLI data access** | player grades, routes run, target/route data, pass-blocking grades, coverage grades | Grades are opinion scores on top of charting; the CHARTING underneath is what predicts, and we already have it free (FTN charting + participation give routes, pressure, contested, drops). Our lab found efficiency/charting adds nothing beyond usage for yardage. Not needed now; if we ever want it, PFF Pro at $200/yr with the API is the only tier worth buying — and it would go through the same feature lab before touching a pick. |
| X API | pay-per-use $0.005/read (7-day search), Basic $200/mo (legacy) | beat reporters in real time | ~40 beat accounts × 10 posts/day ≈ $2/day, $60/mo. Optional: the ESPN/RotoWire notes + RSS already surface the same news within the hour, with sources. Revisit if the watch misses game-day inactives. |
| FTN Fantasy / DVOA | ~$60-100/yr | DVOA, FTN's own projections | DVOA is team-level; our team-game EPA/pressure/pace tables cover it. Skip. |
| SportsDataIO / MySportsFeeds / SIS / TruMedia | $hundreds–thousands/mo | consolidated feeds, more charting | Everything we use is free; skip until a specific missing field is identified. |
| Odds API historical | credits | past prop prices | The one paid thing that would unlock a real test (market-vs-model on history). Our ledger + line history builds the same thing forward for free from Week 1. |

## 3. How it plugs into the model (aligned with the existing structure)
```
ingest_universe.py   --> data/universe/*         raw feeds, cached, inventoried (data/data_ledger.json)
universe_features.py --> universe_features_player_game.parquet / _team_game.parquet   derived, causal-ready signals
universe_lab.py      --> data/universe_lab.json   every signal tried the repo's way (feature_lab, two held-out seasons)
td_role_validate.py  --> data/td_role_validation.json   TD role reads tried in-season (xtd_validate style)
multifactor.py       ROLE_FEATURES (route_rate_l3, offense_pct_d3) earned -> weights re-fit (8 features)
recency.py           role features per player for the live board (rest games ignored)
xtd_model.py         GL_INCLUSION_BLEND (goal-line package inclusion) in the TD role
market_lag.py        what the market already prices of ALL our tilts (form + role), weekly, pooled
intel_feed.py        beat notes / RSS / Sleeper -> player_intel_<season>_w<week>.json (automatic)
second_opinions.py   Underdog line + RotoWire + ESPN projections on every row (logged)
monitor.py           The Watch: diffs the feeds every 2h -> data/monitor/*; the Model Room shows it
refresh_data.py      in-season: refreshes this season's pbp/participation/snaps and rebuilds the role rows
```
To add a source: one entry in `ingest_universe.SOURCES` (url, seasons, freshness, what it feeds),
one derived column in `universe_features.py`, one line in `universe_lab.CANDIDATES`. Run the lab.

## 4. What the universe has proven so far (day 3)
- **98 candidate signals** (role, efficiency, charting, opponent) through the feature lab on two
  held-out seasons: the only winners were **usage** — route participation, offense snap %, and
  goal-line / red-zone / inside-10 snap inclusion. Nothing from efficiency (separation, RYOE,
  CPOE, YAC over expected, yards after contact), charting (catchable, contested, drops, play
  action, motion, RPO) or opponent tendencies (man rate, blitz, box, pressure) added to form.
- Adopted **route_rate_l3 + offense_pct_d3** (rest games <10% snaps ignored): strong-signal hit
  rate 2024/2025 — rush .604/.618 → .605/.633, rec .600/.616 → .629/.658, receptions .625/.633 →
  .640/.668, pass .667/.608 → .688/.642. Across the offseason the role tilt is on for the
  pass-catching markets and OFF for rushing (measured on weeks 1-3).
- Adopted **goal-line package inclusion** (participation) blended 50/50 with goal-line carry
  share in xTD: backs' Brier .18138 → .18118 (2024), .18157 → .18011 (2025).
- Beat intel is now automatic; hard gates are clause-level about THIS player and ignore
  preseason rest ("Allen will not play in Thursday's preseason finale" is not an OUT).

## 5. Roadmap (each item has its test path)
1. Second market vs the books: after 4 graded weeks, does Underdog's number lead the books' move? (line history + `ud_line` on every row).
2. Outside projections vs the market baseline in-season (`proj_rw`, `proj_espn` logged; grade with CLV).
3. Rookies: draft capital + preseason snaps + projections as the no-history baseline (backlog; market-role baseline covers Week 1).
4. Kicker lane: `stats_player` FG distance columns + pbp FG-range drive rate (Alex's favorite).
5. Officials → totals (games lane), when the games lane earns a place.
6. FTN charting for the TD model (contested/catchable in the red zone; sneak share as the vulture).
7. X API pay-per-use for a curated beat list if the watch ever misses a game-day inactive.
