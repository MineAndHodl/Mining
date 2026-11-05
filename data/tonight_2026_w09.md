# Thursday, Nov 5 2026 — Jaguars @ Ravens (Week 9) — The Desk

**Market (Action Network, 6 books, 18:20 UTC):** JAX -10 (-105/-115), total 39.5, BAL team total 13.5-14.5,
JAX 24.5. Opened (Nov 2) BAL -3.5 / 49 — a 13.5-point spread move and a 9.5-point total drop.
**Why:** Lamar Jackson (thumb surgery, IR, return Dec 13) and Tyler Huntley (turf toe, IR) are out; undrafted
rookie **Joe Fagnano makes his first career start** (10/21, 91 yds, 1 INT in relief vs BUF; practicing in an
ankle brace). Jacoby Brissett was acquired but the trade is not finalized in time. Kicker Tyler Loop on IR
(Jake Moody kicks). JAX: DT DaVon Hamilton IR, Travon Walker (ribs) and Bhayshul Tuten (ankle) questionable.

## What the data says (measured today, not asserted)
- **Debut-QB lever (new, `qb_debut.py`):** emergency starters with <=1 prior start, 2021-25: the team's top-2
  pass catchers' receiving yards come in at .776x their last-3 average vs .882x in an ordinary week (n=135) ->
  x0.879; receptions do NOT move (.904 vs .896, n=140 — check-downs: catches hold, yards per catch collapse);
  the lead back's rushing yards .817x vs .993x (n=63) -> x0.823. Secondary targets and the RB2: no effect.
- **Home dog of 6+ starting a debut QB:** 34-18 ATS (65%, n=53 since 2006); mid-season starts 27-12; 2020+ 16-6.
  Home team outright 23-27% in those spots.
- **Big road favorites (9+):** home covers 58% all-era (n=191) but era-unstable (67% 2006-12, 40% 2013-19,
  62% 2020+) — a lean, never a Card pick.
- **The total:** no under edge left once the number is already <=42 (debut-QB games: 49.5% over, n=104); the
  under trend lives in high totals (40% over when >42). The market has already shrunk this one. PASS.
- 2026 through Week 8: JAX offense #3 by EPA/play (28.0 ppg), defense #3 (15.9 ppg allowed, 3.53 yds/carry,
  1.86 TD/g allowed); JAX pass rate .41 when up 10+ (.52 overall; 17 Lawrence attempts in last week's 42-7).
  BAL offense EPA negative in each of the last three games (10, 13, 10 points), Henry 17/17/11 carries.

## Props (lines = ESPN BET as quoted in ESPN's preview this morning; shop your book — no Odds API key in session)
| Pick | Line / price | Our number | Market mean | P | Edge | Grade | Units |
|---|---|---|---|---|---|---|---|
| **Derrick Henry UNDER rushing yards** | 67.5 (-114) | 65.7 (95.2 base, form -14, debut QB x.823) | 73.5 | .591 | +8.7% | A- | **1u** |
| **Zay Flowers OVER receiving yards** | 46.5 (-113) | 68.9 (79.3 base, total -3.5, routes +5.7, debut QB x.880) | 54.1 | .673 | +16.7% | A- | **0.5u** (head quant trims: his one game with Fagnano was 0-for-5; the measured discount is an average emergency starter, Fagnano may be worse) |
| Trevor Lawrence OVER passing yards | 235.5 (-108) | 245.0 | 234.8 | .553 | +5.7% | B+ | lean, unsized — the blowout script (24.5 vs 13.5) is the model's blind spot |
| Justice Hill UNDER rushing yards | 14.5 (-114) | 17.5 (mean above the line; the under is a median call) | 19.8 | .559 | +5.5% | B+ | reference only (coherence rail) |
| Trevor Lawrence UNDER rushing yards | 11.5 (-107) | 15.5 | 17.0 | .528 | +3.9% | B | reference |

Anytime TD: Henry +135 (implied 42.6%). BAL implied 14.75 pts -> 1.33 team TDs (measured env scale); Henry has 9 of
BAL's 21 TDs (43%) and 17 of 20 goal-line carries -> ~.57 xTD -> P(score) ~.43 -> fair ~+132. No edge; JAX allows
3 goal-line rush TDs on 12 carries. PASS.

## Games lane (intel, never on the Card)
- **BAL +10** — B lean. Both measured reads (debut-QB home dog 65%; big road favorite fade 62% since 2020) point the
  same way; the market has already moved 13.5 points, which is more than Lamar's usual ~6-7 points of value plus a
  backup-to-third-string step.
- **BAL ML +400 to +425** — lotto expression of the same read (home outright 23-27% in the comps vs 19-20% implied).
- Total 39.5 — pass.

## Honest caveats
- These lines are from one book this morning; they will have moved. Confirm Fagnano starts and Tuten/Walker status.
- Week 1 Card graded today: 5-3, +1.07u. Pool n=135 hit 51.1%. Weeks 2-8 were never priced by the Desk
  (the scheduled jobs never pushed), so the in-season calibration sample restarts here.
