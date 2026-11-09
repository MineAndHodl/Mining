# UPGRADE PLAN — from "raw form" to "opportunity-aware form" (Nov 8 2026)

**Owner's brief (Alex):** "We shouldn't just immediately cut their receivers because it was a monsoon.
You should be able to retroactively adjust." He is right. This is the plan to fix the class of error,
not the instance.

## 1. The diagnosis, stated plainly
The multifactor model's edge is recency: a player's last-3 and last-1 games vs his season baseline. Those
inputs are RAW TOTALS (yards, catches). A raw total is `team volume × player share × efficiency`. When the
TEAM's volume is forced by the environment — a monsoon (TB: 3 pass attempts, 56 carries), a 42-7 blowout
(JAX: 17 attempts), a mid-game QB loss — the raw total collapses or balloons for every player on that team,
and the model reads it as a change in HIM. It is not. It is a change in the weather.

Measured today (`data/form_anomaly_effect.json`, 2021-25, slope of next-game residual on last-game deviation):

| market | normal team volume | team passed/ran ≤55% of its usual | ≥150% of usual |
|---|---|---|---|
| receiving yards | **+.165** (n=5,221) | **−.029** (n=63) — NO information | +.145 |
| receptions | +.205 | +.142 | +.127 |
| passing yards | +.190 | n=10 (unmeasurable) | +.308 |
| rushing yards | +.257 | +.383 (n=66) | +.211 (n=163) |

A receiving-yards game played when the team threw ≤55% of its usual attempts predicts NOTHING about next
week. Even target SHARE dies in those games (+8.7 yds per 10% share vs +53 normally): with 3-11 attempts the
share is noise too. The rushing mirror (a 56-carry game) still carries information (+.21 vs +.26) — the
team that ran 56 times revealed something real about its identity and its backfield split.

Why the earlier test did not catch this: `full_game_form.py` dropped games where the PLAYER's opportunities
were <50% of his usual and LOST in both seasons — because a player's own short game (rest, early exit,
benching) IS a role signal. This rule keys off the TEAM's volume, which is unambiguously environment. Same
symptom, different cause, opposite answer. The lesson for the model: condition on the cause, not the symptom.

## 2. Phase 0 — ship this week (earned above): team-anomaly weighting in the form window
- Per team-game, compute pass attempts and rush attempts from play-by-play; the team's season median.
- For receiving-yards rows: a game where the team threw ≤55% of its median gets WEIGHT 0 in last-3, last-1
  and the season baseline (the window rolls back to the previous game).
- Receptions: weight .7 (measured .142/.205). Rushing: no change (still predictive at every volume).
- Passing yards: tried by analogy and **REJECTED by the walk-forward A/B** (strong-signal hit .688→.671 in 2024,
  .642→.634 in 2025, MAE up) — a QB's own low-attempt game is usually HIS exit (benched/hurt), a role signal.
  The raw window stays for QBs. Mayfield's number tonight therefore stays unreliable and is not used; the
  measured bounce-back read (0.82× prior form, n=35) is the honest one for him. Shipped Nov 8 (`form_weights.py`):
  rec_yds .629→.632 / .658→.658, receptions .640→.640 / .668→.669 — not worse in any season, as expected for
  a 1%-of-games rule whose value is concentrated in the weeks after an anomaly.
- The row carries `form_excluded: [game, reason]`; the bridge gets no new bar (the baseline itself moves);
  the paragraph says "his Week 8 (3 team pass attempts in the rain) is not counted as form".
- Acceptance: `multifactor.fit_walk_forward` on 2024 AND 2025 with the weighting on vs off — strong-signal
  hit rate and residual MAE must not get worse (1% of team-games are anomalies, so expect small gains in the
  weeks after one). Test added to `tests/test_pipeline.py`. Re-price tonight's board as the proof.

## 3. Phase 1 — two weeks: the opportunity-decomposed projection (T5 challenger)
The structural fix. Project the three factors separately and multiply:
1. **Team volume** (pass attempts, rush attempts) from tonight's environment — implied team total, spread,
   the team's pass rate over expectation, pace, opponent, weather — weights earned on 2021-25 team-games.
2. **Player share** of those opportunities (targets / team attempts; carries / team carries) — form on the
   SHARE, precision-weighted by the team volume it was measured in (the share slope rises 24 → 57 → 71 → 89
   yds per 10% as team attempts rise: a share seen on 48 attempts is worth more than one seen on 26).
3. **Efficiency per opportunity** (yards per target, yards per carry) — shrunk hard toward the player's
   multi-season rate (the universe lab showed efficiency form adds little beyond usage).
Walk-forward vs the current raw-form model on 2024 and 2025: strong-signal hit, MAE, log-loss of P(over).
Adopt only if it wins; otherwise keep Phase 0 and the raw model. This is what makes "retroactive
adjustment" automatic — a game's team volume is a known fact after the fact and the model conditions on it,
instead of us hand-editing a monsoon out of a window.

## 4. Phase 2 — the general QB-quality lever (the Flowers lesson)
The debut-QB lever shipped Thursday treats every emergency starter as the average one; Fagnano was worse
(6 yards to Flowers). Replace with a QB DELTA: the new starter's career EPA per dropback and yards per
attempt (prior = draft capital for no-history QBs) vs the departed starter's — applied to the team's pass
volume and the pass-catchers' efficiency, and to the lead back's yards. Earn on every 2021-25 QB change
(n ≈ 150+ team-games); the debut case becomes the low-prior end of the same scale. Covers Brissett next week.

## 5. Phase 3 — the calibration loop, every week, including hand-fed boards
- Every pick we send goes into the ledger — including this week's Thursday and SNF picks, which were priced
  without the Odds API and so bypassed `ledger.snapshot`. Add a manual-board path so they count on the record.
- Monday: `grade_week` → W/L, units, CLV, calibration by p-bucket. Flag extreme p's (Williams U rush .71) and
  check them as their own bucket.
- After 3 graded weeks: re-earn the Desk Score constants, the grade bands, the intel weight, the cascade.

## 6. Phase 4 — data and plumbing (what is blocked on the owner is marked)
- **Odds API key** (`ODDS_API_KEY` as an environment variable in the cloud environment settings) — restores
  the 8-book shop, the TD lane, line history and Last Call. **[owner]**
- **DK as the execution book**: price every pick at DraftKings' line AND best-anywhere; the Card shows both.
- **Merge the branch** so the scheduled jobs stop running September code. **[owner]**
- 2026 route participation is not published yet, so the earned role feature runs on LAST SEASON's routes;
  switch the in-season role read to 2026 snap counts (published) with a measured fallback.
- Line history from Week 9 forward (the opener → close record restarts).

## 7. Phase 5 — the games lane earns a model, not a trend list
This week's measured reads (debut-QB home dogs 65%; home favorites of 5.5-7.5 at 44% since 2018; SNF unders
40% over since 2018; big road favorites era-unstable) become FEATURES in a walk-forward games model (team
EPA, QB status, rest, line move, situational flags). A trend is a hypothesis; the model earns its weight.

## 8. Rules that govern every item
Measure before asserting; walk-forward on two held-out seasons; adopt only if it wins; log the result in
`data/*.json`; a test in `tests/test_pipeline.py`; the bridge must close; the paragraph names the
adjustment; the ledger freezes the inputs. Any constant that is assumed is labeled ASSUMED until earned.

## 9. Tonight's corrected reads (share-based, Tampa receivers)
At the team's usual ~32 attempts: Godwin 34 (season share) to 56 (last-3 normal-game share .25, rising),
Egbuka 36-43, Otton 19-26, Mayfield ~215-220 — i.e. the market's numbers, not the model's 148/33/36. No play
either way on the Tampa passing game; the Card (Williams U rush, Odunze U rec, Swift U rush, Loveland O rec)
is unaffected because none of those windows contain an anomaly game.
