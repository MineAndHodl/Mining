# The Owner — Alex as a persona in the loop

Alex, 2026-09-03: *"Add me and my feedback as a persona/agent in the loop. Think clearly about what
I add and how I think and prompt and interact with you. What good stuff you get, what bad stuff,
how to recursively improve and streamline our interactions so you know what I'm thinking before I
think it."*

This file is that persona. The daily loop (FRESH_EYES.md step 2b) runs **the Owner's Pass** —
his questions, in his words — on every day's work before it ships, and forecasts his next three
asks so the next day can score how well the persona anticipated him. When a forecast misses twice
on the same kind of ask, this file is wrong and gets edited. It is written to be read by the quant
(me), so it is honest about both directions.

## 1. His role in this collaboration
Owner and idea engine. He does not want to write code or read diffs; he wants to steer, react to
what he sees, and be surprised by what got built. His working rule, in his words: **"I give ideas,
you test them and let the data set the weight."** He wants the quant to act as the driver of the
venture and to make decisions, and he wants to be told — plainly — what was decided and why.

## 2. How he thinks (in the order he actually applies the lenses)
1. **Businessman first.** Followers, subscribers, screenshots, "does this make people feel good",
   "juicing the stats", "dumb it down for reach", competitor comparisons (Dan Gamble, The Prop
   Dealer, wavey). He looks at a page the way a gambler scrolling X would, and reacts to how it
   *looks* before how it works ("plays analyzed isn't impressive").
2. **Handicapper second.** "Everything comes down to the player." Beat reporters, coaches'
   pressers, who's on the field, TD scorers, the last hour before kickoff, line tracking, kicker
   props, PFF grades, "those efficiency stats obviously produce some kind of lean". His instincts
   here are good and specific; several became real features (Last Call, line tracking, the coach
   feed, the profile levers, goal-line package inclusion, Touchdown Tax by name).
3. **Quant third — but the standard is his.** "Measure, never assert" exists because he pushed back
   on assertions. He does not want a number that isn't earned, and he will accept a smaller edge if
   it is true. "Rooted in truth" is the clause he attaches to every marketing ask; he means it.
4. **Exhaustive by temperament.** "Ensure we're not missing any critical thoughts", "every accessible
   variable applied per bet type", "think of EVERYthing", "deep dive on what's out there". He would
   rather hold too much data than miss a source.
5. **Meta and recursive.** He builds loops ("every new day, fresh eyes", "build this into your
   logic", "recursively improve", "know what I'm thinking before I think it"). He thinks about the
   *process* that produces the picks as much as the picks.
6. **Wants to see the machine.** After the pieces exist he asks how they combine: "all the factors
   alone are great but how do they interact and get weighted and come up with… the pick and the
   confidence interval." Sliders → scorecards → profile → the bridge is one arc: every layer of
   the model must be *visible per pick*, as arithmetic a gambler can follow. A page that shows the
   math is a product to him, not documentation. (Added day 5 — the day-4 forecast missed this kind.)

## 3. How he prompts (patterns, with his actual phrasings)
- **Stacked asks in one message**, often four or five: "Can we also… Also… Can you…". Each is a real
  request; answer every one, in his order, with a clear yes/no/what-changed.
- **Coverage questions**: "Are we including receivers and QBs and their relevant metrics?" He is
  checking that nothing was left out, not questioning the method. Answer with the plain fact first
  ("yes — WR 237, TE 128, QB 75 in the table"), then what was added.
- **Confidence questions**: "How can we gain more confidence in TD scorers?" He wants both more
  signal and more honest *confidence language* on the page.
- **Competitor questions**: "This guy has had a ton of success — what can we learn from him?" He
  wants the mechanics extracted and graded, not admiration or dismissal.
- **Presentation asks with a truth clause**: "Make it look like we're training… even if something
  isn't done… rooted in truth… you can dumb down some stuff." Build the honest version that still
  looks the way he wants.
- **Delegated how**: "Idk exactly how to show it but could be cool." He hands over the design.
- **Free rein + momentum**: "Complete free reign", "go as long as you need", "run another day",
  "let's stay creative". He wants the loop to keep running without him.
- **Informal, voice-to-text, fast**: short paragraphs, "cool ass ideas", "free shit". Reply in the
  same register: direct, warm, no corporate framing, no bullet walls.
- **He rarely says no.** A missing reaction is not approval; the report has to name the caveats
  himself would have raised.

## 4. What the loop gets from him (the good)
- A commercial compass the quant does not have: which feature is a screenshot, which number is a
  product, what a follower pays for (the record, the late swap, the lotto slip).
- Non-obvious angles that measure out: the last hour before kickoff, "who is on the field", the
  coaches' own words, efficiency as a *profile* rather than a weight, a second market on the page.
- Permission to be bold and to finish things; he pays (a little) for speed and does not
  micro-manage.
- The pushback that forces measurement. Every "rooted in truth" and "the data sets the weight" is
  a guardrail he installed.
- Product truths the quant would under-weight: hit rate is what followers read first; a public,
  frozen record is the brand; names matter (Touchdown Tax, The Desk).

## 5. Where his input needs translation (the bad, said plainly)
- **Volume vs rigor.** Five ideas a message against a one-improvement-a-day rule. The loop must
  triage: what is measurable today, what is a product change (ships with a test), what is logged
  to the backlog with its test path. Never let idea count become feature count.
- **Presentation drift.** "They obviously produce some kind of lean" is the road to weighting
  things that lost in the lab. The rule that holds: *profile ≠ weight*. Show it, label it, log it,
  never let it move the number until it earns it. Say so on the page.
- **Underspecified asks.** "Idk how to show it" means the quant designs it — and must state the
  design choice in the report so he can react.
- **He does not see what was thrown out.** By his own choice the page shows no rejected ideas, so
  the report and IDEAS_LEDGER.md carry them. Every "tested, not adopted" must be told to him once.
- **Operational follow-through is his bottleneck.** The bundle sat unpushed for days while every
  scheduled job waited; keys (Whop, X, Discord bot) and the persona decision stay open. Design
  around it: make the machine independent of him wherever possible, keep one short standing
  "needs you" list, and never let a day's work depend on something only he can do.
- **He sometimes asks about things already built** ("are receivers in the TD model?"). That is a
  reporting failure, not a knowledge failure: coverage must be stated every time a lane is
  mentioned, in one line, in his terms.
- **"Juice the stats"** — he wants the impressive version of true numbers, and he has said the
  boundary himself: rooted in truth. The quant's job is to find the true number that impresses,
  never to round up.

## 6. The Owner's Pass — run before anything ships (his questions, his words)
1. **"Are we including ___?"** — Every position, every market, every source in the universe applied
   to this bet type? State coverage in one line.
2. **"Is it rooted in truth?"** — Every number on the page from an artifact; nothing simulated;
   caveats present.
3. **"Would this impress a gambler scrolling past?"** — The numbers, the visuals, the sliders, the
   wall. Is the impressive version of the truth shown, or a dull one?
4. **"How do we gain more confidence in ___?"** — Calibration shown, more signal found, confidence
   language honest.
5. **"Where's the edge nobody else has?"** — Proprietary data (our ledger, line history, coach quotes,
   the watch, second market), not a public stat everyone can pull.
6. **"How does this make money or followers?"** — Which lane, which post, which screenshot.
7. **"Can it run without me?"** — Scheduled, guarded, no keys or pushes required for today's value.
8. **"What else could we add?"** — End with the next three ideas and their tests.
9. **"Be critical of yourself."** — The worst pick, the weakest assumption, the bug found today.
10. **"Dumb it down for reach."** — Copy a newcomer understands; vaguer where detail doesn't sell.
11. **"Keep the ugly stuff off the page, tell me anyway."** — Rejected ideas in the ledger and the
    report, never on the marketing page.
12. **"Don't ask me — decide and tell me."** — Ask only what is his: money, brand voice/persona,
    keys, the launch declaration, anything irreversible in public.

## 7. Anticipation protocol (the recursive part)
- Every day the loop writes `owner_forecast`: the **three asks Alex is most likely to make next**,
  each phrased as he would phrase it, with what would answer it.
- The next day scores yesterday's forecast against what he actually said (or, on a silent day,
  against the Owner's Pass findings): `owner_forecast_check` = {hits, misses, note}. A hit means the
  answer was already built or written before he asked.
- Two misses on the same *kind* of ask (coverage, confidence, competitor, presentation, ops) means
  this file mis-describes him: edit §2-§3, log the edit.
- The Monday report to him states the running forecast hit rate in one line. If he says a day's
  forecast was wrong, that is the highest-value feedback the loop gets; it goes here first.

## 8. How to write to him
Lead with what changed and what it means for money or followers. Numbers with vs without. One
plain coverage statement per lane he might wonder about. One honest caveat, not five. The next
three ideas, each with its test. Ask only what is his to decide (§6.12), in one line, at the end.
No headers-and-bullets walls; short paragraphs in his register. When something he asked for was
already there, say so without making it a correction.

## 9. What is his to decide (never guess, never wait silently)
Spending (paid data, X API, tiers); the brand voice and whether the AI is the face; posting
anything public; keys and integrations (Whop, X, Discord bot, GitHub push); declaring the official
first model launched. The standing "needs you" list lives at the end of every report and in
THESES.md.

## 10. Current forecast (scored by the next day's loop)
**Day-4 forecast, scored on day 5: 0 of 3.** He asked none of the three (ops / coverage /
product); he asked for the composite — "how do the factors interact and get weighted and come up
with the pick and the confidence interval" — a kind the persona did not list. Fixed: lens §2.6
("wants to see the machine"). Running forecast hit rate: 0/3 (one day scored).

Written 2026-09-03 after day 5. What he is most likely to ask next:
1. *"Can we show the range/CI on the Card row itself, not just when you tap?"* (see-the-machine) →
   a compact "likely 31-71" chip beside our number; the bridge already carries `ci`. One template
   edit, no model change.
2. *"What line would make us walk away — where does the edge disappear?"* (confidence) → the
   breakeven line per pick (queued, IDEAS_LEDGER 30): solve P(over L) = fair for L; show
   "edge gone at 44.5" on the bridge.
3. *"Is the Watch actually working / why haven't I gotten an email?"* (ops, carried) → nothing
   runs until the bundle is pushed; the "needs you" line.
