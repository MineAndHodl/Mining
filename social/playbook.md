# Engagement Lab — playbook v1 (priors, to be beaten by data)

## Vet priors (starting weights, all subject to experiment)
- HASHTAGS: near-zero value on X since 2023 algo; max 1 niche tag, only during game
  windows (#NFL on gameday). Never 3+. Hypothesis: 0 tags >= 1 tag > 2+ tags.
- REPLIES are the growth engine at <1k followers: 5-10 substantive replies/day to
  big in-niche accounts (beat writers, betting media, wavey-tier cappers) within
  10 min of their post. Add a number/graphic no one else has. NEVER generic
  ("great post!"), never spam cadence.
- MEDIA: card graphics > plain text. Every pick post ships with a branded card.
- HOOK: first 8 words decide the impression->read conversion. Test: number-led
  ("57.6% of wind games...") vs contrarian ("Home dogs are dead.") vs machine-voice
  ("I simulated the season 20,000 times.") vs question.
- TIMING slots (ET): Tue 10a (card), Wed 12p, Thu 12p (Tax), Sat 11a, Sun 9a,
  Sun 12:55p (pre-kick), Sun/Mon/Thu primetime pregame, Mon 10a (receipts).
- QTs with a real take > RTs. Data QTs of big accounts' takes = discovery.
- Threads: 1-2/week max, always with a payoff tweet.
- GUARDRAILS: no follow-churn, no pods, no bought anything, giveaways rare and real,
  no off-topic trend hijacking, automation label ON. Nothing X's spam policy hates.

## Experiment protocol
- One variable per test where possible; log EVERY post to social/experiments.csv.
- Metrics at ~24h and ~7d via X API batch lookup (public_metrics).
- Weekly lab run scores variables (engagement rate = (likes+2*RT+3*replies+bookmarks)/impressions
  when impressions available, else raw), updates weights, emits next-week plan.
- Explore/exploit: 70% of posts use current best-known combo, 30% test variants.

## experiments.csv schema
post_id,posted_at_utc,slot,format(card|text|thread|qt|reply),hook_type,media(y/n),
tags_count,topic,likes,rts,replies,bookmarks,impressions,followers_at_post,notes
