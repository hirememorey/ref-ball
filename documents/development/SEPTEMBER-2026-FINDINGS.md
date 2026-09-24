# September 2026 findings

Work between September 5 and 24, 2026, after the July handoff. Everything here was run
outside the July pipeline; only the L2M contact work (sections 8-10) has code in this
repository, under [`src/l2m_contact/`](../../src/l2m_contact/). Artifacts for sections 1-7
(clips, grader databases, model outputs) are not in the repository.

**Net position, September 24:**

1. Broad referee-level free-throw-rate differences partly survive controls. Player-specific
   referee "fingerprints" do not show reliable predictive value in a chronological holdout.
2. In L2M windows, with contact type held constant, crews do not detectably differ in how often
   they let league-judged illegal contact go.
3. The landing-foul angle is closed: called landing contact is rare, and the June grader's
   high recall came from a yes-bias.
4. LLM vision models (Sonnet and GPT-6 Luna on frames, Gemini 3.8 Flash on native video) do
   not reliably separate league-judged illegal from marginal contact in broadcast clips. With
   audio, Gemini's "reaction suggests a foul" flag is precise (5 of 29 missed calls, 0 of 32
   marginal no-calls) but low recall and home-biased.
5. League verdicts exist outside L2M only for called fouls (coach's challenges). Uncalled
   contact outside L2M still requires human grading.

---

## 1. Do the referee fingerprints survive controls? (Sep 5)

Reanalysis of 2014-15 to 2024-25 player box scores and crews from pinned public archives
(12,702 player-games in the primary 20-player sample; 23,753 for all 40 players). Outcome:
FTA per 36 minutes.

- **Broad associations partly survive.** Profile dispersion falls from 0.449 to 0.317 FTA/36
  (about 29%) after controls for player-season, opponent-season, team-season, home, postseason,
  rest and minutes, with all three officials estimated jointly. Omnibus test over 85 officials
  with 100+ player-games: F = 2.007, p ≈ 1.6e-7. Three officials pass 5% FDR (Goble -0.62,
  Dalen -0.57, Goldenberg +0.74 FTA/36). Controlled ranks correlate 0.38 between 2014-19 and
  2019-25.
- **Predictive value is not established.** Train through 2020-21, tune on 2021-23, test on
  2023-25 (2,587 player-games). Crew main effects improve RMSE by 0.00062 FTA/36 (about 0.02%);
  player x official terms make it worse. Both bootstrap intervals include zero.

Implication: the July r = 0.406 crew-prediction result and the player-specific
suppressor/amplifier framing should not be presented as validated prediction.

## 2. Drive-event pilot and Harden film room (Sep 5-18)

Ten 2019-20 games (five with original labels, five seeded random). Deterministic extraction
from archived PBP produced 1,049 candidate event groups, 184 involving Harden; all 20 original
foul labels in those games were recovered by exact event ID (11 tests passing). NBA.com video
retrieval works through the browser: `videoeventsasset` per event, and box-score FGA/TOV
playlists via `videodetailsasset`. 181 of 184 Harden clips were cached and duration-verified.

A local grading app ("film room") recorded human grades. Review on September 18: 62 of 181
graded. Of 38 confirmed drives (rubric v2), 25 had visible contact; 14 of those had a linked
recorded foul and 11 did not. No-contact drives never had a linked foul. Small, one grader,
outcome-visible film: descriptive only.

## 3. Tracking-data screen for landing contact (Sep 18)

`leaguedashplayerptshot` closest-defender buckets are aggregates (player x filters), not
event-level. Joining date + quarter + player to PBP reconciled exactly (170/170 3PA in two
games) and cut the first-pass queue from 170 to 47 clips. That is a workload number, not
recall. A later check found the 144 historical landing-foul YES labels are all non-FGA foul
events, so none are eligible to validate a shot-based distance screen (design error, not an
empirical failure).

## 4. Terra closeout benchmark (Sep 19)

GPT-5.6 Terra screening of eight clips: 8 YES / 0 NO, so no discrimination and no viewing time
saved. Native 8 fps frames fixed one localization error in a follow-up. No accuracy estimate.

## 5. Automated contact retrieval (Sep 20-22)

Blinded Terra descriptions, then descriptor-similarity retrieval. Only 2 of 6 top suggestions
carried the matching human label; Terra marked definite contact on 4 of 6 negative controls.
Gemini 3.8 Flash through the Cursor CLI found definite contact on 5 of 12 known positives and
reported receiving text descriptions of images rather than pixels.

## 6. Landing-contact side check (Sep 22-23)

Does the June Gemini grader's false-positive problem come from real landing contact in clips
labeled NO? Blind re-grade of 30 NO clips plus 6 YES controls: 3/29 NO clips contain landing
contact (10%, Wilson 95% CI 4-26%); controls 6/6. The June grader said YES to three-point
shooting fouls generally. **Verdict: close the landing angle.** The July VideoMAE runs' per-clip
predictions were not saved, so their false positives were not re-examined.

## 7. NBA.com exploration (Sep 23)

- `videodetailsasset` with `ContextMeasure=DEF_FGA` returns event-level shots a player defended
  (tracking-assigned defender), which the September 18 audit had missed. No distance or
  contact field.
- Tracking ContextMeasures (drives, touches, hustle) return 400; drive counts are not linked
  to video.
- **L2M referee clips**: every L2M row has a `VideolLink` event number with a clip at
  `https://ak-static.cms.nba.com/wp-content/uploads/referee-clips/{gameId}_{VideolLink}_DF%20BCAST_1509kbps.mp4`,
  1080p, about 20 s, downloadable without a browser. `VideolLink` is not a stats `GameEventID`.

## 8. L2M contact data (Sep 23)

- All 3,132 L2M reports, 2018-19 to 2025-26, fetched sequentially with no blocks: 64,302 rated
  plays, 57,209 fouls (CNC 43,032, CC 11,305, INC 2,399, IC 419).
- Crews: existing crew file plus 415 new 2025-26 games from nba.com game pages, working
  officials only (playoff alternates dropped). 5 plays lack a crew.
- 2023-24 clip pilot: all 297 foul INC plus 701 stratified foul CNC; 988 of 993 unique clips
  valid (981 at 1080p), 11.4 GB. Spot checks confirmed clip content matches the comment and
  the burned-in game clock locates the play. About 6% of rows share a clip with another row.

**How this differs from Step 6 (June):** Step 6 used shooting fouls only and correlated each
official's INC/(INC+CC) with full-game metrics. It never used the clips, the comment text,
CNC plays as examples of uncalled contact, or non-shooting fouls (about two thirds of CNC/INC).

## 9. Comment tagging and crew test (Sep 23-24)

Each foul comment is tagged `none / marginal / affecting / legal / not_contact_judgment /
unclear` for the committing player, without seeing the league decision.

- **Rules** (`classify_comments.py`): transparent regexes; 8 of 25 human-graded rows correct.
- **LLM hybrid** (`llm_tag.py`, `hybrid.py`): GPT-6 Luna tags every row (batch 200); Sonnet
  re-checks the 16.2% flagged by a decision-blind rule (`recheck_rule.py`). Sonnet changed 24%
  of re-checked labels. Hybrid: 23 of 25 human-graded rows (the prompt was developed against
  these rows, so this is optimistic; 65 audit rows remain ungraded as a clean test).
  Missed calls tagged affecting: 85% (most of the rest are take fouls / three seconds). Foul
  no-calls: 59.5% marginal, 28.1% legal, 9.4% no contact, 1.0% affecting. Cost: about $1.50
  for Luna, $16.29 (list) for the Sonnet re-check.

**Crew test** (`analyze_crews.py`, `heldout_null.py`): plays the league judged fouls (CC + INC),
excluding take fouls / three seconds / timeouts; about 9,100 plays, 2,050 missed. L2-penalized
logistic model with all three officials jointly plus foul type, body region, contact wording,
Q4 vs OT and season. Permutations shuffle whole crews across games within season.

| Test | Rules tags | Hybrid tags |
|---|---|---|
| Omnibus permutation p | 0.21 | 0.16 |
| Officials passing 5% FDR | 0 of 84 | 0 of 84 (lowest q = 0.11) |
| Held-out gain vs crew shuffles, p | 0.16 | 0.19 |

A game-bootstrap interval had made the held-out gain look significant; the crew-shuffle null
shows it is season drift. The same three officials top the fewer-misses list under both tag
sets (Van Duyne, Foster, Zarba), not significant.

**Unexpected:** the missed share of league-judged fouls fell from 33% (2018-19) to 13-17%
(2024-26). Better officiating, a change in how reports are written, or a change in which plays
are listed; unexplained. Step 6's r = -0.45 (shooting fouls per game vs INC rate) may partly
reflect this drift; unverified.

## 10. Coach's challenges and the video test (Sep 23-24)

**Coach's challenges** (`coach_challenges.py`): the only league verdicts on called fouls
outside L2M. 2019-20 to 2022-23: 3,023 challenges; 1,990 linked to a called foul at the same
clock; 1,605 outside the last two minutes (465 overturned, 592 ruling stands, 548 support
ruling). The foul text names the calling official. Caveats: coaches choose what to challenge;
a charge overturned to a block is still a foul; outcome codes 4 and 6 each matched one v3 example.

**Video test** (`video_test/`): can a model separate league-judged illegal from marginal or
wrongly called contact in 2023-24 L2M clips? Two tests with matched whistle status and foul
types: uncalled (INC vs marginal CNC) and called (CC vs IC). Models saw 20 frames (5.5-15 s,
1280 px) plus foul type, players and clock, never the decision.

| Model | Test | Clips | AUC | p |
|---|---|---:|---:|---:|
| Sonnet | Uncalled | 61 | 0.64 | 0.035 |
| Sonnet | Called | 46 | 0.52 | 0.41 |
| GPT-6 Luna | Uncalled | 100 | 0.51 | 0.45 |
| GPT-6 Luna | Called | 72 | 0.38 | 0.96 |

Sonnet's uncalled result does not survive correction for four tests (Bonferroni 0.0125); it
was right 62% of the time on judgeable clips and could not judge about a quarter. Sonnet stopped
at 107 of 172 clips ($30.64 list).

**Gemini native video (Sep 24, Vertex).** Gemini 3.8 Flash (`gemini_video.py`) on the same 61
uncalled clips, whole ~20 s clip at 10 fps and high media resolution (~52K tokens per clip).
Two arms: audio removed (a clean test of seeing contact; the frame models had no audio) and
with broadcast audio (a practical filter, since crowd and commentator reactions are evidence
that a play was contestable).

| Model | Uncalled clips | AUC | p | Right when judged | Cannot see |
|---|---:|---:|---:|---:|---:|
| Gemini, muted | 61 | 0.62 | 0.06 | 60% | 1 |
| Gemini, with audio | 61 | 0.63 | 0.04 | 63% | 1 |
| Sonnet, 20 frames | 61 | 0.64 | 0.03 | 62% | 14 |

Gemini sees nearly every play but calls almost everything marginal: of 29 missed calls, it said
illegal on 4 (muted) or 6 (audio). Native video fixed visibility, not discrimination.

The audio arm's own reaction field is more interesting than its score: "audio suggests a foul"
fired on 5 of 29 missed calls and 0 of 32 marginal no-calls (one-sided Fisher p = 0.02). Precise
but low recall (17%), and home-biased: all 5 flags were on plays where the fouled team was at
home (5/17 home vs 0/11 away, p = 0.06). Broadcast audio mostly finds home-team missed calls
that drew a reaction. Cost: $2.72 muted, $2.68 audio (introductory pricing, $0.75 / $3.75 per
million input / output tokens).

Likely limits of broadcast video: one camera angle, contact shorter than frame spacing, and the
play not exactly centered in the clip.
