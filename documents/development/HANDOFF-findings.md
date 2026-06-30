
---

## Key Findings (40 Players, Full Crew — Current)

Analysis run on **40 target players** with full crew data (13,464 games). All outputs below are current.

### The signal is real and lives between officials

- Between-official ANOVA on defense-adjusted FTA/36 deltas: **F = 1.93, p = 0.000003**
- No individual player's mean delta across all officials is significantly different from zero — the variation is *cross-official for the same player*, not cross-player
- This means: which official is on your game matters for FTA; no player is systematically inflated or deflated by all officials

### Effect size

- Cross-official std of mean adjusted delta: **0.50 FTA/36**
- 80th-percentile spread (P10→P90 official): **0.86 FTA/36** ≈ **0.8 FTA/game** for a 34-min starter
- Extreme pairs exist (Giannis + Dagher = +4.72, Giannis + David Jones = −3.84) but are the tails
- The reliable, repeatable effect is closer to ±0.5 FTA/36 for a typical suppressor/amplifier

### Suppressor consistency across players

Suppressors are not one-player effects — they are **official-level traits**:

| Official | Suppressor Score | Mean Adj Δ | N Players |
|---|---|---|---|
| Phenizee Ransom | 84% | −0.66 | 19 |
| Aaron Smith | 80% | −0.51 | 20 |
| Brandon Adair | 80% | −0.64 | 20 |
| Kevin Scott | 80% | −0.36 | 20 |
| Eric Dalen | 75% | −0.65 | 20 |

When these officials are on the floor, high-FTA players across the board get fewer calls.

### Top amplifiers

| Official | Suppressor Score | Mean Adj Δ | N Players |
|---|---|---|---|
| Bill Spooner | 11% | +1.15 | 9 |
| Monty McCutchen | 20% | +1.12 | 10 |
| Haywoode Workman | 25% | +0.87 | 12 |
| Dedric Taylor | 25% | +0.53 | 20 |
| Eric Lewis | 25% | +0.40 | 20 |

### The amplifier paradox

Top amplifiers have *lower* overall SF rates (e.g., Spooner 3.9 SF/G vs league ~5.7). Correlation between `mean_adj_fta36_delta` and `sf_per_game` is **r = −0.29**. A low overall SF rate does not mean an official is a suppressor for specific players. The player×official interaction is a separate dimension from overall foul-calling volume.

### Defensive adjustment remains minor

- Raw vs adjusted delta correlation: **0.969**
- Sign flip rate: **6.1%**
- Conclusion unchanged: opponent defensive quality is not a major confound.

### RS vs PO: no league-wide individual pattern yet

Among 40 officials with RS/PO splits:
- 20 get more suppressive in PO, 20 get less suppressive
- Mean `rs_po_delta`: +0.09 (essentially zero)
- Among suppressors (adj_Δ < 0): 46% get more suppressive in PO
- Among amplifiers (adj_Δ > 0): 52% get more suppressive in PO
- No evidence of a systematic "playoff whistle" at the individual-official level

The does-harden-choke finding that FTA drops in the playoffs may be crew-composition-driven rather than individual-behavior-driven — Step 5 (predictive model) and Step 7 (DHC merge) will test this.

### Layer 1 cross-validation

Correlation between player-derived suppressor metrics and Layer 1 foul-rate metrics:
- `mean_adj_fta36_delta` vs `sf_pct_of_fouls`: **r = −0.33**
- `suppressor_score` vs `sf_pct_of_fouls`: **r = +0.30**

Moderate alignment — officials who suppress player FTA also call slightly fewer shooting fouls as a share of all fouls, but the player-level metric captures signal beyond what overall foul rates reveal.

### Step 5: Predictive models (June 2026)

**Game-level SF count — weak signal.** Honest temporal holdout (train 2014–22, test 2023–24 + 2024–25):

| Model | RMSE | R² |
|---|---|---|
| League average | 4.56 | — |
| OLS additive (crew features) | 4.53 | 0.005 |

Crew assignment does not meaningfully predict game-level shooting foul volume. Context (teams, pace, style) dominates.

**Player-level FTA/36 — modest signal.** Same holdout, 2,675 test player-games:

| Model | RMSE | R² |
|---|---|---|
| Player baseline only | 3.98 | 0.12 |
| Baseline + crew mean adj delta | **3.96** | **0.13** |
| Static/leaky upper bound (ridge) | 3.53 | 0.31 |

12 of 20 target players improve with crew info. Largest RMSE lift: Russell Westbrook (−0.33), Chris Paul (−0.15), James Harden (−0.13).

**Crew interaction effects beyond additive model:**
- 529 official pairs with ≥20 shared L2M-era games
- 53 pairs with |z|>1.96 on additive residuals (expected 26.5 at 5%)
- Examples: amplifier pairs `J.Goble|M.Lindsay` (z=+3.6), suppressor pairs `E.Malloy|N.Buchert` (z=−3.9)

### Step 6: L2M validation (June 2026)

Cross-check player-derived suppressor metrics against league-audited L2M shooting-foul outcomes (2,698 games, 79 qualified officials).

**Primary hypothesis not confirmed:**

| Comparison | r | p |
|---|---|---|
| `suppressor_score` vs L2M INC/(INC+CC) | +0.02 | 0.86 |
| `mean_adj_fta36_delta` vs L2M INC rate | −0.02 | 0.90 |
| Player×official adj Δ vs L2M INC rate | +0.00 | 0.98 |

**Layer 1 volume metrics validated:**

| Comparison | r | p |
|---|---|---|
| `sf_per_game` vs L2M INC rate | **−0.45** | **<0.001** |
| `sf_pct_of_fouls` vs L2M INC rate | **−0.42** | **<0.001** |

Officials who call more shooting fouls in full-game PBP data have lower missed-foul (INC) rates in L2M clutch situations. The player×official suppressor score does not replicate this pattern — it measures a different phenomenon (full-game player-specific FTA shifts) that does not map cleanly onto L2M clutch adjudication.

**Paper implication:** Do not claim L2M validation for `suppressor_score`. Use L2M validation for Layer 1 metrics. Present player×official effects as full-game predictive/descriptive findings (Step 5b holdout), not L2M-grounded.

### Step 7: does-harden-choke merge (June 2026)

Tests whether playoff FTA collapse is crew-mediated. Coverage: 543 playoff player-games with crew data across 28 players, 2014-15 onward.

**A. RS vs PO crew suppression (game level):**
- RS crew_mean_suppressor_score = 0.482, PO = 0.479 — Mann-Whitney p=0.720 (not significant)
- Crew composition does not systematically differ between RS and PO.

**B. Floor game crew composition:**
- Floor PO games (n=63) vs non-floor PO (n=480): crew_mean_suppressor_score 0.476 vs 0.478 — p=0.764 (not significant)
- BUT actual FTA/36 delta: floor mean = −2.889, non-floor = +0.408 (p<0.001)
- **Floor-game FTA crashes are real and large but not crew-driven.**

**C. Player-specific crew suppression vs actual FTA deviation:**
- Predicted crew suppression (mean player×official adj delta) vs actual fta36_delta
- Spearman r=+0.406, p<0.001 (n=433 player-games, 20 players)
- Consistent direction in 18/18 players; 8/18 individually significant
- Methodological caveat: ~6% of adj delta is PO look-ahead. Clean RS-only holdout is a remaining task.

**Paper implication:** Crew assignment explains continuous FTA variance (r=0.406) but is **not** the mechanism behind DHC floor games. Paper 3 framing needs revision: the crew effect is predictive for individual game outcomes, not a collapse trigger. The floor-game mechanism is upstream of crew (defensive pressure, player state, game context).

### Foul-type classification findings (from does-harden-choke, motivating Layer 2)

The foul-type classification work in does-harden-choke established several findings that motivate the landing foul approach:

**The timing axis is dead as a discriminator:**
- LLM grader achieved 71% binary accuracy (PRE_COMMIT/IN_ACT) on Harden clips using event-ordering prompt (best of four prompt modes)
- But the Giannis counterexample killed timing as a predictor: Giannis gets BEFORE fouls on genuine hard drives (4/16 clips) and loses *more* FTAs (-4.5/36) than Harden (-3.3/36) in playoffs
- Timing doesn't separate fouls-that-persist from fouls-that-disappear

**Mechanism distributions differ descriptively:**
- Harden: 40% ARM-HOOK/PUMP-JUMP (manufactured contact)
- Giannis: 50% DRV-FINISH (genuine finishing contact)
- But the predictive chain (mechanism composition -> FTA shift) is untested

**LLM grading lessons (applicable to landing foul grader):**
- 13-field observation prompt: 40% accuracy (cognitive overload, state classification too hard)
- 3-field observation prompt: 50% accuracy (better but still state-classification framing)
- Event-ordering sequence prompt: 71% accuracy (temporal ordering is easier than state classification)
- Key insight: reformulate the question as something the model can observe (spatial/temporal relationships) rather than classify (instantaneous states)
- Landing fouls should be more gradable because the signal is spatial (defender under landing zone) and the temporal window is wide (~1 second)

**Sample size:** 36 manually classified clips total (20 Harden, 16 Giannis). Insufficient for scale but enough to validate the descriptive hypothesis and kill the timing axis.

### Landing foul LLM grader — validation runs (ref-ball Step 10, 2026-06-29)

Four prompt approaches tested on the primary set (93 YES/NO clips, Vertex `gemini-3.5-flash`):

| Prompt mode | Accuracy | Precision (YES) | Recall (YES) | F1 (YES) | Key failure |
|---|---|---|---|---|---|
| Spatial V1 (no `who_initiated`) | 58.1% | 55.3% | 97.9% | 70.7% | 38/45 GT-NO predicted YES (massive YES bias) |
| Spatial V2 (+`who_initiated`) | ~58% | improved on pump-fakes | new FNs | — | Traded FPs for FNs; no net improvement |
| Whistle attribution | ~58% | similar | similar | — | Model cannot reliably parse whistle from audio |
| gemini-2.5-flash on Vertex | N/A | N/A | N/A | N/A | API Error 400 — `mediaResolution` rejected |

**Precision target: ≥ 85%. Recall target: ≥ 70%.** Recall clears the target on every run (≥ 98%). Precision fails on every run (~55%). The binding constraint is false positives.

**Confusion matrix (Spatial V1):** 7 TN, 38 FP, 1 FN, 47 TP.

**False positive pattern:** Human notes say "contest," "pump-fake," or "shooter-initiated." Model consistently outputs `shot_type=JUMP_SHOT`, `defender_position_at_landing=UNDER_SHOOTER`, `contact_moment=DURING_DESCENT_OR_LANDING` at HIGH confidence on clips where the defender's feet were legal. The model sees any closeout contact on a perimeter jump shot and labels the defender as under the shooter.

**Sequence prompt (2026-06-29):** ~54% accuracy, ~53% precision, 100% recall — worse than spatial. LLM path exhausted.

**Pivot (2026-06-30):** Manual grading complete (300/300 clips). Fine-tuned video classifier (VideoMAE/SlowFast) is the active path. See HANDOFF.md Step 10b.

**Cross-project LLM grading lessons (7 approaches across DHC + ref-ball):**

| Project | Approach | Accuracy | Lesson |
|---|---|---|---|
| DHC | 13-field observation | 40% | Cognitive overload → degenerate output (identical vectors for all clips) |
| DHC | 3-field observation | 50% | State classification still too hard for LLMs |
| DHC | Event-ordering sequence | **71%** | Temporal ordering is cognitively easier than state classification |
| DHC | Timing axis overall | — | Killed by Giannis counterexample (not a grading problem — a hypothesis problem) |
| ref-ball | Spatial V1 | 58% | YES bias — model cannot distinguish contest from undercut |
| ref-ball | Spatial V2 | ~58% | Adding `who_initiated` trades FPs for FNs |
| ref-ball | Whistle attribution | ~58% | Audio signal unreliable — model hallucinates whistle timing |

**Structural diagnosis:** Current multimodal LLMs process video as loosely-connected frames, not continuous physical simulations. They can identify objects and describe spatial relationships but cannot: (a) track sub-second temporal ordering between simultaneous body movements (~200–400ms windows), (b) distinguish cause from consequence in fast interactions (did the shooter jump into the defender, or vice versa?), (c) detect absence (defender's feet were NOT under the landing zone), or (d) use audio as a reliable signal.

**If sequence + few-shot fails to reach 75% precision:** Switch to a hybrid pipeline — use the LLM's 98% recall as a pre-filter, then manually review predicted-YES clips. This cuts manual workload by ~50% while maintaining perfect precision. See HANDOFF.md Step 10 for full details.
