
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
