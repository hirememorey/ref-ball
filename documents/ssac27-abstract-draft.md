# SSAC27 Abstract Draft — Paper 1

**Status:** DRAFT v2 — June 29, 2026
**Deadline:** October 1, 2026 11:59 PM EST
**Track:** Basketball
**Word limit:** 500 (including title)
**Figures/tables allowed:** Up to 2
**Figures:** `output/figures/table_a_suppressor_amplifier.png` (Table 1), `output/figures/figure_b_crew_prediction_scatter.png` (Figure 1)
**Generator:** `src/generate_abstract_figures.py`

**TODO (v3):** Incorporate foul-type specificity finding — SF/NSF per-official rate independence (r=0.152, p=0.15), residualized ANOVA (100% of effect survives), η² by foul type table. This strengthens the "interpretive" vs "volumetric" claim in Results.

---

## Individual NBA Referees Have Systematically Different Effects on Free Throw Rates: Evidence from 13,278 Games

### Introduction

NBA officiating is analyzed almost exclusively at the crew level or in aggregate. Prior work has examined racial bias in foul-calling (Price & Wolfers, 2010), referee profiles from Last Two Minute reports (Pelechrinis, 2023), and referee impact via win-probability models (Duma & Benaharon, 2026), but none have achieved whistle-by-whistle attribution to individual officials across full games. We exploit an overlooked feature of the NBA's play-by-play data — the calling official's name embedded in every foul description — to construct per-official shooting foul profiles and measure how individual referees systematically shift free throw attempt (FTA) rates for high-usage players.

### Methods

We parse the calling official from the `description` field of the NBA PlayByPlayV3 API for all 13,278 regular season and playoff games from 2014-15 through 2024-25. We construct per-official shooting foul profiles for 101 qualified officials, then compute defense-adjusted FTA/36 deltas for 40 high-FTA players (FTA/36 >= 5.0, >= 400 career games) across 3,846 player-official interaction pairs. Each delta measures how a player's FTA/36 shifts when a specific official is on the crew versus all other games, adjusted for opponent defensive rating. We validate volume metrics against league-audited Last Two Minute (L2M) report outcomes, test for predictive power using temporal holdout models (train 2014-22, test 2023-25), and merge with playoff FTA data to test whether crew assignment explains postseason FTA shifts.

### Results

Individual officials produce significant heterogeneity in player FTA rates (ANOVA F=1.93, p=0.000003). The 80th-percentile spread across officials is 0.86 FTA/36, equivalent to approximately 0.8 FTA per game for a 34-minute starter. This heterogeneity is an official-level trait, not a one-player effect: Phenizee Ransom suppresses FTA for 84% of high-FTA players (mean adjusted delta -0.66 FTA/36); Brandon Adair, Aaron Smith, and Kevin Scott each suppress 80% (Table 1). Counterintuitively, top FTA amplifiers like Bill Spooner (+1.15 FTA/36 delta) and Monty McCutchen (+1.12) call fewer total shooting fouls per game than average (3.9 SF/game vs. league ~5.7; r=-0.29), indicating that player-specific FTA effects are a separate dimension from overall foul-calling volume. Officials who call more shooting fouls have lower missed-call rates in L2M clutch situations (r=-0.45, p<0.001). In a temporal holdout, player-specific crew suppression scores predict actual playoff FTA deviation (Spearman r=0.406, p<0.001, n=433 player-games across 20 players; Figure 1). However, the large FTA collapses observed in playoff "floor games" (mean -2.889 FTA/36) are not crew-driven (p=0.764).

### Conclusion

Individual NBA referees have measurable, consistent, and predictable effects on player free throw rates. These effects are identifiable from publicly available play-by-play data that has been hiding in plain sight. The complete dataset — per-official profiles, player-official interaction tables, and predictive model outputs for 101 officials across 11 seasons — is published as an open-source resource for teams, researchers, and the league.

---

## Word Count

~460 words (under 500 limit)

## Figures

**Table 1:** `output/figures/table_a_suppressor_amplifier.png` — Top 5 suppressor and amplifier officials with named officials, suppressor scores, mean adjusted FTA/36 deltas, SF/game, and N players. Shows the amplifier paradox (Spooner/McCutchen have +1.15/+1.12 deltas but only 3.9 SF/game).

**Figure 1:** `output/figures/figure_b_crew_prediction_scatter.png` — Predicted crew suppression vs. actual FTA/36 deviation from regular season baseline. Spearman r=0.406, p<0.001, n=433 playoff player-games, 20 players. Shows the predictive claim is real.
