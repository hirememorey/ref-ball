# ref-ball

**Per-official NBA shooting-foul profiles, predictive crew models, and contact-type classification — understanding how individual referees interpret contact differently.**

> **Picking up development?** See [documents/development/HANDOFF.md](documents/development/HANDOFF.md) for current data inventory, findings, and the exact next steps to run.

## The Strategy

**Primary aim:** Understand how each individual NBA referee interprets contact. The aggregate question (do refs call different games?) is answered — ANOVA p=0.000003. The next question is *why*: do refs differ in how they interpret specific types of contact, starting with landing fouls?

**Secondary aim:** Predict the officiating profile of a game from crew assignment. Steps 5-7 (predictive crew models + does-harden-choke merge) are complete. Game-level prediction is weak (R^2~0.005); player-level prediction is modest (R^2=0.13). The continuous prediction works (r=0.406) but crew assignment is not the mechanism behind playoff FTA collapse.

The dataset is the asset. Papers are downstream products. Build the dataset once, then query it for multiple findings.

The dataset has three layers, each with a different novelty moat:

| Layer | What it is | Novelty moat | Status |
|---|---|---|---|
| **Layer 1: Per-official attribution** | Official name parsed from PBP `description` field | Weak (anyone can parse it) | **Complete** — 13,278 games ingested, 13,464 with crew |
| **Player x official profiles** | FTA/36 deltas per player under each official, defense-adjusted | Medium (requires crew + game logs) | **Built (40 players)** — full crew, 3,846 pairs, ANOVA p=0.000003 |
| **Layer 2: Contact-type classification** | LLM-graded foul categories from video (starting with landing fouls) | Strong (multimodal LLM + video at scale) | **Active** — tooling migrated from does-harden-choke; landing foul grader is next build |
| **Layer 3: No-call detection** | Predicted missed fouls on non-called contact plays | Strong (requires video model + full-game video) | **Shelved** — L2M INC available for validation; video path not pursued |

**Current build order:** Layers 1 + player x official profiles + predictive models (Steps 1-7) are **complete**. DHC tooling merge (Step 8) is **complete**. The active frontier is **Layer 2: landing foul classification** — build an LLM grader to measure per-official landing foul calling rates and test for inter-ref variance.

## The Paper Sequence

Each paper builds on the dataset from the previous one. We do not need all three layers before publishing.

### Paper 1 — "Refs call different games" (Layer 1 + player x official profiles)

**Claim:** Individual NBA referees have systematically different effects on shooting-foul and FTA rates — especially for high-FTA players — and those effects are predictable from crew assignment.

- Per-official called SF rates (full game, Layer 1)
- Per-official x player FTA deltas, opponent-defense-adjusted
- RS vs PO comparison per official
- Validation against L2M INC shooting fouls (league-audited ground truth) — **Layer 1 validated; player-derived suppressor score not confirmed (see Step 6)**
- Does not require foul-type classification or a video model
- **Status: Steps 1-7 complete. Findings support the claim but predictive R^2 is modest. Strongest result is descriptive heterogeneity (ANOVA p=0.000003) + crew interaction effects.**

### Paper 2 — "Refs interpret contact differently" (Layer 1 + Layer 2)

**Claim:** Individual referees call specific *types* of contact at significantly different rates, and these type-specific rates explain the aggregate suppressor/amplifier effects from Paper 1.

- Start with **landing fouls** — a single, well-defined category with an explicit rule (landing space protection, 2022-23 point of emphasis)
- LLM-graded binary classification (landing foul yes/no) from shooting foul video clips
- Measure per-official landing foul calling rates; test for significant inter-ref variance
- If variance is real: landing foul tolerance becomes a *mechanism* for the suppressor/amplifier effect — the "why" behind Paper 1's "what"
- Expand to additional contact types if landing fouls prove the concept

**Why landing fouls first:**
- The visual signal is spatial (defender's feet under shooter's landing zone), not temporal — much more LLM-gradable than the timing axis that achieved only 71% binary accuracy in does-harden-choke
- The rule is explicit (landing space protection), reducing subjective judgment in both human and LLM grading
- The sequence plays out over ~1 second (jump, release, descend, land), giving the model a wide temporal window
- If refs vary on a category with an *explicit* rule, variance on ambiguous categories is almost guaranteed

**Previous approaches and why they failed (from does-harden-choke):**
- The v3 five-axis taxonomy (mechanism/body part/timing/severity/location) was built top-down from basketball domain knowledge. It produced ~500+ combinatorial states when only 3-4 distinctions matter.
- The timing axis (BEFORE/DURING/AFTER) was the most engineered — four prompt iterations, the entire LLM grader — but the Giannis counterexample killed it as a discriminator. Giannis gets BEFORE fouls on genuine drives and loses *more* FTAs than Harden in the playoffs. Timing doesn't separate fouls-that-persist from fouls-that-disappear.
- The 13-field observation prompt (40% accuracy) asked the model to classify freeze-frame states — the hardest possible formulation. The event-ordering prompt (71% accuracy) reformulated as temporal sequencing, which was better but still insufficient for scale on a fundamentally ambiguous temporal boundary.
- The manufactured/genuine binary (defined in config but never operationalized) is descriptively valid (Harden 40% ARM-HOOK/PUMP-JUMP vs Giannis 50% DRV-FINISH) but the predictive chain through it is untested.

### Paper 3 — "Referee assignment and playoff FTA shifts" (all layers + collapse data)

**Claim (revised):** Crew assignment explains continuous variance in individual playoff FTA outcomes but is not the mechanism behind floor-game FTA collapse.

- **Step 7 complete.** Crew composition does not differ between RS and PO (p=0.720). Floor-game FTA crashes (mean -2.889 FTA/36) are not crew-driven (p=0.764). But player-specific crew prediction correlates with actual FTA deviation (r=0.406, p<0.001).
- Paper 3 framing needs revision: crew is a continuous modulator, not a collapse trigger. The floor-game mechanism is upstream (defensive pressure, player state, game context).
- If Paper 2 succeeds (contact-type-specific ref variance), Paper 3 may be reframeable: certain contact types disappear under certain refs, and playoff crew composition shifts the contact-type mix.

## The Data Source

The NBA play-by-play JSON (from the `playbyplayv3` API endpoint) includes the calling official's name in the foul `description` field. For example:

```
"Gasol S.FOUL (P1.T1) (R.Garretson)"
```

The parenthesized name after the foul classification is the official who blew the whistle. Parseable with regex `\(([A-Z]\.\s*\w+)\)\s*$`.

**Coverage:** 100% for games from 2014-15 onward (game IDs `00414...` and later). 0% before that. In the does-harden-choke dataset, all 39 playoff games from 2014-15 through 2024-25 have 100% official attribution on shooting fouls (680/680). The per-official analysis is constrained to 2014-15 onward — 11+ seasons.

**Why nobody has used this:** The official name is in an unstructured text field, not a typed API field. Programmatic consumers parse the structured fields (`actionType`, `subType`, `personId`) and skip `description`. The data was hiding in plain sight. Owen Phillips (The F5) found it and built a descriptive database, but nobody has used it for hypothesis-driven research.

## Literature Positioning

| Paper | What they did | What they couldn't do | What this project does |
|---|---|---|---|
| Price & Wolfers (2010) | Racial bias in NBA referee foul-calling | Crew-level, all foul types | Per-official, shooting fouls only |
| Price, Remer, Stone (2012) | Shooting foul advantage, profitable biases | "We do not have data on individual ref calls" | Per-official shooting foul profiles |
| Pelechrinis (2023) | Referee profiles from L2M data | L2M only (last 2 min), no foul-type breakdown | Full-game, foul-type-specific |
| Duma & Benaharon (2026) | Referee Impact Metric (win-probability) | "Referee-game association, not whistle-by-whistle attribution" | Whistle-by-whistle attribution |
| Noecker & Roback (2012) | Individual referee effects on foul evening-out | NCAA only, foul evening-out not shooting profiles | NBA, shooting foul profiles |
| Owen Phillips / The F5 | Individual referee database, 60K+ shooting fouls | Descriptive (counting/ranking), no hypothesis testing | Hypothesis-driven, connected to game outcomes |
| **This project** | **Per-official foul call + no-call profiles, connected to collapse dynamics** | — | — |

## Pipeline

```
LAYER 1 + PROFILES (Steps 1-9 — COMPLETE)
1. Fetch PBP           →  src/fetch_pbp.py                 →  data/raw/pbp/*.json
2. Fetch L2M           →  src/fetch_l2m.py                 →  data/processed/l2m_events.parquet
3. Fetch crew (all)    →  src/fetch_crew_all.py            →  data/processed/crew_assignments.parquet
4. Ingest              →  src/ingest.py                    →  data/processed/games/{game_id}.parquet
5. Layer 1 profiles    →  src/ref_profiles.py              →  data/processed/ref_profiles.parquet
6. Player game logs    →  src/player_official_profiles.py  →  data/processed/player_official/player_games/
7. Player x official   →  src/player_official_profiles.py  →  player_official_interactions.parquet
8. Defense adjustment  →  src/defensive_adjustment.py      →  defensive_adjusted_interactions.parquet
9. Official profiles   →  src/official_calling_profiles.py →  official_calling_profiles.parquet

PREDICTIVE MODELS + VALIDATION (Steps 10-13 — COMPLETE)
10. Game crew model    →  src/crew_predictive_model.py      →  data/processed/model/
11. Player crew model  →  src/player_crew_predictive_model.py → data/processed/model/player/
12. L2M validation     →  src/l2m_validation.py             →  data/processed/model/l2m/
13. DHC merge          →  src/dhc_merge.py                  →  data/processed/model/dhc_merge/

LAYER 2: CONTACT-TYPE CLASSIFICATION (Steps 14-18 — NEXT)
14. Build clip manifest →  src/foul_type_scraper.py          →  data/processed/foul_type_manifest_*.json
15. Manual ground truth →  src/foul_type_classifier.py       →  data/foul_type_classifications.csv
16. LLM grading        →  src/foul_type_llm_grader.py       →  data/processed/foul_type_llm_results_*.json
17. Per-official rates  →  (TBD)                             →  (TBD)
18. Variance analysis   →  (TBD)                             →  (TBD)
```

Steps 1-13 are **complete**. Step 14 (DHC tooling merge) is **complete**. Steps 15-16 have tooling in place (needs adaptation for landing foul binary + official-diversity sampling). Steps 17-18 are new analysis code to be built.

### 1. Ingest (Layer 1 — built)

Parse PBP JSON files and extract per-official foul call records. The calling official is parsed from the `description` field.

Each record includes:

| Field | Description |
|---|---|
| `game_id` | NBA game ID (10-digit) |
| `season` | Season string (e.g. 2023-24) |
| `season_type` | Regular Season / Playoffs |
| `event_num` | Action number within game |
| `period` | Quarter |
| `clock` | Game clock |
| `event_type` | Foul, Violation, etc. |
| `foul_type` | Shooting, Personal, Offensive, Technical, Flagrant, etc. |
| `caller_official_name` | Name parsed from description (e.g. "R.Garretson") |
| `committing_player_id` | Player who committed the foul |
| `committing_player_name` | Player name |
| `committing_team_id` | Team ID |
| `committing_team_tricode` | e.g. HOU |
| `score_home` | Home score at time of foul |
| `score_away` | Away score at time of foul |
| `margin` | Score differential |
| `description` | Raw PBP description string |

Output: one parquet file per game.

```bash
make ingest                    # all PBP files
make ingest --season 2023-24  # single season
```

### 2. Player×official interaction profiles (current focus)

For each high-FTA target player, compute how their FTA/36 shifts under different officials vs. their baseline (games without that official). Defense-adjusted using `opponent_defrtg` from does-harden-choke.

```bash
PYTHONPATH=. .venv/bin/python src/player_official_profiles.py fetch    # download game logs
PYTHONPATH=. .venv/bin/python src/player_official_profiles.py build    # compute interactions
PYTHONPATH=. .venv/bin/python src/player_official_profiles.py summary  # print table

PYTHONPATH=. .venv/bin/python src/defensive_adjustment.py build        # opponent-adjusted deltas
PYTHONPATH=. .venv/bin/python src/defensive_adjustment.py summary
```

**Output:** `data/processed/player_official/` — per-player game logs, interaction pairs, defense-adjusted deltas.

**Target players:** 40 high-FTA players defined in `config/target_players.py` (FTA/36 ≥ 5.0, ≥ 400 career games, 2014-15 onward). All IDs verified via `commonplayerinfo` API.

### 2a. Predictive models + L2M validation + DHC merge (Steps 5-7 — complete)

```bash
make model-crew-temporal         # game-level SF model (honest holdout)
make model-player-crew           # player FTA/36 model (temporal)
make model-crew-diagnose         # compare static vs temporal signal
make l2m-validate                # L2M INC cross-check
make dhc-merge                   # does-harden-choke merge (Step 7)
```

**Outputs:** `data/processed/model/`, `data/processed/model/player/`, `data/processed/model/l2m/`, `data/processed/model/dhc_merge/`

### 2b. Landing foul classification (Layer 2 — next build)

Video-based binary classification of shooting fouls as landing fouls, using multimodal LLM grading. See "Foul-type classification" section for full plan.

```bash
# (post-merge, tooling from does-harden-choke)
# Step 14: build clip manifest for target officials
PYTHONPATH=. .venv/bin/python src/foul_type_scraper.py --by-official
# Step 15: manual ground truth via HTML classifier
PYTHONPATH=. .venv/bin/python src/foul_type_classifier.py --mode landing
# Step 16: LLM grading
PYTHONPATH=. .venv/bin/python src/foul_type_llm_grader.py --prompt-mode landing
```

### 3. Profile

Build per-official profiles from called fouls (Layer 1) and contact-type classification (Layer 2).

**Profile dimensions (Paper 1 — current, without Layer 2):**

| Axis | Metric | Source |
|---|---|---|
| **Called foul volume** | Shooting fouls per game | Layer 1 (PBP) |
| **Player x official FTA delta** | Defense-adjusted FTA/36 shift per player | Layer 1 + crew |
| **Suppressor score** | Fraction of target players suppressed | Layer 1 + crew |
| **Context** | % called in leading vs. trailing situations | Layer 1 (PBP) |
| **Season type** | RS vs. PO profile delta | Layer 1 |

**Profile dimensions (Paper 2 — planned, with Layer 2):**

| Axis | Metric | Source |
|---|---|---|
| **Landing foul rate** | Landing fouls / total shooting fouls called | Layer 2 (LLM classification) |
| **Landing foul variance** | Inter-ref variance on landing foul calling rate | Layer 2 |
| Additional contact types | TBD — expand if landing fouls prove the concept | Layer 2 |

## Relationship to sibling projects

### does-harden-choke (merged — now frozen archive)

does-harden-choke was a **sibling project, now partially merged** into ref-ball. The original split was player-side vs. referee-side, but the active frontier (foul-type classification via LLM) serves ref-ball's question directly (how do refs interpret specific contact types?). The merge is **complete** (Step 8).

**What merged into ref-ball (active tooling):**
- `src/foul_type_scraper.py` — clip manifest builder from PBP data
- `src/foul_type_classifier.py` — HTML tool for manual foul-type classification
- `src/foul_type_llm_grader.py` — multimodal LLM grader (Gemini/OpenAI/Anthropic, ~1244 lines)
- `foul_type_classifications.csv` — 36 manually classified clips (Harden + Giannis ground truth)
- Foul-type manifests and LLM results (data artifacts)
- `analysis_table.csv` — copied into `data/processed/` (ref-ball is self-contained)
- `nba_client.py` — missing methods (`get_common_player_info`, `get_league_game_finder`, `get_league_team_stats`) merged from DHC

**What stays in does-harden-choke (frozen research):**
- All analysis screens (screen_a through screen_f) — finished pass 1 research
- FTA dependency deep-dive (r=-0.528, p=0.002) — strongest finding, published
- Architecture model, trigger taxonomy, mechanism models — all finished or killed
- Pass 2 possession pipeline — production-ready but separate research question
- are_you_gonna_floor prediction model — complete
- All documentation, blog posts, README (the publishable article)
- Raw player/team data and all processed analysis outputs

**Stubs deleted during merge:**
- `src/feasibility_study.py` — never executed, has bugs, superseded by the LLM grader
- `src/nocall_model.py` — all methods raise `NotImplementedError`
- `src/analyze.py` — all tracks raise `NotImplementedError`

**does-harden-choke is now a frozen archive** — the published findings repo. Its README points to ref-ball for active development. The only remaining DHC dependency is the PBP data symlink (`data/raw/pbp/` → DHC).

### cranky-scott-foster

Another sibling project on the same L2M data. Key distinction:

| | cranky-scott-foster | ref-ball |
|---|---|---|
| Question | What *situations* produce errors? | How will *this official* call shooting fouls? |
| Unit of analysis | Decision context (taxonomy bins) | Official identity / crew assignment |
| Finding | Signal is primarily structural context, not referee identity | Official×player FTA heterogeneity is significant (p=0.000003 on full data) |
| Reuse | Taxonomy, crew features, experience tiers — import for conditioning, don't rebuild | — |

ref-ball's claim must be tested *conditional on decision context* to avoid conflating assignment composition with competence. See HANDOFF Step 6 (L2M validation complete; CSF taxonomy conditioning still open).

## Foul-type classification (Paper 2 — active frontier)

### Current approach: landing foul binary

The immediate build is a **single-category binary classifier**: is this shooting foul a landing foul (yes/no)?

Landing fouls are the ideal starting category because:
1. The rule is explicit (defender must give shooter landing space)
2. The visual signal is spatial, not temporal — defender's feet under shooter's landing zone
3. The sequence is slow (~1 second from jump to land) — wide temporal window for LLM
4. If inter-ref variance exists on an explicit rule, it validates the broader approach

### Implementation plan

1. **Ground truth (manual, ~50-60 clips):** Adapt the HTML classifier for a single binary question. Pull clips from games selected for official diversity (not player diversity). Target ~50-60 shooting fouls across 5-6 games with different crews.
2. **LLM prompt design:** Spatial-observation prompt — three questions about shot type, defender position at descent, and contact moment. Derive landing foul from the answers. Test direct prompt as alternative. Gemini native video upload (best performer on prior timing work).
3. **Validation:** Binary accuracy, precision (prioritized — false positives inflate per-official rates), recall. Target 85%+ precision, 70%+ recall.
4. **Scale:** Sampled classification — ~100-150 shooting foul clips per target official (10-15 officials spanning the suppressor/amplifier spectrum). Estimate per-official landing foul rate from sample.
5. **Analysis:** ANOVA on per-official landing foul rates. Correlation with existing suppressor/amplifier profiles.

### Legacy taxonomy (reference, not active)

The v3 five-axis taxonomy from does-harden-choke is preserved for reference but is not the active classification approach. It was built top-down from basketball domain knowledge and proved inadequate — the timing axis was killed by the Giannis counterexample, and 12 mechanism categories are too granular for statistical power at the per-official level.

| Axis | Values | Status |
|---|---|---|
| **Mechanism** | DRV-FINISH, DRV-INIT, ARM-HOOK, CONTEST, LANDING, PUMP-JUMP, RIP-THRU, POST, PUTBACK, OFFBALL, TAKE, AMB | Descriptively valid, predictively untested |
| **Body Part** | HEAD, ARM, CHEST, SHOULDER, LOWER | Manual only |
| **Timing** | BEFORE, DURING, AFTER | **Killed** — does not discriminate FTA shift |
| **Severity** | STRONG, MEDIUM, MARGINAL | Manual only |
| **Location** | RA, PAINT, MID, PERIM | Derivable from PBP |

Manufactured vs. genuine sets (defined in `config.py`, never operationalized):
- **Manufactured**: ARM-HOOK, PUMP-JUMP, RIP-THRU, DRV-INIT (contact-seeking)
- **Genuine**: DRV-FINISH, CONTEST, LANDING, PUTBACK (real basketball contact)

## Project structure

```
ref-ball/
├── README.md                         # This spec doc
├── config.py                         # Paths, seasons, constants
├── config/
│   ├── __init__.py                   # Re-exports config.py (package shadows module)
│   └── target_players.py             # CORE_PLAYERS + EXPANDED_PLAYERS (40 total)
├── requirements.txt
├── Makefile
├── data/
│   ├── foul_type_classifications.csv # Manual ground truth (36 clips, from DHC)
│   ├── raw/
│   │   └── pbp/                     # PBP JSON (symlink → does-harden-choke)
│   └── processed/
│       ├── analysis_table.csv       # Copied from DHC (opponent_defrtg, player game logs)
│       ├── games/                   # Per-game foul parquets (13,278)
│       ├── crew_assignments.parquet # Full 3-person crew per game (13,464 games)
│       ├── l2m_events.parquet       # L2M ground truth (56K events)
│       ├── ref_profiles.parquet     # Layer 1 per-official called-foul profiles
│       ├── foul_type_manifest_*.json  # Clip manifests (from DHC)
│       ├── foul_type_llm_results_*.json # Prior LLM grading results (from DHC)
│       ├── model/                   # Step 5-7 model outputs (gitignored parquet)
│       │   ├── game_crew_dataset.parquet
│       │   ├── evaluation.parquet
│       │   ├── crew_interactions.parquet
│       │   ├── player/              # Player-level FTA/36 predictions
│       │   ├── l2m/                 # L2M validation tables
│       │   └── dhc_merge/           # Step 7 DHC merge outputs
│       └── player_official/         # Player x official interaction pipeline
│           ├── player_games/        # Per-player game logs (FTA, minutes)
│           ├── player_official_interactions.parquet
│           ├── defensive_adjusted_interactions.parquet
│           └── official_calling_profiles.parquet
├── src/
│   ├── nba_client.py                # NBA Stats API client (merged DHC methods)
│   ├── fetch_pbp.py                 # Download PBP JSON
│   ├── fetch_l2m.py                 # Scrape L2M reports
│   ├── fetch_crew_all.py            # Expand crew to all PBP games
│   ├── ingest.py                    # Parse PBP → structured foul records
│   ├── ref_profiles.py              # Layer 1 per-official profiles
│   ├── player_official_profiles.py  # Per-official x player FTA profiles
│   ├── defensive_adjustment.py      # Opponent-DEF_RATING adjustment
│   ├── official_calling_profiles.py # Per-official aggregate profiles (Step 4)
│   ├── crew_predictive_model.py     # Game-level SF prediction (Step 5)
│   ├── player_crew_predictive_model.py # Player-level FTA/36 prediction (Step 5b)
│   ├── l2m_validation.py            # L2M INC cross-check (Step 6)
│   ├── dhc_merge.py                 # does-harden-choke merge (Step 7)
│   ├── foul_type_scraper.py         # Video clip manifest builder (merged from DHC)
│   ├── foul_type_classifier.py      # HTML manual classification tool (merged from DHC)
│   └── foul_type_llm_grader.py      # Multimodal LLM grader (merged from DHC)
├── output/
│   ├── figures/
│   └── tables/
└── documents/
    └── development/
        ├── HANDOFF.md               # Current state + next steps (start here)
        └── HANDOFF-findings.md      # Detailed findings tables
```

## Decisions made

1. **Primary path: player FTA profiles, not video model.** Per-official x player FTA/36 deltas (defense-adjusted) are the core signal for Paper 1. Layer 3 video no-call detection is shelved; L2M INC is used for validation only.

2. **Crew fetch is complete.** `fetch_crew_all.py` expanded crew assignments to 13,464 games. All downstream profiles built on this full dataset.

3. **Defensive adjustment is applied but minor.** Opponent `defrtg` from does-harden-choke barely moves official x player FTA deltas (r=0.98 raw vs adjusted). Still apply in production outputs.

4. **Expand player set before building predictive model.** 16 target players gave borderline ANOVA (p~0.07). Expanded to 40 (FTA/36 >= 5.0, >= 400 games) — ANOVA now p=0.000003. Player definitions centralized in `config/target_players.py`.

5. **No-call attribution is game-level, not play-level.** PBP doesn't record which official was responsible for a non-called play. Sufficient for rate-based analysis.

6. **L2M INC as validation, not primary data.** Step 6 complete: Layer 1 `sf_per_game` correlates with L2M INC rate (r=-0.45, p<0.001). Player-derived `suppressor_score` does **not** validate against L2M (r=+0.02, p=0.86). Use L2M to validate volume metrics, not player x official suppressor scores.

7. **Predictive unit is player-game, not game-level SF.** Game-level SF count from crew assignment is barely predictable (R^2~0.01 honest holdout). Player-level FTA/36 improves modestly with crew features (R^2 0.13 temporal vs 0.12 baseline-only).

8. **Crew interaction effects exist.** 529 official pairs with >= 20 shared games; 53 significant at |z|>1.96 (2x expected). Additive crew model is insufficient for some pairs.

9. **Crew assignment is not the floor-game mechanism.** Step 7 complete. Floor-game FTA crashes (mean -2.889 FTA/36) are not crew-driven (p=0.764). But player-specific crew prediction correlates with actual FTA deviation (r=0.406, p<0.001). Crew is a continuous modulator, not a collapse trigger.

10. **Landing foul binary as the entry point for Layer 2.** The v3 five-axis taxonomy from does-harden-choke was top-down and too granular. The timing axis was killed by the Giannis counterexample. Start with one well-defined category (landing fouls) and test for inter-ref variance before expanding.

11. **DHC tooling merge complete.** Scraper, HTML classifier, and LLM grader merged from does-harden-choke. DHC is now a frozen research archive. Ref-ball is self-contained (local `analysis_table.csv`, no DHC path dependencies except PBP symlink).

12. **Dataset is the asset, papers are downstream.** Build once, query for Papers 1-3.

## Open questions

1. **Does official heterogeneity survive taxonomy conditioning?** cranky-scott-foster found context dominates error rates. CSF taxonomy import for L2M conditioning not yet done.

2. **Can the LLM reliably grade landing fouls?** The spatial signal (defender under shooter's landing zone) should be more gradable than the temporal boundary (BEFORE/DURING) that topped out at 71%. Target: 85%+ precision. Untested.

3. **Do landing foul calling rates vary significantly across officials?** This is the core Paper 2 hypothesis. If yes, it provides a mechanism for the suppressor/amplifier effects. If no, landing fouls may be too well-defined (every ref agrees) and a more ambiguous category is needed.

4. **What is the right sample design for per-official classification?** Player-diverse sampling (used in DHC) vs. official-diverse sampling (needed here). Need enough clips per official for statistical power (~100-150 per ref, 10-15 refs).

5. **Playoff assignment confound.** NBA assigns officials to playoff games strategically. RS->PO comparisons are descriptive, not causal. No individual-level "playoff whistle" found (rs_po_delta ~ 0).

6. **Release strategy.** Publish dataset (citations) vs. keep proprietary (advantage). Decide before Sloan submission.
