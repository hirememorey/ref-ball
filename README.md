# ref-ball

**Per-official NBA shooting-foul profiles and a predictive model of how referee crews shape the free-throw environment — especially for high-FTA, manufactured-contact players.**

> **Picking up development?** See [documents/development/HANDOFF.md](documents/development/HANDOFF.md) for current data inventory, findings, and the exact next steps to run.

## The Strategy

**Primary aim:** Model how each individual referee (and crew) calls shooting fouls, then predict the officiating profile of a game from crew assignment. The downstream question is whether playoff FTA shifts (from [does-harden-choke](../does-harden-choke)) are crew-mediated.

The dataset is the asset. Papers are downstream products. Build the dataset once, then query it for multiple findings.

The dataset has three layers, each with a different novelty moat:

| Layer | What it is | Novelty moat | Status |
|---|---|---|---|
| **Layer 1: Per-official attribution** | Official name parsed from PBP `description` field | Weak (anyone can parse it) | **Complete** — 13,278 games ingested, 13,464 with crew |
| **Player×official profiles** | FTA/36 deltas per player under each official, defense-adjusted | Medium (requires crew + game logs) | **Built (40 players)** — full crew, 3,846 pairs, ANOVA p=0.000003 |
| **Layer 2: Foul-type classification** | Mechanism/severity/location/body part for called fouls | Medium (requires video classification at scale) | Deferred — player-level FTA is the proxy for now |
| **Layer 3: No-call detection** | Predicted missed fouls on non-called contact plays | Strong (requires video model + full-game video) | **Shelved** — L2M INC available for validation; video path not pursued |

**Current build order:** Layer 1 + crew → player×official FTA profiles → official calling profiles → predictive crew model → L2M validation → does-harden-choke merge. Layer 3 (video no-call model) and Layer 2 (foul-type classification) are deferred. Steps 1–4 are complete; Step 5 (predictive model) is next.

## The Paper Sequence

Each paper builds on the dataset from the previous one. We do not need all three layers before publishing.

### Paper 1 — "Refs call different games" (Layer 1 + player×official profiles)

**Claim:** Individual NBA referees have systematically different effects on shooting-foul and FTA rates — especially for high-FTA players — and those effects are predictable from crew assignment.

- Per-official called SF rates (full game, Layer 1)
- Per-official × player FTA deltas, opponent-defense-adjusted
- RS vs PO comparison per official
- Validation against L2M INC shooting fouls (league-audited ground truth)
- Does not require foul-type classification or a video model

### Paper 2 — "Refs miss different *types* of fouls" (Layer 1 + Layer 2 + Layer 3)

**Claim:** The *types* of fouls officials miss are official-specific, and the type-specific miss rates predict which players lose free throws in the playoffs.

- Adds foul-type classification (mechanism, severity, location, body part) to both called fouls and predicted no-calls
- Connects to does-harden-choke's manufactured vs. genuine framework
- Player-official interaction: does Harden's manufactured-contact FTA survive under Ref X but not Ref Y?

### Paper 3 — "Referee assignment is a mechanism variable in team collapse" (all layers + collapse data)

**Claim:** Referee no-call profiles predict the FTA shift that does-harden-choke showed correlates with team collapse.

- Merges with does-harden-choke collapse game data
- Requires an identification strategy for the referee selection problem (NBA assigns refs strategically)
- This is the Sloan paper if the causal claim holds

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
1. Fetch PBP           →  src/fetch_pbp.py                 →  data/raw/pbp/*.json
2. Fetch L2M           →  src/fetch_l2m.py                 →  data/processed/l2m_events.parquet
3. Fetch crew (all)    →  src/fetch_crew_all.py            →  data/processed/crew_assignments.parquet
4. Ingest              →  src/ingest.py                    →  data/processed/games/{game_id}.parquet
5. Layer 1 profiles    →  src/ref_profiles.py              →  data/processed/ref_profiles.parquet
6. Player game logs    →  src/player_official_profiles.py  →  data/processed/player_official/player_games/
7. Player×official     →  src/player_official_profiles.py  →  player_official_interactions.parquet
8. Defense adjustment  →  src/defensive_adjustment.py      →  defensive_adjusted_interactions.parquet
9. Official profiles   →  src/official_calling_profiles.py →  official_calling_profiles.parquet
10. Predictive model   →  (not yet built)                  →  crew predictions
11. Analyze            →  src/analyze.py                   →  output/figures/ + output/tables/
```

Steps 1–4 and L2M fetch are **complete**. Steps 5–8 have been **rebuilt on full crew** with 40 target players. Step 9 (official calling profiles) is built. Step 10 (predictive model) is the next code to write.

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

### 2b. No-call model (Layer 3 — shelved)

Binary video classifier (`src/nocall_model.py`) — stub only. Shelved in favor of player FTA profiles + L2M INC validation. `src/feasibility_study.py` exists but was never executed.

```bash
make train-nocall             # not implemented
make predict-nocalls          # not implemented
```

### 3. Profile

Build per-official profiles from called fouls (Layer 1) and predicted no-calls (Layer 3).

**Profile dimensions (Paper 1 — without Layer 2):**

| Axis | Metric | Source |
|---|---|---|
| **Called foul volume** | Shooting fouls / 100 possessions | Layer 1 (PBP) |
| **No-call volume** | Predicted no-calls / 100 possessions | Layer 3 (model) |
| **No-call rate** | Predicted no-calls / (predicted no-calls + called fouls) | Layer 1 + Layer 3 |
| **Context** | % called in leading vs. trailing situations | Layer 1 (PBP) |
| **Season type** | RS vs. PO profile delta | Layer 1 + Layer 3 |

**Profile dimensions (Paper 2 — with Layer 2 added):**

| Axis | Metric | Source |
|---|---|---|
| **Called foul composition** | % manufactured vs. % genuine | Layer 2 (classification) |
| **No-call composition** | % manufactured vs. % genuine of predicted no-calls | Layer 2 + Layer 3 |
| **Severity** | % MARGINAL / MEDIUM / STRONG | Layer 2 |
| **Location** | % RA / PAINT / MID / PERIM | Layer 2 |
| **Body Part** | % ARM / CHEST / SHOULDER / LOWER | Layer 2 |

```bash
make profile                   # build all profiles
make profile --official 2041  # single official
```

### 4. Analyze

```bash
make analyze              # all tracks
make analyze --track A    # descriptive only
make analyze --track B    # mechanism only
make analyze --track C    # causal only
```

#### Track A: Descriptive — "The Referee Landscape" (Paper 1)

- Distribution of no-call rates across officials
- Which officials are outliers (high/low no-call rate)?
- How much variance in no-call rates is between-official vs. within-official?
- Comparison with Owen Phillips' The F5 database (where overlapping)

#### Track B: Mechanism — "The Choke Referee" (Paper 2)

- Do certain referees' no-call profiles correlate with team collapse dynamics?
- Merge with does-harden-choke collapse game data
- Is there a "choke-amplifying" referee profile (high no-call rate on manufactured contact)?
- Player-referee interaction: do certain players' FTA shift under specific officials?

#### Track C: Causal — "The Playoff Whistle" (Paper 1 + Paper 3)

- Do individual referees change their no-call rates in the playoffs?
- Is the RS→PO shift official-specific or league-wide?
- Does the playoff whistle effect concentrate in specific officials?
- Does referee no-call profile predict team collapse? (Paper 3 — requires identification strategy)

## Relationship to sibling projects

### does-harden-choke

ref-ball is a **sibling project**, not a replacement:

- **does-harden-choke** studies the *player-side* of the FTA shift: which types of fouls disappear, which players are affected, what the mechanism is
- **ref-ball** studies the *referee-side*: which officials suppress/amplify FTA for high-FTA players, and whether crew assignment mediates the playoff FTA shift

**Shared assets:**
- NBA Stats API client (`src/nba_client.py` adapted from DHC)
- Raw PBP data (`data/raw/pbp` symlinks to DHC)
- `analysis_table.csv` — provides `opponent_defrtg` for defensive adjustment
- Foul-type taxonomy (Paper 2)
- Collapse game definitions (Paper 3)

**ref-ball adds:**
- Full crew assignments for all 13K+ games (`fetch_crew_all.py`)
- Per-official × player FTA interaction profiles
- Predictive crew → game officiating model (in progress)

### cranky-scott-foster

Another sibling project on the same L2M data. Key distinction:

| | cranky-scott-foster | ref-ball |
|---|---|---|
| Question | What *situations* produce errors? | How will *this official* call shooting fouls? |
| Unit of analysis | Decision context (taxonomy bins) | Official identity / crew assignment |
| Finding | Signal is primarily structural context, not referee identity | Official×player FTA heterogeneity is significant (p=0.000003 on full data) |
| Reuse | Taxonomy, crew features, experience tiers — import for conditioning, don't rebuild | — |

ref-ball's claim must be tested *conditional on decision context* to avoid conflating assignment composition with competence. See HANDOFF Step 6.

## Foul-type taxonomy (deferred to Paper 2)

Carried over from does-harden-choke. Not needed for Paper 1.

| Axis | Values |
|---|---|
| **Mechanism** | DRV-FINISH, DRV-INIT, ARM-HOOK, CONTEST, LANDING, PUMP-JUMP, RIP-THRU, POST, PUTBACK, OFFBALL, TAKE, AMB |
| **Body Part** | HEAD, ARM, CHEST, SHOULDER, LOWER |
| **Timing** | BEFORE, DURING, AFTER (drive mechanisms only) |
| **Severity** | STRONG, MEDIUM, MARGINAL |
| **Location** | RA, PAINT, MID, PERIM |

Manufactured vs. genuine:
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
│   ├── raw/
│   │   └── pbp/                     # PBP JSON (symlink → does-harden-choke)
│   └── processed/
│       ├── games/                   # Per-game foul parquets (13,278)
│       ├── crew_assignments.parquet # Full 3-person crew per game (13,464 games)
│       ├── l2m_events.parquet       # L2M ground truth (56K events)
│       ├── ref_profiles.parquet     # Layer 1 per-official called-foul profiles
│       └── player_official/         # Player×official interaction pipeline
│           ├── player_games/        # Per-player game logs (FTA, minutes)
│           ├── player_official_interactions.parquet
│           ├── defensive_adjusted_interactions.parquet
│           └── official_calling_profiles.parquet
├── src/
│   ├── fetch_pbp.py                 # Download PBP JSON
│   ├── fetch_l2m.py                 # Scrape L2M reports
│   ├── fetch_crew_all.py            # Expand crew to all PBP games
│   ├── ingest.py                    # Parse PBP → structured foul records
│   ├── ref_profiles.py              # Layer 1 per-official profiles
│   ├── player_official_profiles.py  # Per-official × player FTA profiles
│   ├── defensive_adjustment.py      # Opponent-DEF_RATING adjustment
│   ├── official_calling_profiles.py  # Per-official aggregate profiles (Step 4)
│   ├── feasibility_study.py         # Video feasibility (shelved)
│   ├── nocall_model.py              # Layer 3 stub
│   ├── analyze.py                   # Three-track analysis stub
│   └── nba_client.py                # NBA Stats API client
├── output/
│   ├── figures/
│   └── tables/
└── documents/
    └── development/
        └── HANDOFF.md               # Current state + next steps (start here)
```

## Decisions made

1. **Primary path: player FTA profiles, not video model.** Per-official × player FTA/36 deltas (defense-adjusted) are the core signal. Layer 3 video no-call detection is shelved; L2M INC is used for validation only.

2. **Crew fetch is complete.** `fetch_crew_all.py` expanded crew assignments to 13,464 games. All downstream profiles must be rebuilt on this full dataset.

3. **Defensive adjustment is applied but minor.** Opponent `defrtg` from does-harden-choke barely moves official×player FTA deltas (r=0.98 raw vs adjusted). Still apply in production outputs.

4. **Expand player set before building predictive model.** 16 target players gave borderline ANOVA (p≈0.07). Expanded to 40 (FTA/36 ≥ 5.0, ≥ 400 games) — ANOVA now p=0.000003. Player definitions centralized in `config/target_players.py`.

5. **No-call attribution is game-level, not play-level.** PBP doesn't record which official was responsible for a non-called play. Sufficient for rate-based analysis.

6. **L2M INC as validation, not primary data.** 685 INC shooting fouls validate whether suppressor/amplifier officials also miss more calls in clutch situations.

7. **Foul-type taxonomy deferred.** Player-level FTA is the proxy for manufactured-contact tendency until Layer 2 is built.

8. **Dataset is the asset, papers are downstream.** Build once, query for Papers 1–3.

## Open questions

1. **Does official heterogeneity survive taxonomy conditioning?** cranky-scott-foster found context dominates error rates. ref-ball must test whether official×player FTA effects persist within decision categories (HANDOFF Step 6).

2. **Predictive model architecture.** Start with regularized linear / gradient boosting on per-official historical profiles. Season holdout validation. Key question: additive (3 profiles sum) vs interactive (crew effects).

3. **Playoff assignment confound.** NBA assigns officials to playoff games strategically. RS→PO comparisons are descriptive, not causal. No individual-level "playoff whistle" found (rs_po_delta ≈ 0) — FTA drop may be crew-composition-driven.

4. **Release strategy.** Publish dataset (citations) vs. keep proprietary (advantage). Decide before Sloan submission.

5. **Crew vs individual decomposition.** Is the suppressor/amplifier effect additive across a 3-person crew, or are there interaction effects? Test in Step 5.
