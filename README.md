# ref-ball

**A novel dataset of per-official NBA foul call and no-call data, built to enable research on referee heterogeneity and its effect on game outcomes.**

## The Strategy

The dataset is the asset. Papers are downstream products. Build the dataset once, then query it for multiple findings.

The dataset has three layers, each with a different novelty moat:

| Layer | What it is | Novelty moat | Status |
|---|---|---|---|
| **Layer 1: Per-official attribution** | Official name parsed from PBP `description` field | Weak (anyone can parse it) | **Scaled** — 11 seasons (RS+PO) pulled |
| **Layer 2: Foul-type classification** | Mechanism/severity/location/body part for called fouls | Medium (requires video classification at scale) | Deferred |
| **Layer 3: No-call detection** | Predicted missed fouls on non-called contact plays | Strong (requires video model + full-game video) | **Next** — L2M Ground Truth Collected |

**The build order is Layer 1 → Layer 3 → Layer 2.** This is deliberate, not sequential by difficulty. Layer 3 (no-call detection) is a binary problem (foul / no foul) that can be trained directly on NBA API labels — every event is already tagged. Layer 2 (foul-type classification) is a 12-class problem that requires manual annotation. Binary first, multiclass later.

## The Paper Sequence

Each paper builds on the dataset from the previous one. We do not need all three layers before publishing.

### Paper 1 — "Refs miss different fouls" (Layer 1 + Layer 3)

**Claim:** Individual NBA referees have systematically different no-call rates, and those rates shift in the playoffs.

- Per-official no-call rates computed at the game level (predicted missed fouls / predicted fouls)
- RS vs PO comparison per official
- Validation against L2M INC labels (league-audited ground truth)
- Does not require foul-type classification
- Does not require collapse data
- Sloan-possible if the variation is real and the shift pattern is surprising

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
1. Fetch PBP      →  src/fetch_pbp.py        →  data/raw/pbp/*.json
2. Fetch L2M      →  src/fetch_l2m.py        →  data/processed/l2m_events.parquet (and crew)
3. Ingest         →  src/ingest.py           →  data/processed/games/{game_id}.parquet
4. Profile        →  src/ref_profiles.py     →  data/processed/ref_profiles.parquet
5. No-call model  →  src/nocall_model.py     →  data/processed/nocalls.parquet
6. Analyze        →  src/analyze.py          →  output/figures/ + output/tables/
```

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

### 2. No-call model (Layer 3 — next to build)

Binary classifier trained on NBA API labels. Every PBP event is already labeled (S.FOUL, P.FOUL, or not-a-foul). The model learns what contact patterns produce a whistle vs. no whistle.

**Training data:** Video clips from the NBA video API (`videoeventsasset`), labeled by the PBP event type. Called fouls = positive class. Non-foul events (shots, turnovers, rebounds) = negative class.

**Validation:** L2M INC labels. The league's own auditors flag plays that should have been called. If the model flags plays the league also flagged, that's ground-truth validation. Precision/recall against L2M INC is the pre-registered validation standard.

**Inference:** Run the model across all non-foul events in a game. High-confidence "foul" predictions on non-foul events = predicted no-calls.

**Output:** `data/processed/nocalls.parquet` — one row per predicted no-call, with game_id, event_num, predicted foul probability, and the officials assigned to the game.

**Attribution:** No-calls are attributed at the game level, not the play level. We compute no-call *rates* per official across their game sample — "in Foster's games, predicted no-call rate was 15%; in Brothers' games, 8%." This sidesteps the problem that PBP doesn't record which official was responsible for a non-called play. Game-level crew assignment comes from the NBA API.

```bash
make train-nocall             # train the binary classifier
make predict-nocalls          # run inference across all games
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

## Relationship to does-harden-choke

ref-ball is a **sibling project** to does-harden-choke, not a replacement:

- **does-harden-choke** studies the *player-side* of the FTA shift: which types of fouls disappear, which players are affected, what the mechanism is
- **ref-ball** studies the *referee-side*: which officials call and miss which types of fouls, and whether that variation is a mechanism variable in the collapse

The projects share:
- NBA Stats API client (adapted from does-harden-choke)
- Foul-type taxonomy (for Paper 2, when Layer 2 is added)
- Collapse game definitions (for Paper 3)

ref-ball adds:
- Per-official attribution (parsed from PBP description field)
- No-call detection model (binary video classifier)
- Referee profile construction
- Referee-game-collapse interaction analysis

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
├── README.md                    # This spec doc
├── config.py                    # Paths, seasons, constants
├── requirements.txt
├── Makefile
├── data/
│   ├── raw/
│   │   └── pbp/                # PBP JSON files (symlinked from does-harden-choke)
│   ├── cache/                   # API response cache
│   └── processed/
│       ├── games/               # Per-game parquet files (ingest output)
│       ├── nocalls.parquet      # Predicted no-calls (model output)
│       ├── ref_profiles.parquet # Per-official profiles (profile output)
│       └── classifications.csv # Foul-type video classifications (Paper 2)
├── src/
│   ├── ingest.py                # Parse PBP JSON → structured records with official attribution
│   ├── nocall_model.py          # Train binary no-call classifier + run inference
│   ├── ref_profiles.py         # Build per-official profiles
│   ├── analyze.py               # Three-track analysis
│   └── nba_client.py            # NBA Stats API client (from does-harden-choke)
├── output/
│   ├── figures/
│   └── tables/
└── documents/
    └── development/
```

## Decisions made

1. **Build order: Layer 1 → Layer 3 → Layer 2.** Binary no-call detection first (trainable on API labels), multiclass foul-type classification later (requires manual annotation). Paper 1 ships without Layer 2.

2. **No-call attribution is game-level, not play-level.** PBP doesn't record which official was responsible for a non-called play. We compute no-call *rates* per official across their game sample, not per-play attribution. This is sufficient for the research questions and avoids the unsolved problem of official positioning inference.

3. **Validation against L2M INC labels.** The league's own auditors flag plays that should have been called. L2M INC is ground truth for no-call detection. Precision/recall against L2M INC is the pre-registered validation standard. *Update: 7 seasons of L2M reports have been successfully collected, providing 685 ground-truth INC shooting foul events.*

4. **Dataset is the asset, papers are downstream.** Build the dataset once, query it for multiple papers. The dataset enables Papers 1-3 sequentially; each paper doesn't require a new data collection effort.

5. **Foul-type taxonomy is deferred.** The 12-mechanism taxonomy from does-harden-choke is carried over but not needed for Paper 1. It's added in Paper 2 when Layer 2 is built.

6. **Crew Confound / Game Denominators.** PBP data only lists officials who blew a whistle. To calculate true rates (fouls per game worked), we need the full 3-person crew assignment. We scrape `__NEXT_DATA__` from `www.nba.com` game pages to generate `crew_assignments.parquet`, which provides the denominator.

## Open questions

1. **Video availability boundary.** The no-call model requires full-game video, not just clip-by-event video from the API. What's the earliest season where full-game video is available? This determines the left boundary of the dataset.

2. **Model architecture.** Clip-level classifier (is this clip a foul?) vs. frame-level model (where in this sequence is the contact?). Clip-level is simpler and sufficient for binary detection. Frame-level is needed only if we want to localize contact for the no-call detector. Decide before training.

3. **Sample boundary.** All games or a stratified sample? Full-game video processing is expensive. A sample covering 60 officials × 5 games × 11 seasons = 3,300 games is substantial and may be sufficient for Paper 1.

4. **Release strategy.** Publish the dataset (lose moat, gain citations) or keep it proprietary (keep advantage, limit impact)? Sloan papers typically require sharing methodology; some require sharing data. Decide before submitting.
