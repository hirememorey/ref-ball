# Development Handoff (June 28, 2026)

Operational snapshot for a new developer or LLM picking up this codebase. For project goals, literature positioning, and long-term paper sequence, see the root [README.md](../../README.md).

---

## What This Project Is Trying To Do

**Primary aim (current):** Build predictive profiles of how each NBA referee (and crew) calls shooting fouls — especially for high-FTA, manufactured-contact players — and use those profiles to predict the officiating environment of a game from crew assignment.

**Not the primary aim:** Descriptive L2M error-rate analysis. That question is largely covered by the sibling project [cranky-scott-foster](../../../cranky-scott-foster). ref-ball's value-add is the *official-conditional, predictive* question.

**Downstream consumer:** [does-harden-choke](../../../does-harden-choke) — test whether playoff FTA shifts are crew-mediated (certain officials suppress/amplify FTA for target players).

---

## Current Data Inventory

| Asset | Path | Count | Status |
|---|---|---|---|
| Raw PBP JSON | `data/raw/pbp/` (symlink → does-harden-choke) | 13,278 games | **Complete** |
| Ingested foul parquets | `data/processed/games/{game_id}.parquet` | 13,278 | **Complete** |
| Crew assignments | `data/processed/crew_assignments.parquet` | 13,464 games, 40,804 rows | **Complete** (5 fetch failures) |
| L2M events | `data/processed/l2m_events.parquet` | 56,219 events, 2,717 games | **Complete** |
| L2M reports | `data/processed/l2m_reports.parquet` | 2,717 | **Complete** |
| Layer 1 ref profiles | `data/processed/ref_profiles.parquet` | 101 officials | **Current** |
| Player game logs | `data/processed/player_official/player_games/*.parquet` | 40 players | **Complete** |
| Player×official interactions | `data/processed/player_official/player_official_interactions.parquet` | 3,846 pairs; 2,819 with ≥10 games both sides | **Current** |
| Defense-adjusted interactions | `data/processed/player_official/defensive_adjusted_interactions.parquet` | 1,678 pairs; 1,431 qualified | **Current** |
| Official calling profiles | `data/processed/player_official/official_calling_profiles.parquet` | 88 officials | **Current** |

### External dependencies (sibling projects)

| Project | What ref-ball uses | Path |
|---|---|---|
| does-harden-choke | Raw PBP symlink; `analysis_table.csv` for `opponent_defrtg` | `../does-harden-choke/data/processed/analysis_table.csv` |
| cranky-scott-foster | L2M taxonomy, crew features, structural-risk findings (reference only for now) | `../cranky-scott-foster/` |

---

## What Has Been Built (Code)

| Script | Purpose | CLI |
|---|---|---|
| `src/fetch_pbp.py` | Download PBP JSON | `make fetch-pbp` |
| `src/fetch_l2m.py` | Scrape L2M reports + crew for L2M games | `make fetch-l2m` |
| `src/fetch_crew_all.py` | Expand crew to all PBP game IDs | `python src/fetch_crew_all.py --resume` |
| `src/ingest.py` | Parse PBP → per-foul parquet with official attribution | `make ingest` |
| `src/ref_profiles.py` | Layer 1 per-official called-foul profiles | `make profile` |
| `src/player_official_profiles.py` | Per-official × player FTA interaction profiles | `fetch` / `build` / `summary` |
| `src/defensive_adjustment.py` | Opponent-DEF_RATING-adjusted FTA deltas | `build` / `summary` |
| `src/official_calling_profiles.py` | Per-official aggregate calling profiles (Step 4) | `build` / `summary` |
| `src/feasibility_study.py` | Video clip classifier feasibility (shelved) | Not executed |
| `src/nocall_model.py` | Layer 3 video model (stub) | Not implemented |
| `src/analyze.py` | Three-track analysis (stub) | Not implemented |

All commands require `PYTHONPATH=.` from the project root (or use `make` targets).

```bash
cd /Users/harrisgordon/Documents/Development/ref-ball
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

### Target player configuration

Player definitions are centralized in `config/target_players.py`:

- `CORE_PLAYERS`: 16 original high-FTA players (all IDs verified via `commonplayerinfo`)
- `EXPANDED_PLAYERS`: 24 additional players meeting FTA/36 ≥ 5.0 and ≥ 400 career games (2014-15 onward)
- `ALL_TARGET_PLAYERS`: merged dict (40 total)

Both `player_official_profiles.py` and `defensive_adjustment.py` import from this module rather than duplicating.

Selection rule for expansion:
```
Career RS FTA/36 >= 5.0  AND  >= 400 career games  (2014-15 onward)
```

Near-miss players not included (insufficient FTA/36 or GP): Jalen Brunson (4.79), LaMarcus Aldridge (4.76), Zach LaVine (4.74), Kyrie Irving (4.39), Chris Paul (3.57). Zion Williamson (9.06 FTA/36 but only 214 GP), Ja Morant (7.25 but 307 GP), Victor Wembanyama (5.51 but 117 GP) — have the rate but not the sample.

---

## Key Findings (40 Players, Full Crew — Current)

> **Detailed findings with tables are in [HANDOFF-findings.md](HANDOFF-findings.md).** Summary below.

- **Signal confirmed.** Between-official ANOVA: F=1.93, p=0.000003. Which official is on your game matters for FTA.
- **Effect size.** 80th-percentile spread: 0.86 FTA/36 ≈ 0.8 FTA/game for a 34-min starter. Typical suppressor/amplifier: ±0.5 FTA/36.
- **Suppressors are consistent.** Phenizee Ransom suppresses 84% of players, Aaron Smith 80%, Brandon Adair 80%. Not one-player effects — official-level traits.
- **Amplifier paradox.** Top amplifiers (Spooner, McCutchen) have lower overall SF rates (r=−0.29). Player×official interaction is separate from overall foul-calling volume.
- **Defensive adjustment minor.** Raw vs adjusted r=0.969. Opponent quality is not a confound.
- **No individual-level playoff whistle.** 40/82 officials with RS/PO splits show 50/50 direction. Mean rs_po_delta ≈ 0. FTA playoff drop may be crew-composition-driven, not individual-behavior-driven.
- **Layer 1 cross-validation.** suppressor_score vs sf_pct_of_fouls: r=+0.30. Moderate alignment; player-level metric captures signal beyond overall foul rates.

---

## Recommended Next Steps (Priority Order)

Steps 1–4 are **complete**. Steps 5–7 remain.

### Step 5: Predictive model — crew → game FTA environment

**Input:** 3-official crew assignment + season type (RS/PO) + optional matchup context

**Output:**
- Expected SF/game for the crew
- Expected FTA/36 for each target player in the game
- Crew-level suppressor/amplifier index

**Validation:** Season holdout (train seasons 2014–2022, predict 2023–24). Same structure as cranky-scott-foster's rolling holdout.

**Suggested first model:** Regularized linear model or gradient boosting on per-official historical profiles from `official_calling_profiles.parquet`. Keep it simple until signal is confirmed.

**Key design question:** Is the effect additive (3 individual profiles sum to a crew profile) or interactive (some officials amplify each other)? Start with additive and test residuals for interaction effects.

**Implementation:**
1. Create `src/crew_predictive_model.py`
2. For each game with crew data, look up the 3 officials' historical profiles
3. Train: predict actual game SF rate or player-level FTA from crew features
4. Evaluate: RMSE, correlation, calibration — compare to baseline (league-average SF rate)
5. Holdout: train on 2014–2022, test on 2023–24

---

### Step 6: L2M validation cross-check

Use L2M INC shooting fouls as ground truth: do officials with high suppressor scores on full-game FTA also have higher INC SF rates in L2M clutch games?

**Data:** `l2m_events.parquet` + `crew_assignments.parquet`

**Not yet implemented.** Join and compute per-official `INC_SF / (INC_SF + CC_SF)` in L2M games. Correlate with suppressor scores.

**Implementation:**
1. Load `l2m_events.parquet`, filter to shooting fouls
2. Join with `crew_assignments.parquet` on `game_id`
3. Per official: count INC vs CC shooting fouls in L2M games
4. Compute `inc_sf_rate = INC_SF / (INC_SF + CC_SF)`
5. Correlate with `suppressor_score` and `mean_adj_fta36_delta` from `official_calling_profiles.parquet`

---

### Step 7: Connect to does-harden-choke (Paper 3 mechanism)

Merge crew suppressor scores with does-harden-choke collapse game data. Test whether playoff FTA shifts concentrate in games with high-suppression officials.

**Hypothesis:** The league-wide playoff FTA drop is partly crew-mediated — suppressor-heavy crews may be assigned to playoff games at higher rates.

**Status:** RS/PO analysis shows no individual-official "playoff whistle" effect. The remaining hypothesis is crew-composition. Test with crew-level suppressor indices from Step 5.

---

## How to Rebuild All Outputs From Scratch

```bash
cd /Users/harrisgordon/Documents/Development/ref-ball

# Layer 1 ref profiles
PYTHONPATH=. .venv/bin/python src/ref_profiles.py

# Player×official interactions (fetch + build)
PYTHONPATH=. .venv/bin/python src/player_official_profiles.py fetch    # ~35 min for 40 players
PYTHONPATH=. .venv/bin/python src/player_official_profiles.py build

# Defensive-adjusted deltas
PYTHONPATH=. .venv/bin/python src/defensive_adjustment.py build

# Official calling profiles (aggregate)
PYTHONPATH=. .venv/bin/python src/official_calling_profiles.py build
PYTHONPATH=. .venv/bin/python src/official_calling_profiles.py summary
```

Or via Makefile:
```bash
make profile                          # Layer 1 ref profiles
make profile-calling                  # Official calling profiles
make profile-calling-summary          # Print top suppressors/amplifiers
```

---

## Shelved / Deferred Work

| Item | Why shelved |
|---|---|
| `src/nocall_model.py` (Layer 3 video) | User chose predictive FTA path over video classification |
| `src/feasibility_study.py` | Written but never executed; video deps not installed |
| `src/analyze.py` (Tracks A/B/C) | Superseded by player_official_profiles + official_calling_profiles pipeline; rewrite after Step 5 |
| Layer 2 foul-type classification | Deferred to Paper 2; player-level FTA is the proxy for now |
| Import cranky-scott-foster taxonomy | Useful for Step 6 conditioning, not blocking Step 5 |

---

## Known Issues

1. **5 crew fetch failures** — check `crew_fetch.log` for game IDs; re-run `fetch_crew_all.py --resume` to retry
2. **PBP symlink** — `data/raw/pbp` points to does-harden-choke; don't duplicate PBP data
3. **Official name matching** — PBP uses abbreviated names (e.g. `R.Garretson`); crew uses full names (e.g. `Rodney Garretson`). `player_official_profiles.py` has a mapping step; verified working with full crew
4. ~~**Duplicate TARGET_PLAYERS**~~ — **Fixed.** Centralized in `config/target_players.py`
5. ~~**Hardcoded DHC path**~~ — **Fixed.** `defensive_adjustment.py` now uses `config.PROJECT_ROOT.parent / "does-harden-choke"`
6. ~~**Makefile gaps**~~ — **Partially fixed.** Added `profile-calling` and `profile-calling-summary` targets. Still missing targets for `fetch_crew_all`, `player_official_profiles`, and `defensive_adjustment`
7. **`config/` package shadows `config.py`** — `config/__init__.py` re-exports root `config.py` module attributes. If adding new root-level config, must update `config/__init__.py`
8. **rs_po_delta coverage** — Only 40/82 officials with ≥3 players have RS/PO splits. PO sample sizes are thin (median 22.5 games per player). May need to relax thresholds or aggregate differently for Step 5.

---

## Quick Verification Commands

```bash
cd /Users/harrisgordon/Documents/Development/ref-ball

# Data counts
PYTHONPATH=. .venv/bin/python -c "
import pandas as pd
from pathlib import Path
crew = pd.read_parquet('data/processed/crew_assignments.parquet')
games = len(list(Path('data/processed/games').glob('*.parquet')))
ix = pd.read_parquet('data/processed/player_official/player_official_interactions.parquet')
print(f'PBP games: {games}')
print(f'Crew games: {crew.game_id.nunique()}')
print(f'Interaction pairs: {len(ix)}')
q = ((ix.n_games_with_official>=10)&(ix.n_games_without_official>=10)).sum()
print(f'Pairs >=10 games: {q}')
"

# Check does-harden-choke dependency
test -f ../does-harden-choke/data/processed/analysis_table.csv && echo "DHC analysis_table: OK" || echo "DHC analysis_table: MISSING"

# Official calling profiles
PYTHONPATH=. .venv/bin/python src/official_calling_profiles.py summary
```

---

## Environment

- Python 3.13, venv at `.venv/`
- Installed: pandas, pyarrow, numpy, requests, tqdm, scipy
- **Not installed:** torch, sklearn, opencv (only needed if video feasibility study is revived or for Step 5 gradient boosting)
- NBA API: use `NBAStatsClient` in `src/nba_client.py` (same pattern as does-harden-choke — rate limits are endpoint-specific, not global)
