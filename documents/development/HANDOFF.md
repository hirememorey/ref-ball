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
| `src/crew_predictive_model.py` | Game-level SF prediction from crew (Step 5) | `build` / `summary` / `diagnose` / `interactions` |
| `src/player_crew_predictive_model.py` | Player-level FTA/36 prediction from crew (Step 5b) | `build` / `summary` / `diagnose` |
| `src/l2m_validation.py` | L2M INC cross-check vs suppressor metrics (Step 6) | `build` / `summary` |
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
- **Layer 1 cross-validation (internal).** suppressor_score vs sf_pct_of_fouls: r=+0.30. Moderate alignment; player-level metric captures signal beyond overall foul rates.
- **Step 5 — game-level prediction weak.** Honest temporal holdout: OLS R²≈0.005 for game SF count; league average is competitive. Crew-pair interaction effects are real (53/529 pairs |z|>1.96, 2× expected).
- **Step 5b — player-level prediction modest.** Temporal holdout: R²=0.13 (crew + baseline) vs 0.12 (baseline only). Static/leaky upper bound R²≈0.31. Westbrook, CP3, Harden benefit most.
- **Step 6 — L2M validation mixed.** `suppressor_score` vs L2M INC/(INC+CC): r=+0.02, p=0.86 (not confirmed). `sf_per_game` vs L2M INC rate: r=−0.45, p<0.001 (Layer 1 validated). Player-conditioned L2M test also not significant.

---

## Recommended Next Steps (Priority Order)

Steps 1–6 are **complete**. Step 7 remains.

### Step 5: Predictive model — crew → game FTA environment — **COMPLETE**

**Scripts:** `src/crew_predictive_model.py`, `src/player_crew_predictive_model.py`

**Game-level (SF count):**
- Train 2014–22, test 2023–24 + 2024–25
- Best honest model: OLS additive, RMSE=4.53 vs baseline 4.56, R²≈0.005
- Conclusion: game-level SF volume is mostly context-driven; crew features add little

**Player-level (FTA/36):**
- 11,493 player-games (temporal profiles), 2,675 test
- Best honest model: baseline + crew mean adj delta, RMSE=3.96, R²=0.13 vs baseline R²=0.12
- 12/20 target players improve with crew info; Westbrook, CP3, Harden largest lift

**Crew interactions:**
- Additive residuals tested on all modeling-season games (not just test holdout)
- 529 pairs with ≥20 shared games; 53 significant (expected 26.5)

```bash
make model-crew                  # static profiles + train
make model-crew-temporal         # honest prior-season profiles
make model-crew-diagnose
make model-player-crew           # player FTA/36 (temporal)
make model-player-crew-diagnose
```

**Outputs:** `data/processed/model/` (game), `data/processed/model/player/` (player)

---

### Step 6: L2M validation cross-check — **COMPLETE**

**Script:** `src/l2m_validation.py`

Joins L2M shooting-foul events → crew assignments → `official_calling_profiles` and `defensive_adjusted_interactions`.

**Official-level results (n=79 qualified):**
| Metric | vs L2M INC/(INC+CC) | r | p |
|---|---|---|---|
| `suppressor_score` | primary test | +0.02 | 0.86 |
| `mean_adj_fta36_delta` | | −0.02 | 0.90 |
| `sf_per_game` (Layer 1) | | **−0.45** | **<0.001** |
| `sf_pct_of_fouls` (Layer 1) | | **−0.42** | **<0.001** |

**Player-conditioned (4,348 target-player L2M events, 1,129 adjudicated):**
- Crew mean adj Δ vs INC: r=−0.03, p=0.37 (not significant)

**Conclusion:** Layer 1 volume metrics validate against L2M. Player-derived suppressor score does **not** — cannot claim L2M ground-truth validation for the core player×official metric. Reframe suppressor scores as full-game FTA tools validated by predictive holdout, not L2M.

```bash
make l2m-validate
make l2m-validate-summary
```

**Outputs:** `data/processed/model/l2m/`

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

# Step 5: predictive models
make model-crew-temporal
make model-player-crew

# Step 6: L2M validation
make l2m-validate
```

Or via Makefile:
```bash
make profile                          # Layer 1 ref profiles
make profile-calling                  # Official calling profiles
make profile-calling-summary          # Print top suppressors/amplifiers
make model-crew-temporal                # Game-level SF model (honest holdout)
make model-player-crew                  # Player-level FTA/36 model
make l2m-validate                       # L2M cross-check
```

---

## Shelved / Deferred Work

| Item | Why shelved |
|---|---|
| `src/nocall_model.py` (Layer 3 video) | User chose predictive FTA path over video classification |
| `src/feasibility_study.py` | Written but never executed; video deps not installed |
| `src/analyze.py` (Tracks A/B/C) | Superseded by Steps 5–6 pipeline; rewrite optional |
| Layer 2 foul-type classification | Deferred to Paper 2; player-level FTA is the proxy for now |
| Import cranky-scott-foster taxonomy | Useful for conditioned L2M re-test; not blocking Step 7 |

---

## Known Issues

1. **5 crew fetch failures** — check `crew_fetch.log` for game IDs; re-run `fetch_crew_all.py --resume` to retry
2. **PBP symlink** — `data/raw/pbp` points to does-harden-choke; don't duplicate PBP data
3. **Official name matching** — PBP uses abbreviated names (e.g. `R.Garretson`); crew uses full names (e.g. `Rodney Garretson`). `player_official_profiles.py` has a mapping step; verified working with full crew
4. ~~**Duplicate TARGET_PLAYERS**~~ — **Fixed.** Centralized in `config/target_players.py`
5. ~~**Hardcoded DHC path**~~ — **Fixed.** `defensive_adjustment.py` now uses `config.PROJECT_ROOT.parent / "does-harden-choke"`
6. ~~**Makefile gaps**~~ — **Fixed.** Added targets for `model-crew`, `model-player-crew`, `l2m-validate`, `profile-calling`
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

# Step 5–6 summaries
make model-crew-diagnose
make model-player-crew-diagnose
make l2m-validate-summary
```

---

## Environment

- Python 3.13, venv at `.venv/`
- Installed: pandas, pyarrow, numpy, requests, tqdm, scipy
- **Not installed:** torch, sklearn, opencv (only needed if video feasibility study is revived or for Step 5 gradient boosting)
- NBA API: use `NBAStatsClient` in `src/nba_client.py` (same pattern as does-harden-choke — rate limits are endpoint-specific, not global)
