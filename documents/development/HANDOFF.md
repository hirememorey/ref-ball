# Development Handoff (June 29, 2026)

Operational snapshot for a new developer or LLM picking up this codebase. For project goals, literature positioning, and long-term paper sequence, see the root [README.md](../../README.md).

---

## What This Project Is Trying To Do

**Primary aim (current):** Understand how individual NBA referees interpret specific types of contact differently. The aggregate question (do refs call different games?) is answered — ANOVA p=0.000003. The next question is *why*: do refs differ in how they interpret specific contact types?

**Active frontier:** **Step 10 — landing foul LLM grader.** Manual ground truth is complete (99/100 clips classified, merged with 35 legacy v3 labels → 134-row ground truth). Build a spatial landing-foul prompt (adapt `foul_type_llm_grader.py`), validate at 85%+ precision / 70%+ recall on binary YES/NO, then scale to per-official measurement (Steps 11–12).

**Completed work:** Per-official x player FTA profiles, predictive crew models (Steps 5-7), L2M validation, and does-harden-choke merge. See "Key Findings" below and [HANDOFF-findings.md](HANDOFF-findings.md) for details.

**Not the primary aim:** Descriptive L2M error-rate analysis (covered by [cranky-scott-foster](../../../cranky-scott-foster)). Game-level SF prediction (R^2~0.005, too weak to be useful).

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
| v3 foul-type ground truth | `data/foul_type_classifications.csv` | 36 clips (Harden + Giannis) | **Complete** — 1 LANDING, 35 non-LANDING |
| Landing foul manifest | `data/processed/landing_foul_manifest.json` | 100 clips (reproducible) | **Built** — 3-FT shooting fouls, 53 officials, 2019–25 |
| Landing foul classifier | `output/landing_foul_classifier.html` | 100 clips embedded | **Built** — regenerate via `make landing-classifier` |
| Landing foul classifications | `data/landing_foul_classifications.csv` | 99 clips (48 YES, 45 NO, 6 UNCLEAR) | **Complete** — exported 2026-06-29; 1 manifest clip unlabeled |
| Merged ground truth | `data/landing_foul_ground_truth.csv` | 134 rows (49 YES, 79 NO, 6 UNCLEAR) | **Complete** — `make landing-merge` (99 classifier + 35 v3, 1 overlap) |

### External dependencies (sibling projects)

| Project | What ref-ball uses | Path | Status |
|---|---|---|---|
| does-harden-choke | Raw PBP symlink (data only) | `../does-harden-choke/data/raw/pbp/` | **Frozen** — active tooling merged into ref-ball; DHC is research archive only |
| cranky-scott-foster | L2M taxonomy, crew features, structural-risk findings (reference only) | `../cranky-scott-foster/` | No change |

**Self-contained data:** `analysis_table.csv` is now in `data/processed/` (copied from DHC during Step 8 merge). The only remaining DHC dependency is the PBP symlink (`data/raw/pbp/` → DHC).

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
| `src/dhc_merge.py` | does-harden-choke merge — crew vs FTA collapse (Step 7) | `build` / `summary` |
| `src/foul_type_scraper.py` | Video clip manifest builder by player (**merged from DHC**) | `--player` / `--season` / `--games` |
| `src/foul_type_classifier.py` | HTML v3 five-axis classifier (**merged from DHC**) | `--manifest` |
| `src/foul_type_llm_grader.py` | Multimodal LLM grader — timing axis (**merged from DHC**) | `--player` / `--provider` / `--validate-only` |
| `src/landing_foul_manifest.py` | **Step 9:** scan PBP for 3-FT shooting fouls, sample, fetch video | `make landing-manifest` |
| `src/landing_foul_classifier.py` | **Step 9:** binary YES/NO/UNCLEAR HTML classifier | `make landing-classifier` |
| `src/landing_foul_merge.py` | Merge landing export + v3 labels into ground truth CSV | `make landing-merge` |

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

Steps 1-9 are **complete** (one manifest clip still unlabeled — see Step 9 notes). **Step 10 (LLM grader) is the immediate next task.** Steps 11-12 follow after validation.

### Step 8: Merge does-harden-choke tooling — COMPLETE

Migrated active foul-type tooling from does-harden-choke into ref-ball. DHC is now a frozen research archive.

1. Copied `foul_type_scraper.py`, `foul_type_classifier.py`, `foul_type_llm_grader.py` into `src/`
2. Added `player_slug()` and `ALL_PLAYERS` alias (→ `ALL_TARGET_PLAYERS`) to `config.py` + `config/__init__.py`
3. Copied `foul_type_classifications.csv` → `data/`, manifests + LLM results → `data/processed/`
4. Copied `analysis_table.csv` into `data/processed/` — ref-ball is now self-contained (no DHC path dependency)
5. Updated paths in `defensive_adjustment.py` and `dhc_merge.py` to reference local `analysis_table.csv`
6. Merged missing `nba_client.py` methods from DHC: `get_common_player_info()`, `get_league_game_finder()`, `get_league_team_stats()`
7. Fixed `foul_type_llm_grader.py` ground truth path: `config.DATA_DIR` instead of `config.PROJECT_ROOT`
8. Deleted stubs: `feasibility_study.py`, `nocall_model.py`, `analyze.py`
9. Added freeze note to DHC README
10. Verified all imports and data paths

### Step 9: Landing foul ground truth — COMPLETE (2026-06-29)

**Sampling strategy (enrichment, not population representativeness):** Scan local PBP JSON for shooting fouls followed by exactly 3 free throw attempts (= 3-point shooting foul). This enriches for perimeter closeout fouls where landing space violations are most common. Sample 100 clips across seasons 2019–25 with per-game caps and season diversity. Document enrichment when interpreting base rates — this is a validation set for the LLM grader, not a prevalence estimate.

**Built:**
1. `src/landing_foul_manifest.py` — scans 13,278 games, finds ~1,789 candidates (2019+), samples 100, fetches video URLs (~3 min)
2. `src/landing_foul_classifier.py` — binary YES / NO / UNCLEAR + note; keyboard shortcuts Y/N/U
3. `src/landing_foul_merge.py` — merges browser export with 36 v3 labels (LANDING → YES, else NO)

**Manual classification results** (`data/landing_foul_classifications.csv`):

| Label | Count |
|---|---|
| YES | 48 |
| NO | 45 |
| UNCLEAR | 6 |
| **Total** | **99 / 100** |

- **42 clips** have free-text notes (borderline cases, contest vs landing, etc.) — useful for few-shot examples and mismatch review.
- **1 unlabeled clip:** `0022000114` / event `603` (Ball S.FOUL on S. Gilgeous-Alexander, J. Capers). Optional: re-open classifier and label before Step 10.
- **48% YES rate** in the enrichment sample is higher than the ~15–20% design target — good for LLM validation (more positive examples) but **not** a league-wide landing-foul prevalence estimate.

**Merged ground truth** (`make landing-merge` → `data/landing_foul_ground_truth.csv`):

| Source | Rows | Notes |
|---|---|---|
| `landing_classifier` | 99 | Step 9 export |
| `v3_foul_type` | 35 | 36 v3 clips minus 1 overlap with Step 9 |
| **Merged total** | **134** | YES=49, NO=79, UNCLEAR=6 |

Regenerate merged file any time:
```bash
cd /Users/harrisgordon/Documents/Development/ref-ball
make landing-merge
```

**Rubric:** Landing foul = defender's feet/body under or moving into shooter's landing zone while shooter is airborne on a jump shot, and the foul is called because of that positioning. Standard arm/hand contest on the shot = NO.

**Re-run classifier (if manifest changes or to label the missing clip):**
```bash
make landing-classifier
python -m http.server 8080 --directory output
# → http://localhost:8080/landing_foul_classifier.html
# Export CSV → data/landing_foul_classifications.csv → make landing-merge
```

### Step 10: Landing foul LLM grader — NEXT (unblocked)

**Goal:** Binary YES/NO landing foul classification from video. Adapt the pattern in `foul_type_llm_grader.py` (timing axis) with a **spatial** prompt.

**Build checklist:**
1. Create `src/landing_foul_llm_grader.py` (or extend grader with `--task landing`) — load clips from `data/processed/landing_foul_manifest.json`, ground truth from `data/landing_foul_ground_truth.csv`
2. Design spatial-observation prompt: shot type (jump shot vs drive), defender feet/body at shooter descent, contact moment relative to landing zone
3. `--validate-only` mode: grade only clips with ground truth, print confusion matrix + mismatches (mirror timing grader validation output)
4. Use Gemini native video upload first (`gemini-2.5-flash` or similar)
5. Iterate prompt; event-ordering fallback if needed (DEFENDER_CLOSEOUT → SHOOTER_DESCENDING → CONTACT)

**Evaluation protocol (binary metrics):**
- **Primary set:** YES + NO rows only — **93 clips** from Step 9 export (exclude 6 UNCLEAR unless re-labeled)
- **Extended set:** full merged ground truth (134 rows) including v3 legacy clips
- **Targets:** precision ≥ 85% on YES, recall ≥ 70% on YES
- **Inspect mismatches** on borderline YES clips with notes (e.g. Thybulle/Shamet, Williamson/Hield, Roddy/Durant)

**Example invocation (once built):**
```bash
PYTHONPATH=. .venv/bin/python src/landing_foul_llm_grader.py \
  --provider gemini --model gemini-2.5-flash --validate-only
```

### Step 11: Scale to per-official measurement — PENDING

1. Select 10-15 officials spanning suppressor/amplifier spectrum
2. Sample ~100-150 shooting foul clips per official
3. Run LLM grader on all clips
4. Compute per-official landing foul calling rate

### Step 12: Variance analysis — PENDING

1. ANOVA on per-official landing foul rates — do rates differ significantly?
2. Correlation with existing suppressor/amplifier profiles — does landing foul tolerance explain the effect?
3. If significant: expand to additional contact types. If not: landing fouls are too well-defined and a more ambiguous category is needed.

---

### Completed steps (summary)

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

### Step 7: Connect to does-harden-choke (Paper 3 mechanism) — **COMPLETE**

**Script:** `src/dhc_merge.py`

**Data:** DHC `analysis_table.csv` (2014-15+) × `crew_assignments.parquet` × `official_calling_profiles.parquet` × `defensive_adjusted_interactions.parquet`

**Coverage:** 16,697 player-games with crew data (16,154 RS + 543 PO across 28 players, 9,872 games).

**Three analyses:**

**A. RS vs PO crew composition (is crew more suppressive in playoffs?)**
- RS games n=9,657, crew_mean_suppressor_score mean=0.482
- PO games n=215, crew_mean_suppressor_score mean=0.479
- Mann-Whitney p=0.720 — **not significant**
- Confirms individual-level RS/PO finding: no systematic "playoff whistle" at the crew level either.

**B. Floor game crew composition (do floor games have more suppressive crews?)**
- Floor PO player-games n=63, crew_mean_suppressor_score mean=0.476
- Non-floor PO player-games n=480, mean=0.478
- Mann-Whitney p=0.764 — **not significant**
- BUT: actual FTA/36 delta is dramatically different (floor mean −2.889 vs non-floor +0.408, p<0.001)
- **Key result: DHC floor games are characterized by large FTA drops, but crew composition is not the mechanism. The crashes happen regardless of who is officiating.**

**C. Player-specific predicted crew suppression vs actual FTA delta**
- For each playoff player-game, computed predicted suppression = mean player×official adj delta across the 3 crew officials
- Spearman r=+0.406, p<0.001 (n=433 player-games, 20 players)
- Correct expected direction: positive (amplifying crew → positive predicted → positive actual delta)
- Consistent across 18 of 18 players with ≥5 games; 8/18 individually significant
- **Key result: Player-specific crew prediction explains meaningful variance in individual playoff FTA. But this is a continuous prediction, not a floor-game trigger.**
- Methodological caveat: adj deltas include PO games (median PO fraction ≈6%). A clean RS-only holdout would require recomputing adj deltas excluding PO games.

**Overall conclusion for Paper 3 framing:**
Crew assignment is **not** the mediating variable for playoff FTA collapse. The DHC floor-game FTA crashes (mean −2.889 FTA/36) are not explained by crew composition (p=0.764). However, crew-based predictions do explain variance in individual FTA outcomes (r=0.406) — it's a continuous effect, not a threshold/collapse driver. The mechanism behind floor games lies elsewhere (defensive pressure, player fatigue, psychological, etc.).

```bash
make dhc-merge          # build
make dhc-merge-summary  # print results
```

**Outputs:** `data/processed/model/dhc_merge/`

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
| ~~`src/nocall_model.py` (Layer 3 video)~~ | **Deleted (Step 8).** Stub, never implemented |
| ~~`src/feasibility_study.py`~~ | **Deleted (Step 8).** Never executed, has bugs, superseded by LLM grader |
| ~~`src/analyze.py` (Tracks A/B/C)~~ | **Deleted (Step 8).** Stub, superseded by Steps 5-7 pipeline |
| v3 five-axis foul taxonomy | Timing axis killed by Giannis counterexample; 12 mechanisms too granular for per-official stats. Replaced by landing foul binary as entry point |
| Import cranky-scott-foster taxonomy | Useful for conditioned L2M re-test; not blocking current work |
| Full-game manufactured/genuine classification | Descriptively valid but predictive chain untested. Revisit after landing foul variance results |

---

## Known Issues

1. **5 crew fetch failures** — check `crew_fetch.log` for game IDs; re-run `fetch_crew_all.py --resume` to retry
2. **PBP symlink** — `data/raw/pbp` points to does-harden-choke; don't duplicate PBP data
3. **Official name matching** — PBP uses abbreviated names (e.g. `R.Garretson`); crew uses full names (e.g. `Rodney Garretson`). `player_official_profiles.py` has a mapping step; verified working with full crew
4. ~~**Duplicate TARGET_PLAYERS**~~ — **Fixed.** Centralized in `config/target_players.py`
5. ~~**Hardcoded DHC path**~~ — **Fixed (Step 8).** `defensive_adjustment.py` and `dhc_merge.py` now reference local `data/processed/analysis_table.csv`
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

# Verify local analysis_table (no DHC dependency)
test -f data/processed/analysis_table.csv && echo "analysis_table: OK" || echo "analysis_table: MISSING"

# Official calling profiles
PYTHONPATH=. .venv/bin/python src/official_calling_profiles.py summary

# Step 5–6 summaries
make model-crew-diagnose
make model-player-crew-diagnose
make l2m-validate-summary

# Step 9 landing foul pipeline
test -f data/processed/landing_foul_manifest.json && python3 -c "import json; m=json.load(open('data/processed/landing_foul_manifest.json')); print(f'landing manifest: {m.get(\"num_clips\")} clips, {m.get(\"num_candidates\")} candidates')"
test -f output/landing_foul_classifier.html && echo "landing classifier HTML: OK"
test -f data/landing_foul_classifications.csv && python3 -c "import pandas as pd; df=pd.read_csv('data/landing_foul_classifications.csv'); print(f'classifications: {len(df)} rows'); print(df['landing_foul'].value_counts().to_string())"
make landing-merge && python3 -c "import pandas as pd; df=pd.read_csv('data/landing_foul_ground_truth.csv'); print(f'ground truth: {len(df)} rows'); print(df['landing_foul'].value_counts().to_string())"
```

---

## Environment

- Python 3.13, venv at `.venv/`
- Installed: pandas, pyarrow, numpy, requests, tqdm, scipy
- **Needed for Layer 2:** google-generativeai (Gemini API), openai, anthropic — for `foul_type_llm_grader.py`
- **Not installed:** torch, sklearn, opencv (not needed for current path)
- NBA API: use `NBAStatsClient` in `src/nba_client.py` (same pattern as does-harden-choke — rate limits are endpoint-specific, not global)
