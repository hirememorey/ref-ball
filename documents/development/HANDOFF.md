# Development Handoff (July 1, 2026)

Operational snapshot for a new developer or LLM picking up this codebase. For project goals, literature positioning, and long-term paper sequence, see the root [README.md](../../README.md).

---

## What This Project Is Trying To Do

**Primary aim (current):** Understand how individual NBA referees interpret specific types of contact differently. The aggregate question (do refs call different games?) is answered — ANOVA p=0.000003. The next question is *why*: do refs differ in how they interpret specific contact types?

**Active frontier — two parallel tracks:**

1. **SSAC27 Submission (Paper 1).** Abstract draft complete (~460 words, v2). Two publication-quality figures generated (Table 1: suppressor/amplifier profiles with named officials; Figure 1: crew prediction vs actual FTA deviation scatter, r=0.406). **Abstract deadline: October 1, 2026.** Full paper due December 4, 2026 if selected. Open-source decision resolved: publishing all data, named officials, no anonymization. See `documents/ssac27-abstract-draft.md` and `output/figures/`.

2. **Step 10b — fine-tuned video classifier (Paper 2).** Manual ground truth **complete** (300/300 clips). LLM grader **exhausted** as sole classifier (~55% precision, 98% recall). **Frozen VideoMAE:** zero signal. **Colab fine-tune runs (2026-07-01):**
   - Run 1 (full clip): degenerate constant predictor (51% P, 100% R).
   - Run 2 (real clips, full window): ultra-conservative (80% P, 14% R).
   - Run 3 (anchors ±0.15): **69% P, 76% R** — best recall; gate MARGINAL.
   - Run 4 (anchors ±0.10, `yes_weight=0.7`): **81% P, 59% R** — best precision; 4 FPs (down from 10); recall fails gate.
   - LLM describe → rules (57-val): 50% P, 93% R.
   - **START HERE:** Colab **Run 5** — blend Run 3/4: `yes_weight=0.85`, shorter finetune. See [Step 10b](#step-10b-fine-tuned-video-classifier--colab-run-5).

**Completed work:** Per-official x player FTA profiles, predictive crew models (Steps 1-7), L2M validation, does-harden-choke merge, SSAC27 abstract draft + figures. See "Key Findings" below and [HANDOFF-findings.md](HANDOFF-findings.md) for details.

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
| Landing foul manifest | `data/processed/landing_foul_manifest.json` | 300 clips (seed=42) | **Built** — 3-FT shooting fouls, ~1,789 candidates, 2019–25 |
| Landing foul classifier | `output/landing_foul_classifier.html` | 300 clips embedded | **Built** — `make landing-classifier`; Import CSV + Next Ungraded buttons |
| Landing foul classifications | `data/landing_foul_classifications.csv` | 300 clips (143 YES, 141 NO, 16 UNCLEAR) | **Complete** — exported 2026-06-30 |
| Merged ground truth | `data/landing_foul_ground_truth.csv` | 335 rows (144 YES, 175 NO, 16 UNCLEAR) | **Complete** — `make landing-merge` (300 classifier + 35 v3) |
| Landing foul LLM results | `data/processed/landing_foul_llm_results_vertex_gemini-3_5-flash.json` | 93 clips (spatial validation) | **First run** — 2026-06-29; gitignored (regenerate via validate command) |
| Downloaded video clips | `data/clips/landing_foul/{game_id}_{event_id}.mp4` | 284 clips (YES/NO only) | **Complete** — 960x540, ~8-12s each, gitignored |
| Frozen VideoMAE embeddings | `data/processed/landing_foul_embeddings.npz` | 284 clips × 768-dim | **Complete** — CLS token from `videomae-base-finetuned-kinetics`; zero signal for landing fouls |
| Train/val split | `data/processed/landing_foul_split.json` | 227 train / 57 val | **Complete** — stratified 80/20, seed=42, YES/NO only |
| Fine-tune checkpoint | `data/processed/landing_foul_video_best.pt` | Run 4 (2026-07-01) | **81% P / 59% R** — best precision; gitignored |
| Fine-tune metrics | `data/processed/landing_foul_video_metrics.json` | Run 4 (2026-07-01) | PROMISING — 4 FPs, 12 FNs; gitignored |
| Run 3 checkpoint (reference) | Drive / local backup | Run 3 (2026-07-01) | **69% P / 76% R** — keep for ensemble |
| Clip anchors | `data/processed/landing_foul_clip_anchors.json` | **284 / 284** | **Complete** — committed; foul_frac per clip |
| LLM describe val | `data/processed/landing_foul_llm_results_describe_val57.json` | 57-val | 50% P / 93% R — gitignored |
| Frame cache | `data/processed/landing_foul_frames.npz` | Colab | **Rebuild** when `anchor_half_width` changes; gitignored |
| SSAC27 abstract draft | `documents/ssac27-abstract-draft.md` | ~460 words (v2) | **Draft** — deadline Oct 1, 2026 |
| SSAC27 Table 1 | `output/figures/table_a_suppressor_amplifier.png` | Top 5 suppressors + amplifiers | **Generated** — named officials, SF/game |
| SSAC27 Figure 1 | `output/figures/figure_b_crew_prediction_scatter.png` | r=0.406, 433 player-games | **Generated** — crew prediction vs actual FTA deviation |

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
| `src/landing_foul_llm_grader.py` | **Step 10:** landing foul LLM grader (spatial binary YES/NO) | `make landing-grade` / `landing-grade-validate` |
| `src/landing_foul_video_dataset.py` | **Step 10b:** download clips + extract frozen VideoMAE embeddings | `make video-download` / `make video-extract` |
| `src/landing_foul_video_split.py` | **Step 10b:** stratified train/val split (80/20, seed=42) | `make video-split` |
| `src/landing_foul_video_train.py` | **Step 10b:** train classifier head on embeddings + evaluate | `make video-train` / `make video-cv` |
| `src/landing_foul_video_finetune.py` | **Step 10c:** end-to-end VideoMAE fine-tuning | `make video-finetune` / `make video-finetune-evaluate` |
| `src/landing_foul_annotate_anchors.py` | **Step 10d:** browser UI for per-clip contact temporal anchors | `make video-annotate` |
| `src/generate_abstract_figures.py` | **SSAC27:** Table 1 (suppressor/amplifier profiles) + Figure 1 (crew prediction scatter) | `PYTHONPATH=. .venv/bin/python src/generate_abstract_figures.py` |

All commands require `PYTHONPATH=.` from the project root (or use `make` targets).

```bash
cd /Users/harrisgordon/Documents/Development/ref-ball
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# For Step 10b video classifier (torch, transformers, opencv, scikit-learn):
.venv/bin/pip install -r requirements-ml.txt
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

Steps 1-9 are **complete**. Step 9 manual grading **complete** (300/300). Step 10 LLM grader **exhausted** — pivot to fine-tuned video classifier (Step 10b). **SSAC27 abstract drafted** — submission deadline October 1, 2026.

### SSAC27 Abstract Submission — IN PROGRESS (deadline Oct 1, 2026)

**What's done:**
- Abstract drafted (v2, ~460 words): `documents/ssac27-abstract-draft.md`
- Table 1 generated: `output/figures/table_a_suppressor_amplifier.png` — top 5 suppressors (Phenizee Ransom 84%, Brandon Adair 80%, Aaron Smith 80%, Kevin Scott 80%, Eric Dalen 75%) and top 5 amplifiers (Bill Spooner 11%, Monty McCutchen 20%, Mark Lindsay 25%, Eric Lewis 25%, Matt Boland 25%) with SF/game showing the amplifier paradox
- Figure 1 generated: `output/figures/figure_b_crew_prediction_scatter.png` — predicted crew suppression vs actual FTA/36 deviation (Spearman r=0.406, p<0.001, n=433 player-games, 20 players)
- Open-source decision resolved: full transparency, all data published, named officials
- Figure generator: `src/generate_abstract_figures.py`

**What's remaining before submission:**
1. Final wording pass on abstract (iterate v2 → v3)
2. Prepare GitHub repo for open-source link (required at submission) — clean README, reproducible pipeline, publish parquets
3. Submit via https://bit.ly/4xIaYy9 before October 1, 2026
4. If abstract accepted (notification late-October): full manuscript due December 4, 2026

**Regenerate figures:**
```bash
PYTHONPATH=. .venv/bin/python src/generate_abstract_figures.py
```

---

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

### Step 9: Landing foul ground truth — COMPLETE (2026-06-30)

**Sampling strategy (enrichment, not population representativeness):** Scan local PBP JSON for shooting fouls followed by exactly 3 free throw attempts (= 3-point shooting foul). This enriches for perimeter closeout fouls where landing space violations are most common. Sample 300 clips across seasons 2019–25 with per-game caps and season diversity (`seed=42`). Document enrichment when interpreting base rates — this is a validation/training set, not a prevalence estimate.

**Built:**
1. `src/landing_foul_manifest.py` — scans 13,278 games, finds ~1,789 candidates (2019+), samples 300, fetches video URLs
2. `src/landing_foul_classifier.py` — binary YES / NO / UNCLEAR + note; keyboard shortcuts Y/N/U; **Import CSV** + **Next Ungraded** navigation (2026-06-30)
3. `src/landing_foul_merge.py` — merges browser export with 36 v3 labels (LANDING → YES, else NO)

**Manual classification results** (`data/landing_foul_classifications.csv`):

| Label | Count |
|---|---|
| YES | 143 |
| NO | 141 |
| UNCLEAR | 16 |
| **Total** | **300 / 300** |

- **~48% YES rate** in the enrichment sample — good for classifier training (balanced-ish) but **not** a league-wide landing-foul prevalence estimate.
- Many clips have free-text notes (borderline cases, contest vs landing) — useful for error analysis and few-shot examples.

**Merged ground truth** (`make landing-merge` → `data/landing_foul_ground_truth.csv`):

| Source | Rows | Notes |
|---|---|---|
| `landing_classifier` | 300 | Step 9 export (2026-06-30) |
| `v3_foul_type` | 35 | 36 v3 clips minus 1 overlap with Step 9 |
| **Merged total** | **335** | YES=144, NO=175, UNCLEAR=16 |

Regenerate merged file any time:
```bash
cd /Users/harrisgordon/Documents/Development/ref-ball
make landing-merge
```

**Rubric:** Landing foul = defender's feet/body under or moving into shooter's landing zone while shooter is airborne on a jump shot, and the foul is called because of that positioning. Standard arm/hand contest on the shot = NO.

**Re-run classifier (if manifest changes):**
```bash
make landing-classifier
python -m http.server 8080 --directory output
# → http://localhost:8080/landing_foul_classifier.html
# Import CSV to restore labels; Next Ungraded → to find gaps
# Export CSV → data/landing_foul_classifications.csv → make landing-merge
```

---

### Step 10: Landing foul LLM grader — EXHAUSTED (reference only)

**Goal:** Binary YES/NO landing foul classification from video. Precision target: ≥ 85% on YES. Recall target: ≥ 70% on YES.

**Verdict:** LLM prompting cannot clear the precision bar. Recall passes on every run (≥ 98%); precision stalls at ~55%. **Do not invest further in prompt iteration.** Use LLM only as an optional pre-filter in the hybrid pipeline (see below) if the fine-tuned model also struggles.

**Status:** `src/landing_foul_llm_grader.py` is built. Four prompt modes tested: `spatial`, `spatial_v2`, `sequence`, `whistle`. Provider: Vertex `gemini-3.5-flash` (recommended). Makefile targets: `landing-grade`, `landing-grade-validate`.

| Prompt mode | Accuracy | Precision (YES) | Recall (YES) | Verdict |
|---|---|---|---|---|
| Spatial V1 | 58% | 55% | 98% | YES-biased — 38 contest FPs |
| Spatial V2 (+who_initiated) | ~58% | ~55% | lower | Traded FPs for FNs |
| Whistle | ~58% | ~55% | similar | Audio unreliable |
| Sequence | ~54% | ~53% | 100% | Worse than spatial |

**Why it fails:** Multimodal LLMs process loosely-connected frames, not continuous physics. They see closeout contact on a jump shot and say YES. They cannot reliably track defender feet relative to landing zone over ~200–400ms windows.

**Hybrid fallback (if fine-tuned model precision < 75%):** Run LLM grader as pre-filter (98% recall), manually review predicted-YES clips only (~50% workload reduction). See hybrid section in Step 10b decision tree.

**Regenerate LLM validation (reference):**
```bash
make landing-grade-validate PROVIDER=vertex MODEL=gemini-3.5-flash
make landing-grade-validate PROVIDER=vertex MODEL=gemini-3.5-flash PROMPT_MODE=sequence
```

Full run details, provider setup, and confusion matrices remain in the sections below for reference.

---

### Step 10b: Fine-tuned video classifier — Colab Run 5

**Goal:** Train a supervised video classifier on the 284 labeled manifest clips (YES/NO only, UNCLEAR excluded). Same quality gate as the LLM: **precision ≥ 85% on YES**, **recall ≥ 70% on YES** on a held-out validation set. Do **not** proceed to Steps 11–12 until the gate clears.

#### Run history (57-clip val split, seed=42)

| Run | Date | Setup | P (YES) | R (YES) | Confusion (tp/fp/fn/tn) | Best epoch | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | 2026-06-30 | Full clip `0.0,1.0`, placeholders | 0.51 | 1.00 | 29/28/0/0 | 2 head | BELOW_BASELINE |
| 2 | 2026-07-01 | Real clips, full window | 0.80 | 0.14 | 4/1/25/27 | 1 head | PROMISING — precision trap |
| 3 | 2026-07-01 | Anchors ±0.15, `yes_weight=1.0` | 0.688 | 0.759 | 22/10/7/18 | 7 finetune | MARGINAL — best recall |
| 4 | 2026-07-01 | Anchors ±0.10, `yes_weight=0.7` | **0.810** | 0.586 | 17/4/12/24 | 5 head | PROMISING — best precision |
| — | 2026-07-01 | LLM describe → rules | 0.500 | 0.931 | 27/27/2/1 | — | Rules too permissive |
| **5** | **next** | **±0.10, `yes_weight=0.85`, finetune=5** | ? | ? | — | — | **START HERE** |

**Run 4 takeaways:** FP count dropped 10 → 4 (Teague, Murray, Sochan, Osman still slip through). Recall dropped below gate (76% → 59%) because `yes_weight=0.7` was too aggressive. Best checkpoint was **head epoch 5** — finetune epochs degraded val metrics (overfit). No threshold on Run 4 checkpoint clears both gates (t=0.45 → 64% P / 72% R).

**Run 5 hypothesis:** Keep Run 4's tight crop (±0.10) and low FP count, but raise `yes_weight` to 0.85 and shorten finetune to limit overfit. Target: ~75% P / ~70% R — the Pareto middle between Runs 3 and 4.

#### START HERE: Colab Run 5

Open [`documents/development/colab-finetune.ipynb`](colab-finetune.ipynb) (GPU runtime). Defaults in §5:

| Parameter | Run 3 | Run 4 | **Run 5** |
|---|---|---|---|
| `anchor_half_width` | 0.15 | 0.10 | **0.10** (same cache as Run 4) |
| `yes_weight` | 1.0 | 0.7 | **0.85** |
| `finetune_epochs` | 15 | 15 | **5** |
| `head_epochs` | 5 | 5 | 5 |
| `finetune_lr` | 2e-5 | 2e-5 | 2e-5 |
| Other | — | — | dropout=0.4, patience=6, seed=42 |

**Workflow:**

1. §2 Clone repo (anchors + split come with clone).
2. §4 Upload `landing_foul_clips.zip` from Drive (`make video-package` locally).
3. §5 Confirm Run 5 hyperparameters (notebook defaults updated).
4. **§5b Frame cache** — **skip rebuild if you still have Run 4's cache** (`anchor_half_width=0.10`). Rebuild only if cache is missing or you changed half_width.
5. §6 Fine-tune → §7 Save checkpoint + metrics to Drive.

Local equivalent:

```bash
# Cache only if missing (Run 4 cache at half_width=0.10 is reusable):
PYTHONPATH=. python src/landing_foul_video_finetune.py --build-cache --anchor-half-width 0.10

make video-finetune ANCHOR_HALF_WIDTH=0.10 YES_WEIGHT=0.85 FINETUNE_EPOCHS=5
```

**If Run 5 still misses the gate:**

| Next lever | Rationale |
|---|---|
| `yes_weight=0.9` | Nudge recall up if Run 5 precision ≥80% but recall <70% |
| `phase=head` only | Run 4's best was head epoch 5; skip finetune entirely |
| Ensemble Run 3 + Run 4 | Union for recall, intersection for precision; manual review on disagreement |
| LLM Layer 2 on describe JSON | Text classifier on top of Gemini observations |

#### What's been built and tested

**Infrastructure (complete):**

| Script | Purpose | CLI |
|---|---|---|
| `src/landing_foul_video_dataset.py` | Download clips + extract frozen VideoMAE embeddings | `make video-download` / `make video-extract` |
| `src/landing_foul_video_split.py` | Stratified 80/20 train/val split | `make video-split` |
| `src/landing_foul_video_train.py` | Train classifier head on embeddings + evaluate | `make video-train` / `make video-train-mlp` / `make video-cv` |
| `src/landing_foul_video_finetune.py` | End-to-end VideoMAE fine-tuning (two-phase, frame cache) | `make video-finetune` / `make video-finetune-evaluate` |
| `src/landing_foul_annotate_anchors.py` | Browser UI for per-clip contact temporal anchors | `make video-annotate` |
| `documents/development/colab-finetune.ipynb` | Colab runbook (GPU, frame cache, training) | Open in Google Colab |
| `requirements-ml.txt` | ML dependencies (torch, transformers, opencv, scikit-learn) | `.venv/bin/pip install -r requirements-ml.txt` |

**Frozen baseline results (2026-06-30) — zero signal:**

| Model | YES Precision | YES Recall | Behavior |
|---|---|---|---|
| Logistic Regression (single split) | 0.000 | 0.000 | All-NO (degenerate) |
| MLP (single split) | 0.509 | 1.000 | All-YES (degenerate) |
| Logistic Regression (5-fold CV) | 0.000 | 0.000 | All-NO (degenerate) |

**Colab fine-tune results (2026-06-30) — also degenerate:**

| Metric | Value | Notes |
|---|---|---|
| Best epoch | 2 (head phase) | Early-stopped at epoch 11 |
| YES Precision | 0.509 | = class prior (29 YES / 57 val) |
| YES Recall | 1.000 | Predicts YES on every val clip |
| Confusion | tp=29, fp=28, fn=0, tn=0 | No discrimination |
| `prob_yes` | 0.587 on all clips | Constant output — head collapsed to bias |
| Threshold sweep | YES at t≤0.55, NO at t≥0.60 | No usable operating point |
| Gate verdict | **BELOW_BASELINE** | precision_pass=false, recall_pass=true |

**Why both runs failed:** VideoMAE-Kinetics encodes "playing basketball" — landing fouls and standard contests look identical at that abstraction. The Colab run compounded this by sampling 16 frames across the **entire 8–12s clip** (`--temporal-window 0.0,1.0`), diluting the ~400ms contact window to one frame every ~625ms. The model never saw concentrated contact signal and collapsed to predicting the class prior.

**What this tells us:** End-to-end fine-tuning alone is insufficient without **temporal cropping to the contact moment**. Per-clip anchors are required before retraining.

#### Dataset summary

| Asset | Path | Count | Status |
|---|---|---|---|
| Labels (primary) | `data/landing_foul_classifications.csv` | 300 rows (143 YES, 141 NO, 16 UNCLEAR) | **Complete** |
| Merged ground truth | `data/landing_foul_ground_truth.csv` | 335 rows (144 YES, 175 NO, 16 UNCLEAR) | **Complete** |
| Downloaded clips | `data/clips/landing_foul/{game_id}_{event_id}.mp4` | 284 (YES/NO only) | **Complete** — 960x540, gitignored |
| Frozen embeddings | `data/processed/landing_foul_embeddings.npz` | 284 × 768-dim | **Complete** — zero signal, reference only |
| Train/val split | `data/processed/landing_foul_split.json` | 227 train (114 YES, 113 NO) / 57 val (29 YES, 28 NO) | **Complete** — seed=42 |
| Clip anchors | `data/processed/landing_foul_clip_anchors.json` | 284 / 284 | **Complete** — foul_frac per clip; half_width overridable via CLI |
| Video URLs + metadata | `data/processed/landing_foul_manifest.json` | 300 clips | **Complete** — NBA CDN URLs verified live 2026-06-30 |

**Label handling:**
- **Primary training set:** 284 clips (YES=143, NO=141). UNCLEAR excluded.
- **Optional augmentation:** 35 v3 legacy clips (Harden/Giannis) are in `landing_foul_ground_truth.csv` but were **not downloaded** (no manifest video URLs). They would require separate video fetching via `foul_type_scraper.py` if needed.
- **Split is fixed:** `data/processed/landing_foul_split.json` — do not regenerate unless you have a specific reason.

#### Anchor annotation (complete 2026-07-01)

All 284 clips annotated via `make video-annotate`. Output committed to repo. Annotator stores `foul_frac` + default `half_width=0.15`; **Run 4 overrides to 0.10** without re-annotating:

```bash
--anchor-half-width 0.10
```

**After changing half_width:**

1. Rebuild frame cache: `--build-cache --anchor-half-width 0.10`
2. Re-run fine-tuning with same flag
3. Check `landing_foul_video_metrics.json` gate — target P≥0.85, R≥0.70.

#### Fine-tuning reference (already built — re-run after anchors)

`src/landing_foul_video_finetune.py` implements two-phase training:

1. **Load pre-trained VideoMAE** (`MCG-NJU/videomae-base-finetuned-kinetics`), replace head with 2-class linear layer.
2. **Phase 1 (5 epochs):** Freeze backbone, train head only.
3. **Phase 2 (15 epochs):** Unfreeze top 4 transformer layers at lower LR.
4. **Regularization:** Dropout 0.4, weight decay 0.01, early stopping on val YES precision, temporal jitter + color jitter (no horizontal flip).
5. **Frame cache:** Decode-once NPZ (`landing_foul_frames.npz`) — ~200× faster epochs vs live decode.
6. **Output:** `landing_foul_video_best.pt`, `landing_foul_video_metrics.json`.

Colab runbook: `documents/development/colab-finetune.ipynb`.

**Preprocessing levers to try if fine-tuning alone is insufficient:**

| Lever | What it does | When to try |
|---|---|---|
| **Per-clip anchors (manual)** | Mark contact frame; `resolve_window` crops to ±half_width | **Complete** — 284/284 |
| **Anchor half-width override** | `--anchor-half-width 0.10` tightens crop without re-annotating | **Runs 4–5** — 4 FPs vs 10 at 0.15 |
| **YES class weight** | `--yes-weight` trades precision vs recall | Run 3: 1.0 · Run 4: 0.7 · **Run 5: 0.85** |
| **Shorter finetune** | `--finetune-epochs 5` — Run 4 best was head-only | **Run 5** — limit overfit |
| **Temporal cropping** | Trim clips to the 2-3s around the foul before frame sampling | Automatic once anchors exist |
| **Spatial cropping** | Crop to the relevant court region. Feet are tiny in 960x540 wide-angle broadcast. If the foul location is roughly known (perimeter vs paint), crop before resizing to 224x224. | Second lever if temporal crop doesn't help |
| **Frame count** | Increase from 16 to 32 frames for higher temporal resolution. Trade-off: more memory, more overfitting risk. | Try alongside temporal cropping |
| **Different backbone** | SlowFast (dual-pathway: slow for spatial, fast for motion) or X3D (lightweight). | If VideoMAE fine-tuning hits <70% after all preprocessing levers |
| **Pose estimation** | MediaPipe/OpenPose to extract skeleton keypoints, then classify on foot/body positions directly. Bypasses the "feet are tiny in the frame" problem entirely. | Nuclear option if all video models fail |

**Makefile targets:**
```bash
make video-download              # download 284 clips (~10 min)
make video-extract               # extract frozen embeddings (~12 min, MPS)
make video-split                 # stratified 80/20 split
make video-train                 # logistic regression on frozen embeddings (baseline — degenerate)
make video-train-mlp             # MLP on frozen embeddings (baseline — degenerate)
make video-cv FOLDS=5            # k-fold CV on frozen embeddings (baseline — degenerate)
make video-pipeline              # full frozen baseline pipeline
make video-annotate              # browser UI for per-clip contact anchors (complete)
make video-finetune              # two-phase VideoMAE fine-tuning
make video-finetune ANCHOR_HALF_WIDTH=0.10 YES_WEIGHT=0.85 FINETUNE_EPOCHS=5   # Run 5 defaults
make landing-grade-describe PROVIDER=vertex MODEL=gemini-3.5-flash VAL_SPLIT=1 LOCAL_CLIPS=1
make video-finetune-evaluate     # evaluate saved checkpoint
```

#### Evaluation gate

| Metric | Target | Action if miss |
|---|---|---|
| Precision (YES) | ≥ 85% | Try temporal cropping, spatial cropping, SlowFast, or hybrid LLM pre-filter |
| Recall (YES) | ≥ 70% | Lower classification threshold, add YES class weight, check false negatives |
| Accuracy | — | Informational only; precision is the binding constraint |

**Important:** With 57 val clips (~29 YES, ~28 NO), 85% precision means tolerating ~4 false positives. Small val set = high variance. Consider **5-fold cross-validation** for a more stable estimate once the single-split result looks promising.

#### Error analysis checklist

On val false positives and false negatives:
1. Print `game_id`, `event_id`, human `note` from classifications CSV (already built into `landing_foul_video_train.py`)
2. Compare failure modes to LLM failures — the LLM's 38 contest FPs are the benchmark. If fine-tuned model FPs are the same clips, the visual signal may be genuinely ambiguous for those plays.
3. **Hard-negative mining:** If contest FPs dominate, consider expanding the training set with additional labeled contest fouls.
4. **Confusion with pump-fake jump-intos:** If these are the primary FP, temporal ordering features may be needed (who initiated contact) — exactly what the LLM also struggled with.

#### Decision tree after fine-tuning

```
Fine-tuned model val precision ≥ 85%?
├── YES → Step 11: sample 100–150 SF clips per official (10–15 refs)
│         Build landing_foul_video_predict.py for batch inference
│         Run classifier at scale → per-official landing foul rates
│         → Step 12: ANOVA + correlation with suppressor/amplifier profiles
├── 75–84% → Try preprocessing levers (temporal crop, spatial crop, frame count)
│            → If still <85%: try SlowFast or hybrid (LLM pre-filter → human review)
├── 55–74% → Hybrid pipeline: LLM pre-filter (98% recall) → fine-tuned model on YES-predicted
│            → Manual review remaining borderline cases
└── < 55%  → Pose estimation (MediaPipe) or scale manual classification
             → HTML tool with keyboard shortcuts (~50 min per 100 clips)
```

#### File layout

```
src/
  landing_foul_video_dataset.py   # download + decode + frozen embeddings      ← BUILT
  landing_foul_video_split.py     # stratified train/val split                 ← BUILT
  landing_foul_video_train.py     # classifier head on frozen embeddings       ← BUILT (baseline)
  landing_foul_video_finetune.py  # end-to-end VideoMAE fine-tuning            ← BUILT (Colab run degenerate)
  landing_foul_annotate_anchors.py # browser UI for per-clip contact anchors   ← BUILT (START HERE)
  landing_foul_video_predict.py   # batch inference for Step 11 scale-up       ← build after gate clears
data/
  clips/landing_foul/             # cached MP4s (284 files, gitignored)         ← DOWNLOADED
  processed/
    landing_foul_embeddings.npz   # frozen VideoMAE embeddings (zero signal)    ← EXTRACTED
    landing_foul_split.json       # reproducible train/val indices              ← CREATED
    landing_foul_clip_anchors.json # per-clip contact temporal windows         ← PENDING (annotate)
    landing_foul_frames.npz       # decode-once frame cache (rebuild after anchors) ← Colab
    landing_foul_video_best.pt    # best fine-tuned checkpoint (gitignored)     ← DEGENERATE — discard
    landing_foul_video_metrics.json # training metrics                          ← Colab run logged
```

#### What NOT to do

- **Don't use the Colab checkpoint** — degenerate constant predictor (prob=0.587 on all clips)
- **Don't re-run fine-tuning without anchors** — full-clip window already proven insufficient
- **Don't re-run the frozen baseline** — it has zero signal. The results are documented above.
- **Don't scale to per-official measurement (Steps 11–12)** until val precision ≥ 85%
- **Don't invest more in LLM prompt iteration** — exhausted
- **Don't train on UNCLEAR labels** — abstain at inference or handle separately
- **Don't leak val clips into train** — use the fixed split file (`data/processed/landing_foul_split.json`)
- **Don't use horizontal flip augmentation** — court orientation matters (broadcast camera is always from one side)

---

#### Manual grading phase — COMPLETE (2026-06-30)

The 300-clip manual grading phase is done. If you need to re-open the classifier:

1. `make landing-classifier && python -m http.server 8080 --directory output`
2. Open `http://localhost:8080/landing_foul_classifier.html`
3. **Import CSV** to restore labels; **Next Ungraded →** to find gaps
4. Export → `data/landing_foul_classifications.csv` → `make landing-merge`

---

#### THE HYBRID PIPELINE (fallback if fine-tuned precision < 75%)

The model's 98% recall means it catches nearly every true landing foul. Use it as a **pre-filter**, not a classifier:

1. Run all clips through the LLM grader (any prompt mode — recall is robust across prompts).
2. Manually review only the predicted-YES clips using the HTML classifier (`make landing-classifier`).
3. The LLM eliminates ~50% of clips (the confident NOs); humans handle the rest.

For Steps 11–12 (per-official measurement at 100–150 clips per official), this means: ~800–1,000 predicted-YES clips to manually review instead of 1,500 total. At ~30 seconds per clip, that's ~7–8 hours of manual work spread across sessions — doable, and the accuracy is perfect.

---

#### WHY THE LLM STRUGGLES: structural failure analysis

Seven approaches have been tested across `does-harden-choke` and `ref-ball`. The pattern:

**does-harden-choke (timing axis, `foul_type_llm_grader.py`):**
- 13-field observation: 40% (degenerate output collapse — identical vectors for all clips)
- 3-field observation: 50% (state classification still too hard)
- Event-ordering sequence: **71%** (temporal ordering is cognitively easier than state classification)
- Timing axis killed by Giannis counterexample regardless of grading accuracy

**ref-ball (landing foul binary, `landing_foul_llm_grader.py`):**
- Spatial V1: 58% accuracy, 55% precision, 98% recall (massive YES bias — 38/45 GT-NO predicted YES)
- Spatial V2 (+who_initiated): ~58% accuracy (traded false positives for new false negatives)
- Whistle attribution: similar YES-biased pattern (model cannot reliably parse whistle timing from audio)

**Two structural walls:**

1. **State classification is too hard.** The model cannot distinguish "ball rising without committed release" from "ball on release path" or "normal contest" from "undercut" at the resolution these clips require. The event-ordering approach partially solves this by converting "what state is this?" into "which thing happened first?"

2. **The model says YES to everything that looks like a foul.** On landing fouls, it sees any closeout contact on a perimeter jump shot and labels the defender as "under the shooter." It cannot detect shooter-initiated contact (pump-fake jump-intos) reliably, even when explicitly prompted. It cannot distinguish absence (defender's feet were NOT under the landing zone) from presence.

**Root cause:** Current multimodal LLMs process video as loosely-connected frames, not continuous physical simulations. They can identify objects and describe spatial relationships but cannot track sub-second temporal ordering between simultaneous body movements (~200–400ms windows), distinguish cause from consequence in fast interactions, or use audio as a reliable signal.

**Alternative technologies (if fine-tuned classifier underperforms):**
- Hybrid LLM pre-filter + manual review (98% recall eliminates confident NOs)
- Pose estimation + rules (MediaPipe/OpenPose) — track defender feet under shooter's landing zone
- Scale manual classification — HTML tool with keyboard shortcuts (~50 min per 100 clips)

---

#### Completed validation runs (reference)

**Run 1 — Spatial V1 (2026-06-29):**

| Setting | Value |
|---|---|
| Provider / model | Vertex `gemini-3.5-flash` |
| Prompt mode | `spatial` (V1 — no `who_initiated_contact` field) |
| Set | Primary — 93 YES/NO clips |
| Runtime | ~26 min (~17s/clip) |

| Metric | Result | Target | Status |
|---|---|---|---|
| Accuracy | 58.1% (54/93) | — | — |
| Precision (YES) | 55.3% (47/85) | ≥ 85% | **Miss** |
| Recall (YES) | 97.9% (47/48) | ≥ 70% | **Pass** |
| F1 (YES) | 70.7% | — | — |

Confusion: 7 TN, 38 FP, 1 FN, 47 TP. Failure mode: heavy YES bias on contest / pump-fake / shooter-initiated fouls.

**Run 2 — Spatial V2 with `who_initiated_contact` (2026-06-29):**

~58% accuracy. Correctly classified some pump-fake-jump-into fouls as NO but introduced new false negatives — traded FPs for FNs without net improvement.

**Run 3 — Whistle attribution (2026-06-29):**

Similar YES-biased pattern. The model claims to hear and time the whistle but produces the same spatial-assessment-driven results. Audio signal is unreliable.

**Run 4 — gemini-2.5-flash on Vertex (2026-06-29, smoke test):**

3 clips, all failed with API Error 400: `mediaResolution` rejected for this model. **Do not use gemini-2.5-flash on Vertex.** Use `gemini-3.5-flash`.

---

#### Implementation notes

- Loads clips from `data/processed/landing_foul_manifest.json` (100 Step 9 clips) + Harden/Giannis player manifests for v3 legacy rows' video URLs — all 134 ground-truth rows resolve to a video URL.
- `--validate-only` grades only ground-truth clips. Primary set (default) = `landing_classifier` source rows only → **93 YES/NO** clips. `--extended` adds the 35 v3 legacy rows → **128 YES/NO**. `--include-unclear` adds the 6 UNCLEAR rows (reported separately, excluded from primary metrics).
- Validation prints accuracy, precision/recall/F1 on YES, confusion matrix, UNCLEAR-prediction count, and per-clip mismatch detail with observations + GT notes.
- `--few-shot` selects balanced YES/NO examples (noted clips preferred), held out from the graded set when validating.
- Results → `data/processed/landing_foul_llm_results_<provider>_<model>.json` (gitignored).

**Provider setup:**

| Provider | Auth | Model notes | Video handling |
|---|---|---|---|
| **Vertex** (recommended) | gcloud ADC — no API key | Use `gemini-3.5-flash` | GCS upload via `VertexGeminiGrader` |
| Gemini API | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `gemini-2.5-flash` | Gemini Files API |
| OpenAI | `OPENAI_API_KEY` | frame-based | ffmpeg 3fps, 15 frames |
| Anthropic | `ANTHROPIC_API_KEY` | frame-based | ffmpeg 2fps, 10 frames |

**Vertex setup:**
1. Install [gcloud CLI](https://cloud.google.com/sdk/docs/install)
2. `gcloud auth application-default login`
3. `gcloud config set project project-3984c931-3755-423f-966`
4. Videos upload to GCS bucket `project-3984c931-3755-423f-966-foul-type-grader-tmp` (1-day lifecycle auto-delete)
5. **Do not use `gemini-2.5-flash` on Vertex** — per-part `mediaResolution` is rejected (400). Use `gemini-3.5-flash`.

**All validation commands:**
```bash
# Sequence prompt (YOUR FIRST RUN):
make landing-grade-validate PROVIDER=vertex MODEL=gemini-3.5-flash PROMPT_MODE=sequence

# Sequence + few-shot (YOUR SECOND RUN):
make landing-grade-validate PROVIDER=vertex MODEL=gemini-3.5-flash PROMPT_MODE=sequence FEW_SHOT=1

# Spatial (already run — reference):
make landing-grade-validate PROVIDER=vertex MODEL=gemini-3.5-flash

# Spatial + few-shot (try if sequence+few-shot is 75-84%):
make landing-grade-validate PROVIDER=vertex MODEL=gemini-3.5-flash FEW_SHOT=1

# Whistle (already run — reference):
make landing-grade-validate PROVIDER=vertex MODEL=gemini-3.5-flash PROMPT_MODE=whistle

# Small smoke run (3 clips):
make landing-grade-validate PROVIDER=vertex MODEL=gemini-3.5-flash LIMIT=3

# Extended set (128 clips incl v3 legacy):
make landing-grade-validate PROVIDER=vertex MODEL=gemini-3.5-flash EXTENDED=1
```

### Step 11: Scale to per-official measurement — PENDING (blocked on Step 10b)

1. Select 10–15 officials spanning suppressor/amplifier spectrum
2. Sample ~100–150 shooting foul clips per official (official-diverse, not player-diverse)
3. Run **fine-tuned video classifier** (`landing_foul_video_predict.py`) on all clips
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
- **Base deps (`requirements.txt`):** pandas, pyarrow, numpy, requests, tqdm, scipy
- **ML deps (`requirements-ml.txt`):** torch 2.12+, torchvision 0.27+, transformers 5.12+, opencv-python-headless, scikit-learn — **installed** as of 2026-06-30
- **System:** `ffmpeg` required for OpenAI/Anthropic frame extraction (not needed for Vertex/Gemini native video)
- **gcloud:** Required for Vertex provider. Project `project-3984c931-3755-423f-966`, ADC via `gcloud auth application-default login`.
- NBA API: use `NBAStatsClient` in `src/nba_client.py` (same pattern as does-harden-choke — rate limits are endpoint-specific, not global)
