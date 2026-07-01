PYTHON ?= .venv/bin/python

.PHONY: venv fetch-pbp fetch-pbp-season fetch-l2m fetch-l2m-season ingest train-nocall predict-nocalls validate-nocall profile analyze model-crew model-crew-temporal landing-manifest landing-manifest-dry landing-classifier landing-merge landing-ground-truth landing-grade landing-grade-validate landing-grade-observe video-download video-extract video-split video-train video-train-mlp video-cv video-pipeline video-finetune video-finetune-evaluate video-annotate

venv:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

# --- Fetch: download PBP JSON from NBA Stats API (Layer 1 source data) ---

fetch-pbp:
	PYTHONPATH=. $(PYTHON) src/fetch_pbp.py

fetch-pbp-season:
	PYTHONPATH=. $(PYTHON) src/fetch_pbp.py --season $(SEASON)

fetch-pbp-playoffs:
	PYTHONPATH=. $(PYTHON) src/fetch_pbp.py --season-type Playoffs

# --- Fetch L2M: Last Two Minute reports + crew assignments (validation ground truth) ---

fetch-l2m:
	PYTHONPATH=. $(PYTHON) src/fetch_l2m.py

fetch-l2m-season:
	PYTHONPATH=. $(PYTHON) src/fetch_l2m.py --season $(SEASON)

# --- Ingest: parse PBP JSON into structured records with official attribution (Layer 1) ---

ingest:
	PYTHONPATH=. $(PYTHON) src/ingest.py

ingest-season:
	PYTHONPATH=. $(PYTHON) src/ingest.py --season $(SEASON)

# --- No-call model: train, predict, validate (Layer 3) ---

train-nocall:
	PYTHONPATH=. $(PYTHON) src/nocall_model.py train

predict-nocalls:
	PYTHONPATH=. $(PYTHON) src/nocall_model.py predict

validate-nocall:
	PYTHONPATH=. $(PYTHON) src/nocall_model.py validate

# --- Profile: build per-official profiles ---

profile:
	PYTHONPATH=. $(PYTHON) src/ref_profiles.py

profile-official:
	PYTHONPATH=. $(PYTHON) src/ref_profiles.py --official $(OFFICIAL)

profile-calling:
	PYTHONPATH=. $(PYTHON) src/official_calling_profiles.py build

profile-calling-summary:
	PYTHONPATH=. $(PYTHON) src/official_calling_profiles.py summary

# --- Analyze: three-track analysis ---

analyze:
	PYTHONPATH=. $(PYTHON) src/analyze.py

analyze-track:
	PYTHONPATH=. $(PYTHON) src/analyze.py --track $(TRACK)

# --- Step 5: crew predictive model ---

model-crew:
	PYTHONPATH=. $(PYTHON) src/crew_predictive_model.py build

model-crew-temporal:
	PYTHONPATH=. $(PYTHON) src/crew_predictive_model.py build --temporal

model-crew-diagnose:
	PYTHONPATH=. $(PYTHON) src/crew_predictive_model.py diagnose

# --- Step 5b: player-level crew FTA/36 model ---

model-player-crew:
	PYTHONPATH=. $(PYTHON) src/player_crew_predictive_model.py build

model-player-crew-static:
	PYTHONPATH=. $(PYTHON) src/player_crew_predictive_model.py build --static

model-player-crew-diagnose:
	PYTHONPATH=. $(PYTHON) src/player_crew_predictive_model.py diagnose

l2m-validate:
	PYTHONPATH=. $(PYTHON) src/l2m_validation.py build

l2m-validate-summary:
	PYTHONPATH=. $(PYTHON) src/l2m_validation.py summary

# --- Step 7: does-harden-choke merge ---

dhc-merge:
	PYTHONPATH=. $(PYTHON) src/dhc_merge.py build

dhc-merge-summary:
	PYTHONPATH=. $(PYTHON) src/dhc_merge.py summary

# --- Step 9: landing foul ground truth (3-FT shooting fouls) ---

landing-manifest:
	PYTHONPATH=. $(PYTHON) src/landing_foul_manifest.py --clips $(or $(CLIPS),100)

landing-manifest-dry:
	PYTHONPATH=. $(PYTHON) src/landing_foul_manifest.py --clips $(or $(CLIPS),100) --dry-run

landing-classifier:
	PYTHONPATH=. $(PYTHON) src/landing_foul_classifier.py

landing-merge:
	PYTHONPATH=. $(PYTHON) src/landing_foul_merge.py

# --- Step 10: landing foul LLM grader (spatial binary YES/NO) ---
# Set PROVIDER, MODEL, and flags via env. Gemini native video is recommended.
#   make landing-grade-validate PROVIDER=gemini MODEL=gemini-2.5-flash LIMIT=10
#   make landing-grade-validate PROVIDER=gemini MODEL=gemini-2.5-flash EXTENDED=1
#   make landing-grade PROVIDER=gemini MODEL=gemini-2.5-flash

landing-grade:
	PYTHONPATH=. $(PYTHON) src/landing_foul_llm_grader.py \
		--provider $(PROVIDER) --model $(MODEL) $(if $(PROMPT_MODE),--prompt-mode $(PROMPT_MODE)) \
		$(if $(FEW_SHOT),--few-shot)

landing-grade-validate:
	PYTHONPATH=. $(PYTHON) src/landing_foul_llm_grader.py \
		--provider $(PROVIDER) --model $(MODEL) --validate-only \
		$(if $(EXTENDED),--extended) $(if $(INCLUDE_UNCLEAR),--include-unclear) \
		$(if $(PROMPT_MODE),--prompt-mode $(PROMPT_MODE)) $(if $(FEW_SHOT),--few-shot) \
		$(if $(LIMIT),--limit $(LIMIT))

landing-ground-truth: landing-manifest landing-classifier

# --- Step 10 observe: structured observation-only prompt (Phase 1) ---
# Derives classification from feature vector post-hoc, not from the model.
#   make landing-grade-observe PROVIDER=vertex MODEL=gemini-3.5-flash
#   make landing-grade-observe PROVIDER=vertex MODEL=gemini-3.5-flash LIMIT=10

landing-grade-observe:
	PYTHONPATH=. $(PYTHON) src/landing_foul_llm_grader.py \
		--provider $(PROVIDER) --model $(MODEL) --prompt-mode observe --validate-only \
		$(if $(EXTENDED),--extended) $(if $(LIMIT),--limit $(LIMIT))

# --- Step 10b: Video classifier (frozen VideoMAE → logistic regression) ---
#   Full pipeline: make video-pipeline
#   Individual steps: make video-download && make video-extract && make video-split && make video-train

video-download:
	PYTHONPATH=. $(PYTHON) src/landing_foul_video_dataset.py download $(if $(LIMIT),--limit $(LIMIT))

video-verify:
	PYTHONPATH=. $(PYTHON) src/landing_foul_video_dataset.py verify

video-package:
	PYTHONPATH=. $(PYTHON) src/landing_foul_video_dataset.py package

video-extract:
	PYTHONPATH=. $(PYTHON) src/landing_foul_video_dataset.py extract $(if $(MODEL),--model $(MODEL))

video-split:
	PYTHONPATH=. $(PYTHON) src/landing_foul_video_split.py $(if $(TEST_SIZE),--test-size $(TEST_SIZE)) $(if $(SEED),--seed $(SEED))

video-train:
	PYTHONPATH=. $(PYTHON) src/landing_foul_video_train.py --model logreg

video-train-mlp:
	PYTHONPATH=. $(PYTHON) src/landing_foul_video_train.py --model mlp

video-cv:
	PYTHONPATH=. $(PYTHON) src/landing_foul_video_train.py --model logreg --cv $(or $(FOLDS),5)

video-pipeline: video-download video-extract video-split video-train

# --- Step 10c: end-to-end VideoMAE fine-tuning ---
#   make video-finetune
#   make video-finetune SMOKE=1            # tiny run to validate the pipeline
#   make video-finetune PHASE=head HEAD_EPOCHS=5
#   make video-finetune-evaluate
#   make video-finetune-evaluate CHECKPOINT=data/processed/landing_foul_video_best.pt

video-finetune:
	PYTHONPATH=. $(PYTHON) src/landing_foul_video_finetune.py \
		--phase $(or $(PHASE),two-phase) \
		--head-epochs $(or $(HEAD_EPOCHS),5) --finetune-epochs $(or $(FINETUNE_EPOCHS),15) \
		--head-lr $(or $(HEAD_LR),1e-3) --finetune-lr $(or $(FINETUNE_LR),2e-5) \
		--unfreeze-layers $(or $(UNFREEZE_LAYERS),4) --batch-size $(or $(BATCH_SIZE),4) \
		--temporal-window "$(if $(TEMPORAL_WINDOW),$(TEMPORAL_WINDOW),0.0,1.0)" --jitter $(or $(JITTER),6) \
		--dropout $(or $(DROPOUT),0.4) --weight-decay $(or $(WEIGHT_DECAY),0.01) \
		--yes-weight $(or $(YES_WEIGHT),1.0) --patience $(or $(PATIENCE),6) --seed $(or $(SEED),42) \
		--device $(or $(DEVICE),auto) \
		$(if $(SMOKE),--head-epochs 1 --finetune-epochs 1 --batch-size 2 --patience 0)

video-finetune-evaluate:
	PYTHONPATH=. $(PYTHON) src/landing_foul_video_finetune.py --evaluate-only \
		--checkpoint $(or $(CHECKPOINT),data/processed/landing_foul_video_best.pt) \
		--device $(or $(DEVICE),auto) --batch-size $(or $(BATCH_SIZE),4) \
		--temporal-window "$(if $(TEMPORAL_WINDOW),$(TEMPORAL_WINDOW),0.0,1.0)"

# Annotate per-clip foul anchors (temporal window) in a local browser UI.
#   make video-annotate
#   make video-annotate PORT=8765
video-annotate:
	PYTHONPATH=. $(PYTHON) src/landing_foul_annotate_anchors.py \
		--host 127.0.0.1 --port $(or $(PORT),8765)
