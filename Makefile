PYTHON ?= .venv/bin/python

.PHONY: venv fetch-pbp fetch-pbp-season fetch-l2m fetch-l2m-season ingest train-nocall predict-nocalls validate-nocall profile analyze model-crew model-crew-temporal

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
