PYTHON ?= .venv/bin/python

.PHONY: venv scrape-harden scrape-harden-po scrape-giannis scrape-giannis-po classify-harden classify-giannis analyze serve

venv:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

# --- Scraping: build clip manifests from NBA video API ---

scrape-harden:
	PYTHONPATH=. $(PYTHON) src/foul_scraper.py --player "James Harden" --season 2019-20 --games 5

scrape-harden-po:
	PYTHONPATH=. $(PYTHON) src/foul_scraper.py --player "James Harden" --season 2019-20 --games 5 --season-type Playoffs

scrape-giannis:
	PYTHONPATH=. $(PYTHON) src/foul_scraper.py --player "Giannis Antetokounmpo" --season 2023-24 --games 5

scrape-giannis-po:
	PYTHONPATH=. $(PYTHON) src/foul_scraper.py --player "Giannis Antetokounmpo" --season 2023-24 --games 5 --season-type Playoffs

# --- Classification: generate HTML tools ---

classify-harden:
	PYTHONPATH=. $(PYTHON) src/foul_classifier.py --manifest data/processed/manifest_james_harden_rs.json

classify-giannis:
	PYTHONPATH=. $(PYTHON) src/foul_classifier.py --manifest data/processed/manifest_giannis_antetokounmpo_rs.json

# --- Analysis: compare RS vs PO foul-type composition ---

analyze:
	PYTHONPATH=. $(PYTHON) src/analyze.py

analyze-harden:
	PYTHONPATH=. $(PYTHON) src/analyze.py --player "James Harden"

# --- Serve classifier HTML ---

serve:
	@echo "Serving classifier at http://localhost:8080/"
	$(PYTHON) -m http.server 8080 --directory output
