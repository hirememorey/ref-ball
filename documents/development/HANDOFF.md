# Development Handoff (June 27, 2026)

This document provides a snapshot of the project state and the immediate next steps for any developer picking up the codebase.

## Current State

We transitioned the project from a small, playoff-only proof of concept to a scaled, 11-season dataset with ground-truth validation.

**1. Play-By-Play (PBP) Fetch (In Progress / Nearing Completion)**
*   `src/fetch_pbp.py` was built to download raw PBP JSON files from the NBA API.
*   It is currently fetching all Regular Season (RS) games from 2014-15 through 2024-25 (~13,500 games).
*   Files are written to `data/raw/pbp/`.

**2. L2M Validation Data (Complete)**
*   `src/fetch_l2m.py` was built and successfully executed.
*   It scraped 7 seasons (2018-19 to 2024-25) of Last Two Minute (L2M) reports.
*   **Result:** `data/processed/l2m_events.parquet` contains ~56k events, including ~14,500 shooting fouls. Crucially, this includes **685 INC (Incorrect No-Call)** shooting fouls. These 685 events are the ground-truth validation set for the Layer 3 no-call video model.

**3. Referee Profiles & Crew Assignments (Code Ready, Data Partial)**
*   `src/ref_profiles.py` was built to calculate per-official foul rates and RS vs PO deltas.
*   **The Denominator Problem Resolved:** PBP data only lists officials who *blew a whistle*. To calculate true rates (fouls per game worked), we need the full 3-person crew assignment.
*   `fetch_l2m.py` collected full crew assignments for all L2M games by parsing `__NEXT_DATA__` on `www.nba.com` game pages (the `boxscoresummaryv2` API endpoint is unreliable for officials). These are stored in `data/processed/crew_assignments.parquet`.

## Immediate Next Steps (Day 1)

1. **Run the Ingest:**
   The `data/processed/games/` directory currently only contains a small sample of parsed parquets. Once the background PBP fetch completes, you must run `make ingest` to parse all ~13,500 raw JSON files into structured parquet files.

2. **Generate Layer 1 Profiles:**
   Run `make profile`. This will execute `src/ref_profiles.py`, joining the newly ingested PBP data against the crew assignments to generate the descriptive profiles (Paper 1).

3. **Expand Crew Assignments (Recommended):**
   `crew_assignments.parquet` currently only covers the ~2,700 games that had L2M reports. To get accurate "games worked" denominators for the *entire* 13,500 game dataset, you should expand the crew fetch. 
   *   *Task:* Write a script (or add a CLI flag to `fetch_l2m.py`) that iterates over all game IDs in `data/raw/pbp/` and calls `fetch_crew_assignments()`. Note: At ~2 seconds per game, this will take ~7.5 hours to run in the background.

4. **Layer 3 Feasibility Study (The Go/No-Go Gate):**
   Begin work on `src/nocall_model.py`. 
   *   Take a stratified sample of the L2M shooting foul events (both INC/IC and CC/CNC).
   *   Use the `nba_client.py`'s `get_video_events()` to download the video clips for these events.
   *   Train a simple clip-level binary classifier.
   *   **The Gate:** If the model cannot distinguish between ground-truth misses (INC) and correct non-calls (CNC) with acceptable precision/recall, the project pivots. We drop Layer 3 and publish a descriptive paper using only Layer 1 (called rates) and the L2M data.
