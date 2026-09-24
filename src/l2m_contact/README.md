# L2M contact pipeline

Uses the NBA's Last Two Minute (L2M) reports as league-labeled examples of called and uncalled
contact: the rating (CC / CNC / IC / INC), the league's written comment, and a referee clip per
play. Findings: [SEPTEMBER-2026-FINDINGS.md](../../documents/development/SEPTEMBER-2026-FINDINGS.md)
sections 8-10.

Run scripts from the repo root. Paths are in `paths.py`:

| Path | Contents | Git |
|---|---|---|
| `data/raw/l2m/<season>/` | L2M report JSON | ignored |
| `data/processed/l2m_contact/` | events tables, LLM cache, fetch logs, video-test frames and outputs | ignored |
| `data/clips/l2m/<season>/` | referee-clip MP4s (about 11 MB each) | ignored |
| `data/l2m_contact/` | LLM contact tags, audit sample and grades, crew results, challenges, video-test sample and answers | tracked |

## Steps

```bash
# 1. Reports (about 2 h for all 8 seasons; one request at a time, resumable) and tables
python3 src/l2m_contact/fetch_l2m_reports.py json
python3 src/l2m_contact/fetch_l2m_reports.py table
python3 src/l2m_contact/fetch_crews.py                 # crews missing from site/.../crew_assignments.csv

# 2. Clip pilot (2023-24: all foul INC + 700 stratified foul CNC; about 11 GB)
python3 src/l2m_contact/fetch_clips.py select 2023-24 700
python3 src/l2m_contact/fetch_clips.py download

# 3. Contact tags
python3 src/l2m_contact/classify_comments.py           # rule tags -> events_tagged.csv
LLM_BACKEND=codex LLM_BATCH=200 LLM_WORKERS=8 python3 src/l2m_contact/llm_tag.py all   # Luna, all rows
LLM_BACKEND=codex LLM_BATCH=50 python3 src/l2m_contact/luna_repair.py                   # truncated batches
python3 src/l2m_contact/hybrid.py                      # Sonnet re-check of flagged rows -> contact_tags.csv

# 4. Human audit of the tags (blind; grades append to data/l2m_contact/tag_audit_grades.csv)
python3 src/l2m_contact/audit_grader.py                # http://localhost:8793
python3 src/l2m_contact/evaluate_tags.py

# 5. Crew test (needs numpy, pandas, scipy, scikit-learn)
python src/l2m_contact/analyze_crews.py && python src/l2m_contact/heldout_null.py
TAGS=hybrid python src/l2m_contact/analyze_crews.py && TAGS=hybrid python src/l2m_contact/heldout_null.py

# 6. Coach's challenges (hoopR play-by-play parquet per season)
python src/l2m_contact/coach_challenges.py 2019-20=path/to/play_by_play_2019-20.parquet ...

# 7. Video test (sample is frozen in data/l2m_contact/video_test_sample.csv)
python3 src/l2m_contact/video_test/video_test.py download
python3 src/l2m_contact/video_test/video_test.py frames
python3 src/l2m_contact/video_test/video_test.py run sonnet     # or luna
GEMINI_PROVIDER=vertex python3 src/l2m_contact/video_test/gemini_video.py
python3 src/l2m_contact/video_test/video_test.py score
```

The committed `contact_tags.csv` lets step 5 run with `TAGS=hybrid` without repeating the LLM
work. Step 3's LLM calls go through the Claude Code CLI (`claude -p`) and the Codex CLI
(`codex exec`); see the docstring in `llm_tag.py` for backends and usage guards.

## Notes

- Clip URL: `https://ak-static.cms.nba.com/wp-content/uploads/referee-clips/{gameId}_{VideolLink}_DF%20BCAST_1509kbps.mp4`.
  This CDN answers 403 for files that do not exist. A few valid-looking files are audio-only
  stubs; `fetch_clips.py` checks each file with ffprobe.
- The L2M prompt (`llm_prompt.md`, hash 763a74ea) is frozen: changing it invalidates the LLM
  cache and the clean test on the 65 ungraded audit rows.
- LLM cache files are named by a hash of their row ids, so a re-run with a different row list
  cannot reuse another batch's answers.
- `contact_tags.csv` holds ids and labels only. Join to `events.csv` for the league comments.
