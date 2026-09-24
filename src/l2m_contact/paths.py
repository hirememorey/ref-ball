"""Shared paths for the L2M contact pipeline.

Large or reproducible data stays in gitignored directories; small derived results that are
expensive to regenerate (LLM tags, audit grades, test results) are tracked in data/l2m_contact/.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

RAW = REPO / "data" / "raw" / "l2m"                    # L2M report JSON, one folder per season (gitignored)
WORK = REPO / "data" / "processed" / "l2m_contact"     # events tables, LLM cache, fetch logs (gitignored)
CLIPS = REPO / "data" / "clips" / "l2m"                # referee-clip MP4s, one folder per season (gitignored)
TRACKED = REPO / "data" / "l2m_contact"                # small derived results (committed)
CREW = REPO / "site" / "public" / "downloads" / "crew_assignments.csv"

for d in (RAW, WORK, CLIPS, TRACKED):
    d.mkdir(parents=True, exist_ok=True)
