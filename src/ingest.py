r"""Parse PBP JSON files and extract per-official foul call records.

The calling official's name is embedded in the foul description field:
  "Gasol S.FOUL (P1.T1) (R.Garretson)"
  Pattern: \(([A-Z]\.\s*\w+)\)\s*$

Coverage: 100% for games from 2014-15 onward, 0% before that.

Input:  data/raw/pbp/*.json  (playbyplayv3 API responses)
Output: data/processed/games/{game_id}.parquet

Usage:
    python src/ingest.py
    python src/ingest.py --season 2023-24
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OFFICIAL_PATTERN = re.compile(r"\(([A-Z]\.\s*\w+)\)\s*$")


def parse_official_name(description: str) -> str | None:
    m = OFFICIAL_PATTERN.search(description)
    return m.group(1) if m else None


def parse_pbp_game(pbp_path: Path) -> list[dict]:
    with open(pbp_path) as f:
        data = json.load(f)

    game = data.get("game", {})
    game_id = game.get("gameId", pbp_path.stem)
    actions = game.get("actions", [])

    records = []
    for a in actions:
        if a.get("actionType") != "Foul":
            continue

        desc = a.get("description", "")
        official_name = parse_official_name(desc)
        sub_type = a.get("subType", "")

        margin = 0
        sh = a.get("scoreHome", "")
        sa = a.get("scoreAway", "")
        if sh and sa:
            try:
                margin = int(sh) - int(sa)
            except ValueError:
                pass

        records.append({
            "game_id": game_id,
            "event_num": a.get("actionNumber", 0),
            "period": a.get("period", 0),
            "clock": a.get("clock", ""),
            "event_type": "Foul",
            "foul_type": sub_type,
            "caller_official_name": official_name or "",
            "committing_player_id": a.get("personId", ""),
            "committing_player_name": a.get("playerNameI", a.get("playerName", "")),
            "committing_team_id": a.get("teamId", ""),
            "committing_team_tricode": a.get("teamTricode", ""),
            "score_home": sh,
            "score_away": sa,
            "margin": margin,
            "location": a.get("location", ""),
            "description": desc,
            "video_available": a.get("videoAvailable", 0),
        })

    return records


def main():
    parser = argparse.ArgumentParser(description="Ingest PBP JSON files into structured records")
    parser.add_argument("--season", default=None, help="Filter to a single season (e.g. 2023-24)")
    args = parser.parse_args()

    pbp_dir = config.RAW_DIR / "pbp"
    if not pbp_dir.exists():
        logger.error("No PBP directory found at %s", pbp_dir)
        return

    pbp_files = sorted(pbp_dir.glob("*.json"))
    if not pbp_files:
        logger.error("No PBP JSON files found in %s", pbp_dir)
        return

    logger.info("Found %d PBP files to ingest", len(pbp_files))

    all_records = []
    for pf in pbp_files:
        records = parse_pbp_game(pf)
        if records:
            game_id = records[0]["game_id"]
            out_path = config.GAMES_DIR / f"{game_id}.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(records).to_parquet(out_path, index=False)

            with_official = sum(1 for r in records if r["caller_official_name"])
            logger.info("  %s: %d fouls (%d with official attribution)",
                        game_id, len(records), with_official)
            all_records.extend(records)

    logger.info("Total: %d foul records (%d with official attribution)",
                len(all_records), sum(1 for r in all_records if r["caller_official_name"]))


if __name__ == "__main__":
    main()
