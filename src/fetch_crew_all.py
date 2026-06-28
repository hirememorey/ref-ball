r"""Expand crew assignments to all games with PBP data.

Fetches referee assignments for all game IDs found in data/raw/pbp/,
skipping games already present in data/processed/crew_assignments.parquet.

Appends new rows to the existing parquet incrementally so progress is
preserved if the run is interrupted.

Usage:
    python src/fetch_crew_all.py
    python src/fetch_crew_all.py --max-games 50   # testing
    python src/fetch_crew_all.py --resume          # skip games already fetched
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

import config
from src.nba_client import NBAStatsClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_existing_game_ids() -> set[str]:
    path = config.CREW_ASSIGNMENTS_PATH
    if not path.exists():
        return set()
    df = pd.read_parquet(path)
    return set(df["game_id"].unique())


def get_all_pbp_game_ids() -> list[str]:
    pbp_dir = config.RAW_PBP_DIR
    return sorted(f.stem for f in pbp_dir.glob("*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand crew assignments to all PBP games")
    parser.add_argument("--max-games", type=int, default=None, help="Cap games to fetch (testing)")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip games already in crew parquet")
    args = parser.parse_args()

    all_gids = get_all_pbp_game_ids()
    logger.info("Found %d PBP games total", len(all_gids))

    if args.resume:
        existing = get_existing_game_ids()
        missing = [g for g in all_gids if g not in existing]
        logger.info("Already have crew for %d games; %d remaining", len(existing), len(missing))
    else:
        missing = all_gids

    if args.max_games is not None:
        missing = missing[: args.max_games]
        logger.info("Capping to %d games", len(missing))

    if not missing:
        logger.info("Nothing to fetch.")
        return

    client = NBAStatsClient()
    new_rows: list[dict] = []
    failed = 0
    checkpoint_interval = 100

    for i, gid in enumerate(missing, start=1):
        try:
            officials = client.get_game_officials(gid)
            new_rows.extend(officials)
        except Exception as exc:
            failed += 1
            logger.warning("  %s failed: %s", gid, exc)

        if i % 25 == 0 or i == len(missing):
            logger.info(
                "  %d/%d fetched=%d failed=%d",
                i, len(missing), len(new_rows) // 3, failed,
            )

        if i % checkpoint_interval == 0 and new_rows:
            _checkpoint(new_rows)
            new_rows = []

    if new_rows:
        _checkpoint(new_rows)

    logger.info("Done. Total failed: %d / %d", failed, len(missing))


def _checkpoint(new_rows: list[dict]) -> None:
    path = config.CREW_ASSIGNMENTS_PATH
    new_df = pd.DataFrame(new_rows)
    if path.exists():
        existing_df = pd.read_parquet(path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(path, index=False)
    logger.info("Checkpoint: wrote %d total rows to %s", len(combined), path)


if __name__ == "__main__":
    main()
