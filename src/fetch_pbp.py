r"""Download play-by-play JSON files from the NBA Stats API.

Fetches league game logs to enumerate game IDs, then downloads playbyplayv3
JSON for each game not already present in data/raw/pbp/.

Official attribution in the PBP description field is 100% for games from
2014-15 onward, so we constrain fetching to those seasons by default.

Input:  none (uses leaguegamelog endpoint to discover game IDs)
Output: data/raw/pbp/{game_id}.json

Usage:
    python src/fetch_pbp.py                          # all RS seasons 2014-15..2024-25
    python src/fetch_pbp.py --season 2023-24         # single season
    python src/fetch_pbp.py --season-type Playoffs   # playoffs instead of RS
    python src/fetch_pbp.py --max-games 50           # cap for testing
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import config
from src.nba_client import NBAStatsClient, result_set_to_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Official attribution in PBP description is 100% from 2014-15 onward.
DEFAULT_SEASONS = [config.year_to_season(y) for y in range(2014, 2025)]


def discover_game_ids(
    client: NBAStatsClient,
    season: str,
    season_type: str,
) -> list[str]:
    """Return sorted unique game IDs for a season + season type."""
    resp = client.get_league_game_log(season, season_type)
    rows = result_set_to_records(resp)
    ids = sorted({r["GAME_ID"] for r in rows if r.get("GAME_ID")})
    return ids


def fetch_season(
    client: NBAStatsClient,
    season: str,
    season_type: str,
    out_dir: Path,
    max_games: int | None,
    force: bool,
) -> tuple[int, int, int]:
    """Download PBP JSON for one season. Returns (fetched, skipped, failed)."""
    game_ids = discover_game_ids(client, season, season_type)
    logger.info("%s %s: %d games discovered", season, season_type, len(game_ids))

    if max_games is not None:
        game_ids = game_ids[:max_games]
        logger.info("Capping to %d games", len(game_ids))

    fetched = 0
    skipped = 0
    failed = 0
    for i, gid in enumerate(game_ids, start=1):
        out_path = out_dir / f"{gid}.json"
        if out_path.exists() and not force:
            skipped += 1
            continue
        try:
            resp = client.get_play_by_play(gid)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(resp, f)
            fetched += 1
            if i % 25 == 0 or i == len(game_ids):
                logger.info(
                    "  %s %s/%s fetched=%d skipped=%d failed=%d",
                    season, i, len(game_ids), fetched, skipped, failed,
                )
        except Exception as exc:
            failed += 1
            logger.warning("  %s failed: %s", gid, exc)
    return fetched, skipped, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PBP JSON from NBA Stats API")
    parser.add_argument("--season", default=None, help="Single season (e.g. 2023-24)")
    parser.add_argument("--seasons", nargs="+", default=None, help="Multiple seasons")
    parser.add_argument(
        "--season-type",
        default="Regular Season",
        choices=["Regular Season", "Playoffs"],
    )
    parser.add_argument("--max-games", type=int, default=None, help="Cap games per season (testing)")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    if args.season:
        seasons = [args.season]
    elif args.seasons:
        seasons = args.seasons
    else:
        seasons = DEFAULT_SEASONS

    out_dir = config.RAW_PBP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", out_dir)

    client = NBAStatsClient()

    total_fetched = 0
    total_skipped = 0
    total_failed = 0
    for season in seasons:
        fetched, skipped, failed = fetch_season(
            client, season, args.season_type, out_dir, args.max_games, args.force,
        )
        total_fetched += fetched
        total_skipped += skipped
        total_failed += failed
        logger.info(
            "%s %s done: fetched=%d skipped=%d failed=%d",
            season, args.season_type, fetched, skipped, failed,
        )

    logger.info(
        "All done. fetched=%d skipped=%d failed=%d",
        total_fetched, total_skipped, total_failed,
    )


if __name__ == "__main__":
    main()
