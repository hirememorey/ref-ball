r"""Collect NBA Last Two Minute (L2M) reports and crew assignments.

L2M reports are the league's audited ground truth for no-call detection.
Each report covers the last 2 minutes of regulation (and OT) and labels
every officiating decision as CC (correct call), CNC (correct no-call),
IC (incorrect call), or INC (incorrect no-call).

INC events are the validation standard for the no-call model (Layer 3).

Pipeline:
  1. Discover game IDs from NBA's public L2M archive pages
  2. Fetch L2M JSON for each game from official.nba.com
  3. Parse to parquet (events + reports)
  4. Fetch crew assignments from www.nba.com game pages

Input:  none (discovers game IDs from archive pages)
Output: data/processed/l2m_events.parquet
        data/processed/l2m_reports.parquet
        data/processed/crew_assignments.parquet

Usage:
    python src/fetch_l2m.py                          # all seasons
    python src/fetch_l2m.py --season 2023-24         # single season
    python src/fetch_l2m.py --max-games 50           # cap for testing
    python src/fetch_l2m.py --no-crew                # skip crew fetch
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from html import unescape
from pathlib import Path
from typing import Any

import pandas as pd
import requests

import config
from src.nba_client import NBAStatsClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def discover_l2m_game_ids(season: str) -> list[str]:
    """Scrape game IDs from the NBA's public L2M archive page for a season."""
    url = config.L2M_ARCHIVE_URLS[season]
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    text = response.text
    ids = re.findall(r"L2MReport\.html\?gameId=(\d{10})", text)
    return list(dict.fromkeys(ids))


def parse_l2m_payload(
    payload: dict[str, Any], game_id: str, season: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse L2M JSON into a report row and event rows."""
    game = (payload.get("game") or [{}])[0]
    events = payload.get("l2m") or []
    rows = []
    for index, event in enumerate(events):
        decision = str(event.get("CallRatingName") or "").strip().upper()
        rows.append({
            "game_id": game_id,
            "event_index": index,
            "season": season,
            "period": event.get("PeriodName"),
            "game_clock": event.get("PCTime"),
            "call_type": event.get("CallType"),
            "review_decision": decision,
            "incorrect": 1 if decision in {"IC", "INC"} else 0,
            "committing_player": _clean(event.get("CP")),
            "disadvantaged_player": _clean(event.get("DP")),
            "comment": _clean(event.get("Comment")),
            "difficulty": event.get("Difficulty"),
            "possession_id": event.get("posID"),
            "possession_start": event.get("posStart"),
            "possession_end": event.get("posEnd"),
            "team_id_in_favor": event.get("teamIdInFavor"),
            "error_in_favor": event.get("errorInFavor"),
            "video_event_num": event.get("VideolLink"),
        })

    report = {
        "game_id": game_id,
        "season": season,
        "game_date": game.get("GameDate"),
        "home_team_id": game.get("HomeTeamId"),
        "away_team_id": game.get("AwayTeamId"),
        "home_team": game.get("Home_team"),
        "away_team": game.get("Away_team"),
        "home_score": game.get("HomeTeamScore"),
        "away_score": game.get("VisitorTeamScore"),
        "event_count": len(rows),
        "incorrect_count": sum(r["incorrect"] for r in rows),
        "has_report": 1 if rows else 0,
        "comments": _clean(game.get("L2M_Comments")),
    }
    return report, rows


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    return unescape(str(value))


def fetch_l2m_season(
    client: NBAStatsClient,
    season: str,
    max_games: int | None,
) -> tuple[list[dict], list[dict]]:
    """Fetch all L2M reports for one season. Returns (reports, events)."""
    game_ids = discover_l2m_game_ids(season)
    logger.info("%s: %d L2M games discovered", season, len(game_ids))
    if max_games is not None:
        game_ids = game_ids[:max_games]
        logger.info("Capping to %d games", len(game_ids))

    reports = []
    events = []
    failed = 0
    for i, gid in enumerate(game_ids, start=1):
        try:
            payload = client.get_l2m_report(gid)
            report, event_rows = parse_l2m_payload(payload, gid, season)
            reports.append(report)
            events.extend(event_rows)
        except Exception as exc:
            failed += 1
            logger.warning("  L2M %s failed: %s", gid, exc)
        if i % 25 == 0 or i == len(game_ids):
            logger.info(
                "  %s %d/%d reports=%d events=%d failed=%d",
                season, i, len(game_ids), len(reports), len(events), failed,
            )
    return reports, events


def fetch_crew_assignments(
    client: NBAStatsClient,
    game_ids: list[str],
) -> list[dict]:
    """Fetch referee assignments for a list of game IDs."""
    rows = []
    failed = 0
    for i, gid in enumerate(game_ids, start=1):
        try:
            officials = client.get_game_officials(gid)
            rows.extend(officials)
        except Exception as exc:
            failed += 1
            logger.warning("  crew %s failed: %s", gid, exc)
        if i % 25 == 0 or i == len(game_ids):
            logger.info(
                "  crew %d/%d assignments=%d failed=%d",
                i, len(game_ids), len(rows), failed,
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect L2M reports and crew assignments")
    parser.add_argument("--season", default=None, help="Single season (e.g. 2023-24)")
    parser.add_argument("--seasons", nargs="+", default=None, help="Multiple seasons")
    parser.add_argument("--max-games", type=int, default=None, help="Cap games per season (testing)")
    parser.add_argument("--no-crew", action="store_true", help="Skip crew assignment fetch")
    args = parser.parse_args()

    if args.season:
        seasons = [args.season]
    elif args.seasons:
        seasons = args.seasons
    else:
        seasons = config.L2M_SEASONS

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    client = NBAStatsClient()

    all_reports = []
    all_events = []
    for season in seasons:
        reports, events = fetch_l2m_season(client, season, args.max_games)
        all_reports.extend(reports)
        all_events.extend(events)
        logger.info(
            "%s done: %d reports, %d events (%d incorrect)",
            season, len(reports), len(events), sum(r["incorrect_count"] for r in reports),
        )

    if all_reports:
        reports_df = pd.DataFrame(all_reports)
        events_df = pd.DataFrame(all_events)
        reports_df.to_parquet(config.L2M_REPORTS_PATH, index=False)
        events_df.to_parquet(config.L2M_EVENTS_PATH, index=False)
        logger.info(
            "Wrote %d reports to %s", len(reports_df), config.L2M_REPORTS_PATH
        )
        logger.info(
            "Wrote %d events to %s (%d INC/IC)",
            len(events_df), config.L2M_EVENTS_PATH, int(events_df["incorrect"].sum()),
        )

    if all_reports and not args.no_crew:
        game_ids = sorted({r["game_id"] for r in all_reports if r.get("has_report")})
        logger.info("Fetching crew assignments for %d games", len(game_ids))
        crew_rows = fetch_crew_assignments(client, game_ids)
        if crew_rows:
            crew_df = pd.DataFrame(crew_rows)
            crew_df.to_parquet(config.CREW_ASSIGNMENTS_PATH, index=False)
            logger.info("Wrote %d crew assignments to %s", len(crew_df), config.CREW_ASSIGNMENTS_PATH)


if __name__ == "__main__":
    main()
