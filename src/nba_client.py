"""NBA Stats API client — adapted from does-harden-choke.

Provides rate-limited, cached access to stats.nba.com endpoints:
  - playergamelogs: per-player game logs (RS + PO)
  - teamgamelogs: per-team game logs (for team scores / margins)
  - playbyplayv3: full play-by-play for a single game
  - videoeventsasset: video clip URLs for a specific play event
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError, RetryError
from urllib3.util.retry import Retry

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

CACHE_SEASON_DAYS = 30
CACHE_GAMELOG_DAYS = 7


def normalize_game_id(game_id: str | int) -> str:
    """Normalize game IDs to 10-digit NBA Stats format (e.g. 0041600236)."""
    gid = str(game_id).strip()
    if gid.isdigit():
        return gid.zfill(10)
    return gid


def playbyplay_actions(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract action rows from playbyplayv3 response."""
    game = response.get("game") or {}
    actions = game.get("actions") or []
    return actions if isinstance(actions, list) else []


def result_set_to_records(
    response: Dict[str, Any],
    result_set_index: int = 0,
) -> List[Dict[str, Any]]:
    """Convert NBA Stats API resultSets into list-of-dicts."""
    result_sets = response.get("resultSets") or response.get("resultSet")
    if not result_sets:
        return []
    if isinstance(result_sets, dict):
        result_sets = [result_sets]
    if result_set_index >= len(result_sets):
        return []
    rs = result_sets[result_set_index]
    headers = rs.get("headers", [])
    rows = rs.get("rowSet", [])
    return [dict(zip(headers, row)) for row in rows]


class NBAStatsClient:
    """HTTP client for stats.nba.com with rate limiting, caching, and retries."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        min_interval: float = 1.2,
        cache_days_season: int = CACHE_SEASON_DAYS,
        cache_days_gamelog: int = CACHE_GAMELOG_DAYS,
        max_attempts: int = 10,
    ):
        self.base_url = "https://stats.nba.com/stats"
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.cache_days_season = cache_days_season
        self.cache_days_gamelog = cache_days_gamelog
        self.max_attempts = max_attempts
        self.last_request_time = 0.0

        self.session = requests.Session()
        retry_strategy = Retry(
            total=2,
            connect=2,
            read=0,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(
            {
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Origin": "https://www.nba.com",
                "Referer": "https://www.nba.com/",
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/148.0.0.0 Safari/537.36"
                ),
                "sec-ch-ua": '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"macOS"',
                "x-nba-stats-origin": "stats",
                "x-nba-stats-token": "true",
            }
        )
        self.timeout = (15, 45)

    def _cache_path(self, endpoint: str, params: Optional[Dict]) -> Path:
        hasher = hashlib.md5()
        if params:
            hasher.update(json.dumps(params, sort_keys=True).encode())
        hasher.update(endpoint.encode())
        return self.cache_dir / f"{hasher.hexdigest()}.json"

    def _cache_ttl(self, endpoint: str) -> timedelta:
        if endpoint in ("playergamelogs", "teamgamelogs"):
            return timedelta(days=self.cache_days_gamelog)
        return timedelta(days=self.cache_days_season)

    def _read_cache(self, path: Path, ttl: timedelta) -> Optional[Dict]:
        if not path.exists():
            return None
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        if age > ttl:
            return None
        with open(path) as f:
            return json.load(f)

    def _write_cache(self, path: Path, data: Dict) -> None:
        with open(path, "w") as f:
            json.dump(data, f)

    def _wait(self) -> None:
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        time.sleep(random.uniform(0.2, 0.8))
        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        cache_path = self._cache_path(endpoint, params)
        ttl = self._cache_ttl(endpoint)
        cached = self._read_cache(cache_path, ttl)
        if cached is not None:
            logger.debug("Cache hit: %s", endpoint)
            return cached

        url = f"{self.base_url}/{endpoint}"
        retry_statuses = {429, 500, 502, 503, 504}
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                self._wait()
                logger.info("GET %s (attempt %d/%d)", endpoint, attempt, self.max_attempts)
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code in retry_statuses:
                    wait = int(response.headers.get("Retry-After", 0)) or min(60, 2**attempt)
                    logger.warning("HTTP %s on %s; sleeping %ss", response.status_code, endpoint, wait)
                    time.sleep(wait + random.uniform(0, 1))
                    continue
                response.raise_for_status()
                data = response.json()
                self._write_cache(cache_path, data)
                return data
            except (RetryError, HTTPError, requests.RequestException) as exc:
                last_error = exc
                if attempt >= self.max_attempts:
                    break
                wait = min(60, 2**attempt) + random.uniform(0, 1)
                logger.warning("Request failed for %s (attempt %d/%d): %s; retrying in %.1fs", endpoint, attempt, self.max_attempts, exc, wait)
                time.sleep(wait)

        raise last_error or RuntimeError(f"Failed to fetch {endpoint}")

    def get_player_game_logs(
        self,
        player_id: int,
        season: str,
        season_type: str = "Regular Season",
        measure_type: str = "Base",
    ) -> Dict[str, Any]:
        return self._make_request(
            "playergamelogs",
            {
                "DateFrom": "", "DateTo": "", "GameSegment": "", "LastNGames": "0",
                "LeagueID": "00", "Location": "", "MeasureType": measure_type,
                "Month": "0", "OpponentTeamID": "0", "Outcome": "", "PORound": "0",
                "PaceAdjust": "N", "PerMode": "Totals", "Period": "0",
                "PlayerID": str(player_id), "PlusMinus": "N", "Rank": "N",
                "Season": season, "SeasonSegment": "", "SeasonType": season_type,
                "ShotClockRange": "", "VsConference": "", "VsDivision": "",
            },
        )

    def get_team_game_logs(
        self,
        team_id: int,
        season: str,
        season_type: str = "Playoffs",
        measure_type: str = "Base",
    ) -> Dict[str, Any]:
        return self._make_request(
            "teamgamelogs",
            {
                "DateFrom": "", "DateTo": "", "GameSegment": "", "LastNGames": "0",
                "LeagueID": "00", "Location": "", "MeasureType": measure_type,
                "Month": "0", "OpponentTeamID": "0", "Outcome": "", "PORound": "0",
                "PaceAdjust": "N", "PerMode": "Totals", "Period": "0",
                "Season": season, "SeasonSegment": "", "SeasonType": season_type,
                "ShotClockRange": "", "TeamID": str(team_id),
                "VsConference": "", "VsDivision": "",
            },
        )

    def get_play_by_play(
        self,
        game_id: str | int,
        start_period: int = 0,
        end_period: int = 10,
    ) -> Dict[str, Any]:
        gid = normalize_game_id(game_id)
        return self._make_request(
            "playbyplayv3",
            {"GameID": gid, "StartPeriod": start_period, "EndPeriod": end_period},
        )

    def get_video_events(self, game_id: str, event_id: str | int) -> Dict[str, Any]:
        """Fetch video clip URLs for a specific game event."""
        gid = normalize_game_id(game_id)
        return self._make_request(
            "videoeventsasset",
            {"GameEventID": str(event_id), "GameID": gid},
        )


def create_client() -> NBAStatsClient:
    return NBAStatsClient()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = create_client()
    harden_id = config.PLAYERS["James Harden"]["nba_id"]
    resp = client.get_player_game_logs(harden_id, "2023-24", "Regular Season", "Base")
    rows = result_set_to_records(resp)
    print(f"Harden 2023-24 RS games: {len(rows)}")
    if rows:
        print("Sample keys:", list(rows[0].keys())[:8])
