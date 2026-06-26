"""Build a clip manifest of shooting fouls drawn by a target player.

For each game, fetches PBP, finds shooting fouls drawn by the player, and
fetches video clip URLs from the NBA video API.

Usage:
    python src/foul_scraper.py --player "James Harden" --season 2019-20 --games 5
    python src/foul_scraper.py --player "James Harden" --season 2019-20 --games 5 --season-type Playoffs

Output: data/processed/manifest_{player_slug}_{rs|po}.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from src.nba_client import NBAStatsClient, playbyplay_actions, result_set_to_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_shooting_fouls_drawn(
    actions: List[Dict[str, Any]],
    target_person_id: int,
) -> List[Dict[str, Any]]:
    """Return shooting-foul actions where target_person_id drew the foul.

    The foul event's personId is the *committer*. The fouled player is
    identified by the personId on the immediately subsequent Free Throw events.
    """
    drawn = []
    for i, a in enumerate(actions):
        if a.get("actionType") != "Foul" or a.get("subType") != "Shooting":
            continue
        for j in range(i + 1, min(i + 6, len(actions))):
            na = actions[j]
            if na.get("actionType") == "Free Throw":
                if na.get("personId") == target_person_id:
                    drawn.append(a)
                break
            if na.get("actionType") not in ("Free Throw",):
                break
    return drawn


def get_opponent_tricode(actions: List[Dict[str, Any]], target_team_id: int) -> str:
    for a in actions:
        tid = a.get("teamId", 0)
        if tid and int(tid) != target_team_id and a.get("teamTricode"):
            return a["teamTricode"]
    return "UNK"


def get_target_team_id(actions: List[Dict[str, Any]], target_person_id: int) -> int:
    for a in actions:
        if a.get("personId") == target_person_id and a.get("teamId"):
            return int(a["teamId"])
    return 0


def fetch_video_for_event(
    client: NBAStatsClient,
    game_id: str,
    action_number: int,
) -> Optional[Dict[str, Any]]:
    try:
        resp = client.get_video_events(game_id, action_number)
        meta = resp.get("resultSets", {}).get("Meta", {})
        urls = meta.get("videoUrls", [])
        if urls:
            return urls[0]
    except Exception as e:
        logger.warning("Video fetch failed for %s event %s: %s", game_id, action_number, e)
    return None


def build_clip_entry(
    foul_action: Dict[str, Any],
    game_id: str,
    opponent_tricode: str,
    video_info: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not video_info or not video_info.get("murl"):
        return None
    return {
        "game_id": game_id,
        "event_id": foul_action.get("actionNumber", 0),
        "period": foul_action.get("period", 0),
        "clock": foul_action.get("clock", ""),
        "description": foul_action.get("description", ""),
        "opponent": opponent_tricode,
        "score_home": foul_action.get("scoreHome", ""),
        "score_away": foul_action.get("scoreAway", ""),
        "video_url_960": video_info.get("murl", ""),
        "video_url_320": video_info.get("surl", ""),
        "video_url_720": video_info.get("lurl", ""),
        "thumbnail_url": video_info.get("mth", ""),
        "duration_ms": video_info.get("sdur", 0),
    }


def build_manifest(
    player_name: str,
    season: str,
    num_games: Optional[int] = None,
    season_type: str = "Regular Season",
) -> List[Dict[str, Any]]:
    player_info = config.PLAYERS.get(player_name)
    if not player_info:
        raise ValueError(f"Player '{player_name}' not in config.PLAYERS")

    player_id = player_info["nba_id"]
    client = NBAStatsClient()

    logger.info("Fetching game logs for %s (%s %s)", player_name, season, season_type)
    resp = client.get_player_game_logs(player_id, season, season_type, "Base")
    games = result_set_to_records(resp)
    logger.info("Found %d games", len(games))

    if num_games:
        games = games[:num_games]

    clips: List[Dict[str, Any]] = []
    for gi, game in enumerate(games):
        game_id = game["GAME_ID"]
        matchup = game.get("MATCHUP", "?")
        logger.info("Game %d/%d: %s (%s)", gi + 1, len(games), game_id, matchup)

        pbp_resp = client.get_play_by_play(game_id)
        actions = playbyplay_actions(pbp_resp)

        target_team_id = get_target_team_id(actions, player_id)
        opponent_tricode = get_opponent_tricode(actions, target_team_id)

        fouls_drawn = find_shooting_fouls_drawn(actions, player_id)
        logger.info("  %d shooting fouls drawn", len(fouls_drawn))

        for foul in fouls_drawn:
            action_number = foul.get("actionNumber", 0)
            video_info = fetch_video_for_event(client, game_id, action_number)
            entry = build_clip_entry(foul, game_id, opponent_tricode, video_info)
            if entry:
                clips.append(entry)
                logger.info("  Clip: event %d — %s", action_number, foul.get("description", "")[:50])
            else:
                logger.info("  Skipped event %d (no video)", action_number)

    return clips


def main():
    parser = argparse.ArgumentParser(description="Build shooting-foul clip manifest")
    parser.add_argument("--player", required=True, help="Player name (must match config.py)")
    parser.add_argument("--season", required=True, help="Season string (e.g. 2019-20)")
    parser.add_argument("--games", type=int, default=None, help="Limit to N games")
    parser.add_argument("--season-type", default="Regular Season", help="Regular Season or Playoffs")
    args = parser.parse_args()

    clips = build_manifest(args.player, args.season, args.games, args.season_type)

    slug = config.player_slug(args.player)
    suffix = "_po" if args.season_type == "Playoffs" else "_rs"
    out_path = config.PROCESSED_DIR / f"manifest_{slug}{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "player": args.player,
        "season": args.season,
        "season_type": args.season_type,
        "num_games_requested": args.games,
        "num_clips": len(clips),
        "clips": clips,
    }

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Wrote %d clips to %s", len(clips), out_path)


if __name__ == "__main__":
    main()
