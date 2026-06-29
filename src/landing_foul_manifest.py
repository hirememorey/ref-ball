"""Build a clip manifest of 3-point shooting fouls from ingested PBP data.

Scans local play-by-play JSON for shooting fouls followed by exactly 3 free
throw attempts (3-point shooting fouls), samples candidates, and fetches video
URLs from the NBA Stats API.

Usage:
    python src/landing_foul_manifest.py --clips 100
    python src/landing_foul_manifest.py --clips 100 --min-season 22
    python src/landing_foul_manifest.py --clips 50 --seed 7 --dry-run

Output: data/processed/landing_foul_manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from src.foul_type_scraper import build_clip_entry, fetch_video_for_event
from src.nba_client import NBAStatsClient, playbyplay_actions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OFFICIAL_PATTERN = re.compile(r"\(([A-Z]\.\s*\w+)\)\s*$")


def season_start_yy(game_id: str) -> int:
    """Return the two-digit season start year from a 10-digit game ID."""
    return int(str(game_id).zfill(10)[3:5])


def parse_official_name(description: str) -> str:
    m = OFFICIAL_PATTERN.search(description or "")
    return m.group(1) if m else ""


def find_three_ft_shooting_fouls(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return shooting fouls followed by exactly 3 free throw attempts."""
    found: List[Dict[str, Any]] = []
    for i, action in enumerate(actions):
        if action.get("actionType") != "Foul" or action.get("subType") != "Shooting":
            continue

        ft_count = 0
        first_ft: Optional[Dict[str, Any]] = None
        for j in range(i + 1, min(i + 8, len(actions))):
            nxt = actions[j]
            if nxt.get("actionType") == "Free Throw":
                ft_count += 1
                if first_ft is None:
                    first_ft = nxt
            elif ft_count > 0:
                break
            else:
                break

        if ft_count != 3 or first_ft is None:
            continue

        found.append(
            {
                "event_id": action.get("actionNumber", 0),
                "period": action.get("period", 0),
                "clock": action.get("clock", ""),
                "description": action.get("description", ""),
                "committing_player_id": action.get("personId", 0),
                "committing_player_name": action.get("playerNameI", action.get("playerName", "")),
                "committing_team_tricode": action.get("teamTricode", ""),
                "fouled_player_id": first_ft.get("personId", 0),
                "fouled_player_name": first_ft.get("playerNameI", first_ft.get("playerName", "")),
                "fouled_team_tricode": first_ft.get("teamTricode", ""),
                "caller_official_name": parse_official_name(action.get("description", "")),
                "video_available": bool(action.get("videoAvailable")),
            }
        )
    return found


def scan_candidates(
    min_season_yy: int = 19,
    require_video: bool = True,
) -> List[Dict[str, Any]]:
    """Scan local PBP JSON and return all 3-FT shooting foul candidates."""
    pbp_dir = config.RAW_PBP_DIR
    if not pbp_dir.exists():
        raise FileNotFoundError(f"PBP directory not found: {pbp_dir}")

    candidates: List[Dict[str, Any]] = []
    for pbp_path in sorted(pbp_dir.glob("*.json")):
        game_id = pbp_path.stem
        if season_start_yy(game_id) < min_season_yy:
            continue

        with open(pbp_path) as f:
            data = json.load(f)

        actions = playbyplay_actions(data)
        for foul in find_three_ft_shooting_fouls(actions):
            if require_video and not foul["video_available"]:
                continue
            foul["game_id"] = game_id
            foul["season_start_yy"] = season_start_yy(game_id)
            candidates.append(foul)

    logger.info("Found %d 3-FT shooting foul candidates (min season 20%02d)", len(candidates), min_season_yy)
    return candidates


def sample_candidates(
    candidates: List[Dict[str, Any]],
    num_clips: int,
    seed: int,
    max_per_game: int = 2,
) -> List[Dict[str, Any]]:
    """Sample candidates with per-game caps and season diversity."""
    if len(candidates) <= num_clips:
        return list(candidates)

    rng = random.Random(seed)
    by_season: Dict[int, List[Dict[str, Any]]] = {}
    for c in candidates:
        by_season.setdefault(c["season_start_yy"], []).append(c)

    seasons = sorted(by_season)
    per_season = max(1, num_clips // len(seasons))
    selected: List[Dict[str, Any]] = []
    per_game: Dict[str, int] = {}

    for season in seasons:
        pool = list(by_season[season])
        rng.shuffle(pool)
        taken = 0
        for c in pool:
            gid = c["game_id"]
            if per_game.get(gid, 0) >= max_per_game:
                continue
            selected.append(c)
            per_game[gid] = per_game.get(gid, 0) + 1
            taken += 1
            if taken >= per_season:
                break

    if len(selected) < num_clips:
        remaining = [c for c in candidates if c not in selected]
        rng.shuffle(remaining)
        for c in remaining:
            gid = c["game_id"]
            if per_game.get(gid, 0) >= max_per_game:
                continue
            selected.append(c)
            per_game[gid] = per_game.get(gid, 0) + 1
            if len(selected) >= num_clips:
                break

    rng.shuffle(selected)
    return selected[:num_clips]


def foul_action_stub(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Build a minimal foul action dict for build_clip_entry()."""
    return {
        "actionNumber": candidate["event_id"],
        "period": candidate["period"],
        "clock": candidate["clock"],
        "description": candidate["description"],
        "scoreHome": "",
        "scoreAway": "",
    }


def fetch_clips(
    candidates: List[Dict[str, Any]],
    client: NBAStatsClient,
) -> List[Dict[str, Any]]:
    """Fetch video URLs and build manifest clip entries."""
    clips: List[Dict[str, Any]] = []
    for i, cand in enumerate(candidates):
        game_id = cand["game_id"]
        event_id = cand["event_id"]
        logger.info(
            "Fetching video %d/%d: %s event %s — %s",
            i + 1,
            len(candidates),
            game_id,
            event_id,
            cand["description"][:60],
        )
        video_info = fetch_video_for_event(client, game_id, event_id)
        entry = build_clip_entry(
            foul_action_stub(cand),
            game_id,
            cand["committing_team_tricode"] or "UNK",
            video_info,
        )
        if not entry:
            logger.info("  Skipped (no video URL)")
            continue

        entry.update(
            {
                "fouled_player_id": cand["fouled_player_id"],
                "fouled_player_name": cand["fouled_player_name"],
                "fouled_team_tricode": cand["fouled_team_tricode"],
                "committing_player_id": cand["committing_player_id"],
                "committing_player_name": cand["committing_player_name"],
                "committing_team_tricode": cand["committing_team_tricode"],
                "caller_official_name": cand["caller_official_name"],
                "season_start_yy": cand["season_start_yy"],
                "enrichment": "3ft_shooting_foul",
            }
        )
        clips.append(entry)
        logger.info("  Clip added")

    return clips


def build_manifest(
    num_clips: int = 100,
    seed: int = 42,
    min_season_yy: int = 19,
    max_per_game: int = 2,
    dry_run: bool = False,
) -> Dict[str, Any]:
    candidates = scan_candidates(min_season_yy=min_season_yy, require_video=True)
    sampled = sample_candidates(candidates, num_clips, seed, max_per_game=max_per_game)
    logger.info("Sampled %d candidates from %d total", len(sampled), len(candidates))

    if dry_run:
        return {
            "manifest_type": "landing_foul_ground_truth",
            "enrichment": "3ft_shooting_foul",
            "min_season_yy": min_season_yy,
            "seed": seed,
            "num_clips_requested": num_clips,
            "num_candidates": len(candidates),
            "num_sampled": len(sampled),
            "num_clips": 0,
            "clips": [],
            "sampled_candidates": sampled,
        }

    client = NBAStatsClient()
    clips = fetch_clips(sampled, client)

    return {
        "manifest_type": "landing_foul_ground_truth",
        "enrichment": "3ft_shooting_foul",
        "description": "3-point shooting fouls (3 FT awarded) for landing foul ground truth",
        "min_season_yy": min_season_yy,
        "seed": seed,
        "num_clips_requested": num_clips,
        "num_candidates": len(candidates),
        "num_sampled": len(sampled),
        "num_clips": len(clips),
        "clips": clips,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build landing foul clip manifest from 3-FT shooting fouls")
    parser.add_argument("--clips", type=int, default=100, help="Number of clips to fetch (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--min-season",
        type=int,
        default=19,
        help="Minimum season start year suffix (19 = 2019-20, 22 = 2022-23)",
    )
    parser.add_argument("--max-per-game", type=int, default=2, help="Max clips per game")
    parser.add_argument("--dry-run", action="store_true", help="Sample only; do not fetch videos")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: data/processed/landing_foul_manifest.json)",
    )
    args = parser.parse_args()

    manifest = build_manifest(
        num_clips=args.clips,
        seed=args.seed,
        min_season_yy=args.min_season,
        max_per_game=args.max_per_game,
        dry_run=args.dry_run,
    )

    out_path = Path(args.output) if args.output else config.PROCESSED_DIR / "landing_foul_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Wrote manifest to %s (%d clips)", out_path, manifest.get("num_clips", 0))


if __name__ == "__main__":
    main()
