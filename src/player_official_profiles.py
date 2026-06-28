r"""Per-official × player shooting foul interaction profiles.

For each target player, compute how their shooting foul / FTA profile
shifts under different officials. This is the core of ref-ball's predictive
thesis: can we predict the officiating environment of a game from the crew?

Pipeline:
  1. Fetch player game logs (FTA, minutes, game IDs) for target players
  2. Load crew assignments and ingested foul data
  3. For each player × official pair, compute interaction metrics
  4. Output per-official player-adjusted profiles

Usage:
    python src/player_official_profiles.py fetch     # Step 1: download game logs
    python src/player_official_profiles.py build      # Steps 2-4: compute profiles
    python src/player_official_profiles.py summary    # Print summary table
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

import config
from config.target_players import ALL_TARGET_PLAYERS
from src.nba_client import NBAStatsClient, result_set_to_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROFILES_DIR = config.PROCESSED_DIR / "player_official"
PLAYER_GAMES_DIR = PROFILES_DIR / "player_games"
INTERACTIONS_PATH = PROFILES_DIR / "player_official_interactions.parquet"
SUMMARY_PATH = PROFILES_DIR / "summary.parquet"

TARGET_PLAYERS = ALL_TARGET_PLAYERS

SEASONS = [config.year_to_season(y) for y in range(2014, 2025)]
SEASON_TYPES = ["Regular Season", "Playoffs"]


def fetch_player_game_logs() -> None:
    """Download player game logs for all target players and seasons."""
    client = NBAStatsClient()
    PLAYER_GAMES_DIR.mkdir(parents=True, exist_ok=True)

    total = len(TARGET_PLAYERS) * len(SEASONS) * len(SEASON_TYPES)
    done = 0
    fetched = 0
    cached = 0
    failed = 0

    for player_name, player_id in TARGET_PLAYERS.items():
        slug = player_name.lower().replace(" ", "_").replace("'", "").replace("-", "")
        out_path = PLAYER_GAMES_DIR / f"{slug}.parquet"

        all_rows: list[dict] = []
        for season in SEASONS:
            for season_type in SEASON_TYPES:
                done += 1
                try:
                    resp = client.get_player_game_logs(player_id, season, season_type)
                    rows = result_set_to_records(resp)
                    if rows:
                        all_rows.extend(rows)
                    fetched += 1
                except Exception as exc:
                    failed += 1
                    logger.warning("  %s %s %s failed: %s", player_name, season, season_type, exc)

                if done % 10 == 0 or done == total:
                    logger.info("  %d/%d fetched=%d failed=%d", done, total, fetched, failed)

        if all_rows:
            df = pd.DataFrame(all_rows)
            df["player_name"] = player_name
            df["nba_id"] = player_id
            df["season_type"] = df["GAME_ID"].str[:3].map({"002": "RS", "004": "PO"})
            df.to_parquet(out_path, index=False)
            logger.info("Wrote %d games for %s → %s", len(df), player_name, out_path)
        else:
            logger.warning("No game logs for %s", player_name)

    logger.info("Done. fetched=%d failed=%d", fetched, failed)


def _load_player_games() -> pd.DataFrame:
    """Load all cached player game logs."""
    files = sorted(PLAYER_GAMES_DIR.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No player game logs in {PLAYER_GAMES_DIR}. Run 'fetch' first.")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    df["game_id"] = df["GAME_ID"].astype(str).str.zfill(10)
    df["fta"] = pd.to_numeric(df.get("FTA", 0), errors="coerce").fillna(0)
    df["min"] = pd.to_numeric(df.get("MIN", 0), errors="coerce").fillna(0)
    df["fga"] = pd.to_numeric(df.get("FGA", 0), errors="coerce").fillna(0)

    df["season"] = df["GAME_ID"].apply(_game_id_to_season)
    df["season_type"] = df["GAME_ID"].str[:3].map({"002": "RS", "004": "PO"})

    logger.info("Loaded %d player-game rows for %d players",
                len(df), df["player_name"].nunique())
    return df


def _game_id_to_season(game_id: str) -> str:
    start_year = 2000 + int(game_id[3:5])
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _load_foul_data() -> pd.DataFrame:
    """Load ingested foul data from game parquets."""
    files = sorted(config.GAMES_DIR.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No game parquets in {config.GAMES_DIR}")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["season_type"] = df["game_id"].str[:3].map({"002": "RS", "004": "PO"})
    df["season"] = df["game_id"].apply(_game_id_to_season)
    logger.info("Loaded %d foul records", len(df))
    return df


def _load_crew() -> pd.DataFrame:
    """Load crew assignments."""
    path = config.CREW_ASSIGNMENTS_PATH
    if not path.exists():
        raise RuntimeError(f"No crew assignments at {path}")
    crew = pd.read_parquet(path)
    crew["pbp_name"] = crew["first_name"].str[0] + "." + crew["last_name"]
    crew["season_type"] = crew["game_id"].str[:3].map({"002": "RS", "004": "PO"})
    crew["season"] = crew["game_id"].apply(_game_id_to_season)
    logger.info("Loaded %d crew assignment rows for %d games",
                len(crew), crew["game_id"].nunique())
    return crew


def build_interactions() -> pd.DataFrame:
    """Build per-official × per-player interaction profiles.

    For each player × official pair:
    - Games they shared (official assigned, player appeared)
    - Shooting fouls called on the player in those games
    - Player's FTA in those games
    - Same metrics for games where the official was NOT assigned (baseline)
    - Delta between official-specific and baseline rates
    """
    player_games = _load_player_games()
    fouls = _load_foul_data()
    crew = _load_crew()

    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    # Build game-level crew lookup: game_id -> list of officials
    crew_by_game = crew.groupby("game_id").agg(
        officials=("pbp_name", list),
        official_ids=("official_id", list),
        official_names=("official_name", list),
        n_officials=("pbp_name", "count"),
    ).reset_index()

    # Shooting fouls per player per game
    sf = fouls[fouls["foul_type"] == "Shooting"].copy()
    sf_per_player_game = (
        sf.groupby(["game_id", "committing_player_id", "committing_player_name"])
        .agg(n_sf=("event_num", "count"))
        .reset_index()
    )

    # Match PBP player names to target players
    name_map = {}
    for player_name in TARGET_PLAYERS:
        parts = player_name.split()
        last = parts[-1]
        first_init = parts[0][0]
        pbp_pattern = f"{first_init}. {last}"
        name_map[pbp_pattern] = player_name
        if len(last) > 8:
            name_map[f"{first_init}. {last[:8]}"] = player_name

    # Build interactions
    records: list[dict] = []

    for player_name, player_id in TARGET_PLAYERS.items():
        slug = player_name.lower().replace(" ", "_").replace("'", "").replace("-", "")
        pg_path = PLAYER_GAMES_DIR / f"{slug}.parquet"
        if not pg_path.exists():
            logger.warning("No game logs for %s — skipping", player_name)
            continue

        pg = pd.read_parquet(pg_path)
        pg["game_id"] = pg["GAME_ID"].astype(str).str.zfill(10)
        pg["fta"] = pd.to_numeric(pg.get("FTA", 0), errors="coerce").fillna(0)

        player_games_list = set(pg["game_id"].unique())
        player_fouls = sf_per_player_game[
            sf_per_player_game["committing_player_name"].str.contains(
                player_name.split()[-1], case=False, na=False
            )
        ]

        # All officials this player has encountered
        player_games_with_crew = crew_by_game[crew_by_game["game_id"].isin(player_games_list)]

        # Get all unique officials across these games
        all_officials: set[str] = set()
        for _, row in player_games_with_crew.iterrows():
            all_officials.update(row["officials"])

        # Baseline: all games for this player
        total_games = len(player_games_list)
        total_fta = pg["fta"].sum()
        total_sf = len(player_fouls)
        baseline_fta_per_game = total_fta / max(total_games, 1)
        baseline_sf_per_game = total_sf / max(total_games, 1)

        for official_pbp_name in sorted(all_officials):
            # Games with this official
            games_with = player_games_with_crew[
                player_games_with_crew["officials"].apply(lambda x: official_pbp_name in x)
            ]
            game_ids_with = set(games_with["game_id"])

            # Games without this official
            game_ids_without = player_games_list - game_ids_with

            # FTA with/without
            fta_with = pg[pg["game_id"].isin(game_ids_with)]["fta"].sum()
            fta_without = pg[pg["game_id"].isin(game_ids_without)]["fta"].sum()
            n_games_with = len(game_ids_with)
            n_games_without = len(game_ids_without)

            # SF with/without
            sf_with = len(player_fouls[player_fouls["game_id"].isin(game_ids_with)])
            sf_without = len(player_fouls[player_fouls["game_id"].isin(game_ids_without)])

            # RS vs PO split
            pg_with_rs = len(pg[pg["game_id"].isin(game_ids_with) & (pg["game_id"].str.startswith("002"))])
            pg_with_po = len(pg[pg["game_id"].isin(game_ids_with) & (pg["game_id"].str.startswith("004"))])

            records.append({
                "player_name": player_name,
                "official_pbp_name": official_pbp_name,
                "n_games_with_official": n_games_with,
                "n_games_without_official": n_games_without,
                "n_games_with_rs": pg_with_rs,
                "n_games_with_po": pg_with_po,
                "fta_with": fta_with,
                "fta_without": fta_without,
                "fta_per_game_with": fta_with / max(n_games_with, 1),
                "fta_per_game_without": fta_without / max(n_games_without, 1),
                "fta_delta": (fta_with / max(n_games_with, 1)) - (fta_without / max(n_games_without, 1)),
                "sf_with": sf_with,
                "sf_without": sf_without,
                "sf_per_game_with": sf_with / max(n_games_with, 1),
                "sf_per_game_without": sf_without / max(n_games_without, 1),
                "sf_delta": (sf_with / max(n_games_with, 1)) - (sf_without / max(n_games_without, 1)),
                "baseline_fta_per_game": baseline_fta_per_game,
                "baseline_sf_per_game": baseline_sf_per_game,
            })

    result = pd.DataFrame(records)

    # Add official full names from crew data
    name_lookup = (
        crew.drop_duplicates("pbp_name")[["pbp_name", "official_name", "official_id"]]
        .rename(columns={"pbp_name": "official_pbp_name"})
    )
    result = result.merge(name_lookup, on="official_pbp_name", how="left")

    result.to_parquet(INTERACTIONS_PATH, index=False)
    logger.info("Wrote %d player-official interactions to %s", len(result), INTERACTIONS_PATH)
    return result


def print_summary() -> None:
    """Print summary of interaction profiles, sorted by largest FTA deltas."""
    if not INTERACTIONS_PATH.exists():
        raise RuntimeError(f"No interactions at {INTERACTIONS_PATH}. Run 'build' first.")

    df = pd.read_parquet(INTERACTIONS_PATH)

    # Filter to meaningful sample sizes
    df = df[(df["n_games_with_official"] >= 10) & (df["n_games_without_official"] >= 10)].copy()

    print(f"\n{'Player':<25} {'Official':<20} {'G':>4} {'FTA/G':>7} {'Base':>7} {'Delta':>7} {'SF/G':>6} {'Base':>6} {'Delta':>6}")
    print("-" * 105)

    # Sort by absolute FTA delta — biggest interactions first
    df["abs_fta_delta"] = df["fta_delta"].abs()
    for _, row in df.sort_values("abs_fta_delta", ascending=False).head(40).iterrows():
        official = row.get("official_name") or row["official_pbp_name"]
        print(
            f"{row['player_name']:<25} {official:<20} "
            f"{row['n_games_with_official']:>4} "
            f"{row['fta_per_game_with']:>7.2f} {row['fta_per_game_without']:>7.2f} "
            f"{row['fta_delta']:>+7.2f} "
            f"{row['sf_per_game_with']:>6.2f} {row['sf_per_game_without']:>6.2f} "
            f"{row['sf_delta']:>+6.2f}"
        )

    # Per-player summary: which officials have the largest FTA effects?
    print(f"\n\nPer-player FTA suppression/amplification (min {10} games together):")
    print("-" * 80)
    for player in sorted(df["player_name"].unique()):
        pdf = df[df["player_name"] == player]
        if pdf.empty:
            continue
        most_supp = pdf.loc[pdf["fta_delta"].idxmin()]
        most_amp = pdf.loc[pdf["fta_delta"].idxmax()]
        supp_name = most_supp.get("official_name") or most_supp["official_pbp_name"]
        amp_name = most_amp.get("official_name") or most_amp["official_pbp_name"]
        print(
            f"  {player:<25} "
            f"Most suppressed: {supp_name} ({most_supp['fta_delta']:+.2f} FTA/G, n={most_supp['n_games_with_official']})  "
            f"Most amplified: {amp_name} ({most_amp['fta_delta']:+.2f} FTA/G, n={most_amp['n_games_with_official']})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-official × player interaction profiles")
    parser.add_argument("command", choices=["fetch", "build", "summary"])
    args = parser.parse_args()

    if args.command == "fetch":
        fetch_player_game_logs()
    elif args.command == "build":
        build_interactions()
    elif args.command == "summary":
        print_summary()


if __name__ == "__main__":
    main()
