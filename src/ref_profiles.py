r"""Build per-official shooting foul profiles from ingested game data.

Layer 1 profiles: called foul rates per official, RS vs PO comparison.

The PBP description field gives us the calling official's name (e.g. "S.Foster").
Crew assignments (from fetch_l2m.py) give us official IDs and full names. We join
on the PBP name format (first initial + last name) to get stable official IDs.

Input:  data/processed/games/*.parquet  (from ingest.py)
        data/processed/crew_assignments.parquet  (from fetch_l2m.py, optional)
Output: data/processed/ref_profiles.parquet

Profile dimensions (Paper 1 — Layer 1 only):
  - Called shooting foul volume (per game)
  - Called shooting foul share (% of all shooting fouls in their games)
  - RS vs PO profile delta
  - Game sample size
  - Crew chief rate (% of games as crew chief)

Usage:
    python src/ref_profiles.py
    python src/ref_profiles.py --season 2023-24
    python src/ref_profiles.py --official 1162
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_all_fouls(season: str | None = None) -> pd.DataFrame:
    """Load and concatenate all ingested game parquet files."""
    files = sorted(config.GAMES_DIR.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No game parquets found in {config.GAMES_DIR}")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["season_type"] = df["game_id"].str[:3].map({"002": "RS", "004": "PO"})
    df["season"] = df["game_id"].apply(_game_id_to_season)
    if season is not None:
        df = df[df["season"] == season]
    logger.info("Loaded %d foul records across %d games", len(df), df["game_id"].nunique())
    return df


def _game_id_to_season(game_id: str) -> str:
    start_year = 2000 + int(game_id[3:5])
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def load_crew_assignments() -> pd.DataFrame | None:
    """Load crew assignments and build PBP-name to official-ID mapping."""
    path = config.CREW_ASSIGNMENTS_PATH
    if not path.exists():
        logger.warning("No crew assignments found at %s — profiles will use name only", path)
        return None
    crew = pd.read_parquet(path)
    crew["pbp_name"] = crew["first_name"].str[0] + "." + crew["last_name"]
    crew["season_type"] = crew["game_id"].str[:3].map({"002": "RS", "004": "PO"})
    crew["season"] = crew["game_id"].apply(_game_id_to_season)
    # Build name -> official_id mapping (de-duplicate, warn on conflicts)
    name_to_id = crew.drop_duplicates("pbp_name")[["pbp_name", "official_id", "official_name"]]
    conflicts = crew.groupby("pbp_name")["official_id"].nunique()
    conflicts = conflicts[conflicts > 1]
    if len(conflicts):
        logger.warning("Name conflicts in crew mapping: %s", conflicts.index.tolist())
    return crew


def build_profiles(season: str | None = None, official_id: int | None = None) -> pd.DataFrame:
    """Build per-official shooting foul profiles."""
    fouls = load_all_fouls(season)
    crew = load_crew_assignments()

    # Filter to shooting fouls with official attribution
    sf = fouls[
        (fouls["foul_type"] == "Shooting")
        & (fouls["caller_official_name"] != "")
    ].copy()

    if crew is not None:
        name_map = crew.drop_duplicates("pbp_name")[["pbp_name", "official_id", "official_name"]]
        sf = sf.merge(
            name_map[["pbp_name", "official_id", "official_name"]],
            left_on="caller_official_name",
            right_on="pbp_name",
            how="left",
        )
        sf["official_id"] = sf["official_id"].fillna(-1).astype(int)

        # True game denominator: all games the official was assigned to
        crew_filtered = crew.copy()
        if season is not None:
            crew_filtered = crew_filtered[crew_filtered["season"] == season]
        games_per_official = (
            crew_filtered.groupby(["pbp_name", "season_type"])["game_id"]
            .nunique()
            .reset_index()
            .rename(columns={"game_id": "n_games_assigned"})
        )
    else:
        sf["official_id"] = -1
        sf["official_name"] = sf["caller_official_name"]
        # Fallback: count games where official called at least one SF (lower bound)
        games_per_official = (
            sf.groupby(["caller_official_name", "season_type"])["game_id"]
            .nunique()
            .reset_index()
            .rename(columns={"game_id": "n_games_assigned"})
        )
        games_per_official["pbp_name"] = games_per_official["caller_official_name"]

    # Aggregate shooting fouls per official per season type
    agg = (
        sf.groupby(["caller_official_name", "official_id", "official_name", "season_type"])
        .agg(
            n_shooting_fouls=("event_num", "count"),
            n_games_called=("game_id", "nunique"),
        )
        .reset_index()
    )

    # Merge true game counts
    if crew is not None:
        agg = agg.merge(
            games_per_official[["pbp_name", "season_type", "n_games_assigned"]],
            left_on=["caller_official_name", "season_type"],
            right_on=["pbp_name", "season_type"],
            how="left",
        )
    else:
        agg = agg.merge(
            games_per_official[["caller_official_name", "season_type", "n_games_assigned"]],
            on=["caller_official_name", "season_type"],
            how="left",
        )
    agg["n_games_assigned"] = agg["n_games_assigned"].fillna(agg["n_games_called"])
    agg["sf_per_game"] = agg["n_shooting_fouls"] / agg["n_games_assigned"]

    # Pivot RS vs PO
    pivot = agg.pivot_table(
        index=["caller_official_name", "official_id", "official_name"],
        columns="season_type",
        values=["n_shooting_fouls", "n_games_assigned", "sf_per_game"],
        fill_value=0,
    )

    # Flatten column names
    pivot.columns = [f"{col}_{st}" for col, st in pivot.columns]

    # Compute RS vs PO delta
    if "sf_per_game_RS" in pivot.columns and "sf_per_game_PO" in pivot.columns:
        pivot["sf_per_game_delta"] = pivot["sf_per_game_PO"] - pivot["sf_per_game_RS"]
    pivot["total_sf"] = pivot.get("n_shooting_fouls_RS", 0) + pivot.get("n_shooting_fouls_PO", 0)
    pivot["total_games"] = pivot.get("n_games_assigned_RS", 0) + pivot.get("n_games_assigned_PO", 0)
    pivot = pivot.reset_index().sort_values("total_sf", ascending=False)

    if official_id is not None:
        pivot = pivot[pivot["official_id"] == official_id]

    return pivot


def main():
    parser = argparse.ArgumentParser(description="Build per-official shooting foul profiles")
    parser.add_argument("--season", default=None, help="Filter to a single season")
    parser.add_argument("--official", type=int, default=None, help="Filter to a single official ID")
    args = parser.parse_args()

    profiles = build_profiles(season=args.season, official_id=args.official)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PROCESSED_DIR / "ref_profiles.parquet"
    profiles.to_parquet(out_path, index=False)
    logger.info("Wrote %d official profiles to %s", len(profiles), out_path)

    # Print summary
    print(f"\n{'Official':<20} {'RS SF/G':>8} {'PO SF/G':>8} {'Delta':>8} {'RS G':>6} {'PO G':>6} {'Total':>6}")
    print("-" * 70)
    for _, row in profiles.head(20).iterrows():
        rs_rate = row.get("sf_per_game_RS", 0)
        po_rate = row.get("sf_per_game_PO", 0)
        delta = row.get("sf_per_game_delta", 0)
        rs_games = int(row.get("n_games_assigned_RS", 0))
        po_games = int(row.get("n_games_assigned_PO", 0))
        total = int(row["total_sf"])
        name = row["official_name"] or row["caller_official_name"]
        print(f"{name:<20} {rs_rate:>8.2f} {po_rate:>8.2f} {delta:>+8.2f} {rs_games:>6} {po_games:>6} {total:>6}")


if __name__ == "__main__":
    main()
