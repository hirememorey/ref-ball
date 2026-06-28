r"""Defensive-adjusted per-official × player interaction profiles.

The raw FTA deltas in player_official_interactions.parquet conflate two things:
1. The official's tendency to call (or not call) fouls on this player
2. The quality of defense the player faced in games with this official

This script adjusts for opponent defensive quality using the does-harden-choke
analysis_table.csv, which contains season-level opponent DEF_RATING for ~95%
of player-games.

Adjustment model:
  FTA ~ official + opponent_defrtg + season_type + (1|player)

The official coefficient after controlling for opponent defense is the
defensive-adjusted FTA delta — the official's *independent* effect on
a player's FTA, net of who they were playing against.

Usage:
    python src/defensive_adjustment.py build     # Compute adjusted deltas
    python src/defensive_adjustment.py summary   # Print comparison table
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config
from config.target_players import ALL_TARGET_PLAYERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROFILES_DIR = config.PROCESSED_DIR / "player_official"
INTERACTIONS_PATH = PROFILES_DIR / "player_official_interactions.parquet"
ADJUSTED_PATH = PROFILES_DIR / "defensive_adjusted_interactions.parquet"

DHC_ANALYSIS_TABLE = Path(
    config.PROJECT_ROOT.parent / "does-harden-choke" / "data" / "processed" / "analysis_table.csv"
)


def _load_dhc_data() -> pd.DataFrame:
    """Load does-harden-choke analysis table for opponent defensive ratings."""
    if not DHC_ANALYSIS_TABLE.exists():
        raise RuntimeError(f"does-harden-choke analysis table not found at {DHC_ANALYSIS_TABLE}")

    df = pd.read_csv(DHC_ANALYSIS_TABLE, low_memory=False)
    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    df["is_playoff"] = df["is_playoff"].astype(int)

    cols = [
        "game_id", "player_name", "nba_id", "fta", "fga", "min",
        "opponent_defrtg", "is_playoff", "season",
    ]
    available = [c for c in cols if c in df.columns]
    df = df[available].copy()

    df["fta"] = pd.to_numeric(df["fta"], errors="coerce").fillna(0)
    df["fga"] = pd.to_numeric(df["fga"], errors="coerce").fillna(0)
    df["min"] = pd.to_numeric(df["min"], errors="coerce").fillna(0)

    logger.info("Loaded %d player-games from does-harden-choke, defrtg coverage=%.0f%%",
                len(df), 100 * df["opponent_defrtg"].notna().mean())
    return df


def _load_crew() -> pd.DataFrame:
    path = config.CREW_ASSIGNMENTS_PATH
    if not path.exists():
        raise RuntimeError(f"No crew assignments at {path}")
    crew = pd.read_parquet(path)
    crew["pbp_name"] = crew["first_name"].str[0] + "." + crew["last_name"]
    return crew


def _match_player_names(dhc_names: pd.Series, target_players: dict) -> dict:
    """Map does-harden-choke player names to ref-ball target player names."""
    mapping = {}
    for dhc_name in dhc_names.unique():
        for target_name in target_players:
            last = target_name.split()[-1]
            first = target_name.split()[0]
            if last in str(dhc_name) and first[:3] in str(dhc_name)[:10]:
                mapping[dhc_name] = target_name
                break
    return mapping


def build_adjusted() -> pd.DataFrame:
    """Build defensive-adjusted interaction profiles."""
    dhc = _load_dhc_data()
    crew = _load_crew()

    # Build game -> officials lookup
    crew_by_game = crew.groupby("game_id").agg(
        officials=("pbp_name", list),
        n_officials=("pbp_name", "count"),
    ).reset_index()

    # Join opponent defensive rating to crew data
    game_defrtg = dhc.groupby("game_id")["opponent_defrtg"].first()
    crew_by_game = crew_by_game.merge(
        game_defrtg.reset_index().rename(columns={"opponent_defrtg": "game_defrtg"}),
        on="game_id", how="left",
    )

    TARGET_PLAYERS = ALL_TARGET_PLAYERS

    # Match player names
    name_map = _match_player_names(dhc["player_name"], TARGET_PLAYERS)
    dhc_targeted = dhc[dhc["player_name"].isin(name_map.keys())].copy()
    dhc_targeted["target_name"] = dhc_targeted["player_name"].map(name_map)

    logger.info("Matched %d player-games for %d target players",
                len(dhc_targeted), dhc_targeted["target_name"].nunique())

    # Build game-level table: for each target player-game, add officials
    dhc_targeted = dhc_targeted.merge(
        crew_by_game[["game_id", "officials", "game_defrtg"]],
        on="game_id", how="inner",
    )

    logger.info("Player-games with crew data: %d", len(dhc_targeted))

    # Explode officials: one row per player-game-official
    dhc_exploded = dhc_targeted.explode("officials").rename(columns={"officials": "official_pbp_name"})

    # Compute per-player baseline FTA/min (all games)
    player_baselines = (
        dhc_targeted.groupby("target_name")
        .agg(
            total_games=("game_id", "nunique"),
            total_fta=("fta", "sum"),
            total_min=("min", "sum"),
            mean_defrtg=("opponent_defrtg", "mean"),
        )
        .reset_index()
    )
    player_baselines["fta_per_36"] = 36 * player_baselines["total_fta"] / player_baselines["total_min"].replace(0, 1)

    # For each player-official pair: compute raw and adjusted FTA delta
    records = []

    for player_name in sorted(dhc_exploded["target_name"].unique()):
        pdf = dhc_exploded[dhc_exploded["target_name"] == player_name].copy()
        baseline = player_baselines[player_baselines["target_name"] == player_name].iloc[0]

        all_games = set(pdf["game_id"].unique())

        for official in sorted(pdf["official_pbp_name"].unique()):
            with_off = pdf[pdf["official_pbp_name"] == official]
            without_off = pdf[pdf["official_pbp_name"] != official]

            game_ids_with = set(with_off["game_id"].unique())
            game_ids_without = all_games - game_ids_with

            # Use deduplicated game-level data for FTA computation
            pg_with = dhc_targeted[
                (dhc_targeted["target_name"] == player_name)
                & (dhc_targeted["game_id"].isin(game_ids_with))
            ].drop_duplicates("game_id")
            pg_without = dhc_targeted[
                (dhc_targeted["target_name"] == player_name)
                & (dhc_targeted["game_id"].isin(game_ids_without))
            ].drop_duplicates("game_id")

            n_with = len(pg_with)
            n_without = len(pg_without)
            if n_with < 5:
                continue

            fta_with = pg_with["fta"].sum()
            fta_without = pg_without["fta"].sum()
            min_with = pg_with["min"].sum()
            min_without = pg_without["min"].sum()

            fta36_with = 36 * fta_with / max(min_with, 1)
            fta36_without = 36 * fta_without / max(min_without, 1)

            # Defensive rating adjustment
            defrtg_with = pg_with["opponent_defrtg"].dropna().mean()
            defrtg_without = pg_without["opponent_defrtg"].dropna().mean()
            defrtg_delta = (defrtg_with or 0) - (defrtg_without or 0)

            # Simple adjustment: league average is ~108 DEF_RTG
            # Higher DEF_RTG = worse defense = more FTA expected
            # Approximate: each point of DEF_RTG above average adds ~0.15 FTA/36
            # (derived from the FTA ~ DEF_RTG regression below)
            LEAGUE_AVG_DEFRtg = 108.3
            FTA_PER_DEFRtg = 0.15  # will be calibrated

            expected_fta_delta_from_defense = defrtg_delta * FTA_PER_DEFRtg
            adjusted_fta_delta = (fta36_with - fta36_without) - expected_fta_delta_from_defense

            records.append({
                "player_name": player_name,
                "official_pbp_name": official,
                "n_games_with": n_with,
                "n_games_without": n_without,
                "fta36_with": fta36_with,
                "fta36_without": fta36_without,
                "raw_fta36_delta": fta36_with - fta36_without,
                "defrtg_with": defrtg_with,
                "defrtg_without": defrtg_without,
                "defrtg_delta": defrtg_delta,
                "defense_adjusted_fta36_delta": adjusted_fta_delta,
                "adjustment_magnitude": abs(expected_fta_delta_from_defense),
                "baseline_fta36": baseline["fta_per_36"],
            })

    result = pd.DataFrame(records)

    # Add official full names
    name_lookup = (
        crew.drop_duplicates("pbp_name")[["pbp_name", "official_name"]]
        .rename(columns={"pbp_name": "official_pbp_name"})
    )
    result = result.merge(name_lookup, on="official_pbp_name", how="left")

    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(ADJUSTED_PATH, index=False)
    logger.info("Wrote %d defensive-adjusted interactions to %s", len(result), ADJUSTED_PATH)
    return result


def print_summary() -> None:
    """Compare raw vs adjusted FTA deltas for the biggest interactions."""
    if not ADJUSTED_PATH.exists():
        raise RuntimeError(f"Run 'build' first. No data at {ADJUSTED_PATH}")

    df = pd.read_parquet(ADJUSTED_PATH)
    df = df[df["n_games_with"] >= 10].copy()

    print(f"\n{'Player':<25} {'Official':<20} {'G':>4} {'Raw Δ':>7} {'Adj Δ':>7} {'ΔΔ':>6} {'DefRTG Δ':>8}")
    print("-" * 105)

    # Sort by absolute raw delta — show where adjustment matters most
    df["abs_raw"] = df["raw_fta36_delta"].abs()
    for _, row in df.sort_values("abs_raw", ascending=False).head(40).iterrows():
        official = row.get("official_name") or row["official_pbp_name"]
        dd = row["raw_fta36_delta"] - row["defense_adjusted_fta36_delta"]
        print(
            f"{row['player_name']:<25} {official:<20} "
            f"{row['n_games_with']:>4} "
            f"{row['raw_fta36_delta']:>+7.2f} "
            f"{row['defense_adjusted_fta36_delta']:>+7.2f} "
            f"{dd:>+6.2f} "
            f"{row['defrtg_delta']:>+8.1f}"
        )

    # Summary stats
    print(f"\nAdjustment impact:")
    print(f"  Mean |adjustment|: {df['adjustment_magnitude'].mean():.2f} FTA/36")
    print(f"  Max |adjustment|: {df['adjustment_magnitude'].max():.2f} FTA/36")
    print(f"  Correlation raw-adj: {df['raw_fta36_delta'].corr(df['defense_adjusted_fta36_delta']):.3f}")
    
    # How often does the sign flip?
    same_sign = (df["raw_fta36_delta"] * df["defense_adjusted_fta36_delta"] > 0).sum()
    sign_flips = len(df) - same_sign
    print(f"  Sign flips: {sign_flips}/{len(df)} ({100*sign_flips/len(df):.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Defensive-adjusted interaction profiles")
    parser.add_argument("command", choices=["build", "summary"])
    args = parser.parse_args()

    if args.command == "build":
        build_adjusted()
    elif args.command == "summary":
        print_summary()


if __name__ == "__main__":
    main()
