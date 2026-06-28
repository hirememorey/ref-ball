r"""Per-official aggregate calling profiles (Step 4).

Aggregates defense-adjusted player×official FTA deltas into one row per
official, then joins Layer 1 shooting-foul rates from ref_profiles.

Output columns:
  - mean_adj_fta36_delta: mean defense-adjusted FTA/36 delta across target players
  - n_players: target players with >=10 games with and without the official
  - suppressor_score: fraction of players with negative adjusted delta
  - rs_po_delta: mean PO adjusted delta minus mean RS adjusted delta (official level)
  - sf_per_game: overall shooting fouls per assigned game (Layer 1)
  - sf_pct_of_fouls: shooting fouls / all fouls in games the official worked

Usage:
    python src/official_calling_profiles.py build
    python src/official_calling_profiles.py summary
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

import config
from config.target_players import ALL_TARGET_PLAYERS
from src.defensive_adjustment import (
    ADJUSTED_PATH,
    _load_crew,
    _load_dhc_data,
    _match_player_names,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROFILES_DIR = config.PROCESSED_DIR / "player_official"
OUTPUT_PATH = PROFILES_DIR / "official_calling_profiles.parquet"
REF_PROFILES_PATH = config.PROCESSED_DIR / "ref_profiles.parquet"

MIN_GAMES_WITH = 10
MIN_GAMES_WITHOUT = 10
MIN_GAMES_RS = 5
MIN_GAMES_PO = 3

FTA_PER_DEFRTG = 0.15


def _adjusted_fta36_delta(
    pg_with: pd.DataFrame,
    pg_without: pd.DataFrame,
    min_games: int,
) -> float | None:
    """Compute defense-adjusted FTA/36 delta for a with/without official split."""
    n_with = len(pg_with)
    n_without = len(pg_without)
    if n_with < min_games or n_without < min_games:
        return None

    fta_with = pg_with["fta"].sum()
    fta_without = pg_without["fta"].sum()
    min_with = pg_with["min"].sum()
    min_without = pg_without["min"].sum()

    fta36_with = 36 * fta_with / max(min_with, 1)
    fta36_without = 36 * fta_without / max(min_without, 1)

    defrtg_with = pg_with["opponent_defrtg"].dropna().mean()
    defrtg_without = pg_without["opponent_defrtg"].dropna().mean()
    defrtg_delta = (defrtg_with or 0) - (defrtg_without or 0)
    expected_fta_delta_from_defense = defrtg_delta * FTA_PER_DEFRTG

    return (fta36_with - fta36_without) - expected_fta_delta_from_defense


def _season_type_adj_delta(
    dhc_targeted: pd.DataFrame,
    dhc_exploded: pd.DataFrame,
    player_name: str,
    official: str,
    season_type: str,
    min_games: int,
) -> float | None:
    """Defense-adjusted FTA/36 delta for one player×official within a season type."""
    pdf = dhc_exploded[
        (dhc_exploded["target_name"] == player_name)
        & (dhc_exploded["season_type"] == season_type)
    ]
    st_games = set(pdf["game_id"].unique())
    if not st_games:
        return None

    game_ids_with = set(pdf[pdf["official_pbp_name"] == official]["game_id"].unique())
    game_ids_without = st_games - game_ids_with

    pg_with = dhc_targeted[
        (dhc_targeted["target_name"] == player_name)
        & (dhc_targeted["game_id"].isin(game_ids_with))
        & (dhc_targeted["season_type"] == season_type)
    ].drop_duplicates("game_id")
    pg_without = dhc_targeted[
        (dhc_targeted["target_name"] == player_name)
        & (dhc_targeted["game_id"].isin(game_ids_without))
        & (dhc_targeted["season_type"] == season_type)
    ].drop_duplicates("game_id")

    return _adjusted_fta36_delta(pg_with, pg_without, min_games)


def _compute_rs_po_deltas() -> pd.DataFrame:
    """Per-official RS vs PO split on defense-adjusted FTA deltas."""
    dhc = _load_dhc_data()
    crew = _load_crew()

    crew_by_game = crew.groupby("game_id").agg(
        officials=("pbp_name", list),
    ).reset_index()

    name_map = _match_player_names(dhc["player_name"], ALL_TARGET_PLAYERS)
    dhc_targeted = dhc[dhc["player_name"].isin(name_map.keys())].copy()
    dhc_targeted["target_name"] = dhc_targeted["player_name"].map(name_map)
    dhc_targeted["season_type"] = dhc_targeted["is_playoff"].map({0: "RS", 1: "PO"})

    dhc_targeted = dhc_targeted.merge(
        crew_by_game[["game_id", "officials"]],
        on="game_id",
        how="inner",
    )
    dhc_exploded = dhc_targeted.explode("officials").rename(columns={"officials": "official_pbp_name"})

    pair_records: list[dict] = []
    season_mins = {"RS": MIN_GAMES_RS, "PO": MIN_GAMES_PO}

    for player_name in sorted(dhc_exploded["target_name"].unique()):
        officials = dhc_exploded.loc[
            dhc_exploded["target_name"] == player_name, "official_pbp_name"
        ].unique()
        for official in sorted(officials):
            row: dict = {
                "player_name": player_name,
                "official_pbp_name": official,
            }
            for season_type, min_games in season_mins.items():
                row[f"adj_fta36_delta_{season_type.lower()}"] = _season_type_adj_delta(
                    dhc_targeted,
                    dhc_exploded,
                    player_name,
                    official,
                    season_type,
                    min_games,
                )
            pair_records.append(row)

    pairs = pd.DataFrame(pair_records)
    if pairs.empty:
        return pairs

    rs_po_agg = []
    for official, pdf in pairs.groupby("official_pbp_name"):
        rs_vals = pdf["adj_fta36_delta_rs"].dropna()
        po_vals = pdf["adj_fta36_delta_po"].dropna()
        if rs_vals.empty or po_vals.empty:
            continue
        mean_rs = rs_vals.mean()
        mean_po = po_vals.mean()
        rs_po_agg.append({
            "official_pbp_name": official,
            "mean_adj_fta36_delta_rs": mean_rs,
            "mean_adj_fta36_delta_po": mean_po,
            "rs_po_delta": mean_po - mean_rs,
            "n_players_rs": len(rs_vals),
            "n_players_po": len(po_vals),
        })

    return pd.DataFrame(rs_po_agg)


def _compute_sf_pct_of_fouls() -> pd.DataFrame:
    """Shooting fouls / all fouls in games each official worked."""
    crew = _load_crew()
    files = sorted(config.GAMES_DIR.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No game parquets in {config.GAMES_DIR}")

    fouls = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    fouls["is_shooting"] = fouls["foul_type"] == "Shooting"

    crew_games = crew[["game_id", "pbp_name"]].drop_duplicates()
    fouls_by_game = (
        fouls.groupby("game_id")
        .agg(
            n_fouls=("event_num", "count"),
            n_shooting_fouls=("is_shooting", "sum"),
        )
        .reset_index()
    )

    official_games = crew_games.merge(fouls_by_game, on="game_id", how="inner")
    agg = (
        official_games.groupby("pbp_name")
        .agg(
            n_games=("game_id", "nunique"),
            n_fouls=("n_fouls", "sum"),
            n_shooting_fouls=("n_shooting_fouls", "sum"),
        )
        .reset_index()
        .rename(columns={"pbp_name": "official_pbp_name"})
    )
    agg["sf_pct_of_fouls"] = agg["n_shooting_fouls"] / agg["n_fouls"].replace(0, np.nan)
    return agg[["official_pbp_name", "sf_pct_of_fouls", "n_games"]]


def build_official_profiles() -> pd.DataFrame:
    """Aggregate defense-adjusted interactions into per-official profiles."""
    if not ADJUSTED_PATH.exists():
        raise RuntimeError(f"No adjusted interactions at {ADJUSTED_PATH}. Run defensive_adjustment build first.")
    if not REF_PROFILES_PATH.exists():
        raise RuntimeError(f"No Layer 1 profiles at {REF_PROFILES_PATH}. Run ref_profiles first.")

    adjusted = pd.read_parquet(ADJUSTED_PATH)
    qualified = adjusted[
        (adjusted["n_games_with"] >= MIN_GAMES_WITH)
        & (adjusted["n_games_without"] >= MIN_GAMES_WITHOUT)
    ].copy()

    logger.info(
        "Aggregating %d qualified player×official pairs (>= %d games each side)",
        len(qualified),
        MIN_GAMES_WITH,
    )

    agg = (
        qualified.groupby(["official_pbp_name", "official_name"], dropna=False)
        .agg(
            mean_adj_fta36_delta=("defense_adjusted_fta36_delta", "mean"),
            std_adj_fta36_delta=("defense_adjusted_fta36_delta", "std"),
            n_players=("player_name", "nunique"),
            n_pairs=("player_name", "count"),
            suppressor_score=(
                "defense_adjusted_fta36_delta",
                lambda s: (s < 0).mean(),
            ),
            mean_raw_fta36_delta=("raw_fta36_delta", "mean"),
        )
        .reset_index()
    )

    rs_po = _compute_rs_po_deltas()
    if not rs_po.empty:
        agg = agg.merge(rs_po, on="official_pbp_name", how="left")
    else:
        agg["rs_po_delta"] = np.nan
        agg["mean_adj_fta36_delta_rs"] = np.nan
        agg["mean_adj_fta36_delta_po"] = np.nan
        agg["n_players_rs"] = 0
        agg["n_players_po"] = 0

    ref = pd.read_parquet(REF_PROFILES_PATH)
    ref = ref.rename(columns={"caller_official_name": "official_pbp_name"})
    ref["sf_per_game"] = ref["total_sf"] / ref["total_games"].replace(0, np.nan)
    ref_cols = [
        "official_pbp_name",
        "official_id",
        "sf_per_game",
        "sf_per_game_RS",
        "sf_per_game_PO",
        "sf_per_game_delta",
        "total_games",
        "total_sf",
    ]
    agg = agg.merge(ref[ref_cols], on="official_pbp_name", how="left", suffixes=("", "_ref"))

    if "official_name_ref" in agg.columns:
        agg["official_name"] = agg["official_name"].fillna(agg["official_name_ref"])
        agg = agg.drop(columns=["official_name_ref"])

    sf_pct = _compute_sf_pct_of_fouls()
    agg = agg.merge(sf_pct, on="official_pbp_name", how="left", suffixes=("", "_sf"))

    if "n_games_sf" in agg.columns:
        agg["n_games"] = agg["n_games_sf"].fillna(agg.get("n_games"))
        agg = agg.drop(columns=["n_games_sf"])

    agg = agg.sort_values("mean_adj_fta36_delta")
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Wrote %d official calling profiles to %s", len(agg), OUTPUT_PATH)
    return agg


def print_summary() -> None:
    """Print top suppressors and amplifiers."""
    if not OUTPUT_PATH.exists():
        raise RuntimeError(f"No profiles at {OUTPUT_PATH}. Run 'build' first.")

    df = pd.read_parquet(OUTPUT_PATH)
    df = df[df["n_players"] >= 3].copy()

    print(f"\n{'Official':<22} {'Adj Δ':>7} {'N':>4} {'Supp':>5} {'RS-PO':>7} {'SF/G':>6} {'SF%':>6}")
    print("-" * 72)

    print("\nTop suppressors (lowest mean adj FTA/36 delta, n_players >= 3):")
    for _, row in df.nsmallest(15, "mean_adj_fta36_delta").iterrows():
        name = row.get("official_name") or row["official_pbp_name"]
        print(
            f"{name:<22} {row['mean_adj_fta36_delta']:>+7.2f} "
            f"{int(row['n_players']):>4} {row['suppressor_score']:>5.0%} "
            f"{row.get('rs_po_delta', float('nan')):>+7.2f} "
            f"{row.get('sf_per_game', float('nan')):>6.2f} "
            f"{row.get('sf_pct_of_fouls', float('nan')):>6.1%}"
        )

    print("\nTop amplifiers (highest mean adj FTA/36 delta, n_players >= 3):")
    for _, row in df.nlargest(15, "mean_adj_fta36_delta").iterrows():
        name = row.get("official_name") or row["official_pbp_name"]
        print(
            f"{name:<22} {row['mean_adj_fta36_delta']:>+7.2f} "
            f"{int(row['n_players']):>4} {row['suppressor_score']:>5.0%} "
            f"{row.get('rs_po_delta', float('nan')):>+7.2f} "
            f"{row.get('sf_per_game', float('nan')):>6.2f} "
            f"{row.get('sf_pct_of_fouls', float('nan')):>6.1%}"
        )

    rs_po_cov = df["rs_po_delta"].notna().sum()
    print(f"\nProfile count: {len(df)} officials with >=3 target players")
    print(f"rs_po_delta coverage: {rs_po_cov}/{len(df)} officials")
    print(f"Mean adj delta std across officials: {df['mean_adj_fta36_delta'].std():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-official aggregate calling profiles")
    parser.add_argument("command", choices=["build", "summary"])
    args = parser.parse_args()

    if args.command == "build":
        build_official_profiles()
    elif args.command == "summary":
        print_summary()


if __name__ == "__main__":
    main()
