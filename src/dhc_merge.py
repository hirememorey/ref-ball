r"""Step 7: does-harden-choke × ref-ball merge.

Tests whether playoff FTA shifts for high-FTA players are crew-mediated —
i.e., whether suppressor-heavy crews appear more often in playoff games
and especially in games where DHC target players hit floor-level performances.

Three analyses:
  A. Crew composition: RS vs PO — do playoff games get more suppressive crews?
  B. Floor game crew: floor vs non-floor PO — are floor games more suppressor-heavy?
  C. Player-crew predicted suppression vs actual FTA deviation (player-game level).

Coverage constraint: ref-ball crew data is 100% complete from 2014-15 onward.
Seasons before 2014-15 are excluded from all analyses.

Usage:
    PYTHONPATH=. .venv/bin/python src/dhc_merge.py build
    PYTHONPATH=. .venv/bin/python src/dhc_merge.py summary
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DHC_ANALYSIS_PATH = config.PROCESSED_DIR / "analysis_table.csv"

CREW_PATH = config.CREW_ASSIGNMENTS_PATH
PROFILES_PATH = config.PROCESSED_DIR / "player_official" / "official_calling_profiles.parquet"
ADJ_PATH = config.PROCESSED_DIR / "player_official" / "defensive_adjusted_interactions.parquet"

OUTPUT_DIR = config.PROCESSED_DIR / "model" / "dhc_merge"
GAME_CREW_PATH = OUTPUT_DIR / "game_crew_metrics.parquet"
PLAYER_GAME_PATH = OUTPUT_DIR / "player_game_merged.parquet"
PLAYER_CREW_SUPPRESSION_PATH = OUTPUT_DIR / "player_crew_suppression.parquet"
RESULTS_PATH = OUTPUT_DIR / "dhc_merge_results.parquet"

# Crew coverage starts 2014-15 (official attribution available in PBP from that season)
CREW_COVERAGE_START = "2014-15"

# Minimum minutes played to include a player-game
MIN_MINUTES = 10.0

# Minimum number of adj pairs matched per player-game for player-specific analysis
MIN_OFFICIALS_MATCHED = 1


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _load_dhc() -> pd.DataFrame:
    """Load DHC analysis table, filter to crew-coverage window, add derived cols."""
    logger.info("Loading DHC analysis_table from %s", DHC_ANALYSIS_PATH)
    dhc = pd.read_csv(DHC_ANALYSIS_PATH, low_memory=False)

    # Normalize game_id to 10-char zero-padded string (matches crew_assignments)
    dhc["game_id_str"] = dhc["game_id"].astype(str).str.zfill(10)

    # Filter to crew-coverage window
    before = len(dhc)
    dhc = dhc[dhc["season"] >= CREW_COVERAGE_START].copy()
    logger.info(
        "DHC rows: %d total → %d after ≥%s filter", before, len(dhc), CREW_COVERAGE_START
    )

    # Compute FTA/36 (normalize for playing time)
    dhc["fta36"] = np.where(dhc["min"] > 0, dhc["fta"] / dhc["min"] * 36.0, np.nan)

    # Min-minutes filter
    dhc = dhc[dhc["min"] >= MIN_MINUTES].copy()
    logger.info("DHC rows after min≥%g filter: %d", MIN_MINUTES, len(dhc))

    # Compute per-player RS FTA36 baseline (from RS games in coverage window)
    rs = dhc[dhc["is_playoff"] == False]  # noqa: E712
    rs_baselines = (
        rs.groupby("player_name")
        .agg(rs_fta36_mean=("fta36", "mean"), rs_n_games=("game_id_str", "count"))
        .reset_index()
    )
    dhc = dhc.merge(rs_baselines, on="player_name", how="left")
    dhc["fta36_delta"] = dhc["fta36"] - dhc["rs_fta36_mean"]

    return dhc


def _build_game_crew_metrics() -> pd.DataFrame:
    """Join crew assignments → official profiles → aggregate to game-level crew metrics."""
    logger.info("Building game-level crew metrics")
    crew = pd.read_parquet(CREW_PATH)
    ocp = pd.read_parquet(PROFILES_PATH)

    # Cast official_id to float for join (ocp stores as float)
    crew["official_id_f"] = crew["official_id"].astype(float)

    crew_prof = crew.merge(
        ocp[["official_id", "suppressor_score", "mean_adj_fta36_delta", "sf_per_game"]],
        left_on="official_id_f",
        right_on="official_id",
        how="left",
        suffixes=("_crew", "_ocp"),
    )

    match_rate = crew_prof["suppressor_score"].notna().mean()
    logger.info("crew→profile match rate: %.1f%%", match_rate * 100)

    game_crew = (
        crew_prof.groupby("game_id")
        .agg(
            crew_mean_suppressor_score=("suppressor_score", "mean"),
            crew_mean_adj_delta=("mean_adj_fta36_delta", "mean"),
            crew_min_adj_delta=("mean_adj_fta36_delta", "min"),  # most suppressive official
            crew_mean_sf_per_game=("sf_per_game", "mean"),
            crew_n_profiled=("suppressor_score", "count"),  # how many of 3 officials have profiles
        )
        .reset_index()
    )

    logger.info("Game-level crew metrics built: %d games", len(game_crew))
    return game_crew, crew


def _build_player_crew_suppression(
    dhc_po: pd.DataFrame, crew: pd.DataFrame, adj: pd.DataFrame
) -> pd.DataFrame:
    """
    For each player-game, compute predicted crew suppression = mean of the 3 officials'
    player-specific defense-adjusted FTA36 delta.

    Returns a DataFrame with one row per player-game that has ≥1 official matched.
    """
    logger.info("Building player-specific crew suppression (vectorized)")

    # Keep only the columns we need from DHC
    pg = dhc_po[["player_name", "game_id_str"]].drop_duplicates().copy()

    # Expand: one row per (player_game × official in crew)
    crew_subset = crew[["game_id", "official_name"]].copy()
    pg_officials = pg.merge(
        crew_subset, left_on="game_id_str", right_on="game_id", how="inner"
    )

    # Join player-specific adj delta on (player_name, official_name)
    pg_officials = pg_officials.merge(
        adj[["player_name", "official_name", "defense_adjusted_fta36_delta"]],
        on=["player_name", "official_name"],
        how="left",
    )

    # Aggregate back to player-game
    player_suppression = (
        pg_officials.groupby(["player_name", "game_id_str"])
        .agg(
            player_predicted_suppression=("defense_adjusted_fta36_delta", "mean"),
            n_officials_matched=("defense_adjusted_fta36_delta", "count"),
        )
        .reset_index()
    )

    logger.info(
        "Player-crew suppression: %d player-games, %d with ≥%d official(s) matched",
        len(player_suppression),
        (player_suppression["n_officials_matched"] >= MIN_OFFICIALS_MATCHED).sum(),
        MIN_OFFICIALS_MATCHED,
    )
    return player_suppression


def build() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dhc = _load_dhc()
    game_crew, crew = _build_game_crew_metrics()
    adj = pd.read_parquet(ADJ_PATH)

    # Join DHC → game-level crew metrics
    merged = dhc.merge(game_crew, left_on="game_id_str", right_on="game_id", how="inner")
    logger.info(
        "DHC→crew merge: %d player-games with crew data (of %d total DHC rows in window)",
        len(merged),
        len(dhc),
    )

    merged.to_parquet(PLAYER_GAME_PATH, index=False)
    logger.info("Saved player-game merged table → %s", PLAYER_GAME_PATH)

    # Build player-specific crew suppression for all merged games
    po_merged = merged[merged["is_playoff"] == True]  # noqa: E712
    player_supp = _build_player_crew_suppression(po_merged, crew, adj)
    player_supp.to_parquet(PLAYER_CREW_SUPPRESSION_PATH, index=False)
    logger.info("Saved player-crew suppression → %s", PLAYER_CREW_SUPPRESSION_PATH)

    # Save game-level crew metrics
    game_crew.to_parquet(GAME_CREW_PATH, index=False)
    logger.info("Saved game-crew metrics → %s", GAME_CREW_PATH)


# ---------------------------------------------------------------------------
# Summary / Analysis
# ---------------------------------------------------------------------------


def _mannwhitney_summary(a: pd.Series, b: pd.Series, label_a: str, label_b: str) -> dict:
    """Run Mann-Whitney U and return a results dict."""
    a_clean = a.dropna()
    b_clean = b.dropna()
    stat, p = stats.mannwhitneyu(a_clean, b_clean, alternative="two-sided")
    return {
        f"{label_a}_n": len(a_clean),
        f"{label_a}_mean": a_clean.mean(),
        f"{label_a}_median": a_clean.median(),
        f"{label_b}_n": len(b_clean),
        f"{label_b}_mean": b_clean.mean(),
        f"{label_b}_median": b_clean.median(),
        "mannwhitney_U": stat,
        "p_value": p,
    }


def _spearman_summary(x: pd.Series, y: pd.Series) -> dict:
    mask = x.notna() & y.notna()
    x_clean, y_clean = x[mask], y[mask]
    r, p = stats.spearmanr(x_clean, y_clean)
    return {"n": len(x_clean), "spearman_r": r, "p_value": p}


def summary() -> None:
    if not PLAYER_GAME_PATH.exists():
        print("No merged data found. Run 'build' first.")
        sys.exit(1)

    merged = pd.read_parquet(PLAYER_GAME_PATH)
    player_supp = pd.read_parquet(PLAYER_CREW_SUPPRESSION_PATH)

    rs = merged[merged["is_playoff"] == False]  # noqa: E712
    po = merged[merged["is_playoff"] == True]  # noqa: E712

    print("\n" + "=" * 70)
    print("Step 7: does-harden-choke × ref-ball Merge Results")
    print("=" * 70)
    print(f"Coverage window: {CREW_COVERAGE_START} onward (crew attribution available)")
    print(f"Total player-games with crew data: {len(merged):,}")
    print(f"  Regular season: {len(rs):,}")
    print(f"  Playoff: {len(po):,}")
    print(f"  Players represented: {merged['player_name'].nunique()}")
    print(f"  Games represented: {merged['game_id_str'].nunique():,}")

    # ------------------------------------------------------------------
    # Analysis A: RS vs PO crew suppression composition
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ANALYSIS A: Crew Composition — Regular Season vs Playoffs")
    print("-" * 70)
    print("Hypothesis: Playoff games have more suppressive crews than RS games.")
    print("Metric: crew_mean_suppressor_score (fraction of target players suppressed)")

    # Deduplicate to game level (one row per game)
    rs_games = rs.drop_duplicates("game_id_str")
    po_games = po.drop_duplicates("game_id_str")

    res_a = _mannwhitney_summary(
        rs_games["crew_mean_suppressor_score"],
        po_games["crew_mean_suppressor_score"],
        "RS",
        "PO",
    )
    print(f"\n  RS games: n={res_a['RS_n']:,}, mean={res_a['RS_mean']:.3f}, median={res_a['RS_median']:.3f}")
    print(f"  PO games: n={res_a['PO_n']:,}, mean={res_a['PO_mean']:.3f}, median={res_a['PO_median']:.3f}")
    print(f"  Mann-Whitney U={res_a['mannwhitney_U']:.0f}, p={res_a['p_value']:.4f}")

    direction = "more suppressive" if res_a["PO_mean"] > res_a["RS_mean"] else "less suppressive"
    sig = res_a["p_value"] < 0.05
    print(f"  → Playoff crews are {direction} than RS crews (p{'<0.05' if sig else '>0.05'}: {'significant' if sig else 'not significant'})")

    # Also test crew_mean_adj_delta
    res_a2 = _mannwhitney_summary(
        rs_games["crew_mean_adj_delta"],
        po_games["crew_mean_adj_delta"],
        "RS",
        "PO",
    )
    print(f"\n  [crew_mean_adj_delta check]")
    print(f"  RS mean={res_a2['RS_mean']:.3f}, PO mean={res_a2['PO_mean']:.3f}, p={res_a2['p_value']:.4f}")

    # ------------------------------------------------------------------
    # Analysis B: Floor vs non-floor playoff crew composition
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ANALYSIS B: Floor Game Crew Composition — Floor vs Non-Floor Playoff")
    print("-" * 70)
    print("Hypothesis: Games where players hit floor-level performance have more suppressive crews.")
    print("Metric: crew_mean_suppressor_score")

    # is_floor_primary: True = game where player crashed vs RS baseline
    po_floor = po[po["is_floor_primary"] == True]
    po_nonfloor = po[po["is_floor_primary"] == False]

    # Game-level (deduplicate) — note: a game can be floor for one player but not another
    # Use player-game level to preserve floor/nonfloor per player
    print(f"\n  (Player-game level: floor={len(po_floor)}, non-floor={len(po_nonfloor)})")

    res_b = _mannwhitney_summary(
        po_floor["crew_mean_suppressor_score"],
        po_nonfloor["crew_mean_suppressor_score"],
        "floor",
        "nonfloor",
    )
    print(f"  Floor PO:     n={res_b['floor_n']:,}, mean={res_b['floor_mean']:.3f}, median={res_b['floor_median']:.3f}")
    print(f"  Non-floor PO: n={res_b['nonfloor_n']:,}, mean={res_b['nonfloor_mean']:.3f}, median={res_b['nonfloor_median']:.3f}")
    print(f"  Mann-Whitney U={res_b['mannwhitney_U']:.0f}, p={res_b['p_value']:.4f}")

    direction_b = "more" if res_b["floor_mean"] > res_b["nonfloor_mean"] else "less"
    sig_b = res_b["p_value"] < 0.05
    print(f"  → Floor games have {direction_b} suppressive crews (p{'<0.05' if sig_b else '>0.05'}: {'significant' if sig_b else 'not significant'})")

    # FTA delta comparison
    res_b_fta = _mannwhitney_summary(
        po_floor["fta36_delta"],
        po_nonfloor["fta36_delta"],
        "floor",
        "nonfloor",
    )
    print(f"\n  [Actual FTA/36 delta check: floor vs non-floor]")
    print(f"  Floor mean FTA36 delta={res_b_fta['floor_mean']:.3f}, Non-floor mean={res_b_fta['nonfloor_mean']:.3f}, p={res_b_fta['p_value']:.4f}")

    # By player breakdown
    if len(po_floor) > 0:
        print("\n  Floor game crew suppression by player:")
        player_floor = (
            po_floor.groupby("player_name")
            .agg(
                n_floor=("game_id_str", "count"),
                crew_suppressor_mean=("crew_mean_suppressor_score", "mean"),
                fta36_delta_mean=("fta36_delta", "mean"),
            )
            .reset_index()
            .sort_values("crew_suppressor_mean", ascending=False)
        )
        for _, row in player_floor.iterrows():
            print(
                f"    {row['player_name']:<30} n={row['n_floor']:>3}, "
                f"crew_suppressor={row['crew_suppressor_mean']:.3f}, "
                f"fta36_delta={row['fta36_delta_mean']:+.2f}"
            )

    # ------------------------------------------------------------------
    # Analysis C: Player-specific predicted crew suppression vs actual FTA delta
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ANALYSIS C: Player-Specific Predicted Crew Suppression vs Actual FTA Delta")
    print("-" * 70)
    print("Hypothesis: Games predicted to be suppressive (from player×official profiles)")
    print("  show lower actual FTA/36 vs RS baseline.")

    # Join player suppression back to po merged table
    po_with_supp = po.merge(player_supp, on=["player_name", "game_id_str"], how="inner")
    po_with_supp = po_with_supp[
        po_with_supp["n_officials_matched"] >= MIN_OFFICIALS_MATCHED
    ].copy()

    print(f"\n  Player-games with ≥{MIN_OFFICIALS_MATCHED} official matched: {len(po_with_supp)}")
    print(f"  Players represented: {po_with_supp['player_name'].nunique()}")

    if len(po_with_supp) > 10:
        res_c = _spearman_summary(
            po_with_supp["player_predicted_suppression"],
            po_with_supp["fta36_delta"],
        )
        print(
            f"\n  NOTE: adj deltas are computed from ALL games (RS + PO). Median PO fraction ≈6%."
        )
        print(f"  This is partial look-ahead; a clean holdout would recompute adj deltas RS-only.")
        print(f"\n  Spearman r={res_c['spearman_r']:.3f}, p={res_c['p_value']:.4f} (n={res_c['n']})")
        sig_c = res_c["p_value"] < 0.05
        correct_direction = res_c["spearman_r"] > 0  # positive is expected: suppressor crew → fewer FTA → both negative
        expected = (
            "Expected direction: POSITIVE "
            "(suppressor crew → negative predicted value → negative actual FTA delta → r>0)"
        )
        print(f"  {expected}")
        direction_c = "positive" if res_c["spearman_r"] > 0 else "negative"
        confirmed = sig_c and correct_direction
        print(f"  → {direction_c} correlation ({'significant' if sig_c else 'not significant'}) "
              f"— {'CONFIRMED' if confirmed else 'NOT confirmed'}")

        # Per-player correlations
        print("\n  Per-player: predicted suppression vs actual FTA delta")
        player_corrs = []
        for player, grp in po_with_supp.groupby("player_name"):
            if len(grp) >= 5:
                r, p = stats.spearmanr(
                    grp["player_predicted_suppression"], grp["fta36_delta"]
                )
                player_corrs.append({"player_name": player, "n": len(grp), "r": r, "p": p})
        if player_corrs:
            pc_df = pd.DataFrame(player_corrs).sort_values("r")
            for _, row in pc_df.iterrows():
                print(
                    f"    {row['player_name']:<30} n={row['n']:>3}, r={row['r']:+.3f}, p={row['p']:.3f}"
                )
    else:
        print("  Insufficient player-games for correlation analysis.")

    # ------------------------------------------------------------------
    # Summary of findings
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY OF FINDINGS")
    print("=" * 70)
    findings = []
    findings.append(
        f"A (RS vs PO crew): PO crews {'MORE' if res_a['PO_mean'] > res_a['RS_mean'] else 'NOT more'} "
        f"suppressive (p={res_a['p_value']:.3f})"
    )
    findings.append(
        f"B (Floor game crew): {'CONFIRMED' if sig_b else 'NOT confirmed'} — "
        f"floor games {'have more' if direction_b == 'more' else 'do NOT have more'} suppressive crews "
        f"(p={res_b['p_value']:.3f})"
    )
    if len(po_with_supp) > 10:
        confirmed_c = sig_c and res_c["spearman_r"] > 0
        findings.append(
            f"C (Player-predicted suppression): r={res_c['spearman_r']:.3f}, "
            f"p={res_c['p_value']:.3f} — "
            f"{'CONFIRMED (correct direction, significant)' if confirmed_c else 'NOT confirmed'}"
        )
    for f in findings:
        print(f"  {f}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "summary"])
    args = parser.parse_args()

    if args.command == "build":
        build()
    elif args.command == "summary":
        summary()


if __name__ == "__main__":
    main()
