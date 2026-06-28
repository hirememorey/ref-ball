r"""L2M validation cross-check (Step 6).

Tests whether player-derived official suppressor metrics align with
league-audited Last Two Minute report outcomes.

Primary hypothesis:
  Officials who suppress target-player FTA in full-game data should show
  higher incorrect-non-call (INC) rates on shooting fouls in L2M games.

Approach:
  1. Filter L2M events to shooting-foul decisions (CC, INC, IC, CNC)
  2. Join crew assignments → attribute game-level L2M counts to each official
  3. Per official: inc_sf_rate = INC / (INC + CC)
  4. Correlate with suppressor_score and mean_adj_fta36_delta

Usage:
    python src/l2m_validation.py build
    python src/l2m_validation.py summary
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import config
from config.target_players import ALL_TARGET_PLAYERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

L2M_PATH = config.L2M_EVENTS_PATH
CREW_PATH = config.CREW_ASSIGNMENTS_PATH
PROFILES_PATH = config.PROCESSED_DIR / "player_official" / "official_calling_profiles.parquet"
ADJ_PATH = config.PROCESSED_DIR / "player_official" / "defensive_adjusted_interactions.parquet"
OUTPUT_DIR = config.PROCESSED_DIR / "model" / "l2m"
OFFICIAL_L2M_PATH = OUTPUT_DIR / "official_l2m_rates.parquet"
VALIDATION_PATH = OUTPUT_DIR / "l2m_validation.parquet"
CORRELATIONS_PATH = OUTPUT_DIR / "l2m_correlations.parquet"
PLAYER_EVENTS_PATH = OUTPUT_DIR / "player_l2m_events.parquet"
PLAYER_CORR_PATH = OUTPUT_DIR / "player_l2m_correlations.parquet"

SF_CALL_TYPE = "Foul: Shooting"
ADJUDICATED = {"CC", "INC", "IC"}
ALL_SF_DECISIONS = {"CC", "INC", "IC", "CNC"}
MIN_CC_INC = 15  # minimum CC+INC for stable INC rate


def _match_target_player(name: str | float) -> str | None:
    """Map L2M player name string to ref-ball target player name."""
    if pd.isna(name) or not name:
        return None
    name_lower = str(name).lower()
    for target in ALL_TARGET_PLAYERS:
        parts = target.lower().split()
        if parts[-1] in name_lower and parts[0] in name_lower:
            return target
    return None


def _load_l2m_shooting_fouls() -> pd.DataFrame:
    """Load L2M shooting-foul events with review decisions."""
    l2m = pd.read_parquet(L2M_PATH)
    sf = l2m[l2m["call_type"] == SF_CALL_TYPE].copy()
    sf = sf[sf["review_decision"].isin(ALL_SF_DECISIONS)].copy()
    sf["game_id"] = sf["game_id"].astype(str)
    logger.info("L2M shooting-foul events: %d across %d games", len(sf), sf["game_id"].nunique())
    return sf


def _load_crew_long() -> pd.DataFrame:
    """Crew assignments in long format: one row per game × official."""
    crew = pd.read_parquet(CREW_PATH)
    crew["pbp_name"] = crew["first_name"].str[0] + "." + crew["last_name"]
    crew["game_id"] = crew["game_id"].astype(str)
    return crew[["game_id", "pbp_name", "official_name", "official_id"]].drop_duplicates()


def build_official_l2m_rates() -> pd.DataFrame:
    """Aggregate L2M shooting-foul outcomes per official via crew assignments."""
    sf = _load_l2m_shooting_fouls()
    crew = _load_crew_long()

    game_counts = (
        sf.groupby(["game_id", "review_decision"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ALL_SF_DECISIONS:
        if col not in game_counts.columns:
            game_counts[col] = 0

    game_counts["n_adjudicated"] = game_counts[list(ADJUDICATED)].sum(axis=1)
    game_counts["n_cc_inc"] = game_counts["CC"] + game_counts["INC"]

    crew_games = crew.merge(game_counts, on="game_id", how="inner")
    logger.info(
        "Matched %d L2M games to crew (%d official-game rows)",
        crew_games["game_id"].nunique(),
        len(crew_games),
    )

    official = (
        crew_games.groupby(["pbp_name", "official_name", "official_id"], as_index=False)
        .agg(
            n_l2m_games=("game_id", "nunique"),
            cc_sf=("CC", "sum"),
            inc_sf=("INC", "sum"),
            ic_sf=("IC", "sum"),
            cnc_sf=("CNC", "sum"),
            n_adjudicated=("n_adjudicated", "sum"),
            n_cc_inc=("n_cc_inc", "sum"),
        )
    )

    official["inc_sf_rate"] = official["inc_sf"] / official["n_cc_inc"].clip(lower=1)
    official["ic_sf_rate"] = official["ic_sf"] / (official["ic_sf"] + official["cc_sf"]).clip(lower=1)
    official["cc_share"] = official["cc_sf"] / official["n_cc_inc"].clip(lower=1)
    official["cnc_rate"] = official["cnc_sf"] / (
        official["cc_sf"] + official["inc_sf"] + official["ic_sf"] + official["cnc_sf"]
    ).clip(lower=1)

    official = official.rename(columns={"pbp_name": "official_pbp_name"})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    official.to_parquet(OFFICIAL_L2M_PATH, index=False)
    logger.info("Wrote official L2M rates for %d officials", len(official))
    return official


def build_validation_table() -> pd.DataFrame:
    """Join L2M rates with official calling profiles and compute correlations."""
    if not OFFICIAL_L2M_PATH.exists():
        build_official_l2m_rates()

    l2m = pd.read_parquet(OFFICIAL_L2M_PATH)
    profiles = pd.read_parquet(PROFILES_PATH)

    merged = profiles.merge(
        l2m,
        on="official_pbp_name",
        how="inner",
        suffixes=("_profile", "_l2m"),
    )
    merged["qualified"] = merged["n_cc_inc"] >= MIN_CC_INC

    corr_rows = []
    metrics = [
        ("suppressor_score", "inc_sf_rate", "Suppressor score vs L2M INC/(INC+CC)"),
        ("mean_adj_fta36_delta", "inc_sf_rate", "Mean adj FTA delta vs L2M INC rate"),
        ("sf_per_game", "inc_sf_rate", "SF/game vs L2M INC rate"),
        ("sf_pct_of_fouls", "inc_sf_rate", "SF% of fouls vs L2M INC rate"),
        ("suppressor_score", "ic_sf_rate", "Suppressor score vs L2M IC rate"),
        ("mean_adj_fta36_delta", "ic_sf_rate", "Mean adj FTA delta vs L2M IC rate"),
        ("suppressor_score", "cnc_rate", "Suppressor score vs L2M CNC share"),
        ("mean_adj_fta36_delta", "cc_share", "Mean adj FTA delta vs L2M CC share"),
    ]

    for x_col, y_col, label in metrics:
        for qualified_only in [True, False]:
            sub = merged[merged["qualified"]] if qualified_only else merged
            sub = sub[[x_col, y_col]].dropna()
            if len(sub) < 10:
                continue
            r, p = stats.pearsonr(sub[x_col], sub[y_col])
            rho, p_rho = stats.spearmanr(sub[x_col], sub[y_col])
            corr_rows.append({
                "comparison": label,
                "x": x_col,
                "y": y_col,
                "qualified_only": qualified_only,
                "n": len(sub),
                "pearson_r": r,
                "pearson_p": p,
                "spearman_rho": rho,
                "spearman_p": p_rho,
            })

    corr_df = pd.DataFrame(corr_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(VALIDATION_PATH, index=False)
    corr_df.to_parquet(CORRELATIONS_PATH, index=False)
    logger.info("Wrote validation table (%d officials matched)", len(merged))
    return merged


def build_player_l2m_validation() -> pd.DataFrame:
    """Event-level validation: crew player×official deltas vs L2M outcomes."""
    sf = _load_l2m_shooting_fouls()
    crew = _load_crew_long()
    adj = pd.read_parquet(ADJ_PATH)
    adj_lookup = adj.set_index(["player_name", "official_pbp_name"])["defense_adjusted_fta36_delta"]
    league_delta = float(adj["defense_adjusted_fta36_delta"].median())

    sf["player_name"] = sf["disadvantaged_player"].apply(_match_target_player)
    sf = sf[sf["player_name"].notna()].copy()

    crew_wide = crew.groupby("game_id")["pbp_name"].apply(list).reset_index(name="officials")
    sf = sf.merge(crew_wide, on="game_id", how="inner")

    records = []
    for _, row in sf.iterrows():
        player = row["player_name"]
        deltas = []
        for off in row["officials"]:
            key = (player, off)
            if key in adj_lookup.index:
                val = adj_lookup.loc[key]
                deltas.append(float(val.iloc[0]) if isinstance(val, pd.Series) else float(val))
            else:
                deltas.append(league_delta)

        records.append({
            "game_id": row["game_id"],
            "player_name": player,
            "review_decision": row["review_decision"],
            "is_inc": int(row["review_decision"] == "INC"),
            "is_cc": int(row["review_decision"] == "CC"),
            "is_ic": int(row["review_decision"] == "IC"),
            "is_adjudicated": int(row["review_decision"] in {"CC", "INC"}),
            "crew_mean_adj_delta": float(np.mean(deltas)),
            "crew_min_adj_delta": float(np.min(deltas)),
            "crew_max_adj_delta": float(np.max(deltas)),
        })

    events = pd.DataFrame(records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events.to_parquet(PLAYER_EVENTS_PATH, index=False)
    logger.info("Built %d target-player L2M events", len(events))

    corr_rows = []
    adjudicated = events[events["is_adjudicated"] == 1].copy()

    for x_col, y_col, label in [
        ("crew_mean_adj_delta", "is_inc", "Crew mean adj Δ vs INC (adjudicated SF)"),
        ("crew_min_adj_delta", "is_inc", "Crew min adj Δ vs INC (most suppressive ref)"),
        ("crew_mean_adj_delta", "is_cc", "Crew mean adj Δ vs CC (correct calls)"),
    ]:
        sub = adjudicated[[x_col, y_col]].dropna()
        if len(sub) < 30:
            continue
        r, p = stats.pearsonr(sub[x_col], sub[y_col])
        rho, p_rho = stats.spearmanr(sub[x_col], sub[y_col])
        corr_rows.append({
            "comparison": label, "x": x_col, "y": y_col, "n": len(sub),
            "pearson_r": r, "pearson_p": p, "spearman_rho": rho, "spearman_p": p_rho,
        })

    po_records = []
    for player in adj["player_name"].unique():
        for off in adj.loc[adj["player_name"] == player, "official_pbp_name"]:
            game_ids = set(crew.loc[crew["pbp_name"] == off, "game_id"])
            pe = events[(events["player_name"] == player) & events["game_id"].isin(game_ids)]
            adj_events = pe[pe["is_adjudicated"] == 1]
            if len(adj_events) < 5:
                continue
            key = (player, off)
            delta = adj_lookup.loc[key] if key in adj_lookup.index else np.nan
            if isinstance(delta, pd.Series):
                delta = delta.iloc[0]
            po_records.append({
                "player_name": player,
                "official_pbp_name": off,
                "adj_fta36_delta": float(delta) if pd.notna(delta) else np.nan,
                "n_l2m_adjudicated": len(adj_events),
                "inc_rate": adj_events["is_inc"].mean(),
            })

    po_df = pd.DataFrame(po_records)
    sub = po_df[["adj_fta36_delta", "inc_rate"]].dropna()
    if len(sub) >= 20:
        r, p = stats.pearsonr(sub["adj_fta36_delta"], sub["inc_rate"])
        rho, p_rho = stats.spearmanr(sub["adj_fta36_delta"], sub["inc_rate"])
        corr_rows.append({
            "comparison": "Player×official adj Δ vs L2M INC rate",
            "x": "adj_fta36_delta", "y": "inc_rate", "n": len(sub),
            "pearson_r": r, "pearson_p": p, "spearman_rho": rho, "spearman_p": p_rho,
        })

    pd.DataFrame(corr_rows).to_parquet(PLAYER_CORR_PATH, index=False)
    logger.info("Wrote player-level L2M correlations")
    return events


def print_summary() -> None:
    """Print L2M validation results."""
    if not VALIDATION_PATH.exists():
        raise RuntimeError(f"No validation at {VALIDATION_PATH}. Run 'build' first.")

    merged = pd.read_parquet(VALIDATION_PATH)
    qualified = merged[merged["qualified"]].copy()

    print("\n  Step 6: L2M Validation Cross-Check")
    print("  " + "=" * 80)
    print(f"  Officials matched: {len(merged)} ({len(qualified)} with >={MIN_CC_INC} CC+INC events)")
    print(f"  L2M games per official (median): {merged['n_l2m_games'].median():.0f}")

    if CORRELATIONS_PATH.exists():
        corr_df = pd.read_parquet(CORRELATIONS_PATH)
        print("\n  Correlations (qualified officials only):")
        print(f"  {'Comparison':<45} {'n':>4} {'Pearson r':>10} {'p':>10} {'Spearman':>10}")
        print("  " + "-" * 85)
        q = corr_df[corr_df["qualified_only"]]
        for _, row in q.iterrows():
            sig = "*" if row["pearson_p"] < 0.05 else " "
            print(
                f"  {row['comparison']:<45} {row['n']:>4} "
                f"{row['pearson_r']:>+10.3f} {row['pearson_p']:>9.4f}{sig} "
                f"{row['spearman_rho']:>+10.3f}"
            )

    print("\n  Top suppressors (player-derived) — L2M shooting foul outcomes:")
    print(f"  {'Official':<22} {'Suppress':>8} {'Adj Δ':>7} {'INC':>5} {'CC':>5} "
          f"{'INC%':>7} {'IC%':>7} {'L2M G':>6}")
    print("  " + "-" * 80)
    top = qualified.sort_values("suppressor_score", ascending=False).head(12)
    for _, row in top.iterrows():
        name = row.get("official_name_l2m") or row.get("official_name") or row["official_pbp_name"]
        print(
            f"  {str(name):<22} {row['suppressor_score']:>8.2f} "
            f"{row['mean_adj_fta36_delta']:>+7.2f} "
            f"{row['inc_sf']:>5.0f} {row['cc_sf']:>5.0f} "
            f"{row['inc_sf_rate']:>7.1%} {row['ic_sf_rate']:>7.1%} "
            f"{row['n_l2m_games']:>6.0f}"
        )

    print("\n  Top amplifiers (low suppressor score):")
    bottom = qualified.sort_values("suppressor_score", ascending=True).head(8)
    for _, row in bottom.iterrows():
        name = row.get("official_name_l2m") or row.get("official_name") or row["official_pbp_name"]
        print(
            f"  {str(name):<22} {row['suppressor_score']:>8.2f} "
            f"{row['mean_adj_fta36_delta']:>+7.2f} "
            f"{row['inc_sf']:>5.0f} {row['cc_sf']:>5.0f} "
            f"{row['inc_sf_rate']:>7.1%} {row['ic_sf_rate']:>7.1%} "
            f"{row['n_l2m_games']:>6.0f}"
        )

    if CORRELATIONS_PATH.exists():
        key = corr_df[
            (corr_df["x"] == "suppressor_score")
            & (corr_df["y"] == "inc_sf_rate")
            & (corr_df["qualified_only"])
        ]
        if not key.empty:
            r = key.iloc[0]["pearson_r"]
            p = key.iloc[0]["pearson_p"]
            print(f"\n  Primary test: suppressor_score vs INC/(INC+CC)")
            print(f"    r = {r:+.3f}, p = {p:.4f}", end="")
            if p < 0.05 and r > 0:
                print(" → suppressor officials have higher L2M missed-foul rates (validated)")
            elif p < 0.05 and r < 0:
                print(" → significant but opposite direction (needs investigation)")
            else:
                print(" → not significant at α=0.05 (player-derived metric not confirmed by L2M)")

    if PLAYER_CORR_PATH.exists():
        pc = pd.read_parquet(PLAYER_CORR_PATH)
        events = pd.read_parquet(PLAYER_EVENTS_PATH) if PLAYER_EVENTS_PATH.exists() else None
        print("\n  Player-conditioned L2M validation (target players as disadvantaged):")
        if events is not None:
            adj_n = events[events["is_adjudicated"] == 1]
            print(f"    Events: {len(events)} total, {len(adj_n)} adjudicated (CC/INC)")
            print(f"    INC rate: {adj_n['is_inc'].mean():.1%}  CC rate: {adj_n['is_cc'].mean():.1%}")
        print(f"  {'Comparison':<45} {'n':>6} {'Pearson r':>10} {'p':>10}")
        print("  " + "-" * 75)
        for _, row in pc.iterrows():
            sig = "*" if row["pearson_p"] < 0.05 else " "
            print(
                f"  {row['comparison']:<45} {row['n']:>6} "
                f"{row['pearson_r']:>+10.3f} {row['pearson_p']:>9.4f}{sig}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="L2M validation cross-check (Step 6)")
    parser.add_argument("command", choices=["build", "summary"])
    args = parser.parse_args()

    if args.command == "build":
        build_official_l2m_rates()
        build_validation_table()
        build_player_l2m_validation()
        print_summary()
    elif args.command == "summary":
        print_summary()


if __name__ == "__main__":
    main()
