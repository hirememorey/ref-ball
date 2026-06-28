r"""Player-level FTA/36 predictions from crew assignment.

Predicts a target player's free-throw rate in a game from the 3-official crew,
using defense-adjusted player×official interaction profiles.

Approach:
  1. Build player-game rows from does-harden-choke + crew assignments
  2. Temporal profiles: for season S, interaction deltas use only seasons < S
  3. Additive model: player baseline + mean official interaction delta
  4. Season holdout: train 2014-22, test 2023-24 + 2024-25

Usage:
    python src/player_crew_predictive_model.py build
    python src/player_crew_predictive_model.py build --static
    python src/player_crew_predictive_model.py summary
    python src/player_crew_predictive_model.py diagnose
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config
from config.target_players import ALL_TARGET_PLAYERS
from src.defensive_adjustment import _load_crew, _load_dhc_data, _match_player_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = config.PROCESSED_DIR / "model" / "player"
DATASET_PATH = MODEL_DIR / "player_game_crew_dataset.parquet"
DATASET_STATIC_PATH = MODEL_DIR / "player_game_crew_dataset_static.parquet"
TEMPORAL_INTERACTIONS_PATH = MODEL_DIR / "temporal_player_official_deltas.parquet"
STATIC_INTERACTIONS_PATH = MODEL_DIR / "static_player_official_deltas.parquet"
EVALUATION_PATH = MODEL_DIR / "player_evaluation.parquet"
EVALUATION_STATIC_PATH = MODEL_DIR / "player_evaluation_static.parquet"
PREDICTIONS_PATH = MODEL_DIR / "player_predictions.parquet"

TRAIN_SEASONS = [f"{y}-{(y+1)%100:02d}" for y in range(2014, 2023)]
TEST_SEASONS = ["2023-24", "2024-25"]
MODEL_SEASONS = TRAIN_SEASONS + TEST_SEASONS
MIN_MINUTES = 10
MIN_GAMES_WITH = 5
FTA_PER_DEFRTG = 0.15


def _game_id_to_season(game_id: str) -> str:
    start_year = 2000 + int(game_id[3:5])
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _season_sort_key(season: str) -> int:
    return int(season.split("-")[0])


def _adjusted_fta36_delta(pg_with: pd.DataFrame, pg_without: pd.DataFrame, min_games: int) -> float | None:
    """Defense-adjusted FTA/36 delta for a with/without official split."""
    n_with = len(pg_with)
    n_without = len(pg_without)
    if n_with < min_games or n_without < min_games:
        return None

    fta36_with = 36 * pg_with["fta"].sum() / max(pg_with["min"].sum(), 1)
    fta36_without = 36 * pg_without["fta"].sum() / max(pg_without["min"].sum(), 1)

    defrtg_with = pg_with["opponent_defrtg"].dropna().mean()
    defrtg_without = pg_without["opponent_defrtg"].dropna().mean()
    defrtg_delta = (defrtg_with or 0) - (defrtg_without or 0)
    expected_fta_delta = defrtg_delta * FTA_PER_DEFRTG

    return (fta36_with - fta36_without) - expected_fta_delta


def _load_player_games_with_crew() -> pd.DataFrame:
    """Target-player game logs joined to 3-official crew assignments."""
    dhc = _load_dhc_data()
    crew = _load_crew()

    name_map = _match_player_names(dhc["player_name"], ALL_TARGET_PLAYERS)
    dhc = dhc[dhc["player_name"].isin(name_map.keys())].copy()
    dhc["player_name"] = dhc["player_name"].map(name_map)
    dhc["season"] = dhc["game_id"].apply(_game_id_to_season)
    dhc["season_type"] = np.where(dhc["is_playoff"].astype(int) == 1, "PO", "RS")
    dhc["fta36"] = 36 * dhc["fta"] / dhc["min"].clip(lower=1)

    crew_by_game = (
        crew.groupby("game_id")
        .agg(officials=("pbp_name", list), n_officials=("pbp_name", "count"))
        .reset_index()
    )
    crew_by_game = crew_by_game[crew_by_game["n_officials"] == 3]

    games = dhc.merge(crew_by_game[["game_id", "officials"]], on="game_id", how="inner")
    games = games[games["min"] >= MIN_MINUTES].copy()
    games = games[games["season"].isin(MODEL_SEASONS)].copy()

    logger.info(
        "Loaded %d player-games (min>=%d, 3-official crew) for %d players",
        len(games),
        MIN_MINUTES,
        games["player_name"].nunique(),
    )
    return games


def _compute_interaction_table(games: pd.DataFrame) -> pd.DataFrame:
    """Compute player×official defense-adjusted FTA/36 deltas from a game subset."""
    records = []
    for player_name in sorted(games["player_name"].unique()):
        pdf = games[games["player_name"] == player_name]
        all_games = set(pdf["game_id"].unique())
        officials = sorted({o for offs in pdf["officials"] for o in offs})

        for official in officials:
            game_ids_with = {
                gid
                for gid, offs in zip(pdf["game_id"], pdf["officials"])
                if official in offs
            }
            game_ids_without = all_games - game_ids_with

            pg_with = pdf[pdf["game_id"].isin(game_ids_with)].drop_duplicates("game_id")
            pg_without = pdf[pdf["game_id"].isin(game_ids_without)].drop_duplicates("game_id")

            delta = _adjusted_fta36_delta(pg_with, pg_without, MIN_GAMES_WITH)
            if delta is None:
                continue

            records.append({
                "player_name": player_name,
                "official_pbp_name": official,
                "adj_fta36_delta": delta,
                "n_games_with": len(pg_with),
                "n_games_without": len(pg_without),
            })

    return pd.DataFrame(records)


def build_interaction_profiles(temporal: bool = True) -> pd.DataFrame:
    """Build static or temporal player×official interaction tables."""
    games = _load_player_games_with_crew()

    if not temporal:
        table = _compute_interaction_table(games)
        table["profile_season"] = "all"
        out = STATIC_INTERACTIONS_PATH
    else:
        seasons = sorted(games["season"].unique(), key=_season_sort_key)
        chunks = []
        for target_season in seasons:
            prior = games[games["season"].map(_season_sort_key) < _season_sort_key(target_season)]
            if prior.empty:
                continue
            chunk = _compute_interaction_table(prior)
            chunk["profile_season"] = target_season
            chunks.append(chunk)
        table = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        out = TEMPORAL_INTERACTIONS_PATH

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    logger.info("Wrote %d player×official deltas (%s) to %s", len(table), "temporal" if temporal else "static", out)
    return table


def _player_baselines(games: pd.DataFrame, through_season: str | None = None) -> pd.DataFrame:
    """Per-player FTA/36 baseline from prior games."""
    subset = games.copy()
    if through_season is not None:
        subset = subset[subset["season"].map(_season_sort_key) < _season_sort_key(through_season)]

    baselines = (
        subset.groupby("player_name", as_index=False)
        .agg(
            baseline_fta36=("fta36", "mean"),
            baseline_games=("game_id", "nunique"),
            baseline_fta=("fta", "sum"),
            baseline_min=("min", "sum"),
        )
    )
    return baselines


def build_player_game_dataset(temporal: bool = True) -> pd.DataFrame:
    """One row per player-game with crew interaction features and FTA/36 target."""
    games = _load_player_games_with_crew()
    interactions_path = TEMPORAL_INTERACTIONS_PATH if temporal else STATIC_INTERACTIONS_PATH

    if not interactions_path.exists():
        build_interaction_profiles(temporal=temporal)

    interactions = pd.read_parquet(interactions_path)
    league_delta = interactions["adj_fta36_delta"].median()

    if temporal:
        interaction_lookup = {
            season: df.set_index(["player_name", "official_pbp_name"])
            for season, df in interactions.groupby("profile_season")
        }
        player_defaults = interactions.groupby("player_name")["adj_fta36_delta"].median().to_dict()
    else:
        static_lookup = interactions.set_index(["player_name", "official_pbp_name"])
        player_defaults = interactions.groupby("player_name")["adj_fta36_delta"].median().to_dict()

    records = []
    for _, row in games.iterrows():
        player = row["player_name"]
        season = row["season"]
        officials = row["officials"]

        if temporal:
            lookup = interaction_lookup.get(season)
            if lookup is None:
                continue
            deltas = []
            for off in officials:
                key = (player, off)
                if key in lookup.index:
                    val = lookup.loc[key, "adj_fta36_delta"]
                    if isinstance(val, pd.Series):
                        val = val.iloc[0]
                    deltas.append(float(val))
                else:
                    deltas.append(player_defaults.get(player, league_delta))
        else:
            deltas = []
            for off in officials:
                key = (player, off)
                if key in static_lookup.index:
                    val = static_lookup.loc[key, "adj_fta36_delta"]
                    if isinstance(val, pd.Series):
                        val = val.iloc[0]
                    deltas.append(float(val))
                else:
                    deltas.append(player_defaults.get(player, league_delta))

        if temporal:
            prior = games[
                (games["player_name"] == player)
                & (games["season"].map(_season_sort_key) < _season_sort_key(season))
            ]
            if len(prior) < MIN_GAMES_WITH:
                continue
            baseline = 36 * prior["fta"].sum() / max(prior["min"].sum(), 1)
        else:
            baseline = 36 * games[games["player_name"] == player]["fta"].sum() / max(
                games[games["player_name"] == player]["min"].sum(), 1,
            )

        record = {
            "game_id": row["game_id"],
            "player_name": player,
            "season": season,
            "season_type": row["season_type"],
            "fta36": row["fta36"],
            "fta": row["fta"],
            "min": row["min"],
            "opponent_defrtg": row["opponent_defrtg"],
            "is_playoff": int(row["season_type"] == "PO"),
            "player_baseline_fta36": baseline,
            "crew_mean_adj_delta": float(np.mean(deltas)),
            "crew_sum_adj_delta": float(np.sum(deltas)),
            "crew_max_adj_delta": float(np.max(deltas)),
            "crew_min_adj_delta": float(np.min(deltas)),
            "crew_std_adj_delta": float(np.std(deltas)) if len(deltas) > 1 else 0.0,
            "n_officials_matched": sum(
                1 for d in deltas if d != player_defaults.get(player, league_delta)
            ),
        }
        for i, (off, delta) in enumerate(zip(officials, deltas)):
            record[f"off{i+1}_name"] = off
            record[f"off{i+1}_adj_delta"] = delta

        records.append(record)

    dataset = pd.DataFrame(records)
    out = DATASET_PATH if temporal else DATASET_STATIC_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out, index=False)
    logger.info("Built player-game dataset (%s): %d rows", "temporal" if temporal else "static", len(dataset))
    return dataset


def train_and_evaluate(dataset: pd.DataFrame | None = None, temporal: bool = True) -> dict:
    """Train on TRAIN_SEASONS, evaluate on TEST_SEASONS."""
    path = DATASET_PATH if temporal else DATASET_STATIC_PATH
    eval_path = EVALUATION_PATH if temporal else EVALUATION_STATIC_PATH

    if dataset is None:
        if not path.exists():
            raise RuntimeError(f"No dataset at {path}. Run 'build' first.")
        dataset = pd.read_parquet(path)

    train = dataset[dataset["season"].isin(TRAIN_SEASONS)].copy()
    test = dataset[dataset["season"].isin(TEST_SEASONS)].copy()
    logger.info("Train: %d player-games | Test: %d player-games", len(train), len(test))

    feature_cols = [
        "player_baseline_fta36",
        "crew_mean_adj_delta",
        "crew_sum_adj_delta",
        "crew_max_adj_delta",
        "crew_min_adj_delta",
        "opponent_defrtg",
        "is_playoff",
    ]
    feature_cols = [c for c in feature_cols if c in dataset.columns]

    train_x = train[feature_cols].copy()
    test_x = test[feature_cols].copy()
    medians = train_x.median(numeric_only=True)
    train_x = train_x.fillna(medians)
    test_x = test_x.fillna(medians)

    y_train = train["fta36"].values
    y_test = test["fta36"].values

    results = {}

    train_mean = y_train.mean()
    results["league_avg"] = _evaluate(y_test, np.full(len(y_test), train_mean), "League average FTA/36")

    player_train_means = train.groupby("player_name")["fta36"].mean()
    player_pred = test["player_name"].map(player_train_means).fillna(train_mean).values
    results["player_baseline"] = _evaluate(y_test, player_pred, "Player baseline (train mean)")

    prior_baseline_pred = test["player_baseline_fta36"].values
    results["player_prior_baseline"] = _evaluate(
        y_test, prior_baseline_pred, "Player prior baseline (pre-season)",
    )

    additive_pred = prior_baseline_pred + test["crew_mean_adj_delta"].values
    results["additive_crew"] = _evaluate(y_test, additive_pred, "Additive: prior + crew mean delta")

    ols_features = ["player_baseline_fta36", "crew_mean_adj_delta", "opponent_defrtg", "is_playoff"]
    ols_features = [f for f in ols_features if f in feature_cols]
    beta_ols = None
    if ols_features:
        X_tr = np.column_stack([np.ones(len(train_x)), train_x[ols_features].values])
        X_te = np.column_stack([np.ones(len(test_x)), test_x[ols_features].values])
        beta_ols, _, _, _ = np.linalg.lstsq(X_tr, y_train, rcond=None)
        results["ols_crew"] = _evaluate(y_test, X_te @ beta_ols, "OLS: baseline + crew + defrtg")

    ridge_features = feature_cols
    X_tr = train_x[ridge_features].values
    X_te = test_x[ridge_features].values
    means = X_tr.mean(axis=0)
    stds = X_tr.std(axis=0)
    stds[stds == 0] = 1.0
    X_tr = (X_tr - means) / stds
    X_te = (X_te - means) / stds
    X_tr = np.column_stack([np.ones(len(X_tr)), X_tr])
    X_te = np.column_stack([np.ones(len(X_te)), X_te])

    for alpha in [1.0, 10.0, 100.0]:
        beta = _ridge(X_tr, y_train, alpha)
        results[f"ridge_a{alpha}"] = _evaluate(y_test, X_te @ beta, f"Ridge α={alpha}")

    eval_df = pd.DataFrame([
        {
            "model": name,
            "label": res["label"],
            "rmse": res["rmse"],
            "mae": res["mae"],
            "r2": res["r2"],
            "corr": res["corr"],
            "n_test": res["n"],
        }
        for name, res in results.items()
    ])

    best_name = eval_df.loc[eval_df["rmse"].idxmin(), "model"]
    pred_df = test[["game_id", "player_name", "season", "season_type", "fta36"]].copy()
    pred_df["predicted_fta36"] = results[best_name]["predictions"]
    pred_df["residual"] = pred_df["fta36"] - pred_df["predicted_fta36"]
    pred_df["model"] = best_name

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    eval_df.to_parquet(eval_path, index=False)
    pred_df.to_parquet(PREDICTIONS_PATH, index=False)

    logger.info("Wrote evaluation to %s", eval_path)
    logger.info("Best model: %s (RMSE=%.3f)", results[best_name]["label"], results[best_name]["rmse"])

    return results


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    n_features = X.shape[1]
    A = X.T @ X + alpha * np.eye(n_features)
    A[0, 0] -= alpha
    return np.linalg.solve(A, X.T @ y)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0

    logger.info("%-42s RMSE=%.3f  MAE=%.3f  R²=%.4f  r=%.4f", label, rmse, mae, r2, corr)
    return {"label": label, "rmse": rmse, "mae": mae, "r2": r2, "corr": corr, "n": len(y_true), "predictions": y_pred}


def diagnose() -> None:
    """Compare static vs temporal signal strength."""
    print("\n  Player-Level FTA/36 Signal Diagnostics")
    print("  " + "=" * 80)

    for label, path in [("temporal", DATASET_PATH), ("static", DATASET_STATIC_PATH)]:
        if not path.exists():
            print(f"\n  [{label}] dataset not found")
            continue
        ds = pd.read_parquet(path)
        model = ds[ds["season"].isin(MODEL_SEASONS)]
        train = model[model["season"].isin(TRAIN_SEASONS)]
        test = model[model["season"].isin(TEST_SEASONS)]

        print(f"\n  [{label.upper()}] {len(model)} player-games (train={len(train)}, test={len(test)})")
        for split_name, split in [("train", train), ("test", test)]:
            r = split["fta36"].corr(split["crew_mean_adj_delta"])
            r_base = split["fta36"].corr(split["player_baseline_fta36"])
            print(f"    {split_name}: r(fta36, crew_delta)={r:+.3f}  r(fta36, baseline)={r_base:+.3f}")

        by_player = (
            test.groupby("player_name")
            .apply(lambda g: g["fta36"].corr(g["crew_mean_adj_delta"]) if len(g) > 20 else np.nan)
            .dropna()
            .sort_values(ascending=False)
        )
        if not by_player.empty:
            print(f"    test top players by r(crew_delta): {by_player.head(3).to_dict()}")

    for label, path in [("temporal", EVALUATION_PATH), ("static", EVALUATION_STATIC_PATH)]:
        if path.exists():
            print(f"\n  {label.title()} evaluation:")
            for _, row in pd.read_parquet(path).iterrows():
                print(f"    {row['label']:<42} RMSE={row['rmse']:.3f}  R²={row['r2']:.4f}  r={row['corr']:.4f}")


def print_summary(temporal: bool = True) -> None:
    eval_path = EVALUATION_PATH if temporal else EVALUATION_STATIC_PATH
    if not eval_path.exists():
        raise RuntimeError(f"No evaluation at {eval_path}. Run 'build' first.")

    eval_df = pd.read_parquet(eval_path)
    label = "Temporal" if temporal else "Static"
    print(f"\n  Player FTA/36 Model Evaluation ({label}, Test Set)")
    print("  " + "=" * 80)
    print(f"  {'Model':<42} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'r':>8}")
    print("  " + "-" * 80)
    for _, row in eval_df.iterrows():
        print(f"  {row['label']:<42} {row['rmse']:>8.3f} {row['mae']:>8.3f} {row['r2']:>8.4f} {row['corr']:>8.4f}")

    best = eval_df.loc[eval_df["rmse"].idxmin()]
    print(f"\n  Best model: {best['label']} (RMSE={best['rmse']:.3f}, R²={best['r2']:.4f})")

    if PREDICTIONS_PATH.exists():
        preds = pd.read_parquet(PREDICTIONS_PATH)
        print(f"\n  Sample predictions (n={len(preds)} test player-games):")
        print(f"  {'Player':<22} {'Season':>8} {'Actual':>7} {'Predicted':>10} {'Residual':>9}")
        print("  " + "-" * 62)
        for _, row in preds.head(15).iterrows():
            print(
                f"  {row['player_name']:<22} {row['season']:>8} "
                f"{row['fta36']:>7.1f} {row['predicted_fta36']:>10.1f} {row['residual']:>+9.1f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Player-level crew FTA/36 model")
    parser.add_argument("command", choices=["build", "summary", "diagnose"])
    parser.add_argument("--static", action="store_true", help="Use all-time interaction profiles (leaky)")
    args = parser.parse_args()

    temporal = not args.static

    if args.command == "build":
        dataset = build_player_game_dataset(temporal=temporal)
        train_and_evaluate(dataset, temporal=temporal)
        print_summary(temporal=temporal)
    elif args.command == "summary":
        print_summary(temporal=temporal)
    elif args.command == "diagnose":
        diagnose()


if __name__ == "__main__":
    main()
