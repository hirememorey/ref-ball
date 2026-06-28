r"""Predictive model: crew assignment → game shooting-foul environment.

Step 5 of the ref-ball pipeline. Predicts game-level SF count (and rate)
from the 3-official crew assignment using per-official historical profiles.

Approach:
  1. Additive baseline — sum/average each official's profile features
  2. Season holdout — train 2014-15 through 2022-23, test 2023-24 + 2024-25
  3. Compare against league-average baseline
  4. Test additive residuals for crew interaction effects

Profiles can be static (all-time, for exploration) or temporal (prior seasons
only, for honest holdout evaluation).

Usage:
    python src/crew_predictive_model.py build       # build dataset + train + evaluate
    python src/crew_predictive_model.py build --no-train  # build dataset only
    python src/crew_predictive_model.py build --temporal  # use prior-season profiles
    python src/crew_predictive_model.py summary      # print evaluation results
    python src/crew_predictive_model.py interactions # test crew interaction effects
    python src/crew_predictive_model.py diagnose     # signal diagnostics
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROFILES_DIR = config.PROCESSED_DIR / "player_official"
OFFICIAL_PROFILES_PATH = PROFILES_DIR / "official_calling_profiles.parquet"
REF_PROFILES_PATH = config.PROCESSED_DIR / "ref_profiles.parquet"
CREW_PATH = config.CREW_ASSIGNMENTS_PATH
MODEL_DIR = config.PROCESSED_DIR / "model"
DATASET_PATH = MODEL_DIR / "game_crew_dataset.parquet"
DATASET_TEMPORAL_PATH = MODEL_DIR / "game_crew_dataset_temporal.parquet"
TEMPORAL_PROFILES_PATH = MODEL_DIR / "temporal_official_profiles.parquet"
EVALUATION_PATH = MODEL_DIR / "evaluation.parquet"
EVALUATION_TEMPORAL_PATH = MODEL_DIR / "evaluation_temporal.parquet"
PREDICTIONS_PATH = MODEL_DIR / "predictions.parquet"
PREDICTIONS_TEMPORAL_PATH = MODEL_DIR / "predictions_temporal.parquet"
ALL_PREDICTIONS_PATH = MODEL_DIR / "predictions_all.parquet"
ALL_PREDICTIONS_TEMPORAL_PATH = MODEL_DIR / "predictions_all_temporal.parquet"

TRAIN_SEASONS = [f"{y}-{(y+1)%100:02d}" for y in range(2014, 2023)]
TEST_SEASONS = ["2023-24", "2024-25"]
MODEL_SEASONS = TRAIN_SEASONS + TEST_SEASONS

PROFILE_FEATURE_COLS = [
    "sf_per_game",
    "sf_pct_of_fouls",
    "mean_adj_fta36_delta",
    "suppressor_score",
    "total_games",
]


def _game_id_to_season(game_id: str) -> str:
    start_year = 2000 + int(game_id[3:5])
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _game_id_to_season_type(game_id: str) -> str:
    prefix = game_id[:3]
    return "PO" if prefix == "004" else "RS"


def _season_sort_key(season: str) -> int:
    return int(season.split("-")[0])


def _load_game_sf_counts() -> pd.DataFrame:
    """Load per-game shooting foul counts from ingested game parquets."""
    files = sorted(config.GAMES_DIR.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No game parquets in {config.GAMES_DIR}")

    records = []
    for f in files:
        df = pd.read_parquet(f, columns=["game_id", "foul_type"])
        sf_count = (df["foul_type"] == "Shooting").sum()
        total_fouls = len(df)
        game_id = df["game_id"].iloc[0]
        records.append({
            "game_id": game_id,
            "sf_count": sf_count,
            "total_fouls": total_fouls,
        })

    result = pd.DataFrame(records)
    result["season"] = result["game_id"].apply(_game_id_to_season)
    result["season_type"] = result["game_id"].apply(_game_id_to_season_type)
    logger.info("Loaded SF counts for %d games", len(result))
    return result


def _load_official_features() -> pd.DataFrame:
    """Merge official_calling_profiles with ref_profiles for complete feature set."""
    profiles = pd.read_parquet(OFFICIAL_PROFILES_PATH)
    ref = pd.read_parquet(REF_PROFILES_PATH)
    ref = ref.rename(columns={"caller_official_name": "official_pbp_name"})

    ref_cols = [
        "official_pbp_name",
        "sf_per_game_RS",
        "sf_per_game_PO",
        "sf_per_game_delta",
        "total_games",
        "total_sf",
    ]
    ref_available = [c for c in ref_cols if c in ref.columns]
    ref_subset = ref[ref_available].copy()

    merged = profiles.merge(ref_subset, on="official_pbp_name", how="left", suffixes=("", "_ref"))
    logger.info("Merged official features: %d officials, %d columns", len(merged), len(merged.columns))
    return merged


def build_temporal_official_profiles() -> pd.DataFrame:
    """Build per-official cumulative profiles by season (prior seasons only).

    For season S, profiles use only games from seasons strictly before S.
    This avoids leakage when evaluating season holdouts.
    """
    crew = pd.read_parquet(CREW_PATH)
    crew["pbp_name"] = crew["first_name"].str[0] + "." + crew["last_name"]
    crew["season"] = crew["game_id"].apply(_game_id_to_season)
    crew["season_type"] = crew["game_id"].apply(_game_id_to_season_type)

    games_assigned = (
        crew.groupby(["season", "pbp_name", "season_type"])
        .agg(n_games_assigned=("game_id", "nunique"))
        .reset_index()
    )

    sf_records = []
    files = sorted(config.GAMES_DIR.glob("*.parquet"))
    for f in files:
        df = pd.read_parquet(f, columns=["game_id", "foul_type", "caller_official_name"])
        game_id = df["game_id"].iloc[0]
        season = _game_id_to_season(game_id)
        season_type = _game_id_to_season_type(game_id)
        sf = df[df["foul_type"] == "Shooting"]
        if sf.empty:
            continue
        for official, n_sf in sf.groupby("caller_official_name").size().items():
            if not official:
                continue
            sf_records.append({
                "season": season,
                "season_type": season_type,
                "game_id": game_id,
                "official_pbp_name": official,
                "n_sf": int(n_sf),
            })

    sf_by_official = (
        pd.DataFrame(sf_records)
        .groupby(["season", "season_type", "official_pbp_name"], as_index=False)
        .agg(n_sf=("n_sf", "sum"))
    )

    season_official = games_assigned.merge(
        sf_by_official,
        left_on=["season", "pbp_name", "season_type"],
        right_on=["season", "official_pbp_name", "season_type"],
        how="left",
    )
    season_official["n_sf"] = season_official["n_sf"].fillna(0)
    season_official = season_official.drop(columns=["official_pbp_name"]).rename(
        columns={"pbp_name": "official_pbp_name"},
    )

    seasons = sorted(season_official["season"].unique(), key=_season_sort_key)
    profile_rows = []

    for target_season in seasons:
        prior = season_official[season_official["season"].map(_season_sort_key) < _season_sort_key(target_season)]
        if prior.empty:
            continue

        cum = (
            prior.groupby("official_pbp_name", as_index=False)
            .agg(
                n_sf=("n_sf", "sum"),
                n_games_assigned=("n_games_assigned", "sum"),
            )
        )
        cum["profile_season"] = target_season
        cum["sf_per_game"] = cum["n_sf"] / cum["n_games_assigned"].clip(lower=1)
        cum["total_games"] = cum["n_games_assigned"]
        profile_rows.append(cum)

    if not profile_rows:
        raise RuntimeError("No temporal profiles built — check crew/game data")

    temporal = pd.concat(profile_rows, ignore_index=True)

    static = _load_official_features()
    adj_cols = ["official_pbp_name", "mean_adj_fta36_delta", "suppressor_score", "sf_pct_of_fouls"]
    adj_cols = [c for c in adj_cols if c in static.columns]
    temporal = temporal.merge(static[adj_cols], on="official_pbp_name", how="left")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temporal.to_parquet(TEMPORAL_PROFILES_PATH, index=False)
    logger.info(
        "Built temporal profiles: %d official×season rows across %d seasons",
        len(temporal),
        temporal["profile_season"].nunique(),
    )
    return temporal


def _aggregate_crew_features(
    officials: list[str],
    official_lookup: pd.DataFrame,
    feature_cols: list[str],
    impute_values: dict[str, float] | None = None,
) -> dict:
    """Aggregate per-official features into crew-level summary stats."""
    feats = official_lookup.reindex(officials)
    if impute_values:
        for col in feature_cols:
            if col in feats.columns and col in impute_values:
                feats[col] = feats[col].fillna(impute_values[col])

    n_missing = int(feats[feature_cols].isna().all(axis=1).sum()) if feature_cols else 0

    record: dict = {"n_crew_profiled": 3 - n_missing}
    for col in feature_cols:
        vals = feats[col].dropna()
        if len(vals) == 0:
            record[f"crew_mean_{col}"] = np.nan
            record[f"crew_min_{col}"] = np.nan
            record[f"crew_max_{col}"] = np.nan
            record[f"crew_std_{col}"] = np.nan
            record[f"crew_sum_{col}"] = np.nan
            for i in range(3):
                record[f"off{i+1}_{col}"] = np.nan
        else:
            record[f"crew_mean_{col}"] = vals.mean()
            record[f"crew_min_{col}"] = vals.min()
            record[f"crew_max_{col}"] = vals.max()
            record[f"crew_std_{col}"] = vals.std() if len(vals) > 1 else 0.0
            record[f"crew_sum_{col}"] = vals.sum()
            for i, off in enumerate(officials):
                val = feats.loc[off, col] if off in feats.index else np.nan
                record[f"off{i+1}_{col}"] = val

    for i, off in enumerate(officials):
        record[f"off{i+1}_name"] = off

    return record


def build_game_crew_dataset(temporal: bool = False) -> pd.DataFrame:
    """Build the modeling dataset: one row per game with crew features.

    For each game, look up the 3 officials and aggregate their profile
    features (mean, min, max, std, sum) across the crew.

    When temporal=True, officials' features come from cumulative stats
  through the prior season only (no leakage).
    """
    crew = pd.read_parquet(CREW_PATH)
    crew["pbp_name"] = crew["first_name"].str[0] + "." + crew["last_name"]
    crew["season"] = crew["game_id"].apply(_game_id_to_season)
    crew["season_type"] = crew["game_id"].apply(_game_id_to_season_type)

    game_sf = _load_game_sf_counts()

    if temporal:
        if not TEMPORAL_PROFILES_PATH.exists():
            build_temporal_official_profiles()
        temporal_profiles = pd.read_parquet(TEMPORAL_PROFILES_PATH)
        feature_cols = [c for c in PROFILE_FEATURE_COLS if c in temporal_profiles.columns]
    else:
        official_features = _load_official_features()
        temporal_profiles = None
        feature_cols = [c for c in PROFILE_FEATURE_COLS if c in official_features.columns]

    crew_by_game = (
        crew.groupby("game_id")
        .agg(officials=("pbp_name", list), n_officials=("pbp_name", "count"))
        .reset_index()
    )
    crew_by_game = crew_by_game[crew_by_game["n_officials"] == 3].copy()

    game_sf = game_sf.merge(
        crew_by_game[["game_id", "officials", "n_officials"]],
        on="game_id",
        how="inner",
    )

    feature_records = []
    if temporal:
        profile_by_season = {
            season: df.set_index("official_pbp_name")
            for season, df in temporal_profiles.groupby("profile_season")
        }
        league_defaults = temporal_profiles.groupby("profile_season")[feature_cols].median().to_dict("index")
        global_impute = temporal_profiles[feature_cols].median().to_dict()
    else:
        official_lookup = official_features.set_index("official_pbp_name")
        global_impute = official_features[feature_cols].median().to_dict()

    for _, row in game_sf.iterrows():
        officials = row["officials"]
        if temporal:
            lookup = profile_by_season.get(row["season"])
            if lookup is None:
                continue
            defaults = league_defaults.get(row["season"], global_impute)
            filled = lookup.reindex(officials)
            for col in feature_cols:
                if col in filled.columns:
                    filled[col] = filled[col].fillna(defaults.get(col, global_impute.get(col, np.nan)))
            record = _aggregate_crew_features(officials, filled, feature_cols, impute_values=defaults)
        else:
            record = _aggregate_crew_features(
                officials, official_lookup, feature_cols, impute_values=global_impute,
            )

        record.update({
            "game_id": row["game_id"],
            "season": row["season"],
            "season_type": row["season_type"],
            "sf_count": row["sf_count"],
            "total_fouls": row["total_fouls"],
        })
        feature_records.append(record)

    dataset = pd.DataFrame(feature_records)
    dataset = dataset[dataset["n_crew_profiled"] >= 2].copy()

    dataset["sf_rate"] = dataset["sf_count"] / dataset["total_fouls"].replace(0, np.nan)
    dataset["is_playoff"] = (dataset["season_type"] == "PO").astype(int)

    out_path = DATASET_TEMPORAL_PATH if temporal else DATASET_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out_path, index=False)
    logger.info(
        "Built %s game-crew dataset: %d games, %d features",
        "temporal" if temporal else "static",
        len(dataset),
        len(dataset.columns),
    )
    return dataset


def train_and_evaluate(
    dataset: pd.DataFrame | None = None,
    dataset_path: Path | None = None,
    temporal: bool = False,
) -> dict:
    """Train additive model on TRAIN_SEASONS, evaluate on TEST_SEASONS."""
    if dataset is None:
        path = dataset_path or (DATASET_TEMPORAL_PATH if temporal else DATASET_PATH)
        if not path.exists():
            raise RuntimeError(f"No dataset at {path}. Run 'build' first.")
        dataset = pd.read_parquet(path)

    eval_path = EVALUATION_TEMPORAL_PATH if temporal else EVALUATION_PATH
    pred_path = PREDICTIONS_TEMPORAL_PATH if temporal else PREDICTIONS_PATH
    all_pred_path = ALL_PREDICTIONS_TEMPORAL_PATH if temporal else ALL_PREDICTIONS_PATH

    train = dataset[dataset["season"].isin(TRAIN_SEASONS)].copy()
    test = dataset[dataset["season"].isin(TEST_SEASONS)].copy()

    logger.info("Train: %d games (%s)", len(train), ", ".join(TRAIN_SEASONS))
    logger.info("Test: %d games (%s)", len(test), ", ".join(TEST_SEASONS))

    if len(test) == 0:
        logger.warning("No test games found for seasons %s", TEST_SEASONS)
        test = dataset[~dataset["season"].isin(TRAIN_SEASONS)].copy()
        logger.info("Using all non-train seasons as test: %d games", len(test))

    crew_feature_cols = [
        c for c in dataset.columns
        if c.startswith("crew_") or c.startswith("off1_") or c.startswith("off2_") or c.startswith("off3_")
    ]
    crew_feature_cols = [c for c in crew_feature_cols if c not in {
        "off1_name", "off2_name", "off3_name",
    }]

    context_cols = ["is_playoff"]
    all_features = [c for c in crew_feature_cols + context_cols if c in dataset.columns]

    train_features = train[all_features].copy()
    test_features = test[all_features].copy()

    numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
    medians = train[numeric_cols].median()
    train_features = train_features[numeric_cols].fillna(medians)
    test_features = test_features[numeric_cols].fillna(medians)

    y_train = train["sf_count"].values
    y_test = test["sf_count"].values
    y_rate_train = train["sf_rate"].values
    y_rate_test = test["sf_rate"].values

    train_mean = y_train.mean()
    league_avg_pred = np.full(len(y_test), train_mean)

    results = {}
    results["baseline"] = _evaluate(y_test, league_avg_pred, "League Average (SF count)")

    if "crew_sum_sf_per_game" in test_features.columns:
        sum_pred = test_features["crew_sum_sf_per_game"].values
        results["crew_sf_sum"] = _evaluate(y_test, sum_pred, "Crew SF/G Sum (3 officials)")

    ols_features = [
        "crew_sum_sf_per_game",
        "crew_mean_sf_pct_of_fouls",
        "crew_mean_mean_adj_fta36_delta",
        "crew_mean_suppressor_score",
        "is_playoff",
    ]
    ols_features = [f for f in ols_features if f in numeric_cols]

    beta_ols = None
    if ols_features:
        X_train_ols = train_features[ols_features].values
        X_test_ols = test_features[ols_features].values
        X_train_ols = np.column_stack([np.ones(len(X_train_ols)), X_train_ols])
        X_test_ols = np.column_stack([np.ones(len(X_test_ols)), X_test_ols])

        beta_ols, _, _, _ = np.linalg.lstsq(X_train_ols, y_train, rcond=None)
        ols_pred = X_test_ols @ beta_ols
        results["ols_additive"] = _evaluate(y_test, ols_pred, "OLS Additive (SF count)")

    if "crew_mean_sf_pct_of_fouls" in numeric_cols:
        rate_features = ["crew_mean_sf_pct_of_fouls", "crew_sum_sf_per_game", "is_playoff"]
        rate_features = [f for f in rate_features if f in numeric_cols]
        X_train_rate = train_features[rate_features].values
        X_test_rate = test_features[rate_features].values
        X_train_rate = np.column_stack([np.ones(len(X_train_rate)), X_train_rate])
        X_test_rate = np.column_stack([np.ones(len(X_test_rate)), X_test_rate])
        beta_rate, _, _, _ = np.linalg.lstsq(X_train_rate, y_rate_train, rcond=None)
        rate_pred = X_test_rate @ beta_rate
        results["ols_sf_rate"] = _evaluate(y_rate_test, rate_pred, "OLS Additive (SF rate)", is_rate=True)

    ridge_features = numeric_cols
    X_train_ridge = train_features[ridge_features].values
    X_test_ridge = test_features[ridge_features].values

    means = X_train_ridge.mean(axis=0)
    stds = X_train_ridge.std(axis=0)
    stds[stds == 0] = 1.0
    X_train_ridge = (X_train_ridge - means) / stds
    X_test_ridge = (X_test_ridge - means) / stds

    X_train_ridge = np.column_stack([np.ones(len(X_train_ridge)), X_train_ridge])
    X_test_ridge = np.column_stack([np.ones(len(X_test_ridge)), X_test_ridge])

    for alpha in [1.0, 10.0, 100.0]:
        beta_ridge = _ridge(X_train_ridge, y_train, alpha)
        ridge_pred = X_test_ridge @ beta_ridge
        results[f"ridge_a{alpha}"] = _evaluate(y_test, ridge_pred, f"Ridge α={alpha}")

    eval_rows = []
    for name, res in results.items():
        eval_rows.append({
            "model": name,
            "label": res["label"],
            "rmse": res["rmse"],
            "mae": res["mae"],
            "r2": res["r2"],
            "corr": res["corr"],
            "n_test": res["n"],
        })
    eval_df = pd.DataFrame(eval_rows)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    eval_df.to_parquet(eval_path, index=False)

    count_models = [name for name in results if "rate" not in results[name]["label"].lower()]
    count_eval = eval_df[eval_df["model"].isin(count_models)]
    best_name = count_eval.loc[count_eval["rmse"].idxmin(), "model"]
    best_pred = results[best_name]["predictions"]
    pred_df = test[["game_id", "season", "season_type", "sf_count", "sf_rate"]].copy()
    pred_df["predicted_sf"] = best_pred
    pred_df["residual"] = pred_df["sf_count"] - pred_df["predicted_sf"]
    pred_df["model"] = best_name
    pred_df.to_parquet(pred_path, index=False)

    all_pred_df = _predict_all_games(
        dataset=dataset,
        train=train,
        numeric_cols=numeric_cols,
        medians=medians,
        ols_features=ols_features,
        beta_ols=beta_ols,
    )
    all_pred_df.to_parquet(all_pred_path, index=False)

    logger.info("Wrote evaluation to %s", eval_path)
    logger.info("Wrote test predictions to %s", pred_path)
    logger.info("Wrote all-game predictions to %s", all_pred_path)

    return results


def _predict_all_games(
    dataset: pd.DataFrame,
    train: pd.DataFrame,
    numeric_cols: list[str],
    medians: pd.Series,
    ols_features: list[str],
    beta_ols: np.ndarray | None,
) -> pd.DataFrame:
    """Apply the OLS additive model to every game (for interaction analysis)."""
    model_df = dataset[dataset["season"].isin(MODEL_SEASONS)].copy()
    features = model_df[[c for c in numeric_cols if c in model_df.columns]].fillna(medians)

    if beta_ols is not None and ols_features:
        X = features[ols_features].values
        X = np.column_stack([np.ones(len(X)), X])
        predicted = X @ beta_ols
        model_name = "ols_additive"
    elif "crew_sum_sf_per_game" in features.columns:
        predicted = features["crew_sum_sf_per_game"].values
        model_name = "crew_sf_sum"
    else:
        predicted = np.full(len(model_df), train["sf_count"].mean())
        model_name = "baseline"

    out = model_df[["game_id", "season", "season_type", "sf_count", "sf_rate"]].copy()
    out["predicted_sf"] = predicted
    out["residual"] = out["sf_count"] - out["predicted_sf"]
    out["model"] = model_name
    return out


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge regression: (X'X + αI)^-1 X'y."""
    n_features = X.shape[1]
    A = X.T @ X + alpha * np.eye(n_features)
    A[0, 0] -= alpha
    b = X.T @ y
    return np.linalg.solve(A, b)


def _evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
    is_rate: bool = False,
) -> dict:
    """Compute evaluation metrics."""
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0

    unit = "rate" if is_rate else "count"
    logger.info(
        "%-40s RMSE=%.4f  MAE=%.4f  R²=%.4f  r=%.4f  (%s)",
        label, rmse, mae, r2, corr, unit,
    )

    return {
        "label": label,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "corr": corr,
        "n": len(y_true),
        "predictions": y_pred,
    }


def test_interactions(
    dataset: pd.DataFrame | None = None,
    dataset_path: Path | None = None,
    min_games: int = 20,
    use_train_only: bool = False,
    temporal: bool = False,
) -> pd.DataFrame:
    """Test whether additive residuals contain crew interaction effects.

    Uses residuals from all modeling-season games (not just the test holdout)
    so crew pairs have enough shared games for stable estimates.
    """
    path = dataset_path or (DATASET_TEMPORAL_PATH if temporal else DATASET_PATH)
    all_pred_path = ALL_PREDICTIONS_TEMPORAL_PATH if temporal else ALL_PREDICTIONS_PATH
    if dataset is None:
        if not path.exists():
            raise RuntimeError(f"No dataset at {path}. Run 'build' first.")
        dataset = pd.read_parquet(path)

    if not all_pred_path.exists():
        raise RuntimeError("No all-game predictions found. Run 'build' (with train) first.")

    preds = pd.read_parquet(all_pred_path)
    if use_train_only:
        preds = preds[preds["season"].isin(TRAIN_SEASONS)]
        scope = "train seasons"
    else:
        preds = preds[preds["season"].isin(MODEL_SEASONS)]
        scope = "all modeling seasons (2014-25)"

    dataset = dataset.merge(preds[["game_id", "residual"]], on="game_id", how="inner")
    logger.info("Interaction analysis on %d games (%s)", len(dataset), scope)

    crew = pd.read_parquet(CREW_PATH)
    crew["pbp_name"] = crew["first_name"].str[0] + "." + crew["last_name"]
    crew = crew[crew["game_id"].isin(dataset["game_id"])]

    crew_pairs = []
    for game_id, group in crew.groupby("game_id"):
        officials = sorted(group["pbp_name"].tolist())
        if len(officials) < 3:
            continue
        for i in range(len(officials)):
            for j in range(i + 1, len(officials)):
                crew_pairs.append({
                    "game_id": game_id,
                    "pair_key": f"{officials[i]}|{officials[j]}",
                })

    pairs_df = pd.DataFrame(crew_pairs)
    if pairs_df.empty:
        logger.warning("No crew pairs found")
        return pd.DataFrame()

    pairs_df = pairs_df.merge(dataset[["game_id", "residual"]], on="game_id", how="inner")

    pair_residuals = (
        pairs_df.groupby("pair_key")
        .agg(
            mean_residual=("residual", "mean"),
            n_games=("game_id", "count"),
            std_residual=("residual", "std"),
        )
        .reset_index()
    )

    eligible = pair_residuals[pair_residuals["n_games"] >= min_games].copy()

    if eligible.empty:
        logger.warning("No pairs with >= %d games", min_games)
        return eligible

    eligible["se_mean"] = eligible["std_residual"] / np.sqrt(eligible["n_games"])
    eligible["z_score"] = eligible["mean_residual"] / eligible["se_mean"]

    n_significant = (eligible["z_score"].abs() > 1.96).sum()
    expected = len(eligible) * 0.05

    logger.info("Interaction test: %d crew pairs with >= %d games", len(eligible), min_games)
    logger.info("  Pairs with |z| > 1.96: %d (expected %.1f at 5%%)", n_significant, expected)
    logger.info("  Mean |z|: %.3f", eligible["z_score"].abs().mean())

    top_interactions = eligible.nlargest(10, "z_score")
    worst_interactions = eligible.nsmallest(10, "z_score")

    logger.info("\nTop amplifier pairs (positive residual = more SF than additive predicts):")
    for _, row in top_interactions.iterrows():
        logger.info(
            "  %s: mean_resid=%+.2f, z=%.2f, n=%d",
            row["pair_key"], row["mean_residual"], row["z_score"], row["n_games"],
        )

    logger.info("\nTop suppressor pairs (negative residual = fewer SF than additive predicts):")
    for _, row in worst_interactions.iterrows():
        logger.info(
            "  %s: mean_resid=%+.2f, z=%.2f, n=%d",
            row["pair_key"], row["mean_residual"], row["z_score"], row["n_games"],
        )

    interaction_path = MODEL_DIR / "crew_interactions.parquet"
    eligible.to_parquet(interaction_path, index=False)
    logger.info("Wrote crew interaction analysis to %s", interaction_path)

    return eligible


def diagnose(dataset_path: Path | None = None) -> None:
    """Print signal diagnostics comparing static vs temporal profiles."""
    paths = {
        "static": dataset_path or DATASET_PATH,
        "temporal": DATASET_TEMPORAL_PATH,
    }

    print("\n  Step 5 Signal Diagnostics")
    print("  " + "=" * 80)

    for label, path in paths.items():
        if not path.exists():
            print(f"\n  [{label}] dataset not found at {path}")
            continue

        ds = pd.read_parquet(path)
        model = ds[ds["season"].isin(MODEL_SEASONS)].copy()
        train = model[model["season"].isin(TRAIN_SEASONS)]
        test = model[model["season"].isin(TEST_SEASONS)]

        print(f"\n  [{label.upper()}] {len(model)} games (train={len(train)}, test={len(test)})")

        if "crew_sum_sf_per_game" in model.columns:
            for split_name, split in [("train", train), ("test", test), ("all", model)]:
                corr = split["sf_count"].corr(split["crew_sum_sf_per_game"])
                base_rmse = np.sqrt(((split["sf_count"] - train["sf_count"].mean()) ** 2).mean())
                sum_rmse = np.sqrt(((split["sf_count"] - split["crew_sum_sf_per_game"]) ** 2).mean())
                print(
                    f"    {split_name:5s}: r(sf_count, crew_sum_sf)={corr:+.3f}  "
                    f"RMSE_sum={sum_rmse:.3f}  RMSE_baseline={base_rmse:.3f}"
                )

        if "crew_mean_sf_pct_of_fouls" in model.columns:
            r_rate = model["sf_rate"].corr(model["crew_mean_sf_pct_of_fouls"])
            print(f"    all: r(sf_rate, crew_sf_pct)={r_rate:+.3f}")

        if "crew_mean_suppressor_score" in model.columns:
            r_sup = model["sf_count"].corr(model["crew_mean_suppressor_score"])
            print(f"    all: r(sf_count, crew_suppressor)={r_sup:+.3f}")

    for eval_label, eval_path in [
        ("static", EVALUATION_PATH),
        ("temporal", EVALUATION_TEMPORAL_PATH),
    ]:
        if eval_path.exists():
            print(f"\n  {eval_label.title()} evaluation (test holdout):")
            eval_df = pd.read_parquet(eval_path)
            for _, row in eval_df.iterrows():
                print(
                    f"    {row['label']:<40} RMSE={row['rmse']:.3f}  "
                    f"R²={row['r2']:.4f}  r={row['corr']:.4f}"
                )


def print_summary(temporal: bool = False) -> None:
    """Print evaluation results and top predictions."""
    eval_path = EVALUATION_TEMPORAL_PATH if temporal else EVALUATION_PATH
    pred_path = PREDICTIONS_TEMPORAL_PATH if temporal else PREDICTIONS_PATH
    if not eval_path.exists():
        raise RuntimeError(f"No evaluation at {eval_path}. Run 'build' first.")

    eval_df = pd.read_parquet(eval_path)
    label = "Temporal" if temporal else "Static"
    print(f"\n  Model Evaluation ({label}, Test Set)")
    print("  " + "=" * 80)
    print(f"  {'Model':<40} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'r':>8}")
    print("  " + "-" * 80)
    for _, row in eval_df.iterrows():
        print(
            f"  {row['label']:<40} {row['rmse']:>8.3f} {row['mae']:>8.3f} "
            f"{row['r2']:>8.4f} {row['corr']:>8.4f}"
        )

    count_eval = eval_df[~eval_df["label"].str.contains("rate", case=False)]
    best_count = count_eval.loc[count_eval["rmse"].idxmin()]
    print(
        f"\n  Best SF-count model: {best_count['label']} "
        f"(RMSE={best_count['rmse']:.3f}, R²={best_count['r2']:.4f})"
    )

    if pred_path.exists():
        preds = pd.read_parquet(pred_path)
        print(f"\n  Prediction sample (n={len(preds)} test games):")
        print(f"  {'Game ID':<14} {'Season':>8} {'Type':>4} {'Actual':>7} {'Predicted':>10} {'Residual':>9}")
        print("  " + "-" * 60)
        for _, row in preds.head(20).iterrows():
            print(
                f"  {row['game_id']:<14} {row['season']:>8} {row['season_type']:>4} "
                f"{row['sf_count']:>7} {row['predicted_sf']:>10.1f} {row['residual']:>+9.1f}"
            )

    if MODEL_DIR.joinpath("crew_interactions.parquet").exists():
        interactions = pd.read_parquet(MODEL_DIR / "crew_interactions.parquet")
        n_sig = (interactions["z_score"].abs() > 1.96).sum()
        expected = len(interactions) * 0.05
        print(f"\n  Crew Interaction Test:")
        print(f"  Pairs analyzed: {len(interactions)} (>= 20 shared games)")
        print(f"  Significant pairs (|z| > 1.96): {n_sig} (expected {expected:.1f} at 5%)")
        if n_sig > expected * 1.5:
            print("  → Evidence of crew interaction effects beyond additive model")
        else:
            print("  → No strong evidence of interaction effects; additive model sufficient")


def main() -> None:
    parser = argparse.ArgumentParser(description="Crew predictive model (Step 5)")
    parser.add_argument(
        "command",
        choices=["build", "summary", "interactions", "diagnose"],
        help="build=dataset+train+eval, summary=print results, interactions=test crew effects",
    )
    parser.add_argument("--no-train", action="store_true", help="Build dataset only, skip training")
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Use prior-season official profiles (honest holdout, no leakage)",
    )
    parser.add_argument(
        "--train-only-interactions",
        action="store_true",
        help="Limit interaction test to train seasons only",
    )
    args = parser.parse_args()

    dataset_path = DATASET_TEMPORAL_PATH if args.temporal else DATASET_PATH

    if args.command == "build":
        dataset = build_game_crew_dataset(temporal=args.temporal)
        if not args.no_train:
            train_and_evaluate(dataset, dataset_path=dataset_path, temporal=args.temporal)
            print_summary(temporal=args.temporal)
            test_interactions(
                dataset,
                dataset_path=dataset_path,
                use_train_only=args.train_only_interactions,
                temporal=args.temporal,
            )
    elif args.command == "summary":
        print_summary(temporal=args.temporal)
    elif args.command == "interactions":
        test_interactions(
            dataset_path=dataset_path,
            use_train_only=args.train_only_interactions,
            temporal=args.temporal,
        )
    elif args.command == "diagnose":
        diagnose()


if __name__ == "__main__":
    main()
