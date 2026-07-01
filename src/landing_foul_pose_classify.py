"""Step 10e (Phase 3): Pose-based landing foul classifier.

Trains/evaluates a classifier on the geometric pose features from
landing_foul_pose_features.py, using the same fixed train/val split and quality
gate (precision >= 0.85, recall >= 0.70 on the 57-clip val YES class) as the
VideoMAE track.

Three modes (plan priority order):
  rules    Transparent rule-based classifier (zero training data). Thresholds are
           tuned on the TRAIN set only and evaluated on val.
  xgboost  Gradient-boosted trees with 5-fold stratified CV on train for
           hyperparameter selection, trained on full train, evaluated on val.
           Reports feature importances (Paper 2 interpretability).
  cv       Cross-validated metrics on the TRAIN set only (no val peeking) for a
           more stable estimate than the 57-clip val.

A threshold sweep on val YES-probability is printed (precision is the binding
gate). Predictions + probabilities are saved for Phase 4 ensembling.

Usage:
  PYTHONPATH=. python src/landing_foul_pose_classify.py --mode rules
  PYTHONPATH=. python src/landing_foul_pose_classify.py --mode xgboost
  PYTHONPATH=. python src/landing_foul_pose_classify.py --mode cv
  PYTHONPATH=. python src/landing_foul_pose_classify.py --evaluate-only

Output:
  data/processed/landing_foul_pose_model.json       (config + metrics + predictions)
  data/processed/landing_foul_pose_model.xgb        (saved xgboost model, gitignored)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURES_PATH = config.PROCESSED_DIR / "landing_foul_pose_features.npz"
MODEL_JSON_PATH = config.PROCESSED_DIR / "landing_foul_pose_model.json"
MODEL_XGB_PATH = config.PROCESSED_DIR / "landing_foul_pose_model.xgb"

PRECISION_GATE = 0.85
RECALL_GATE = 0.70
DEFAULT_THRESHOLD = 0.5

# Features excluded from learned-classifier input (meta / leakage risks).
EXCLUDED_FEATURES = {"has_missing_data", "role_assignment_confidence", "shooter_landing_frame_offset"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> dict[str, Any]:
    d = np.load(FEATURES_PATH, allow_pickle=True)
    names = list(d["feature_names"])
    X = d["features"].astype(float)
    y = d["labels"].astype(int)
    split = np.asarray(d["split"]).astype(str)
    keys = list(d["keys"])
    return {"X": X, "y": y, "names": names, "split": split, "keys": keys}


def model_features(names: list[str]) -> tuple[list[int], list[str]]:
    idx = [i for i, n in enumerate(names) if n not in EXCLUDED_FEATURES]
    return idx, [names[i] for i in idx]


def split_indices(split: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "train": np.where(split == "train")[0],
        "val": np.where(split == "val")[0],
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    c = confusion(y_true, y_pred)
    prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": float(np.mean(y_pred == y_true)), **c}


def report_metrics(label: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    m = prf(y_true, y_pred)
    logger.info(
        "%s: P=%.3f R=%.3f F1=%.3f acc=%.3f  (tp=%d fp=%d fn=%d tn=%d)",
        label, m["precision"], m["recall"], m["f1"], m["accuracy"],
        m["tp"], m["fp"], m["fn"], m["tn"],
    )
    p_pass = m["precision"] >= PRECISION_GATE
    r_pass = m["recall"] >= RECALL_GATE
    logger.info("  gate: precision %s, recall %s", "PASS" if p_pass else "MISS", "PASS" if r_pass else "MISS")
    return m


def threshold_sweep(y_true: np.ndarray, proba: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        pred = (proba >= t).astype(int)
        m = prf(y_true, pred)
        out.append({"threshold": t, **m})
        logger.info("  t=%.2f  P=%.3f R=%.3f F1=%.3f (tp=%d fp=%d fn=%d tn=%d)",
                    t, m["precision"], m["recall"], m["f1"], m["tp"], m["fp"], m["fn"], m["tn"])
    return out


# ---------------------------------------------------------------------------
# Rule-based classifier (Option A). Thresholds tuned on TRAIN only.
# ---------------------------------------------------------------------------


def _tune_rules(Xtr: np.ndarray, ytr: np.ndarray, names: list[str], feat_idx: list[int]) -> dict[str, float]:
    """Grid-search simple thresholds on train to maximize F1, constrained to
    the basketball-expected direction (higher zone incursion => YES)."""
    nm = {names[i]: i for i in feat_idx}

    def col(name: str) -> np.ndarray:
        return Xtr[:, nm[name]]

    zone = col("defender_ankle_in_zone_frac")
    overlap = col("overlap_duration_frames")
    min_dist = col("min_ankle_distance")
    contact_h = col("contact_height")

    best = None
    for z_t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        for o_t in [1, 2, 3, 4]:
            for d_t in [0.20, 0.30, 0.40, 0.50]:
                for ch_t in [0.5, 0.7, 0.9, 1.1]:
                    pred = ((zone > z_t) | (overlap >= o_t) | ((min_dist < d_t) & (contact_h < ch_t))).astype(int)
                    m = prf(ytr, pred)
                    score = m["f1"] - 0.1 * max(0, 0.6 - m["precision"])  # lean toward precision
                    if best is None or score > best[0]:
                        best = (score, {"zone": z_t, "overlap": o_t, "min_dist": d_t, "contact_h": ch_t})
    return best[1]


def apply_rules(X: np.ndarray, names: list[str], feat_idx: list[int], t: dict[str, float]) -> np.ndarray:
    nm = {names[i]: i for i in feat_idx}
    zone = X[:, nm["defender_ankle_in_zone_frac"]]
    overlap = X[:, nm["overlap_duration_frames"]]
    min_dist = X[:, nm["min_ankle_distance"]]
    contact_h = X[:, nm["contact_height"]]
    return (
        (zone > t["zone"])
        | (overlap >= t["overlap"])
        | ((min_dist < t["min_dist"]) & (contact_h < t["contact_h"]))
    ).astype(int)


def run_rules(data: dict[str, Any]) -> None:
    idx, sub_names = model_features(data["names"])
    sp = split_indices(data["split"])
    tr, va = sp["train"], sp["val"]
    Xtr, ytr = data["X"][tr][:, idx], data["y"][tr]
    Xva, yva = data["X"][va][:, idx], data["y"][va]

    thresholds = _tune_rules(Xtr, ytr, data["names"], idx)
    logger.info("Tuned rule thresholds (train): %s", thresholds)

    train_metrics = report_metrics("train", ytr, apply_rules(Xtr, data["names"], idx, thresholds))
    val_metrics = report_metrics("val", yva, apply_rules(Xva, data["names"], idx, thresholds))

    out = {
        "mode": "rules",
        "thresholds": thresholds,
        "train": train_metrics,
        "val": val_metrics,
        "gate": {"precision": PRECISION_GATE, "recall": RECALL_GATE},
    }
    MODEL_JSON_PATH.write_text(json.dumps(out, indent=2))
    logger.info("Wrote %s", MODEL_JSON_PATH.name)


# ---------------------------------------------------------------------------
# XGBoost classifier (Option B)
# ---------------------------------------------------------------------------


def run_xgboost(data: dict[str, Any], threshold: float = DEFAULT_THRESHOLD, cv_folds: int = 5) -> None:
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold

    idx, sub_names = model_features(data["names"])
    sp = split_indices(data["split"])
    tr, va = sp["train"], sp["val"]
    Xtr, ytr = data["X"][tr][:, idx], data["y"][tr]
    Xva, yva = data["X"][va][:, idx], data["y"][va]

    # Conservative hyperparameters for n=227, ~19 features (plan guidance).
    base_params = {
        "max_depth": 3,
        "n_estimators": 200,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "verbosity": 0,
        "n_jobs": -1,
        "random_state": 42,
    }

    # 5-fold CV on train for a stable estimate (no val peeking).
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_folds_metrics = []
    oof_proba = np.zeros(len(ytr))
    for fold, (fit_idx, oof_idx) in enumerate(skf.split(Xtr, ytr), 1):
        # Rough class balance weighting.
        pos = float(ytr[fit_idx].sum()); neg = float(len(fit_idx) - pos)
        spw = neg / max(1.0, pos)
        params = {**base_params, "scale_pos_weight": spw}
        model = xgb.XGBClassifier(**params)
        model.fit(Xtr[fit_idx], ytr[fit_idx])
        p = model.predict_proba(Xtr[oof_idx])[:, 1]
        oof_proba[oof_idx] = p
        cv_folds_metrics.append(prf(ytr[oof_idx], (p >= threshold).astype(int)))
    cv_oof = report_metrics(f"train OOF ({cv_folds}-fold)", ytr, (oof_proba >= threshold).astype(int))

    # Final model on full train.
    pos = float(ytr.sum()); neg = float(len(ytr) - pos)
    final_params = {**base_params, "scale_pos_weight": neg / max(1.0, pos)}
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(Xtr, ytr)

    train_metrics = report_metrics("train", ytr, (final_model.predict_proba(Xtr)[:, 1] >= threshold).astype(int))
    val_proba = final_model.predict_proba(Xva)[:, 1]
    val_metrics = report_metrics("val", yva, (val_proba >= threshold).astype(int))

    logger.info("Val threshold sweep:")
    sweep = threshold_sweep(yva, val_proba)

    # Feature importances.
    imp = final_model.get_booster().get_score(importance_type="gain")
    # Map f0/f1/.. back to names.
    fi = {sub_names[int(k[1:])]: float(v) for k, v in imp.items()}
    fi_sorted = sorted(fi.items(), key=lambda t: t[1], reverse=True)
    logger.info("Top feature importances (gain):")
    for name, v in fi_sorted[:10]:
        logger.info("  %-32s %.2f", name, v)

    # Save predictions for ensembling.
    preds = {
        "train_keys": [data["keys"][i] for i in tr],
        "train_proba": oof_proba.tolist(),  # OOF to avoid train leakage in ensemble
        "val_keys": [data["keys"][i] for i in va],
        "val_proba": val_proba.tolist(),
    }

    out = {
        "mode": "xgboost",
        "threshold": threshold,
        "params": final_params,
        "features": sub_names,
        "cv_oof": cv_oof,
        "train": train_metrics,
        "val": val_metrics,
        "val_threshold_sweep": sweep,
        "feature_importance_gain": dict(fi_sorted),
        "predictions": preds,
        "gate": {"precision": PRECISION_GATE, "recall": RECALL_GATE},
    }
    MODEL_JSON_PATH.write_text(json.dumps(out, indent=2))
    final_model.save_model(str(MODEL_XGB_PATH))
    logger.info("Wrote %s + %s", MODEL_JSON_PATH.name, MODEL_XGB_PATH.name)


# ---------------------------------------------------------------------------
# Cross-validated train-only estimate (no val peeking)
# ---------------------------------------------------------------------------


def run_cv(data: dict[str, Any], cv_folds: int = 5, threshold: float = DEFAULT_THRESHOLD) -> None:
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold

    idx, _ = model_features(data["names"])
    sp = split_indices(data["split"])
    tr = sp["train"]
    Xtr, ytr = data["X"][tr][:, idx], data["y"][tr]

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    all_metrics = []
    for fold, (fit_idx, oof_idx) in enumerate(skf.split(Xtr, ytr), 1):
        pos = float(ytr[fit_idx].sum()); neg = float(len(fit_idx) - pos)
        model = xgb.XGBClassifier(
            max_depth=3, n_estimators=200, learning_rate=0.05, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            verbosity=0, n_jobs=-1, random_state=42, scale_pos_weight=neg / max(1.0, pos),
        )
        model.fit(Xtr[fit_idx], ytr[fit_idx])
        p = model.predict_proba(Xtr[oof_idx])[:, 1]
        m = prf(ytr[oof_idx], (p >= threshold).astype(int))
        all_metrics.append(m)
        logger.info("fold %d: P=%.3f R=%.3f F1=%.3f", fold, m["precision"], m["recall"], m["f1"])
    agg = {
        "precision": float(np.mean([m["precision"] for m in all_metrics])),
        "recall": float(np.mean([m["recall"] for m in all_metrics])),
        "f1": float(np.mean([m["f1"] for m in all_metrics])),
    }
    logger.info("CV mean (k=%d): P=%.3f R=%.3f F1=%.3f", cv_folds, agg["precision"], agg["recall"], agg["f1"])


# ---------------------------------------------------------------------------
# Evaluate saved model
# ---------------------------------------------------------------------------


def run_evaluate(data: dict[str, Any], threshold: float = DEFAULT_THRESHOLD) -> None:
    import xgboost as xgb

    if not MODEL_XGB_PATH.exists():
        raise SystemExit(f"No saved model at {MODEL_XGB_PATH}; run --mode xgboost first")
    idx, sub_names = model_features(data["names"])
    sp = split_indices(data["split"])
    tr, va = sp["train"], sp["val"]
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_XGB_PATH))
    report_metrics("train", data["y"][tr], (model.predict_proba(data["X"][tr][:, idx])[:, 1] >= threshold).astype(int))
    val_proba = model.predict_proba(data["X"][va][:, idx])[:, 1]
    report_metrics("val", data["y"][va], (val_proba >= threshold).astype(int))
    logger.info("Val threshold sweep:")
    threshold_sweep(data["y"][va], val_proba)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 10e Phase 3: pose-based classifier")
    parser.add_argument("--mode", default="xgboost", choices=["rules", "xgboost", "cv"], help="Classifier mode")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate the saved xgboost model")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="YES probability threshold")
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds for xgboost/cv modes")
    args = parser.parse_args()

    data = load_data()
    if args.evaluate_only:
        run_evaluate(data, threshold=args.threshold)
    elif args.mode == "rules":
        run_rules(data)
    elif args.mode == "xgboost":
        run_xgboost(data, threshold=args.threshold, cv_folds=args.cv_folds)
    elif args.mode == "cv":
        run_cv(data, cv_folds=args.cv_folds, threshold=args.threshold)


if __name__ == "__main__":
    main()
