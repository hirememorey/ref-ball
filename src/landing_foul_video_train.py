"""Train and evaluate a classifier on frozen VideoMAE embeddings.

Proof-of-concept: can frozen spatiotemporal features distinguish landing fouls
from standard contests? Trains logistic regression (and optionally a small MLP)
on the 80/20 split from landing_foul_video_split.py.

Usage:
    python src/landing_foul_video_train.py
    python src/landing_foul_video_train.py --model mlp
    python src/landing_foul_video_train.py --cv 5

Output: prints classification report + precision/recall on YES class.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EMBEDDINGS_PATH = config.PROCESSED_DIR / "landing_foul_embeddings.npz"
SPLIT_PATH = config.PROCESSED_DIR / "landing_foul_split.json"
GROUND_TRUTH_PATH = config.DATA_DIR / "landing_foul_ground_truth.csv"


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load embeddings and split into train/val arrays."""
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    game_ids = data["game_ids"]
    event_ids = data["event_ids"]

    with open(SPLIT_PATH) as f:
        split = json.load(f)

    train_idx = np.array(split["train"]["indices"])
    val_idx = np.array(split["val"]["indices"])

    X_train = embeddings[train_idx]
    y_train = labels[train_idx]
    X_val = embeddings[val_idx]
    y_val = labels[val_idx]

    meta = {
        "game_ids": game_ids,
        "event_ids": event_ids,
        "val_idx": val_idx,
    }

    return X_train, y_train, X_val, y_val, meta


def build_model(model_type: str = "logreg"):
    """Build a classifier head."""
    if model_type == "logreg":
        return LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=42,
        )
    elif model_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(256, 64),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_and_evaluate(model_type: str = "logreg") -> None:
    """Train on train split, evaluate on val split."""
    X_train, y_train, X_val, y_val, meta = load_data()

    logger.info("Train: %d samples (YES=%d, NO=%d)", len(y_train), y_train.sum(), len(y_train) - y_train.sum())
    logger.info("Val:   %d samples (YES=%d, NO=%d)", len(y_val), y_val.sum(), len(y_val) - y_val.sum())

    # Standardize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Train
    clf = build_model(model_type)
    clf.fit(X_train_s, y_train)

    # Evaluate
    y_pred = clf.predict(X_val_s)
    y_prob = clf.predict_proba(X_val_s)[:, 1] if hasattr(clf, "predict_proba") else None

    print(f"\n{'='*60}")
    print(f"Model: {model_type} | Train: {len(y_train)} | Val: {len(y_val)}")
    print(f"{'='*60}")
    print(classification_report(y_val, y_pred, target_names=["NO", "YES"]))

    prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, pos_label=1, average="binary")
    print(f"YES Precision: {prec:.3f}  (target: >=0.85)")
    print(f"YES Recall:    {rec:.3f}  (target: >=0.70)")
    print(f"YES F1:        {f1:.3f}")
    print()

    if prec >= 0.85:
        print(">> PASSES precision gate (>=85%). Fine-tuning likely to improve further.")
    elif prec >= 0.70:
        print(">> PROMISING (70-84%). Fine-tuning or temporal cropping may clear the gate.")
    elif prec >= 0.55:
        print(">> MARGINAL (55-69%). Signal exists but frozen features may not be enough.")
    else:
        print(">> BELOW BASELINE (<55%). Frozen VideoMAE features may not encode the right signal.")
        print("   Consider: temporal cropping, spatial cropping, or a different backbone.")

    # Error analysis: print val misclassifications with ground truth notes
    _error_analysis(y_val, y_pred, meta)


def _error_analysis(y_val: np.ndarray, y_pred: np.ndarray, meta: dict) -> None:
    """Print misclassified clips with their ground truth notes."""
    import pandas as pd

    gt = pd.read_csv(GROUND_TRUTH_PATH)
    gt["game_id"] = gt["game_id"].astype(str).str.zfill(10)
    gt["event_id"] = gt["event_id"].astype(int)
    gt_lookup = {(r["game_id"], r["event_id"]): r for _, r in gt.iterrows()}

    val_idx = meta["val_idx"]
    game_ids = meta["game_ids"]
    event_ids = meta["event_ids"]

    fps = []  # False positives (predicted YES, actual NO)
    fns = []  # False negatives (predicted NO, actual YES)

    for i, vi in enumerate(val_idx):
        if y_pred[i] != y_val[i]:
            gid = str(game_ids[vi])
            eid = int(event_ids[vi])
            row = gt_lookup.get((gid, eid), {})
            entry = {
                "game_id": gid,
                "event_id": eid,
                "actual": "YES" if y_val[i] == 1 else "NO",
                "predicted": "YES" if y_pred[i] == 1 else "NO",
                "description": row.get("description", ""),
                "note": row.get("note", ""),
            }
            if y_pred[i] == 1 and y_val[i] == 0:
                fps.append(entry)
            else:
                fns.append(entry)

    if fps:
        print(f"\n--- False Positives ({len(fps)}) — predicted YES, actually NO ---")
        for e in fps:
            note = f' | note: {e["note"]}' if e["note"] else ""
            print(f'  {e["game_id"]}_{e["event_id"]}: {e["description"][:70]}{note}')

    if fns:
        print(f"\n--- False Negatives ({len(fns)}) — predicted NO, actually YES ---")
        for e in fns:
            note = f' | note: {e["note"]}' if e["note"] else ""
            print(f'  {e["game_id"]}_{e["event_id"]}: {e["description"][:70]}{note}')


def cross_validate(model_type: str = "logreg", n_folds: int = 5) -> None:
    """Run k-fold cross-validation on the full dataset for a more stable estimate."""
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    X = data["embeddings"]
    y = data["labels"]

    logger.info("Running %d-fold CV on %d samples (YES=%d, NO=%d)", n_folds, len(y), y.sum(), len(y) - y.sum())

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = build_model(model_type)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X_s, y, cv=cv)

    print(f"\n{'='*60}")
    print(f"{n_folds}-Fold Cross-Validation | Model: {model_type} | N={len(y)}")
    print(f"{'='*60}")
    print(classification_report(y, y_pred, target_names=["NO", "YES"]))

    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, pos_label=1, average="binary")
    print(f"YES Precision: {prec:.3f}  (target: >=0.85)")
    print(f"YES Recall:    {rec:.3f}  (target: >=0.70)")
    print(f"YES F1:        {f1:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classifier on frozen VideoMAE embeddings")
    parser.add_argument("--model", default="logreg", choices=["logreg", "mlp"], help="Classifier head type")
    parser.add_argument("--cv", type=int, default=0, help="Run k-fold CV instead of single split (0=off)")
    args = parser.parse_args()

    if args.cv > 0:
        cross_validate(model_type=args.model, n_folds=args.cv)
    else:
        train_and_evaluate(model_type=args.model)


if __name__ == "__main__":
    main()
