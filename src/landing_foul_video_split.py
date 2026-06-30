"""Create stratified train/val split for landing foul video classification.

Reads the embeddings file (which contains only YES/NO labels, UNCLEAR excluded)
and produces a reproducible 80/20 stratified split.

Usage:
    python src/landing_foul_video_split.py
    python src/landing_foul_video_split.py --test-size 0.25 --seed 7

Output: data/processed/landing_foul_split.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EMBEDDINGS_PATH = config.PROCESSED_DIR / "landing_foul_embeddings.npz"
SPLIT_PATH = config.PROCESSED_DIR / "landing_foul_split.json"


def create_split(test_size: float = 0.20, seed: int = 42) -> dict:
    """Create stratified train/val split from embeddings file."""
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    labels = data["labels"]
    game_ids = data["game_ids"]
    event_ids = data["event_ids"]

    n_total = len(labels)
    n_pos = int(labels.sum())
    n_neg = n_total - n_pos
    logger.info("Total clips: %d (YES=%d, NO=%d)", n_total, n_pos, n_neg)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(n_total), labels))

    train_keys = [
        {"game_id": str(game_ids[i]), "event_id": int(event_ids[i])}
        for i in train_idx
    ]
    val_keys = [
        {"game_id": str(game_ids[i]), "event_id": int(event_ids[i])}
        for i in val_idx
    ]

    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    split = {
        "seed": seed,
        "test_size": test_size,
        "n_total": n_total,
        "train": {
            "n": len(train_idx),
            "n_yes": int(train_labels.sum()),
            "n_no": int(len(train_labels) - train_labels.sum()),
            "indices": train_idx.tolist(),
            "keys": train_keys,
        },
        "val": {
            "n": len(val_idx),
            "n_yes": int(val_labels.sum()),
            "n_no": int(len(val_labels) - val_labels.sum()),
            "indices": val_idx.tolist(),
            "keys": val_keys,
        },
    }

    logger.info(
        "Train: %d (YES=%d, NO=%d) | Val: %d (YES=%d, NO=%d)",
        split["train"]["n"], split["train"]["n_yes"], split["train"]["n_no"],
        split["val"]["n"], split["val"]["n_yes"], split["val"]["n_no"],
    )

    return split


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stratified train/val split")
    parser.add_argument("--test-size", type=float, default=0.20, help="Fraction held out for val (default: 0.20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    split = create_split(test_size=args.test_size, seed=args.seed)

    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_PATH, "w") as f:
        json.dump(split, f, indent=2)

    logger.info("Wrote split to %s", SPLIT_PATH)


if __name__ == "__main__":
    main()
