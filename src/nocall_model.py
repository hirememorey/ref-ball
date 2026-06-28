"""Train a binary no-call classifier and run inference across all games.

Layer 3 of the ref-ball dataset. Binary problem: foul or no foul.
Trains on NBA API labels (every PBP event is already tagged).

Training data: video clips from videoeventsasset API, labeled by PBP event type.
  - Called fouls (S.FOUL, P.FOUL) = positive class
  - Non-foul events (shots, turnovers, rebounds) = negative class

Validation: L2M INC labels (league-audited ground truth for no-calls).

Inference: run model across all non-foul events in a game.
  High-confidence "foul" predictions on non-foul events = predicted no-calls.

Output: data/processed/nocalls.parquet

Usage:
    python src/nocall_model.py train
    python src/nocall_model.py predict
    python src/nocall_model.py validate
"""

from __future__ import annotations

import argparse
import logging

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train() -> None:
    raise NotImplementedError("Train binary classifier on API labels")


def predict() -> None:
    raise NotImplementedError("Run inference across all non-foul events")


def validate() -> None:
    raise NotImplementedError("Validate against L2M INC labels")


def main():
    parser = argparse.ArgumentParser(description="No-call detection model")
    parser.add_argument("command", choices=["train", "predict", "validate"])
    args = parser.parse_args()

    if args.command == "train":
        train()
    elif args.command == "predict":
        predict()
    elif args.command == "validate":
        validate()


if __name__ == "__main__":
    main()
