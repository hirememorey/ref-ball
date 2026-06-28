"""Three-track analysis of per-official shooting foul profiles.

Track A: Descriptive — "The Referee Landscape"
Track B: Mechanism — "The Choke Referee"
Track C: Causal — "The Playoff Whistle"

Input:  data/processed/ref_profiles.parquet  (from ref_profiles.py)
        data/processed/games/*.parquet       (from ingest.py)
        does-harden-choke collapse game data
Output: output/figures/ + output/tables/

Usage:
    python src/analyze.py
    python src/analyze.py --track A
    python src/analyze.py --track B
    python src/analyze.py --track C
"""

from __future__ import annotations

import argparse
import logging

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_track_a() -> None:
    raise NotImplementedError("Track A: Descriptive analysis")


def run_track_b() -> None:
    raise NotImplementedError("Track B: Mechanism analysis (choke referee)")


def run_track_c() -> None:
    raise NotImplementedError("Track C: Causal analysis (playoff whistle)")


def main():
    parser = argparse.ArgumentParser(description="Analyze per-official shooting foul profiles")
    parser.add_argument("--track", choices=["A", "B", "C"], default=None,
                        help="Run a single track (default: all)")
    args = parser.parse_args()

    if args.track is None or args.track == "A":
        logger.info("Running Track A: Descriptive")
        run_track_a()
    if args.track is None or args.track == "B":
        logger.info("Running Track B: Mechanism")
        run_track_b()
    if args.track is None or args.track == "C":
        logger.info("Running Track C: Causal")
        run_track_c()


if __name__ == "__main__":
    main()
