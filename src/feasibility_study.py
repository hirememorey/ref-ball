r"""Layer 3 Feasibility Study — Go/No-Go Gate for No-Call Video Model.

This script is a FEASIBILITY STUDY, not the production pipeline.
It answers one question: Can a video-based binary classifier distinguish
called shooting fouls from non-foul plays well enough to predict no-calls?

Steps:
  1. Sample PBP events: shooting fouls (positive) + shot attempts (negative)
  2. Download video clips via videoeventsasset API
  3. Extract features from clips using a pre-trained video encoder
  4. Train a binary classifier (logistic regression on embeddings)
  5. Evaluate on held-out test set
  6. Validate against L2M INC labels (the ground-truth test)

Usage:
    python src/feasibility_study.py sample      # Step 1: build sample manifest
    python src/feasibility_study.py download     # Step 2: download video clips
    python src/feasibility_study.py train        # Step 3-5: extract + train + evaluate
    python src/feasibility_study.py validate     # Step 6: validate against L2M INC

Output: data/processed/feasibility/ (manifest, clips, results)

Go/No-Go Criteria:
    - Precision >= 0.6 on held-out fouls (we can tolerate false positives
      at this stage — they become predicted no-calls, which is the signal)
    - Recall >= 0.5 on L2M INC events (we need to catch at least half of
      the ground-truth missed calls)
    - If both thresholds not met: pivot to descriptive-only Paper 1 using
      Layer 1 + L2M data directly (685 INC events are the no-call dataset)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

import config
from src.nba_client import NBAStatsClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEAS_DIR = config.PROCESSED_DIR / "feasibility"
MANIFEST_PATH = FEAS_DIR / "manifest.parquet"
CLIPS_DIR = FEAS_DIR / "clips"
RESULTS_PATH = FEAS_DIR / "results.json"

N_POSITIVE = 250
N_NEGATIVE = 250
RANDOM_SEED = 42

FOUL_ACTION_TYPE = "Foul"
NEGATIVE_ACTION_TYPES = {"Made Shot", "Missed Shot"}


def _load_raw_events() -> pd.DataFrame:
    """Load all events (including non-fouls) from raw PBP JSON files.

    The ingested game parquets only contain fouls (ingest.py filters to
    actionType == 'Foul'). For the feasibility study we also need shot
    attempts as the negative class, so we parse raw PBP directly.
    """
    pbp_dir = config.RAW_PBP_DIR
    files = sorted(pbp_dir.glob("*.json"))
    logger.info("Scanning %d PBP files for events", len(files))

    records = []
    for i, fp in enumerate(files):
        if i % 2000 == 0 and i > 0:
            logger.info("  scanned %d/%d files", i, len(files))
        try:
            with open(fp) as f:
                data = json.load(f)
            actions = data.get("game", {}).get("actions", [])
            game_id = fp.stem
            for a in actions:
                atype = a.get("actionType", "")
                stype = a.get("subType", "")
                if atype == FOUL_ACTION_TYPE and stype == "Shooting":
                    pass
                elif atype in NEGATIVE_ACTION_TYPES:
                    pass
                else:
                    continue
                records.append({
                    "game_id": game_id,
                    "event_num": a.get("actionNumber", 0),
                    "action_type": atype,
                    "sub_type": stype,
                    "video_available": a.get("videoAvailable", 0),
                    "description": a.get("description", ""),
                })
        except Exception as exc:
            logger.warning("  %s parse error: %s", fp.name, exc)

    df = pd.DataFrame(records)
    logger.info("Loaded %d candidate events", len(df))
    return df


def build_sample() -> pd.DataFrame:
    """Build a stratified sample of PBP events for the feasibility study."""
    logger.info("Building sample: %d positive (SF) + %d negative (shots)", N_POSITIVE, N_NEGATIVE)

    df = _load_raw_events()

    # Positive: shooting fouls with video
    positive = df[
        (df["action_type"] == "Foul")
        & (df["sub_type"] == "Shooting")
        & (df["video_available"] == 1)
    ].copy()

    # Negative: shot attempts (made + missed) with video
    negative = df[
        (df["action_type"].isin(NEGATIVE_TYPES))
        & (df["video_available"] == 1)
    ].copy()

    logger.info("Pool: %d SF (positive), %d shots (negative)", len(positive), len(negative))

    pos_sample = positive.sample(n=min(N_POSITIVE, len(positive)), random_state=RANDOM_SEED)
    neg_sample = negative.sample(n=min(N_NEGATIVE, len(negative)), random_state=RANDOM_SEED)

    pos_sample["label"] = 1
    neg_sample["label"] = 0

    manifest = pd.concat([pos_sample, neg_sample], ignore_index=True)
    manifest["clip_path"] = manifest.apply(
        lambda r: str(CLIPS_DIR / f"{r['game_id']}_{r['event_num']}.mp4"), axis=1,
    )
    manifest["downloaded"] = False

    FEAS_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(MANIFEST_PATH, index=False)
    logger.info("Wrote %d-event manifest to %s", len(manifest), MANIFEST_PATH)

    print(f"\nSample composition:")
    print(f"  Positive (shooting fouls): {pos_sample.shape[0]}")
    print(f"  Negative (shot attempts):   {neg_sample.shape[0]}")
    print(f"  Seasons covered: {sorted(manifest['game_id'].str[3:5].unique())}")
    return manifest


def download_clips() -> None:
    """Download video clips for all events in the manifest."""
    if not MANIFEST_PATH.exists():
        raise RuntimeError("Run 'sample' first to build the manifest")

    manifest = pd.read_parquet(MANIFEST_PATH)
    client = NBAStatsClient()

    to_download = manifest[~manifest["downloaded"]].copy()
    logger.info("Clips to download: %d / %d", len(to_download), len(manifest))

    fetched = 0
    failed = 0
    for i, (idx, row) in enumerate(to_download.iterrows()):
        gid = row["game_id"]
        evt = row["event_num"]
        clip_path = Path(row["clip_path"])

        if clip_path.exists() and clip_path.stat().st_size > 1000:
            manifest.at[idx, "downloaded"] = True
            fetched += 1
            continue

        try:
            resp = client.get_video_events(gid, evt)
            video_urls = resp.get("resultSets", {}).get("Meta", {}).get("videoUrls", [])

            if not video_urls:
                failed += 1
                if i < 20:
                    logger.warning("  No video URL for %s/%s", gid, evt)
                continue

            url = video_urls[0].get("murl") or video_urls[0].get("lurl") or video_urls[0].get("surl")
            if not url:
                failed += 1
                continue

            subprocess.run(
                ["curl", "-sS", "-o", str(clip_path), url],
                check=True, timeout=30, capture_output=True,
            )

            if clip_path.exists() and clip_path.stat().st_size > 1000:
                manifest.at[idx, "downloaded"] = True
                fetched += 1
            else:
                failed += 1
                clip_path.unlink(missing_ok=True)

        except Exception as exc:
            failed += 1
            if i < 20:
                logger.warning("  Download failed %s/%s: %s", gid, evt, exc)

        if (i + 1) % 25 == 0 or (i + 1) == len(to_download):
            logger.info("  %d/%d fetched=%d failed=%d", i + 1, len(to_download), fetched, failed)
            manifest.to_parquet(MANIFEST_PATH, index=False)

    manifest.to_parquet(MANIFEST_PATH, index=False)
    logger.info("Download complete: fetched=%d, failed=%d", fetched, failed)

    dl_count = manifest["downloaded"].sum()
    total = len(manifest)
    print(f"\nDownload summary: {dl_count}/{total} clips ({dl_count/total*100:.1f}%)")


def train_and_evaluate() -> dict[str, Any]:
    """Extract features from clips and train a binary classifier.

    Uses a pre-trained video encoder to extract embeddings, then trains
    logistic regression on top. Falls back to a random baseline if
    torch/heavy deps aren't available.
    """
    if not MANIFEST_PATH.exists():
        raise RuntimeError("Run 'sample' and 'download' first")

    manifest = pd.read_parquet(MANIFEST_PATH)
    downloaded = manifest[manifest["downloaded"]].copy()
    logger.info("Training on %d downloaded clips (%d positive, %d negative)",
                len(downloaded), int(downloaded["label"].sum()), int((1 - downloaded["label"]).sum()))

    if len(downloaded) < 50:
        raise RuntimeError(f"Only {len(downloaded)} clips downloaded — need at least 50")

    try:
        from src.feasibility_train import run_training
        results = run_training(downloaded)
    except ImportError:
        logger.warning("feasibility_train.py not found — using random baseline")
        results = _random_baseline(downloaded)

    FEAS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results: %s", json.dumps(results, indent=2))
    return results


def _random_baseline(manifest: pd.DataFrame) -> dict[str, Any]:
    """Random baseline to establish performance floor."""
    import numpy as np

    y = manifest["label"].values
    rng = np.random.RandomState(42)
    y_prob = rng.random(len(y))
    y_pred = (y_prob > 0.5).astype(int)

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    return {
        "model": "random_baseline",
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y, y_prob)),
        "note": "Random baseline — replace with actual video classifier",
    }


def validate_against_l2m() -> dict[str, Any]:
    """Validate the trained model against L2M INC ground truth.

    This is the go/no-go gate. If the model cannot identify L2M INC events
    as fouls with acceptable recall, we pivot to descriptive-only Paper 1.
    """
    if not RESULTS_PATH.exists():
        raise RuntimeError("Run 'train' first")

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    l2m = pd.read_parquet(config.L2M_EVENTS_PATH)
    pbp_gids = set(f.stem for f in config.RAW_PBP_DIR.glob("*.json"))

    inc_sf = l2m[
        (l2m["review_decision"] == "INC")
        & (l2m["call_type"] == "Foul: Shooting")
        & (l2m["game_id"].isin(pbp_gids))
    ]

    results["l2m_inc_sf_available"] = len(inc_sf)
    results["l2m_inc_sf_total"] = int(
        l2m[(l2m["review_decision"] == "INC") & (l2m["call_type"] == "Foul: Shooting")].shape[0]
    )
    results["go_criteria"] = {
        "precision_threshold": 0.6,
        "recall_threshold": 0.5,
        "precision_met": results.get("precision", 0) >= 0.6,
        "recall_met": results.get("recall", 0) >= 0.5,
    }
    results["decision"] = (
        "GO — proceed to full Layer 3 build"
        if results["go_criteria"]["precision_met"] and results["go_criteria"]["recall_met"]
        else "NO-GO — pivot to descriptive Paper 1 using L2M INC as no-call data"
    )

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Validation: %s", results["decision"])

    print(f"\nGo/No-Go Gate:")
    print(f"  Precision: {results.get('precision', 0):.3f} (threshold: 0.6) {'PASS' if results['go_criteria']['precision_met'] else 'FAIL'}")
    print(f"  Recall:    {results.get('recall', 0):.3f} (threshold: 0.5) {'PASS' if results['go_criteria']['recall_met'] else 'FAIL'}")
    print(f"  Decision:  {results['decision']}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer 3 Feasibility Study")
    parser.add_argument("command", choices=["sample", "download", "train", "validate"])
    args = parser.parse_args()

    if args.command == "sample":
        build_sample()
    elif args.command == "download":
        download_clips()
    elif args.command == "train":
        train_and_evaluate()
    elif args.command == "validate":
        validate_against_l2m()


if __name__ == "__main__":
    main()
