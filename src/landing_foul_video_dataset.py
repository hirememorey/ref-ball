"""Download landing foul clips and extract frozen VideoMAE embeddings.

Phase 1: Download MP4 clips from NBA CDN into data/clips/landing_foul/.
Phase 2: Extract per-clip embeddings using a frozen VideoMAE backbone.

Usage:
    python src/landing_foul_video_dataset.py download
    python src/landing_foul_video_dataset.py download --limit 10
    python src/landing_foul_video_dataset.py extract
    python src/landing_foul_video_dataset.py extract --model MCG-NJU/videomae-base-finetuned-kinetics

Output:
    data/clips/landing_foul/{game_id}_{event_id}.mp4   (downloaded clips)
    data/processed/landing_foul_embeddings.npz          (embeddings + metadata)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

import config
from nba_client import NBAStatsClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CLIPS_DIR = config.DATA_DIR / "clips" / "landing_foul"
MANIFEST_PATH = config.PROCESSED_DIR / "landing_foul_manifest.json"
GROUND_TRUTH_PATH = config.DATA_DIR / "landing_foul_ground_truth.csv"
EMBEDDINGS_PATH = config.PROCESSED_DIR / "landing_foul_embeddings.npz"

DEFAULT_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"
NUM_FRAMES = 16

# NBA CDN returns this generic "video not available" MP4 when requests lack
# NBA.com Referer / stats headers (plain requests.get gets the placeholder).
PLACEHOLDER_MD5 = "2dd8e05a98fc6949fa7ec979b0905464"
PLACEHOLDER_SIZE = 31_580_089


def is_placeholder_bytes(data: bytes) -> bool:
    if len(data) == PLACEHOLDER_SIZE:
        return hashlib.md5(data).hexdigest() == PLACEHOLDER_MD5
    return False


def is_placeholder_file(path: Path, *, verify_md5: bool = False) -> bool:
    """Detect NBA CDN 'video not available' placeholder saved locally.

  Real PBP clips are typically 3–8 MB. The placeholder is always exactly
  PLACEHOLDER_SIZE bytes. Set verify_md5=True when writing downloads; for
  bulk scans (annotator startup) size-only is sufficient.
    """
    if not path.exists():
        return False
    if path.stat().st_size != PLACEHOLDER_SIZE:
        return False
    if not verify_md5:
        return True
    return hashlib.md5(path.read_bytes()).hexdigest() == PLACEHOLDER_MD5


def load_manifest() -> List[Dict[str, Any]]:
    """Load the landing foul manifest and return clip entries."""
    with open(MANIFEST_PATH) as f:
        data = json.load(f)
    return data["clips"]


def load_ground_truth_keys() -> set[tuple[str, int]]:
    """Return set of (game_id, event_id) pairs from ground truth, excluding UNCLEAR."""
    import pandas as pd

    df = pd.read_csv(GROUND_TRUTH_PATH)
    df = df[df["landing_foul"].isin(["YES", "NO"])]
    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    return set(zip(df["game_id"], df["event_id"].astype(int)))


def clip_path(game_id: str, event_id: int) -> Path:
    return CLIPS_DIR / f"{game_id}_{event_id}.mp4"


# ---------------------------------------------------------------------------
# Phase 1: Download clips
# ---------------------------------------------------------------------------


def download_clips(limit: int | None = None, resume: bool = True) -> None:
    """Download MP4 clips from NBA CDN for all ground-truth-labeled clips.

    Uses NBAStatsClient session headers — plain requests.get() receives a
    generic "video not available" placeholder (~31 MB) instead of the clip.
    With resume=True, existing placeholder files are re-downloaded automatically.
    """
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    clips = load_manifest()
    gt_keys = load_ground_truth_keys()

    # Filter manifest to only clips with ground truth labels (excluding UNCLEAR)
    labeled_clips = [
        c for c in clips
        if (str(c["game_id"]).zfill(10), int(c["event_id"])) in gt_keys
    ]
    logger.info(
        "Found %d manifest clips with ground truth labels (of %d total manifest, %d GT keys)",
        len(labeled_clips), len(clips), len(gt_keys),
    )

    if limit:
        labeled_clips = labeled_clips[:limit]

    client = NBAStatsClient()
    session = client.session

    downloaded, skipped, failed, refreshed = 0, 0, 0, 0
    for clip in tqdm(labeled_clips, desc="Downloading clips"):
        gid = str(clip["game_id"]).zfill(10)
        eid = int(clip["event_id"])
        out = clip_path(gid, eid)

        if resume and out.exists() and out.stat().st_size > 1000 and not is_placeholder_file(out, verify_md5=True):
            skipped += 1
            continue

        if out.exists() and is_placeholder_file(out, verify_md5=True):
            refreshed += 1

        url = clip.get("video_url_960") or clip.get("video_url_720")
        if not url:
            logger.warning("No video URL for %s_%s", gid, eid)
            failed += 1
            continue

        try:
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
            if is_placeholder_bytes(resp.content):
                logger.warning(
                    "CDN returned placeholder for %s_%s (check NBA session headers)",
                    gid, eid,
                )
                failed += 1
                continue
            out.write_bytes(resp.content)
            downloaded += 1
        except Exception as e:
            logger.warning("Failed to download %s_%s: %s", gid, eid, e)
            failed += 1

    logger.info(
        "Download complete: %d downloaded, %d skipped (valid existing), "
        "%d refreshed (was placeholder), %d failed",
        downloaded, skipped, refreshed, failed,
    )


# ---------------------------------------------------------------------------
# Phase 2: Extract VideoMAE embeddings
# ---------------------------------------------------------------------------


def sample_frames_from_video(video_path: Path, num_frames: int = NUM_FRAMES) -> np.ndarray:
    """Decode video and uniformly sample num_frames frames.

    Returns: ndarray of shape (num_frames, H, W, 3) in uint8 RGB.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {video_path}")

    # Uniformly sample frame indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # Fallback: duplicate last good frame
            if frames:
                frames.append(frames[-1].copy())
            else:
                raise ValueError(f"Cannot read frame {idx} from {video_path}")
            continue
        # BGR -> RGB
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return np.stack(frames)


def extract_embeddings(model_name: str = DEFAULT_MODEL) -> None:
    """Extract frozen VideoMAE embeddings for all downloaded clips with labels."""
    import torch
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    logger.info("Loading VideoMAE model: %s", model_name)
    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    model = VideoMAEForVideoClassification.from_pretrained(model_name)
    model.eval()

    # We extract from the last hidden state, not the classification head
    # The classification head is Kinetics-specific; we want general features
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    logger.info("Using device: %s", device)

    import pandas as pd

    gt = pd.read_csv(GROUND_TRUTH_PATH)
    gt = gt[gt["landing_foul"].isin(["YES", "NO"])].copy()
    gt["game_id"] = gt["game_id"].astype(str).str.zfill(10)
    gt["event_id"] = gt["event_id"].astype(int)

    embeddings = []
    game_ids = []
    event_ids = []
    labels = []
    skipped = []

    for _, row in tqdm(gt.iterrows(), total=len(gt), desc="Extracting embeddings"):
        gid = row["game_id"]
        eid = int(row["event_id"])
        vpath = clip_path(gid, eid)

        if not vpath.exists():
            skipped.append(f"{gid}_{eid}")
            continue

        try:
            frames = sample_frames_from_video(vpath, NUM_FRAMES)
        except Exception as e:
            logger.warning("Frame extraction failed for %s_%s: %s", gid, eid, e)
            skipped.append(f"{gid}_{eid}")
            continue

        # Processor expects list of frames as list of numpy arrays
        inputs = processor(
            list(frames),
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Use CLS token from last hidden state as the embedding
            # hidden_states[-1] shape: (1, num_patches+1, hidden_dim)
            cls_embedding = outputs.hidden_states[-1][:, 0, :]  # (1, 768)
            embeddings.append(cls_embedding.cpu().numpy().squeeze())

        game_ids.append(gid)
        event_ids.append(eid)
        labels.append(1 if row["landing_foul"] == "YES" else 0)

    if skipped:
        logger.warning("Skipped %d clips (not downloaded or unreadable): %s", len(skipped), skipped[:5])

    embeddings_array = np.stack(embeddings)
    logger.info(
        "Extracted %d embeddings, shape %s (skipped %d)",
        len(embeddings), embeddings_array.shape, len(skipped),
    )

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        EMBEDDINGS_PATH,
        embeddings=embeddings_array,
        game_ids=np.array(game_ids),
        event_ids=np.array(event_ids),
        labels=np.array(labels),
        model_name=model_name,
    )
    logger.info("Saved embeddings to %s", EMBEDDINGS_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download clips and extract VideoMAE embeddings")
    sub = parser.add_subparsers(dest="command")

    dl = sub.add_parser("download", help="Download MP4 clips from NBA CDN")
    dl.add_argument("--limit", type=int, default=None, help="Max clips to download")
    dl.add_argument("--no-resume", action="store_true", help="Re-download existing clips")

    ext = sub.add_parser("extract", help="Extract frozen VideoMAE embeddings")
    ext.add_argument("--model", default=DEFAULT_MODEL, help=f"HuggingFace model ID (default: {DEFAULT_MODEL})")

    args = parser.parse_args()

    if args.command == "download":
        download_clips(limit=args.limit, resume=not args.no_resume)
    elif args.command == "extract":
        extract_embeddings(model_name=args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
