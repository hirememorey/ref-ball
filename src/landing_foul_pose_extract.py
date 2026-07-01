"""Step 10e (Phase 1): Multi-person pose extraction for landing foul clips.

Extracts per-frame COCO-17 keypoints for every detected person in the contact
window of each labeled clip, using a YOLOv8-Pose top-down model with the built-in
BoT-SORT tracker for cross-frame identity. Output schema matches the plan in
documents/development/POSE-ESTIMATION-PLAN.md (Phase 1) — all 17 named joints
are stored (a superset of the lower-body subset the plan lists) so Phase 2
feature engineering can pick any combination.

Reuses existing pipeline assets:
  - data/clips/landing_foul/{game_id}_{event_id}.mp4   (284 clips)
  - data/processed/landing_foul_clip_anchors.json       (foul_frac ± half_width)
  - data/landing_foul_ground_truth.csv                  (YES/NO clip list)

Subcommands:
  extract     Run pose extraction for all (or a subset of) clips → landing_foul_poses.json
  validate    Phase 0: keypoint-quality report on a small clip set (go/no-go)
  visualize   Overlay skeletons + track ids on one clip → output/pose_viz/<clip>.mp4

Usage:
  PYTHONPATH=. python src/landing_foul_pose_extract.py extract
  PYTHONPATH=. python src/landing_foul_pose_extract.py extract --clip 0021900028_532
  PYTHONPATH=. python src/landing_foul_pose_extract.py validate --limit 10
  PYTHONPATH=. python src/landing_foul_pose_extract.py visualize --clip 0021900028_532

Output:
  data/processed/landing_foul_poses.json          (raw keypoints, gitignored)
  data/processed/landing_foul_pose_validation.json (Phase 0 quality report)
  output/pose_viz/{clip}.mp4                       (visualized clips)
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
from landing_foul_video_dataset import CLIPS_DIR, GROUND_TRUTH_PATH, clip_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ANCHORS_PATH = config.PROCESSED_DIR / "landing_foul_clip_anchors.json"
POSES_PATH = config.PROCESSED_DIR / "landing_foul_poses.json"
VALIDATION_PATH = config.PROCESSED_DIR / "landing_foul_pose_validation.json"
VIZ_DIR = config.OUTPUT_DIR / "pose_viz"

DEFAULT_POSE_MODEL = "yolov8s-pose.pt"
DEFAULT_HALF_WIDTH = 0.15
DEFAULT_MAX_FRAMES = 60  # cap frames per clip (uniform subsample within window)
KP_CONF_THRESHOLD = 0.3  # plan's Phase 0 success bar for lower-body joints

# COCO-17 keypoint names in the order YOLOv8-Pose emits them.
COCO_KP_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
LOWER_BODY_JOINTS = {"left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"}
ANKLE_JOINTS = {"left_ankle", "right_ankle"}


# ---------------------------------------------------------------------------
# Shared loaders
# ---------------------------------------------------------------------------


def load_anchors() -> dict[str, dict[str, float]]:
    if not ANCHORS_PATH.exists():
        return {}
    with open(ANCHORS_PATH) as f:
        return json.load(f)


def load_labeled_clips() -> list[tuple[str, int]]:
    """Return sorted (game_id, event_id) pairs with YES/NO labels (UNCLEAR excluded)."""
    import pandas as pd

    df = pd.read_csv(GROUND_TRUTH_PATH)
    df = df[df["landing_foul"].isin(["YES", "NO"])].copy()
    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    df["event_id"] = df["event_id"].astype(int)
    df = df.drop_duplicates(["game_id", "event_id"]).sort_values(["game_id", "event_id"])
    return list(zip(df["game_id"], df["event_id"]))


def clip_key(game_id: str, event_id: int) -> str:
    return f"{game_id}_{event_id}"


def resolve_window(
    key: str,
    anchors: dict[str, dict[str, float]],
    half_width_override: float | None = None,
) -> tuple[float, float]:
    """Per-clip [start_frac, end_frac] from the anchor sidecar. Defaults to full clip."""
    anchor = anchors.get(key)
    if anchor and "foul_frac" in anchor:
        frac = float(anchor["foul_frac"])
        hw = half_width_override if half_width_override is not None else float(anchor.get("half_width", DEFAULT_HALF_WIDTH))
        lo = max(0.0, frac - hw)
        hi = min(1.0, frac + hw)
        if hi - lo >= 0.05:
            return lo, hi
    return 0.0, 1.0


# ---------------------------------------------------------------------------
# Frame decoding within a window (single sequential pass; per-frame seeks are
# catastrophically slow on these MP4s — see landing_foul_video_finetune.py).
# ---------------------------------------------------------------------------


def decode_window_frames(
    video_path: Path,
    start_frac: float,
    end_frac: float,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> tuple[list[np.ndarray], float, int, list[int]]:
    """Decode frames in [start_frac, end_frac], uniform-subsampling to <= max_frames.

    Returns (frames_bgr, fps, total_frames, frame_indices). Frames are BGR uint8
    (native cv2 format — ultralytics and cv2.VideoWriter both expect BGR).
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    start_frame = max(0, int(round(start_frac * total)))
    end_frame = min(total - 1, int(round(end_frac * total)))
    if end_frame <= start_frame:
        end_frame = min(total - 1, start_frame + 1)

    span = end_frame - start_frame + 1
    n = min(max_frames, span)
    indices = [start_frame + int(round(i * (span - 1) / max(1, n - 1))) for i in range(n)] if n > 1 else [start_frame]
    # De-duplicate while preserving order.
    seen: set[int] = set()
    indices = [i for i in indices if not (i in seen or seen.add(i))]
    target = set(indices)

    frames: list[np.ndarray] = []
    collected: list[int] = []
    frame_idx = 0
    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in target:
            frames.append(frame)
            collected.append(frame_idx)
        frame_idx += 1
    cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from window [{start_frac},{end_frac}] of {video_path}")
    return frames, float(fps), total, collected


# ---------------------------------------------------------------------------
# Pose model
# ---------------------------------------------------------------------------


def load_pose_model(model_name: str, device: str):
    from ultralytics import YOLO

    model = YOLO(model_name)
    if device != "auto":
        model.to(device)
    return model


def pick_device(pref: str) -> str:
    if pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def run_pose_on_frames(model, frames: list[np.ndarray], device: str):
    """Run tracked pose inference over a clip's frames, one frame at a time.

    Frame-by-frame (rather than passing the whole list as a batch) keeps BoT-SORT
    state via persist=True while avoiding batched-NMS timeouts — with ~10 persons
    per broadcast frame, batched NMS regularly exceeds ultralytics' 5s limit and
    silently drops detections for whole frames.
    """
    results = []
    for fr in frames:
        r = model.track(
            source=fr,
            persist=True,
            tracker="botsort.yaml",
            verbose=False,
            device=device if device != "cpu" else "cpu",
        )
        results.append(r[0] if isinstance(r, list) else r)
    return results


# ---------------------------------------------------------------------------
# Per-frame detection → serializable records
# ---------------------------------------------------------------------------


def result_to_detections(result) -> list[dict[str, Any]]:
    """Convert one ultralytics Results object to a list of per-person dicts."""
    dets: list[dict[str, Any]] = []
    if result.boxes is None or result.keypoints is None:
        return dets

    kpts_xy = result.keypoints.xy  # (N, 17, 2)
    kpts_conf = result.keypoints.conf  # (N, 17)
    boxes_xyxy = result.boxes.xyxy  # (N, 4)
    boxes_conf = result.boxes.conf  # (N,)
    ids = result.boxes.id  # (N,) or None

    n = int(boxes_xyxy.shape[0]) if boxes_xyxy is not None else 0
    for i in range(n):
        x1, y1, x2, y2 = (float(v) for v in boxes_xyxy[i].tolist())
        tid = int(ids[i].item()) if ids is not None else None
        kp: dict[str, list[float]] = {}
        for j, name in enumerate(COCO_KP_NAMES):
            xy = kpts_xy[i, j].tolist()
            c = float(kpts_conf[i, j].item()) if kpts_conf is not None else 0.0
            kp[name] = [float(xy[0]), float(xy[1]), c]
        dets.append({
            "track_id": tid,
            "bbox": [x1, y1, x2, y2],
            "conf": float(boxes_conf[i].item()),
            "keypoints": kp,
        })
    return dets


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------


def extract_clip(
    model,
    video_path: Path,
    key: str,
    anchors: dict[str, dict[str, float]],
    half_width: float | None,
    max_frames: int,
    device: str,
) -> dict[str, Any]:
    start_frac, end_frac = resolve_window(key, anchors, half_width)
    frames, fps, total, indices = decode_window_frames(video_path, start_frac, end_frac, max_frames)
    results = run_pose_on_frames(model, frames, device)

    # Group detections by track id across frames. Untracked detections (id None)
    # are kept under a per-frame sentinel so no data is silently dropped.
    persons: dict[str, dict[str, Any]] = {}
    for frame_pos, (result, fidx) in enumerate(zip(results, indices)):
        for det in result_to_detections(result):
            tid = det["track_id"]
            pid = str(tid) if tid is not None else f"untracked_{fidx}_{det['bbox'][0]:.0f}"
            person = persons.setdefault(pid, {"track_id": tid, "frames": {}})
            person["frames"][str(fidx)] = {
                "bbox": det["bbox"],
                "conf": det["conf"],
                "keypoints": det["keypoints"],
            }

    persons_list = [
        {"track_id": p["track_id"], "frames": p["frames"]} for p in persons.values()
    ]
    return {
        "fps": fps,
        "total_frames": total,
        "anchor_frac": float(anchors.get(key, {}).get("foul_frac", 0.5)),
        "half_width": float(
            half_width if half_width is not None else anchors.get(key, {}).get("half_width", DEFAULT_HALF_WIDTH)
        ),
        "window": [start_frac, end_frac],
        "frame_indices": indices,
        "n_frames_processed": len(indices),
        "n_persons": len(persons_list),
        "persons": persons_list,
    }


def extract_all(
    model_name: str,
    half_width: float | None,
    max_frames: int,
    device: str,
    clip_filter: str | None,
    limit: int | None,
    overwrite: bool,
) -> None:
    anchors = load_anchors()
    clips = load_labeled_clips()
    if clip_filter:
        clips = [(gid, eid) for gid, eid in clips if clip_key(gid, eid) == clip_filter]
    if limit:
        clips = clips[:limit]
    logger.info("Extracting poses for %d clips (model=%s, device=%s)", len(clips), model_name, device)

    device_resolved = pick_device(device)
    model = load_pose_model(model_name, device_resolved)

    existing: dict[str, Any] = {}
    if not overwrite and POSES_PATH.exists():
        with open(POSES_PATH) as f:
            existing = json.load(f)
        logger.info("Resuming: %d clips already in %s", len(existing), POSES_PATH.name)

    out = dict(existing)
    done = 0
    for gid, eid in clips:
        key = clip_key(gid, eid)
        vpath = clip_path(gid, eid)
        if not vpath.exists():
            logger.warning("Missing clip %s — skipping", key)
            continue
        if key in out and not overwrite:
            done += 1
            continue
        try:
            out[key] = extract_clip(model, vpath, key, anchors, half_width, max_frames, device_resolved)
            done += 1
            logger.info(
                "%s: %d frames, %d persons (window [%.3f, %.3f])",
                key, out[key]["n_frames_processed"], out[key]["n_persons"], *out[key]["window"],
            )
        except Exception as e:
            logger.warning("Failed on %s: %s", key, e)

    POSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(POSES_PATH, "w") as f:
        json.dump(out, f)
    logger.info("Wrote %d clip poses to %s", len(out), POSES_PATH)


# ---------------------------------------------------------------------------
# Validate (Phase 0)
# ---------------------------------------------------------------------------


def _kp(detection: dict[str, Any], name: str) -> tuple[float, float, float] | None:
    kp = detection["keypoints"].get(name)
    if kp is None:
        return None
    return float(kp[0]), float(kp[1]), float(kp[2])


def _person_frame(person: dict[str, Any], fidx: int) -> dict[str, Any] | None:
    return person["frames"].get(str(fidx))


def _frame_persons(clip_poses: dict[str, Any], fidx: int) -> list[dict[str, Any]]:
    out = []
    for p in clip_poses["persons"]:
        fr = _person_frame(p, fidx)
        if fr is not None:
            out.append(fr)
    return out


def validate_clip_quality(clip_poses: dict[str, Any]) -> dict[str, Any]:
    """Compute Phase 0 keypoint-quality stats for one clip.

    Role assignment is Phase 2, so we report across all detected persons:
      - mean confidence per lower-body joint
      - fraction of frames with >=2 persons detected (need shooter + defender)
      - fraction of frames where some person has both ankles > threshold
      - fraction of frames where >=2 persons each have both ankles > threshold
    """
    indices = clip_poses["frame_indices"]
    n_frames = len(indices)
    if n_frames == 0:
        return {"error": "no frames processed"}

    conf_sums = {j: [] for j in LOWER_BODY_JOINTS | ANKLE_JOINTS}
    n_persons_per_frame = []
    frames_someone_both_ankles = 0
    frames_two_both_ankles = 0

    for fidx in indices:
        persons = _frame_persons(clip_poses, fidx)
        n_persons_per_frame.append(len(persons))
        both_ankles_count = 0
        for det in persons:
            la = _kp(det, "left_ankle")
            ra = _kp(det, "right_ankle")
            la_ok = la is not None and la[2] > KP_CONF_THRESHOLD
            ra_ok = ra is not None and ra[2] > KP_CONF_THRESHOLD
            if la_ok and ra_ok:
                both_ankles_count += 1
            for j in conf_sums:
                k = _kp(det, j)
                if k is not None and k[2] > 0:
                    conf_sums[j].append(k[2])
        if both_ankles_count >= 1:
            frames_someone_both_ankles += 1
        if both_ankles_count >= 2:
            frames_two_both_ankles += 1

    mean_conf = {j: (float(np.mean(v)) if v else 0.0) for j, v in conf_sums.items()}
    mean_n_persons = float(np.mean(n_persons_per_frame)) if n_persons_per_frame else 0.0

    return {
        "n_frames": n_frames,
        "n_persons_total": clip_poses["n_persons"],
        "mean_persons_per_frame": mean_n_persons,
        "frames_ge2_persons_frac": float(np.mean([n >= 2 for n in n_persons_per_frame])),
        "frames_someone_both_ankles_frac": frames_someone_both_ankles / n_frames,
        "frames_two_both_ankles_frac": frames_two_both_ankles / n_frames,
        "mean_joint_conf": mean_conf,
        "ankle_mean_conf": float(np.mean(conf_sums["left_ankle"] + conf_sums["right_ankle"])) if (conf_sums["left_ankle"] or conf_sums["right_ankle"]) else 0.0,
    }


def validate(
    model_name: str,
    half_width: float | None,
    max_frames: int,
    device: str,
    limit: int | None,
    clip_filter: str | None = None,
) -> None:
    anchors = load_anchors()
    clips = load_labeled_clips()
    if clip_filter:
        clips = [(gid, eid) for gid, eid in clips if clip_key(gid, eid) == clip_filter]
    if limit:
        clips = clips[:limit]
    logger.info("Phase 0 validation on %d clips (model=%s)", len(clips), model_name)
    device_resolved = pick_device(device)
    model = load_pose_model(model_name, device_resolved)

    per_clip = {}
    for gid, eid in clips:
        key = clip_key(gid, eid)
        vpath = clip_path(gid, eid)
        if not vpath.exists():
            logger.warning("Missing clip %s — skipping", key)
            continue
        try:
            clip_poses = extract_clip(model, vpath, key, anchors, half_width, max_frames, device_resolved)
            stats = validate_clip_quality(clip_poses)
            per_clip[key] = stats
            logger.info(
                "%s: persons/frame=%.1f, two-both-ankles frac=%.2f, ankle conf=%.2f",
                key, stats["mean_persons_per_frame"], stats["frames_two_both_ankles_frac"], stats["ankle_mean_conf"],
            )
        except Exception as e:
            logger.warning("Failed on %s: %s", key, e)
            per_clip[key] = {"error": str(e)}

    # Aggregate
    valid = [v for v in per_clip.values() if "error" not in v]
    n = len(valid)
    if n == 0:
        logger.error("No clips validated successfully")
        report = {"n_clips": 0, "per_clip": per_clip}
    else:
        report = {
            "model": model_name,
            "half_width": half_width if half_width is not None else DEFAULT_HALF_WIDTH,
            "max_frames": max_frames,
            "kp_conf_threshold": KP_CONF_THRESHOLD,
            "n_clips": n,
            "mean_persons_per_frame": float(np.mean([v["mean_persons_per_frame"] for v in valid])),
            "mean_frames_ge2_persons_frac": float(np.mean([v["frames_ge2_persons_frac"] for v in valid])),
            "mean_frames_someone_both_ankles_frac": float(np.mean([v["frames_someone_both_ankles_frac"] for v in valid])),
            "mean_frames_two_both_ankles_frac": float(np.mean([v["frames_two_both_ankles_frac"] for v in valid])),
            "mean_ankle_conf": float(np.mean([v["ankle_mean_conf"] for v in valid])),
            # Plan's Phase 0 success bars (applied across all persons, pre-role-assignment):
            #   shooter ankles tracked in >=80% of descent frames  (proxy: >=1 person both ankles)
            #   defender feet tracked in >=60% of contact frames   (proxy: >=2 persons both ankles)
            "meets_someone_both_ankles_bar": float(np.mean([v["frames_someone_both_ankles_frac"] for v in valid])) >= 0.80,
            "meets_two_both_ankles_bar": float(np.mean([v["frames_two_both_ankles_frac"] for v in valid])) >= 0.60,
            "per_clip": per_clip,
        }

    VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VALIDATION_PATH, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 60)
    logger.info("Phase 0 validation report (%d clips)", n)
    if n:
        logger.info("  mean persons/frame:        %.2f", report["mean_persons_per_frame"])
        logger.info("  >=2 persons/frame frac:    %.2f", report["mean_frames_ge2_persons_frac"])
        logger.info("  someone-both-ankles frac:  %.2f  (bar >=0.80: %s)", report["mean_frames_someone_both_ankles_frac"], report["meets_someone_both_ankles_bar"])
        logger.info("  two-both-ankles frac:      %.2f  (bar >=0.60: %s)", report["mean_frames_two_both_ankles_frac"], report["meets_two_both_ankles_bar"])
        logger.info("  mean ankle confidence:     %.2f", report["mean_ankle_conf"])
    logger.info("Wrote report to %s", VALIDATION_PATH)


# ---------------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------------


def visualize_clip(model_name: str, key: str, half_width: float | None, max_frames: int, device: str) -> Path:
    import cv2

    anchors = load_anchors()
    try:
        gid, eid = key.rsplit("_", 1)
        eid = int(eid)
    except ValueError:
        raise SystemExit(f"Bad clip key '{key}', expected {{game_id}}_{{event_id}}")
    vpath = clip_path(gid, eid)
    if not vpath.exists():
        raise SystemExit(f"Missing clip {vpath}")

    device_resolved = pick_device(device)
    model = load_pose_model(model_name, device_resolved)
    clip_poses = extract_clip(model, vpath, key, anchors, half_width, max_frames, device_resolved)

    frames, fps, total, indices = decode_window_frames(vpath, *clip_poses["window"], max_frames)
    results = run_pose_on_frames(model, frames, device_resolved)

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIZ_DIR / f"{key}.mp4"
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for i, (result, fidx) in enumerate(zip(results, indices)):
        annotated = result.plot()  # BGR ndarray with boxes + skeleton + ids
        # Annotate frame index + n persons for inspection.
        cv2.putText(annotated, f"frame {fidx}  persons={len(result.boxes or [])}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        writer.write(annotated)
    writer.release()
    logger.info("Wrote visualization to %s (%d frames)", out_path, len(frames))
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 10e Phase 1: pose extraction (YOLOv8-Pose)")
    parser.add_argument("subcommand", nargs="?", default="extract", choices=["extract", "validate", "visualize"])
    parser.add_argument("--clip", default=None, help="Single clip key (game_id_event_id)")
    parser.add_argument("--limit", type=int, default=None, help="Max clips to process")
    parser.add_argument("--model", default=DEFAULT_POSE_MODEL, help=f"YOLOv8-Pose weights (default: {DEFAULT_POSE_MODEL})")
    parser.add_argument("--anchor-half-width", type=float, default=None, help="Override anchor half_width (default: per-clip stored value)")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help=f"Max frames per clip (default: {DEFAULT_MAX_FRAMES})")
    parser.add_argument("--device", default="auto", help="cuda | mps | cpu | auto (default: auto)")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract clips already in poses.json")
    args = parser.parse_args()

    if args.subcommand == "extract":
        extract_all(
            model_name=args.model,
            half_width=args.anchor_half_width,
            max_frames=args.max_frames,
            device=args.device,
            clip_filter=args.clip,
            limit=args.limit,
            overwrite=args.overwrite,
        )
    elif args.subcommand == "validate":
        validate(
            args.model, args.anchor_half_width, args.max_frames, args.device,
            limit=args.limit, clip_filter=args.clip,
        )
    elif args.subcommand == "visualize":
        if not args.clip:
            raise SystemExit("visualize requires --clip")
        visualize_clip(args.model, args.clip, args.anchor_half_width, args.max_frames, args.device)


if __name__ == "__main__":
    main()
