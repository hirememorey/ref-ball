"""End-to-end fine-tuning of VideoMAE for landing foul classification.

The frozen-feature baseline (landing_foul_video_train.py) produced zero
discriminative signal: Kinetics-pretrained features cannot distinguish a
landing foul from a standard contest at the "playing basketball" level of
abstraction. This script trains the backbone itself to learn that distinction.

Approach:
  - Replace the 400-class Kinetics head with a 2-class linear layer (NO=0, YES=1).
  - Two-phase training:
      Phase 1 ("head"):  freeze the backbone, train only the classifier (+ fc_norm).
      Phase 2 ("finetune"): unfreeze the top-N transformer layers and train them
                            at a lower learning rate than the head.
  - Heavy regularization (n=227 train): dropout, weight decay, early stopping on
    val YES precision, random temporal jitter + color jitter. No horizontal flip
    (broadcast camera orientation is consistent and load-bearing).
  - Checkpoint the best model by val precision on YES — the binding gate (>= 85%).

Temporal windowing:
  Clips are ~8-12s; the contact window is ~400ms. Sampling 16 frames across the
  whole clip (the frozen approach) dilutes the signal. The window is resolved
  per-clip with a fallback chain:
    1. Optional sidecar `landing_foul_clip_anchors.json` mapping
       "{game_id}_{event_id}" -> {"foul_frac": float, "half_width": float}.
       (Future hook for whistle / arm-up detection — not built here.)
    2. Global --temporal-window (default 0.0,1.0 = full clip).
  Jitter is applied around the resolved window.

Usage:
    python src/landing_foul_video_finetune.py                         # two-phase
    python src/landing_foul_video_finetune.py --phase head --head-epochs 5
    python src/landing_foul_video_finetune.py --evaluate-only \
        --checkpoint data/processed/landing_foul_video_best.pt

Outputs:
    data/processed/landing_foul_video_best.pt        (gitignored checkpoint)
    data/processed/landing_foul_video_metrics.json   (config + history + eval)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

import config
from landing_foul_video_dataset import (
    CLIPS_DIR,
    DEFAULT_MODEL,
    GROUND_TRUTH_PATH,
    clip_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SPLIT_PATH = config.PROCESSED_DIR / "landing_foul_split.json"
ANCHORS_PATH = config.PROCESSED_DIR / "landing_foul_clip_anchors.json"
BEST_CKPT_PATH = config.PROCESSED_DIR / "landing_foul_video_best.pt"
METRICS_PATH = config.PROCESSED_DIR / "landing_foul_video_metrics.json"
DEFAULT_CACHE_PATH = config.PROCESSED_DIR / "landing_foul_frames.npz"

# Quality gate
PRECISION_GATE = 0.85
RECALL_GATE = 0.70


# ---------------------------------------------------------------------------
# Config dataclass (also serialized into metrics JSON)
# ---------------------------------------------------------------------------


def parse_window(s: str) -> tuple[float, float]:
    a, b = s.split(",")
    lo, hi = float(a), float(b)
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError(f"temporal window must satisfy 0 <= lo < hi <= 1, got {s}")
    return lo, hi


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def pick_device(pref: str) -> str:
    import torch

    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Labels + split
# ---------------------------------------------------------------------------


def load_labeled_keys() -> dict[tuple[str, int], int]:
    """Return {(game_id, event_id): label} for YES/NO rows only."""
    import pandas as pd

    df = pd.read_csv(GROUND_TRUTH_PATH)
    df = df[df["landing_foul"].isin(["YES", "NO"])].copy()
    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    df["event_id"] = df["event_id"].astype(int)
    return {
        (r["game_id"], int(r["event_id"])): (1 if r["landing_foul"] == "YES" else 0)
        for _, r in df.iterrows()
    }


def load_split_keys() -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    with open(SPLIT_PATH) as f:
        split = json.load(f)

    def to_keys(items):
        return [(str(x["game_id"]).zfill(10), int(x["event_id"])) for x in items]

    return to_keys(split["train"]["keys"]), to_keys(split["val"]["keys"])


def load_anchors() -> dict[str, dict[str, float]]:
    if not ANCHORS_PATH.exists():
        return {}
    with open(ANCHORS_PATH) as f:
        return json.load(f)


def resolve_window(
    key: tuple[str, int],
    anchors: dict[str, dict[str, float]],
    global_window: tuple[float, float],
) -> tuple[float, float]:
    """Per-clip window via anchor sidecar, else global window."""
    anchor = anchors.get(f"{key[0]}_{key[1]}")
    if anchor and "foul_frac" in anchor:
        frac = float(anchor["foul_frac"])
        hw = float(anchor.get("half_width", 0.15))
        lo = max(0.0, frac - hw)
        hi = min(1.0, frac + hw)
        if hi - lo >= 0.05:
            return lo, hi
    return global_window


# ---------------------------------------------------------------------------
# Frame sampling (generalizes landing_foul_video_dataset.sample_frames_from_video)
# ---------------------------------------------------------------------------


def sample_frames_windowed(
    video_path: Path,
    num_frames: int,
    start_frac: float,
    end_frac: float,
    jitter_extra: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Decode video and sample `num_frames` frames uniformly within [start_frac, end_frac].

    Uses a single sequential decode pass and collects only the target frame indices.
    Per-frame cv2 seeks are catastrophically slow on these MP4s (FFmpeg re-decodes from
    the nearest keyframe on every CAP_PROP_POS_FRAMES set), so we avoid them entirely.

    If jitter_extra > 0, oversample (num_frames + jitter_extra) frames from the window
    and return a random contiguous block of `num_frames` — a temporal shift augmentation
    that preserves the 16-frame model constraint.

    Returns: ndarray of shape (num_frames, H, W, 3) in uint8 RGB.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    lo = int(total * start_frac)
    hi = int(total * end_frac)
    oversample = num_frames + max(0, jitter_extra)
    hi = max(hi, lo + oversample + 1)
    hi = min(hi, total - 1)
    lo = max(0, min(lo, hi - oversample))

    # Sorted target indices within the window (uniform spacing).
    targets = np.linspace(lo, hi - 1, oversample, dtype=int)
    target_set = set(int(t) for t in targets)

    collected: dict[int, np.ndarray] = {}
    frame_idx = 0
    try:
        while len(collected) < len(target_set) and frame_idx <= hi:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in target_set:
                collected[frame_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_idx += 1
    finally:
        cap.release()

    # Fill any missing targets (decode ended early) by duplicating the nearest collected frame.
    if not collected:
        raise ValueError(f"No frames decoded from {video_path}")
    fallback = next(iter(collected.values()))
    frames = [collected.get(int(t), fallback) for t in targets]
    arr = np.stack(frames)

    if jitter_extra > 0 and oversample > num_frames:
        rng = rng or np.random.default_rng()
        max_start = oversample - num_frames
        start = int(rng.integers(0, max_start + 1))
        arr = arr[start : start + num_frames]
    return arr


# ---------------------------------------------------------------------------
# Frame cache: decode each clip once, reuse across epochs.
#
# Decoding 1080p MP4s is the dominant cost (~7s/clip); re-decoding every epoch
# makes a 20-epoch run take ~10 hours. Caching ~32 frames/clip at 256x256 uint8
# (~1.8 GB) drops epoch time to ~1-2 min. Cache is built with the same window
# resolution logic as live decoding, so results are identical modulo resize.
# ---------------------------------------------------------------------------


def _resize_frames(frames: np.ndarray, size: int) -> np.ndarray:
    import cv2

    out = np.empty((frames.shape[0], size, size, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        out[i] = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
    return out


def build_frame_cache(
    keys: list[tuple[str, int]],
    labels: dict[tuple[str, int], int],
    anchors: dict[str, dict[str, float]],
    global_window: tuple[float, float],
    cache_frames: int,
    cache_size: int,
    out_path: Path,
) -> None:
    """Decode every clip once and store `cache_frames` resized frames per clip."""
    import cv2  # noqa: F401  (sample_frames_windowed imports cv2 lazily)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_arrays, key_tags, label_list = [], [], []
    failed = []
    for i, key in enumerate(keys):
        gid, eid = key
        vpath = clip_path(gid, eid)
        if not vpath.exists():
            failed.append(f"{gid}_{eid} (missing)")
            continue
        try:
            win = resolve_window(key, anchors, global_window)
            frames = sample_frames_windowed(
                vpath, num_frames=cache_frames, start_frac=win[0], end_frac=win[1],
                jitter_extra=0,
            )
            frames = _resize_frames(frames, cache_size)
        except Exception as e:
            failed.append(f"{gid}_{eid} ({e})")
            continue
        frame_arrays.append(frames)
        key_tags.append(f"{gid}_{eid}")
        label_list.append(labels[key])
        if (i + 1) % 25 == 0:
            logger.info("  cache: %d/%d clips decoded", i + 1, len(keys))

    if failed:
        logger.warning("Cache build: %d clips failed: %s", len(failed), failed[:5])
    if not frame_arrays:
        raise SystemExit("No clips decoded for cache.")

    stacked = np.stack(frame_arrays)  # (N, cache_frames, size, size, 3) uint8
    np.savez(
        out_path,
        frames=stacked,
        keys=np.array(key_tags, dtype=object),
        labels=np.array(label_list, dtype=np.int64),
        cache_frames=cache_frames,
        cache_size=cache_size,
    )
    logger.info("Saved frame cache: %s | %d clips x %d frames x %dx%d (~%.1f GB)",
                out_path, len(frame_arrays), cache_frames, cache_size, cache_size,
                stacked.nbytes / 1e9)


def load_frame_cache(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    frames = data["frames"]  # (N, T, H, W, 3)
    keys = [str(k) for k in data["keys"]]
    return {keys[i]: frames[i] for i in range(len(keys))}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class LandingFoulDataset:
    """PyTorch Dataset over on-disk clips.

    Augmentation (train only): random temporal jitter + color jitter.
    No horizontal flip (broadcast camera orientation is load-bearing).
    """

    def __init__(
        self,
        keys: list[tuple[str, int]],
        labels: dict[tuple[str, int], int],
        anchors: dict[str, dict[str, float]],
        global_window: tuple[float, float],
        augment: bool,
        num_frames: int = 16,
        jitter_extra: int = 6,
        model_name: str = DEFAULT_MODEL,
        seed: int = 42,
        frame_cache: dict[str, np.ndarray] | None = None,
        cache_frames: int = 32,
    ) -> None:
        self.keys = keys
        self.labels = labels
        self.anchors = anchors
        self.global_window = global_window
        self.augment = augment
        self.num_frames = num_frames
        self.jitter_extra = jitter_extra if augment else 0
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.frame_cache = frame_cache
        self.cache_frames = cache_frames

        # Lazily build the processor (HuggingFace).
        from transformers import VideoMAEImageProcessor

        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)

        if augment:
            from torchvision.transforms import ColorJitter

            self.color_jitter = ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0
            )
        else:
            self.color_jitter = None

    def __len__(self) -> int:
        return len(self.keys)

    def _read_frames(self, key: tuple[str, int]) -> np.ndarray:
        # Cached path: frames already decoded + resized. Apply temporal subsample/jitter.
        if self.frame_cache is not None:
            cached = self.frame_cache.get(f"{key[0]}_{key[1]}")
            if cached is not None:
                T = cached.shape[0]
                if self.augment and T > self.num_frames:
                    max_start = T - self.num_frames
                    start = int(self._rng.integers(0, max_start + 1))
                    return cached[start : start + self.num_frames]
                idx = np.linspace(0, T - 1, self.num_frames, dtype=int)
                return cached[idx]
        # Live decode path.
        gid, eid = key
        vpath = clip_path(gid, eid)
        if not vpath.exists():
            raise FileNotFoundError(f"Clip missing: {vpath}")
        win = resolve_window(key, self.anchors, self.global_window)
        return sample_frames_windowed(
            vpath,
            num_frames=self.num_frames,
            start_frac=win[0],
            end_frac=win[1],
            jitter_extra=self.jitter_extra,
            rng=self._rng,
        )

    def _apply_color_jitter(self, frames: np.ndarray) -> np.ndarray:
        from PIL import Image

        out = []
        for f in frames:
            out.append(np.array(self.color_jitter(Image.fromarray(f))))
        return np.stack(out)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch

        key = self.keys[idx]
        frames = self._read_frames(key)
        if self.augment and self.color_jitter is not None:
            frames = self._apply_color_jitter(frames)

        inputs = self.processor(list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # (T, C, H, W)
        label = self.labels[key]
        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(label, dtype=torch.long),
            "key": key,
        }


def make_collate(device: str):
    def collate(batch):
        import torch

        pv = torch.stack([b["pixel_values"] for b in batch]).to(device)
        labels = torch.stack([b["label"] for b in batch]).to(device)
        keys = [b["key"] for b in batch]
        return {"pixel_values": pv, "labels": labels, "keys": keys}

    return collate


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_model(model_name: str, dropout: float, num_classes: int = 2):
    """Load VideoMAE and replace the Kinetics head with a 2-class linear layer."""
    import torch.nn as nn
    from transformers import VideoMAEForVideoClassification

    model = VideoMAEForVideoClassification.from_pretrained(model_name)
    hidden = model.config.hidden_size

    # Replace classifier. Wrap in a small block so dropout is applied pre-logits.
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(hidden, num_classes),
    )
    return model


def set_trainable(model, phase: str, unfreeze_layers: int) -> None:
    """Configure which parameters require grad for the given phase."""
    for p in model.parameters():
        p.requires_grad = False

    # fc_norm always trains when we're training anything (it's cheap and head-adjacent).
    for p in model.fc_norm.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    if phase == "finetune":
        n = min(unfreeze_layers, len(model.videomae.encoder.layer))
        for layer in model.videomae.encoder.layer[-n:]:
            for p in layer.parameters():
                p.requires_grad = True


def build_optimizer(model, phase: str, head_lr: float, finetune_lr: float, weight_decay: float):
    """Two param groups: head-side (higher LR) and unfrozen backbone (lower LR)."""
    head_params = [p for p in model.classifier.parameters() if p.requires_grad] + [
        p for p in model.fc_norm.parameters() if p.requires_grad
    ]
    backbone_params = []
    if phase == "finetune":
        backbone_params = [
            p for p in model.videomae.encoder.parameters() if p.requires_grad
        ]
    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": finetune_lr})
    import torch

    return torch.optim.AdamW(groups, weight_decay=weight_decay)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


def evaluate(model, loader, device, threshold: float = 0.5, max_batches: int = 0):
    """Return dict of metrics + per-clip predictions for error analysis.

    If max_batches > 0, only the first max_batches batches are evaluated (smoke testing).
    """
    import torch

    model.eval()
    all_probs, all_labels, all_keys = [], [], []
    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            if max_batches and bidx >= max_batches:
                break
            logits = model(pixel_values=batch["pixel_values"]).logits
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
            all_keys.extend(batch.get("keys", []))

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0

    return {
        "n": len(labels),
        "precision_yes": precision,
        "recall_yes": recall,
        "f1_yes": f1,
        "accuracy": accuracy,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "probs": probs,
        "labels": labels,
        "keys": all_keys,
    }


def threshold_sweep(probs: np.ndarray, labels: np.ndarray) -> list[dict]:
    """For a range of thresholds, report precision/recall — find best precision at recall>=0.70."""
    out = []
    for t in np.arange(0.3, 0.81, 0.05):
        preds = (probs >= t).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out.append({"threshold": float(round(t, 2)), "precision": prec, "recall": rec})
    return out


def error_report(metrics: dict, keys: list[tuple[str, int]]) -> dict:
    """Attach ground-truth notes to false positives / false negatives."""
    import pandas as pd

    probs = metrics["probs"]
    labels = metrics["labels"]
    preds = (probs >= 0.5).astype(int)

    gt = pd.read_csv(GROUND_TRUTH_PATH)
    gt["game_id"] = gt["game_id"].astype(str).str.zfill(10)
    gt["event_id"] = gt["event_id"].astype(int)
    lookup = {(r["game_id"], int(r["event_id"])): r for _, r in gt.iterrows()}

    fps, fns = [], []
    for i, key in enumerate(keys):
        if preds[i] == labels[i]:
            continue
        row = lookup.get(key, {})
        note_val = row.get("note", "")
        note_str = "" if (note_val is None or (isinstance(note_val, float) and np.isnan(note_val)) or str(note_val) == "nan") else str(note_val)
        entry = {
            "game_id": key[0],
            "event_id": key[1],
            "actual": "YES" if labels[i] == 1 else "NO",
            "predicted": "YES" if preds[i] == 1 else "NO",
            "prob_yes": float(round(float(probs[i]), 3)),
            "description": str(row.get("description", "")),
            "note": note_str,
        }
        (fps if preds[i] == 1 else fns).append(entry)
    return {"false_positives": fps, "false_negatives": fns}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_phase(
    model,
    loader,
    val_loader,
    optimizer,
    device,
    epochs: int,
    patience: int,
    best_state: dict,
    history: list,
    phase_name: str,
    yes_weight: float,
    max_train_batches: int = 0,
    max_val_batches: int = 0,
) -> tuple[dict, list, float]:
    """Run `epochs` of training. Updates best_state in place. Returns (best_state, history, best_precision)."""
    import torch
    from torch.nn import CrossEntropyLoss

    weights = torch.tensor([1.0, yes_weight], dtype=torch.float32).to(device)
    loss_fn = CrossEntropyLoss(weight=weights)

    best_prec = best_state.get("val_precision_yes", 0.0)
    epochs_since_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running, nseen = 0.0, 0
        for bidx, batch in enumerate(loader):
            if max_train_batches and bidx >= max_train_batches:
                break
            optimizer.zero_grad()
            logits = model(pixel_values=batch["pixel_values"]).logits
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * batch["labels"].shape[0]
            nseen += batch["labels"].shape[0]

        train_loss = running / max(nseen, 1)
        val = evaluate(model, val_loader, device, max_batches=max_val_batches)
        val_loss = float("nan")  # we don't compute val loss to keep eval in no_grad; track via metrics

        rec = {
            "phase": phase_name,
            "epoch": len(history) + 1,
            "train_loss": train_loss,
            "val_precision_yes": val["precision_yes"],
            "val_recall_yes": val["recall_yes"],
            "val_f1_yes": val["f1_yes"],
            "val_accuracy": val["accuracy"],
            "confusion": val["confusion"],
        }
        history.append(rec)
        logger.info(
            "[%s] epoch %d | train_loss %.4f | val P %.3f R %.3f F1 %.3f acc %.3f | %s",
            phase_name, rec["epoch"], train_loss, val["precision_yes"], val["recall_yes"],
            val["f1_yes"], val["accuracy"], val["confusion"],
        )

        if val["precision_yes"] > best_prec:
            best_prec = val["precision_yes"]
            best_state.clear()
            best_state.update(
                {
                    "val_precision_yes": val["precision_yes"],
                    "val_recall_yes": val["recall_yes"],
                    "val_f1_yes": val["f1_yes"],
                    "val_accuracy": val["accuracy"],
                    "confusion": val["confusion"],
                    "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    "epoch": rec["epoch"],
                    "phase": phase_name,
                }
            )
            epochs_since_improve = 0
            logger.info("  >> new best val precision %.3f @ epoch %d", best_prec, rec["epoch"])
        else:
            epochs_since_improve += 1
            if patience and epochs_since_improve >= patience:
                logger.info("  >> early stopping (%d epochs without precision improvement)", patience)
                break

    return best_state, history, best_prec


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run_training(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader

    set_seed(args.seed)
    device = pick_device(args.device)
    logger.info("Device: %s", device)

    labels = load_labeled_keys()
    train_keys, val_keys = load_split_keys()
    # Keep only keys that actually have labels (YES/NO).
    train_keys = [k for k in train_keys if k in labels]
    val_keys = [k for k in val_keys if k in labels]
    anchors = load_anchors()
    window = parse_window(args.temporal_window)

    # Optional frame cache: build-once, reuse across epochs (eliminates per-epoch decode).
    if args.build_cache:
        logger.info("Building frame cache -> %s", args.frame_cache)
        build_frame_cache(
            train_keys + val_keys, labels, anchors, window,
            cache_frames=args.cache_frames, cache_size=args.cache_size,
            out_path=Path(args.frame_cache),
        )
        logger.info("Cache built. Re-run without --build-cache to train.")
        return

    frame_cache = None
    if Path(args.frame_cache).exists():
        logger.info("Loading frame cache: %s", args.frame_cache)
        frame_cache = load_frame_cache(Path(args.frame_cache))
        logger.info("Cache loaded: %d clips", len(frame_cache))

    logger.info(
        "Split: train=%d val=%d | window=%s | jitter_extra=%d | augment=%s | cache=%s",
        len(train_keys), len(val_keys), window, args.jitter if args.augment else 0, args.augment,
        "yes" if frame_cache is not None else "no (live decode)",
    )

    train_ds = LandingFoulDataset(
        train_keys, labels, anchors, window,
        augment=args.augment, num_frames=args.num_frames,
        jitter_extra=args.jitter if args.augment else 0,
        model_name=args.model, seed=args.seed,
        frame_cache=frame_cache, cache_frames=args.cache_frames,
    )
    val_ds = LandingFoulDataset(
        val_keys, labels, anchors, window,
        augment=False, num_frames=args.num_frames,
        jitter_extra=0, model_name=args.model, seed=args.seed + 1,
        frame_cache=frame_cache, cache_frames=args.cache_frames,
    )

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
        collate_fn=make_collate(device), num_workers=0, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        collate_fn=make_collate(device), num_workers=0,
    )

    model = build_model(args.model, dropout=args.dropout)
    model = model.to(device)

    history: list[dict] = []
    best_state: dict[str, Any] = {"val_precision_yes": 0.0}

    phases = []
    if args.phase in ("head", "two-phase"):
        phases.append(("head", args.head_epochs, args.head_lr))
    if args.phase in ("finetune", "two-phase"):
        phases.append(("finetune", args.finetune_epochs, args.finetune_lr))

    for phase_name, epochs, lr in phases:
        set_trainable(model, phase_name, args.unfreeze_layers)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("=== Phase: %s | epochs=%d | lr=%g | trainable params=%d ===",
                    phase_name, epochs, lr, n_train)
        optimizer = build_optimizer(
            model, phase_name, head_lr=args.head_lr, finetune_lr=args.finetune_lr,
            weight_decay=args.weight_decay,
        )
        best_state, history, _ = train_phase(
            model, train_loader, val_loader, optimizer, device,
            epochs=epochs, patience=args.patience,
            best_state=best_state, history=history,
            phase_name=phase_name, yes_weight=args.yes_weight,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
        )

    # ---- Final evaluation on best checkpoint ----
    logger.info("Loading best checkpoint (epoch %s, phase %s)",
                best_state.get("epoch"), best_state.get("phase"))
    model.load_state_dict({k: v.to(device) for k, v in best_state["state_dict"].items()})
    final = evaluate(model, val_loader, device, max_batches=args.max_val_batches)
    sweep = threshold_sweep(final["probs"], final["labels"])
    errors = error_report(final, final["keys"])

    prec, rec = final["precision_yes"], final["recall_yes"]
    gate = {
        "precision_target": PRECISION_GATE,
        "recall_target": RECALL_GATE,
        "precision_pass": bool(prec >= PRECISION_GATE),
        "recall_pass": bool(rec >= RECALL_GATE),
        "verdict": (
            "PASS" if (prec >= PRECISION_GATE and rec >= RECALL_GATE)
            else "PROMISING" if prec >= 0.70
            else "MARGINAL" if prec >= 0.55
            else "BELOW_BASELINE"
        ),
    }
    logger.info("FINAL val: P %.3f R %.3f F1 %.3f acc %.3f | %s",
                prec, rec, final["f1_yes"], final["accuracy"], gate["verdict"])
    logger.info("Confusion: %s", final["confusion"])

    # ---- Save checkpoint + metrics ----
    BEST_CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state["state_dict"],
            "config": vars(args),
            "best": {
                "epoch": best_state.get("epoch"),
                "phase": best_state.get("phase"),
                "val_precision_yes": best_state["val_precision_yes"],
                "val_recall_yes": best_state["val_recall_yes"],
                "val_f1_yes": best_state["val_f1_yes"],
                "val_accuracy": best_state["val_accuracy"],
                "confusion": best_state["confusion"],
            },
        },
        BEST_CKPT_PATH,
    )
    logger.info("Saved checkpoint to %s", BEST_CKPT_PATH)

    metrics_out = {
        "config": vars(args),
        "device": device,
        "n_train": len(train_keys),
        "n_val": len(val_keys),
        "history": history,
        "best": {
            "epoch": best_state.get("epoch"),
            "phase": best_state.get("phase"),
            "val_precision_yes": best_state["val_precision_yes"],
            "val_recall_yes": best_state["val_recall_yes"],
            "val_f1_yes": best_state["val_f1_yes"],
            "val_accuracy": best_state["val_accuracy"],
            "confusion": best_state["confusion"],
        },
        "final_eval": {
            "precision_yes": prec,
            "recall_yes": rec,
            "f1_yes": final["f1_yes"],
            "accuracy": final["accuracy"],
            "confusion": final["confusion"],
            "threshold_sweep": sweep,
        },
        "gate": gate,
        "errors": errors,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    logger.info("Saved metrics to %s", METRICS_PATH)

    # Console error summary
    if errors["false_positives"]:
        print(f"\n--- False Positives ({len(errors['false_positives'])}) ---")
        for e in errors["false_positives"][:20]:
            note = f" | {e['note']}" if e["note"] else ""
            print(f"  {e['game_id']}_{e['event_id']} p={e['prob_yes']}: {e['description'][:60]}{note}")
    if errors["false_negatives"]:
        print(f"\n--- False Negatives ({len(errors['false_negatives'])}) ---")
        for e in errors["false_negatives"][:20]:
            note = f" | {e['note']}" if e["note"] else ""
            print(f"  {e['game_id']}_{e['event_id']} p={e['prob_yes']}: {e['description'][:60]}{note}")


def run_evaluate_only(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader

    set_seed(args.seed)
    device = pick_device(args.device)

    labels = load_labeled_keys()
    _, val_keys = load_split_keys()
    val_keys = [k for k in val_keys if k in labels]
    anchors = load_anchors()
    window = parse_window(args.temporal_window)

    frame_cache = None
    if Path(args.frame_cache).exists():
        frame_cache = load_frame_cache(Path(args.frame_cache))

    val_ds = LandingFoulDataset(
        val_keys, labels, anchors, window,
        augment=False, num_frames=args.num_frames,
        jitter_extra=0, model_name=args.model, seed=args.seed + 1,
        frame_cache=frame_cache, cache_frames=args.cache_frames,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        collate_fn=make_collate(device), num_workers=0,
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(args.model, dropout=args.dropout)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)

    final = evaluate(model, val_loader, device, max_batches=args.max_val_batches)
    sweep = threshold_sweep(final["probs"], final["labels"])
    errors = error_report(final, final["keys"])

    prec, rec = final["precision_yes"], final["recall_yes"]
    print(f"\n{'='*60}\nEvaluate-only | checkpoint: {args.checkpoint}\n{'='*60}")
    print(f"Val n={final['n']} | P_yes={prec:.3f} R_yes={rec:.3f} F1={final['f1_yes']:.3f} acc={final['accuracy']:.3f}")
    print(f"Confusion: {final['confusion']}")
    print(f"Gate: precision {'PASS' if prec>=PRECISION_GATE else 'FAIL'} ({prec:.3f}>={PRECISION_GATE}) | "
          f"recall {'PASS' if rec>=RECALL_GATE else 'FAIL'} ({rec:.3f}>={RECALL_GATE})")
    print("\nThreshold sweep:")
    for row in sweep:
        print(f"  t={row['threshold']:.2f}  P={row['precision']:.3f}  R={row['recall']:.3f}")
    if errors["false_positives"]:
        print(f"\nFalse Positives ({len(errors['false_positives'])}):")
        for e in errors["false_positives"][:20]:
            print(f"  {e['game_id']}_{e['event_id']} p={e['prob_yes']}: {e['description'][:60]}")
    if errors["false_negatives"]:
        print(f"\nFalse Negatives ({len(errors['false_negatives'])}):")
        for e in errors["false_negatives"][:20]:
            print(f"  {e['game_id']}_{e['event_id']} p={e['prob_yes']}: {e['description'][:60]}")


def main() -> None:
    p = argparse.ArgumentParser(description="End-to-end VideoMAE fine-tuning for landing fouls")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"HuggingFace model ID (default: {DEFAULT_MODEL})")
    p.add_argument("--phase", default="two-phase", choices=["head", "finetune", "two-phase"])
    p.add_argument("--head-epochs", type=int, default=5)
    p.add_argument("--finetune-epochs", type=int, default=15)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--finetune-lr", type=float, default=2e-5)
    p.add_argument("--unfreeze-layers", type=int, default=4, help="Top-N encoder layers to unfreeze in finetune phase")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--temporal-window", default="0.0,1.0", help="Global window as 'lo,hi' fractions of clip")
    p.add_argument("--jitter", type=int, default=6, help="Oversample frames for random temporal jitter (train only)")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable augmentation")
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--yes-weight", type=float, default=1.0, help="CrossEntropy weight on YES class")
    p.add_argument("--patience", type=int, default=6, help="Early stopping patience on val precision (0=off)")
    p.add_argument("--max-train-batches", type=int, default=0, help="Cap train batches/epoch (0=unlimited; smoke testing)")
    p.add_argument("--max-val-batches", type=int, default=0, help="Cap val batches/epoch (0=unlimited; smoke testing)")
    p.add_argument("--frame-cache", default=str(DEFAULT_CACHE_PATH), help="Path to frame cache npz (decode-once)")
    p.add_argument("--build-cache", action="store_true", help="Decode all clips once and write frame cache, then exit")
    p.add_argument("--cache-frames", type=int, default=32, help="Frames per clip stored in cache")
    p.add_argument("--cache-size", type=int, default=256, help="Cached frame side length (resized before storage)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--checkpoint", default=str(BEST_CKPT_PATH))
    args = p.parse_args()

    if args.evaluate_only:
        run_evaluate_only(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
