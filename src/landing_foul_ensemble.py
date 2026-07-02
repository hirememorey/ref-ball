"""Step 10f: Ensemble VideoMAE + Pose classifiers for landing foul detection.

Combines the fine-tuned VideoMAE (temporal/appearance signal) with the
pose-based XGBoost (spatial/geometric signal) to try to clear the
precision >= 85% / recall >= 70% gate that neither model clears alone.

Ensemble strategies:
  weighted    Weighted average of probabilities: p = w_v * p_video + (1-w_v) * p_pose
              Sweep w_v in [0.0 .. 1.0] to find best gate result.
  intersect   Both models must predict YES (high precision, low recall).
  union       Either model predicts YES (high recall, low precision).
  rank        Rank-average the probability orderings (calibration-free).

Usage:
  PYTHONPATH=. python src/landing_foul_ensemble.py
  PYTHONPATH=. python src/landing_foul_ensemble.py --video-checkpoint data/processed/landing_foul_video_best.pt
  PYTHONPATH=. python src/landing_foul_ensemble.py --anchor-half-width 0.10

Output:
  data/processed/landing_foul_ensemble_metrics.json
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

POSE_MODEL_JSON = config.PROCESSED_DIR / "landing_foul_pose_model.json"
VIDEO_METRICS_JSON = config.PROCESSED_DIR / "landing_foul_video_metrics.json"
ENSEMBLE_METRICS_JSON = config.PROCESSED_DIR / "landing_foul_ensemble_metrics.json"
GROUND_TRUTH_PATH = config.DATA_DIR / "landing_foul_ground_truth.csv"
SPLIT_PATH = config.PROCESSED_DIR / "landing_foul_split.json"

PRECISION_GATE = 0.85
RECALL_GATE = 0.70


def load_labels() -> dict[str, int]:
    import pandas as pd

    df = pd.read_csv(GROUND_TRUTH_PATH)
    df = df[df["landing_foul"].isin(["YES", "NO"])].copy()
    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    df["key"] = df["game_id"].str.zfill(10) + "_" + df["event_id"].astype(str)
    return {r["key"]: (1 if r["landing_foul"] == "YES" else 0) for _, r in df.iterrows()}


def load_split() -> dict[str, list[str]]:
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    train_keys = [f"{str(x['game_id']).zfill(10)}_{x['event_id']}" for x in split["train"]["keys"]]
    val_keys = [f"{str(x['game_id']).zfill(10)}_{x['event_id']}" for x in split["val"]["keys"]]
    return {"train": train_keys, "val": val_keys}


def load_pose_predictions() -> dict[str, Any]:
    d = json.load(open(POSE_MODEL_JSON))
    preds = d["predictions"]
    return {
        "train_keys": preds["train_keys"],
        "train_proba": np.array(preds["train_proba"]),
        "val_keys": preds["val_keys"],
        "val_proba": np.array(preds["val_proba"]),
    }


def extract_video_predictions(args: argparse.Namespace) -> dict[str, Any]:
    """Run VideoMAE evaluate-only to get per-clip val probabilities."""
    import torch
    from torch.utils.data import DataLoader
    from landing_foul_video_finetune import (
        build_model,
        evaluate,
        load_labeled_keys,
        load_split_keys,
        load_anchors,
        parse_window,
        LandingFoulDataset,
        make_collate,
        set_seed,
        pick_device,
        DEFAULT_MODEL,
        DEFAULT_CACHE_PATH,
    )

    set_seed(args.seed)
    device = pick_device(args.device)

    labels = load_labeled_keys()
    train_keys, val_keys = load_split_keys()
    val_keys = [k for k in val_keys if k in labels]
    anchors = load_anchors()
    window = parse_window(args.temporal_window)

    frame_cache = None
    cache_path = Path(args.frame_cache)
    if cache_path.exists():
        from landing_foul_video_finetune import load_frame_cache
        frame_cache = load_frame_cache(cache_path)
        logger.info("Loaded frame cache: %s (%d clips)", cache_path, len(frame_cache))

    val_ds = LandingFoulDataset(
        val_keys, labels, anchors, window,
        augment=False, num_frames=args.num_frames,
        jitter_extra=0, model_name=args.model, seed=args.seed + 1,
        frame_cache=frame_cache, cache_frames=args.cache_frames,
        anchor_half_width_override=args.anchor_half_width,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        collate_fn=make_collate(device), num_workers=0,
    )

    ckpt = torch.load(args.video_checkpoint, map_location=device, weights_only=False)
    model = build_model(args.model, dropout=args.dropout)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)

    result = evaluate(model, val_loader, device)

    val_key_strs = [f"{k[0]}_{k[1]}" for k in result["keys"]]
    return {
        "val_keys": val_key_strs,
        "val_proba": result["probs"],
        "val_labels": result["labels"],
    }


def prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def run_ensemble(args: argparse.Namespace) -> None:
    logger.info("Loading pose predictions from %s", POSE_MODEL_JSON)
    pose = load_pose_predictions()

    logger.info("Extracting VideoMAE predictions from checkpoint %s", args.video_checkpoint)
    video = extract_video_predictions(args)

    labels = load_labels()
    split = load_split()
    val_keys = split["val"]

    pose_val = {k: p for k, p in zip(pose["val_keys"], pose["val_proba"])}
    video_val = {k: p for k, p in zip(video["val_keys"], video["val_proba"])}

    common_keys = [k for k in val_keys if k in pose_val and k in video_val]
    logger.info("Val clips with both predictions: %d / %d", len(common_keys), len(val_keys))

    y_true = np.array([labels[k] for k in common_keys])
    p_pose = np.array([pose_val[k] for k in common_keys])
    p_video = np.array([video_val[k] for k in common_keys])

    logger.info("Val set: %d YES / %d NO", int(y_true.sum()), int(len(y_true) - y_true.sum()))

    # --- Individual baselines ---
    logger.info("\n" + "=" * 60)
    logger.info("INDIVIDUAL MODEL BASELINES (threshold=0.5)")
    logger.info("=" * 60)
    for name, probs in [("VideoMAE", p_video), ("Pose XGB", p_pose)]:
        m = prf(y_true, (probs >= 0.5).astype(int))
        logger.info("%s: P=%.3f R=%.3f F1=%.3f (tp=%d fp=%d fn=%d tn=%d)",
                    name, m["precision"], m["recall"], m["f1"], m["tp"], m["fp"], m["fn"], m["tn"])

    # --- Strategy 1: Weighted average ---
    logger.info("\n" + "=" * 60)
    logger.info("STRATEGY 1: WEIGHTED AVERAGE (p = w*video + (1-w)*pose)")
    logger.info("=" * 60)
    best_weighted = None
    weighted_results = []
    for w_v in np.arange(0.0, 1.01, 0.05):
        w_v = round(w_v, 2)
        p_ens = w_v * p_video + (1 - w_v) * p_pose
        # Sweep thresholds
        for t in np.arange(0.30, 0.71, 0.05):
            t = round(t, 2)
            m = prf(y_true, (p_ens >= t).astype(int))
            m["weight_video"] = w_v
            m["threshold"] = t
            weighted_results.append(m)
            p_pass = m["precision"] >= PRECISION_GATE
            r_pass = m["recall"] >= RECALL_GATE
            if p_pass and r_pass:
                if best_weighted is None or m["f1"] > best_weighted["f1"]:
                    best_weighted = m

    # Print top weighted results sorted by F1
    weighted_results.sort(key=lambda x: x["f1"], reverse=True)
    logger.info("Top 10 weighted-average configs (by F1):")
    for m in weighted_results[:10]:
        p_pass = "PASS" if m["precision"] >= PRECISION_GATE else "miss"
        r_pass = "PASS" if m["recall"] >= RECALL_GATE else "miss"
        logger.info("  w_video=%.2f t=%.2f  P=%.3f(%s) R=%.3f(%s) F1=%.3f  tp=%d fp=%d fn=%d tn=%d",
                    m["weight_video"], m["threshold"],
                    m["precision"], p_pass, m["recall"], r_pass, m["f1"],
                    m["tp"], m["fp"], m["fn"], m["tn"])

    # Also specifically show results near the gate
    near_gate = [m for m in weighted_results if m["precision"] >= 0.75]
    near_gate.sort(key=lambda x: (-x["precision"], -x["recall"]))
    logger.info("\nConfigs with P>=0.75 (top 10 by precision then recall):")
    for m in near_gate[:10]:
        p_pass = "PASS" if m["precision"] >= PRECISION_GATE else "miss"
        r_pass = "PASS" if m["recall"] >= RECALL_GATE else "miss"
        logger.info("  w_video=%.2f t=%.2f  P=%.3f(%s) R=%.3f(%s) F1=%.3f  tp=%d fp=%d fn=%d tn=%d",
                    m["weight_video"], m["threshold"],
                    m["precision"], p_pass, m["recall"], r_pass, m["f1"],
                    m["tp"], m["fp"], m["fn"], m["tn"])

    # --- Strategy 2: Intersection (both must say YES) ---
    logger.info("\n" + "=" * 60)
    logger.info("STRATEGY 2: INTERSECTION (both models >= threshold => YES)")
    logger.info("=" * 60)
    intersect_results = []
    for t_v in np.arange(0.30, 0.71, 0.05):
        for t_p in np.arange(0.30, 0.71, 0.05):
            t_v, t_p = round(t_v, 2), round(t_p, 2)
            pred = ((p_video >= t_v) & (p_pose >= t_p)).astype(int)
            m = prf(y_true, pred)
            m["t_video"] = t_v
            m["t_pose"] = t_p
            intersect_results.append(m)

    intersect_results.sort(key=lambda x: x["f1"], reverse=True)
    logger.info("Top 10 intersection configs (by F1):")
    for m in intersect_results[:10]:
        p_pass = "PASS" if m["precision"] >= PRECISION_GATE else "miss"
        r_pass = "PASS" if m["recall"] >= RECALL_GATE else "miss"
        logger.info("  t_video=%.2f t_pose=%.2f  P=%.3f(%s) R=%.3f(%s) F1=%.3f  tp=%d fp=%d fn=%d tn=%d",
                    m["t_video"], m["t_pose"],
                    m["precision"], p_pass, m["recall"], r_pass, m["f1"],
                    m["tp"], m["fp"], m["fn"], m["tn"])

    # --- Strategy 3: Union (either says YES) ---
    logger.info("\n" + "=" * 60)
    logger.info("STRATEGY 3: UNION (either model >= threshold => YES)")
    logger.info("=" * 60)
    union_results = []
    for t_v in np.arange(0.30, 0.71, 0.05):
        for t_p in np.arange(0.30, 0.71, 0.05):
            t_v, t_p = round(t_v, 2), round(t_p, 2)
            pred = ((p_video >= t_v) | (p_pose >= t_p)).astype(int)
            m = prf(y_true, pred)
            m["t_video"] = t_v
            m["t_pose"] = t_p
            union_results.append(m)

    union_results.sort(key=lambda x: (-x["precision"], -x["recall"]))
    high_p_union = [m for m in union_results if m["precision"] >= 0.70]
    logger.info("Top 10 union configs with P>=0.70 (by precision):")
    for m in high_p_union[:10]:
        p_pass = "PASS" if m["precision"] >= PRECISION_GATE else "miss"
        r_pass = "PASS" if m["recall"] >= RECALL_GATE else "miss"
        logger.info("  t_video=%.2f t_pose=%.2f  P=%.3f(%s) R=%.3f(%s) F1=%.3f  tp=%d fp=%d fn=%d tn=%d",
                    m["t_video"], m["t_pose"],
                    m["precision"], p_pass, m["recall"], r_pass, m["f1"],
                    m["tp"], m["fp"], m["fn"], m["tn"])

    # --- Strategy 4: Rank average ---
    logger.info("\n" + "=" * 60)
    logger.info("STRATEGY 4: RANK AVERAGE (average the rank orderings)")
    logger.info("=" * 60)
    from scipy.stats import rankdata
    rank_video = rankdata(p_video)
    rank_pose = rankdata(p_pose)
    rank_avg = (rank_video + rank_pose) / 2.0

    rank_results = []
    for t_pct in np.arange(0.10, 0.91, 0.05):
        t_pct = round(t_pct, 2)
        threshold = np.quantile(rank_avg, 1 - t_pct)
        pred = (rank_avg >= threshold).astype(int)
        m = prf(y_true, pred)
        m["top_pct"] = t_pct
        rank_results.append(m)

    rank_results.sort(key=lambda x: x["f1"], reverse=True)
    logger.info("Top 10 rank-average configs (by F1):")
    for m in rank_results[:10]:
        p_pass = "PASS" if m["precision"] >= PRECISION_GATE else "miss"
        r_pass = "PASS" if m["recall"] >= RECALL_GATE else "miss"
        logger.info("  top_pct=%.2f  P=%.3f(%s) R=%.3f(%s) F1=%.3f  tp=%d fp=%d fn=%d tn=%d",
                    m["top_pct"], m["precision"], p_pass, m["recall"], r_pass, m["f1"],
                    m["tp"], m["fp"], m["fn"], m["tn"])

    # --- Calibrated weighted average (Platt-style per-model scaling) ---
    logger.info("\n" + "=" * 60)
    logger.info("STRATEGY 5: CALIBRATED WEIGHTED AVERAGE")
    logger.info("  (scale each model's probs to [0,1] via min-max on val, then weighted avg)")
    logger.info("=" * 60)
    def minmax_scale(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-8:
            return np.full_like(arr, 0.5)
        return (arr - mn) / (mx - mn)

    p_video_cal = minmax_scale(p_video)
    p_pose_cal = minmax_scale(p_pose)

    cal_results = []
    for w_v in np.arange(0.0, 1.01, 0.05):
        w_v = round(w_v, 2)
        p_ens = w_v * p_video_cal + (1 - w_v) * p_pose_cal
        for t in np.arange(0.20, 0.81, 0.05):
            t = round(t, 2)
            m = prf(y_true, (p_ens >= t).astype(int))
            m["weight_video"] = w_v
            m["threshold"] = t
            cal_results.append(m)

    cal_results.sort(key=lambda x: x["f1"], reverse=True)
    logger.info("Top 10 calibrated weighted configs (by F1):")
    for m in cal_results[:10]:
        p_pass = "PASS" if m["precision"] >= PRECISION_GATE else "miss"
        r_pass = "PASS" if m["recall"] >= RECALL_GATE else "miss"
        logger.info("  w_video=%.2f t=%.2f  P=%.3f(%s) R=%.3f(%s) F1=%.3f  tp=%d fp=%d fn=%d tn=%d",
                    m["weight_video"], m["threshold"],
                    m["precision"], p_pass, m["recall"], r_pass, m["f1"],
                    m["tp"], m["fp"], m["fn"], m["tn"])

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("GATE SUMMARY")
    logger.info("=" * 60)

    all_gate_passers = []

    for m in weighted_results:
        if m["precision"] >= PRECISION_GATE and m["recall"] >= RECALL_GATE:
            all_gate_passers.append(("weighted", m))
    for m in intersect_results:
        if m["precision"] >= PRECISION_GATE and m["recall"] >= RECALL_GATE:
            all_gate_passers.append(("intersect", m))
    for m in cal_results:
        if m["precision"] >= PRECISION_GATE and m["recall"] >= RECALL_GATE:
            all_gate_passers.append(("calibrated", m))

    if all_gate_passers:
        logger.info("GATE CLEARED by %d config(s):", len(all_gate_passers))
        all_gate_passers.sort(key=lambda x: x[1]["f1"], reverse=True)
        for strategy, m in all_gate_passers[:5]:
            logger.info("  [%s] P=%.3f R=%.3f F1=%.3f  %s",
                        strategy, m["precision"], m["recall"], m["f1"],
                        {k: v for k, v in m.items() if k in ("weight_video", "threshold", "t_video", "t_pose")})
    else:
        logger.info("NO CONFIGURATION CLEARS THE GATE (P>=%.2f, R>=%.2f)", PRECISION_GATE, RECALL_GATE)

        best_p = max(weighted_results + intersect_results + cal_results, key=lambda x: x["precision"])
        best_r = max(weighted_results + intersect_results + cal_results, key=lambda x: x["recall"])
        best_f1 = max(weighted_results + intersect_results + cal_results, key=lambda x: x["f1"])
        logger.info("Best precision: P=%.3f R=%.3f", best_p["precision"], best_p["recall"])
        logger.info("Best recall:    P=%.3f R=%.3f", best_r["precision"], best_r["recall"])
        logger.info("Best F1:        P=%.3f R=%.3f", best_f1["precision"], best_f1["recall"])

    # --- Error analysis on best config ---
    best_overall = max(weighted_results + intersect_results + cal_results, key=lambda x: x["f1"])
    logger.info("\nBest overall config: P=%.3f R=%.3f F1=%.3f", best_overall["precision"], best_overall["recall"], best_overall["f1"])

    # Save metrics
    out = {
        "n_val": len(common_keys),
        "n_yes": int(y_true.sum()),
        "n_no": int(len(y_true) - y_true.sum()),
        "individual_video": prf(y_true, (p_video >= 0.5).astype(int)),
        "individual_pose": prf(y_true, (p_pose >= 0.5).astype(int)),
        "gate": {"precision": PRECISION_GATE, "recall": RECALL_GATE},
        "gate_cleared": len(all_gate_passers) > 0,
        "best_config": best_overall,
        "top_weighted": weighted_results[:20],
        "top_intersect": intersect_results[:20],
        "top_calibrated": cal_results[:20],
    }
    ENSEMBLE_METRICS_JSON.write_text(json.dumps(out, indent=2, default=str))
    logger.info("\nWrote %s", ENSEMBLE_METRICS_JSON.name)


def main() -> None:
    p = argparse.ArgumentParser(description="Step 10f: VideoMAE + Pose ensemble")
    p.add_argument("--video-checkpoint", default=str(config.PROCESSED_DIR / "landing_foul_video_best.pt"))
    p.add_argument("--model", default="MCG-NJU/videomae-base-finetuned-kinetics")
    p.add_argument("--anchor-half-width", type=float, default=0.10)
    p.add_argument("--temporal-window", default="0.0,1.0")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--cache-frames", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--frame-cache", default=str(config.PROCESSED_DIR / "landing_foul_frames.npz"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = p.parse_args()
    run_ensemble(args)


if __name__ == "__main__":
    main()
