"""One-off sweep: narrowed temporal windows on Run 5 checkpoint (evaluate-only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landing_foul_video_finetune import (
    DEFAULT_MODEL,
    PRECISION_GATE,
    RECALL_GATE,
    LandingFoulDataset,
    build_model,
    evaluate,
    load_anchors,
    load_frame_cache,
    load_labeled_keys,
    load_split_keys,
    make_collate,
    parse_window,
    pick_device,
    set_seed,
    threshold_sweep,
)

CKPT = Path("data/processed/landing_foul_video_best.pt")
CACHE = Path("data/processed/landing_foul_frames.npz")
OUT = Path("data/processed/landing_foul_temporal_window_sweep.json")
DEVICE = pick_device("auto")
SEED = 42


def pr_at_threshold(probs, labels, t=0.5):
    pred = (probs >= t).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "tp": tp, "fp": fp, "fn": fn}


def best_gate_passing(sweep):
    hits = []
    for row in sweep:
        p, r = row["precision"], row["recall"]
        if p >= PRECISION_GATE and r >= RECALL_GATE:
            f1 = 2 * p * r / (p + r + 1e-9)
            hits.append({**row, "f1": f1})
    return max(hits, key=lambda x: x["f1"]) if hits else None


def run_eval(
    model,
    val_keys,
    labels,
    anchors,
    full_cache,
    frame_cache,
    anchor_hw=None,
    cache_crop=None,
    global_window=(0.0, 1.0),
):
    set_seed(SEED)

    if cache_crop is not None:
        lo, hi = cache_crop
        cropped = {}
        for k, frames in full_cache.items():
            t = frames.shape[0]
            i0 = int(round(lo * (t - 1)))
            i1 = int(round(hi * (t - 1)))
            if i1 <= i0:
                i1 = min(t - 1, i0 + 1)
            cropped[k] = frames[i0 : i1 + 1]
        frame_cache = cropped

    ds = LandingFoulDataset(
        val_keys,
        labels,
        anchors,
        global_window,
        augment=False,
        num_frames=16,
        jitter_extra=0,
        seed=SEED + 1,
        frame_cache=frame_cache,
        cache_frames=32,
        anchor_half_width_override=anchor_hw,
    )
    loader = DataLoader(
        ds, batch_size=4, shuffle=False, collate_fn=make_collate(DEVICE), num_workers=0
    )
    final = evaluate(model, loader, DEVICE)
    probs = final["probs"]
    labels_arr = final["labels"]
    sweep = threshold_sweep(probs, labels_arr)
    t05 = pr_at_threshold(probs, labels_arr, 0.5)
    best = None
    best_f1 = 0.0
    for row in sweep:
        f1 = 2 * row["precision"] * row["recall"] / (row["precision"] + row["recall"] + 1e-9)
        if f1 > best_f1:
            best_f1 = f1
            best = {**row, "f1": f1}
    gate = best_gate_passing(sweep)
    return {"t05": t05, "best_thresh": best, "gate": gate}


def save_results(results: list) -> None:
    out = []
    for name, kind, v1, v2, r in results:
        out.append(
            {
                "name": name,
                "kind": kind,
                "param1": v1,
                "param2": v2,
                "t05": r["t05"],
                "best_thresh": r["best_thresh"],
                "gate_cleared": r["gate"] is not None,
                "gate_config": r["gate"],
            }
        )
    OUT.write_text(json.dumps(out, indent=2))


def print_summary(results: list) -> None:
    print("\n" + "=" * 90)
    print(f"{'Config':<28} {'t=0.5 P':>8} {'R':>6} {'best_t':>7} {'best P':>8} {'best R':>8} {'gate?':>6}")
    print("=" * 90)
    for name, *_rest, r in results:
        t05 = r["t05"]
        bt = r["best_thresh"]
        gate = "YES" if r["gate"] else "no"
        bt_t = bt["threshold"] if bt else float("nan")
        print(
            f"{name:<28} {t05['precision']:>8.3f} {t05['recall']:>6.3f} "
            f"{bt_t:>7.2f} {bt['precision']:>8.3f} {bt['recall']:>8.3f} {gate:>6}"
        )


def main() -> None:
    labels = load_labeled_keys()
    _, val_keys = load_split_keys()
    val_keys = [k for k in val_keys if k in labels]
    anchors = load_anchors()
    full_cache = load_frame_cache(CACHE)

    print("Loading checkpoint...", flush=True)
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model = build_model(DEFAULT_MODEL, dropout=0.4)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(DEVICE)
    model.eval()

    results = []
    common = dict(
        model=model,
        val_keys=val_keys,
        labels=labels,
        anchors=anchors,
        full_cache=full_cache,
    )

    def run_and_record(name, kind, v1, v2, **kwargs):
        print(f"Running {name}...", flush=True)
        r = run_eval(**common, **kwargs)
        results.append((name, kind, v1, v2, r))
        save_results(results)
        t05 = r["t05"]
        bt = r["best_thresh"]
        gate = "YES" if r["gate"] else "no"
        print(
            f"  -> t=0.5 P={t05['precision']:.3f} R={t05['recall']:.3f} | "
            f"best P={bt['precision']:.3f} R={bt['recall']:.3f} gate={gate}",
            flush=True,
        )

    run_and_record("baseline cache hw=0.10", None, None, None, frame_cache=full_cache, anchor_hw=0.10)

    for lo, hi, name in [
        (0.15, 0.85, "crop 15-85%"),
        (0.20, 0.80, "crop 20-80%"),
        (0.25, 0.75, "crop 25-75%"),
        (0.30, 0.70, "crop 30-70%"),
        (0.35, 0.65, "crop 35-65%"),
    ]:
        run_and_record(name, "cache_crop", lo, hi, frame_cache=True, cache_crop=(lo, hi), anchor_hw=0.10)

    for hw in [0.05, 0.06, 0.07, 0.08, 0.09]:
        run_and_record(
            f"live hw={hw}",
            "anchor_hw",
            hw,
            None,
            frame_cache=None,
            anchor_hw=hw,
        )

    for win in ["0.3,0.9", "0.35,0.85", "0.4,0.8", "0.45,0.75"]:
        run_and_record(
            f"live window {win}",
            "global_window",
            win,
            None,
            frame_cache=None,
            anchor_hw=0.10,
            global_window=parse_window(win),
        )

    print_summary(results)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
