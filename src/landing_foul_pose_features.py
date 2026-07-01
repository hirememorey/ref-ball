"""Step 10e (Phase 2): Geometric feature engineering from pose keypoints.

Transforms raw per-frame keypoints (landing_foul_poses.json) into ~20
interpretable geometric features per clip that encode the landing-foul signal:
defender foot position relative to the shooter's landing zone, shooter vertical
trajectory, contact geometry, and trajectory shape.

Image coordinate convention: y increases downward (cv2/ultralytics native).
"Up"/"elevation" = decreasing y; "down"/"descent" = increasing y.

Role assignment is robust to BoT-SORT track fragmentation (the extractor averages
~24 track IDs per clip on crowded broadcast frames): the shooter is chosen as the
track with the largest hip vertical excursion, and the defender is matched
per-frame by nearest bounding-box center to its contact-frame position so ID
switches do not break its trajectory.

Input:
  data/processed/landing_foul_poses.json
  data/processed/landing_foul_split.json   (optional; attaches split membership)
  data/landing_foul_ground_truth.csv       (attaches YES/NO labels)

Output:
  data/processed/landing_foul_pose_features.npz   (X, feature_names, keys, labels, split)
  data/processed/landing_foul_pose_roles.json     (per-clip role assignment diagnostics)

Usage:
  PYTHONPATH=. python src/landing_foul_pose_features.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

POSES_PATH = config.PROCESSED_DIR / "landing_foul_poses.json"
SPLIT_PATH = config.PROCESSED_DIR / "landing_foul_split.json"
GROUND_TRUTH_PATH = config.DATA_DIR / "landing_foul_ground_truth.csv"
FEATURES_PATH = config.PROCESSED_DIR / "landing_foul_pose_features.npz"
ROLES_PATH = config.PROCESSED_DIR / "landing_foul_pose_roles.json"

KP_CONF_THRESHOLD = 0.3
NAN = float("nan")

FEATURE_NAMES = [
    # A. Shooter vertical trajectory
    "shooter_peak_height",
    "shooter_descent_velocity",
    "shooter_landing_frame_offset",
    "shooter_airtime_frames",
    "shooter_lateral_drift",
    # B. Defender foot position relative to landing zone
    "defender_ankle_in_zone_frac",
    "min_ankle_distance",
    "defender_ankle_at_landing",
    "defender_foot_direction",
    "defender_stance_width",
    "defender_ankle_below_shooter",
    "overlap_duration_frames",
    "defender_retreat_velocity",
    # C. Contact geometry
    "contact_height",
    "body_overlap_area",
    "relative_facing_angle",
    "shooter_defender_distance_at_peak",
    # D. Trajectory shape
    "shooter_vertical_symmetry",
    "defender_closing_speed",
    "landing_zone_incursion_onset",
    # meta
    "role_assignment_confidence",
    "has_missing_data",
]


# ---------------------------------------------------------------------------
# Keypoint helpers
# ---------------------------------------------------------------------------


def _kp(frame: dict[str, Any], name: str) -> np.ndarray | None:
    """Return [x, y, conf] or None if missing/low-confidence."""
    k = frame.get("keypoints", {}).get(name)
    if k is None:
        return None
    arr = np.asarray(k, dtype=float)
    if arr[2] < KP_CONF_THRESHOLD:
        return None
    return arr


def _point(frame: dict[str, Any], name: str) -> np.ndarray | None:
    """Return [x, y] or None."""
    k = _kp(frame, name)
    return None if k is None else k[:2]


def _mid(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return (a + b) / 2.0


def _hip_center(frame: dict[str, Any]) -> np.ndarray | None:
    return _mid(_point(frame, "left_hip"), _point(frame, "right_hip"))


def _shoulder_center(frame: dict[str, Any]) -> np.ndarray | None:
    return _mid(_point(frame, "left_shoulder"), _point(frame, "right_shoulder"))


def _shoulder_width(frame: dict[str, Any]) -> float | None:
    ls = _point(frame, "left_shoulder")
    rs = _point(frame, "right_shoulder")
    if ls is None or rs is None:
        return None
    return float(np.linalg.norm(ls - rs))


def _ankle_points(frame: dict[str, Any]) -> list[np.ndarray]:
    out = []
    for n in ("left_ankle", "right_ankle"):
        p = _point(frame, n)
        if p is not None:
            out.append(p)
    return out


def _bbox_center(frame: dict[str, Any]) -> np.ndarray | None:
    b = frame.get("bbox")
    if not b or len(b) != 4:
        return None
    x1, y1, x2, y2 = b
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])


def _dist(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------
# Per-clip frame access
# ---------------------------------------------------------------------------


def _frame_persons(clip: dict[str, Any], fidx: int) -> list[dict[str, Any]]:
    out = []
    for p in clip["persons"]:
        fr = p["frames"].get(str(fidx))
        if fr is not None:
            out.append(fr)
    return out


def _contact_frame(clip: dict[str, Any]) -> int:
    """Frame index in the window closest to the annotated foul anchor."""
    target = int(round(clip["anchor_frac"] * clip["total_frames"]))
    return min(clip["frame_indices"], key=lambda f: abs(f - target))


def _build_nn_trajectory(
    clip: dict[str, Any],
    frames: list[int],
    anchor_center: np.ndarray,
    exclude_center: np.ndarray | None,
    max_match_dist: float | None = None,
) -> dict[int, dict[str, Any]]:
    """Per-frame person matching by nearest bbox center, robust to track ID switches.

    For each frame, pick the detected person whose bbox center is closest to
    `anchor_center`, skipping any within 20px of `exclude_center` (the other
    involved player). Frames with no acceptable match are simply absent.
    """
    traj: dict[int, dict[str, Any]] = {}
    if anchor_center is None:
        return traj
    for fidx in frames:
        best = None
        best_d = math.inf
        for fr in _frame_persons(clip, fidx):
            bc = _bbox_center(fr)
            if bc is None:
                continue
            if exclude_center is not None and np.linalg.norm(bc - exclude_center) < 20:
                continue
            d = float(np.linalg.norm(bc - anchor_center))
            if max_match_dist is not None and d > max_match_dist:
                continue
            if d < best_d:
                best_d = d
                best = fr
        if best is not None:
            traj[fidx] = best
    return traj


def _traj_hip_excursion(traj: dict[int, dict[str, Any]]) -> tuple[float, int, list[int]]:
    """Robust hip-y range over a NN trajectory (p10–p90). Returns (exc, peak_fidx, fidxs)."""
    ys = []
    for fidx in sorted(traj):
        hc = _hip_center(traj[fidx])
        if hc is not None:
            ys.append((fidx, float(hc[1])))
    if len(ys) < 3:
        return 0.0, -1, [f for f in sorted(traj)]
    yvals = np.array([y for _, y in ys])
    exc = float(np.percentile(yvals, 90) - np.percentile(yvals, 10))
    peak = min(ys, key=lambda t: t[1])[0]
    return exc, peak, sorted(traj)


def _closest_pair_at(clip: dict[str, Any], fidx: int) -> list[dict[str, Any]]:
    """Return the two closest persons at a frame by bbox-center distance (<=1 person → that)."""
    persons = _frame_persons(clip, fidx)
    centers = []
    for fr in persons:
        bc = _bbox_center(fr)
        if bc is not None:
            centers.append((bc, fr))
    if len(centers) < 2:
        return [c[1] for c in centers]
    best = None
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = float(np.linalg.norm(centers[i][0] - centers[j][0]))
            if best is None or d < best[0]:
                best = (d, centers[i], centers[j])
    return [best[1][1], best[2][1]]


# ---------------------------------------------------------------------------
# Role assignment
# ---------------------------------------------------------------------------


def _person_height_at(frame: dict[str, Any]) -> float | None:
    nose = _point(frame, "nose")
    ankles = _ankle_points(frame)
    mid_ankle = ankles[0] if len(ankles) == 1 else (_mid(ankles[0], ankles[1]) if len(ankles) >= 2 else None)
    h_nose_ankle = _dist(nose, mid_ankle)
    if h_nose_ankle and h_nose_ankle > 10:
        return h_nose_ankle
    hc = _hip_center(frame)
    h_hip_ankle = _dist(hc, mid_ankle)
    if h_hip_ankle and h_hip_ankle > 10:
        return h_hip_ankle * 1.7
    sw = _shoulder_width(frame)
    if sw and sw > 5:
        return sw * 3.5
    return None


def assign_roles(clip: dict[str, Any]) -> dict[str, Any]:
    """Identify shooter + primary defender and key temporal frames.

    Strategy (robust to BoT-SORT track fragmentation on crowded broadcast frames):
      1. At the contact frame, take the closest pair of persons by bbox-center
         distance — the two players involved in the foul are necessarily close.
      2. Build each player's trajectory across the whole window by nearest-neighbour
         matching to their contact-frame bbox center (survives ID switches).
      3. Shooter = the member of the pair with the larger hip vertical excursion
         (the jumper); defender = the other.
      4. Resolve peak/landing frames and the landing zone from the shooter trajectory.

    Falls back to a single-track plausible-jump shooter if no contact pair exists.
    """
    persons = clip["persons"]
    contact_fidx = _contact_frame(clip)
    window = list(clip["frame_indices"])

    pair = _closest_pair_at(clip, contact_fidx)
    if len(pair) >= 2:
        a, b = pair[0], pair[1]
        ca, cb = _bbox_center(a), _bbox_center(b)
        traj_a = _build_nn_trajectory(clip, window, ca, cb)
        traj_b = _build_nn_trajectory(clip, window, cb, ca)
        exc_a, peak_a, _ = _traj_hip_excursion(traj_a)
        exc_b, peak_b, _ = _traj_hip_excursion(traj_b)
        if exc_a >= exc_b:
            shooter_traj, defender_traj = traj_a, traj_b
            peak_fidx = peak_a
            shooter_contact = a
            defender_contact = b
            exc = exc_a
        else:
            shooter_traj, defender_traj = traj_b, traj_a
            peak_fidx = peak_b
            shooter_contact = b
            defender_contact = a
            exc = exc_b
        defender_anchor_pos = _bbox_center(defender_contact)
    elif len(pair) == 1:
        # Only one person at contact — use them as shooter, NN-match a defender.
        shooter_contact = pair[0]
        ca = _bbox_center(shooter_contact)
        shooter_traj = _build_nn_trajectory(clip, window, ca, None)
        exc, peak_fidx, _ = _traj_hip_excursion(shooter_traj)
        # Defender = nearest other person at contact to the shooter (fallback).
        defender_contact = None
        defender_anchor_pos = None
        defender_traj = {}
        for fr in _frame_persons(clip, contact_fidx):
            bc = _bbox_center(fr)
            if bc is None or ca is None:
                continue
            if np.linalg.norm(bc - ca) < 20:
                continue
            if defender_anchor_pos is None or np.linalg.norm(bc - ca) < np.linalg.norm(defender_anchor_pos - ca):
                defender_contact = fr
                defender_anchor_pos = bc
        if defender_anchor_pos is not None:
            defender_traj = _build_nn_trajectory(clip, window, defender_anchor_pos, ca)
    else:
        return {"shooter_id": None, "contact_frame": contact_fidx, "ok": False}

    if not shooter_traj or peak_fidx < 0:
        return {"shooter_id": None, "contact_frame": contact_fidx, "ok": False,
                "peak_frame": peak_fidx if peak_fidx >= 0 else contact_fidx}

    # Landing frame: after peak, the shooter frame with ankles lowest (max y).
    after = [f for f in sorted(shooter_traj) if f > peak_fidx]
    landing_fidx = peak_fidx
    ankle_ys = []
    for fidx in after:
        anks = _ankle_points(shooter_traj[fidx])
        if anks:
            ankle_ys.append((fidx, float(np.mean([a[1] for a in anks]))))
    if ankle_ys:
        landing_fidx = max(ankle_ys, key=lambda t: t[1])[0]

    # Person height from a grounded shooter frame (max ankle y overall = standing).
    grounded = sorted(shooter_traj.values(), key=lambda fr: max(
        (a[1] for a in _ankle_points(fr)), default=0.0), reverse=True)
    person_height = None
    for fr in grounded[:3]:
        h = _person_height_at(fr)
        if h and h > 0:
            person_height = h
            break
    if not person_height:
        person_height = _person_height_at(shooter_traj.get(landing_fidx) or shooter_traj.get(peak_fidx))
    if not person_height or person_height <= 0:
        return {"shooter_id": None, "contact_frame": contact_fidx, "ok": False,
                "peak_frame": peak_fidx, "landing_frame": landing_fidx}

    landing_frame_data = shooter_traj.get(landing_fidx) or shooter_traj.get(peak_fidx)
    landing_ankles = _ankle_points(landing_frame_data) if landing_frame_data else []
    shooter_landing_pos = (
        landing_ankles[0] if len(landing_ankles) == 1
        else (_mid(landing_ankles[0], landing_ankles[1]) if len(landing_ankles) >= 2 else None)
    )
    if shooter_landing_pos is None:
        shooter_landing_pos = _hip_center(landing_frame_data) if landing_frame_data else None

    peak_frame_data = shooter_traj.get(peak_fidx)
    peak_hip = _hip_center(peak_frame_data) if peak_frame_data else None
    zone_center = None
    if peak_hip is not None and shooter_landing_pos is not None:
        zone_center = np.array([peak_hip[0], shooter_landing_pos[1]])
    elif shooter_landing_pos is not None:
        zone_center = shooter_landing_pos.copy()
    zone_radius = (_shoulder_width(peak_frame_data) if peak_frame_data else None) or person_height * 0.28

    exc_norm = exc / person_height
    ok = defender_anchor_pos is not None and shooter_landing_pos is not None
    # Plausibility: a real jump raises the hip 0.05–0.7 body heights.
    plausibility_factor = 1.0 if 0.05 <= exc_norm <= 0.7 else 0.5
    confidence = float(min(1.0, exc_norm / 0.35)) * (0.5 + 0.5 * ok) * plausibility_factor

    return {
        "ok": ok,
        "shooter_id": None,  # NN trajectories are not tied to a single track id
        "shooter_traj": shooter_traj,
        "defender_traj": defender_traj,
        "defender_anchor_pos": defender_anchor_pos,
        "contact_frame": contact_fidx,
        "peak_frame": peak_fidx,
        "landing_frame": landing_fidx,
        "person_height": person_height,
        "zone_center": zone_center,
        "zone_radius": zone_radius,
        "shooter_landing_pos": shooter_landing_pos,
        "excursion": exc,
        "exc_norm": exc_norm,
        "confidence": confidence,
        "n_shooter_frames": len(shooter_traj),
    }


# ---------------------------------------------------------------------------
# Descent window
# ---------------------------------------------------------------------------


def _descent_frames(roles: dict[str, Any], clip: dict[str, Any]) -> list[int]:
    """Frame indices in the window from peak to landing (inclusive)."""
    lo, hi = roles["peak_frame"], roles["landing_frame"]
    if hi < lo:
        lo, hi = hi, lo
    return [f for f in clip["frame_indices"] if lo <= f <= hi]


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def _safe_div(x: float | None, d: float) -> float:
    if x is None or d is None or d == 0 or not np.isfinite(x):
        return NAN
    return float(x) / d


def _velocity(positions: list[tuple[int, np.ndarray]], n_tail: int) -> float | None:
    """Mean per-step displacement (pixels/frame) over the last n_tail steps.

    positions: sorted (fidx, point) with consistent spacing assumed ~1 frame.
    """
    pts = [p for _, p in positions if p is not None]
    if len(pts) < 2:
        return None
    tail = pts[-n_tail:] if len(pts) >= n_tail else pts
    diffs = [np.linalg.norm(tail[i + 1] - tail[i]) for i in range(len(tail) - 1)]
    return float(np.mean(diffs)) if diffs else None


def compute_features(clip: dict[str, Any], roles: dict[str, Any]) -> dict[str, float]:
    f: dict[str, float] = {name: NAN for name in FEATURE_NAMES}
    f["has_missing_data"] = 1.0
    f["role_assignment_confidence"] = roles.get("confidence", 0.0)

    if not roles.get("ok"):
        return f

    H = roles["person_height"]
    peak_f = roles["peak_frame"]
    land_f = roles["landing_frame"]
    contact_f = roles["contact_frame"]
    zone_c = roles["zone_center"]
    zone_r = roles["zone_radius"]
    shooter_landing = roles["shooter_landing_pos"]
    def_anchor = roles["defender_anchor_pos"]

    shooter_traj = roles.get("shooter_traj", {})
    defender_traj = roles.get("defender_traj", {})

    def shoot_fr(idx: int) -> dict[str, Any] | None:
        return shooter_traj.get(idx)

    def def_fr(idx: int) -> dict[str, Any] | None:
        return defender_traj.get(idx)

    shooter_peak_fr = shoot_fr(peak_f)
    shooter_land_fr = shoot_fr(land_f)
    if shooter_peak_fr is None or shooter_land_fr is None:
        return f

    # --- A. Shooter vertical trajectory ---
    hip_peak = _hip_center(shooter_peak_fr)
    hip_land = _hip_center(shooter_land_fr)
    if hip_peak is not None and hip_land is not None:
        f["shooter_peak_height"] = _safe_div(float(hip_land[1] - hip_peak[1]), H)

    # descent velocity: mean downward hip velocity over last 5 frames before landing
    descent = _descent_frames(roles, clip)
    hip_seq = []
    for fidx in descent:
        fr = shoot_fr(fidx)
        if fr is None:
            continue
        hc = _hip_center(fr)
        if hc is not None:
            hip_seq.append((fidx, hc))
    if len(hip_seq) >= 2:
        ys = [p[1] for _, p in hip_seq]
        tail = ys[-5:] if len(ys) >= 5 else ys
        dv = float(np.mean([tail[i + 1] - tail[i] for i in range(len(tail) - 1)])) if len(tail) >= 2 else None
        f["shooter_descent_velocity"] = _safe_div(dv, H)

    window_start = min(clip["frame_indices"])
    f["shooter_landing_frame_offset"] = float(land_f - window_start)
    f["shooter_airtime_frames"] = float(abs(land_f - peak_f))
    if hip_peak is not None and hip_land is not None:
        f["shooter_lateral_drift"] = _safe_div(abs(float(hip_land[0] - hip_peak[0])), H)

    # Defender trajectory is prebuilt in roles (NN match to contact-frame position).

    # --- B. Defender foot position relative to landing zone ---
    in_zone = 0
    incursion_onset = None
    min_ankle_dist = math.inf
    overlap_run = 0
    max_overlap_run = 0
    prev_in_zone = False
    for fidx in descent:
        dfr = def_fr(fidx)
        if dfr is None:
            prev_in_zone = False
            overlap_run = 0
            continue
        anks = _ankle_points(dfr)
        if not anks:
            prev_in_zone = False
            overlap_run = 0
            continue
        for a in anks:
            d_land = _dist(a, shooter_landing)
            if d_land is not None and d_land < min_ankle_dist:
                min_ankle_dist = d_land
        # zone membership
        in_zone_flag = False
        if zone_c is not None:
            for a in anks:
                if float(np.linalg.norm(a - zone_c)) < zone_r:
                    in_zone_flag = True
                    break
        if in_zone_flag:
            in_zone += 1
            if incursion_onset is None:
                incursion_onset = fidx
            overlap_run = prev_in_zone + 1
            max_overlap_run = max(max_overlap_run, overlap_run)
        else:
            overlap_run = 0
        prev_in_zone = in_zone_flag

    if descent:
        f["defender_ankle_in_zone_frac"] = float(in_zone) / len(descent)
    if min_ankle_dist != math.inf:
        f["min_ankle_distance"] = _safe_div(min_ankle_dist, H)
    f["overlap_duration_frames"] = float(max_overlap_run)
    if incursion_onset is not None:
        f["landing_zone_incursion_onset"] = float(incursion_onset - peak_f)

    # defender ankle at landing
    dfr_land = def_fr(land_f)
    if dfr_land is not None:
        anks = _ankle_points(dfr_land)
        if anks and shooter_landing is not None:
            d = min((_dist(a, shooter_landing) or math.inf) for a in anks)
            if d != math.inf:
                f["defender_ankle_at_landing"] = _safe_div(d, H)
        # stance width at contact frame (use contact for the posture snapshot)
        dfr_contact = def_fr(contact_f) or dfr_land
        la = _point(dfr_contact, "left_ankle")
        ra = _point(dfr_contact, "right_ankle")
        sw = _dist(la, ra)
        if sw is not None:
            f["defender_stance_width"] = _safe_div(sw, H)
        # ankle below shooter (feet planted lower than shooter's landing feet)
        if anks and shooter_landing is not None:
            f["defender_ankle_below_shooter"] = 1.0 if max(a[1] for a in anks) > shooter_landing[1] else 0.0

    # defender foot direction & retreat velocity from ankle trajectory
    ankle_traj = []
    for fidx in descent:
        dfr = def_fr(fidx)
        if dfr is None:
            continue
        anks = _ankle_points(dfr)
        if anks and shooter_landing is not None:
            # closest ankle to shooter landing
            a = min(anks, key=lambda x: _dist(x, shooter_landing) or math.inf)
            ankle_traj.append((fidx, a))
    if len(ankle_traj) >= 2 and shooter_landing is not None:
        a0 = ankle_traj[0][1]
        v = ankle_traj[-1][1] - ankle_traj[0][1]
        target = shooter_landing - a0
        nv, nt = np.linalg.norm(v), np.linalg.norm(target)
        if nv > 1e-6 and nt > 1e-6:
            f["defender_foot_direction"] = float(np.dot(v, target) / (nv * nt))
        # retreat velocity: change in distance-to-zone-center over last 3 frames
        if zone_c is not None and len(ankle_traj) >= 2:
            tail = ankle_traj[-3:] if len(ankle_traj) >= 3 else ankle_traj
            ds = [float(np.linalg.norm(p - zone_c)) for _, p in tail]
            if len(ds) >= 2:
                # positive = distance growing = clearing out; negative = moving in
                f["defender_retreat_velocity"] = _safe_div(float(ds[-1] - ds[0]) / max(1, len(ds) - 1), H)

    # --- C. Contact geometry ---
    sfr_c = shoot_fr(contact_f)
    dfr_c = def_fr(contact_f)
    if sfr_c is not None and dfr_c is not None:
        # body overlap (IoU of bboxes)
        sb = sfr_c.get("bbox")
        db = dfr_c.get("bbox")
        if sb and db:
            f["body_overlap_area"] = float(_iou(sb, db))
        # contact height: vertical position of closest keypoint pair, relative to
        # shooter's ground (landing y). Lower => feet/legs.
        spts = [(_point(sfr_c, n), n) for n in ("left_ankle", "right_ankle", "left_knee", "right_knee",
                                                 "left_hip", "right_hip", "left_shoulder", "right_shoulder")]
        dpts = [(_point(dfr_c, n), n) for n in ("left_ankle", "right_ankle", "left_knee", "right_knee",
                                                 "left_hip", "right_hip", "left_shoulder", "right_shoulder")]
        best = None
        for sp, _ in spts:
            if sp is None:
                continue
            for dp, _ in dpts:
                if dp is None:
                    continue
                ymid = (sp[1] + dp[1]) / 2.0
                d = np.linalg.norm(sp - dp)
                if best is None or d < best[0]:
                    best = (d, ymid)
        if best is not None and shooter_landing is not None:
            # height above ground: landing_y - contact_y (positive = higher up)
            f["contact_height"] = _safe_div(float(shooter_landing[1] - best[1]), H)
        # relative facing angle: shooter shoulder line vs shooter->defender vector
        sl = _point(sfr_c, "left_shoulder")
        sr = _point(sfr_c, "right_shoulder")
        sc = _shoulder_center(sfr_c)
        dc = _bbox_center(dfr_c)
        if sl is not None and sr is not None and sc is not None and dc is not None:
            shoulder_vec = sl - sr
            to_def = dc - sc
            ns, nd = np.linalg.norm(shoulder_vec), np.linalg.norm(to_def)
            if ns > 1e-6 and nd > 1e-6:
                f["relative_facing_angle"] = float(abs(np.dot(shoulder_vec, to_def) / (ns * nd)))

    # shooter-defender distance at peak
    sfr_p = shoot_fr(peak_f)
    dfr_p = def_fr(peak_f)
    if sfr_p is not None and dfr_p is not None:
        sp = _hip_center(sfr_p)
        dp = _hip_center(dfr_p)
        d = _dist(sp, dp)
        if d is not None:
            f["shooter_defender_distance_at_peak"] = _safe_div(d, H)

    # --- D. Trajectory shape ---
    # vertical symmetry: ascent / descent duration
    pre = [fidx for fidx in clip["frame_indices"] if fidx <= peak_f]
    post = [fidx for fidx in clip["frame_indices"] if fidx >= peak_f]
    if pre and post:
        f["shooter_vertical_symmetry"] = float(len(pre)) / float(len(post))

    # defender closing speed: mean decrease in hip-to-hip distance over descent
    sep = []
    for fidx in descent:
        sfr = shoot_fr(fidx)
        dfr = def_fr(fidx)
        if sfr is None or dfr is None:
            continue
        sh = _hip_center(sfr)
        dh = _hip_center(dfr)
        d = _dist(sh, dh)
        if d is not None:
            sep.append(d)
    if len(sep) >= 2:
        # closing = distance decreasing => positive closing speed
        closing = float(sep[0] - sep[-1]) / max(1, len(sep) - 1)
        f["defender_closing_speed"] = _safe_div(closing, H)

    # If we got this far with a real shooter + defender, clear the missing flag.
    f["has_missing_data"] = 0.0
    return f


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def load_labels() -> dict[str, int]:
    import pandas as pd

    df = pd.read_csv(GROUND_TRUTH_PATH)
    df = df[df["landing_foul"].isin(["YES", "NO"])].copy()
    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    df["event_id"] = df["event_id"].astype(int)
    return {f"{r.game_id}_{r.event_id}": (1 if r.landing_foul == "YES" else 0) for r in df.itertuples()}


def load_split_membership() -> dict[str, str]:
    if not SPLIT_PATH.exists():
        return {}
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    out = {}
    for k in split.get("train", {}).get("keys", []):
        out[f"{k['game_id']}_{k['event_id']}"] = "train"
    for k in split.get("val", {}).get("keys", []):
        out[f"{k['game_id']}_{k['event_id']}"] = "val"
    return out


def build(poses_path: Path = POSES_PATH) -> None:
    with open(poses_path) as f:
        poses = json.load(f)
    labels = load_labels()
    split_mem = load_split_membership()

    keys = sorted(poses.keys())
    X = np.full((len(keys), len(FEATURE_NAMES)), NAN, dtype=float)
    game_ids, event_ids, y, split_arr = [], [], [], []
    roles_out: dict[str, Any] = {}
    n_ok = 0

    for i, key in enumerate(keys):
        clip = poses[key]
        roles = assign_roles(clip)
        feat = compute_features(clip, roles)
        for j, name in enumerate(FEATURE_NAMES):
            X[i, j] = feat[name]
        gid, eid = key.rsplit("_", 1)
        game_ids.append(gid)
        event_ids.append(int(eid))
        y.append(labels.get(key, -1))
        split_arr.append(split_mem.get(key, "none"))
        roles_out[key] = {
            "ok": roles.get("ok", False),
            "shooter_id": roles.get("shooter_id"),
            "peak_frame": roles.get("peak_frame"),
            "landing_frame": roles.get("landing_frame"),
            "contact_frame": roles.get("contact_frame"),
            "person_height": roles.get("person_height"),
            "exc_norm": roles.get("exc_norm"),
            "confidence": roles.get("confidence"),
            "n_shooter_frames": roles.get("n_shooter_frames"),
        }
        if roles.get("ok") and feat["has_missing_data"] == 0.0:
            n_ok += 1

    X = np.nan_to_num(X, nan=0.0)  # trees handle 0; NaN mainly signals missing roles
    y = np.array(y, dtype=int)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        FEATURES_PATH,
        features=X,
        feature_names=np.array(FEATURE_NAMES),
        game_ids=np.array(game_ids),
        event_ids=np.array(event_ids),
        labels=y,
        split=np.array(split_arr),
        keys=np.array(keys),
    )
    with open(ROLES_PATH, "w") as f:
        json.dump(roles_out, f, indent=2)

    logger.info("Built %d clips × %d features → %s", len(keys), len(FEATURE_NAMES), FEATURES_PATH.name)
    logger.info("Role assignment succeeded on %d/%d clips (%.0f%%)", n_ok, len(keys), 100.0 * n_ok / max(1, len(keys)))
    logger.info("Label coverage: %d labeled, %d unlabeled", int((y >= 0).sum()), int((y < 0).sum()))
    miss = np.mean(X == 0.0, axis=0)
    worst = sorted(zip(FEATURE_NAMES, miss.tolist()), key=lambda t: t[1], reverse=True)[:5]
    logger.info("Top-5 zero/missing-rate features: %s", worst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 10e Phase 2: geometric pose features")
    parser.add_argument("--poses", default=str(POSES_PATH), help="Input poses JSON path")
    args = parser.parse_args()
    build(Path(args.poses))


if __name__ == "__main__":
    main()
