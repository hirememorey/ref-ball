# Pose Estimation for Landing Foul Detection

Implementation plan for skeleton-based landing foul classification. This approach extracts human body keypoints from broadcast video and classifies fouls based on geometric relationships between shooter and defender body positions — bypassing the "feet are tiny in the frame" problem that limits both LLM grading and end-to-end video model approaches.

---

## Why Pose Estimation

The current VideoMAE fine-tuning (Run 4: 81% P / 59% R) and LLM grading (55% P / 98% R) both fail the quality gate (≥ 85% P / ≥ 70% R) for the same underlying reason: the discriminative signal for landing fouls lives in the spatial relationship between two players' lower bodies during a ~400ms window, and both approaches struggle to resolve small body parts (feet, ankles) in 960×540 broadcast footage resized to 224×224 model input.

Pose estimation addresses this directly:

| Problem | VideoMAE / LLM | Pose estimation |
|---|---|---|
| Feet are ~10 pixels in 224×224 input | Model must learn to attend to tiny regions | Keypoint detector is trained specifically to localize body joints at any scale |
| Shooter vs defender identity | Implicit; model must learn who is who | Explicit multi-person tracking; assign roles from PBP context (shooter = player in shooting motion) |
| Landing zone geometry | Encoded in pixel space; learned end-to-end | Computed directly from ankle/hip keypoints; interpretable |
| Temporal dynamics (ascent → descent → landing) | 16 frames across ~800ms; coarse | Per-frame keypoints at native fps; smooth trajectories |
| Training data requirement | 227 train clips; deep model is data-hungry | Pose extractor is pretrained; downstream classifier needs only geometric features — lightweight |

**Risk:** Broadcast camera angles are wide and oblique. Pose estimators trained on frontal/lab data may lose accuracy on distant, side-angle NBA players. Occlusion during contested shots is common. The plan includes explicit quality checks and a fallback to ensemble with VideoMAE if standalone pose classification underperforms.

---

## Architecture Overview

```
MP4 clip (960×540, 8-12s)
    │
    ▼
┌──────────────────────────┐
│  Temporal crop            │   anchor ± half_width (reuse existing clip_anchors.json)
│  → ~0.8-2.4s contact     │
│    window at native fps   │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Multi-person pose        │   ViTPose-Base or MediaPipe Pose
│  extraction               │   → N persons × 17+ keypoints × (x, y, confidence) per frame
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Person tracking +        │   Simple IoU tracker (SORT/ByteTrack) or pose-based matching
│  role assignment          │   → shooter track, defender track(s)
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Geometric feature        │   Per-frame and trajectory features from keypoints:
│  extraction               │   foot positions, vertical velocity, spatial overlap,
│                           │   landing zone projection, contact proximity
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Classifier               │   Option A: Rule-based (geometric thresholds)
│                           │   Option B: Gradient-boosted trees (XGBoost/LightGBM)
│                           │   Option C: Small temporal model (1D-CNN or GRU on keypoint sequences)
└──────────┬───────────────┘
           │
           ▼
    YES / NO prediction
```

---

## Implementation Phases

### Phase 0: Pose Estimator Selection and Validation (1-2 days)

**Goal:** Confirm that a pretrained pose estimator produces usable keypoints on NBA broadcast footage before building the full pipeline.

**Candidates:**

| Model | Pros | Cons | COCO AP |
|---|---|---|---|
| **ViTPose-Base** (ViT-B, COCO) | SOTA accuracy; robust to occlusion; transformer backbone handles long-range context | Requires top-down detection (person bbox first); heavier inference | 75.8 |
| **MMPose RTMPose-L** | Fast; good accuracy/speed trade-off; bottom-up (no detector needed) | Slightly lower accuracy than ViTPose on distant figures | 72.3 |
| **MediaPipe Pose** | Zero-setup; runs on CPU; single-person per call | Single-person only (need to crop per player); lower accuracy on distant subjects; not designed for multi-person sports | ~65 (est.) |
| **YOLOv8-Pose** | Integrated detection + pose; fast; multi-person | Lower keypoint accuracy than ViTPose for fine-grained joint localization | 69.2 |

**Recommendation:** Start with **ViTPose-Base** via the `mmpose` library (or the standalone `easy_ViTPose` wrapper). It has the best accuracy for distant, partially-occluded sports figures. Use **YOLOv8** as the person detector (top-down pipeline: detect people → estimate pose per bbox). Fall back to **MediaPipe** if ViTPose setup proves too complex for the Colab environment.

**Validation protocol:**

1. Select 10 clips spanning the difficulty range: 3 clear landing fouls, 3 clear non-landing fouls, 4 borderline/difficult cases (from val FPs and FNs).
2. Extract keypoints at native fps (30 fps) for the anchor ± 0.15 window.
3. Visually inspect overlaid skeletons. Success criteria:
   - Both ankles visible and tracked for the shooter in ≥ 80% of descent frames.
   - At least one defender's feet tracked in ≥ 60% of contact frames.
   - Keypoint confidence > 0.3 for lower-body joints in the contact window.
4. If ankle keypoints are unreliable on > 50% of clips, try spatial cropping (crop to the shooting-foul region before pose estimation) or switch to MediaPipe with per-player bbox crops.

**Output:** Validation report (which model, keypoint quality stats, failure modes). Go/no-go for Phase 1.

**Dependencies:**
```
mmpose>=1.3           # or easy-vitpose
mmdet>=3.3            # for top-down person detection
mmengine>=0.10
# OR for MediaPipe fallback:
mediapipe>=0.10
# OR for YOLOv8:
ultralytics>=8.2
```

---

### Phase 1: Pose Extraction Pipeline (`landing_foul_pose_extract.py`) (2-3 days)

**Goal:** Extract and store per-clip, per-person, per-frame keypoint data for all 284 labeled clips.

**Input:** `data/clips/landing_foul/*.mp4` + `data/processed/landing_foul_clip_anchors.json`

**Output:** `data/processed/landing_foul_poses.json` — one entry per clip:

```json
{
  "0021900028_532": {
    "fps": 30,
    "anchor_frac": 0.42,
    "half_width": 0.15,
    "frame_range": [108, 162],
    "persons": [
      {
        "track_id": 0,
        "role": "shooter",
        "frames": {
          "108": {
            "bbox": [320, 180, 420, 380],
            "keypoints": {
              "left_ankle": [345.2, 368.1, 0.87],
              "right_ankle": [358.7, 370.3, 0.91],
              "left_hip": [348.0, 290.5, 0.95],
              "right_hip": [356.2, 291.8, 0.94],
              "left_shoulder": [342.1, 220.3, 0.96],
              "right_shoulder": [360.8, 218.7, 0.97]
            }
          }
        }
      },
      {
        "track_id": 1,
        "role": "defender",
        "frames": { }
      }
    ]
  }
}
```

**Keypoint set (COCO-17 subset, lower body focus):**
- `left_ankle`, `right_ankle` — primary signal for foot position
- `left_knee`, `right_knee` — leg extension / stance width
- `left_hip`, `right_hip` — torso center, vertical velocity reference
- `left_shoulder`, `right_shoulder` — shooting motion detection
- `nose` — rough head position for person height estimation

**Processing steps per clip:**
1. Load clip, compute frame range from anchor ± half_width.
2. Run person detector (YOLOv8) on each frame → bounding boxes.
3. Run pose estimator (ViTPose) on each detected person bbox → keypoints.
4. Track persons across frames (IoU-based or pose-based matching).
5. Store raw keypoints + bboxes + track IDs.

**Role assignment (Phase 2 concern, but stub here):** Defer shooter/defender labeling to Phase 2 feature extraction. Store all tracked persons; role assignment uses heuristics (tallest vertical displacement = shooter; closest other person at contact = primary defender).

**CLI:**
```bash
PYTHONPATH=. python src/landing_foul_pose_extract.py                    # all 284 clips
PYTHONPATH=. python src/landing_foul_pose_extract.py --clip 0021900028_532  # single clip
PYTHONPATH=. python src/landing_foul_pose_extract.py --visualize --clip 0021900028_532  # overlay skeletons on video
```

**Makefile:**
```makefile
pose-extract:
	PYTHONPATH=. $(PYTHON) src/landing_foul_pose_extract.py

pose-visualize:
	PYTHONPATH=. $(PYTHON) src/landing_foul_pose_extract.py --visualize --clip $(CLIP)
```

---

### Phase 2: Geometric Feature Engineering (`landing_foul_pose_features.py`) (2-3 days)

**Goal:** Transform raw keypoint sequences into interpretable geometric features that encode the landing foul signal.

**Input:** `data/processed/landing_foul_poses.json` + `data/processed/landing_foul_split.json`

**Output:** `data/processed/landing_foul_pose_features.npz` — feature matrix (284 clips × D features) + feature names.

#### Role Assignment Heuristics

The shooter and primary defender must be identified from multi-person tracks. In broadcast NBA footage:

1. **Shooter detection:** The person with the largest upward vertical displacement (hip y-coordinate decreasing then increasing, assuming y=0 at top) in the 0.5s before the anchor frame. Shooting motion = hip rises then falls.
2. **Primary defender:** The non-shooter person whose ankle keypoints are closest to the shooter's projected landing position at the contact frame.
3. **Fallback:** If only one or two persons are reliably tracked, use them. If zero, the clip gets a missing-data flag and is excluded from pose-based classification (fall back to VideoMAE prediction).

#### Feature Categories

**A. Shooter vertical trajectory (5 features)**

| Feature | Definition | Signal |
|---|---|---|
| `shooter_peak_height` | Max hip elevation above landing-frame hip position (pixels, normalized by person height) | Confirms jump shot (vs. layup) |
| `shooter_descent_velocity` | Mean downward hip velocity in the 5 frames before landing | Faster descent = less time for defender to clear |
| `shooter_landing_frame` | Frame index where shooter ankles reach minimum elevation (landing moment) | Temporal anchor for spatial overlap check |
| `shooter_airtime_frames` | Number of frames from peak to landing | Longer hang time = wider temporal window for defender intrusion |
| `shooter_lateral_drift` | Horizontal displacement of hip centroid from peak to landing (pixels, normalized) | Forward/lateral drift into defender vs. straight up-and-down |

**B. Defender foot position relative to landing zone (8 features — the core signal)**

| Feature | Definition | Signal |
|---|---|---|
| `defender_ankle_in_zone_frac` | Fraction of descent frames where any defender ankle is within the shooter's projected landing zone | **Primary landing foul indicator** — high fraction = defender under shooter |
| `min_ankle_distance` | Minimum distance between any defender ankle and the shooter's landing position across all descent frames (normalized by person height) | Closer = more likely landing foul |
| `defender_ankle_at_landing` | Distance between closest defender ankle and shooter ankle at the landing frame | Snapshot at the critical moment |
| `defender_foot_direction` | Cosine similarity between defender ankle velocity vector and shooter landing position | Moving toward = closing out into landing zone |
| `defender_stance_width` | Distance between defender's left and right ankles at contact | Wide stance in landing zone = classic landing foul posture |
| `defender_ankle_below_shooter` | Whether defender ankle y-coordinate is below (closer to ground than) shooter ankle at landing | Feet planted while shooter lands |
| `overlap_duration_frames` | Consecutive frames where defender ankle is within landing zone threshold | Sustained intrusion vs. momentary brush |
| `defender_retreat_velocity` | Defender ankle velocity away from landing zone in the 3 frames before landing | Negative = moving into zone; positive = clearing out |

**C. Contact geometry (4 features)**

| Feature | Definition | Signal |
|---|---|---|
| `contact_height` | Height of closest keypoint pair between shooter and defender at the contact frame (normalized) | Low contact (legs/feet) = more likely landing foul |
| `body_overlap_area` | Overlap area of shooter and defender bounding boxes at contact (normalized) | More overlap = more body contact |
| `relative_facing_angle` | Angle between shooter's shoulder line and the shooter-defender vector | Defender behind/beside vs. in front |
| `shooter_defender_distance_at_peak` | Distance at shot apex — how close was the defender during the contest | Close contest vs. late closeout |

**D. Trajectory shape (3 features)**

| Feature | Definition | Signal |
|---|---|---|
| `shooter_vertical_symmetry` | Ratio of ascent duration to descent duration | Asymmetric (slow descent) may indicate contact during landing |
| `defender_closing_speed` | Mean defender-to-shooter approach velocity over the window | Fast closeout = higher landing foul risk |
| `landing_zone_incursion_onset` | Frame offset from shot release to first defender ankle entering landing zone | Early incursion = more time under shooter |

**Total: ~20 features.** Deliberately low-dimensional to avoid overfitting on 227 training clips.

**Landing zone definition:** A circle (in pixel space) centered on the shooter's hip x-coordinate at shot peak, with radius = 1.0× the shooter's shoulder width (estimated from keypoints). This approximates the "cylinder" landing space rule: the shooter is entitled to land in roughly the space they occupied at takeoff plus natural forward momentum.

**Normalization:** All distance features normalized by estimated person height (nose-to-ankle distance) to handle varying camera distances across clips.

---

### Phase 3: Pose-Based Classifier (`landing_foul_pose_classify.py`) (1-2 days)

**Goal:** Train a classifier on geometric features. Three approaches in priority order.

#### Option A: Rule-Based Classifier (try first)

Transparent, zero training data needed, directly encodes the rule:

```python
def classify_landing_foul(features: dict) -> tuple[str, float]:
    """Rule-based landing foul classification from pose features.

    Returns (prediction, confidence) where prediction is "YES" or "NO".
    """
    if features["defender_ankle_in_zone_frac"] > 0.40:
        return ("YES", features["defender_ankle_in_zone_frac"])

    if (features["min_ankle_distance"] < 0.3
            and features["overlap_duration_frames"] >= 3):
        return ("YES", 0.7)

    if (features["defender_ankle_at_landing"] < 0.25
            and features["defender_retreat_velocity"] < 0):
        return ("YES", 0.65)

    return ("NO", 1.0 - features["defender_ankle_in_zone_frac"])
```

Thresholds tuned on training set; evaluate on val. If rule-based achieves ≥ 80% precision, iterate thresholds. If not, move to Option B.

#### Option B: Gradient-Boosted Trees (XGBoost / LightGBM)

20 features, 227 training samples — perfect regime for tree-based models:

- 5-fold stratified CV on training set for hyperparameter selection.
- Final model trained on full training set, evaluated on 57-clip val.
- Feature importance analysis reveals which geometric relationships drive predictions — directly interpretable for the paper.
- Hyperparameters: `max_depth=3`, `n_estimators=100-300`, `min_child_weight=5` (conservative to prevent overfit).

#### Option C: Temporal Sequence Model (1D-CNN or GRU)

If per-frame keypoint sequences carry more signal than aggregated features:

- Input: (T frames × K keypoints × 3 coordinates) for shooter + defender.
- Small 1D-CNN (3 layers, 32→64→128 channels) or GRU (hidden=64, 1 layer) with global pooling → 2-class output.
- Risk: 227 training clips may not support a sequence model. Use only if A and B underperform.

**CLI:**
```bash
PYTHONPATH=. python src/landing_foul_pose_classify.py --mode rules      # Option A
PYTHONPATH=. python src/landing_foul_pose_classify.py --mode xgboost    # Option B
PYTHONPATH=. python src/landing_foul_pose_classify.py --mode gru        # Option C
PYTHONPATH=. python src/landing_foul_pose_classify.py --evaluate-only   # eval saved model
```

**Makefile:**
```makefile
pose-classify:
	PYTHONPATH=. $(PYTHON) src/landing_foul_pose_classify.py --mode $(MODE)

pose-evaluate:
	PYTHONPATH=. $(PYTHON) src/landing_foul_pose_classify.py --evaluate-only
```

---

### Phase 4: Ensemble with VideoMAE (1 day)

**Goal:** Combine pose-based predictions with VideoMAE predictions for maximum performance.

If both VideoMAE and pose estimation independently approach but don't clear the gate, their failure modes are likely complementary:

| Model | Strength | Weakness |
|---|---|---|
| VideoMAE | Holistic scene understanding; captures context beyond the two involved players | Cannot resolve small body parts; struggles with precise spatial relationships |
| Pose estimation | Precise spatial relationships between keypoints; explicit geometric reasoning | Fails when keypoints are unreliable (occlusion, distant camera); misses non-geometric cues |

**Ensemble strategies (in order of complexity):**

1. **Voting:** Both must agree on YES for a YES prediction (intersection → higher precision). If pose says YES and VideoMAE says NO (or vice versa), predict NO.
2. **Stacking:** Train a logistic regression on (VideoMAE prob, pose prob, pose confidence) → final prediction. Uses val set with leave-one-out to avoid leaking.
3. **Cascade:** Run pose classifier first; if confidence > threshold, use its prediction. Otherwise, defer to VideoMAE. Leverages pose precision on clear cases and VideoMAE coverage on ambiguous ones.

**Implementation:** `landing_foul_ensemble.py` — takes saved predictions from both models, produces final predictions + confusion matrix.

---

### Phase 5: Integration and Scale-Up (1-2 days)

**Goal:** Integrate pose-based (or ensemble) classifier into the batch prediction pipeline for Steps 11-12.

1. Update `landing_foul_video_predict.py` (not yet built) to run pose extraction + classification on new clips.
2. Add pose extraction to the Colab notebook workflow (GPU accelerates pose estimation significantly).
3. Ensure the pipeline handles missing-keypoint clips gracefully (fall back to VideoMAE or flag for manual review).

---

## Timeline and Decision Points

```
Week 1:
  Day 1-2:  Phase 0 — Pose estimator validation on 10 clips
            DECISION: Which model? Go/no-go on pose approach.

  Day 3-5:  Phase 1 — Extract keypoints for all 284 clips
            Phase 2 — Feature engineering + landing zone geometry

Week 2:
  Day 6-7:  Phase 3 — Train pose classifier (rules → XGBoost → GRU)
            DECISION: Does pose alone clear the gate?

  Day 8:    Phase 4 — Ensemble with VideoMAE (if needed)
            DECISION: Does ensemble clear the gate?

  Day 9-10: Phase 5 — Integration into batch prediction pipeline
            → Proceed to Step 11 (per-official measurement)
```

**Decision tree:**

```
Phase 0: Keypoints usable on ≥ 80% of clips?
├── NO  → Spatial crop (zoom to shooting region) + retry
│         Still NO → Abort pose approach; continue VideoMAE tuning
│
└── YES → Phase 1-3: Pose classifier val metrics
          ├── P ≥ 85%, R ≥ 70% → Gate cleared. Use pose standalone.
          │                        Proceed to Step 11.
          │
          ├── P ≥ 75% or R ≥ 65% → Phase 4: Ensemble with VideoMAE
          │   ├── Ensemble clears gate → Proceed to Step 11
          │   └── Ensemble still misses → Hybrid: ensemble pre-filter + manual review
          │
          └── P < 75% and R < 65% → Keypoints too noisy for this task.
                                     Fall back to VideoMAE tuning + manual scale-up.
```

---

## Anticipated Challenges and Mitigations

| Challenge | Impact | Mitigation |
|---|---|---|
| **Distant players in wide-angle broadcast** | Low keypoint confidence on ankles/feet | Spatial crop to shooting region before pose estimation; use higher-resolution model (ViTPose-L) |
| **Occlusion during contested shots** | Defender keypoints hidden behind shooter | Use keypoints from frames before/after contact; interpolate short gaps; flag heavily-occluded clips for manual review |
| **Multiple defenders near shooter** | Ambiguous "primary defender" assignment | Track all nearby defenders; compute landing zone features for the closest; flag clips with 2+ defenders in zone |
| **Camera angle variation across arenas** | Inconsistent pixel-space geometry | Normalize all distances by estimated person height; consider homography estimation for court-plane projection (stretch goal) |
| **Shooter-initiated contact (pump-fake jump-into)** | Shooter jumps forward into stationary defender — not a landing foul | Track shooter lateral drift; if drift > threshold toward defender who is stationary, classify NO. This is the same signal the LLM couldn't resolve; pose features make it geometric and explicit. |
| **Small training set (227 clips)** | Overfitting risk for learned classifiers | Start with rule-based; use tree-based models with strong regularization; avoid deep models unless A/B fail |

---

## Relationship to Existing Pipeline

Pose estimation slots into the existing pipeline as a parallel track alongside VideoMAE fine-tuning:

```
Step 10c: VideoMAE fine-tuning (Runs 3-5)  ─┐
                                              ├──→  Step 10f: Ensemble  ──→  Step 11: Scale
Step 10e: Pose estimation (this plan)       ─┘
```

It reuses:
- **Clip anchors** (`landing_foul_clip_anchors.json`) — same temporal windows.
- **Train/val split** (`landing_foul_split.json`) — same evaluation protocol.
- **Ground truth** (`landing_foul_ground_truth.csv`) — same labels.
- **Quality gate** (P ≥ 85%, R ≥ 70% on 57-val) — same success criterion.

New scripts:
- `src/landing_foul_pose_extract.py` — Phase 1
- `src/landing_foul_pose_features.py` — Phase 2
- `src/landing_foul_pose_classify.py` — Phase 3
- `src/landing_foul_ensemble.py` — Phase 4

New data:
- `data/processed/landing_foul_poses.json` — raw keypoints (gitignored; ~50-100 MB)
- `data/processed/landing_foul_pose_features.npz` — feature matrix (small; git-tracked)

New dependencies (add to `requirements-ml.txt`):
```
mmpose>=1.3
mmdet>=3.3
mmengine>=0.10
xgboost>=2.1
# OR for lightweight fallback:
mediapipe>=0.10
ultralytics>=8.2
```

---

## Success Criteria

| Metric | Target | Stretch |
|---|---|---|
| Pose standalone precision (YES) | ≥ 75% | ≥ 85% |
| Pose standalone recall (YES) | ≥ 65% | ≥ 70% |
| Ensemble precision (YES) | **≥ 85%** | ≥ 90% |
| Ensemble recall (YES) | **≥ 70%** | ≥ 80% |
| Keypoint extraction coverage | ≥ 80% of clips | ≥ 95% |
| Feature interpretability | Top 3 features align with landing foul rule | All features have basketball-meaningful interpretation |

**The binding success criterion is the same as the overall project gate: ensemble (or standalone) precision ≥ 85% and recall ≥ 70% on the 57-clip validation set.** If pose estimation enables clearing this gate — whether standalone or in ensemble with VideoMAE — it succeeds.

---

## Paper 2 Implications

If pose estimation works, it strengthens the Paper 2 narrative:

1. **Interpretability.** "The defender's ankle was 0.3 person-heights inside the shooter's landing zone for 4 consecutive frames" is a more compelling finding than "the model predicted YES with probability 0.87." Per-official variance can be described in geometric terms.

2. **Feature importance.** XGBoost feature importance directly answers "what makes a landing foul?" — which geometric relationships most predict a YES label. This is publishable analysis beyond binary classification.

3. **Officiating style profiles.** If per-official landing foul rates (Step 12) show variance, pose features can characterize *how* officials differ: do suppressors tolerate more ankle incursion before calling a foul? Do amplifiers have a lower threshold for landing zone overlap? This is the "why behind the why" — mechanism for Paper 2's claim.
