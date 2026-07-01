"""Landing-Foul Video Grader using Multimodal LLMs (Step 10).

Binary YES/NO/UNCLEAR classification of whether a shooting foul is a *landing
foul* — defender's feet/body under or moving into the shooter's landing zone
while the shooter is airborne on a jump shot. The signal is spatial, not
temporal, so this uses a dedicated spatial prompt rather than the timing-axis
prompts in ``foul_type_llm_grader.py``.

Ground truth: ``data/landing_foul_ground_truth.csv`` (134 rows: 99 from the
Step 9 HTML classifier + 35 legacy v3 labels). Clip video URLs come from
``data/processed/landing_foul_manifest.json`` (100 clips) plus the Harden /
Giannis player manifests for the v3 legacy rows.

Recommended (Gemini native video):
    PYTHONPATH=. .venv/bin/python src/landing_foul_llm_grader.py \
        --provider gemini --model gemini-2.5-flash --validate-only

Validate against the primary set (Step 9 YES+NO, excludes UNCLEAR):
    PYTHONPATH=. .venv/bin/python src/landing_foul_llm_grader.py \
        --provider gemini --model gemini-2.5-flash --validate-only --limit 10

Extended set (include v3 legacy clips):
    PYTHONPATH=. .venv/bin/python src/landing_foul_llm_grader.py \
        --provider gemini --model gemini-2.5-flash --validate-only --extended

Target: precision >= 85% on YES, recall >= 70% on YES.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from landing_foul_video_dataset import clip_path
from nba_client import NBAStatsClient
from foul_type_llm_grader import (
    GeminiGrader,
    OpenAIGrader,
    AnthropicGrader,
    VertexGeminiGrader,
    extract_frames_with_ffmpeg,
    encode_image_base64,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# OBSERVE_LANDING_PROMPT — structured observation-only prompt. No
# classification ask. The model describes what it sees across six binary
# feature dimensions, then produces a free-text narrative. The classification
# is derived post-hoc from the feature vector, not by the model itself.
#
# Design rationale: Phase 0 analysis of the spatial V1 run showed that the
# model's classification is 100% YES-biased on `defender_in_landing_zone` and
# `contact_vs_descent` — the structured fields carry zero discriminative
# information because the classification prompt collapses the model's
# perception into a single template narrative. The observation prompt forces
# the model to answer specific, narrow questions about what it sees without
# the pressure to classify. Each feature is independently correlatable
# against ground truth post-hoc.
OBSERVE_LANDING_PROMPT = """You are watching a short video clip of a basketball shooting foul. Your job is to DESCRIBE what you see — NOT to classify or judge the play. You are a camera, not a referee.

Answer each question based ONLY on what you can directly observe in the video. If you cannot tell, say UNCLEAR. Do not infer or guess.

QUESTIONS:

1. SHOT_TYPE: What kind of shot is the fouled player attempting?
   - JUMP_SHOT: Player leaves the ground on a perimeter/wing 3-point attempt
   - DRIVE: Player is driving to the rim or attempting a layup/floater
   - STEP_BACK: Player steps back and rises for a shot
   - PULL_UP: Player pulls up off the dribble for a shot
   - OTHER: None of the above
   - UNCLEAR: Cannot determine

2. SHOOTER_AIRBORNE: Is the shooter in the air (both feet off the floor) at the moment of contact?
   - YES / NO / UNCLEAR

3. WHO_INITIATED: Who caused the contact that was called?
   - DEFENDER: The defender moved into the shooter's space (closeout, step-in, undercut)
   - SHOOTER: The shooter moved into the defender's space (pump-fake then jump into, lean-in, rip-through)
   - MUTUAL: Both players moved toward each other simultaneously
   - UNCLEAR: Cannot determine

4. PRIMARY_CONTACT_BODY_PART: Where on the SHOOTER'S body is the main contact that caused the whistle?
   - FEET_OR_LOWER_LEGS: Contact on the shooter's feet, ankles, or lower legs (below the knee)
   - UPPER_LEGS_OR_HIPS: Contact on the shooter's thighs, hips, or pelvis
   - TORSO: Contact on the shooter's chest, back, or midsection
   - ARM_OR_HAND: Contact on the shooter's arm, wrist, or hand
   - HEAD: Contact on the shooter's head or face
   - MULTIPLE: Contact across multiple body areas simultaneously
   - UNCLEAR: Cannot determine

5. DEFENDER_FEET_AT_SHOOTER_LANDING: When the shooter comes down to land, where are the defender's feet relative to the shooter's landing spot?
   - DIRECTLY_UNDER: The defender's feet are underneath or inside the spot where the shooter needs to land
   - NEAR_BUT_NOT_UNDER: The defender is nearby but their feet are to the side or just outside the landing zone
   - AWAY_FROM_LANDING: The defender's feet are clearly not near the shooter's landing area
   - UNCLEAR: Cannot determine

6. CONTACT_TIMING: When does the called contact occur relative to the shot?
   - BEFORE_RELEASE: While the ball is still in the shooter's hands
   - AT_RELEASE: As the ball is leaving the shooter's hands
   - AFTER_RELEASE_DESCENDING: After the ball is released, while the shooter is still in the air descending
   - AFTER_LANDING: After the shooter has already come down to the floor
   - UNCLEAR: Cannot determine

Finally, write a 2-3 sentence NARRATIVE describing what you see in the clip chronologically — who does what, in what order, and what contact occurs. Be specific about body parts and directions. Do NOT use the phrase "landing foul" or make any judgment about the call.

Return a raw JSON object (no markdown):
{
  "shot_type": "...",
  "shooter_airborne": "...",
  "who_initiated": "...",
  "primary_contact_body_part": "...",
  "defender_feet_at_shooter_landing": "...",
  "contact_timing": "...",
  "narrative": "..."
}"""


# DESCRIBE_LANDING_PROMPT — Layer 1 only. No classification. The model
# describes contact timing relative to the jump apex and lists contacts in
# order. Layer 2 (`classify_from_description`) applies landing-foul rules.
DESCRIBE_LANDING_PROMPT = """You are watching a short video clip of a basketball shooting foul. Describe what you see — do NOT classify whether it is a landing foul or judge the call.

Watch the full clip. Focus on the shooter on a jump shot: identify the apex of their jump (highest point), when the ball is released, and when they land.

For EACH distinct contact between the defender and the shooter (in chronological order), report:
  - When it occurs relative to the jump: ASCENT (still rising, before apex), AT_APEX, DESCENT (falling after apex, before/at landing), or ON_GROUND
  - Which DEFENDER body part makes contact (HAND, ARM, TORSO, HIP, LEG, FOOT, etc.)
  - Which SHOOTER body part is contacted
  - One short phrase describing the contact

Then identify which contact index (0-based) appears to be the foul that was called — the primary penalized contact. If unclear, use the last significant contact before the whistle or landing.

Also report shot_type: JUMP_SHOT, DRIVE, STEP_BACK, PULL_UP, OTHER, or UNCLEAR.

Do NOT use the phrase "landing foul". Do NOT output YES/NO.

Return a raw JSON object (no markdown):
{
  "shot_type": "JUMP_SHOT | DRIVE | STEP_BACK | PULL_UP | OTHER | UNCLEAR",
  "narrative": "2-4 sentence chronological description of the play",
  "contacts": [
    {
      "order": 1,
      "shooter_motion": "ASCENT | AT_APEX | DESCENT | ON_GROUND",
      "defender_body_part": "HAND | ARM | TORSO | HIP | LEG | FOOT | ...",
      "shooter_body_part": "...",
      "description": "brief phrase"
    }
  ],
  "primary_foul_contact_index": 0
}"""


# ---------------------------------------------------------------------------

# SPATIAL_LANDING_PROMPT — the primary prompt. Asks for the spatial
# observations that determine whether a foul is a landing foul, plus a direct
# YES/NO/UNCLEAR classification. The model's `landing_foul` field is the
# prediction; the observations anchor its reasoning and aid mismatch review.
SPATIAL_LANDING_PROMPT = """You are an expert NBA officiating analyst. Watch a short video clip of a shooting foul and determine whether it is a LANDING FOUL.

DEFINITION (landing foul): A foul where the defender's feet or body are under or moving into the shooter's landing zone WHILE THE SHOOTER IS AIRBORNE on a jump shot, and the foul is called because of that positioning / the contact that occurs as the shooter comes down. The shooter must be in the air on a jump shot (typically a perimeter/wing 3-point attempt) and the illegal contact is tied to the landing space — the defender undercut or stepped under the airborne shooter.

NOT a landing foul (answer NO):
  - Standard arm/hand contest on the shot (defender swipes or reaches the arm while the shooter is going up or at the release, feet in legal position).
  - Arm/hand contact on the shooter's FOLLOW-THROUGH AFTER the release (the defender's arm comes down on the shooter after the ball is gone). Post-release contact is NOT automatically a landing foul — it must be the defender's feet/body under the shooter, not an arm.
  - The SHOOTER INITIATES the contact (see Step 2 — the pump-fake jump-into is the most common trap here).
  - Contact on a drive to the rim or layup (shooter not in an airborne jump shot, or contact is body-to-body on the drive).
  - Off-ball or screen contact unrelated to the shooter's landing zone.
  - Reach-in or body foul before the shooter leaves the ground.

KEY SPATIAL CHECK (watch the clip twice):
  1. Is the shooter airborne on a JUMP SHOT (both feet off the floor, rising/falling on a perimeter shot)? If it is a drive/layup, the answer is almost certainly NO.
  2. WHO INITIATED THE CONTACT? Watch the closeout and the takeoff carefully.
       - PUMP-FAKE JUMP-INTO: the shooter pump-fakes the defender into the air (or otherwise gets the defender to leave his feet / commit), THEN the shooter jumps and launches the shot INTO the airborne/committed defender. The defender's body ends up under the shooter at landing ONLY because the shooter jumped into him — the shooter created the contact. This is NOT a landing foul; answer NO even though the defender appears to be under the shooter at landing.
       - SHOOTER LEANS/JUMPS INTO A STATIONARY DEFENDER, or a rip-through: also shooter-initiated → NO.
       - DEFENDER STEPS/UNDERCUTS: the defender closes out and moves his feet/body into the shooter's landing zone independent of the shooter's motion → the defender initiated → continue to Step 3.
  3. Where are the defender's FEET/BODY as the shooter DESCENDS to land? Are they under the shooter's landing spot (undercut / stepped into the landing zone), or is the defender to the side / in legal position? "Near the shooter at landing" is NOT enough — the defender must actually be under / in the landing zone for it to be a landing foul; a normal closeout that ends up near the shooter is a contest, not an undercut.
  4. WHAT is the called contact, and roughly when? Is it the defender's feet/body under the shooter during the descent/landing (landing foul), or an arm/hand contest on the shot release or follow-through (not a landing foul)?

IMPORTANT — do not let timing alone decide: contact that occurs at or around the release can still be a landing foul if the defender is undercutting the shooter; contact after the release can still be NO if it is an arm on the follow-through. Judge by WHAT the contact is and WHO initiated it, not only by WHEN it occurs. A landing foul requires ALL THREE: airborne jump shot + DEFENDER-initiated contact + the defender's feet/body under the shooter's landing zone. If the shooter jumped into the defender, answer NO. If the contact is a standard arm contest / follow-through with the defender's feet legal, answer NO.

Answer these observation questions, then the classification:

{
  "shot_type": "JUMP_SHOT" | "DRIVE" | "OTHER" | "UNCLEAR",
  "who_initiated_contact": "SHOOTER_JUMPED_INTO_DEFENDER" | "DEFENDER_STEPPED_UNDER_SHOOTER" | "MUTUAL_OR_INCIDENTAL" | "UNCLEAR",
  "defender_position_at_landing": "UNDER_SHOOTER" | "NEAR_BUT_LEGAL" | "NOT_AT_LANDING" | "UNCLEAR",
  "contact_moment": "DURING_DESCENT_OR_LANDING" | "DURING_SHOT_MOTION" | "BEFORE_SHOT" | "UNCLEAR",
  "landing_foul": "YES" | "NO" | "UNCLEAR",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "reasoning": "One sentence citing who initiated contact, defender feet position at landing, and what the called contact was."
}"""


# SEQUENCE_LANDING_PROMPT — event-ordering fallback (per HANDOFF): if the
# freeze-frame spatial prompt underperforms, frame the decision as the temporal
# ordering of DEFENDER_CLOSEOUT -> SHOOTER_DESCENDING -> CONTACT.
SEQUENCE_LANDING_PROMPT = """You are an expert NBA officiating analyst. Watch a short video clip of a shooting foul and determine whether it is a LANDING FOUL.

A landing foul = the shooter is airborne on a jump shot and the defender undercuts or steps into the shooter's landing zone, with the illegal contact occurring as the shooter descends/lands. A standard arm/hand contest on the shot release with the defender's feet in legal position is NOT a landing foul.

STEP 1 — NARRATIVE: Write 2-3 sentences describing the clip chronologically. Focus on the order of: when does the shooter leave the ground and release the ball? When does the defender close out / where are the defender's feet? When does contact occur relative to the shooter coming down?

STEP 2 — EVENT ORDERING: Report which of these occurred, and the order of contact relative to the shooter's descent:
  DEFENDER_CLOSEOUT: The defender closes out and his feet end up under / moving into the shooter's landing zone.
  SHOOTER_RELEASE: The ball leaves the shooter's hand on a jump shot.
  SHOOTER_DESCENDING: The airborne shooter begins falling toward his landing spot.
  CONTACT: The first illegal contact.

STEP 3 — CLASSIFY: Is this a landing foul?
  YES: Airborne jump shot + defender in the landing zone + contact during the shooter's descent/landing.
  NO: Standard arm/hand contest on the shot, a drive/layup foul, shooter-initiated contact, or any contact not tied to the landing zone.
  UNCLEAR: Cannot tell from the video.

Return a JSON object:
{
  "narrative": "2-3 sentence chronological description.",
  "defender_in_landing_zone": "YES" | "NO" | "UNCLEAR",
  "contact_vs_descent": "BEFORE_DESCENT" | "DURING_DESCENT_OR_LANDING" | "DURING_SHOT_MOTION" | "UNCLEAR",
  "landing_foul": "YES" | "NO" | "UNCLEAR",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "reasoning": "One sentence citing the event ordering that determined the call."
}"""


# WHISTLE_LANDING_PROMPT — attribution-based. The clip's audio contains the
# referee's whistle. The whistle is a near-instantaneous proxy for the contact
# the referee judged to be the foul, so it both *selects* which contact to
# evaluate and *times* it. A play often has several contacts (an incidental
# bump on the closeout, a hand on the hip, the shooter landing on the
# defender's foot); only the contact coincident with the whistle is the called
# foul — all other contact must be ignored, no matter how visually prominent.
# This is the prompt that targets the observed failure mode: the spatial prompt
# mis-attributes the foul to whatever contact it notices and over-calls YES.
# Best with native-video providers (Gemini / Vertex) that pass audio through.
WHISTLE_LANDING_PROMPT = """You are an expert NBA officiating analyst. Watch a short video clip of a shooting foul — INCLUDING THE AUDIO — and determine whether it is a LANDING FOUL.

DEFINITION (landing foul): A foul where the defender's feet or body are under or moving into the shooter's landing zone WHILE THE SHOOTER IS AIRBORNE on a jump shot, and the foul is called because of that contact as the shooter comes down. The shooter must be in the air on a jump shot (typically a perimeter/wing 3-point attempt) and the illegal contact is the defender undercutting or stepping under the airborne shooter.

NOT a landing foul (answer NO):
  - Standard arm/hand contest on the shot (defender swipes or reaches the arm while the shooter is going up or at the release, feet in legal position).
  - Arm/hand contact on the shooter's FOLLOW-THROUGH AFTER the release (the defender's arm comes down on the shooter after the ball is gone). Contact after the release is NOT automatically a landing foul — it must be the defender's feet/body under the shooter, not an arm.
  - Contact on a drive to the rim or layup (shooter not in an airborne jump shot).
  - The shooter initiates contact (jumping into / leaning into a stationary defender, rip-through, pump-fake jump-into).
  - Off-ball or screen contact unrelated to the shooter's landing zone.
  - Reach-in or body foul before the shooter leaves the ground.

METHOD — three steps. Use the audio.

STEP 1 — JUMP SHOT GATE: Is the shooter airborne on a JUMP SHOT (both feet off the floor on a perimeter shot)? If it is a drive/layup, answer NO.

STEP 2 — FIND THE CALLED CONTACT VIA THE WHISTLE: Listen for the referee's whistle. The contact that coincides with the whistle is the contact the referee judged to be the foul — that is the ONLY contact you should evaluate. If the whistle is late, the called contact is the one immediately before the whistle. IGNORE every other contact in the clip (an early bump, a hand on the hip, a brush on the closeout), no matter how visible — if the referee did not whistle it, it is not the foul. Note when the whistle blows relative to the shot: at/before the release (shooter still rising / at the release point) or during the descent / at landing.

STEP 3 — CLASSIFY THE CALLED CONTACT:
  - If the called contact occurs AT OR BEFORE THE RELEASE (contact on the upward shot motion) -> NO (standard arm contest on the shot).
  - If the called contact occurs DURING DESCENT / AT LANDING:
      * If the contact is the defender's FEET/BODY UNDER THE SHOOTER (undercut / stepped into the landing zone) -> YES.
      * If the contact is the defender's ARM/HAND on the shooter (e.g. coming down on the follow-through) with the defender's feet in legal position -> NO.

YES requires ALL THREE: airborne jump shot + called contact during descent/landing + the called contact is the defender's feet/body under the shooter. Post-release arm contact is NO. Pre-release arm contact is NO. Any contact the referee did NOT whistle is irrelevant to the decision.

Answer these observation questions, then the classification:

{
  "shot_type": "JUMP_SHOT" | "DRIVE" | "OTHER" | "UNCLEAR",
  "whistle_timing": "AT_OR_BEFORE_RELEASE" | "DURING_DESCENT_OR_LANDING" | "NO_WHISTLE_HEARD" | "UNCLEAR",
  "called_contact": "DEFENDER_FEET_BODY_UNDER_SHOOTER" | "ARM_HAND_ON_SHOOTER" | "OTHER_CONTACT" | "NO_CLEAR_CONTACT_AT_WHISTLE" | "UNCLEAR",
  "landing_foul": "YES" | "NO" | "UNCLEAR",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "reasoning": "One sentence citing the whistle timing and what the called contact was."
}"""


# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

@dataclass
class LandingFewShotExample:
    game_id: str
    event_id: int
    description: str
    label: str  # YES | NO
    video_url: str
    note: str = ""


def select_landing_few_shot(
    clips_by_key: Dict[Tuple[str, int], Dict[str, Any]],
    ground_truth: Dict[Tuple[str, int], str],
    n_per_class: int = 2,
) -> List[LandingFewShotExample]:
    """Select balanced YES/NO few-shot examples from the ground truth.

    Prefers clips that carry a human note (borderline/illustrative) so the
    model sees why a YES or NO was assigned.
    """
    examples: List[LandingFewShotExample] = []
    by_class: Dict[str, List[LandingFewShotExample]] = {"YES": [], "NO": []}

    for key, label in ground_truth.items():
        label = label.strip().upper()
        if label not in ("YES", "NO"):
            continue
        clip = clips_by_key.get(key)
        if not clip:
            continue
        video_url = clip.get("video_url_720") or clip.get("video_url_960")
        if not video_url:
            continue
        by_class[label].append(LandingFewShotExample(
            game_id=key[0],
            event_id=key[1],
            description=clip.get("description", ""),
            label=label,
            video_url=video_url,
            note=str(clip.get("note") or ""),
        ))

    # Prefer noted examples first, then fill.
    for label in ("YES", "NO"):
        pool = sorted(by_class[label], key=lambda e: (e.note == "", e.game_id))
        examples.extend(pool[:n_per_class])
    return examples


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

GT_PATH = config.DATA_DIR / "landing_foul_ground_truth.csv"
LANDING_MANIFEST_PATH = config.PROCESSED_DIR / "landing_foul_manifest.json"
SPLIT_PATH = config.PROCESSED_DIR / "landing_foul_split.json"

# Layer-2 rule helpers (describe → classify)
_ARM_PARTS = frozenset({
    "HAND", "ARM", "ARMS", "WRIST", "FOREARM", "FINGERS", "FINGER",
    "ARM_OR_HAND", "HAND_OR_ARM",
})
_ASCENT_PHASES = frozenset({
    "ASCENT", "RISING", "GOING_UP", "BEFORE_APEX", "BEFORE_RELEASE",
    "DURING_SHOT_MOTION", "AT_RELEASE",
})
_DESCENT_PHASES = frozenset({
    "DESCENT", "FALLING", "DESCENDING", "LANDING", "AFTER_APEX",
    "AFTER_RELEASE", "AFTER_RELEASE_DESCENDING", "AT_LANDING",
})
_DRIVE_SHOTS = frozenset({"DRIVE", "LAYUP", "FLOATER"})


def _norm_token(s: Any) -> str:
    return str(s or "").strip().upper().replace("-", "_").replace(" ", "_")


def _is_arm_contact(part: str) -> bool:
    p = _norm_token(part)
    if not p or p == "UNCLEAR":
        return False
    if p in _ARM_PARTS:
        return True
    return any(p.startswith(x) or x in p for x in ("HAND", "ARM", "WRIST", "FOREARM"))


def _primary_contact(obs: Dict[str, Any]) -> Dict[str, Any]:
    contacts = obs.get("contacts") or []
    if not isinstance(contacts, list) or not contacts:
        return {}
    idx = obs.get("primary_foul_contact_index", 0)
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        idx = 0
    if idx < 0 or idx >= len(contacts):
        idx = len(contacts) - 1
    c = contacts[idx]
    return c if isinstance(c, dict) else {}


def classify_from_description(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Layer 2: derive landing_foul YES/NO/UNCLEAR from describe-mode observations.

    Rules (from domain):
      - Pump-fake jump-into and arm swipes occur on ASCENT → NO
      - Landing foul requires contact during DESCENT and NOT hand/arm contact
    """
    shot = _norm_token(obs.get("shot_type"))
    primary = _primary_contact(obs)
    motion = _norm_token(primary.get("shooter_motion"))
    def_part = _norm_token(primary.get("defender_body_part"))

    reasons: List[str] = []

    if shot in _DRIVE_SHOTS:
        return {
            "landing_foul": "NO",
            "confidence": "HIGH",
            "reasoning": "Layer 2: drive/layup — not an airborne jump-shot landing foul.",
            "layer2_rule": "drive",
        }

    if not primary:
        return {
            "landing_foul": "UNCLEAR",
            "confidence": "LOW",
            "reasoning": "Layer 2: no contacts described.",
            "layer2_rule": "no_contacts",
        }

    if motion in _ASCENT_PHASES or motion == "AT_APEX":
        reasons.append(f"primary contact during {motion} (ascending/apex)")
        return {
            "landing_foul": "NO",
            "confidence": "HIGH",
            "reasoning": "Layer 2: " + "; ".join(reasons) + " — not a descent landing contact.",
            "layer2_rule": "ascent_contact",
        }

    if _is_arm_contact(def_part):
        reasons.append(f"defender contact via {def_part}")
        return {
            "landing_foul": "NO",
            "confidence": "HIGH",
            "reasoning": "Layer 2: " + "; ".join(reasons) + " — arm/hand contact cannot be a landing foul.",
            "layer2_rule": "arm_contact",
        }

    if motion in _DESCENT_PHASES:
        reasons.append(f"primary contact during {motion} via defender {def_part}")
        return {
            "landing_foul": "YES",
            "confidence": "MEDIUM" if motion == "ON_GROUND" else "HIGH",
            "reasoning": "Layer 2: " + "; ".join(reasons) + " — body contact on descent.",
            "layer2_rule": "descent_body_contact",
        }

    if motion == "ON_GROUND":
        return {
            "landing_foul": "NO",
            "confidence": "MEDIUM",
            "reasoning": "Layer 2: primary contact on ground, not during descent.",
            "layer2_rule": "on_ground",
        }

    return {
        "landing_foul": "UNCLEAR",
        "confidence": "LOW",
        "reasoning": f"Layer 2: could not classify (shot={shot}, motion={motion}, def_part={def_part}).",
        "layer2_rule": "unclear",
    }


def load_val_split_keys() -> List[Tuple[str, int]]:
    """Return (game_id, event_id) keys from the fixed video train/val split."""
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Split not found: {SPLIT_PATH}")
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    return [
        (str(x["game_id"]).zfill(10), int(x["event_id"]))
        for x in split["val"]["keys"]
    ]


def load_landing_ground_truth() -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Load merged landing foul ground truth -> (game_id, event_id) -> row dict."""
    if not GT_PATH.exists():
        return {}
    df = pd.read_csv(GT_PATH)
    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    df["event_id"] = df["event_id"].astype(int)
    gt: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for _, row in df.iterrows():
        gt[(row["game_id"], int(row["event_id"]))] = row.to_dict()
    return gt


def _clip_video_url(clip: Dict[str, Any]) -> Optional[str]:
    return clip.get("video_url_720") or clip.get("video_url_960") or clip.get("video_url_320")


def load_clips_by_key(extended: bool = False) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Build a (game_id, event_id) -> clip lookup.

    Always loads the landing manifest (100 Step 9 clips). When ``extended`` is
    True (or always, since they are small), also loads the Harden / Giannis
    player manifests so v3 legacy ground-truth rows resolve to a video URL.
    """
    clips_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}

    if LANDING_MANIFEST_PATH.exists():
        with open(LANDING_MANIFEST_PATH) as f:
            manifest = json.load(f)
        for clip in manifest.get("clips", []):
            key = (str(clip["game_id"]).zfill(10), int(clip["event_id"]))
            clips_by_key[key] = clip

    # Player manifests carry the v3 legacy clips' video URLs.
    for slug_file in sorted(config.PROCESSED_DIR.glob("foul_type_manifest_*.json")):
        with open(slug_file) as f:
            manifest = json.load(f)
        for clip in manifest.get("clips", []):
            key = (str(clip["game_id"]).zfill(10), int(clip["event_id"]))
            # Don't overwrite landing-manifest entries (they carry extra fields).
            clips_by_key.setdefault(key, clip)

    return clips_by_key


# ---------------------------------------------------------------------------
# Grader subclasses (reuse upload / frame infrastructure from the timing grader)
# ---------------------------------------------------------------------------

class _LandingMixin:
    """Shared landing-specific behavior mixed into each provider subclass."""

    prompt_mode: str  # "spatial" | "sequence" | "whistle" | "observe" | "describe"

    def _landing_system_prompt(self) -> str:
        if self.prompt_mode == "sequence":
            return SEQUENCE_LANDING_PROMPT
        if self.prompt_mode == "whistle":
            return WHISTLE_LANDING_PROMPT
        if self.prompt_mode == "observe":
            return OBSERVE_LANDING_PROMPT
        if self.prompt_mode == "describe":
            return DESCRIBE_LANDING_PROMPT
        return SPATIAL_LANDING_PROMPT

    def _normalize_landing(self, grade: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize the model's landing response.

        For "describe" mode, Layer 1 observations are classified by
        ``classify_from_description`` (Layer 2 rules). For "observe" mode,
        derives classification from the feature vector rather than trusting
        a model-produced label. For other modes, trusts the model's
        ``landing_foul`` field; derives it from observations only if missing.
        """
        if self.prompt_mode == "describe":
            layer2 = classify_from_description(grade)
            grade["landing_foul"] = layer2["landing_foul"]
            grade["confidence"] = layer2["confidence"]
            grade["reasoning"] = layer2["reasoning"]
            grade["layer2_rule"] = layer2.get("layer2_rule", "")
            primary = _primary_contact(grade)
            if primary:
                grade["primary_shooter_motion"] = primary.get("shooter_motion", "")
                grade["primary_defender_body_part"] = primary.get("defender_body_part", "")
            return grade

        if self.prompt_mode == "observe":
            label = "UNCLEAR"
            shot = str(grade.get("shot_type", "")).strip().upper()
            airborne = str(grade.get("shooter_airborne", "")).strip().upper()
            who = str(grade.get("who_initiated", "")).strip().upper()
            body = str(grade.get("primary_contact_body_part", "")).strip().upper()
            feet = str(grade.get("defender_feet_at_shooter_landing", "")).strip().upper()
            timing = str(grade.get("contact_timing", "")).strip().upper()

            # Decision rule derived from Phase 0 FP analysis:
            # YES requires: airborne jump shot + defender-initiated + feet under
            #   + lower-body contact + contact after release
            # NO if: shooter-initiated, arm/hand contact, defender not under,
            #   or contact before/at release
            if shot in ("DRIVE",) or airborne == "NO":
                label = "NO"
            elif who == "SHOOTER":
                label = "NO"
            elif body == "ARM_OR_HAND":
                label = "NO"
            elif feet == "DIRECTLY_UNDER" and who == "DEFENDER" and airborne == "YES":
                label = "YES"
            elif feet in ("NEAR_BUT_NOT_UNDER", "AWAY_FROM_LANDING"):
                label = "NO"
            elif timing in ("BEFORE_RELEASE", "AT_RELEASE") and body not in ("FEET_OR_LOWER_LEGS",):
                label = "NO"
            # Default for observe: if enough signals align, classify, else UNCLEAR
            grade["landing_foul"] = label
            grade["confidence"] = "MEDIUM" if label != "UNCLEAR" else "LOW"
            grade["reasoning"] = (
                f"Derived from observe features: shot={shot} airborne={airborne} "
                f"who={who} body={body} feet={feet} timing={timing}"
            )
            grade["obs_shot_type"] = shot
            grade["obs_shooter_airborne"] = airborne
            grade["obs_who_initiated"] = who
            grade["obs_primary_contact_body_part"] = body
            grade["obs_defender_feet"] = feet
            grade["obs_contact_timing"] = timing
            # Populate legacy fields for validation comparison
            if feet == "DIRECTLY_UNDER":
                grade["defender_in_landing_zone"] = "YES"
            elif feet in ("NEAR_BUT_NOT_UNDER", "AWAY_FROM_LANDING"):
                grade["defender_in_landing_zone"] = "NO"
            else:
                grade["defender_in_landing_zone"] = "UNCLEAR"
            if timing == "AFTER_RELEASE_DESCENDING":
                grade["contact_vs_descent"] = "DURING_DESCENT_OR_LANDING"
            elif timing in ("BEFORE_RELEASE", "AT_RELEASE"):
                grade["contact_vs_descent"] = "DURING_SHOT_MOTION"
            else:
                grade["contact_vs_descent"] = "UNCLEAR"
            return grade

        label = str(grade.get("landing_foul", "")).strip().upper()
        if label not in ("YES", "NO", "UNCLEAR"):
            shot = str(grade.get("shot_type", "")).upper()
            pos = str(grade.get("defender_position_at_landing", "")).upper()
            moment = str(grade.get("contact_moment", "")).upper()
            desc = str(grade.get("contact_vs_descent", "")).upper()
            in_zone = str(grade.get("defender_in_landing_zone", "")).upper()
            whistle_timing = str(grade.get("whistle_timing", "")).upper()
            called_contact = str(grade.get("called_contact", "")).upper()
            who_initiated = str(grade.get("who_initiated_contact", "")).upper()

            # Shooter-initiated contact (pump-fake jump-into, lean-in,
            # rip-through) is a hard NO regardless of where the defender ends
            # up at landing — the defender being under the shooter at landing
            # does not make it a landing foul if the shooter jumped into him.
            if who_initiated == "SHOOTER_JUMPED_INTO_DEFENDER":
                label = "NO"
            # Whistle-mode derivation: the called contact + its timing decide.
            elif whistle_timing == "AT_OR_BEFORE_RELEASE":
                label = "NO"
            elif called_contact == "ARM_HAND_ON_SHOOTER":
                label = "NO"
            elif (called_contact == "DEFENDER_FEET_BODY_UNDER_SHOOTER"
                  and whistle_timing == "DURING_DESCENT_OR_LANDING"):
                label = "YES"
            elif shot == "DRIVE" or shot == "OTHER":
                label = "NO"
            # Fall back to the spatial/sequence observation fields.
            elif pos == "UNDER_SHOOTER" and (
                moment in ("DURING_DESCENT_OR_LANDING",)
                or desc in ("DURING_DESCENT_OR_LANDING",)
            ):
                label = "YES"
            elif in_zone == "YES" and (
                moment in ("DURING_DESCENT_OR_LANDING",)
                or desc in ("DURING_DESCENT_OR_LANDING",)
            ):
                label = "YES"
            elif pos in ("NEAR_BUT_LEGAL", "NOT_AT_LANDING") or in_zone == "NO":
                label = "NO"
            else:
                label = "UNCLEAR"
        grade["landing_foul"] = label
        grade["confidence"] = str(grade.get("confidence", "LOW")).strip().upper()
        return grade

    def _few_shot_label_text(self, ex: LandingFewShotExample, idx: int) -> str:
        note = f" (note: {ex.note})" if ex.note else ""
        return f"Example {idx + 1}: {ex.description}\nCorrect label: {ex.label}{note}\n"

    def _instruction_text(self, description: str) -> str:
        if self.prompt_mode == "whistle":
            return (
                f"\n\nPlay-by-play description: {description}\n\n"
                "Watch the clip above (audio is included — use the whistle). "
                "First locate the referee's whistle and identify the contact "
                "that coincides with it; ignore every other contact in the "
                "clip. Then classify THAT contact. Return a raw JSON object."
            )
        if self.prompt_mode == "sequence":
            return (
                f"\n\nPlay-by-play description: {description}\n\n"
                "Watch the clip above. Determine whether this is a landing foul "
                "using the event-ordering steps. Return a raw JSON object."
            )
        if self.prompt_mode == "observe":
            return (
                f"\n\nPlay-by-play description: {description}\n\n"
                "Watch the clip above carefully. Answer the six observation "
                "questions based on what you see. Do NOT classify the play. "
                "Do NOT say whether it is a landing foul. Just describe what "
                "happened. Return a raw JSON object."
            )
        if self.prompt_mode == "describe":
            return (
                f"\n\nPlay-by-play description: {description}\n\n"
                "Watch the full clip above. Describe contacts in chronological "
                "order relative to the shooter's jump apex. Do NOT classify "
                "whether this is a landing foul. Return a raw JSON object."
            )
        return (
            f"\n\nPlay-by-play description: {description}\n\n"
            "Watch the clip above. Apply the spatial check: shot type, WHO "
            "initiated the contact (watch for a pump-fake jump-into by the "
            "shooter), defender feet at landing, and what the called contact "
            "was. Return a raw JSON object."
        )

    def _frame_instruction_text(self, description: str) -> str:
        """Instruction for frame-based providers (OpenAI / Anthropic).

        These providers receive still frames only — no audio — so the whistle
        is not directly audible. The model should instead identify the called
        contact from the referee's visible signal and the most clearly
        penalized contact, then apply the same attribution logic.
        """
        if self.prompt_mode == "whistle":
            return (
                f"Play-by-play description: {description}\n\n"
                "Analyze the chronological frames below. Audio is not available, "
                "so identify the CALLED contact from the referee's visible signal "
                "and the contact that appears to be penalized; ignore other "
                "contact. Note when that called contact occurs relative to the "
                "release vs. the descent, and what body part made it. Return a "
                "raw JSON object."
            )
        return (
            f"Play-by-play description: {description}\n\n"
            "Analyze the chronological sequence of frames below and determine "
            "whether this is a landing foul using the spatial check."
        )


class GeminiLandingGrader(_LandingMixin, GeminiGrader):
    """Gemini (native video upload) landing foul grader."""

    def __init__(self, api_key: str, model_name: str, prompt_mode: str = "spatial",
                 few_shot_examples: Optional[List[LandingFewShotExample]] = None):
        super().__init__(api_key=api_key, model_name=model_name,
                         prompt_mode=prompt_mode, few_shot_examples=few_shot_examples or [])

    def grade_clip(self, video_path: str, description: str) -> Dict[str, Any]:
        target_upload = self._upload_file(video_path)
        if not target_upload:
            return {"landing_foul": "UNCLEAR", "confidence": "LOW",
                    "reasoning": "Failed to upload target video to Gemini Files API."}
        target_uri, target_name = target_upload

        few_shot_uploads: List[Tuple[str, str]] = []
        for i, ex in enumerate(self.few_shot_examples):
            try:
                fs_local = os.path.join(tempfile.mkdtemp(), f"fewshot_{i}.mp4")
                fs_session = NBAStatsClient().session
                fs_resp = fs_session.get(ex.video_url, timeout=30)
                if fs_resp.status_code != 200:
                    logger.warning("Failed downloading few-shot video %s", ex.video_url)
                    continue
                with open(fs_local, "wb") as f:
                    f.write(fs_resp.content)
                fs_upload = self._upload_file(fs_local)
                os.remove(fs_local)
                if fs_upload:
                    few_shot_uploads.append(fs_upload)
            except Exception as exc:
                logger.warning("Failed processing few-shot example %d: %s", i, exc)

        parts: List[Dict[str, Any]] = []
        for i, (fs_uri, _) in enumerate(few_shot_uploads):
            parts.append({"file_data": {"mime_type": "video/mp4", "file_uri": fs_uri}})
            parts.append({"text": self._few_shot_label_text(self.few_shot_examples[i], i)})
        parts.append({"file_data": {"mime_type": "video/mp4", "file_uri": target_uri}})
        parts.append({"text": self._landing_system_prompt() + self._instruction_text(description)})

        generate_url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                        f"{self.model_name}:generateContent?key={self.api_key}")
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.0,
            },
        }
        resp_gen = requests.post(generate_url, headers={"Content-Type": "application/json"}, json=payload)

        requests.delete(f"https://generativelanguage.googleapis.com/v1beta/{target_name}?key={self.api_key}")
        for _, fs_name in few_shot_uploads:
            requests.delete(f"https://generativelanguage.googleapis.com/v1beta/{fs_name}?key={self.api_key}")

        if resp_gen.status_code != 200:
            return {"landing_foul": "UNCLEAR", "confidence": "LOW",
                    "reasoning": f"Gemini Generation Error: {resp_gen.text}"}
        try:
            res_text = resp_gen.json()["candidates"][0]["content"]["parts"][0]["text"]
            return self._normalize_landing(self._parse_json_response(res_text))
        except Exception as exc:
            return {"landing_foul": "UNCLEAR", "confidence": "LOW",
                    "reasoning": f"Failed parsing Gemini candidate: {exc}"}


class VertexLandingGrader(_LandingMixin, VertexGeminiGrader):
    """Vertex AI Gemini (gcloud ADC, GCS upload) landing foul grader."""

    def __init__(self, model_name: str, prompt_mode: str = "spatial",
                 few_shot_examples: Optional[List[LandingFewShotExample]] = None):
        super().__init__(model_name=model_name, prompt_mode=prompt_mode,
                         few_shot_examples=few_shot_examples or [])

    def grade_clip(self, video_path: str, description: str) -> Dict[str, Any]:
        token = self._access_token()
        object_name = f"grader_tmp/{os.path.basename(video_path)}"
        try:
            gcs_uri = self._upload_to_gcs(video_path, object_name)
        except Exception as exc:
            return {"landing_foul": "UNCLEAR", "confidence": "LOW",
                    "reasoning": f"GCS upload error: {exc}"}

        parts: List[Dict[str, Any]] = []
        for i, ex in enumerate(self.few_shot_examples):
            try:
                fs_local = os.path.join(tempfile.mkdtemp(), f"fewshot_{i}.mp4")
                fs_session = NBAStatsClient().session
                fs_resp = fs_session.get(ex.video_url, timeout=30)
                if fs_resp.status_code != 200:
                    continue
                with open(fs_local, "wb") as f:
                    f.write(fs_resp.content)
                fs_object = f"grader_tmp/fewshot_{i}_{os.path.basename(ex.video_url)}"
                fs_uri = self._upload_to_gcs(fs_local, fs_object)
                os.remove(fs_local)
                parts.append({
                    "file_data": {"mime_type": "video/mp4", "file_uri": fs_uri},
                    "video_metadata": {"fps": self.VIDEO_SAMPLE_FPS},
                    "mediaResolution": {"level": "MEDIA_RESOLUTION_HIGH"},
                })
                parts.append({"text": self._few_shot_label_text(ex, i)})
            except Exception as exc:
                logger.warning("Failed processing few-shot example %d: %s", i, exc)

        parts.append({
            "file_data": {"mime_type": "video/mp4", "file_uri": gcs_uri},
            "video_metadata": {"fps": self.VIDEO_SAMPLE_FPS},
            "mediaResolution": {"level": "MEDIA_RESOLUTION_HIGH"},
        })
        parts.append({"text": self._landing_system_prompt() + self._instruction_text(description)})

        generate_url = self._generate_content_url(self.project, self.LOCATION, self.model_name)
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.0,
                "maxOutputTokens": 8192,
            },
        }
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        try:
            resp = requests.post(generate_url, headers=headers, json=payload)
            if resp.status_code != 200:
                return {"landing_foul": "UNCLEAR", "confidence": "LOW",
                        "reasoning": f"Vertex AI Error {resp.status_code}: {resp.text[:500]}"}
            res_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            return self._normalize_landing(self._parse_json_response(res_text))
        except Exception as exc:
            return {"landing_foul": "UNCLEAR", "confidence": "LOW",
                    "reasoning": f"Vertex AI parse error: {exc}"}
        finally:
            try:
                self._delete_gcs_object(object_name)
            except Exception:
                logger.warning("Failed to delete GCS object %s", object_name)
            for i in range(len(self.few_shot_examples)):
                try:
                    fs_object = f"grader_tmp/fewshot_{i}_{os.path.basename(self.few_shot_examples[i].video_url)}"
                    self._delete_gcs_object(fs_object)
                except Exception:
                    pass


class OpenAILandingGrader(_LandingMixin, OpenAIGrader):
    """OpenAI (frame-based) landing foul grader."""

    def __init__(self, api_key: str, model_name: str, prompt_mode: str = "spatial",
                 few_shot_examples: Optional[List[LandingFewShotExample]] = None):
        super().__init__(api_key=api_key, model_name=model_name,
                         prompt_mode=prompt_mode, few_shot_examples=few_shot_examples or [])

    def grade_clip(self, video_path: str, description: str) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Landing sequence plays out over ~1s; 3fps captures the descent.
            frames = extract_frames_with_ffmpeg(video_path, temp_dir, fps=3.0)
            if not frames:
                return {"landing_foul": "UNCLEAR", "confidence": "LOW",
                        "reasoning": "Could not extract frames from video."}
            frames = frames[:15]

            instruction = self._frame_instruction_text(description)
            content: List[Dict[str, Any]] = [{"type": "text", "text": instruction}]
            for f in frames:
                b64 = encode_image_base64(f)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                })

            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self._landing_system_prompt()},
                    {"role": "user", "content": content},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0,
            }
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            resp = requests.post("https://api.openai.com/v1/chat/completions",
                                 headers=headers, json=payload)
            if resp.status_code != 200:
                return {"landing_foul": "UNCLEAR", "confidence": "LOW",
                        "reasoning": f"OpenAI API Error {resp.status_code}: {resp.text}"}
            res_text = resp.json()["choices"][0]["message"]["content"]
            return self._normalize_landing(self._parse_json_response(res_text))


class AnthropicLandingGrader(_LandingMixin, AnthropicGrader):
    """Anthropic (frame-based) landing foul grader."""

    def __init__(self, api_key: str, model_name: str, prompt_mode: str = "spatial",
                 few_shot_examples: Optional[List[LandingFewShotExample]] = None):
        super().__init__(api_key=api_key, model_name=model_name,
                         prompt_mode=prompt_mode, few_shot_examples=few_shot_examples or [])

    def grade_clip(self, video_path: str, description: str) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory() as temp_dir:
            frames = extract_frames_with_ffmpeg(video_path, temp_dir, fps=2.0)
            if not frames:
                return {"landing_foul": "UNCLEAR", "confidence": "LOW",
                        "reasoning": "Could not extract frames from video."}
            frames = frames[:10]

            content: List[Dict[str, Any]] = []
            for f in frames:
                b64 = encode_image_base64(f)
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                })
            prompt = self._frame_instruction_text(description)
            content.append({"type": "text", "text": prompt})

            payload = {
                "model": self.model_name,
                "max_tokens": 600,
                "system": self._landing_system_prompt() + "\nReturn ONLY raw JSON, do not wrap in markdown code blocks.",
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.0,
            }
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
            if resp.status_code != 200:
                return {"landing_foul": "UNCLEAR", "confidence": "LOW",
                        "reasoning": f"Anthropic API Error {resp.status_code}: {resp.text}"}
            res_text = resp.json()["content"][0]["text"]
            return self._normalize_landing(self._parse_json_response(res_text))


# ---------------------------------------------------------------------------
# Validation analytics
# ---------------------------------------------------------------------------

def print_validation(val_comparisons: List[Dict[str, Any]], prompt_mode: str,
                     include_unclear: bool) -> None:
    if not val_comparisons:
        print("No ground truth comparisons were available to validate predictions.")
        return

    vdf = pd.DataFrame(val_comparisons)

    # Primary binary set: GT in {YES, NO}. UNCLEAR GT rows are reported separately.
    binary_df = vdf[vdf["gt"].isin(["YES", "NO"])].copy()
    unclear_gt = vdf[vdf["gt"] == "UNCLEAR"]

    print("*" * 70)
    print("VALIDATION vs MANUAL GROUND TRUTH (landing foul binary)")
    print("*" * 70)
    print(f"Prompt mode:       {prompt_mode}")
    print(f"Matched clips:     {len(vdf)}")
    print(f"Binary set (GT YES/NO): {len(binary_df)}   (UNCLEAR GT rows: {len(unclear_gt)})")
    print(f"Include UNCLEAR in metrics: {include_unclear}")
    print()

    if len(binary_df) == 0:
        print("No YES/NO ground truth rows to score.\n")
        return

    # Treat model UNCLEAR as a separate prediction bucket.
    eval_df = binary_df[binary_df["pred"].isin(["YES", "NO"])].copy()
    n_decided = len(eval_df)
    n_unclear_pred = (binary_df["pred"] == "UNCLEAR").sum()

    tp = int(((eval_df["gt"] == "YES") & (eval_df["pred"] == "YES")).sum())
    fp = int(((eval_df["gt"] == "NO") & (eval_df["pred"] == "YES")).sum())
    fn = int(((eval_df["gt"] == "YES") & (eval_df["pred"] == "NO")).sum())
    tn = int(((eval_df["gt"] == "NO") & (eval_df["pred"] == "NO")).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / n_decided if n_decided else 0.0

    print("Primary metrics (model YES/NO predictions vs GT YES/NO):")
    print(f"  Decided predictions: {n_decided}/{len(binary_df)}  (model UNCLEAR: {n_unclear_pred})")
    print(f"  Accuracy:  {accuracy:.1%}  ({tp + tn}/{n_decided})")
    print(f"  Precision (YES): {precision:.1%}  ({tp}/{tp + fp})   TARGET >= 85%")
    print(f"  Recall    (YES): {recall:.1%}  ({tp}/{tp + fn})   TARGET >= 70%")
    print(f"  F1        (YES): {f1:.1%}")
    print()

    print("Confusion matrix (rows=GT, cols=Pred):")
    ct = pd.crosstab(binary_df["gt"], binary_df["pred"], margins=True)
    print(ct.to_string())
    print()

    if len(unclear_gt) > 0:
        print(f"Model predictions on {len(unclear_gt)} UNCLEAR-ground-truth clips "
              "(excluded from primary metrics):")
        print(unclear_gt["pred"].value_counts().to_string())
        print()

    print("MISMATCHED cases (GT YES/NO where model disagreed or was UNCLEAR):")
    mismatch = binary_df[binary_df["gt"] != binary_df["pred"]]
    for _, row in mismatch.iterrows():
        print(f"  Clip {row['game_id']}_{row['event_id']}:")
        print(f"    PBP:    {str(row['desc'])[:80]}")
        print(f"    GT:     {row['gt']}  (note: {row.get('gt_note', '')})")
        print(f"    Pred:   {row['pred']}  (conf={row.get('confidence', '')})")
        obs_fields = []
        for k in ("shot_type", "who_initiated_contact", "defender_position_at_landing",
                  "contact_moment", "defender_in_landing_zone", "contact_vs_descent",
                  "narrative", "whistle_timing", "called_contact",
                  "primary_shooter_motion", "primary_defender_body_part",
                  "layer2_rule", "contacts"):
            if k not in row:
                continue
            val = row.get(k)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            if k == "contacts" and isinstance(val, list):
                val = f"{len(val)} contacts"
            obs_fields.append(f"{k}={str(val)[:80]}")
        if obs_fields:
            print(f"    Obs:    {' | '.join(obs_fields)}")
        print(f"    Reason: {row.get('reasoning', '')}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Landing foul LLM video grader (Step 10)")
    parser.add_argument("--provider", required=True,
                        choices=["gemini", "vertex", "openai", "anthropic"],
                        help="LLM provider")
    parser.add_argument("--model", required=True,
                        help="Model name (e.g. gemini-2.5-flash, claude-sonnet-4-6, gpt-5.4-mini)")
    parser.add_argument("--prompt-mode", default="spatial",
                        choices=["spatial", "sequence", "whistle", "observe", "describe"],
                        help="Prompt strategy: spatial (default), event-ordering sequence, "
                             "whistle (attribution via the referee's whistle — audio, "
                             "best with gemini/vertex native video), "
                             "observe (structured observation only — no classification, "
                             "derive post-hoc from feature vector), "
                             "describe (Layer 1: apex/contact sequence; Layer 2: rules)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only grade clips that have manual ground truth")
    parser.add_argument("--val-split", action="store_true",
                        help="Grade only the 57-clip video val split (requires --validate-only)")
    parser.add_argument("--local-clips", action="store_true",
                        help="Use data/clips/landing_foul/*.mp4 when present (skip CDN download)")
    parser.add_argument("--extended", action="store_true",
                        help="Use the extended ground truth set (includes v3 legacy clips)")
    parser.add_argument("--include-unclear", action="store_true",
                        help="Include UNCLEAR ground-truth rows in the graded set "
                             "(still excluded from primary YES/NO metrics)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit grading to first N clips (after filtering)")
    parser.add_argument("--few-shot", action="store_true",
                        help="Include YES/NO few-shot video examples from ground truth")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: data/processed/landing_foul_llm_results_<model>.json)")
    args = parser.parse_args()

    if args.val_split and not args.validate_only:
        print("Error: --val-split requires --validate-only.")
        sys.exit(1)

    # API key (vertex uses gcloud ADC — no key needed)
    key_env_map = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}
    if args.provider == "vertex":
        api_key = ""
    else:
        env_var = key_env_map[args.provider]
        api_key = os.getenv(env_var)
        if not api_key and args.provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print(f"Error: API key for {args.provider} not found in environment ({env_var}).")
            sys.exit(1)

    ground_truth = load_landing_ground_truth()
    if not ground_truth:
        print(f"Error: ground truth not found at {GT_PATH}. Run `make landing-merge` first.")
        sys.exit(1)
    clips_by_key = load_clips_by_key(extended=args.extended)

    # Build the list of clips to grade.
    if args.validate_only:
        keys = sorted(k for k, row in ground_truth.items())
    else:
        # Full run: grade everything in the landing manifest (the scale set).
        keys = sorted(k for k in clips_by_key.keys())

    # Filter by label inclusion.
    def label_allowed(label: str) -> bool:
        label = label.strip().upper()
        if label in ("YES", "NO"):
            return True
        if label == "UNCLEAR":
            return args.include_unclear
        return False

    if args.validate_only:
        keys = [k for k in keys if label_allowed(ground_truth[k].get("landing_foul", ""))]
        # Primary set = Step 9 `landing_classifier` rows only. v3 legacy rows
        # (source == v3_foul_type) require --extended. This matches the HANDOFF
        # evaluation protocol (93 primary YES/NO clips vs 128 extended).
        if not args.extended:
            keys = [k for k in keys
                    if str(ground_truth[k].get("source", "")).strip() != "v3_foul_type"]

    if args.val_split:
        val_keys = set(load_val_split_keys())
        keys = [k for k in keys if k in val_keys]
        logger.info("Filtered to video val split: %d clips", len(keys))

    def _has_video(k: Tuple[str, int]) -> bool:
        if args.local_clips and clip_path(k[0], k[1]).exists():
            return True
        clip = clips_by_key.get(k)
        return bool(clip and _clip_video_url(clip))

    keys = [k for k in keys if _has_video(k)]

    if args.limit:
        keys = keys[: args.limit]
    if not keys:
        print("ERROR: no clips to grade after filtering. Check manifest / ground truth / --extended.")
        sys.exit(1)

    # Few-shot examples (held out from the graded set when validating).
    few_shot_examples: List[LandingFewShotExample] = []
    if args.few_shot:
        graded_keys = set(keys)
        gt_for_fs = {k: v["landing_foul"] for k, v in ground_truth.items() if k not in graded_keys}
        if not gt_for_fs:
            # Fall back to the full GT if everything is being graded (few-shot leakage risk noted).
            gt_for_fs = {k: v["landing_foul"] for k, v in ground_truth.items()}
        few_shot_examples = select_landing_few_shot(clips_by_key, gt_for_fs)
        if not few_shot_examples:
            print("WARNING: --few-shot specified but no usable YES/NO examples found.")

    # Initialize grader
    if args.provider == "gemini":
        grader = GeminiLandingGrader(api_key, args.model, args.prompt_mode, few_shot_examples)
    elif args.provider == "vertex":
        grader = VertexLandingGrader(args.model, args.prompt_mode, few_shot_examples)
    elif args.provider == "openai":
        grader = OpenAILandingGrader(api_key, args.model, args.prompt_mode, few_shot_examples)
    else:
        grader = AnthropicLandingGrader(api_key, args.model, args.prompt_mode, few_shot_examples)

    # Summary header
    gt_count = sum(1 for k in keys if k in ground_truth)
    print("\n" + "=" * 70)
    print("LANDING FOUL LLM GRADER (Step 10)")
    print(f"Provider:    {args.provider.upper()} ({args.model})")
    print(f"Prompt mode: {args.prompt_mode}")
    print(f"Few-shot:    {len(few_shot_examples)} examples" if few_shot_examples else "Few-shot:    off")
    print(f"Clips:       {len(keys)}{' (validate-only)' if args.validate_only else ' (full manifest)'}"
          f"{' [val split]' if args.val_split else ''}")
    print(f"Extended:    {args.extended}   Include UNCLEAR: {args.include_unclear}")
    print(f"Local clips: {args.local_clips}")
    print(f"Ground truth matches: {gt_count} / {len(keys)}")
    print("=" * 70 + "\n")

    results: List[Dict[str, Any]] = []
    val_comparisons: List[Dict[str, Any]] = []
    temp_video_dir = tempfile.mkdtemp()
    video_session = NBAStatsClient().session

    try:
        for key in tqdm(keys, desc="Grading clips"):
            game_id, event_id = key
            clip = clips_by_key.get(key, {})
            video_url = _clip_video_url(clip) if clip else None
            description = clip.get("description", "") if clip else ""
            local_on_disk = clip_path(game_id, event_id)
            if args.local_clips and local_on_disk.exists():
                grade_path = str(local_on_disk)
            else:
                if not video_url:
                    logger.warning("No video for %s_%s", game_id, event_id)
                    continue
                grade_path = os.path.join(temp_video_dir, f"clip_{game_id}_{event_id}.mp4")
                try:
                    resp = video_session.get(video_url, timeout=30)
                    if resp.status_code == 200:
                        with open(grade_path, "wb") as f:
                            f.write(resp.content)
                    else:
                        logger.warning("Failed downloading video %s: HTTP %d", video_url, resp.status_code)
                        continue
                except Exception as exc:
                    logger.warning("Error downloading video %s: %s", video_url, exc)
                    continue

            grade = grader.grade_clip(grade_path, description)

            res_entry: Dict[str, Any] = {
                "game_id": game_id,
                "event_id": event_id,
                "description": description,
                "predicted_landing_foul": grade.get("landing_foul", "UNCLEAR"),
                "confidence": grade.get("confidence", "LOW"),
                "reasoning": grade.get("reasoning", ""),
                "caller_official_name": clip.get("caller_official_name", "") if clip else "",
            }
            for k in ("shot_type", "defender_position_at_landing", "contact_moment",
                      "defender_in_landing_zone", "contact_vs_descent", "narrative",
                      "whistle_timing", "called_contact", "who_initiated_contact",
                      "contacts", "primary_foul_contact_index",
                      "primary_shooter_motion", "primary_defender_body_part", "layer2_rule"):
                if k in grade:
                    res_entry[k] = grade[k]
            results.append(res_entry)

            if key in ground_truth:
                gt_row = ground_truth[key]
                val_comparisons.append({
                    "game_id": game_id,
                    "event_id": event_id,
                    "gt": str(gt_row.get("landing_foul", "")).strip().upper(),
                    "gt_note": str(gt_row.get("note", "") or ""),
                    "pred": str(grade.get("landing_foul", "UNCLEAR")).upper(),
                    "confidence": grade.get("confidence", "LOW"),
                    "desc": description,
                    "reasoning": grade.get("reasoning", ""),
                    **{k: grade.get(k) for k in ("shot_type", "defender_position_at_landing",
                       "contact_moment", "defender_in_landing_zone", "contact_vs_descent",
                       "narrative", "whistle_timing", "called_contact",
                       "who_initiated_contact", "contacts", "primary_shooter_motion",
                       "primary_defender_body_part", "layer2_rule") if k in grade},
                })
    finally:
        shutil.rmtree(temp_video_dir, ignore_errors=True)

    # Save results
    out_path = Path(args.output) if args.output else (
        config.PROCESSED_DIR
        / f"landing_foul_llm_results_{args.provider}_{args.model.replace('.', '_')}_{args.prompt_mode}.json"
    )
    output_payload = {
        "task": "landing_foul",
        "provider": args.provider,
        "model": args.model,
        "prompt_mode": args.prompt_mode,
        "val_split": args.val_split,
        "local_clips": args.local_clips,
        "extended": args.extended,
        "include_unclear": args.include_unclear,
        "few_shot_count": len(few_shot_examples),
        "timestamp": pd.Timestamp.now().isoformat(),
        "num_graded": len(results),
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_payload, f, indent=2)

    print("\n" + "=" * 70)
    print(f"RUN COMPLETE. Results saved to {out_path}")
    print("=" * 70 + "\n")

    if val_comparisons:
        print_validation(val_comparisons, args.prompt_mode, args.include_unclear)
    else:
        print("No ground truth comparisons in this run (run with --validate-only to score).")


if __name__ == "__main__":
    main()
