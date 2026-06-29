"""Foul-Type Video Grader using Multimodal LLMs.

Grades the TIMING axis (BEFORE, DURING, AFTER) of shooting fouls from video clips.
Supports Google (Gemini — recommended, native video), OpenAI (GPT-5.4 mini), and Anthropic (Claude Sonnet 4.6).

Recommended usage (Gemini — native video understanding, no frame extraction):
    python src/foul_type_llm_grader.py --player "James Harden" --provider "gemini" --model "gemini-2.5-flash"

Validate against manual ground truth first:
    python src/foul_type_llm_grader.py --player "James Harden" --provider "gemini" --model "gemini-2.5-flash" --validate-only

Alternative providers (frame-based, less temporal precision):
    python src/foul_type_llm_grader.py --player "James Harden" --provider "openai" --model "gpt-5.4-mini"
    python src/foul_type_llm_grader.py --player "James Harden" --provider "anthropic" --model "claude-sonnet-4-6"
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.nba_client import NBAStatsClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt modes
# ---------------------------------------------------------------------------

# OBSERVATION_PROMPT — collapsed 3-field schema (replaces the 13-field legacy prompt).
# Asks for the three discriminative observations that determine phase.
OBSERVATION_PROMPT = """You are an expert NBA officiating analyst. Watch a short video clip of a shooting foul and answer three observation questions about the EXACT frame of first illegal contact.

Your job is NOT to judge if the call was correct. Do NOT output a phase/timing classification — only observations. Classification is computed from your answers.

CRITICAL ERROR TO AVOID: Do NOT assume a committed shot just because the player is jumping or arms are elevated. In bait fouls and arm-hooks, the shooter RISES into a mostly-still defender with a sideways hook arm or chest-high ball. That is NOT a shot release.

Watch the clip twice: (1) who moves into whom at contact, (2) ball + arm geometry at the freeze frame. Ignore what happens 0.2s after contact.

Answer these three questions:

Q1 — WHO INITIATED CONTACT (answer first — most important):
  SHOOTER: The shooter's arm/body traveled INTO the defender to create contact while the defender was mostly still or already in legal position (arm-hooks, bait rises, rip-throughs where shooter seeks contact).
  DEFENDER: The defender's arm/hand clearly traveled toward the shooter/ball to make contact (genuine contest, swipe, reach-in).
  MUTUAL: Both players moved into each other simultaneously.
  UNCLEAR: Cannot determine who initiated.

Q2 — BALL STATE at contact:
  GATHERING: Ball at hip/chest, dribble just stopped or gather-to-shot transition not yet complete. Shooter could still pass, pivot, or pump-fake.
  RISING_NO_RELEASE: Player airborne/rising but ball NOT on release path — gather, pump, or bait rise. Both hands not yet driving the ball toward the rim on a vertical arc.
  ON_RELEASE_PATH: Both hands under the ball driving it straight toward the rim on a vertical arc. Ball above shoulder or at release point. Shooter is in the act of shooting.
  RELEASED: Ball visibly off fingertips before contact.
  UNCLEAR: Cannot determine.

Q3 — ARM GEOMETRY at contact:
  HORIZONTAL_HOOK: The contacting arm is a sideways hook/bar — elbow out, forearm roughly horizontal into defender. This includes off-arm hooks at chest/shoulder height.
  VERTICAL_SHOT_ARC: Both elbows under/behind the ball on a vertical path toward the rim. A genuine shooting motion.
  UNCLEAR: Cannot determine.

Return a JSON object (do NOT include phase or timing):
{
  "who_initiated": "SHOOTER" | "DEFENDER" | "MUTUAL" | "UNCLEAR",
  "ball_state_at_contact": "GATHERING" | "RISING_NO_RELEASE" | "ON_RELEASE_PATH" | "RELEASED" | "UNCLEAR",
  "arm_geometry": "HORIZONTAL_HOOK" | "VERTICAL_SHOT_ARC" | "UNCLEAR",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "reasoning": "One sentence citing who initiated contact, ball state, and arm geometry at the freeze frame."
}"""


# DIRECT_TIMING_PROMPT — asks for timing directly with justification.
# Simpler than the observation-proxy architecture; test as a baseline.
DIRECT_TIMING_PROMPT = """You are an expert NBA officiating analyst. Watch a short video clip of a shooting foul and classify the TIMING of the first illegal contact relative to the shot release.

Your job is NOT to judge if the call was correct. Focus only on WHEN the contact happens relative to the shooting motion.

Definitions:
  BEFORE: Contact happens during the gather/dribble-gather, before the shooting motion begins. The shooter could still pass, pivot, or pump-fake. Ball is at hip/chest or the player is rising without a committed release path. Includes arm-hooks, bait rises, rip-throughs, pump-fake jump-intos, and drive-initiate fouls.
  DURING: Contact happens simultaneous with the upward shooting motion. Both hands are on a vertical arc driving the ball toward the rim. The shooter is in the act of shooting and cannot abort.
  AFTER: Contact happens after the ball leaves the shooter's hand. The ball is visibly released before contact occurs. Includes landing space fouls.

CRITICAL ERROR TO AVOID: Do NOT classify as DURING just because the player is jumping or arms are elevated. In bait fouls and arm-hooks, the shooter RISES into a mostly-still defender with a sideways hook arm or chest-high ball — that is BEFORE, not DURING. DURING requires both hands driving the ball toward the rim on a vertical arc.

Watch the clip twice: (1) identify the exact frame of first illegal contact, (2) determine the ball state at that frame.

Return a JSON object:
{
  "timing": "BEFORE" | "DURING" | "AFTER",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "reasoning": "One sentence explaining the ball state and body position at the frame of first contact."
}"""


# SEQUENCE_PROMPT — event-ordering approach.
# Asks the model to identify observable events and report their temporal order
# relative to contact, rather than classifying a freeze-frame state.
SEQUENCE_PROMPT = """You are an expert NBA officiating analyst. Watch a short video clip of a shooting foul and determine when the first illegal contact occurs relative to the shooting motion.

Your job is NOT to judge if the call was correct. Focus only on the temporal ordering of observable events.

STEP 1 — NARRATIVE: Write 2-3 sentences describing what happens in the clip chronologically. Focus on the sequence: when does the defender's arm begin moving toward the shooter? When does the shooter's arm begin extending upward for the shot? When does contact occur? When does the ball leave the hand (if it does)?

STEP 2 — EVENT ORDERING: Identify which of these observable events occurred FIRST in the clip. These are temporal milestones — report which one happened earliest:
  DEFENDER_REACH: The defender's arm/hand begins moving toward the shooter or ball (the contest or reach-in motion starts).
  FEET_SET: The shooter's feet are set and lower body has stopped moving — the gather is complete and the shooter is planted for the shot.
  ARM_EXTEND: The shooter's shooting arm begins extending upward — the shooting motion has started (not just rising, but the arm is actively pushing the ball toward the rim).
  BALL_RELEASED: The ball has left the shooter's fingertips.
  CONTACT: The first illegal contact between the players.

STEP 3 — CONTACT TIMING: Based on the event ordering, classify when contact occurred:
  BEFORE: Contact happens before ARM_EXTEND — the defender reaches in or the shooter initiates contact while still gathering or rising but before the shooting arm has started extending toward the rim. The shooter has not yet committed to the shot.
  DURING: Contact happens after ARM_EXTEND but before BALL_RELEASED — the shooting arm is actively extending upward, driving the ball toward the rim, and contact occurs during this motion.
  AFTER: Contact happens after BALL_RELEASED — the ball has left the shooter's hand before contact occurs.

CRITICAL: The key distinction is ARM_EXTEND, not whether the player is jumping or rising. A player can be airborne with the ball at chest height (rising to shoot or baiting contact) without the shooting arm having started its extension. If the arm is still cocked or the ball is still being brought up, that is BEFORE, not DURING. DURING requires the arm to be actively extending the ball toward the rim.

Return a JSON object:
{
  "narrative": "2-3 sentence chronological description of what happens in the clip.",
  "first_event": "DEFENDER_REACH" | "FEET_SET" | "ARM_EXTEND" | "BALL_RELEASED" | "CONTACT",
  "contact_vs_arm_extend": "BEFORE_ARM_EXTEND" | "DURING_ARM_EXTEND" | "AFTER_RELEASE",
  "timing": "BEFORE" | "DURING" | "AFTER",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "reasoning": "One sentence citing the event ordering that determined the timing."
}"""


# Legacy 13-field prompt — kept for backward compatibility with existing result files.
SYSTEM_PROMPT = OBSERVATION_PROMPT


def phase_to_timing(grade: Dict[str, Any]) -> str:
    """Map binary phase (+ release flag) to legacy BEFORE/DURING/AFTER timing label."""
    phase = str(grade.get("phase", "UNKNOWN")).upper()
    if phase == "PRE_COMMIT":
        return "BEFORE"
    if phase == "IN_ACT":
        # New 3-field schema
        if grade.get("ball_state_at_contact") == "RELEASED":
            return "AFTER"
        # Legacy 13-field schema
        released = grade.get("ball_released_before_contact")
        if released is True:
            return "AFTER"
        return "DURING"
    return "UNKNOWN"


def _derive_phase_from_observations(grade: Dict[str, Any]) -> str:
    """Deterministically derive PRE_COMMIT/IN_ACT from model observations.

    Supports two schemas:
      - New 3-field: who_initiated, ball_state_at_contact, arm_geometry
      - Legacy 13-field: shooter_initiates_contact, hook_or_bar_arm, etc.
    """
    # --- New 3-field schema ---
    new_keys = ("who_initiated", "ball_state_at_contact", "arm_geometry")
    if any(grade.get(k) is not None for k in new_keys):
        who = str(grade.get("who_initiated", "")).upper()
        ball = str(grade.get("ball_state_at_contact", "")).upper()
        arm = str(grade.get("arm_geometry", "")).upper()

        # PRE_COMMIT signals — shooter initiated, hook arm, or ball not on release path
        if who == "SHOOTER":
            return "PRE_COMMIT"
        if arm == "HORIZONTAL_HOOK":
            return "PRE_COMMIT"
        if ball in ("GATHERING", "RISING_NO_RELEASE"):
            return "PRE_COMMIT"

        # IN_ACT signals — ball released or on release path with vertical arc
        if ball == "RELEASED":
            return "IN_ACT"
        if ball == "ON_RELEASE_PATH" and arm == "VERTICAL_SHOT_ARC" and who != "SHOOTER":
            return "IN_ACT"

        # Tie-breaker: doubt → PRE_COMMIT
        return "PRE_COMMIT"

    # --- Legacy 13-field schema (backward compatibility) ---
    obs_keys = (
        "contact_initiator", "arm_direction_at_contact", "ball_location_at_contact",
        "shooter_body_state", "shooter_initiates_contact", "hook_or_bar_arm",
        "both_hands_on_vertical_arc", "rising_without_release",
    )
    if not any(grade.get(k) is not None for k in obs_keys):
        return str(grade.get("phase", "UNKNOWN")).upper()

    initiator = str(grade.get("contact_initiator", "")).upper()
    arm = str(grade.get("arm_direction_at_contact", "")).upper()
    body = str(grade.get("shooter_body_state", "")).upper()
    ball = str(grade.get("ball_location_at_contact", "")).upper()

    # PRE_COMMIT veto — any true → PRE_COMMIT
    if grade.get("shooter_initiates_contact") is True:
        return "PRE_COMMIT"
    if grade.get("hook_or_bar_arm") is True:
        return "PRE_COMMIT"
    if grade.get("rising_without_release") is True:
        return "PRE_COMMIT"
    if initiator == "SHOOTER_REACHES_INTO_DEFENDER":
        return "PRE_COMMIT"
    if arm in ("HORIZONTAL_OUT", "LATERAL", "UPWARD_BAIT"):
        return "PRE_COMMIT"
    if body in ("GATHERING", "DRIVE_STEP", "BAIT_RISE"):
        return "PRE_COMMIT"
    if ball in ("HIP", "CHEST"):
        return "PRE_COMMIT"
    if grade.get("gather_complete_at_contact") is False:
        return "PRE_COMMIT"
    if grade.get("could_still_abort") is True:
        return "PRE_COMMIT"
    if grade.get("both_hands_on_vertical_arc") is False:
        return "PRE_COMMIT"

    # IN_ACT — all required signals present
    if grade.get("ball_released_before_contact") is True:
        return "IN_ACT"
    if (
        grade.get("gather_complete_at_contact") is True
        and grade.get("both_hands_on_vertical_arc") is True
        and grade.get("shooter_initiates_contact") is False
        and grade.get("hook_or_bar_arm") is False
        and grade.get("rising_without_release") is False
        and arm == "UPWARD_SHOT_ARC"
        and grade.get("could_still_abort") is False
        and ball in ("ABOVE_SHOULDER", "AT_RELEASE", "RELEASED")
        and body in ("SHOT_RELEASE", "LANDING")
    ):
        return "IN_ACT"

    # Tie-breaker: doubt → PRE_COMMIT
    return "PRE_COMMIT"


def timing_to_binary_phase(timing: str) -> Optional[str]:
    """Collapse manual 3-way timing labels to binary ground truth."""
    timing = timing.strip().upper()
    if timing == "BEFORE":
        return "PRE_COMMIT"
    if timing in ("DURING", "AFTER"):
        return "IN_ACT"
    return None


# ---------------------------------------------------------------------------
# Few-Shot Examples
# ---------------------------------------------------------------------------

@dataclass
class FewShotExample:
    """A labeled video clip used as a few-shot example in the prompt."""
    game_id: str
    event_id: int
    description: str
    timing: str  # BEFORE | DURING | AFTER
    video_url: str
    reasoning: str = ""


def select_few_shot_examples(clips: List[Dict[str, Any]], ground_truth: Dict[Tuple[str, int], str]) -> List[FewShotExample]:
    """Select 2-3 few-shot examples from the manifest using manual ground truth.

    Picks one BEFORE and one DURING/AFTER example to anchor the model to both
    classes. Prefers clips with clear, unambiguous descriptions.
    """
    examples: List[FewShotExample] = []
    seen_timings: set = set()

    for clip in clips:
        gid = str(clip["game_id"]).zfill(10)
        eid = int(clip["event_id"])
        key = (gid, eid)
        if key not in ground_truth:
            continue
        timing = ground_truth[key].strip().upper()
        if timing not in ("BEFORE", "DURING", "AFTER"):
            continue
        if timing in seen_timings and len(seen_timings) >= 2:
            continue

        video_url = clip.get("video_url_720") or clip.get("video_url_960")
        if not video_url:
            continue

        examples.append(FewShotExample(
            game_id=gid,
            event_id=eid,
            description=clip["description"],
            timing=timing,
            video_url=video_url,
        ))
        seen_timings.add(timing)

        if len(examples) >= 3 or len(seen_timings) >= 3:
            break

    return examples


# ---------------------------------------------------------------------------
# Frame Extraction Helper using local ffmpeg
# ---------------------------------------------------------------------------

def extract_frames_with_ffmpeg(video_path: str, output_dir: str, fps: float = 1.0) -> List[str]:
    """Extract frames from video at specific fps using local ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)
    # Target naming: frame_0001.jpg, frame_0002.jpg
    frame_pattern = os.path.join(output_dir, "frame_%04d.jpg")
    
    # We find ffmpeg path on mac
    ffmpeg_path = "/opt/local/bin/ffmpeg" if os.path.exists("/opt/local/bin/ffmpeg") else "ffmpeg"
    
    cmd = [
        ffmpeg_path, "-y", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",  # high quality JPEGs
        frame_pattern
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # List and sort the generated frames
    frames = sorted([
        os.path.join(output_dir, f) for f in os.listdir(output_dir)
        if f.startswith("frame_") and f.endswith(".jpg")
    ])
    return frames


def encode_image_base64(image_path: str) -> str:
    """Encode binary image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Grader Base and Provider Subclasses (Pure requests-based to avoid library dependencies)
# ---------------------------------------------------------------------------

class LLMGrader(ABC):
    """Abstract Base Class for LLM-based video grading.

    Args:
        api_key: API key for the provider (empty for Vertex ADC).
        model_name: Model identifier string.
        prompt_mode: "observation" (3-field schema), "direct" (timing directly),
                     or "legacy" (13-field schema). Defaults to "observation".
        few_shot_examples: Optional list of FewShotExample to include in the prompt.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        prompt_mode: str = "observation",
        few_shot_examples: Optional[List["FewShotExample"]] = None,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.prompt_mode = prompt_mode
        self.few_shot_examples = few_shot_examples or []

    @abstractmethod
    def grade_clip(self, video_path: str, description: str) -> Dict[str, Any]:
        """Grade a single video clip given its path and PBP play description."""
        pass

    def _get_system_prompt(self) -> str:
        """Return the system prompt for the active prompt mode."""
        if self.prompt_mode == "direct":
            return DIRECT_TIMING_PROMPT
        if self.prompt_mode == "sequence":
            return SEQUENCE_PROMPT
        if self.prompt_mode == "legacy":
            return SYSTEM_PROMPT
        return OBSERVATION_PROMPT

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Utility to safely extract and parse JSON object from LLM response text."""
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            chunk = text[start:end + 1]
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                # Try repairing truncated JSON (common when output token limit cuts off)
                for suffix in ('"}', '"}', 'null}', 'false}', 'true}'):
                    try:
                        return json.loads(chunk + suffix)
                    except json.JSONDecodeError:
                        continue
        return {
            "phase": "UNKNOWN",
            "timing": "UNKNOWN",
            "confidence": "LOW",
            "reasoning": f"Failed to parse model response: {text[:200]}",
        }

    def _normalize_grade(self, grade: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure phase/timing fields are present and consistent.

        Handles four prompt modes:
          - direct: model outputs timing directly; derive phase from it.
          - sequence: model outputs timing + event ordering; derive phase from timing.
          - observation: model outputs 3-field schema; derive phase from observations.
          - legacy: model outputs 13-field schema; derive phase from observations.
        """
        # Direct-timing and sequence modes: model outputs timing directly
        if self.prompt_mode in ("direct", "sequence"):
            timing = str(grade.get("timing", "UNKNOWN")).upper()
            if timing in ("BEFORE", "DURING", "AFTER"):
                grade["phase"] = timing_to_binary_phase(timing) or "UNKNOWN"
            else:
                grade["phase"] = "UNKNOWN"
                grade["timing"] = "UNKNOWN"
            return grade

        # Observation and legacy modes: derive phase from observations
        if "phase" not in grade and "timing" in grade:
            timing = str(grade.get("timing", "UNKNOWN")).upper()
            if timing == "BEFORE":
                grade["phase"] = "PRE_COMMIT"
            elif timing in ("DURING", "AFTER"):
                grade["phase"] = "IN_ACT"
            else:
                grade["phase"] = "UNKNOWN"

        # New 3-field schema keys
        new_keys = ("who_initiated", "ball_state_at_contact", "arm_geometry")
        # Legacy 13-field schema keys
        legacy_keys = (
            "shooter_initiates_contact", "hook_or_bar_arm", "both_hands_on_vertical_arc",
            "rising_without_release", "contact_initiator", "arm_direction_at_contact",
        )
        if any(grade.get(k) is not None for k in new_keys + legacy_keys):
            grade["phase"] = _derive_phase_from_observations(grade)
        grade["timing"] = phase_to_timing(grade)
        return grade


class OpenAIGrader(LLMGrader):
    """Grader using OpenAI's chat completions API (GPT-5.4-mini / GPT-5.4)."""

    def grade_clip(self, video_path: str, description: str) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract at 3fps to capture the ~0.2s gather-to-release transition; cap at 15 frames
            frames = extract_frames_with_ffmpeg(video_path, temp_dir, fps=3.0)
            if not frames:
                return {"timing": "UNKNOWN", "confidence": "LOW", "reasoning": "Could not extract frames from video."}
            frames = frames[:15]
            
            # OpenAI prompt structure
            if self.prompt_mode in ("direct", "sequence"):
                instruction = f"Play-by-play description: {description}\n\nAnalyze the chronological sequence of frames below and classify the timing of the foul contact."
            else:
                instruction = f"Play-by-play description: {description}\n\nAnalyze the chronological sequence of frames below and answer the observation questions."
            content: List[Dict[str, Any]] = [
                {"type": "text", "text": instruction}
            ]
            
            # Append images
            for f in frames:
                b64 = encode_image_base64(f)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "low"  # low detail saves tokens and cost
                    }
                })

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": content}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0
            }

            resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            if resp.status_code != 200:
                return {"timing": "UNKNOWN", "confidence": "LOW", "reasoning": f"OpenAI API Error {resp.status_code}: {resp.text}"}
            
            res_text = resp.json()["choices"][0]["message"]["content"]
            return self._normalize_grade(self._parse_json_response(res_text))


class AnthropicGrader(LLMGrader):
    """Grader using Anthropic's Messages API (Claude Sonnet 4.6 / Claude Haiku 4.5)."""

    def grade_clip(self, video_path: str, description: str) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory() as temp_dir:
            # 2fps to capture the gather-to-release transition; cap at 10 frames for Claude token limits
            frames = extract_frames_with_ffmpeg(video_path, temp_dir, fps=2.0)
            if not frames:
                return {"timing": "UNKNOWN", "confidence": "LOW", "reasoning": "Could not extract frames from video."}
            
            frames = frames[:10]  # cap at 10 frames to avoid rate limits / context limits
            
            content: List[Dict[str, Any]] = []
            # Append frames first
            for f in frames:
                b64 = encode_image_base64(f)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64
                    }
                })
            
            # Append prompt text at the end
            if self.prompt_mode in ("direct", "sequence"):
                prompt = f"Play-by-play description: {description}\n\nAnalyze the chronological sequence of frames above. Classify the timing of the contact relative to the shot release."
            else:
                prompt = f"Play-by-play description: {description}\n\nAnalyze the chronological sequence of frames above. Answer the observation questions."
            content.append({
                "type": "text",
                "text": prompt
            })

            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "max_tokens": 400,
                "system": self._get_system_prompt() + "\nReturn ONLY raw JSON, do not wrap in markdown code blocks.",
                "messages": [
                    {"role": "user", "content": content}
                ],
                "temperature": 0.0
            }

            resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
            if resp.status_code != 200:
                return {"timing": "UNKNOWN", "confidence": "LOW", "reasoning": f"Anthropic API Error {resp.status_code}: {resp.text}"}
            
            res_text = resp.json()["content"][0]["text"]
            return self._normalize_grade(self._parse_json_response(res_text))


class VertexGeminiGrader(LLMGrader):
    """Grader using Vertex AI Gemini via gcloud ADC — no API key required.

    Authenticates via ``gcloud auth application-default print-access-token``,
    uploads videos to a GCS bucket, and calls the Vertex AI generateContent
    endpoint with native video understanding.
    """

    GCS_BUCKET = "project-3984c931-3755-423f-966-foul-type-grader-tmp"
    # Gemini 3.x models are only available on the global Vertex endpoint in many projects.
    LOCATION = "global"
    # Sample above default 1 fps to capture gather-to-release transitions (~0.2–0.5s).
    VIDEO_SAMPLE_FPS = 5.0

    def __init__(
        self,
        model_name: str,
        project: Optional[str] = None,
        prompt_mode: str = "observation",
        few_shot_examples: Optional[List["FewShotExample"]] = None,
    ):
        super().__init__(api_key="", model_name=model_name, prompt_mode=prompt_mode, few_shot_examples=few_shot_examples)
        self.project = project or self._detect_project()
        self._token_cache: Tuple[float, str] = (0.0, "")

    @staticmethod
    def _generate_content_url(project: str, location: str, model_name: str) -> str:
        path = (
            f"projects/{project}/locations/{location}/"
            f"publishers/google/models/{model_name}:generateContent"
        )
        if location == "global":
            return f"https://aiplatform.googleapis.com/v1/{path}"
        return f"https://{location}-aiplatform.googleapis.com/v1/{path}"

    @staticmethod
    def _detect_project() -> str:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True, text=True,
        )
        project = result.stdout.strip()
        if not project:
            raise RuntimeError("Could not detect GCP project. Run `gcloud config set project <PROJECT>`.")
        return project

    def _access_token(self) -> str:
        ts, tok = self._token_cache
        if tok and (time.time() - ts) < 300:
            return tok
        result = subprocess.run(
            ["gcloud", "auth", "application-default", "print-access-token"],
            capture_output=True, text=True,
        )
        token = result.stdout.strip()
        if not token:
            raise RuntimeError("Failed to obtain access token via gcloud ADC.")
        self._token_cache = (time.time(), token)
        return token

    def _upload_to_gcs(self, local_path: str, object_name: str) -> str:
        token = self._access_token()
        url = f"https://storage.googleapis.com/upload/storage/v1/b/{self.GCS_BUCKET}/o"
        params = {"uploadType": "media", "name": object_name}
        with open(local_path, "rb") as f:
            data = f.read()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "video/mp4",
        }
        resp = requests.post(url, headers=headers, params=params, data=data)
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"GCS upload failed ({resp.status_code}): {resp.text}")
        return f"gs://{self.GCS_BUCKET}/{object_name}"

    def _delete_gcs_object(self, object_name: str) -> None:
        token = self._access_token()
        url = f"https://storage.googleapis.com/storage/v1/b/{self.GCS_BUCKET}/o/{object_name}"
        headers = {"Authorization": f"Bearer {token}"}
        requests.delete(url, headers=headers)

    def grade_clip(self, video_path: str, description: str) -> Dict[str, Any]:
        token = self._access_token()
        object_name = f"grader_tmp/{os.path.basename(video_path)}"

        try:
            gcs_uri = self._upload_to_gcs(video_path, object_name)
        except Exception as exc:
            return {"timing": "UNKNOWN", "confidence": "LOW", "reasoning": f"GCS upload error: {exc}"}

        system_prompt = self._get_system_prompt()

        # Build the user content parts: few-shot examples first, then the target clip
        parts: List[Dict[str, Any]] = []

        # Few-shot examples (uploaded to GCS)
        few_shot_uris: List[str] = []
        for i, ex in enumerate(self.few_shot_examples):
            try:
                # Download the few-shot video locally, then upload to GCS
                fs_local = os.path.join(tempfile.mkdtemp(), f"fewshot_{i}.mp4")
                fs_session = NBAStatsClient().session
                fs_resp = fs_session.get(ex.video_url, timeout=30)
                if fs_resp.status_code != 200:
                    logger.warning("Failed downloading few-shot video %s", ex.video_url)
                    continue
                with open(fs_local, "wb") as f:
                    f.write(fs_resp.content)
                fs_object = f"grader_tmp/fewshot_{i}_{os.path.basename(ex.video_url)}"
                fs_uri = self._upload_to_gcs(fs_local, fs_object)
                few_shot_uris.append(fs_uri)
                os.remove(fs_local)

                # Add the few-shot video
                parts.append({
                    "file_data": {"mime_type": "video/mp4", "file_uri": fs_uri},
                    "video_metadata": {"fps": self.VIDEO_SAMPLE_FPS},
                    "mediaResolution": {"level": "MEDIA_RESOLUTION_HIGH"},
                })
                # Add the few-shot label
                if self.prompt_mode in ("direct", "sequence"):
                    parts.append({"text": f"Example {i+1}: {ex.description}\nCorrect timing: {ex.timing}\n"})
                else:
                    phase = timing_to_binary_phase(ex.timing) or "UNKNOWN"
                    parts.append({"text": f"Example {i+1}: {ex.description}\nCorrect phase: {phase} (timing={ex.timing})\n"})
            except Exception as exc:
                logger.warning("Failed processing few-shot example %d: %s", i, exc)

        # Target clip
        parts.append({
            "file_data": {"mime_type": "video/mp4", "file_uri": gcs_uri},
            "video_metadata": {"fps": self.VIDEO_SAMPLE_FPS},
            "mediaResolution": {"level": "MEDIA_RESOLUTION_HIGH"},
        })

        # Build the instruction text
        if self.prompt_mode in ("direct", "sequence"):
            instruction = f"\n\nPlay-by-play description: {description}\n\nWatch the clip above. Classify the timing of the first illegal contact relative to the shot release. Return a raw JSON object."
        else:
            instruction = f"\n\nPlay-by-play description: {description}\n\nWatch the clip above. Answer Q1 (who initiated contact) before Q2/Q3. Do NOT output phase or timing — only observations. Return a raw JSON object."
        parts.append({"text": system_prompt + instruction})

        generate_url = self._generate_content_url(self.project, self.LOCATION, self.model_name)
        payload = {
            "contents": [{
                "role": "user",
                "parts": parts
            }],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.0,
                "maxOutputTokens": 8192,
            }
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(generate_url, headers=headers, json=payload)
            if resp.status_code != 200:
                return {"timing": "UNKNOWN", "confidence": "LOW", "reasoning": f"Vertex AI Error {resp.status_code}: {resp.text[:500]}"}
            res_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            return self._normalize_grade(self._parse_json_response(res_text))
        except Exception as exc:
            return {
                "phase": "UNKNOWN",
                "timing": "UNKNOWN",
                "confidence": "LOW",
                "reasoning": f"Vertex AI parse error: {exc}",
            }
        finally:
            try:
                self._delete_gcs_object(object_name)
            except Exception:
                logger.warning("Failed to delete GCS object %s", object_name)
            # Clean up few-shot GCS objects
            for i in range(len(self.few_shot_examples)):
                try:
                    fs_object = f"grader_tmp/fewshot_{i}_{os.path.basename(self.few_shot_examples[i].video_url)}"
                    self._delete_gcs_object(fs_object)
                except Exception:
                    pass


class GeminiGrader(LLMGrader):
    """Grader using Google's Gemini API via direct File uploads (best native video understanding)."""

    def _upload_file(self, video_path: str) -> Optional[Tuple[str, str]]:
        """Upload a single video to the Gemini Files API. Returns (file_uri, file_name) or None."""
        file_size = os.path.getsize(video_path)

        # Initiate resumable upload session
        headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
            "X-Goog-Upload-Header-Content-Type": "video/mp4",
            "Content-Type": "application/json",
        }

        metadata = {
            "file": {"display_name": os.path.basename(video_path)}
        }

        url_upload_init = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={self.api_key}"
        resp_init = requests.post(url_upload_init, headers=headers, json=metadata)
        if resp_init.status_code != 200:
            logger.warning("Gemini Upload Init Error: %s", resp_init.text)
            return None

        upload_url = resp_init.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            logger.warning("Gemini Upload URL not returned.")
            return None

        # Upload actual bytes
        headers_upload = {
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
            "Content-Length": str(file_size),
        }
        with open(video_path, "rb") as f:
            resp_upload = requests.post(upload_url, headers=headers_upload, data=f.read())

        if resp_upload.status_code != 200:
            logger.warning("Gemini Upload Bytes Error: %s", resp_upload.text)
            return None

        file_info = resp_upload.json()
        file_uri = file_info.get("file", {}).get("uri")
        file_name = file_info.get("file", {}).get("name")

        if not file_uri:
            logger.warning("Gemini File URI not returned.")
            return None

        # Poll file processing status until ACTIVE
        headers_get = {"Content-Type": "application/json"}
        get_url = f"https://generativelanguage.googleapis.com/v1beta/{file_name}?key={self.api_key}"

        status = "PROCESSING"
        max_attempts = 15
        attempt = 0
        while status == "PROCESSING" and attempt < max_attempts:
            time.sleep(2)
            resp_get = requests.get(get_url, headers=headers_get)
            if resp_get.status_code == 200:
                status = resp_get.json().get("state", "PROCESSING")
            attempt += 1

        if status != "ACTIVE":
            requests.delete(get_url)
            logger.warning("Gemini Video Processing failed/timeout (state=%s)", status)
            return None

        return (file_uri, file_name)

    def grade_clip(self, video_path: str, description: str) -> Dict[str, Any]:
        # Step 1: Upload target video
        target_upload = self._upload_file(video_path)
        if not target_upload:
            return {"timing": "UNKNOWN", "confidence": "LOW", "reasoning": "Failed to upload target video to Gemini Files API."}
        target_uri, target_name = target_upload

        # Step 1b: Upload few-shot example videos
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

        # Step 2: Build the generateContent payload
        system_prompt = self._get_system_prompt()
        parts: List[Dict[str, Any]] = []

        # Few-shot examples first
        for i, (fs_uri, _) in enumerate(few_shot_uploads):
            parts.append({"file_data": {"mime_type": "video/mp4", "file_uri": fs_uri}})
            ex = self.few_shot_examples[i]
            if self.prompt_mode in ("direct", "sequence"):
                parts.append({"text": f"Example {i+1}: {ex.description}\nCorrect timing: {ex.timing}\n"})
            else:
                phase = timing_to_binary_phase(ex.timing) or "UNKNOWN"
                parts.append({"text": f"Example {i+1}: {ex.description}\nCorrect phase: {phase} (timing={ex.timing})\n"})

        # Target clip
        parts.append({"file_data": {"mime_type": "video/mp4", "file_uri": target_uri}})

        # Instruction text
        if self.prompt_mode in ("direct", "sequence"):
            instruction = f"\n\nPlay-by-play description: {description}\n\nWatch the clip above. Classify the timing of the first illegal contact relative to the shot release. Return a raw JSON object."
        else:
            instruction = f"\n\nPlay-by-play description: {description}\n\nWatch the clip above. Answer Q1 (who initiated contact) before Q2/Q3. Do NOT output phase or timing — only observations. Return a raw JSON object."
        parts.append({"text": system_prompt + instruction})

        generate_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.0
            }
        }

        headers_get = {"Content-Type": "application/json"}
        resp_gen = requests.post(generate_url, headers=headers_get, json=payload)

        # Clean up all files on Google servers immediately
        requests.delete(f"https://generativelanguage.googleapis.com/v1beta/{target_name}?key={self.api_key}")
        for _, fs_name in few_shot_uploads:
            requests.delete(f"https://generativelanguage.googleapis.com/v1beta/{fs_name}?key={self.api_key}")

        if resp_gen.status_code != 200:
            return {"timing": "UNKNOWN", "confidence": "LOW", "reasoning": f"Gemini Generation Error: {resp_gen.text}"}

        try:
            res_text = resp_gen.json()["candidates"][0]["content"]["parts"][0]["text"]
            return self._normalize_grade(self._parse_json_response(res_text))
        except Exception as e:
            return {
                "phase": "UNKNOWN",
                "timing": "UNKNOWN",
                "confidence": "LOW",
                "reasoning": f"Failed parsing Gemini candidate: {e}",
            }


# ---------------------------------------------------------------------------
# Data and Ground Truth Loading Helper
# ---------------------------------------------------------------------------

def load_ground_truth() -> Dict[Tuple[str, int], str]:
    """Load manual timing classifications from foul_type_classifications.csv."""
    gt_path = config.DATA_DIR / "foul_type_classifications.csv"
    if not gt_path.exists():
        return {}
    
    df = pd.read_csv(gt_path, low_memory=False)
    df = df.dropna(subset=["timing"])
    
    # Map (game_id, event_id) -> timing
    # Pad game_id with leading zeros if it's numeric
    gt = {}
    for _, row in df.iterrows():
        gid = str(row["game_id"]).zfill(10)
        eid = int(row["event_id"])
        gt[(gid, eid)] = row["timing"].strip()
    return gt


def load_manifest(player: str, season_type: str = "Regular Season") -> Dict[str, Any]:
    """Load player manifest json. Checks for PO-specific manifest when season_type is Playoffs."""
    slug = config.player_slug(player)
    suffix = "_po" if season_type == "Playoffs" else ""
    manifest_path = config.PROCESSED_DIR / f"foul_type_manifest_{slug}{suffix}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}\nRun the appropriate foul-type-scrape target first.")
    with open(manifest_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multimodal LLM video foul timing grader")
    parser.add_argument("--player", required=True, help="Player name (must match config.py)")
    parser.add_argument("--provider", required=True, choices=["openai", "anthropic", "gemini", "vertex"], help="LLM Provider")
    parser.add_argument("--model", required=True, help="Model name (e.g. gpt-5.4-mini, claude-sonnet-4-6, gemini-2.5-flash)")
    parser.add_argument("--season-type", default="Regular Season", help="Regular Season or Playoffs (selects the correct manifest)")
    parser.add_argument("--limit", type=int, default=None, help="Limit grading to first N clips")
    parser.add_argument("--validate-only", action="store_true", help="Only grade clips that have manual ground truth in foul_type_classifications.csv")
    parser.add_argument("--direct-timing", action="store_true", help="Use direct-timing prompt mode (model outputs BEFORE/DURING/AFTER directly)")
    parser.add_argument("--sequence", action="store_true", help="Use event-ordering prompt mode (model identifies observable events and their temporal order)")
    parser.add_argument("--legacy-prompt", action="store_true", help="Use legacy 13-field observation prompt (default is 3-field observation prompt)")
    parser.add_argument("--few-shot", action="store_true", help="Include few-shot video examples from ground truth in the prompt")
    args = parser.parse_args()

    # Determine prompt mode
    if args.sequence:
        prompt_mode = "sequence"
    elif args.direct_timing:
        prompt_mode = "direct"
    elif args.legacy_prompt:
        prompt_mode = "legacy"
    else:
        prompt_mode = "observation"

    # Get API key from env (vertex uses gcloud ADC — no key needed)
    key_env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY"
    }

    if args.provider == "vertex":
        api_key = ""
    else:
        env_var = key_env_map[args.provider]
        api_key = os.getenv(env_var) or (os.getenv("GOOGLE_API_KEY") if args.provider == "gemini" else None)

        if not api_key:
            print(f"Error: API key for {args.provider} not found in environment ({env_var}).")
            sys.exit(1)

    # Load data (needed before grader init for few-shot examples)
    manifest = load_manifest(args.player, args.season_type)
    ground_truth = load_ground_truth()

    clips = manifest.get("clips", [])

    # Select few-shot examples if requested
    few_shot_examples: List[FewShotExample] = []
    if args.few_shot:
        few_shot_examples = select_few_shot_examples(clips, ground_truth)
        if not few_shot_examples:
            print("WARNING: --few-shot specified but no ground-truth examples found for few-shot selection.")

    # Initialize Grader
    if args.provider == "openai":
        grader = OpenAIGrader(api_key, args.model, prompt_mode=prompt_mode, few_shot_examples=few_shot_examples)
    elif args.provider == "anthropic":
        grader = AnthropicGrader(api_key, args.model, prompt_mode=prompt_mode, few_shot_examples=few_shot_examples)
    elif args.provider == "vertex":
        grader = VertexGeminiGrader(args.model, prompt_mode=prompt_mode, few_shot_examples=few_shot_examples)
    else:
        grader = GeminiGrader(api_key, args.model, prompt_mode=prompt_mode, few_shot_examples=few_shot_examples)

    if args.validate_only:
        clips = [c for c in clips if (str(c['game_id']).zfill(10), int(c['event_id'])) in ground_truth]
        if not clips:
            print("ERROR: --validate-only specified but no clips match ground truth in foul_type_classifications.csv")
            sys.exit(1)

    if args.limit:
        clips = clips[:args.limit]

    gt_count = sum(1 for c in clips if (str(c['game_id']).zfill(10), int(c['event_id'])) in ground_truth)

    print(f"\n" + "="*70)
    print(f"LLM VIDEO GRADER RUN: {args.player}")
    print(f"Provider:  {args.provider.upper()} ({args.model})")
    print(f"Prompt mode: {prompt_mode}")
    print(f"Few-shot:  {len(few_shot_examples)} examples" if few_shot_examples else "Few-shot:  off")
    print(f"Clips:     {len(clips)}{' (validate-only)' if args.validate_only else ' from manifest'}")
    print(f"Ground Truth matches: {gt_count} / {len(clips)}")
    print("="*70 + "\n")

    results = []
    temp_video_dir = tempfile.mkdtemp()
    # NBA CDN requires browser-like headers; plain requests.get returns a placeholder MP4.
    video_session = NBAStatsClient().session
    
    # Store validation analytics
    val_comparisons = []

    try:
        for idx, c in enumerate(tqdm(clips, desc="Grading clips")):
            game_id = str(c["game_id"]).zfill(10)
            event_id = int(c["event_id"])
            video_url = c.get("video_url_720") or c.get("video_url_960")
            description = c["description"]
            
            # Download video locally
            local_video_path = os.path.join(temp_video_dir, f"clip_{game_id}_{event_id}.mp4")
            try:
                resp = video_session.get(video_url, timeout=30)
                if resp.status_code == 200:
                    with open(local_video_path, "wb") as f:
                        f.write(resp.content)
                else:
                    logger.warning("Failed downloading video %s: HTTP %d", video_url, resp.status_code)
                    continue
            except Exception as e:
                logger.warning("Error downloading video %s: %s", video_url, e)
                continue
            
            # Run grading
            grade = grader.grade_clip(local_video_path, description)
            
            # Record result — include all schema fields (new and legacy)
            res_entry = {
                "game_id": game_id,
                "event_id": event_id,
                "description": description,
                "predicted_phase": grade.get("phase", "UNKNOWN"),
                "predicted_timing": grade.get("timing", "UNKNOWN"),
                "confidence": grade.get("confidence", "LOW"),
                "reasoning": grade.get("reasoning", ""),
                "opponent": c["opponent"],
            }
            # Sequence schema fields
            for k in ("narrative", "first_event", "contact_vs_arm_extend"):
                if k in grade:
                    res_entry[k] = grade[k]
            # New 3-field schema
            for k in ("who_initiated", "ball_state_at_contact", "arm_geometry"):
                if k in grade:
                    res_entry[k] = grade[k]
            # Legacy 13-field schema
            for k in (
                "shooter_initiates_contact", "defender_actively_swiped", "hook_or_bar_arm",
                "both_hands_on_vertical_arc", "rising_without_release", "ball_location_at_contact",
                "arm_direction_at_contact", "shooter_body_state", "contact_initiator",
                "gather_complete_at_contact", "ball_released_before_contact", "could_still_abort",
            ):
                if k in grade:
                    res_entry[k] = grade[k]
            
            # Cross-reference with Ground Truth
            gt_key = (game_id, event_id)
            if gt_key in ground_truth:
                gt_timing = ground_truth[gt_key]
                res_entry["ground_truth_timing"] = gt_timing
                res_entry["ground_truth_phase"] = timing_to_binary_phase(gt_timing)
                val_entry = {
                    "game_id": game_id,
                    "event_id": event_id,
                    "gt": gt_timing,
                    "gt_phase": timing_to_binary_phase(gt_timing),
                    "pred": grade.get("timing", "UNKNOWN"),
                    "pred_phase": grade.get("phase", "UNKNOWN"),
                    "reasoning": grade.get("reasoning", ""),
                    "desc": description,
                }
                # Include all observation fields that are present
                for k in (
                    "narrative", "first_event", "contact_vs_arm_extend",
                    "who_initiated", "ball_state_at_contact", "arm_geometry",
                    "shooter_initiates_contact", "defender_actively_swiped", "hook_or_bar_arm",
                    "both_hands_on_vertical_arc", "rising_without_release", "ball_location_at_contact",
                    "arm_direction_at_contact", "shooter_body_state", "contact_initiator",
                    "gather_complete_at_contact", "ball_released_before_contact", "could_still_abort",
                ):
                    if k in grade:
                        val_entry[k] = grade[k]
                val_comparisons.append(val_entry)
            
            results.append(res_entry)
            
    finally:
        shutil.rmtree(temp_video_dir, ignore_errors=True)

    # Save results
    slug = config.player_slug(args.player)
    out_path = config.PROCESSED_DIR / f"foul_type_llm_results_{slug}.json"
    
    output_payload = {
        "player": args.player,
        "provider": args.provider,
        "model": args.model,
        "prompt_mode": prompt_mode,
        "few_shot_count": len(few_shot_examples),
        "timestamp": pd.Timestamp.now().isoformat(),
        "num_graded": len(results),
        "results": results
    }
    with open(out_path, "w") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n" + "="*70)
    print(f"RUN COMPLETE. Results saved to {out_path}")
    print("="*70 + "\n")

    # Display Validation Analytics
    if val_comparisons:
        vdf = pd.DataFrame(val_comparisons)
        correct_3way = (vdf["gt"] == vdf["pred"]).sum()
        accuracy_3way = correct_3way / len(vdf)

        binary_df = vdf[vdf["gt_phase"].notna() & vdf["pred_phase"].isin(["PRE_COMMIT", "IN_ACT"])].copy()
        if len(binary_df) > 0:
            binary_correct = (binary_df["gt_phase"] == binary_df["pred_phase"]).sum()
            binary_accuracy = binary_correct / len(binary_df)
        else:
            binary_correct = 0
            binary_accuracy = 0.0
        
        print("*"*70)
        print("VALIDATION ANALYSIS vs MANUAL GROUND TRUTH (accuracy check)")
        print("*"*70)
        print(f"Prompt mode: {prompt_mode}")
        print(f"Matched comparisons: {len(vdf)}")
        print(f"Binary phase matches (PRE_COMMIT/IN_ACT): {binary_correct}/{len(binary_df)} ({binary_accuracy:.1%})")
        print(f"Exact 3-way timing matches (derived):   {correct_3way}/{len(vdf)} ({accuracy_3way:.1%})\n")

        print("Binary Phase Breakdown (GT BEFORE→PRE_COMMIT, GT DURING/AFTER→IN_ACT):")
        for gt_phase in ["PRE_COMMIT", "IN_ACT"]:
            subset = binary_df[binary_df["gt_phase"] == gt_phase]
            if len(subset) > 0:
                sub_correct = (subset["gt_phase"] == subset["pred_phase"]).sum()
                sub_acc = sub_correct / len(subset)
                print(f"  {gt_phase:12s} : {sub_correct}/{len(subset)} correct ({sub_acc:.1%})")

        print("\nBinary Phase Confusion Matrix:")
        if len(binary_df) > 0:
            ct_bin = pd.crosstab(binary_df["gt_phase"], binary_df["pred_phase"], margins=True)
            print(ct_bin.to_string())
        
        # Display breakdown
        print("\nDerived 3-way Timing Breakdown:")
        for gt_class in ["BEFORE", "DURING", "AFTER"]:
            subset = vdf[vdf["gt"] == gt_class]
            if len(subset) > 0:
                sub_correct = (subset["gt"] == subset["pred"]).sum()
                sub_acc = sub_correct / len(subset)
                print(f"  {gt_class:8s} : {sub_correct}/{len(subset)} correct ({sub_acc:.1%})")
                
        print("\nDerived 3-way Confusion Matrix:")
        ct = pd.crosstab(vdf["gt"], vdf["pred"], margins=True)
        print(ct.to_string())
        
        print("\nMismatched Cases Detail (binary phase):")
        mismatch = binary_df[binary_df["gt_phase"] != binary_df["pred_phase"]]
        for _, row in mismatch.iterrows():
            print(f"  Clip {row['game_id']}_{row['event_id']}:")
            print(f"    PBP:     {row['desc'][:70]}...")
            print(f"    GT:      {row['gt']} ({row['gt_phase']})")
            print(f"    Pred:    {row['pred_phase']} (derived timing={row['pred']})")
            # Show sequence fields if present
            if "narrative" in row and pd.notna(row.get("narrative")):
                print(f"    Narrative: {row['narrative']}")
            seq_fields = []
            for k in ("first_event", "contact_vs_arm_extend"):
                if k in row and pd.notna(row.get(k)):
                    seq_fields.append(f"{k}={row[k]}")
            if seq_fields:
                print(f"    Sequence: {', '.join(seq_fields)}")
            # Show whichever observation fields are present
            new_fields = []
            for k in ("who_initiated", "ball_state_at_contact", "arm_geometry"):
                if k in row and pd.notna(row.get(k)):
                    new_fields.append(f"{k}={row[k]}")
            legacy_fields = []
            for k in (
                "shooter_initiates_contact", "hook_or_bar_arm", "both_hands_on_vertical_arc",
                "rising_without_release", "ball_location_at_contact", "arm_direction_at_contact",
                "shooter_body_state", "contact_initiator", "gather_complete_at_contact",
                "could_still_abort", "ball_released_before_contact",
            ):
                if k in row and pd.notna(row.get(k)):
                    legacy_fields.append(f"{k}={row[k]}")
            if new_fields:
                print(f"    New schema: {', '.join(new_fields)}")
            if legacy_fields:
                print(f"    Legacy schema: {', '.join(legacy_fields)}")
            print(f"    Reason:  {row['reasoning']}")
            print()
            
    else:
        print("No ground truth comparisons were available in this run to validate predictions.")


if __name__ == "__main__":
    main()
