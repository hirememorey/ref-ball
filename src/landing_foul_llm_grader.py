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
from src.nba_client import NBAStatsClient
from src.foul_type_llm_grader import (
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
# ---------------------------------------------------------------------------

# SPATIAL_LANDING_PROMPT — the primary prompt. Asks for the spatial
# observations that determine whether a foul is a landing foul, plus a direct
# YES/NO/UNCLEAR classification. The model's `landing_foul` field is the
# prediction; the observations anchor its reasoning and aid mismatch review.
SPATIAL_LANDING_PROMPT = """You are an expert NBA officiating analyst. Watch a short video clip of a shooting foul and determine whether it is a LANDING FOUL.

DEFINITION (landing foul): A foul where the defender's feet or body are under or moving into the shooter's landing zone WHILE THE SHOOTER IS AIRBORNE on a jump shot, and the foul is called because of that positioning / the contact that occurs as the shooter comes down. The shooter must be in the air on a jump shot (typically a perimeter/wing 3-point attempt) and the illegal contact is tied to the landing space — the defender undercut or stepped under the airborne shooter.

NOT a landing foul (answer NO):
  - Standard arm/hand contest on the shot (defender swipes or reaches the arm while the shooter is going up or at the release, feet in legal position).
  - Contact on a drive to the rim or layup (shooter not in an airborne jump shot, or contact is body-to-body on the drive).
  - The shooter initiates contact (jumping into / leaning into a stationary defender, rip-through, pump-fake jump-into).
  - Off-ball or screen contact unrelated to the shooter's landing zone.
  - Reach-in or body foul before the shooter leaves the ground.

KEY SPATIAL CHECK (watch the clip twice):
  1. Is the shooter airborne on a JUMP SHOT (both feet off the floor, rising/falling on a perimeter shot)? If it is a drive/layup, the answer is almost certainly NO.
  2. Where are the defender's FEET/BODY as the shooter DESCENDS to land? Are they under the shooter's landing spot (undercut / stepped into the landing zone), or is the defender to the side / in legal position?
  3. WHEN does the illegal contact happen — during the shooter's downward descent / landing (after the release), or during the upward shot motion / at the release (a normal contest)?
  4. Is the WHISTLE for the landing-zone contact, or for a routine arm contest on the shot?

A landing foul requires: airborne jump shot + defender in/under the landing zone + contact tied to the descent/landing. If the contact is a standard arm contest on the shot release with the defender's feet in legal position, answer NO even if the call was made.

Answer these observation questions, then the classification:

{
  "shot_type": "JUMP_SHOT" | "DRIVE" | "OTHER" | "UNCLEAR",
  "defender_position_at_landing": "UNDER_SHOOTER" | "NEAR_BUT_LEGAL" | "NOT_AT_LANDING" | "UNCLEAR",
  "contact_moment": "DURING_DESCENT_OR_LANDING" | "DURING_SHOT_MOTION" | "BEFORE_SHOT" | "UNCLEAR",
  "landing_foul": "YES" | "NO" | "UNCLEAR",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "reasoning": "One sentence citing shot type, defender feet position at landing, and when contact occurred."
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

    prompt_mode: str  # "spatial" | "sequence"

    def _landing_system_prompt(self) -> str:
        if self.prompt_mode == "sequence":
            return SEQUENCE_LANDING_PROMPT
        return SPATIAL_LANDING_PROMPT

    def _normalize_landing(self, grade: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize the model's landing response.

        Trusts the model's `landing_foul` field; derives it from observations
        only if missing. Always uppercases and ensures the field is present.
        """
        label = str(grade.get("landing_foul", "")).strip().upper()
        if label not in ("YES", "NO", "UNCLEAR"):
            # Derive from observations when the model omits the direct label.
            shot = str(grade.get("shot_type", "")).upper()
            pos = str(grade.get("defender_position_at_landing", "")).upper()
            moment = str(grade.get("contact_moment", "")).upper()
            desc = str(grade.get("contact_vs_descent", "")).upper()
            in_zone = str(grade.get("defender_in_landing_zone", "")).upper()
            descent_contact = moment in ("DURING_DESCENT_OR_LANDING",) or desc in (
                "DURING_DESCENT_OR_LANDING",
            )
            if shot == "DRIVE" or shot == "OTHER":
                label = "NO"
            elif pos == "UNDER_SHOOTER" and descent_contact:
                label = "YES"
            elif in_zone == "YES" and descent_contact:
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
        if self.prompt_mode == "sequence":
            return (
                f"\n\nPlay-by-play description: {description}\n\n"
                "Watch the clip above. Determine whether this is a landing foul "
                "using the event-ordering steps. Return a raw JSON object."
            )
        return (
            f"\n\nPlay-by-play description: {description}\n\n"
            "Watch the clip above. Apply the spatial check (shot type, defender "
            "feet at landing, when contact occurs). Return a raw JSON object."
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

            instruction = (
                f"Play-by-play description: {description}\n\n"
                "Analyze the chronological sequence of frames below and determine "
                "whether this is a landing foul using the spatial check."
            )
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
            prompt = (
                f"Play-by-play description: {description}\n\n"
                "Analyze the chronological sequence of frames above and determine "
                "whether this is a landing foul using the spatial check."
            )
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
        for k in ("shot_type", "defender_position_at_landing", "contact_moment",
                  "defender_in_landing_zone", "contact_vs_descent", "narrative"):
            if k in row and pd.notna(row.get(k)):
                val = str(row[k])
                obs_fields.append(f"{k}={val[:80]}")
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
    parser.add_argument("--prompt-mode", default="spatial", choices=["spatial", "sequence"],
                        help="Prompt strategy: spatial (default) or event-ordering sequence fallback")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only grade clips that have manual ground truth")
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

    # Only keep keys that have a resolvable video URL.
    keys = [k for k in keys if clips_by_key.get(k) and _clip_video_url(clips_by_key[k])]

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
    print(f"Clips:       {len(keys)}{' (validate-only)' if args.validate_only else ' (full manifest)'}")
    print(f"Extended:    {args.extended}   Include UNCLEAR: {args.include_unclear}")
    print(f"Ground truth matches: {gt_count} / {len(keys)}")
    print("=" * 70 + "\n")

    results: List[Dict[str, Any]] = []
    val_comparisons: List[Dict[str, Any]] = []
    temp_video_dir = tempfile.mkdtemp()
    video_session = NBAStatsClient().session

    try:
        for key in tqdm(keys, desc="Grading clips"):
            game_id, event_id = key
            clip = clips_by_key[key]
            video_url = _clip_video_url(clip)
            description = clip.get("description", "")
            local_video_path = os.path.join(temp_video_dir, f"clip_{game_id}_{event_id}.mp4")
            try:
                resp = video_session.get(video_url, timeout=30)
                if resp.status_code == 200:
                    with open(local_video_path, "wb") as f:
                        f.write(resp.content)
                else:
                    logger.warning("Failed downloading video %s: HTTP %d", video_url, resp.status_code)
                    continue
            except Exception as exc:
                logger.warning("Error downloading video %s: %s", video_url, exc)
                continue

            grade = grader.grade_clip(local_video_path, description)

            res_entry: Dict[str, Any] = {
                "game_id": game_id,
                "event_id": event_id,
                "description": description,
                "predicted_landing_foul": grade.get("landing_foul", "UNCLEAR"),
                "confidence": grade.get("confidence", "LOW"),
                "reasoning": grade.get("reasoning", ""),
                "caller_official_name": clip.get("caller_official_name", ""),
            }
            for k in ("shot_type", "defender_position_at_landing", "contact_moment",
                      "defender_in_landing_zone", "contact_vs_descent", "narrative"):
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
                       "narrative") if k in grade},
                })
    finally:
        shutil.rmtree(temp_video_dir, ignore_errors=True)

    # Save results
    out_path = Path(args.output) if args.output else (
        config.PROCESSED_DIR / f"landing_foul_llm_results_{args.provider}_{args.model.replace('.', '_')}.json"
    )
    output_payload = {
        "task": "landing_foul",
        "provider": args.provider,
        "model": args.model,
        "prompt_mode": args.prompt_mode,
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
