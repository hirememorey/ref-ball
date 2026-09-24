"""Gemini native-video run on the video-test sample (REST, no SDK).

Sends each whole L2M clip (about 20 s) inline, sampled at 10 fps with high media resolution.
The frame models saw only 5.5-15.0 s; a first Gemini result showed the play can fall outside
that window, so Gemini gets the full clip. Same blinded rubric as video_test.py: foul type,
the two players and the game clock, never the league decision or comment. Results go to
outputs/gemini/<clip>.json, which `video_test.py score` picks up.

Two routes, chosen with GEMINI_PROVIDER:

    # Gemini API (AI Studio key in GEMINI_API_KEY, or ~/.config/ref-ball/gemini_api_key)
    python3 src/l2m_contact/video_test/gemini_video.py

    # Vertex AI via gcloud ADC (gcloud auth application-default login; project from
    # GOOGLE_CLOUD_PROJECT or `gcloud config get-value project`)
    GEMINI_PROVIDER=vertex python3 src/l2m_contact/video_test/gemini_video.py

    ... gemini_video.py all      # every clip in the sample, not just the 61 Sonnet scored

GEMINI_MODEL (default gemini-3.8-flash) and GEMINI_WORKERS (default 2) override. The free
Gemini API tier allows 20 requests per day per model, and 503 "high demand" errors count.
"""

from __future__ import annotations

import base64
import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import video_test as v

PROVIDER = os.environ.get("GEMINI_PROVIDER", "gemini")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
WORKERS = int(os.environ.get("GEMINI_WORKERS", "2"))
FPS = 10
OUT = v.OUTS / "gemini"

VIDEO_INTRO = ("The video is an NBA broadcast clip of about 20 seconds around one play. "
               "The broadcast game clock is visible in the score bug.")
SCHEMA = {"type": "OBJECT", "required": ["level", "p_illegal", "reason"],
          "properties": {"level": {"type": "STRING", "enum": ["none", "marginal", "illegal", "cannot_see"]},
                         "p_illegal": {"type": "INTEGER"},
                         "reason": {"type": "STRING"}}}
LEVELS = ("none", "marginal", "illegal", "cannot_see")


def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    path = Path.home() / ".config" / "ref-ball" / "gemini_api_key"
    if not key and path.exists():
        key = path.read_text().strip()
    if not key:
        raise SystemExit("Set GEMINI_API_KEY or write the key to ~/.config/ref-ball/gemini_api_key")
    return key


def gcloud(*args: str) -> str:
    return subprocess.run(["gcloud", *args], capture_output=True, text=True, check=True).stdout.strip()


_token: tuple[float, str] = (0.0, "")


def vertex_headers() -> dict:
    global _token
    if time.time() - _token[0] > 1800:  # ADC tokens last about an hour
        _token = (time.time(), gcloud("auth", "application-default", "print-access-token"))
    return {"Authorization": f"Bearer {_token[1]}", "Content-Type": "application/json"}


def endpoint() -> tuple[str, callable]:
    if PROVIDER == "vertex":
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or gcloud("config", "get-value", "project")
        if not project:
            raise SystemExit("No GCP project: set GOOGLE_CLOUD_PROJECT or `gcloud config set project ...`")
        # Gemini 3.x models are served from the global Vertex endpoint in many projects.
        url = (f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/"
               f"publishers/google/models/{MODEL}:generateContent")
        return url, vertex_headers
    key = api_key()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    return url, lambda: {"Content-Type": "application/json", "x-goog-api-key": key}


URL, HEADERS = endpoint()


def prompt_for(r: dict) -> str:
    text = v.prompt_for(r)
    # Same rubric; swap the frame description for a video one.
    return text.replace("The frames are in time order, about 0.5 seconds apart, from the moments around one play.\n"
                        "The broadcast game clock is visible in the score bug.", VIDEO_INTRO)


def run_one(r: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{r['clip_id']}.json"
    if path.exists():
        return json.loads(path.read_text())
    video = (v.CLIPS / f"{r['clip_id']}.mp4").read_bytes()
    body = {
        "contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": "video/mp4", "data": base64.b64encode(video).decode()},
             "videoMetadata": {"fps": FPS}},
            {"text": prompt_for(r)}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "responseSchema": SCHEMA,
                             "mediaResolution": "MEDIA_RESOLUTION_HIGH"},
    }
    data = json.dumps(body).encode()
    for attempt in range(10):
        req = urllib.request.Request(URL, data=data, method="POST", headers=HEADERS())
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                out = json.loads(resp.read())
            res = json.loads(out["candidates"][0]["content"]["parts"][0]["text"])
            if res.get("level") in LEVELS:
                res["p_illegal"] = max(0, min(100, int(res["p_illegal"])))
                res["usage"] = out.get("usageMetadata", {})
                res["provider"], res["model"] = PROVIDER, MODEL
                path.write_text(json.dumps(res))
                return res
            err = f"bad level {res.get('level')}"
        except urllib.error.HTTPError as e:
            body_text = e.read()[:400].decode("utf-8", "replace")
            err = f"HTTP {e.code}: {body_text}"
            if e.code == 429 and "PerDay" in body_text:
                raise RuntimeError(f"{r['clip_id']}: daily quota exhausted; rerun tomorrow (results resume)")
            if e.code not in (429, 500, 503):
                raise RuntimeError(f"{r['clip_id']}: {err}")
        except (KeyError, IndexError, json.JSONDecodeError, OSError) as e:  # OSError covers socket timeouts
            err = str(e)[:300]
        wait = min(180, 15 * 2 ** attempt) + random.uniform(0, 10)
        print(f"  {r['clip_id']} attempt {attempt + 1}: {err[:160]}; retry in {wait:.0f}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"{r['clip_id']}: gave up")


def main() -> None:
    rows = v.rows()
    if sys.argv[1:] != ["all"]:
        # The 61 uncalled clips the Sonnet frame run scored, for a paired comparison.
        scored = {p.stem for p in (v.OUTS / "sonnet").glob("*.json")}
        if not scored and v.OUTPUTS_CSV.exists():  # fresh clone: use the tracked Sonnet answers
            import csv
            scored = {x["clip_id"] for x in csv.DictReader(v.OUTPUTS_CSV.open()) if x["model"] == "sonnet"}
        rows = [r for r in rows if r["test"] == "uncalled" and r["clip_id"] in scored]
    print(f"{len(rows)} clips -> {PROVIDER}:{MODEL}, {FPS} fps, full clip, high media resolution", flush=True)
    done = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(run_one, r) for r in rows]
        for f in as_completed(futs):
            try:
                f.result()
            except RuntimeError as e:
                print("  failed:", e, flush=True)
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
