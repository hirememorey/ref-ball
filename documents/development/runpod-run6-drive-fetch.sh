#!/usr/bin/env bash
# Download training artifacts from Google Drive on a RunPod pod (no Mac→pod transfer).
#
# Prereqs:
#   - Files shared as "Anyone with the link" (Viewer is enough).
#   - Set GDRIVE_CLIPS_URL (required) and optionally GDRIVE_FRAMES_URL.
#   - If frames URL is unset and cache is missing, builds cache on-pod (~5–10 min).
#
# Usage (on pod, after clone):
#   export GDRIVE_CLIPS_URL='https://drive.google.com/file/d/.../view?usp=sharing'
#   export GDRIVE_FRAMES_URL='https://drive.google.com/file/d/.../view?usp=sharing'  # optional
#   bash documents/development/runpod-run6-drive-fetch.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/ref-ball}"
cd "$REPO_DIR"
export PYTHONPATH=.

CLIPS_ZIP="${CLIPS_ZIP:-$REPO_DIR/landing_foul_clips.zip}"
FRAMES_NPZ="${FRAMES_NPZ:-$REPO_DIR/data/processed/landing_foul_frames.npz}"
ANCHOR_HALF_WIDTH="${ANCHOR_HALF_WIDTH:-0.10}"
TEMPORAL_WINDOW="${TEMPORAL_WINDOW:-0.0,1.0}"
BUILD_FRAME_CACHE="${BUILD_FRAME_CACHE:-auto}"  # auto | always | never

mkdir -p data/clips data/processed

pip install -q gdown

drive_download() {
  local url="$1"
  local dest="$2"
  if [[ -z "$url" ]]; then
    return 1
  fi
  if [[ -f "$dest" ]]; then
    echo "OK  $dest ($(du -h "$dest" | cut -f1)) — skip download"
    return 0
  fi
  echo "Downloading $(basename "$dest") from Drive..."
  local file_id=""
  if [[ "$url" =~ /d/([a-zA-Z0-9_-]+) ]]; then
    file_id="${BASH_REMATCH[1]}"
  fi
  if gdown --help 2>&1 | grep -q fuzzy; then
    gdown --fuzzy "$url" -O "$dest" || gdown "https://drive.google.com/uc?id=${file_id}" -O "$dest"
  else
    gdown "https://drive.google.com/uc?id=${file_id}" -O "$dest"
  fi
  echo "OK  $dest ($(du -h "$dest" | cut -f1))"
}

clip_count() {
  find data/clips/landing_foul -name '*.mp4' 2>/dev/null | wc -l | tr -d ' '
}

# --- clips zip ---
if [[ -z "${GDRIVE_CLIPS_URL:-}" ]]; then
  echo "ERROR: GDRIVE_CLIPS_URL is required (share link for landing_foul_clips.zip)."
  echo "  Drive path: MyDrive/landing_foul_clips.zip"
  exit 1
fi

drive_download "$GDRIVE_CLIPS_URL" "$CLIPS_ZIP"

n_clips="$(clip_count)"
if [[ "$n_clips" -lt 280 ]]; then
  echo "Extracting $CLIPS_ZIP ..."
  unzip -oq "$CLIPS_ZIP" -d data/clips/
  n_clips="$(clip_count)"
fi
echo "clips: ${n_clips} mp4 (expect 284)"
if [[ "$n_clips" -lt 280 ]]; then
  echo "ERROR: expected ~284 clips after extract; got $n_clips"
  exit 1
fi

# --- frame cache ---
need_build=false
if [[ -f "$FRAMES_NPZ" ]]; then
  echo "OK  $FRAMES_NPZ ($(du -h "$FRAMES_NPZ" | cut -f1))"
elif [[ -n "${GDRIVE_FRAMES_URL:-}" ]]; then
  drive_download "$GDRIVE_FRAMES_URL" "$FRAMES_NPZ"
elif [[ "$BUILD_FRAME_CACHE" == "always" ]]; then
  need_build=true
elif [[ "$BUILD_FRAME_CACHE" == "never" ]]; then
  echo "ERROR: frame cache missing and BUILD_FRAME_CACHE=never"
  exit 1
else
  echo "No GDRIVE_FRAMES_URL — will build frame cache on pod."
  need_build=true
fi

if [[ "$need_build" == true ]]; then
  echo ""
  echo "=== Building frame cache (anchor_half_width=$ANCHOR_HALF_WIDTH) ==="
  pip install -q transformers opencv-python-headless tqdm scikit-learn
  python3 src/landing_foul_video_finetune.py --build-cache \
    --temporal-window "$TEMPORAL_WINDOW" \
    --anchor-half-width "$ANCHOR_HALF_WIDTH" \
    --cache-frames 32 \
    --cache-size 256
  python3 - <<'PY'
import numpy as np
from pathlib import Path
p = Path("data/processed/landing_foul_frames.npz")
d = np.load(p)
print("cache shape:", d["frames"].shape, "| ~%.2f GB" % (d["frames"].nbytes / 1e9))
PY
fi

echo ""
echo "Drive fetch complete."
