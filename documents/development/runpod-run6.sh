#!/usr/bin/env bash
# RunPod Run 6 — VideoMAE fine-tune with unfreeze_layers=6
#
# Modes:
#   bash documents/development/runpod-run6.sh --drive
#     Start pod, pull clips (+ optional frames) from Google Drive on-pod, train.
#     Requires documents/development/runpod-drive.env with share links.
#
#   bash documents/development/runpod-run6.sh
#     Legacy: create pod + print Mac upload / SSH steps.
#
# Prerequisites (one-time):
#   runpodctl config --apiKey YOUR_KEY
#   runpodctl ssh add-key --keyfile ~/.ssh/id_ed25519.pub
#
# Estimated cost (drive mode): ~$0.20–0.40 on RTX A5000 ($0.27/hr) — GPU only while downloading + training.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

DRIVE_MODE=0
AUTO_RUN=0
for arg in "$@"; do
  case "$arg" in
    --drive) DRIVE_MODE=1 ;;
    --run)   DRIVE_MODE=1; AUTO_RUN=1 ;;
  esac
done

CLIPS_ZIP="${CLIPS_ZIP:-$ROOT/landing_foul_clips.zip}"
FRAME_CACHE="${FRAME_CACHE:-$ROOT/data/processed/landing_foul_frames.npz}"
ONPOD_SCRIPT="$ROOT/documents/development/runpod-run6-onpod.sh"
DRIVE_FETCH_SCRIPT="$ROOT/documents/development/runpod-run6-drive-fetch.sh"
DRIVE_ENV="$ROOT/documents/development/runpod-drive.env"
SSH_EXP="$ROOT/documents/development/runpod-ssh.exp"

GPU_TYPE="${GPU_TYPE:-NVIDIA RTX A5000}"
POD_NAME="${POD_NAME:-ref-ball-run6}"
VOLUME_GB="${VOLUME_GB:-50}"
IMAGE="${IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"

pod_status() {
  runpodctl get pod "$1" 2>/dev/null | awk -F'\t' 'NR==2 {
    for (i = NF; i >= 1; i--) if ($i != "") { print $i; exit }
  }'
}

pod_ssh_host() {
  local pod_id="$1"
  local attempt host
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    host="$(python3 - <<PY
import json, os, sys, time
pod_id = "$pod_id"
apikey = open(os.path.expanduser("~/.runpod/config.toml")).read().split('apikey = "')[1].split('"')[0]
q = '''query { pod(input: {podId: "%s"}) { machine { podHostId } desiredStatus } }''' % pod_id
import urllib.request
req = urllib.request.Request(
    "https://api.runpod.io/graphql",
    data=json.dumps({"query": q}).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + apikey},
)
try:
    resp = json.load(urllib.request.urlopen(req))
except Exception as e:
    print("", end="")
    sys.exit(0)
host = resp.get("data", {}).get("pod", {}).get("machine", {}).get("podHostId") or ""
print(host)
PY
)"
    if [[ -n "$host" ]]; then
      echo "$host"
      return 0
    fi
    echo "Waiting for pod SSH host (attempt $attempt)..." >&2
    sleep 15
  done
  echo "Could not resolve podHostId (is the pod running?)" >&2
  return 1
}

ensure_pod_running() {
  local pod_id="$1"
  local status
  status="$(pod_status "$pod_id")"
  if [[ "$status" == "EXITED" ]]; then
    echo "Starting pod $pod_id ..."
    if ! runpodctl start pod "$pod_id" 2>&1; then
      echo "Start failed on this host — recreate the pod to pick a new machine."
      return 1
    fi
    echo "Waiting 60s for pod to boot..."
    sleep 60
  elif [[ "$status" != "RUNNING" ]]; then
    echo "Pod status: ${status:-unknown}. Waiting 60s..."
    sleep 60
  fi
}

create_runpod() {
  local gpu="$GPU_TYPE"
  local cloud
  while true; do
    for cloud in secure community; do
      echo "Creating pod '$POD_NAME' ($gpu, ${cloud} cloud)..."
      if [[ "$cloud" == "secure" ]]; then
        runpodctl create pod \
          --name "$POD_NAME" \
          --gpuType "$gpu" \
          --imageName "$IMAGE" \
          --volumeSize "$VOLUME_GB" \
          --volumePath /workspace \
          --containerDiskSize 30 \
          --mem 32 \
          --secureCloud 2>&1 && return 0
      else
        runpodctl create pod \
          --name "$POD_NAME" \
          --gpuType "$gpu" \
          --imageName "$IMAGE" \
          --volumeSize "$VOLUME_GB" \
          --volumePath /workspace \
          --containerDiskSize 30 \
          --mem 32 \
          --communityCloud 2>&1 && return 0
      fi
      echo "${cloud} cloud unavailable for $gpu"
    done
    if [[ "$gpu" == "NVIDIA RTX A5000" ]]; then
      gpu="NVIDIA GeForce RTX 3090"
      echo "Trying fallback GPU: $gpu"
    else
      echo "ERROR: could not create pod on any GPU type"
      return 1
    fi
  done
}

echo "=== ref-ball Run 6 — RunPod launcher ==="
echo "repo: $ROOT"
echo "mode: $([[ "$DRIVE_MODE" == 1 ]] && echo 'drive (Google Drive → pod)' || echo 'manual upload')"

if [[ "$DRIVE_MODE" == 1 ]]; then
  if [[ -f "$DRIVE_ENV" ]]; then
    # shellcheck disable=SC1090
    source "$DRIVE_ENV"
    echo "OK  loaded $DRIVE_ENV"
  else
    echo "ERROR: $DRIVE_ENV not found."
    echo "  cp documents/development/runpod-drive.env.example documents/development/runpod-drive.env"
    echo "  Then paste Google Drive share links for landing_foul_clips.zip (required)."
    exit 1
  fi
  if [[ -z "${GDRIVE_CLIPS_URL:-}" || "$GDRIVE_CLIPS_URL" == *YOUR_CLIPS* ]]; then
    echo "ERROR: set GDRIVE_CLIPS_URL in $DRIVE_ENV"
    exit 1
  fi
  for f in "$ONPOD_SCRIPT" "$DRIVE_FETCH_SCRIPT" "$SSH_EXP"; do
    [[ -f "$f" ]] || { echo "MISSING: $f"; exit 1; }
  done
else
  for f in "$CLIPS_ZIP" "$FRAME_CACHE" "$ONPOD_SCRIPT"; do
    [[ -f "$f" ]] || { echo "MISSING: $f"; exit 1; }
    echo "OK  $f ($(du -h "$f" | cut -f1))"
  done
fi

if ! command -v runpodctl &>/dev/null; then
  echo "ERROR: runpodctl not found. brew install runpod/runpodctl/runpodctl"
  exit 1
fi

if ! runpodctl get pod &>/dev/null; then
  echo "runpodctl is not configured. Run: runpodctl config --apiKey YOUR_RUNPOD_API_KEY"
  exit 1
fi

echo ""
echo "=== Step 1: Create / resume GPU pod ==="
POD_ID="$(runpodctl get pod 2>/dev/null | awk -v n="$POD_NAME" '$2==n {print $1; exit}')"

if [[ -z "${POD_ID:-}" ]]; then
  create_runpod
  echo "Waiting 60s for pod to start..."
  sleep 60
  POD_ID="$(runpodctl get pod 2>/dev/null | awk -F'\t' -v n="$POD_NAME" '$2==n {print $1; exit}')"
fi

if [[ -z "${POD_ID:-}" ]]; then
  echo "Could not find pod. Check: runpodctl get pod"
  exit 1
fi

echo "Pod ID: $POD_ID"

if [[ "$DRIVE_MODE" == 1 ]]; then
  if ! ensure_pod_running "$POD_ID"; then
    echo "Recreating pod on a new host..."
    runpodctl remove pod "$POD_ID" 2>/dev/null || true
    create_runpod
    sleep 60
    POD_ID="$(runpodctl get pod 2>/dev/null | awk -F'\t' -v n="$POD_NAME" '$2==n {print $1; exit}')"
    ensure_pod_running "$POD_ID"
  fi
  POD_HOST="$(pod_ssh_host "$POD_ID")"
  echo "SSH host: $POD_HOST"

  echo ""
  echo "=== Step 2: Clone / update repo on pod ==="
  "$SSH_EXP" "$POD_HOST" \
    "if [[ ! -d /workspace/ref-ball/.git ]]; then git clone --depth 1 https://github.com/hirememorey/ref-ball.git /workspace/ref-ball; else cd /workspace/ref-ball && git pull --ff-only origin main; fi" \
    600

  REMOTE_ENV="FETCH_FROM_DRIVE=1 GDRIVE_CLIPS_URL=$(printf '%q' "$GDRIVE_CLIPS_URL")"
  if [[ -n "${GDRIVE_FRAMES_URL:-}" ]]; then
    REMOTE_ENV="$REMOTE_ENV GDRIVE_FRAMES_URL=$(printf '%q' "$GDRIVE_FRAMES_URL")"
  fi
  if [[ -n "${BUILD_FRAME_CACHE:-}" ]]; then
    REMOTE_ENV="$REMOTE_ENV BUILD_FRAME_CACHE=$(printf '%q' "$BUILD_FRAME_CACHE")"
  fi
  if [[ -n "${UNFREEZE_LAYERS:-}" ]]; then
    REMOTE_ENV="$REMOTE_ENV UNFREEZE_LAYERS=$(printf '%q' "$UNFREEZE_LAYERS")"
  fi

  REMOTE_CMD="cd /workspace/ref-ball && $REMOTE_ENV bash documents/development/runpod-run6-onpod.sh"

  echo ""
  echo "=== Step 3: Drive fetch + Run 6 training on pod ==="
  if [[ "$AUTO_RUN" == 1 ]]; then
    "$SSH_EXP" "$POD_HOST" "$REMOTE_CMD" 14400
    echo ""
    echo "=== Step 4: Stop pod (avoid idle charges) ==="
    runpodctl stop pod "$POD_ID"
    echo "Stopped $POD_ID"
    echo ""
    echo "Download results:"
    echo "  scp -i ~/.ssh/id_ed25519 ${POD_HOST}@ssh.runpod.io:/workspace/ref-ball/data/processed/landing_foul_video_best.pt $ROOT/data/processed/"
    echo "  scp -i ~/.ssh/id_ed25519 ${POD_HOST}@ssh.runpod.io:/workspace/ref-ball/data/processed/landing_foul_video_metrics.json $ROOT/data/processed/"
  else
    echo "Pod is running. To start training:"
    echo "  bash documents/development/runpod-run6.sh --drive --run"
    echo ""
    echo "Or SSH manually:"
    echo "  ssh ${POD_HOST}@ssh.runpod.io -i ~/.ssh/id_ed25519"
    echo "  $REMOTE_ENV bash documents/development/runpod-run6-onpod.sh"
    echo ""
    echo "Stop when done: runpodctl stop pod $POD_ID"
  fi
  exit 0
fi

echo ""
echo "=== Step 2: SSH into pod ==="
echo "Open RunPod console → Pods → $POD_NAME → Connect → SSH"
echo ""
echo "=== Step 3: On the pod ==="
cat <<'REMOTE'

cd /workspace
git clone --depth 1 https://github.com/hirememorey/ref-ball.git
cd ref-ball
mkdir -p data/clips data/processed

REMOTE
echo "# Upload from Mac (legacy — prefer --drive mode):"
echo "  bash documents/development/runpod-run6-transfer.sh"
echo ""
echo "=== Step 4: Launch Run 6 training on pod ==="
echo "  bash documents/development/runpod-run6-onpod.sh"
echo ""
echo "=== Step 5: Stop pod ==="
echo "  runpodctl stop pod $POD_ID"
