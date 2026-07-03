#!/usr/bin/env bash
# RunPod Run 6 — VideoMAE fine-tune with unfreeze_layers=6
#
# Prerequisites (one-time):
#   1. RunPod account + credits: https://console.runpod.io
#   2. API key: console → Settings → API Keys
#   3. runpodctl config --apiKey YOUR_KEY
#   4. SSH key: runpodctl ssh add-key --keyfile ~/.ssh/id_ed25519.pub
#
# Local artifacts (already on your Mac):
#   landing_foul_clips.zip          (~1.4 GB)
#   data/processed/landing_foul_frames.npz  (~1.7 GB)
#
# Estimated cost: ~$0.30–0.70 on RTX A5000 ($0.27/hr) for setup + ~15 min train.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CLIPS_ZIP="${CLIPS_ZIP:-$ROOT/landing_foul_clips.zip}"
FRAME_CACHE="${FRAME_CACHE:-$ROOT/data/processed/landing_foul_frames.npz}"
ONPOD_SCRIPT="$ROOT/documents/development/runpod-run6-onpod.sh"

GPU_TYPE="${GPU_TYPE:-NVIDIA RTX A5000}"
POD_NAME="${POD_NAME:-ref-ball-run6}"
VOLUME_GB="${VOLUME_GB:-50}"
IMAGE="${IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"

echo "=== ref-ball Run 6 — RunPod launcher ==="
echo "repo: $ROOT"

# --- local checks ---
for f in "$CLIPS_ZIP" "$FRAME_CACHE" "$ONPOD_SCRIPT"; do
  [[ -f "$f" ]] || { echo "MISSING: $f"; exit 1; }
  echo "OK  $f ($(du -h "$f" | cut -f1))"
done

if ! command -v runpodctl &>/dev/null; then
  echo "ERROR: runpodctl not found. brew install runpod/runpodctl/runpodctl"
  exit 1
fi

if ! runpodctl get pod &>/dev/null; then
  echo ""
  echo "runpodctl is not configured. Run:"
  echo "  runpodctl config --apiKey YOUR_RUNPOD_API_KEY"
  exit 1
fi

echo ""
echo "=== Step 1: Create GPU pod ==="
echo "GPU:      $GPU_TYPE"
echo "Image:    $IMAGE"
echo "Volume:   ${VOLUME_GB}GB at /workspace"
echo ""
echo "Creating pod (or re-use an existing one named '$POD_NAME')..."

POD_ID="$(runpodctl get pod 2>/dev/null | awk -v n="$POD_NAME" '$2==n {print $1; exit}')"

if [[ -z "${POD_ID:-}" ]]; then
  runpodctl create pod \
    --name "$POD_NAME" \
    --gpuType "$GPU_TYPE" \
    --imageName "$IMAGE" \
    --volumeSize "$VOLUME_GB" \
    --volumePath /workspace \
    --containerDiskSize 30 \
    --mem 32 \
    --secureCloud
  echo "Waiting 60s for pod to start..."
  sleep 60
  POD_ID="$(runpodctl get pod 2>/dev/null | awk -v n="$POD_NAME" '$2==n {print $1; exit}')"
fi

if [[ -z "${POD_ID:-}" ]]; then
  echo "Could not find pod. Check: runpodctl get pod"
  echo "Or create manually at https://console.runpod.io/deploy"
  exit 1
fi

echo "Pod ID: $POD_ID"
echo ""
echo "=== Step 2: SSH into pod ==="
echo "Open the RunPod console → Pods → $POD_NAME → Connect → SSH"
echo "Typical command (copy from console):"
echo "  ssh \${POD_ID}-<port>@ssh.runpod.io -i ~/.ssh/id_ed25519"
echo ""
echo "=== Step 3: On the pod, run these commands ==="
cat <<'REMOTE'

# --- one-time pod setup ---
cd /workspace
git clone --depth 1 https://github.com/hirememorey/ref-ball.git
cd ref-ball
mkdir -p data/clips data/processed

REMOTE
echo "# Upload from your Mac (new terminal, while pod is running):"
echo "  runpodctl send $CLIPS_ZIP"
echo "  runpodctl send $FRAME_CACHE"
echo "# On the pod, move received files:"
echo "  unzip -q ~/landing_foul_clips.zip -d data/clips/"
echo "  mv ~/landing_foul_frames.npz data/processed/"
echo ""
echo "# Or use SCP (SSH command from RunPod console):"
echo "  scp -i ~/.ssh/id_ed25519 $CLIPS_ZIP ssh.runpod.io:~/"
echo "  scp -i ~/.ssh/id_ed25519 $FRAME_CACHE ssh.runpod.io:~/"
echo ""
echo "=== Step 4: Launch Run 6 training on pod ==="
echo "  bash documents/development/runpod-run6-onpod.sh"
echo ""
echo "=== Step 5: Download results to Mac ==="
echo "  scp -i ~/.ssh/id_ed25519 ssh.runpod.io:~/ref-ball/data/processed/landing_foul_video_best.pt $ROOT/data/processed/"
echo "  scp -i ~/.ssh/id_ed25519 ssh.runpod.io:~/ref-ball/data/processed/landing_foul_video_metrics.json $ROOT/data/processed/"
echo ""
echo "=== Step 6: Stop pod (avoid idle charges) ==="
echo "  runpodctl stop pod $POD_ID"
echo ""
echo "Tip: skip frame-cache rebuild — landing_foul_frames.npz is reused (anchor_half_width=0.10)."
