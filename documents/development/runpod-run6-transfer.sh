#!/usr/bin/env bash
# Transfer frames.npz + clips.zip to RunPod pod via croc/runpodctl.
# Usage: POD_HOST=twf1cuwlsmkeos-644113a7 bash documents/development/runpod-run6-transfer.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

POD_HOST="${POD_HOST:-twf1cuwlsmkeos-644113a7}"
SSH_EXP="$ROOT/documents/development/runpod-ssh.exp"
FRAMES="$ROOT/data/processed/landing_foul_frames.npz"
CLIPS="$ROOT/landing_foul_clips.zip"

transfer_file() {
  local code="$1"
  local local_path="$2"
  local remote_cmd="$3"
  local timeout="$4"
  local name
  name="$(basename "$local_path")"
  echo ""
  echo "=== Transfer $name via croc (code=$code) ==="
  echo "Local: $local_path ($(du -h "$local_path" | cut -f1))"
  # Receiver blocks until send completes.
  "$SSH_EXP" "$POD_HOST" "$remote_cmd" "$timeout" &
  local recv_pid=$!
  sleep 5
  runpodctl send --code "$code" "$local_path"
  wait "$recv_pid"
  echo "=== Done: $name ==="
}

transfer_file "refball-frames-6" "$FRAMES" \
  "cd /workspace/ref-ball/data/processed && croc --yes refball-frames-6" 7200

transfer_file "refball-clips-6" "$CLIPS" \
  "cd /workspace/ref-ball && croc --yes refball-clips-6" 7200

echo ""
echo "=== Extract clips + verify on pod ==="
"$SSH_EXP" "$POD_HOST" \
  "cd /workspace/ref-ball && unzip -oq landing_foul_clips.zip -d data/clips/ && ls -lh landing_foul_clips.zip data/processed/landing_foul_frames.npz && find data/clips/landing_foul -name '*.mp4' | wc -l" \
  1800

echo "Transfer complete."
