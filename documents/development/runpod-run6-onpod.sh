#!/usr/bin/env bash
# Run **inside** a RunPod GPU pod after clips + frame cache are on disk.
# Called by runpod-run6.sh or manually after SSH.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/ref-ball}"
cd "$REPO_DIR"
export PYTHONPATH=.

echo "=== Run 6 on-pod setup ==="
echo "cwd: $(pwd)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python3 -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# --- verify artifacts ---
need_ok=true
for f in \
  src/landing_foul_video_finetune.py \
  data/landing_foul_ground_truth.csv \
  data/processed/landing_foul_split.json \
  data/processed/landing_foul_clip_anchors.json \
  data/processed/landing_foul_frames.npz; do
  if [[ -f "$f" ]]; then
    echo "OK  $f"
  else
    echo "MISSING $f"
    need_ok=false
  fi
done

n_clips=$(find data/clips/landing_foul -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')
echo "clips: ${n_clips} mp4 (expect 284)"
if [[ "$n_clips" -lt 280 ]]; then
  echo "ERROR: extract landing_foul_clips.zip into data/clips/ first"
  need_ok=false
fi

if [[ "$need_ok" != true ]]; then
  exit 1
fi

# --- deps (torch usually pre-installed on RunPod PyTorch images) ---
pip install -q -U transformers opencv-python-headless tqdm scikit-learn

# --- Run 6 hyperparameters (HANDOFF.md Step 10b) ---
UNFREEZE_LAYERS="${UNFREEZE_LAYERS:-6}"
ANCHOR_HALF_WIDTH="${ANCHOR_HALF_WIDTH:-0.10}"
YES_WEIGHT="${YES_WEIGHT:-0.85}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-5}"
HEAD_EPOCHS="${HEAD_EPOCHS:-5}"
SEED="${SEED:-42}"

echo ""
echo "=== Run 6 fine-tune | unfreeze_layers=$UNFREEZE_LAYERS ==="
python3 src/landing_foul_video_finetune.py \
  --phase two-phase \
  --head-epochs "$HEAD_EPOCHS" \
  --finetune-epochs "$FINETUNE_EPOCHS" \
  --head-lr 1e-3 \
  --finetune-lr 2e-5 \
  --unfreeze-layers "$UNFREEZE_LAYERS" \
  --batch-size 4 \
  --temporal-window "0.0,1.0" \
  --anchor-half-width "$ANCHOR_HALF_WIDTH" \
  --jitter 6 \
  --dropout 0.4 \
  --weight-decay 0.01 \
  --yes-weight "$YES_WEIGHT" \
  --patience 6 \
  --seed "$SEED" \
  --device cuda

echo ""
echo "=== RESULT ==="
python3 - <<'PY'
import json
from pathlib import Path
p = Path("data/processed/landing_foul_video_metrics.json")
m = json.loads(p.read_text())
b, g = m["best"], m["gate"]
print(f"best epoch: {b['epoch']} | phase: {b['phase']}")
print(f"val precision (YES): {b['val_precision_yes']:.3f}  (gate >= 0.85)")
print(f"val recall    (YES): {b['val_recall_yes']:.3f}  (gate >= 0.70)")
print(f"confusion: {b['confusion']}")
print(f"gate verdict: {g['verdict']}")
PY

echo ""
echo "Outputs:"
ls -lh data/processed/landing_foul_video_best.pt data/processed/landing_foul_video_metrics.json
