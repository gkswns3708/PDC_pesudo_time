#!/bin/bash
# Sequential full-data training of 3 backbones for S14-2289-1-6 ensemble inference.
#
# Each backbone trains on all 8 slides (no LOSO), patch-level 10% random val.
# Output: checkpoints/best_model_<backbone>_full.pth + logs/train_full_<backbone>.log
#
# Total time: ~16h (4-6h per model, sequential on 2×L40)
#
# Usage (in tmux):
#   bash /app/Gland_Seg/Code/run_full_training.sh

set -e
cd /app/Gland_Seg/Code
mkdir -p /app/Gland_Seg/logs

CONFIG=/app/Gland_Seg/Code/config.py
START=$(date +%s)

for BACKBONE in virchow2 uni2 phikon-v2; do
    T0=$(date +%s)
    echo ""
    echo "================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] BEGIN training: $BACKBONE"
    echo "================================================================"

    # Switch config.backbone in-place
    sed -i "s/^    backbone: str = \".*\"\$/    backbone: str = \"$BACKBONE\"/" "$CONFIG"
    grep "^    backbone:" "$CONFIG"

    PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 \
      script -q -f -c "torchrun --standalone --nnodes=1 --nproc_per_node=2 train_full.py" \
      "/app/Gland_Seg/logs/train_full_${BACKBONE}.log"

    DT=$(( $(date +%s) - T0 ))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE $BACKBONE in $((DT/60))m $((DT%60))s"

    # Verify checkpoint
    CKPT="/app/Gland_Seg/checkpoints/best_model_${BACKBONE}_full.pth"
    if [ -f "$CKPT" ]; then
        echo "  ✓ saved $CKPT ($(stat -c%s $CKPT) bytes)"
    else
        echo "  ✗ ERROR: $CKPT not found!"
    fi
done

TOTAL_DT=$(( $(date +%s) - START ))
echo ""
echo "================================================================"
echo "ALL DONE. Total time: $((TOTAL_DT/3600))h $((TOTAL_DT%3600/60))m"
echo "================================================================"
ls -la /app/Gland_Seg/checkpoints/best_model_*_full.pth
