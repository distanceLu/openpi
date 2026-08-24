#!/usr/bin/env bash

set -o pipefail

OUTPUT_DIR="/mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/sft_shiji/output_total/aug05-06-causal-10hz-v2-zonly-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=6 \
UV_CACHE_DIR=/tmp/uv-cache \
UV_LINK_MODE=copy \
uv run \
  --project /mnt/data/lcx1/yiqinworkspace/openpi \
  python -m openpi.sft_shiji.train \
  --hdf5-root /mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/sft_shiji/hdf5_data/2026-08-05-06-command-delta-causal-10hz-v2 \
  --checkpoint /mnt/data/lcx1/yiqinworkspace/openpi/asset_pi05_base/pytorch \
  --output-dir "${OUTPUT_DIR}" \
  --tensorboard-dir "${OUTPUT_DIR}/tensorboard" \
  --prompt "follow the demonstrated robot tool trajectory" \
  --validation-recording 2026-08-06__11-34-53,2026-08-06__11-42-22,2026-08-05__14-35-33 \
  --action-horizon 10 \
  --finetune-mode d \
  --mask-non-z-actions \
  --motion-threshold 1e-4 \
  --balance-motion-samples \
  --max-balance-repeat 8 \
  --action-abs-quantile 0.99 \
  --lora-dropout 0.0 \
  --micro-batch-size 8 \
  --gradient-accumulation-steps 2 \
  --learning-rate 5e-5 \
  --min-learning-rate 1e-6 \
  --warmup-steps 50 \
  --weight-decay 0.0 \
  --max-grad-norm 1.0 \
  --steps 2000 \
  --save-interval 100 \
  --validation-interval 0 \
  --validation-batches 0 \
  --validation-inference-batches 0 \
  --validation-diffusion-steps 20 \
  --validation-noise-seed 20260811 \
  --early-stopping-patience 8 \
  --early-stopping-min-delta 1e-4 \
  --early-stopping-overfit-ratio 1.25 \
  --no-use-robot-state \
  --no-gradient-checkpointing \
  2>&1 | tee "${OUTPUT_DIR}/train.log"
