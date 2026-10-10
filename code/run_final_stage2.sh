#!/bin/bash
# 最终 stage2: 记忆随机丢弃 + lr1e-4 (保留单帧能力, 充分训练记忆模块)
set -x
PY="${PY:-python}"
cd "$(dirname "$0")"

echo "===== [S2E-1] base_t_s2e ====="
$PY -u train.py --stage 2 --variant base_t --init ../runs/base_s1d/best.pt \
  --out ../runs/base_t_s2e --epochs 12 --batch 4 --lr 1e-4 --mem_dropout 0.35 \
  --workers 2 --no_amp --eval_every 2 --val_batches 20 || exit 1

echo "===== [S2E-2] focus_t_s2e ====="
$PY -u train.py --stage 2 --variant focus_t --init ../runs/focus_s1d/best.pt \
  --out ../runs/focus_t_s2e --epochs 12 --batch 4 --lr 1e-4 --mem_dropout 0.35 \
  --workers 2 --no_amp --eval_every 2 --val_batches 20 || exit 1

echo "===== 最终 stage2 完成 ====="
