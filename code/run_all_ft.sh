#!/bin/bash
# 方向2 续训链 (修复学习率过早衰减: 更高lr + LR下限 + 更长周期)
# 从第一轮 best 权重继续, 输出 *_s1b / *_s2b 目录 (保留旧结果)
set -x
PY="${PY:-python}"
cd "$(dirname "$0")"

echo "===== [1/4] base_s1b (续训) ====="
$PY -u train.py --stage 1 --variant base --init ../runs/base_s1/best.pt --out ../runs/base_s1b \
  --epochs 18 --batch 10 --lr 1.5e-4 --warmup 300 --min_lr_ratio 0.08 \
  --workers 2 --no_amp --eval_every 3 --val_batches 40 || exit 1

echo "===== [2/4] focus_s1b (续训) ====="
$PY -u train.py --stage 1 --variant focus --init ../runs/focus_s1/best.pt --out ../runs/focus_s1b \
  --epochs 18 --batch 10 --lr 1.5e-4 --warmup 300 --min_lr_ratio 0.08 \
  --workers 2 --no_amp --eval_every 3 --val_batches 40 || exit 1

echo "===== [3/4] focus_t_s2b ====="
$PY -u train.py --stage 2 --variant focus_t --init ../runs/focus_s1b/best.pt \
  --out ../runs/focus_t_s2b --epochs 15 --batch 4 --lr 1e-4 --workers 2 --no_amp \
  --eval_every 3 --val_batches 20 || exit 1

echo "===== [4/4] base_t_s2b ====="
$PY -u train.py --stage 2 --variant base_t --init ../runs/base_s1b/best.pt \
  --out ../runs/base_t_s2b --epochs 15 --batch 4 --lr 1e-4 --workers 2 --no_amp \
  --eval_every 3 --val_batches 20 || exit 1

echo "===== 续训链完成 ====="
