#!/bin/bash
# 方向2 Round D: base/focus 继续爬升 + focus_g 隔离实验 (Sa-Gn+GIoU) + stage2 重做
set -x
PY="${PY:-python}"
cd "$(dirname "$0")"

echo "===== [D1] focus_g (Sa-Gn+GIoU 隔离, 10ep) ====="
$PY -u train.py --stage 1 --variant focus_g --init ../runs/focus_s1c/best.pt --out ../runs/focus_g_s1 \
  --epochs 10 --batch 10 --lr 1.2e-4 --warmup 200 --min_lr_ratio 0.08 \
  --workers 2 --no_amp --eval_every 2 --val_batches 40 || exit 1

echo "===== [D2] base_s1d (15ep) ====="
$PY -u train.py --stage 1 --variant base --init ../runs/base_s1c/best.pt --out ../runs/base_s1d \
  --epochs 15 --batch 10 --lr 1.0e-4 --warmup 150 --min_lr_ratio 0.08 \
  --workers 2 --no_amp --eval_every 3 --val_batches 40 || exit 1

echo "===== [D3] focus_s1d (16ep) ====="
$PY -u train.py --stage 1 --variant focus --init ../runs/focus_s1c/best.pt --out ../runs/focus_s1d \
  --epochs 16 --batch 10 --lr 1.3e-4 --warmup 200 --min_lr_ratio 0.08 \
  --workers 2 --no_amp --eval_every 3 --val_batches 40 || exit 1

echo "===== [D4] base_t_s2d (10ep, 低lr) ====="
$PY -u train.py --stage 2 --variant base_t --init ../runs/base_s1d/best.pt \
  --out ../runs/base_t_s2d --epochs 10 --batch 4 --lr 5e-5 --workers 2 --no_amp \
  --eval_every 2 --val_batches 20 || exit 1

echo "===== [D5] focus_t_s2d (10ep, 低lr) ====="
$PY -u train.py --stage 2 --variant focus_t --init ../runs/focus_s1d/best.pt \
  --out ../runs/focus_t_s2d --epochs 10 --batch 4 --lr 5e-5 --workers 2 --no_amp \
  --eval_every 2 --val_batches 20 || exit 1

echo "===== Round D 完成 ====="
