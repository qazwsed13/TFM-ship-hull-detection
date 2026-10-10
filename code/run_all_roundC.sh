#!/bin/bash
# 方向2 第三轮训练链: stage1 再续训 18ep (base/focus) + stage2 重做 (低lr, 逐epoch存档)
set -x
PY="${PY:-python}"
cd "$(dirname "$0")"

echo "===== [C1] base_s1c ====="
$PY -u train.py --stage 1 --variant base --init ../runs/base_s1b/best.pt --out ../runs/base_s1c \
  --epochs 18 --batch 10 --lr 1.2e-4 --warmup 200 --min_lr_ratio 0.08 \
  --workers 2 --no_amp --eval_every 3 --val_batches 40 || exit 1

echo "===== [C2] focus_s1c ====="
$PY -u train.py --stage 1 --variant focus --init ../runs/focus_s1b/best.pt --out ../runs/focus_s1c \
  --epochs 18 --batch 10 --lr 1.5e-4 --warmup 200 --min_lr_ratio 0.08 \
  --workers 2 --no_amp --eval_every 3 --val_batches 40 || exit 1

echo "===== [C3] base_t_s2c ====="
$PY -u train.py --stage 2 --variant base_t --init ../runs/base_s1c/best.pt \
  --out ../runs/base_t_s2c --epochs 10 --batch 4 --lr 5e-5 --workers 2 --no_amp \
  --eval_every 2 --val_batches 20 || exit 1

echo "===== [C4] focus_t_s2c ====="
$PY -u train.py --stage 2 --variant focus_t --init ../runs/focus_s1c/best.pt \
  --out ../runs/focus_t_s2c --epochs 10 --batch 4 --lr 5e-5 --workers 2 --no_amp \
  --eval_every 2 --val_batches 20 || exit 1

echo "===== Round C 完成 ====="
