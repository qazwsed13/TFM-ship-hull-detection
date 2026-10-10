#!/bin/bash
# 方向2 完整训练链 (顺序执行, 避免 GPU 争用)
# usage: bash run_all_train.sh
set -x
PY="${PY:-python}"
cd "$(dirname "$0")"

echo "===== [1/4] base_s1 =====" 
$PY -u train.py --stage 1 --variant base --out ../runs/base_s1 \
  --epochs 25 --batch 10 --lr 1e-4 --warmup 500 --workers 2 --no_amp \
  --eval_every 5 --val_batches 40 || exit 1

echo "===== [2/4] focus_s1 ====="
$PY -u train.py --stage 1 --variant focus --out ../runs/focus_s1 \
  --epochs 25 --batch 10 --lr 1e-4 --warmup 500 --workers 2 --no_amp \
  --eval_every 5 --val_batches 40 || exit 1

echo "===== [3/4] focus_t_s2 ====="
$PY -u train.py --stage 2 --variant focus_t --init ../runs/focus_s1/best.pt \
  --out ../runs/focus_t_s2 --epochs 15 --batch 4 --lr 1e-4 --workers 2 --no_amp \
  --eval_every 3 --val_batches 20 || exit 1

echo "===== [4/4] base_t_s2 ====="
$PY -u train.py --stage 2 --variant base_t --init ../runs/base_s1/best.pt \
  --out ../runs/base_t_s2 --epochs 15 --batch 4 --lr 1e-4 --workers 2 --no_amp \
  --eval_every 3 --val_batches 20 || exit 1

echo "===== 全部训练完成 ====="
