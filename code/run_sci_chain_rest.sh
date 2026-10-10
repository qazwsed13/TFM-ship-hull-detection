#!/bin/bash
# SCI 补充实验链·续跑 (修复 age_embed 索引越界后重跑 S7-S10)
set -x
PY="${PY:-python}"
cd "$(dirname "$0")"

echo "===== [S7r] 消融C: 记忆寿命=5 (修复后重跑) ====="
$PY -u eval.py --ckpt ../runs/e2e_s1/ep7.pt --splits test --video --video_splits testbig \
  --motion gt --mem_lifespan 5 --mem_only --out ../runs/eval_sweep_life5 || exit 1

echo "===== [S8r] 鲁棒性A: 里程计噪声 sigma=5px ====="
$PY -u eval.py --ckpt ../runs/e2e_s1/ep7.pt --splits test --video --video_splits testbig \
  --motion gt --motion_noise 5 --mem_only --out ../runs/eval_sweep_noise5 || exit 1

echo "===== [S9r] 鲁棒性B: 里程计噪声 sigma=15px ====="
$PY -u eval.py --ckpt ../runs/e2e_s1/ep7.pt --splits test --video --video_splits testbig \
  --motion gt --motion_noise 15 --mem_only --out ../runs/eval_sweep_noise15 || exit 1

echo "===== [S10r] 无里程计替代: 相位相关运动 ====="
$PY -u eval.py --ckpt ../runs/e2e_s1/ep7.pt --splits test --video --video_splits testbig \
  --motion phase --mem_only --out ../runs/eval_sweep_phase || exit 1

echo "===== SCI 补充实验链·续跑完成 ====="
