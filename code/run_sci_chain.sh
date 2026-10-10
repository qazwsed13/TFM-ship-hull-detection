#!/bin/bash
# SCI 严谨性补充实验链: 多随机种子复现 + 推理期消融/鲁棒性分析
# 预注册协议: 主分析用 ep7 (固定 8 轮训练终态); 消融基于 seed1 ep7, mem-off 对照沿用已有结果
set -x
PY="${PY:-python}"
cd "$(dirname "$0")"

echo "===== [S1] e2e seed2 训练 (seed=2027) ====="
$PY -u train.py --stage 3 --variant base_t --init ../runs/base_s1d/best.pt --out ../runs/e2e_s2 \
  --epochs 8 --steps_per_epoch 220 --seq_every 4 --lr 5e-5 --warmup 100 --min_lr_ratio 0.1 \
  --workers 2 --no_amp --eval_every 4 --val_batches 40 --seed 2027 || exit 1

echo "===== [S2] e2e seed2 评估 (ep7, 全量) ====="
$PY -u eval.py --ckpt ../runs/e2e_s2/ep7.pt --splits test,test_hard --video \
  --video_splits testbig,test,test_hard --motion gt --out ../runs/eval_e2e_s2_ep7 || exit 1

echo "===== [S3] e2e seed3 训练 (seed=2028) ====="
$PY -u train.py --stage 3 --variant base_t --init ../runs/base_s1d/best.pt --out ../runs/e2e_s3 \
  --epochs 8 --steps_per_epoch 220 --seq_every 4 --lr 5e-5 --warmup 100 --min_lr_ratio 0.1 \
  --workers 2 --no_amp --eval_every 4 --val_batches 40 --seed 2028 || exit 1

echo "===== [S4] e2e seed3 评估 (ep7, 全量) ====="
$PY -u eval.py --ckpt ../runs/e2e_s3/ep7.pt --splits test,test_hard --video \
  --video_splits testbig,test,test_hard --motion gt --out ../runs/eval_e2e_s3_ep7 || exit 1

echo "===== [S5] 消融A: 无运动对齐 (seed1 ep7) ====="
$PY -u eval.py --ckpt ../runs/e2e_s1/ep7.pt --splits test --video --video_splits testbig \
  --motion gt --no_align --mem_only --out ../runs/eval_sweep_noalign || exit 1

echo "===== [S6] 消融B: 记忆寿命=1 ====="
$PY -u eval.py --ckpt ../runs/e2e_s1/ep7.pt --splits test --video --video_splits testbig \
  --motion gt --mem_lifespan 1 --mem_only --out ../runs/eval_sweep_life1 || exit 1

echo "===== [S7] 消融C: 记忆寿命=5 ====="
$PY -u eval.py --ckpt ../runs/e2e_s1/ep7.pt --splits test --video --video_splits testbig \
  --motion gt --mem_lifespan 5 --mem_only --out ../runs/eval_sweep_life5 || exit 1

echo "===== [S8] 鲁棒性A: 里程计噪声 sigma=5px ====="
$PY -u eval.py --ckpt ../runs/e2e_s1/ep7.pt --splits test --video --video_splits testbig \
  --motion gt --motion_noise 5 --mem_only --out ../runs/eval_sweep_noise5 || exit 1

echo "===== [S9] 鲁棒性B: 里程计噪声 sigma=15px ====="
$PY -u eval.py --ckpt ../runs/e2e_s1/ep7.pt --splits test --video --video_splits testbig \
  --motion gt --motion_noise 15 --mem_only --out ../runs/eval_sweep_noise15 || exit 1

echo "===== [S10] 无里程计替代: 相位相关运动 ====="
$PY -u eval.py --ckpt ../runs/e2e_s1/ep7.pt --splits test --video --video_splits testbig \
  --motion phase --mem_only --out ../runs/eval_sweep_phase || exit 1

echo "===== SCI 补充实验链完成 ====="
