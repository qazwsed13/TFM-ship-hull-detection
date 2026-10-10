#!/bin/bash
# 方向2 最终评估套件: 4 变体 全量评估 (帧级 + 视频时序 + 运动消融)
set -x
PY="${PY:-python}"
cd "$(dirname "$0")"

echo "===== [E1] base_s1c (帧级) ====="
$PY -u eval.py --ckpt ../runs/base_s1c/best.pt --splits test,test_hard --video --video_splits test,test_hard --motion gt || exit 1

echo "===== [E2] focus_s1c (帧级) ====="
$PY -u eval.py --ckpt ../runs/focus_s1c/best.pt --splits test,test_hard --video --video_splits test,test_hard --motion gt || exit 1

echo "===== [E3] base_t_s2c (帧级+视频, 真值运动) ====="
$PY -u eval.py --ckpt ../runs/base_t_s2c/best.pt --splits test,test_hard --video --video_splits test,test_hard --motion gt || exit 1

echo "===== [E4] focus_t_s2c (帧级+视频, 真值运动) ====="
$PY -u eval.py --ckpt ../runs/focus_t_s2c/best.pt --splits test,test_hard --video --video_splits test,test_hard --motion gt || exit 1

echo "===== [E5] focus_t_s2c 相位相关运动 (无真值) ====="
$PY -u eval.py --ckpt ../runs/focus_t_s2c/best.pt --splits test --video --video_splits test --motion phase || exit 1

echo "===== [E6] focus_t_s2c last 权重对照 ====="
$PY -u eval.py --ckpt ../runs/focus_t_s2c/last.pt --splits test --video --video_splits test --motion gt || exit 1

echo "===== 最终评估完成 ====="
