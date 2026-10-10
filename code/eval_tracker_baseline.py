# -*- coding: utf-8 -*-
"""经典时序滤波基线 (审稿意见 #4): Kalman 恒速模型 + IoU 两阶段关联 + 通道输出
========================================================================
ByteTrack 式设计, 全离线运行在 s1 mem-off 检测流上 (无需重新推理):
  - 每类贪心 IoU 关联 (门限 0.3): 第一轮 tracks×高分检测(≥0.5), 第二轮×低分检测([0.3,0.5))
  - 未关联 track: 恒速 Kalman 预测外推 (coast), 分数 ×0.9/帧, 超过 3 帧删除 (与 TFM 同寿命)
  - 输出 = 全部存活 track 的滤波/外推框 (分数≥0.3)
评估: 与 TFM 完全同协议 (aggregate: TP 稳定度/P/R/mAP50)
输出: runs/eval_tracker_baseline/results.json
注: 跟踪器不含外部位移输入 (纯数据驱动运动模型), 因此无 σ 噪声变体.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_persistence import aggregate                                   # noqa: E402

RUNS = Path(__file__).resolve().parent.parent / "runs"
CACHE = RUNS / "eval_persistence"


def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class KF:
    """恒速 Kalman, 状态 [cx,cy,w,h,vx,vy,vw,vh]."""
    def __init__(self, box):
        cx = (box[0] + box[2]) / 2; cy = (box[1] + box[3]) / 2
        w = box[2] - box[0]; h = box[3] - box[1]
        self.x = np.array([cx, cy, w, h, 0., 0., 0., 0.])
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0
        self.H = np.zeros((4, 8)); self.H[:4, :4] = np.eye(4)
        self.P = np.diag([10., 10., 10., 10., 100., 100., 100., 100.])
        self.Q = np.diag([1., 1., 1., 1., 0.05, 0.05, 0.05, 0.05])
        self.R = np.diag([2., 2., 2., 2.])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.box()

    def update(self, box):
        z = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2,
                      box[2] - box[0], box[3] - box[1]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return self.box()

    def box(self):
        cx, cy, w, h = self.x[:4]
        w = max(w, 2.0); h = max(h, 2.0)
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def track_sequence(frames, stride, hi_thr=0.4, noise=0.0, seed=0, gmc=True,
                   max_age=3, decay=0.9, thr_out=0.3):
    """GMC-坐标实现 (ByteTrack 全局运动补偿标准做法):
    KF 全程在运动补偿(世界)坐标中维护 x_world = x_frame + shift_t;
    关联/更新/外推均在世界坐标, 输出时变换回当前帧坐标.
    noise>0: 每帧补偿位移叠加 N(0,noise) (模拟里程计误差); gmc=False: 关闭补偿."""
    if noise > 0:
        np.random.seed(seed)
    tracks = []   # dict: kf(世界系), cls, score, age
    out_frames = []
    shift = 0.0
    for t, fr in enumerate(frames):
        if t > 0 and gmc:
            shift += stride + (float(np.random.randn() * noise) if noise > 0 else 0.0)

        def to_world(b):
            return [b[0] + shift, b[1], b[2] + shift, b[3]]

        def to_frame(b):
            return [b[0] - shift, b[1], b[2] - shift, b[3]]

        boxes = [to_world(b) for b in fr["boxes"].tolist()]
        labels = fr["labels"].tolist(); scores = fr["scores"].tolist()
        n_det = len(boxes)
        assigned_tr = [False] * len(tracks)
        assigned_det = [False] * n_det
        predicted = [tr["kf"].predict() for tr in tracks]   # 世界系预测

        def match_round(det_idx):
            pairs = []
            for ti, tr in enumerate(tracks):
                if assigned_tr[ti]:
                    continue
                for di in det_idx:
                    if assigned_det[di] or labels[di] != tr["cls"]:
                        continue
                    v = iou_xyxy(predicted[ti], boxes[di])
                    if v >= 0.3:
                        pairs.append((v, ti, di))
            pairs.sort(reverse=True)
            for v, ti, di in pairs:
                if not assigned_tr[ti] and not assigned_det[di]:
                    assigned_tr[ti] = True; assigned_det[di] = True
                    tracks[ti]["kf"].update(boxes[di])
                    tracks[ti]["score"] = scores[di]
                    tracks[ti]["age"] = 0

        hi = [i for i in range(n_det) if scores[i] >= hi_thr]
        lo = [i for i in range(n_det) if 0.3 <= scores[i] < hi_thr]
        match_round(hi)
        match_round(lo)
        # 未关联 track: 老化/衰减/删除
        alive = []
        for ti, tr in enumerate(tracks):
            if not assigned_tr[ti]:
                tr["age"] += 1
                tr["score"] *= decay
            if tr["age"] <= max_age and tr["score"] >= thr_out:
                alive.append(tr)
        tracks = alive
        # 新生 track (高分未关联检测)
        for di in hi:
            if not assigned_det[di]:
                tracks.append({"kf": KF(boxes[di]), "cls": labels[di],
                               "score": scores[di], "age": 0})
        # 输出: 世界系滤波框 -> 当前帧坐标
        ob, ol, os_ = [], [], []
        for tr in tracks:
            b = to_frame(tr["kf"].box())
            ob.append(b); ol.append(tr["cls"]); os_.append(tr["score"])
        order = np.argsort(-np.array(os_)) if os_ else []
        out_frames.append({
            "boxes": torch.tensor([ob[i] for i in order], dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor([ol[i] for i in order], dtype=torch.long),
            "scores": torch.tensor([os_[i] for i in order], dtype=torch.float32),
            "gt_boxes": fr["gt_boxes"], "gt_labels": fr["gt_labels"]})
    return out_frames


def main():
    po = CACHE / "stream_memoff_testbig.pt"
    streams = torch.load(po)
    variants = [
        ("conservative_births_clean", dict(hi_thr=0.4)),
        ("conservative_births_sig15", dict(hi_thr=0.4, noise=15.0)),
        ("open_births_clean", dict(hi_thr=0.3)),
        ("open_births_sig15", dict(hi_thr=0.3, noise=15.0)),
        ("no_gmc_open_births", dict(hi_thr=0.3, gmc=False)),
    ]
    results = {}
    for name, kw in variants:
        out_streams = []
        for seq in streams:
            out_streams.append({"stride": seq["stride"],
                                "frames": track_sequence(seq["frames"], seq["stride"], **kw)})
        results[name] = aggregate(out_streams, f"kalman-iou tracker + GMC [{name}]",
                                  compute_m=("clean" in name))
    ref = {"raw_memoff_tp": 0.5064, "tfm_tp": 0.7244, "persistence_tp": 0.7341,
           "tfm_sig15_tp": 0.5941, "persistence_sig15_tp": 0.5449}
    out = {"method": "Kalman constant-velocity + two-stage IoU association + coasting "
                      "(ByteTrack-style) with global motion compensation (GT stride, "
                      "optional N(0,sigma) noise); no learning. Birth thresholds: "
                      "conservative (0.4, standard ByteTrack high-score tier) vs open "
                      "(0.3, all detections).",
           "variants": results, "references": ref}
    OUT = RUNS / "eval_tracker_baseline"
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved ->", OUT / "results.json")


if __name__ == "__main__":
    main()
