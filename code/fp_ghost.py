# -*- coding: utf-8 -*-
"""Minor 9: 假阳性持久化（幽灵检出）代理统计
============================================
基于官方缓存流 (τ=0.3): 对每帧 t>0, 把 t-1 帧输出按 stride 回绕, 统计当前
假阳性(未命中任何 GT)中与"前帧假阳性"匹配(IoU>=0.5 同类)的比例 = ghost 率。
对比 ON vs OFF; 输出 runs/fp_ghost.json
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import match_boxes_iou  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "runs" / "eval_persistence"


def fp_flags(fr):
    """返回每个框是否 FP (未命中任何 GT, IoU>=0.5 同类)."""
    b, l = fr["boxes"], fr["labels"]
    gt, gl = fr["gt_boxes"], fr["gt_labels"]
    if b.numel() == 0:
        return torch.zeros(0, dtype=torch.bool)
    if gt.numel() == 0:
        return torch.ones(b.shape[0], dtype=torch.bool)
    m = match_boxes_iou(b, gt, 0.5, (l, gl))
    return m < 0


def ghost_rate(streams):
    n_fp = n_ghost = 0
    for seq in streams:
        stride = seq["stride"]
        prev_b = prev_l = prev_fp = None
        for t, fr in enumerate(seq["frames"]):
            fp = fp_flags(fr)
            if t > 0 and prev_b is not None and prev_b.numel() and fr["boxes"].numel():
                warped = prev_b - torch.tensor([stride, 0.0, stride, 0.0])
                m = match_boxes_iou(fr["boxes"], warped, 0.5, (fr["labels"], prev_l))
                for i in range(fr["boxes"].shape[0]):
                    if not bool(fp[i]):
                        continue
                    n_fp += 1
                    if m[i] >= 0 and bool(prev_fp[m[i]]):
                        n_ghost += 1
            elif t > 0:
                n_fp += int(fr["boxes"].numel()) if fr["boxes"].numel() else 0
            prev_b, prev_l, prev_fp = fr["boxes"], fr["labels"], fp
    return {"n_fp": n_fp, "n_ghost_fp": n_ghost,
            "ghost_rate": n_ghost / max(n_fp, 1)}


def main():
    out = {}
    for mode, f in [("mem_on", "stream_memon_testbig.pt"),
                    ("mem_off", "stream_memoff_testbig.pt")]:
        streams = torch.load(CACHE / f, weights_only=False)
        r = ghost_rate(streams)
        out[mode] = r
        print(mode, r, flush=True)
    (ROOT / "runs" / "fp_ghost.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved ->", ROOT / "runs" / "fp_ghost.json")


if __name__ == "__main__":
    main()
