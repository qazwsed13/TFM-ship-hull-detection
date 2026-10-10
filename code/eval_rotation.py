# -*- coding: utf-8 -*-
"""旋转/转弯边界实验 (审稿意见 2.2-3): 逐帧 0.5° 旋转, 仅平移对齐.
相机以 Δθ=0.5°/帧 旋转时, 平移-only 记忆对齐逐步失配 —— 量化退化程度.
GT 框与图像同步绕图像中心旋转 (相机坐标系一致).
输出: runs/eval_rotation/results.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import ckpt_boxes_to_xyxy_px, load_model                              # noqa
from dataset import VideoClipDataset, load_img, load_yolo_label                 # noqa
from ops import postprocess                                                     # noqa
from eval_persistence import aggregate                                          # noqa

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
DATA = ROOT / "data" / "defect_ds"
DTHETA_DEG = 0.5


def rot_mat(center, ang_deg):
    return cv2.getRotationMatrix2D(center, ang_deg, 1.0)


def rot_box(box, M, W, H):
    x0, y0, x1, y1 = box
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    ones = np.ones((4, 1), np.float32)
    p = np.hstack([pts, ones]) @ M.T
    return [float(max(0, p[:, 0].min())), float(max(0, p[:, 1].min())),
            float(min(W, p[:, 0].max())), float(min(H, p[:, 1].max()))]


@torch.no_grad()
def collect(model, dev, split="testbig", score_thr=0.3, use_memory=False,
            dtheta_deg=DTHETA_DEG):
    ds = VideoClipDataset(DATA, split)
    seqs = []
    for si, seq in enumerate(ds.seqs):
        memory = None
        frames = []
        for t, rel in enumerate(seq["frames"]):
            raw = load_img(DATA / rel)
            H, W = raw.shape[:2]
            ang = t * dtheta_deg                      # 相机累计旋转
            img = raw
            gt_boxes_np = None
            if ang != 0.0:
                M = rot_mat((W / 2, H / 2), -ang)     # 图像反向旋转=相机正向旋转
                img = cv2.warpAffine((raw * 255).astype(np.uint8), M, (W, H)).astype(np.float32) / 255.0
            x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(dev)
            motion = (float(seq["stride"]), 0.0) if (use_memory and t > 0) else None
            out = model(x, memory_state=memory, motion=motion, return_memory=True)
            pp = postprocess(out, W, H, score_thr=score_thr)[0]
            boxes, labels = load_yolo_label(DATA / "labels" / seq["split"] /
                                            (Path(rel).stem + ".txt"), W, H)
            gt = ckpt_boxes_to_xyxy_px(boxes, W, H) if boxes.numel() else boxes
            if ang != 0.0 and gt.numel():
                Mf = rot_mat((W / 2, H / 2), -ang)
                gt = torch.tensor([rot_box(b.tolist(), Mf, W, H) for b in gt],
                                  dtype=torch.float32)
            frames.append({"boxes": pp["boxes"].cpu(), "labels": pp["labels"].cpu(),
                           "scores": pp["scores"].cpu(),
                           "gt_boxes": gt.cpu() if gt.numel() else gt,
                           "gt_labels": labels.cpu()})
            if use_memory and model.memory is not None:
                probs = out["pred_logits"].sigmoid()
                sm, lm = probs.max(dim=-1)
                memory = model.update_memory(memory, out["_mem_raw"], W, H, sm, lm)
        seqs.append({"stride": float(seq["stride"]), "frames": frames})
        if (si + 1) % 50 == 0:
            print(f"  ... {si+1}/{len(ds.seqs)}", flush=True)
    return seqs


def main():
    dev = torch.device("cuda")
    model, _ = load_model(str(RUNS / "e2e_s1" / "ep7.pt"), dev)
    OUT = RUNS / "eval_rotation"
    OUT.mkdir(exist_ok=True)
    cache = OUT / "streams_rot05.pt"
    if cache.exists():
        off, on = torch.load(cache)
    else:
        off = collect(model, dev, use_memory=False)
        on = collect(model, dev, use_memory=True)
        torch.save((off, on), cache)
    r_on = aggregate(on, "rotation 0.5 deg/frame mem-on", compute_m=False)
    r_off = aggregate(off, "rotation 0.5 deg/frame mem-off", compute_m=False)
    from scipy import stats
    p = stats.fisher_exact([[r_on["tp_kept"], r_on["tp_prev"] - r_on["tp_kept"]],
                            [r_off["tp_kept"], r_off["tp_prev"] - r_off["tp_kept"]]])[1]
    out = {"dtheta_deg_per_frame": DTHETA_DEG, "cumulative_deg_12frames": 6.0,
           "alignment": "translation-only (motion-gt stride); rotation unmodeled",
           "on": {k: r_on[k] for k in ("TP_stability", "tp_kept", "tp_prev",
                                       "flicker_1frame", "miss_runs",
                                       "precision@0.5", "recall@0.5")},
           "off": {k: r_off[k] for k in ("TP_stability", "tp_kept", "tp_prev",
                                         "flicker_1frame", "miss_runs",
                                         "precision@0.5", "recall@0.5")},
           "gain_pct": (r_on["TP_stability"] / r_off["TP_stability"] - 1) * 100,
           "fisher_p": float(p),
           "clean_reference": {"on": 0.7244, "off": 0.5064, "gain_pct": 43.0}}
    (OUT / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[rotation 0.5deg/f] on={r_on['TP_stability']:.4f} off={r_off['TP_stability']:.4f} "
          f"gain={out['gain_pct']:+.1f}% p={p:.2e}")
    print("saved ->", OUT / "results.json")


if __name__ == "__main__":
    main()
