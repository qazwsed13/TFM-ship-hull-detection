# -*- coding: utf-8 -*-
"""YOLOv5s + 输出级持久化 边界实验 (审稿意见 #9)
================================================
目的: 在近饱和检测器 (YOLO TP-stability 0.966) 上, 零学习持久化包装器
还能带来多少留存? 同时给出 σ=15px 运动误差下的表现, 定位 TFM 适用边界.
流程: ① 采集 YOLO testbig 逐帧检测流 (conf=0.3, 同协议) 并缓存
      ② sanity: 原始流复算官方 0.9656
      ③ persist(decay .9) / persist(σ15) 包装 -> aggregate
输出: runs/eval_yolo_baseline/{stream_yolo_testbig.pt, persistence_results.json}
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "third_party" / "yolov5"))

from models.common import DetectMultiBackend                              # noqa: E402
from eval_yolo_baseline import predict_one, DATA, WEIGHTS                 # noqa: E402
from eval_persistence import persist, aggregate                           # noqa: E402
from eval import ckpt_boxes_to_xyxy_px                                    # noqa: E402
from dataset import load_img, load_yolo_label                             # noqa: E402

OUT = ROOT / "runs" / "eval_yolo_baseline"


@torch.no_grad()
def collect_yolo(model, dev, split="testbig", conf=0.3):
    vi = json.loads((DATA / "video_index.json").read_text(encoding="utf-8"))
    seqs_out = []
    for si, seq in enumerate(vi[split]):
        frames = []
        for rel in seq["frames"]:
            img = load_img(DATA / rel)
            H, W = img.shape[:2]
            pp = predict_one(model, dev, img, conf)
            boxes, labels = load_yolo_label(DATA / "labels" / split /
                                            (Path(rel).stem + ".txt"), W, H)
            gt = ckpt_boxes_to_xyxy_px(boxes, W, H) if boxes.numel() else boxes
            frames.append({"boxes": pp["boxes"], "labels": pp["labels"],
                           "scores": pp["scores"],
                           "gt_boxes": gt, "gt_labels": labels})
        seqs_out.append({"stride": float(seq["stride"]), "frames": frames})
        if (si + 1) % 50 == 0:
            print(f"  ... {si+1}/{len(vi[split])} sequences", flush=True)
    return seqs_out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    pf = OUT / "stream_yolo_testbig.pt"
    if pf.exists():
        streams = torch.load(pf)
        print("stream loaded from cache")
    else:
        model = DetectMultiBackend(str(WEIGHTS), device=dev, fp16=False)
        model.eval()
        streams = collect_yolo(model, dev)
        torch.save(streams, pf)
        print("stream collected & cached")
    results = {}
    print("== sanity: YOLO raw (official TP 0.9656) ==")
    results["raw"] = aggregate(streams, "YOLO raw (no memory)")
    v1 = persist(streams, decay=0.9)
    results["persist"] = aggregate(v1, "YOLO + persistence (decay.9)")
    v2 = persist(streams, decay=0.9, motion_noise=15.0)
    results["persist_sig15"] = aggregate(v2, "YOLO + persistence (sig15px)", compute_m=False)
    (OUT / "persistence_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved ->", OUT / "persistence_results.json")


if __name__ == "__main__":
    main()
