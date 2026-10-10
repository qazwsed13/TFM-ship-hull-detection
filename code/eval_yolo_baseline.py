# -*- coding: utf-8 -*-
"""
eval_yolo_baseline.py — YOLOv5s 跨范式基线评估（方向2 论文表2/5.2 用）
====================================================================
评估内容（与我们全部指标同协议）：
  1) 帧级: test / test_hard 全量 (conf=0.05, 与我们的帧级协议一致, 最大100框)
  2) 视频: testbig 200 序列的 TP-stability / flicker / miss_runs / 视频mAP (conf=0.3,
     与我们视频协议一致) —— 单帧范式的时序参照点
输出: runs/eval_yolo_baseline/results.json
运行: python eval_yolo_baseline.py            （GPU, ~3 分钟）
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "third_party" / "yolov5"
WEIGHTS = (Path(__file__).resolve().parent.parent / "yolo_runs" / "weights" / "best.pt")
            # fine-tuned YOLOv5s checkpoint (third-party; not bundled; see docs/REPRODUCTION.md)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT / "code"))

from models.common import DetectMultiBackend            # noqa: E402
from utils.general import non_max_suppression           # noqa: E402

from ops import compute_map                             # noqa: E402
from dataset import load_img, load_yolo_label           # noqa: E402
from eval import pred_hits_gt, count_runs, match_boxes_iou, ckpt_boxes_to_xyxy_px  # noqa: E402

DATA = ROOT / "data" / "defect_ds"
OUT = ROOT / "runs" / "eval_yolo_baseline"


@torch.no_grad()
def predict_one(model, dev, img_hwc_rgb, conf):
    """img: HWC RGB float [0,1] -> pred dict (CPU tensors, 像素 xyxy)."""
    x = torch.from_numpy(img_hwc_rgb.transpose(2, 0, 1)).unsqueeze(0).to(dev)
    out = model(x)
    pred = non_max_suppression(out, conf_thres=conf, iou_thres=0.6, max_det=100)[0]
    if pred is None or len(pred) == 0:
        return {"scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long),
                "boxes": torch.zeros(0, 4)}
    return {"scores": pred[:, 4].cpu(), "labels": pred[:, 5].long().cpu(),
            "boxes": pred[:, :4].cpu()}


def eval_frames(model, dev, split, conf=0.05):
    imgs = sorted((DATA / "images" / split).glob("*.jpg"))
    preds, gts = [], []
    for p in imgs:
        img = load_img(p)
        H, W = img.shape[:2]
        pp = predict_one(model, dev, img, conf)
        preds.append(pp)
        boxes, labels = load_yolo_label(DATA / "labels" / split / (p.stem + ".txt"), W, H)
        gt = ckpt_boxes_to_xyxy_px(boxes, W, H) if boxes.numel() else boxes
        gts.append({"labels": labels, "boxes": gt})
    return compute_map(preds, gts, num_classes=5)


def eval_video(model, dev, split="testbig", conf=0.3):
    vi = json.loads((DATA / "video_index.json").read_text(encoding="utf-8"))
    seqs = vi[split]
    all_preds, all_gts = [], []
    tcm_num = tcm_den = 0
    stat = {"miss_runs": 0, "delay_sum": 0, "delay_cnt": 0, "flicker": 0,
            "quick_recovered": 0, "tp_prev": 0, "tp_kept": 0}
    n_obj = 0
    for si, seq in enumerate(seqs):
        move = torch.tensor([seq["stride"], 0.0, seq["stride"], 0.0])
        tracks, prev_pred = [], None
        for t, rel in enumerate(seq["frames"]):
            img = load_img(DATA / rel)
            H, W = img.shape[:2]
            pp = predict_one(model, dev, img, conf)
            all_preds.append(pp)
            boxes, labels = load_yolo_label(DATA / "labels" / split /
                                            (Path(rel).stem + ".txt"), W, H)
            gt_xyxy = ckpt_boxes_to_xyxy_px(boxes, W, H) if boxes.numel() else boxes
            all_gts.append({"labels": labels, "boxes": gt_xyxy})
            n_obj += gt_xyxy.shape[0]
            # ---- GT 轨迹与命中 (与 eval.py 同逻辑) ----
            matched = set()
            if tracks:
                pb = torch.stack([tr["box"] for tr in tracks]) - move
                pl = torch.tensor([tr["label"] for tr in tracks])
                m = match_boxes_iou(pb, gt_xyxy, 0.4, (pl, labels))
            else:
                m = torch.full((0,), -1, dtype=torch.long)
            new_tracks = []
            for i, tr in enumerate(tracks):
                if m.numel() and i < m.numel() and m[i] >= 0 and int(m[i]) not in matched:
                    gi = int(m[i]); matched.add(gi)
                    tr["box"] = gt_xyxy[gi]; tr["label"] = int(labels[gi])
                    tr["hits"].append(pred_hits_gt(pp, gt_xyxy[gi], int(labels[gi])))
                    new_tracks.append(tr)
                else:
                    count_runs(tr["hits"], stat)
            for gi in range(gt_xyxy.shape[0]):
                if gi not in matched:
                    new_tracks.append({"box": gt_xyxy[gi], "label": int(labels[gi]),
                                       "hits": [pred_hits_gt(pp, gt_xyxy[gi], int(labels[gi]))]})
            tracks = new_tracks
            # ---- TCM (预测级, 运动补偿) ----
            if prev_pred is not None:
                pb_, pl_ = prev_pred
                if pb_.numel() and pp["boxes"].numel():
                    m2 = match_boxes_iou(pb_ - move, pp["boxes"], 0.5, (pl_, pp["labels"]))
                    tcm_num += int((m2 >= 0).sum()); tcm_den += int(pb_.numel())
            prev_pred = (pp["boxes"], pp["labels"])
        for tr in tracks:
            count_runs(tr["hits"], stat)
        if (si + 1) % 50 == 0:
            print(f"  ... {si+1}/{len(seqs)} sequences", flush=True)
    m = compute_map(all_preds, all_gts, num_classes=5)
    return {"mAP50": m["mAP50"], "mAP50_95": m["mAP50_95"], "mAP@small": m["mAP@small"],
            "TCM": tcm_num / max(tcm_den, 1),
            "TP_stability": stat["tp_kept"] / max(stat["tp_prev"], 1),
            "tp_kept": stat["tp_kept"], "tp_prev": stat["tp_prev"],
            "flicker_1frame": stat["flicker"], "miss_runs": stat["miss_runs"],
            "quick_recovered": stat["quick_recovered"], "n_gt_frames": n_obj}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    model = DetectMultiBackend(str(WEIGHTS), device=dev, fp16=False)
    model.eval()
    print(f"YOLOv5s loaded from {WEIGHTS}", flush=True)
    results = {"model": "YOLOv5s (COCO-pretrained, fine-tuned 80ep)"}
    for split in ("test", "test_hard"):
        m = eval_frames(model, dev, split)
        results[split] = m
        print(f"[{split}] mAP50={m['mAP50']:.4f} mAP50-95={m['mAP50_95']:.4f} "
              f"AP_S={m['mAP@small']:.4f} 每类={['%.3f' % v if v==v else 'NaN' for v in m['AP50_per_class']]}",
              flush=True)
    r = eval_video(model, dev, "testbig")
    results["video_testbig"] = r
    print(f"[testbig] TP={r['TP_stability']:.3f} ({r['tp_kept']}/{r['tp_prev']}) "
          f"TCM={r['TCM']:.3f} flicker={r['flicker_1frame']} miss_runs={r['miss_runs']} "
          f"mAP50={r['mAP50']:.4f}", flush=True)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"saved -> {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
