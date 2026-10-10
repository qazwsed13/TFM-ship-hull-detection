# -*- coding: utf-8 -*-
"""
朴素"对齐框持久化"后处理基线（Nature 审核 P0 补测）
=================================================================
问题：TP-stability 只统计真检测的跨帧留存（不惩罚误检）。一个零学习的
输出级持久化方法——把上一帧的输出框按里程计搬过来、置信度衰减、同类
去重——能否达到相近的留存率？必须实测，连同 mAP/精度一起报告。

协议与 eval.py 完全一致：conf=0.3、motion-gt（warp = -stride）、
GT 轨迹链式匹配（IoU>0.4 同类）、hit=IoU>0.5 同类、mAP=ops.compute_map、
testbig 200 序列。里程计噪声变体与 eval.py 同（np.random.seed(0)）。

流程：
  1) 采集两条流：模型 mem-off 原始检测流 / mem-on 输出流（均 s1 ep7）
  2) sanity：原始流复算官方数字（容许 GPU 抖动），mem-on 复算 0.7244
  3) 变体：persistence(decay .9, 无年龄上限) / (decay .9, age<=3) /
           (decay .9, 无上限, σ=15px 噪声) / (decay .9, age<=3, σ=15px)
输出：runs/eval_persistence/{stream_memoff.pt, stream_memon.pt, results.json}
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dataset import VideoClipDataset, load_img, load_yolo_label
from eval import (load_model, ckpt_boxes_to_xyxy_px, match_boxes_iou,
                  pred_hits_gt, count_runs)
from model import bbox_iou_px
from ops import compute_map, postprocess

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "defect_ds"
OUTDIR = ROOT / "runs" / "eval_persistence"
CONF = 0.3


# ---------------------------------------------------------------- 采集
@torch.no_grad()
def collect(model, dev, split="testbig", score_thr=CONF, use_memory=False):
    """复刻 eval.py eval_video 的推理流程，收集逐帧预测与 GT。"""
    ds = VideoClipDataset(DATA, split)
    seqs = []
    for si, seq in enumerate(ds.seqs):
        memory = None
        frames = []
        for t, rel in enumerate(seq["frames"]):
            img = load_img(DATA / rel)
            H, W = img.shape[:2]
            x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(dev)
            motion = (float(seq["stride"]), 0.0) if (use_memory and t > 0) else None
            out = model(x, memory_state=memory, motion=motion, return_memory=True)
            pp = postprocess(out, W, H, score_thr=score_thr)[0]
            boxes, labels = load_yolo_label(DATA / "labels" / seq["split"] /
                                            (Path(rel).stem + ".txt"), W, H)
            gt_xyxy = ckpt_boxes_to_xyxy_px(boxes, W, H) if boxes.numel() else boxes
            frames.append({
                "boxes": pp["boxes"].cpu(), "labels": pp["labels"].cpu(),
                "scores": pp["scores"].cpu(),
                "gt_boxes": gt_xyxy.cpu() if gt_xyxy.numel() else gt_xyxy,
                "gt_labels": labels.cpu()})
            if use_memory and model.memory is not None:
                probs = out["pred_logits"].sigmoid()
                sm, lm = probs.max(dim=-1)
                memory = model.update_memory(memory, out["_mem_raw"], W, H, sm, lm)
        seqs.append({"stride": float(seq["stride"]), "frames": frames})
        if (si + 1) % 50 == 0:
            print(f"  collect(mem={'on' if use_memory else 'off'}) {si + 1}/{len(ds.seqs)}")
    return seqs


# ---------------------------------------------------------------- 持久化
def persist(streams, decay=0.9, max_age=None, motion_noise=0.0, seed=0):
    """输出级持久化：warp(上一帧输出) + 衰减 + 新鲜优先去重 + 阈值过滤。

    fresh 优先：同类别与携带框 IoU>0.6 时保留新鲜检测（相当于"复现即刷新"）。
    """
    if motion_noise > 0:
        np.random.seed(seed)
    out = []
    for seq in streams:
        stride = seq["stride"]
        carry = None
        frames = []
        for t, fr in enumerate(seq["frames"]):
            dets = {k: fr[k].clone() for k in ("boxes", "labels", "scores")}
            nf = dets["boxes"].shape[0]
            if t == 0:
                merged = {**dets, "age": torch.zeros(nf)}
            else:
                dx, dy = stride, 0.0
                if motion_noise > 0:
                    dx += float(np.random.randn() * motion_noise)
                    dy += float(np.random.randn() * motion_noise)
                mv = torch.tensor([dx, dy, dx, dy])
                cb = carry["boxes"] - mv
                cs = carry["scores"] * decay
                ca = carry["age"] + 1.0
                box = torch.cat([dets["boxes"], cb], 0)
                lab = torch.cat([dets["labels"], carry["labels"]], 0)
                sco = torch.cat([dets["scores"], cs], 0)
                age = torch.cat([torch.zeros(nf), ca], 0)
                fresh = torch.cat([torch.ones(nf), torch.zeros(cb.shape[0])])
                keep = _dedup_fresh_first(box, lab, sco, fresh, 0.6)
                merged = {"boxes": box[keep], "labels": lab[keep],
                          "scores": sco[keep], "age": age[keep]}
            ok = merged["scores"] >= CONF
            if max_age is not None:
                ok &= merged["age"] <= max_age
            merged = {k: v[ok] for k, v in merged.items()}
            carry = {k: merged[k].clone() for k in ("boxes", "labels", "scores", "age")}
            frames.append({"boxes": merged["boxes"], "labels": merged["labels"],
                           "scores": merged["scores"],
                           "gt_boxes": fr["gt_boxes"], "gt_labels": fr["gt_labels"]})
        out.append({"stride": stride, "frames": frames})
    return out


def _nms_per_class(box, lab, sco, iou_thr=0.6):
    """标准纯 NMS (按类, 分数降序, IoU>=thr 抑制) —— 用于 P/R 的统一预处理."""
    if box.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    keep = []
    for c in lab.unique():
        idx = (lab == c).nonzero(as_tuple=True)[0]
        order = idx[sco[idx].argsort(descending=True)]
        b = box[order]
        alive = list(range(order.numel()))
        while alive:
            j0 = alive[0]
            keep.append(int(order[j0]))
            if len(alive) == 1:
                break
            ious = bbox_iou_px(b[j0].unsqueeze(0), b[alive[1:]])[0]
            alive = [a for k, a in enumerate(alive[1:]) if ious[k] < iou_thr]
    return torch.tensor(sorted(keep), dtype=torch.long)


def _dedup_fresh_first(box, lab, sco, fresh, iou_thr):
    """按类去重：新鲜框优先（分数降序），再携带框（分数降序）；IoU>=thr 抑制。"""
    keep = []
    for c in lab.unique():
        idx = (lab == c).nonzero(as_tuple=True)[0]
        fi = idx[fresh[idx] > 0]
        ci = idx[fresh[idx] == 0]
        order = torch.cat([fi[sco[fi].argsort(descending=True)],
                           ci[sco[ci].argsort(descending=True)]])
        b = box[order]
        alive = list(range(order.numel()))
        while alive:
            j0 = alive[0]
            keep.append(int(order[j0]))
            if len(alive) == 1:
                break
            ious = bbox_iou_px(b[j0].unsqueeze(0), b[alive[1:]])[0]
            alive = [a for k, a in enumerate(alive[1:]) if ious[k] < iou_thr]
    return torch.tensor(sorted(keep), dtype=torch.long)


# ---------------------------------------------------------------- 聚合
def aggregate(streams, tag="", compute_m=True):
    """复刻 eval_video 的 GT 轨迹/留存统计 + 精度/召回（IoU0.5 同类）。"""
    stat = {"miss_runs": 0, "delay_sum": 0, "delay_cnt": 0, "flicker": 0,
            "quick_recovered": 0, "tp_prev": 0, "tp_kept": 0}
    all_preds, all_gts = [], []
    tp_p = fp = fn = n_out = n_gt = 0
    for seq in streams:
        tracks = []
        move = torch.tensor([seq["stride"], 0.0, seq["stride"], 0.0])
        for fr in seq["frames"]:
            pp = {"boxes": fr["boxes"], "labels": fr["labels"], "scores": fr["scores"]}
            gt_xyxy, labels = fr["gt_boxes"], fr["gt_labels"]
            all_preds.append({"boxes": pp["boxes"], "labels": pp["labels"],
                              "scores": pp["scores"]})
            all_gts.append({"boxes": gt_xyxy, "labels": labels})
            # ---- 精度/召回: 所有变体统一先做同类 NMS(IoU0.6, 消除官方无NMS流的重复框伪影), 再匹配 ----
            pb, pl, ps = pp["boxes"], pp["labels"], pp["scores"]
            if pb.numel():
                kk = _nms_per_class(pb, pl, ps, 0.6)
                pb, pl, ps = pb[kk], pl[kk], ps[kk]
            n_out += pb.shape[0]
            n_gt += gt_xyxy.shape[0]
            if pb.numel() and gt_xyxy.numel():
                m = match_boxes_iou(pb, gt_xyxy, 0.5, (pl, labels))
                tp_p += int((m >= 0).sum())
                fp += int((m < 0).sum())
                fn += int(gt_xyxy.shape[0]) - int(torch.unique(m[m >= 0]).numel())
            elif pb.numel():
                fp += pb.shape[0]
            else:
                fn += gt_xyxy.shape[0]
            # ---- GT 轨迹与命中 (与 eval_video 逐行一致) ----
            matched_gt = set()
            if tracks:
                prev_boxes = torch.stack([tr["box"] for tr in tracks]) - move
                prev_labels = torch.tensor([tr["label"] for tr in tracks])
                m = match_boxes_iou(prev_boxes, gt_xyxy, 0.4, (prev_labels, labels))
            else:
                m = torch.full((0,), -1, dtype=torch.long)
            new_tracks = []
            for i, tr in enumerate(tracks):
                if m.numel() and i < m.numel() and m[i] >= 0 and int(m[i]) not in matched_gt:
                    gi = int(m[i])
                    matched_gt.add(gi)
                    tr["box"] = gt_xyxy[gi]
                    tr["label"] = int(labels[gi])
                    tr["hits"].append(pred_hits_gt(pp, gt_xyxy[gi], int(labels[gi])))
                    new_tracks.append(tr)
                else:
                    count_runs(tr["hits"], stat)
            for gi in range(gt_xyxy.shape[0]):
                if gi not in matched_gt:
                    new_tracks.append({"box": gt_xyxy[gi], "label": int(labels[gi]),
                                       "hits": [pred_hits_gt(pp, gt_xyxy[gi], int(labels[gi]))]})
            tracks = new_tracks
        for tr in tracks:
            count_runs(tr["hits"], stat)
    cm = compute_map(all_preds, all_gts, num_classes=5) if compute_m else {}
    res = {"tag": tag,
           "TP_stability": stat["tp_kept"] / max(stat["tp_prev"], 1),
           "tp_kept": stat["tp_kept"], "tp_prev": stat["tp_prev"],
           "flicker_1frame": stat["flicker"], "miss_runs": stat["miss_runs"],
           "mAP50": cm.get("mAP50"), "mAP50_95": cm.get("mAP50_95"),
           "precision@0.5": tp_p / max(tp_p + fp, 1),
           "recall@0.5": tp_p / max(tp_p + fn, 1),
           "out_per_frame": n_out / max(len(all_preds), 1)}
    mAP_str = f"mAP50={res['mAP50']:.4f} " if res.get("mAP50") is not None else "mAP50=n/a "
    print(f"[{tag}] TP={res['TP_stability']:.4f} ({res['tp_kept']}/{res['tp_prev']}) "
          f"flicker={res['flicker_1frame']} miss={res['miss_runs']} " + mAP_str +
          f"P={res['precision@0.5']:.3f} R={res['recall@0.5']:.3f} out/f={res['out_per_frame']:.1f}")
    return res


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(ROOT / "runs" / "e2e_s1" / "ep7.pt"))
    ap.add_argument("--split", type=str, default="testbig")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.ckpt, dev)
    print(f"model loaded: {args.ckpt}")

    t0 = time.time()
    p_off = OUTDIR / f"stream_memoff_{args.split}.pt"
    p_on = OUTDIR / f"stream_memon_{args.split}.pt"
    if p_off.exists() and p_on.exists():
        streams_off = torch.load(p_off)
        streams_on = torch.load(p_on)
        print("streams loaded from cache")
    else:
        streams_off = collect(model, dev, args.split, use_memory=False)
        torch.save(streams_off, p_off)
        streams_on = collect(model, dev, args.split, use_memory=True)
        torch.save(streams_on, p_on)
    print(f"collect done in {time.time() - t0:.0f}s")

    results = {"ckpt": str(args.ckpt), "split": args.split, "conf": CONF}
    print("\n=== sanity: 原始流复算 (应为 mem-off 官方数字: 0.5064/675/3811/0.1323) ===")
    results["sanity_memoff_raw"] = aggregate(streams_off, "sanity-memoff-raw")
    print("=== sanity: mem-on 复算 (应为官方 0.7244/613/3393/0.1472) ===")
    results["sanity_memon"] = aggregate(streams_on, "sanity-memon")

    print("\n=== persistence variants (基于 mem-off 原始流) ===")
    v1 = persist(streams_off, decay=0.9, max_age=None)
    results["persist_inf"] = aggregate(v1, "persist(decay.9, no cap)")
    v2 = persist(streams_off, decay=0.9, max_age=3)
    results["persist_L3"] = aggregate(v2, "persist(decay.9, age<=3)")
    v3 = persist(streams_off, decay=0.9, max_age=None, motion_noise=15.0)
    results["persist_inf_sig15"] = aggregate(v3, "persist(no cap, sig15px)", compute_m=False)
    v4 = persist(streams_off, decay=0.9, max_age=3, motion_noise=15.0)
    results["persist_L3_sig15"] = aggregate(v4, "persist(age<=3, sig15px)", compute_m=False)
    v5 = persist(streams_off, decay=0.9, max_age=None, motion_noise=5.0)
    results["persist_inf_sig5"] = aggregate(v5, "persist(no cap, sig5px)", compute_m=False)

    with open(OUTDIR / "results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {OUTDIR / 'results.json'}")


if __name__ == "__main__":
    main()
