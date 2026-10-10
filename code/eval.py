# -*- coding: utf-8 -*-
"""
评估脚本: 帧级 mAP / 时序一致性 (TCM) / 漏检恢复延迟 / FPS / 运动估计模式对比
用法:
  python eval.py --ckpt ../runs/focus_s1/best.pt --splits test,test_hard
  python eval.py --ckpt ../runs/focus_t_s2/best.pt --video --motion gt
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import DefectDataset, VideoClipDataset, load_img, load_yolo_label
from model import FocusDETRLite
from ops import compute_map, postprocess

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "defect_ds"
CLASS_NAMES = ["rust", "crack", "weld_defect", "peel", "pitting"]


# ---------------------------------------------------------------- 工具

def ckpt_boxes_to_xyxy_px(tb, W, H):
    x0 = (tb[:, 0] - tb[:, 2] / 2) * W
    y0 = (tb[:, 1] - tb[:, 3] / 2) * H
    x1 = (tb[:, 0] + tb[:, 2] / 2) * W
    y1 = (tb[:, 1] + tb[:, 3] / 2) * H
    return torch.stack([x0, y0, x1, y1], -1)


def load_model(ckpt_path, dev):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = FocusDETRLite(num_classes=5, variant=ck["variant"])
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    return model, ck


# ---------------------------------------------------------------- 帧级评估

@torch.no_grad()
def eval_frames(model, split, dev, limit=None, score_thr=0.05):
    ds = DefectDataset(DATA, split, augment=False, limit=limit)
    preds, gts = [], []
    for item in ds:
        img = item["img"].unsqueeze(0).to(dev)
        out = model(img)
        pp = postprocess(out, img.shape[-1], img.shape[-2], score_thr=score_thr)[0]
        preds.append({"scores": pp["scores"].cpu(), "labels": pp["labels"].cpu(),
                      "boxes": pp["boxes"].cpu()})
        t = item["target"]
        gts.append({"labels": t["labels"], "boxes": ckpt_boxes_to_xyxy_px(
            t["boxes"], img.shape[-1], img.shape[-2])})
    return compute_map(preds, gts, num_classes=5)


@torch.no_grad()
def measure_fps(model, dev, n_warm=10, n_run=50, memory_state=None, motion=None):
    x = torch.rand(1, 3, 512, 512, device=dev)
    for _ in range(n_warm):
        model(x, memory_state=memory_state, motion=motion)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_run):
        model(x, memory_state=memory_state, motion=motion)
    torch.cuda.synchronize()
    return n_run / (time.time() - t0)


# ---------------------------------------------------------------- 时序指标

def match_boxes_iou(b1, b2, iou_thr=0.5, same_class=None):
    """返回 b1 中每个框在 b2 中的匹配索引 (-1 无). 均 xyxy."""
    if b1.numel() == 0 or b2.numel() == 0:
        return torch.full((b1.shape[0],), -1, dtype=torch.long)
    from model import bbox_iou_px
    iou = bbox_iou_px(b1, b2)
    idx = torch.full((b1.shape[0],), -1, dtype=torch.long)
    for i in range(b1.shape[0]):
        best = int(iou[i].argmax())
        ok = iou[i, best] >= iou_thr
        if same_class is not None and ok:
            ok = same_class[0][i].item() == same_class[1][best].item()
        if ok:
            idx[i] = best
    return idx


def pred_hits_gt(pp, gt_box, gt_label):
    """当前帧预测是否命中某 GT (IoU>0.5 且同类; 设备自动对齐)."""
    if pp["boxes"].numel() == 0 or gt_box.numel() == 0:
        return False
    dev = pp["boxes"].device
    m = match_boxes_iou(gt_box.view(1, 4).to(dev), pp["boxes"], 0.5,
                        (torch.tensor([gt_label], device=dev), pp["labels"]))
    return bool((m >= 0).any())

def count_runs(hits_list, stat):
    """对命中序列统计 miss-run (flicker=单帧漏检; 恢复延迟=连续漏检帧数;
    quick_recovered=短间隙恢复) + TP 稳定性 P(命中_t | 命中_{t-1}) (与输出数量无关)."""
    hs = hits_list
    i = 0
    while i < len(hs):
        if not hs[i]:
            j = i
            while j < len(hs) and not hs[j]:
                j += 1
            run = j - i
            stat["miss_runs"] += 1
            if j < len(hs):                      # 之后恢复
                stat["delay_sum"] += run
                stat["delay_cnt"] += 1
                if run == 1:
                    stat["flicker"] += 1
                if run <= 3 and i > 0:           # 之前命中过 → 短间隙恢复
                    stat["quick_recovered"] += 1
            i = j
        else:
            i += 1
    # TP 条件留存率 (真检测跨帧稳定性, 不受预测数量影响)
    for a, b in zip(hs[:-1], hs[1:]):
        if a:
            stat["tp_prev"] += 1
            if b:
                stat["tp_kept"] += 1

@torch.no_grad()
def eval_video(model, split, dev, motion_mode="gt", use_memory=True, score_thr=0.3,
               no_align=False, motion_noise=0.0):
    """视频模式评估: 逐序列滚动记忆; 输出帧 mAP + TCM + 漏检恢复统计.

    GT 轨迹用运动补偿 IoU 链式匹配维护 (跨帧同一缺陷对象), 命中序列的
    miss-run 统计 flicker(1帧闪断) 与恢复延迟(连续漏检帧数).
    TCM = 相邻帧检测对 (运动补偿后 IoU>0.5 同类) 占比, 越高越稳.
    no_align: 消融——记忆不做运动对齐 (dx=dy=0).
    motion_noise: 里程计噪声 std (px), 评估对运动估计误差的鲁棒性.
    """
    if motion_noise > 0:
        np.random.seed(0)                     # 噪声可复现
    ds = VideoClipDataset(DATA, split)
    all_preds, all_gts = [], []
    tcm_num, tcm_den = 0, 0
    stat = {"miss_runs": 0, "delay_sum": 0, "delay_cnt": 0, "flicker": 0,
            "quick_recovered": 0, "tp_prev": 0, "tp_kept": 0}
    n_objects = 0
    for seq in ds.seqs:
        memory = None
        prev_pred = None
        move = torch.tensor([seq["stride"], 0.0, seq["stride"], 0.0])
        tracks = []                              # 活动 GT 轨迹 {"box","label","hits"}
        seq_preds, seq_gts = [], []
        for t, rel in enumerate(seq["frames"]):
            img = load_img(DATA / rel)
            H, W = img.shape[:2]
            x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(dev)
            # ---- 运动 (记忆对齐用) ----
            motion, dx, dy = None, 0.0, 0.0
            if use_memory and t > 0:
                dx, dy = float(seq["stride"]), 0.0
                if motion_mode == "phase":
                    g_prev = cv2.cvtColor((prev_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    g_cur = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    (ex, ey), _ = cv2.phaseCorrelate(g_prev.astype(np.float32),
                                                     g_cur.astype(np.float32))
                    dx, dy = -float(ex), -float(ey)   # 校准: cur 相对 prev 的位移取反
                if no_align:                          # 消融: 不做运动对齐
                    dx, dy = 0.0, 0.0
                if motion_noise > 0:                  # 里程计噪声鲁棒性
                    dx += float(np.random.randn() * motion_noise)
                    dy += float(np.random.randn() * motion_noise)
                motion = (dx, dy)
            out = model(x, memory_state=memory, motion=motion, return_memory=True)
            pp = postprocess(out, W, H, score_thr=score_thr)[0]
            seq_preds.append({"scores": pp["scores"].cpu(), "labels": pp["labels"].cpu(),
                              "boxes": pp["boxes"].cpu()})
            boxes, labels = load_yolo_label(DATA / "labels" / seq["split"] /
                                            (Path(rel).stem + ".txt"), W, H)
            gt_xyxy = ckpt_boxes_to_xyxy_px(boxes, W, H) if boxes.numel() else boxes
            seq_gts.append({"labels": labels, "boxes": gt_xyxy})
            # ---- GT 轨迹更新与命中记录 ----
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
                else:                            # 轨迹消失: 结算 miss-run
                    count_runs(tr["hits"], stat)
            for gi in range(gt_xyxy.shape[0]):
                if gi not in matched_gt:
                    new_tracks.append({"box": gt_xyxy[gi], "label": int(labels[gi]),
                                       "hits": [pred_hits_gt(pp, gt_xyxy[gi], int(labels[gi]))]})
            tracks = new_tracks
            n_objects += gt_xyxy.shape[0]
            # ---- TCM (所有模型统一度量: 运动补偿后的相邻帧预测稳定性, 全部在 CPU 上) ----
            if prev_pred is not None:
                pb, pl = prev_pred
                cb, cl = seq_preds[-1]["boxes"], seq_preds[-1]["labels"]
                if pb.numel() and cb.numel():
                    warped = pb - move
                    m2 = match_boxes_iou(warped, cb, 0.5, (pl, cl))
                    tcm_num += int((m2 >= 0).sum())
                    tcm_den += int(pb.numel())
            # ---- 记忆更新 ----
            if use_memory and model.memory is not None:
                probs = out["pred_logits"].sigmoid()
                scores_m, labels_m = probs.max(dim=-1)
                memory = model.update_memory(memory, out["_mem_raw"], W, H, scores_m, labels_m)
            prev_pred = (pp["boxes"].cpu(), pp["labels"].cpu())   # TCM 统一在 CPU 上
            prev_img = img
        for tr in tracks:                        # 序列结束结算
            count_runs(tr["hits"], stat)
        all_preds.extend(seq_preds)
        all_gts.extend(seq_gts)
    m = compute_map(all_preds, all_gts, num_classes=5)
    return {"mAP50": m["mAP50"], "mAP50_95": m["mAP50_95"], "mAP@small": m["mAP@small"],
            "TCM": (tcm_num / max(tcm_den, 1)), "tcm_pairs": tcm_den,
            "miss_runs": stat["miss_runs"], "flicker_1frame": stat["flicker"],
            "recovery_delay_avg": (stat["delay_sum"] / max(stat["delay_cnt"], 1)),
            "recovered_runs": stat["delay_cnt"],
            "quick_recovered": stat["quick_recovered"],
            "TP_stability": (stat["tp_kept"] / max(stat["tp_prev"], 1)),
            "tp_kept": stat["tp_kept"], "tp_prev": stat["tp_prev"],
            "n_gt_frames": n_objects, "n_frames": len(all_preds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--splits", type=str, default="test,test_hard")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--video", action="store_true", help="跑视频时序评估")
    ap.add_argument("--video_splits", type=str, default="test")
    ap.add_argument("--motion", type=str, default="gt", choices=["gt", "phase"])
    ap.add_argument("--no_memory", action="store_true", help="时序模型关闭记忆 (消融)")
    ap.add_argument("--out", type=str, default=None)
    # 推理期消融 (SCI 敏感性/鲁棒性分析)
    ap.add_argument("--mem_only", action="store_true", help="仅评估 mem-on (对照沿用已有 mem-off)")
    ap.add_argument("--no_align", action="store_true", help="消融: 记忆不做运动对齐")
    ap.add_argument("--mem_lifespan", type=int, default=None, help="覆盖记忆寿命 (推理期)")
    ap.add_argument("--mem_score_thr", type=float, default=None, help="覆盖记忆入选阈值")
    ap.add_argument("--motion_noise", type=float, default=0.0, help="里程计噪声 std (px)")
    ap.add_argument("--mem_featureless", action="store_true",
                    help="消融: 记忆特征置零(仅框+年龄接力, 预答'是否只是框传递后处理')")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(args.ckpt, dev)
    if model.memory is not None:
        if args.mem_lifespan is not None:
            model.memory.lifespan = args.mem_lifespan
        if args.mem_score_thr is not None:
            model.memory.score_thr = args.mem_score_thr
        if args.mem_featureless:
            model.mem_featureless = True
    print(f"模型: variant={ck['variant']} epoch={ck['epoch']} best={ck['best']:.4f}", flush=True)
    results = {"ckpt": args.ckpt, "variant": ck["variant"]}
    for split in args.splits.split(","):
        m = eval_frames(model, split, dev, limit=args.limit)
        print(f"[{split}] mAP50={m['mAP50']:.4f} mAP50-95={m['mAP50_95']:.4f} "
              f"AP_S={m['mAP@small']:.4f} AP_M={m['mAP@medium']:.4f} 每类AP50="
              f"{[f'{v:.3f}' if v==v else 'NaN' for v in m['AP50_per_class']]}", flush=True)
        results[split] = m
    if args.video:
        for vsplit in args.video_splits.split(","):
            r = eval_video(model, vsplit, dev, motion_mode=args.motion, use_memory=True,
                           no_align=args.no_align, motion_noise=args.motion_noise)
            tag = f"video_{vsplit}_motion-{args.motion}_mem-on"
            results[tag] = r
            print(f"[{tag}] mAP50={r['mAP50']:.4f} TCM={r['TCM']:.3f} "
                  f"flicker={r['flicker_1frame']} 恢复延迟={r['recovery_delay_avg']:.2f} "
                  f"快捷恢复={r['quick_recovered']} miss_runs={r['miss_runs']}", flush=True)
            if model.memory is not None and not args.mem_only:
                r2 = eval_video(model, vsplit, dev, motion_mode=args.motion, use_memory=False)
                tag2 = f"video_{vsplit}_motion-{args.motion}_mem-off"
                results[tag2] = r2
                print(f"[{tag2}] mAP50={r2['mAP50']:.4f} TCM={r2['TCM']:.3f} "
                      f"flicker={r2['flicker_1frame']} 恢复延迟={r2['recovery_delay_avg']:.2f} "
                          f"快捷恢复={r2['quick_recovered']} miss_runs={r2['miss_runs']}", flush=True)
    # FPS
    fps_nm = measure_fps(model, dev)
    results["fps_bs1"] = fps_nm
    if model.memory is not None:
        probs = torch.zeros(1, 1, 5)
        mem = [{"feat": torch.zeros(0, 256, device=dev), "box": torch.zeros(0, 4, device=dev),
                "score": torch.zeros(0, device=dev), "label": torch.zeros(0, dtype=torch.long, device=dev),
                "age": torch.zeros(0, device=dev)} for _ in range(1)]
        fps_m = measure_fps(model, dev, memory_state=mem, motion=(128.0, 0.0))
        results["fps_bs1_with_memory"] = fps_m
        print(f"FPS(bs1): 无记忆={fps_nm:.1f} 有记忆={fps_m:.1f}", flush=True)
    else:
        print(f"FPS(bs1): {fps_nm:.1f}", flush=True)
    out = Path(args.out) if args.out else ROOT / "runs" / f"eval_{Path(args.ckpt).parent.name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"结果已保存 -> {out/'results.json'}", flush=True)


if __name__ == "__main__":
    main()
