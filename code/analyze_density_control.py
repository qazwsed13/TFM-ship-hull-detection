# -*- coding: utf-8 -*-
"""M2 密度/召回对照分析（CPU, 基于低阈值流）
==========================================
1) τ=0.3 复算官方数字 (sanity: ON 0.7244 = 1924/2656; OFF 0.5064 = 1182/2334)
2) OFF 重定阈至与 ON 同召回 (~0.364) → TP 稳定度 + 精度 + Fisher
3) 双轴前沿: 扫 OFF/ON 阈值: (precision, TP stability) 曲线
4) ON 流来源分解: t-1 命中来自 内容查询 vs 记忆 token 的留存率
输出: runs/density_control/summary.json
"""
import json
import sys
from pathlib import Path

import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import match_boxes_iou, pred_hits_gt, count_runs                # noqa: E402
from eval_persistence import _nms_per_class                               # noqa: E402
from ops import compute_map                                               # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DC = ROOT / "runs" / "density_control"
N_Q = 100   # 内容查询数; src>=100 → 记忆 token


def frame_pp(fr, thr):
    ok = fr["scores"] > thr
    return {"boxes": fr["boxes"][ok], "labels": fr["labels"][ok],
            "scores": fr["scores"][ok]}, (fr["src"][ok] if "src" in fr else None)


def hit_src(pp, src, gt_box, gt_label):
    """命中判定 + 命中框来源 (最高 IoU 的同类匹配框)."""
    if pp["boxes"].numel() == 0 or gt_box.numel() == 0:
        return False, -1
    iou = torch.zeros(pp["boxes"].shape[0])
    from model import bbox_iou_px
    iou = bbox_iou_px(gt_box.view(1, 4), pp["boxes"])[0]
    cand = (iou >= 0.5) & (pp["labels"] == gt_label)
    if not bool(cand.any()):
        return False, -1
    cb = cand.nonzero(as_tuple=True)[0]
    best = cb[iou[cb].argmax()]
    s = int(src[best]) if src is not None else 0
    return True, (1 if s >= N_Q else 0)      # 1=记忆来源, 0=内容查询


def aggregate_thr(streams, thr, decompose=False):
    stat = {"flicker": 0, "miss_runs": 0, "delay_sum": 0, "delay_cnt": 0,
            "quick_recovered": 0, "tp_prev": 0, "tp_kept": 0,
            "mem_prev": 0, "mem_kept": 0, "con_prev": 0, "con_kept": 0,
            "hit_boxes_mem": 0, "hit_boxes_con": 0}
    tp_p = fp = fn = n_out = n_gt = n_frames = 0
    all_preds, all_gts = [], []
    for seq in streams:
        tracks = []
        move = torch.tensor([seq["stride"], 0.0, seq["stride"], 0.0])
        for fr in seq["frames"]:
            n_frames += 1
            pp, src = frame_pp(fr, thr)
            gt_xyxy, labels = fr["gt_boxes"], fr["gt_labels"]
            all_preds.append({"boxes": pp["boxes"], "labels": pp["labels"],
                              "scores": pp["scores"]})
            all_gts.append({"boxes": gt_xyxy, "labels": labels})
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
                    if decompose:
                        hit, s = hit_src(pp, src, gt_xyxy[gi], int(labels[gi]))
                        prev_src = tr.get("last_src", 0)
                        if tr["hits"] and tr["hits"][-1]:    # t-1 命中 → 条件转移
                            if prev_src == 1:
                                stat["mem_prev"] += 1
                                stat["mem_kept"] += int(hit)
                            else:
                                stat["con_prev"] += 1
                                stat["con_kept"] += int(hit)
                        tr["hits"].append(hit)
                        tr["last_src"] = s if hit else prev_src
                    else:
                        tr["hits"].append(pred_hits_gt(pp, gt_xyxy[gi], int(labels[gi])))
                    new_tracks.append(tr)
                else:
                    count_runs(tr["hits"], stat)
            for gi in range(gt_xyxy.shape[0]):
                if gi not in matched_gt:
                    if decompose:
                        hit, s = hit_src(pp, src, gt_xyxy[gi], int(labels[gi]))
                        new_tracks.append({"box": gt_xyxy[gi], "label": int(labels[gi]),
                                           "hits": [hit], "last_src": s})
                    else:
                        new_tracks.append({"box": gt_xyxy[gi], "label": int(labels[gi]),
                                           "hits": [pred_hits_gt(pp, gt_xyxy[gi], int(labels[gi]))]})
            tracks = new_tracks
        for tr in tracks:
            count_runs(tr["hits"], stat)
    res = {"thr": thr,
           "TP_stability": stat["tp_kept"] / max(stat["tp_prev"], 1),
           "tp_kept": stat["tp_kept"], "tp_prev": stat["tp_prev"],
           "precision@0.5": tp_p / max(tp_p + fp, 1),
           "recall@0.5": tp_p / max(tp_p + fn, 1),
           "out_per_frame": n_out / max(n_frames, 1)}
    if decompose:
        res["src_decomp"] = {
            "mem_prev": stat["mem_prev"], "mem_kept": stat["mem_kept"],
            "con_prev": stat["con_prev"], "con_kept": stat["con_kept"],
            "P_kept_mem_src": stat["mem_kept"] / max(stat["mem_prev"], 1),
            "P_kept_con_src": stat["con_kept"] / max(stat["con_prev"], 1)}
    return res


def fisher_stab(a, b):
    return float(stats.fisher_exact([[a["tp_kept"], a["tp_prev"] - a["tp_kept"]],
                                     [b["tp_kept"], b["tp_prev"] - b["tp_kept"]]])[1])


def main():
    off = torch.load(DC / "stream_memoff_lowthr.pt", weights_only=False)
    on = torch.load(DC / "stream_memon_lowthr.pt", weights_only=False)
    out = {"n_q_content": N_Q}
    print("streams loaded", flush=True)

    # ---- sanity @0.3 ----
    r_on30 = aggregate_thr(on, 0.30, decompose=True)
    r_off30 = aggregate_thr(off, 0.30)
    out["sanity_on_030"] = r_on30
    out["sanity_off_030"] = r_off30
    print(f"[sanity ON@0.3 ] TP={r_on30['TP_stability']:.4f} ({r_on30['tp_kept']}/{r_on30['tp_prev']}) "
          f"P={r_on30['precision@0.5']:.3f} R={r_on30['recall@0.5']:.3f} dens={r_on30['out_per_frame']:.2f}")
    print(f"[sanity OFF@0.3] TP={r_off30['TP_stability']:.4f} ({r_off30['tp_kept']}/{r_off30['tp_prev']}) "
          f"P={r_off30['precision@0.5']:.3f} R={r_off30['recall@0.5']:.3f} dens={r_off30['out_per_frame']:.2f}")
    R_ON = r_on30["recall@0.5"]

    # ---- 前沿扫描 ----
    grid = [0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
    curve = {"off": [], "on": []}
    for thr in grid:
        ro = aggregate_thr(off, thr)
        rn = aggregate_thr(on, thr)
        curve["off"].append(ro)
        curve["on"].append(rn)
        print(f"[frontier t={thr:.2f}] OFF P={ro['precision@0.5']:.3f} R={ro['recall@0.5']:.3f} "
              f"TP={ro['TP_stability']:.4f} d={ro['out_per_frame']:.2f} | "
              f"ON P={rn['precision@0.5']:.3f} R={rn['recall@0.5']:.3f} TP={rn['TP_stability']:.4f} "
              f"d={rn['out_per_frame']:.2f}", flush=True)
    out["frontier"] = curve

    # ---- 召回匹配: 找 OFF 阈值使 recall ≈ R_ON (τ↓→recall↑) ----
    lo, hi = 0.05, 0.30
    for _ in range(10):
        mid = (lo + hi) / 2
        rm = aggregate_thr(off, mid)["recall@0.5"]
        if rm < R_ON:
            hi = mid          # 召回不足 → 需要更低阈值
        else:
            lo = mid
    thr_match = round((lo + hi) / 2, 4)
    r_match = aggregate_thr(off, thr_match)
    r_match["fisher_vs_on"] = fisher_stab(r_match, r_on30)
    out["recall_matched_off"] = r_match
    print(f"[recall-matched OFF] thr={thr_match} TP={r_match['TP_stability']:.4f} "
          f"({r_match['tp_kept']}/{r_match['tp_prev']}) P={r_match['precision@0.5']:.3f} "
          f"R={r_match['recall@0.5']:.3f} dens={r_match['out_per_frame']:.2f} "
          f"Fisher vs ON p={r_match['fisher_vs_on']:.3g}")

    # ---- 密度匹配 (out/frame ≈ ON 3.33; τ↓→密度↑): 供参照 ----
    lo, hi = 0.05, 0.30
    for _ in range(10):
        mid = (lo + hi) / 2
        dm = aggregate_thr(off, mid)["out_per_frame"]
        if dm < r_on30["out_per_frame"]:
            hi = mid          # 密度不足 → 需要更低阈值
        else:
            lo = mid
    thr_dens = round((lo + hi) / 2, 4)
    r_dens = aggregate_thr(off, thr_dens)
    r_dens["fisher_vs_on"] = fisher_stab(r_dens, r_on30)
    out["density_matched_off"] = r_dens
    print(f"[density-matched OFF] thr={thr_dens} TP={r_dens['TP_stability']:.4f} "
          f"P={r_dens['precision@0.5']:.3f} R={r_dens['recall@0.5']:.3f} "
          f"dens={r_dens['out_per_frame']:.2f} Fisher p={r_dens['fisher_vs_on']:.3g}")

    # ---- 跨种子稳健性: s2/s3 召回匹配对照 (低阈值流若在则运行) ----
    for seed, tag in [(2, "_s2"), (3, "_s3")]:
        f_off = DC / f"stream_memoff_lowthr{tag}.pt"
        f_on = DC / f"stream_memon_lowthr{tag}.pt"
        if not (f_off.exists() and f_on.exists()):
            continue
        o2 = torch.load(f_off, weights_only=False)
        n2 = torch.load(f_on, weights_only=False)
        rn = aggregate_thr(n2, 0.30)
        ro = aggregate_thr(o2, 0.30)
        TARGET = rn["recall@0.5"]
        lo, hi = 0.05, 0.30
        for _ in range(10):
            mid = (lo + hi) / 2
            rm = aggregate_thr(o2, mid)["recall@0.5"]
            if rm < TARGET:
                hi = mid
            else:
                lo = mid
        thr_m = round((lo + hi) / 2, 4)
        rm_ = aggregate_thr(o2, thr_m)
        rm_["fisher_vs_on"] = fisher_stab(rm_, rn)
        out[f"seed_s{seed}"] = {"on_030": rn, "off_030": ro,
                                "recall_matched_off": rm_, "match_thr": thr_m}
        print(f"[seed s{seed}] ON@0.3 TP={rn['TP_stability']:.4f} R={rn['recall@0.5']:.3f} | "
              f"matched OFF thr={thr_m} TP={rm_['TP_stability']:.4f} "
              f"({rm_['tp_kept']}/{rm_['tp_prev']}) P={rm_['precision@0.5']:.3f} "
              f"Fisher p={rm_['fisher_vs_on']:.3g}", flush=True)

    (DC / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved ->", DC / "summary.json")


if __name__ == "__main__":
    main()
