# -*- coding: utf-8 -*-
"""P0-1 补强: 零学习 wrapper + 相位相关运动 组合实验 (推理/CPU only)
- 复刻 eval.py motion_mode='phase' 的运动估计 (同转换/同符号)
- wrapper 用同一条 phase 运动序列回绕 (persist 语义与 eval_persistence 一致)
- 对照: TFM+phase (eval_sweep_phase), TFM 干净 0.7244, wrapper 干净 0.7341
输出: runs/eval_wrapper_phase/results.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import VideoClipDataset, load_img                        # noqa: E402
from eval_persistence import aggregate, CONF, _dedup_fresh_first      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "defect_ds"
OUT = ROOT / "runs" / "eval_wrapper_phase"


def phase_motion_table():
    """复刻 eval.py: g=(img*255)uint8->RGB2GRAY; (ex,ey)=phaseCorrelate(prev,cur); dx,dy=-ex,-ey."""
    ds = VideoClipDataset(DATA, "testbig")
    table = []
    for seq in ds.seqs:
        row = [(0.0, 0.0)]
        prev_img = None
        for t, rel in enumerate(seq["frames"]):
            img = load_img(DATA / rel)
            if t > 0 and prev_img is not None:
                g_prev = cv2.cvtColor((prev_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                g_cur = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                (ex, ey), _ = cv2.phaseCorrelate(g_prev.astype(np.float32),
                                                 g_cur.astype(np.float32))
                row.append((-float(ex), -float(ey)))
            prev_img = img
        table.append(row)
    return table


def persist_with_motion(streams, table, decay=0.9):
    """输出级持久化, 每帧回绕量取自运动表 (dx,dy)."""
    out = []
    for si, seq in enumerate(streams):
        carry = None
        frames = []
        for t, fr in enumerate(seq["frames"]):
            dets = {k: fr[k].clone() for k in ("boxes", "labels", "scores")}
            nf = dets["boxes"].shape[0]
            if t == 0:
                merged = {**dets, "age": torch.zeros(nf)}
            else:
                dx, dy = table[si][t]
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
            merged = {k: v[ok] for k, v in merged.items()}
            carry = {k: merged[k].clone() for k in ("boxes", "labels", "scores", "age")}
            frames.append({"boxes": merged["boxes"], "labels": merged["labels"],
                           "scores": merged["scores"],
                           "gt_boxes": fr["gt_boxes"], "gt_labels": fr["gt_labels"]})
        out.append({"stride": seq["stride"], "frames": frames})
    return out


def fisher_stab(a, b):
    from scipy import stats
    return float(stats.fisher_exact([[a["tp_kept"], a["tp_prev"] - a["tp_kept"]],
                                     [b["tp_kept"], b["tp_prev"] - b["tp_kept"]]])[1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = phase_motion_table()
    dxs = np.array([d for row in table for d, _ in row[1:]])
    dys = np.array([d for row in table for _, d in row[1:]])
    print(f"phase motion: mean dx={dxs.mean():.2f} (stride 128), std={dxs.std():.2f}, "
          f"mean dy={dys.mean():.2f} std={dys.std():.2f}, n={len(dxs)}")

    streams_off = torch.load(ROOT / "runs" / "eval_persistence" /
                             "stream_memoff_testbig.pt", weights_only=False)
    w = persist_with_motion(streams_off, table)
    r_w = aggregate(w, "wrapper+phase", compute_m=False)
    ph = json.loads((ROOT / "runs" / "eval_sweep_phase" / "results.json")
                    .read_text(encoding="utf-8"))
    tfm_ph = (ph.get("video_testbig_motion-phase_mem-on")
              or ph["video_testbig_motion-gt_mem-on"])
    p = json.loads((ROOT / "runs" / "eval_persistence" / "results.json")
                   .read_text(encoding="utf-8"))
    res = {"phase_table": {"mean_dx": float(dxs.mean()), "std_dx": float(dxs.std()),
                           "mean_dy": float(dys.mean()), "std_dy": float(dys.std()),
                           "n": int(len(dxs))},
           "wrapper_phase": r_w,
           "tfm_phase": {"TP_stability": tfm_ph["TP_stability"],
                         "tp_kept": tfm_ph["tp_kept"], "tp_prev": tfm_ph["tp_prev"]},
           "wrapper_clean": p["persist_inf"], "tfm_clean": p["sanity_memon"]}
    res["fisher_wrapper_phase_vs_tfm_phase"] = fisher_stab(r_w, tfm_ph)
    res["fisher_wrapper_phase_vs_tfm_clean"] = fisher_stab(r_w, p["sanity_memon"])
    (OUT / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[wrapper+phase] TP={r_w['TP_stability']:.4f} ({r_w['tp_kept']}/{r_w['tp_prev']}) "
          f"P={r_w['precision@0.5']:.3f} R={r_w['recall@0.5']:.3f}")
    print(f"[TFM+phase] TP={tfm_ph['TP_stability']:.4f} ({tfm_ph['tp_kept']}/{tfm_ph['tp_prev']})")
    print(f"Fisher wrapper+phase vs TFM+phase: p={res['fisher_wrapper_phase_vs_tfm_phase']:.3g}")
    print("saved ->", OUT / "results.json")


if __name__ == "__main__":
    main()
