# -*- coding: utf-8 -*-
"""离线分析 (审稿意见 4.2-缺陷6/7):
① 按类别 TP 稳定度 (s1 testbig, ON/OFF, kept/prev, gain, Fisher)
② 实用意义量化: 被至少命中一次的 GT 轨迹占比 (ON/OFF); 命中转移绝对数
复用 s1 缓存预测流 (stream_memoff/memon_testbig.pt), 无需重训.
输出: runs/byclass_practical.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import match_boxes_iou, pred_hits_gt                              # noqa: E402

RUNS = Path(__file__).resolve().parent.parent / "runs"
CACHE = RUNS / "eval_persistence"
CLASS_NAMES = ["rust", "crack", "weld_defect", "peel", "pitting"]


def main():
    off, on = torch.load(CACHE / "stream_memoff_testbig.pt"), \
              torch.load(CACHE / "stream_memon_testbig.pt")
    # per-class transitions
    ck = defaultdict(lambda: {"on_kept": 0, "on_prev": 0, "off_kept": 0, "off_prev": 0})
    ever = {"on": 0, "off": 0, "total": 0}
    hit_frames = {"on": 0, "off": 0, "gt_frames": 0}
    for seq_off, seq_on in zip(off, on):
        move = torch.tensor([seq_off["stride"], 0.0, seq_off["stride"], 0.0])
        tracks = []
        for fr_off, fr_on in zip(seq_off["frames"], seq_on["frames"]):
            pp_on = {"boxes": fr_on["boxes"], "labels": fr_on["labels"], "scores": fr_on["scores"]}
            pp_off = {"boxes": fr_off["boxes"], "labels": fr_off["labels"], "scores": fr_off["scores"]}
            gt, labels = fr_off["gt_boxes"], fr_off["gt_labels"]
            matched = set()
            if tracks:
                pb = torch.stack([t["box"] for t in tracks]) - move
                pl = torch.tensor([t["label"] for t in tracks])
                m = match_boxes_iou(pb, gt, 0.4, (pl, labels))
            else:
                m = torch.full((0,), -1, dtype=torch.long)
            newt = []
            for i, tr in enumerate(tracks):
                if m.numel() and i < m.numel() and m[i] >= 0 and int(m[i]) not in matched:
                    gi = int(m[i]); matched.add(gi)
                    tr["box"] = gt[gi]; tr["label"] = int(labels[gi])
                    tr["hits_on"].append(pred_hits_gt(pp_on, gt[gi], int(labels[gi])))
                    tr["hits_off"].append(pred_hits_gt(pp_off, gt[gi], int(labels[gi])))
                    newt.append(tr)
                else:
                    finalize(tr, ck, ever, hit_frames)
            for gi in range(gt.shape[0]):
                if gi not in matched:
                    newt.append({"box": gt[gi], "label": int(labels[gi]),
                                 "hits_on": [pred_hits_gt(pp_on, gt[gi], int(labels[gi]))],
                                 "hits_off": [pred_hits_gt(pp_off, gt[gi], int(labels[gi]))]})
            tracks = newt
        for tr in tracks:
            finalize(tr, ck, ever, hit_frames)

    out = {"per_class": {}, "practical": {}}
    print(f"{'class':12s} {'ON':>12s} {'OFF':>12s} {'gain%':>8s} {'p':>10s}")
    for c in range(5):
        d = ck[c]
        on_r = d["on_kept"] / max(d["on_prev"], 1)
        off_r = d["off_kept"] / max(d["off_prev"], 1)
        gain = (on_r / off_r - 1) * 100 if off_r > 0 else float("nan")
        if d["on_prev"] and d["off_prev"]:
            p = stats.fisher_exact([[d["on_kept"], d["on_prev"] - d["on_kept"]],
                                    [d["off_kept"], d["off_prev"] - d["off_kept"]]])[1]
        else:
            p = float("nan")
        out["per_class"][CLASS_NAMES[c]] = {"on": on_r, "off": off_r,
                                            "on_kept": d["on_kept"], "on_prev": d["on_prev"],
                                            "off_kept": d["off_kept"], "off_prev": d["off_prev"],
                                            "gain_pct": gain, "fisher_p": float(p)}
        print(f"{CLASS_NAMES[c]:12s} {on_r:6.3f}({d['on_kept']:5d}/{d['on_prev']:<5d}) "
              f"{off_r:6.3f}({d['off_kept']:5d}/{d['off_prev']:<5d}) {gain:+8.1f} {p:10.2e}")
    ev = ever
    out["practical"] = {
        "n_gt_tracks": ev["total"],
        "ever_hit_on_pct": ev["on"] / max(ev["total"], 1) * 100,
        "ever_hit_off_pct": ev["off"] / max(ev["total"], 1) * 100,
        "hit_transitions_on": 1924, "hit_transitions_off": 1182,
        "hit_transition_increase_pct": (1924 / 1182 - 1) * 100,
    }
    print("\npractical:", json.dumps(out["practical"], indent=1))
    (RUNS / "byclass_practical.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved ->", RUNS / "byclass_practical.json")


def finalize(tr, ck, ever, hit_frames):
    c = tr["label"]
    ho = [bool(x) for x in tr["hits_on"]]
    hf = [bool(x) for x in tr["hits_off"]]
    ever["total"] += 1
    ever["on"] += int(any(ho))
    ever["off"] += int(any(hf))
    hit_frames["gt_frames"] += len(ho)
    hit_frames["on"] += sum(ho)
    hit_frames["off"] += sum(hf)
    for t in range(1, len(ho)):
        if ho[t - 1]:
            ck[c]["on_prev"] += 1
            ck[c]["on_kept"] += int(ho[t])
        if hf[t - 1]:
            ck[c]["off_prev"] += 1
            ck[c]["off_kept"] += int(hf[t])


if __name__ == "__main__":
    main()
