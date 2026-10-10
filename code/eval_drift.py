# -*- coding: utf-8 -*-
"""M3 漂移（相关型）里程计噪声变体
==================================
现实打滑里程计误差是相关/漂移型, 而原 σ 扫描是逐帧 i.i.d. 高斯。
本脚本注入随机游走漂移: 每帧步进 e~N(0, sig_d^2) px (两轴), 序列内累积;
TFM 与输出级持久化包装器使用同一条漂移序列 (公平对照)。
- TFM ON: 逐帧 motion=(stride+cum_x, cum_y)
- wrapper: 逐帧 warp 偏移 (stride+cum_x, cum_y), 其余同 persist()
参照: OFF 官方流 (与运动无关) = 0.5064; 干净 ON = 0.7244
输出: runs/eval_drift/results.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import VideoClipDataset, load_img, load_yolo_label      # noqa: E402
from eval import load_model, ckpt_boxes_to_xyxy_px, pred_hits_gt, count_runs  # noqa: E402
from eval_persistence import aggregate, persist, CONF                # noqa: E402
from ops import postprocess                                          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "defect_ds"
OUT = ROOT / "runs" / "eval_drift"
SIGS = [4.0, 8.0]
SEED = 0


def build_drift(streams_off, sig):
    """每序列内随机游走累积偏移表: table[i][t]=(nx, ny), t=0 为 0."""
    rng = np.random.RandomState(SEED)
    table = []
    for seq in streams_off:
        cum = np.zeros(2)
        row = [(0.0, 0.0)]
        for _ in range(1, len(seq["frames"])):
            cum = cum + rng.normal(0.0, sig, size=2)
            row.append((float(cum[0]), float(cum[1])))
        table.append(row)
    return table


@torch.no_grad()
def collect_drift(model, dev, split, table):
    """TFM ON 采集: motion 加漂移 (与官方 collect 同流程, 仅 motion 不同)."""
    ds = VideoClipDataset(DATA, split)
    seqs = []
    for si, seq in enumerate(ds.seqs):
        memory = None
        frames = []
        for t, rel in enumerate(seq["frames"]):
            img = load_img(DATA / rel)
            H, W = img.shape[:2]
            x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(dev)
            motion = None
            if t > 0:
                nx, ny = table[si][t]
                motion = (float(seq["stride"]) + nx, ny)
            out = model(x, memory_state=memory, motion=motion, return_memory=True)
            pp = postprocess(out, W, H, score_thr=CONF)[0]
            boxes, labels = load_yolo_label(DATA / "labels" / seq["split"] /
                                            (Path(rel).stem + ".txt"), W, H)
            gt_xyxy = ckpt_boxes_to_xyxy_px(boxes, W, H) if boxes.numel() else boxes
            frames.append({
                "boxes": pp["boxes"].cpu(), "labels": pp["labels"].cpu(),
                "scores": pp["scores"].cpu(),
                "gt_boxes": gt_xyxy.cpu() if gt_xyxy.numel() else gt_xyxy,
                "gt_labels": labels.cpu()})
            if model.memory is not None:
                probs = out["pred_logits"].sigmoid()
                sm, lm = probs.max(dim=-1)
                memory = model.update_memory(memory, out["_mem_raw"], W, H, sm, lm)
        seqs.append({"stride": float(seq["stride"]), "frames": frames})
        if (si + 1) % 50 == 0:
            print(f"  drift collect {si+1}/{len(ds.seqs)}", flush=True)
    return seqs


def persist_drift(streams, table, decay=0.9):
    """输出级持久化 + 同一条漂移序列 (mirror eval_persistence.persist)."""
    out = []
    for si, seq in enumerate(streams):
        stride = seq["stride"]
        carry = None
        frames = []
        for t, fr in enumerate(seq["frames"]):
            dets = {k: fr[k].clone() for k in ("boxes", "labels", "scores")}
            nf = dets["boxes"].shape[0]
            if t == 0:
                merged = {**dets, "age": torch.zeros(nf)}
            else:
                nx, ny = table[si][t]
                mv = torch.tensor([stride + nx, ny, stride + nx, ny])
                cb = carry["boxes"] - mv
                cs = carry["scores"] * decay
                ca = carry["age"] + 1.0
                box = torch.cat([dets["boxes"], cb], 0)
                lab = torch.cat([dets["labels"], carry["labels"]], 0)
                sco = torch.cat([dets["scores"], cs], 0)
                age = torch.cat([torch.zeros(nf), ca], 0)
                fresh = torch.cat([torch.ones(nf), torch.zeros(cb.shape[0])])
                from eval_persistence import _dedup_fresh_first
                keep = _dedup_fresh_first(box, lab, sco, fresh, 0.6)
                merged = {"boxes": box[keep], "labels": lab[keep],
                          "scores": sco[keep], "age": age[keep]}
            ok = merged["scores"] >= CONF
            merged = {k: v[ok] for k, v in merged.items()}
            carry = {k: merged[k].clone() for k in ("boxes", "labels", "scores", "age")}
            frames.append({"boxes": merged["boxes"], "labels": merged["labels"],
                           "scores": merged["scores"],
                           "gt_boxes": fr["gt_boxes"], "gt_labels": fr["gt_labels"]})
        out.append({"stride": stride, "frames": frames})
    return out


def fisher_stab(a, b):
    from scipy import stats
    return float(stats.fisher_exact([[a["tp_kept"], a["tp_prev"] - a["tp_kept"]],
                                     [b["tp_kept"], b["tp_prev"] - b["tp_kept"]]])[1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(str(ROOT / "runs" / "e2e_s1" / "ep7.pt"), dev)
    print("model loaded", flush=True)

    streams_off = torch.load(ROOT / "runs" / "eval_persistence" /
                             "stream_memoff_testbig.pt", weights_only=False)
    r_off = aggregate(streams_off, "OFF-reference", compute_m=False)
    results = {"off_reference": r_off, "clean_on_ref": 0.7244, "sig_section": SIGS}
    assert abs(r_off["TP_stability"] - 0.5064) < 0.002, "OFF 参照不符官方 0.5064"

    for sig in SIGS:
        table = build_drift(streams_off, sig)
        on = collect_drift(model, dev, "testbig", table)
        r_on = aggregate(on, f"TFM-drift sd={sig}", compute_m=False)
        w = persist_drift(streams_off, table)
        r_w = aggregate(w, f"persist-drift sd={sig}", compute_m=False)
        rec = {"on_tfm": r_on, "wrapper": r_w,
               "fisher_tfm_vs_wrapper": fisher_stab(r_on, r_w),
               "fisher_tfm_vs_off": fisher_stab(r_on, r_off)}
        results[f"drift_sd{int(sig)}"] = rec
        print(f"  [sd={sig}] TFM {r_on['TP_stability']:.4f} vs wrapper {r_w['TP_stability']:.4f} "
              f"(p={rec['fisher_tfm_vs_wrapper']:.3g}); TFM vs OFF p={rec['fisher_tfm_vs_off']:.3g}",
              flush=True)
        torch.save(on, OUT / f"stream_on_drift_sd{int(sig)}.pt")

    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("saved ->", OUT / "results.json")


if __name__ == "__main__":
    main()
