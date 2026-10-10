# -*- coding: utf-8 -*-
"""M2 密度对照：低阈值重采集 testbig ON/OFF 流（s1 ep7），带来源标签。
- score_thr=0.05, topk=300 保存原始分数；下游离线重定阈
- src = 原始查询行号（>=100 即记忆 token 输出；OFF 恒为内容查询）
- 与 eval_persistence.collect 同流程（官方数字复现口径）
输出: runs/density_control/stream_memoff_lowthr.pt, stream_memon_lowthr.pt
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import VideoClipDataset, load_img, load_yolo_label      # noqa: E402
from eval import load_model, ckpt_boxes_to_xyxy_px                   # noqa: E402
from ops import cxcywh_to_xyxy                                       # noqa: E402
from model import bbox_iou_px  # noqa: F401                          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "defect_ds"
OUTDIR = ROOT / "runs" / "density_control"
LOWTHR, TOPK = 0.05, 300


@torch.no_grad()
def collect(model, dev, split="testbig", use_memory=False):
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
            logits = out["pred_logits"]                                    # [1,Q,C]
            probs = logits.sigmoid()
            scores, labels = probs.max(dim=-1)                             # [1,Q]
            boxes_px = cxcywh_to_xyxy(out["pred_boxes"]) * torch.tensor(
                [W, H, W, H], dtype=out["pred_boxes"].dtype,
                device=out["pred_boxes"].device)
            s, l, bb = scores[0], labels[0], boxes_px[0]
            keep = s > LOWTHR
            idx = keep.nonzero(as_tuple=True)[0]
            s, l, bb = s[keep], l[keep], bb[keep]
            if s.numel() > TOPK:
                tk = s.topk(TOPK).indices
                s, l, bb, idx = s[tk], l[tk], bb[tk], idx[tk]
            boxes, labels_gt = load_yolo_label(DATA / "labels" / seq["split"] /
                                               (Path(rel).stem + ".txt"), W, H)
            gt_xyxy = ckpt_boxes_to_xyxy_px(boxes, W, H) if boxes.numel() else boxes
            frames.append({
                "boxes": bb.cpu(), "labels": l.cpu(), "scores": s.cpu(),
                "src": idx.cpu(),                       # 查询行号 (>=100 → 记忆 token)
                "gt_boxes": gt_xyxy.cpu() if gt_xyxy.numel() else gt_xyxy,
                "gt_labels": labels_gt.cpu()})
            if use_memory and model.memory is not None:
                sm, lm = probs.max(dim=-1)
                memory = model.update_memory(memory, out["_mem_raw"], W, H, sm, lm)
        seqs.append({"stride": float(seq["stride"]), "frames": frames})
        if (si + 1) % 50 == 0:
            print(f"  lowthr collect(mem={'on' if use_memory else 'off'}) {si+1}/{len(ds.seqs)}",
                  flush=True)
    return seqs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs" / "e2e_s1" / "ep7.pt"))
    ap.add_argument("--tag", default="")          # 输出后缀, 如 "_s2"
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.ckpt, dev)
    print("model loaded", args.ckpt, flush=True)
    off = collect(model, dev, "testbig", use_memory=False)
    torch.save(off, OUTDIR / f"stream_memoff_lowthr{args.tag}.pt")
    print("saved off", flush=True)
    on = collect(model, dev, "testbig", use_memory=True)
    torch.save(on, OUTDIR / f"stream_memon_lowthr{args.tag}.pt")
    print("saved on", flush=True)


if __name__ == "__main__":
    main()
