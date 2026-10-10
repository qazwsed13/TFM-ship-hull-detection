# -*- coding: utf-8 -*-
"""Minor 4: 噪声 σ5/σ15 下 TFM ON 的 precision/recall 补算 (i.i.d. 噪声, 与 eval.py 同序)
输出: runs/eval_noise_pr/results.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import VideoClipDataset, load_img, load_yolo_label      # noqa: E402
from eval import load_model, ckpt_boxes_to_xyxy_px                   # noqa: E402
from eval_persistence import aggregate, CONF                         # noqa: E402
from ops import postprocess                                          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "defect_ds"
OUT = ROOT / "runs" / "eval_noise_pr"


@torch.no_grad()
def collect_iid(model, dev, split, sigma):
    """镜像 eval_video: seed(0); 逐 (seq, t>0) 依次 randn*σ x/y."""
    np.random.seed(0)
    ds = VideoClipDataset(DATA, split)
    seqs = []
    for seq in ds.seqs:
        memory = None
        frames = []
        for t, rel in enumerate(seq["frames"]):
            img = load_img(DATA / rel)
            H, W = img.shape[:2]
            x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(dev)
            motion = None
            if t > 0:
                dx = float(seq["stride"]) + float(np.random.randn() * sigma)
                dy = float(np.random.randn() * sigma)
                motion = (dx, dy)
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
    return seqs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(str(ROOT / "runs" / "e2e_s1" / "ep7.pt"), dev)
    results = {}
    for sig in [5.0, 15.0]:
        streams = collect_iid(model, dev, "testbig", sig)
        r = aggregate(streams, f"TFM sigma={sig}", compute_m=False)
        results[f"sigma{int(sig)}"] = r
        print(f"  [sigma={sig}] TP={r['TP_stability']:.4f} P={r['precision@0.5']:.3f} "
              f"R={r['recall@0.5']:.3f}", flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("saved ->", OUT / "results.json")


if __name__ == "__main__":
    main()
