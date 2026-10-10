# -*- coding: utf-8 -*-
"""TFM+phase 的 precision/recall 补算 (对称于 wrapper+phase)."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import VideoClipDataset, load_img, load_yolo_label      # noqa: E402
from eval import load_model, ckpt_boxes_to_xyxy_px                   # noqa: E402
from eval_persistence import aggregate, CONF                         # noqa: E402
from eval_wrapper_phase import phase_motion_table                    # noqa: E402
from ops import postprocess                                          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "defect_ds"
OUT = ROOT / "runs" / "eval_phase_pr"


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(str(ROOT / "runs" / "e2e_s1" / "ep7.pt"), dev)
    table = phase_motion_table()
    ds = VideoClipDataset(DATA, "testbig")
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
                dx, dy = table[si][t]
                motion = (dx, dy)
            out = model(x, memory_state=memory, motion=motion, return_memory=True)
            pp = postprocess(out, W, H, score_thr=CONF)[0]
            boxes, labels = load_yolo_label(DATA / "labels" / seq["split"] /
                                            (Path(rel).stem + ".txt"), W, H)
            gt_xyxy = ckpt_boxes_to_xyxy_px(boxes, W, H) if boxes.numel() else boxes
            frames.append({"boxes": pp["boxes"].cpu(), "labels": pp["labels"].cpu(),
                           "scores": pp["scores"].cpu(),
                           "gt_boxes": gt_xyxy.cpu() if gt_xyxy.numel() else gt_xyxy,
                           "gt_labels": labels.cpu()})
            if model.memory is not None:
                probs = out["pred_logits"].sigmoid()
                sm, lm = probs.max(dim=-1)
                memory = model.update_memory(memory, out["_mem_raw"], W, H, sm, lm)
        seqs.append({"stride": float(seq["stride"]), "frames": frames})
        if (si + 1) % 50 == 0:
            print(f"  phase-PR collect {si+1}/{len(ds.seqs)}", flush=True)
    r = aggregate(seqs, "TFM+phase-PR", compute_m=False)
    (OUT / "results.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    print(f"[TFM+phase PR] TP={r['TP_stability']:.4f} P={r['precision@0.5']:.3f} "
          f"R={r['recall@0.5']:.3f}")
    print("saved ->", OUT / "results.json")


if __name__ == "__main__":
    main()
