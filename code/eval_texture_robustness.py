# -*- coding: utf-8 -*-
"""纹理杂波鲁棒性探针 (响应保真度质疑):
在评测帧上注入标定到"真实纹理统计区间"的杂波 (乘性分形 + 加性高斯 + 斑点),
在记忆开/关下复测 TP 稳定度增益 —— 无需重训, 检验机制对纹理粗糙度的敏感性.

标定目标: 边缘密度 ~20-35% (真实 26.7-46.4%), GLCM 对比度向 1+ 靠拢 (真实 1.23-2.25).
输出: runs/eval_texture_robustness/results.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import load_model                                                # noqa: E402
from eval_persistence import collect, persist, aggregate                    # noqa: E402
from dataset import load_img                                               # noqa: E402
from gen_dataset import fractal_noise                                      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
DATA = ROOT / "data" / "defect_ds"


def perturb(img, rng, amp, sig):
    """img HWC float [0,1] RGB -> 粗糙度探针: 乘性分形 + 加性高斯 + 斑点."""
    h, w = img.shape[:2]
    f = fractal_noise((h, w), rng, octaves=5, base=6)
    f = (f - f.mean()) / (f.std() + 1e-6)
    out = img * (1.0 + amp * f)[..., None]
    speckle = rng.random((h, w, 1)).astype(np.float32)
    out = out * (1.0 - 0.15 * speckle)
    out = out + rng.normal(0, sig, out.shape).astype(np.float32)
    return np.clip(out, 0, 1)


def calibrate():
    """在 8 帧上标定两档参数, 使边缘密度进入真实区间."""
    imgs = sorted((DATA / "images" / "testbig").glob("*.jpg"))[:8]
    grid = {
        "L1_moderate": (0.10, 8.0 / 255),
        "L2_heavy": (0.22, 16.0 / 255),
    }
    report = {}
    for name, (amp, sig) in grid.items():
        rs = np.random.RandomState(0)
        eds, cons = [], []
        for p in imgs:
            g0 = cv2.cvtColor((load_img(p) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            img = perturb(load_img(p), rs, amp, sig)
            g = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            g = cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA)
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx ** 2 + gy ** 2)
            eds.append(float((mag > 40).mean() * 100))
            g32 = (g // 8).astype(np.int32)
            ii, jj = np.mgrid[0:32, 0:32]
            a = g32[:, :-1].ravel(); b = g32[:, 1:].ravel()
            m = np.zeros((32, 32)); np.add.at(m, (a, b), 1); m = m + m.T
            m /= m.sum()
            cons.append(float((m * (ii - jj) ** 2).sum()))
        report[name] = {"edge_density_pct": float(np.mean(eds)),
                        "glcm_contrast": float(np.mean(cons)),
                        "amp": amp, "sig": sig}
        print(f"calib {name}: edge={report[name]['edge_density_pct']:.1f}% "
              f"contrast={report[name]['glcm_contrast']:.2f}")
    return report


@torch.no_grad()
def collect_perturbed(model, dev, amp, sig, split="testbig", score_thr=0.3, use_memory=False):
    """复刻 eval_persistence.collect, 但逐帧注入标定杂波 (按序列确定性种子)."""
    from dataset import VideoClipDataset
    from eval import ckpt_boxes_to_xyxy_px, load_yolo_label
    from ops import postprocess
    ds = VideoClipDataset(DATA, split)
    seqs = []
    for si, seq in enumerate(ds.seqs):
        rs = np.random.RandomState(1234 + si)
        memory = None
        frames = []
        for t, rel in enumerate(seq["frames"]):
            raw = load_img(DATA / rel)
            img = perturb(raw, rs, amp, sig)
            H, W = img.shape[:2]
            x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(dev)
            motion = (float(seq["stride"]), 0.0) if (use_memory and t > 0) else None
            out = model(x, memory_state=memory, motion=motion, return_memory=True)
            pp = postprocess(out, W, H, score_thr=score_thr)[0]
            boxes, labels = load_yolo_label(DATA / "labels" / seq["split"] /
                                            (Path(rel).stem + ".txt"), W, H)
            gt = ckpt_boxes_to_xyxy_px(boxes, W, H) if boxes.numel() else boxes
            frames.append({"boxes": pp["boxes"].cpu(), "labels": pp["labels"].cpu(),
                           "scores": pp["scores"].cpu(),
                           "gt_boxes": gt.cpu() if gt.numel() else gt,
                           "gt_labels": labels.cpu()})
            if use_memory and model.memory is not None:
                probs = out["pred_logits"].sigmoid()
                sm, lm = probs.max(dim=-1)
                memory = model.update_memory(memory, out["_mem_raw"], W, H, sm, lm)
        seqs.append({"stride": float(seq["stride"]), "frames": frames})
        if (si + 1) % 50 == 0:
            print(f"  ... {si+1}/{len(ds.seqs)}", flush=True)
    return seqs


def main():
    dev = torch.device("cuda")
    calib = calibrate()
    model, _ = load_model(str(RUNS / "e2e_s1" / "ep7.pt"), dev)
    results = {"calibration": calib,
               "clean_reference": {"tfm_on": 0.7244, "tfm_off": 0.5064, "gain_pct": 43.0}}
    OUT = RUNS / "eval_texture_robustness"
    OUT.mkdir(exist_ok=True)
    for name, (amp, sig) in [("L1_moderate", (0.10, 8.0 / 255)),
                             ("L2_heavy", (0.22, 16.0 / 255))]:
        cache = OUT / f"streams_{name}.pt"
        if cache.exists():
            off, on = torch.load(cache)
        else:
            off = collect_perturbed(model, dev, amp, sig, use_memory=False)
            on = collect_perturbed(model, dev, amp, sig, use_memory=True)
            torch.save((off, on), cache)
        r_on = aggregate(on, f"{name} mem-on", compute_m=False)
        r_off = aggregate(off, f"{name} mem-off", compute_m=False)
        gain = (r_on["TP_stability"] / r_off["TP_stability"] - 1) * 100
        from scipy import stats
        p = stats.fisher_exact([[r_on["tp_kept"], r_on["tp_prev"] - r_on["tp_kept"]],
                                [r_off["tp_kept"], r_off["tp_prev"] - r_off["tp_kept"]]])[1]
        results[name] = {"on": r_on["TP_stability"], "off": r_off["TP_stability"],
                         "gain_pct": gain, "fisher_p": float(p),
                         "flicker_on": r_on["flicker_1frame"], "flicker_off": r_off["flicker_1frame"]}
        print(f"[{name}] TP on={r_on['TP_stability']:.4f} off={r_off['TP_stability']:.4f} "
              f"gain={gain:+.1f}% p={p:.2e}")
    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    print("saved ->", OUT / "results.json")


if __name__ == "__main__":
    main()
