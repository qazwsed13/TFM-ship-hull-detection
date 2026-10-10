# -*- coding: utf-8 -*-
"""FPS 复测统计 + 相位相关耗时 (审稿意见 #5/#7)
- 5 组交替重复: 记忆关/开 各 measure_fps -> mean±std + 配对 t 检验
- cv2.phaseCorrelate 512² 单帧耗时 (100 次平均)
输出: runs/fps_stats.json
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import load_model, measure_fps                               # noqa: E402

RUNS = Path(__file__).resolve().parent.parent / "runs"


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(str(RUNS / "e2e_s1" / "ep7.pt"), dev)
    mem = [{"feat": torch.zeros(0, 256, device=dev), "box": torch.zeros(0, 4, device=dev),
            "score": torch.zeros(0, device=dev), "label": torch.zeros(0, dtype=torch.long, device=dev),
            "age": torch.zeros(0, device=dev)}]
    nm, wm = [], []
    for rep in range(5):
        nm.append(measure_fps(model, dev, n_run=100))
        wm.append(measure_fps(model, dev, n_run=100, memory_state=mem, motion=(128.0, 0.0)))
        print(f"rep{rep}: off={nm[-1]:.1f} on={wm[-1]:.1f}", flush=True)
    t = stats.ttest_rel(wm, nm)
    fps = {"off_mean": float(np.mean(nm)), "off_std": float(np.std(nm, ddof=1)),
           "on_mean": float(np.mean(wm)), "on_std": float(np.std(wm, ddof=1)),
           "paired_t_p": float(t.pvalue), "n_rep": 5,
           "off_all": nm, "on_all": wm}
    # 相位相关耗时 512x512
    g1 = (np.random.rand(512, 512) * 255).astype(np.float32)
    g2 = np.roll(np.roll(g1, 128, axis=1), 3, axis=0)
    t0 = time.perf_counter()
    for _ in range(100):
        cv2.phaseCorrelate(g1, g2)
    dt = (time.perf_counter() - t0) / 100 * 1000
    fps["phasecorr_ms_per_frame"] = dt
    (RUNS / "fps_stats.json").write_text(json.dumps(fps, indent=2), encoding="utf-8")
    print(f"FPS off={fps['off_mean']:.1f}±{fps['off_std']:.1f}  "
          f"on={fps['on_mean']:.1f}±{fps['on_std']:.1f}  paired p={t.pvalue:.3f}")
    print(f"phaseCorrelate: {dt:.2f} ms/frame (512x512)")
    print("saved ->", RUNS / "fps_stats.json")


if __name__ == "__main__":
    main()
