# -*- coding: utf-8 -*-
"""Minor 5: FPS 复测 v2 --- n=10 交替重复 + 配对差值 95% CI + TOST 等效检验 (±5 FPS).
输出: runs/fps_stats2.json
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
from eval import load_model, measure_fps  # noqa: E402

RUNS = Path(__file__).resolve().parent.parent / "runs"
MARGIN = 5.0   # TOST 等效界 (FPS)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(str(RUNS / "e2e_s1" / "ep7.pt"), dev)
    mem = [{"feat": torch.zeros(0, 256, device=dev), "box": torch.zeros(0, 4, device=dev),
            "score": torch.zeros(0, device=dev), "label": torch.zeros(0, dtype=torch.long, device=dev),
            "age": torch.zeros(0, device=dev)}]
    nm, wm = [], []
    for rep in range(10):
        nm.append(float(measure_fps(model, dev, n_run=100)))
        wm.append(float(measure_fps(model, dev, n_run=100, memory_state=mem, motion=(128.0, 0.0))))
        print(f"rep{rep}: off={nm[-1]:.1f} on={wm[-1]:.1f}", flush=True)
    diff = np.array(wm) - np.array(nm)
    n = len(diff)
    md, sd = float(diff.mean()), float(diff.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    ci = (md - tcrit * se, md + tcrit * se)
    t = stats.ttest_rel(wm, nm)
    # TOST: H0a: diff <= -m ; H0b: diff >= +m
    p1 = float(stats.t.sf((md + MARGIN) / se, n - 1))
    p2 = float(stats.t.cdf((md - MARGIN) / se, n - 1))
    tost_p = max(p1, p2)
    out = {"off_mean": float(np.mean(nm)), "off_std": float(np.std(nm, ddof=1)),
           "on_mean": float(np.mean(wm)), "on_std": float(np.std(wm, ddof=1)),
           "diff_mean": md, "diff_ci95": [ci[0], ci[1]], "paired_t_p": float(t.pvalue),
           "tost_margin": MARGIN, "tost_p": tost_p, "n_rep": n,
           "off_all": nm, "on_all": wm}
    g1 = (np.random.rand(512, 512) * 255).astype(np.float32)
    g2 = np.roll(np.roll(g1, 128, axis=1), 3, axis=0)
    t0 = time.perf_counter()
    for _ in range(100):
        cv2.phaseCorrelate(g1, g2)
    out["phasecorr_ms_per_frame"] = (time.perf_counter() - t0) / 100 * 1000
    (RUNS / "fps_stats2.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"off={out['off_mean']:.1f}±{out['off_std']:.1f}  on={out['on_mean']:.1f}±{out['on_std']:.1f}")
    print(f"diff={md:+.2f} CI95=[{ci[0]:+.2f},{ci[1]:+.2f}]  t p={t.pvalue:.3f}  TOST(±{MARGIN}) p={tost_p:.3f}")
    print(f"phasecorr {out['phasecorr_ms_per_frame']:.2f} ms")
    print("saved ->", RUNS / "fps_stats2.json")


if __name__ == "__main__":
    main()
