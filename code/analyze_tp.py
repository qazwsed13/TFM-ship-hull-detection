# -*- coding: utf-8 -*-
"""TP 稳定性显著性分析: 读取 results.json, 输出 mem-on vs mem-off 的 Fisher 精确检验.
用法: python analyze_tp.py ../runs/eval_e2e_s1/results.json
"""
import json
import sys

import numpy as np
from scipy import stats


def run(path):
    r = json.loads(open(path, encoding="utf-8").read())
    rows = []
    for split in ["test", "test_hard", "testbig"]:
        on = r.get(f"video_{split}_motion-gt_mem-on")
        off = r.get(f"video_{split}_motion-gt_mem-off")
        if on and off:
            rows.append((split, on, off))
    if not rows:
        print("无视频结果"); return
    k1 = sum(o.get("tp_kept", 0) for _, o, _ in rows)
    n1 = sum(o.get("tp_prev", 0) for _, o, _ in rows)
    k2 = sum(o.get("tp_kept", 0) for _, _, o in rows)
    n2 = sum(o.get("tp_prev", 0) for _, _, o in rows)
    all_rows = rows + [("pooled", {"tp_kept": k1, "tp_prev": n1, "flicker_1frame": "-"},
                       {"tp_kept": k2, "tp_prev": n2, "flicker_1frame": "-"})]
    print(f"{'split':10s} | {'mem-on':>16s} | {'mem-off':>16s} | {'相对提升':>9s} | {'Fisher p':>8s}")
    print("-" * 72)
    for split, o, f in all_rows:
        r1 = o["tp_kept"] / max(o["tp_prev"], 1)
        r2 = f["tp_kept"] / max(f["tp_prev"], 1)
        table = np.array([[o["tp_kept"], o["tp_prev"] - o["tp_kept"]],
                          [f["tp_kept"], f["tp_prev"] - f["tp_kept"]]])
        _, p = stats.fisher_exact(table)
        lift = (r1 / r2 - 1) * 100 if r2 > 0 else float("nan")
        print(f"{split:10s} | {r1:.3f} ({o['tp_kept']:>3d}/{o['tp_prev']:<3d}) | "
              f"{r2:.3f} ({f['tp_kept']:>3d}/{f['tp_prev']:<3d}) | {lift:+8.1f}% | {p:8.4f}  "
              f"{'*' if p < 0.05 else ''}")
    print()
    for split, o, f in rows:
        print(f"[{split}] flicker: {o['flicker_1frame']} vs {f['flicker_1frame']} | "
              f"miss_runs: {o['miss_runs']} vs {f['miss_runs']} | "
              f"video mAP50: {o['mAP50']:.4f} vs {f['mAP50']:.4f}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "../runs/eval_e2e_s1/results.json")
