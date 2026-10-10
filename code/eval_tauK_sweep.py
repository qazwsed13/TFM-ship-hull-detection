# -*- coding: utf-8 -*-
"""P1-2 补强: 记忆超参 τ×K 推理期扫描 (testbig, s1 ep7, 纯推理无重训)
- τ=model.memory.score_thr (token 准入阈值), K=model.memory.max_tokens (容量)
- 网格 τ∈{0.2,0.3,0.4,0.5} × K∈{24,48,96}; 每格采集 ON 流并聚合
- sanity: (0.3, 48) 应复现官方 0.7244 (容差 ±0.002)
输出: runs/sweep_tauK/results.json
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import load_model                                     # noqa: E402
from eval_persistence import collect, aggregate                  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "sweep_tauK"
TAUS = [0.2, 0.3, 0.4, 0.5]
KS = [24, 48, 96]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(str(ROOT / "runs" / "e2e_s1" / "ep7.pt"), dev)
    print("model loaded; defaults:", model.memory.score_thr, model.memory.max_tokens, flush=True)
    res = {}
    for tau in TAUS:
        for K in KS:
            model.memory.score_thr = float(tau)
            model.memory.max_tokens = int(K)
            t0 = time.time()
            streams = collect(model, dev, "testbig", use_memory=True)
            r = aggregate(streams, f"tau={tau} K={K}", compute_m=False)
            res[f"tau{tau}_K{K}"] = r
            print(f"[tau={tau} K={K}] TP={r['TP_stability']:.4f} ({r['tp_kept']}/{r['tp_prev']}) "
                  f"P={r['precision@0.5']:.3f} R={r['recall@0.5']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            # 增量落盘防中断
            (OUT / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    san = res["tau0.3_K48"]["TP_stability"]
    print(f"sanity (tau0.3,K48) = {san:.4f} vs official 0.7244")
    assert abs(san - 0.7244) < 0.002, "sanity 失败"
    print("saved ->", OUT / "results.json")


if __name__ == "__main__":
    main()
