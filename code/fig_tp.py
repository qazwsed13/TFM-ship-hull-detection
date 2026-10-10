# -*- coding: utf-8 -*-
"""证据升级实验终图: e2e 模型 TP_stability mem-on vs mem-off (三域+合并, 带显著性)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "figures"


def load(name):
    return json.loads((ROOT / "runs" / name / "results.json").read_text(encoding="utf-8"))


def main():
    r7 = load("eval_e2e_ep7")
    r1 = load("eval_e2e_ep1")
    splits = ["test", "test_hard", "testbig"]
    labels = ["test", "test_hard", "testbig (200 seqs)", "pooled"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, r, title in [(axes[0], r7, "e2e ep7 (7 epochs memory training)"),
                         (axes[1], r1, "e2e ep1 (1 epoch memory training)")]:
        on_vals, off_vals, ps = [], [], []
        k1 = n1 = k2 = n2 = 0
        for sp in splits:
            on = r[f"video_{sp}_motion-gt_mem-on"]
            off = r[f"video_{sp}_motion-gt_mem-off"]
            on_vals.append(on["tp_kept"] / on["tp_prev"])
            off_vals.append(off["tp_kept"] / off["tp_prev"])
            k1 += on["tp_kept"]; n1 += on["tp_prev"]
            k2 += off["tp_kept"]; n2 += off["tp_prev"]
        on_vals.append(k1 / n1); off_vals.append(k2 / n2)
        from scipy import stats
        x = np.arange(len(labels))
        ax.bar(x - 0.2, on_vals, 0.38, label="memory ON", color="#d84341")
        ax.bar(x + 0.2, off_vals, 0.38, label="memory OFF", color="#3b7dd8")
        # p 值标注
        for i, sp in enumerate(splits):
            on = r[f"video_{sp}_motion-gt_mem-on"]
            off = r[f"video_{sp}_motion-gt_mem-off"]
            table = np.array([[on["tp_kept"], on["tp_prev"] - on["tp_kept"]],
                              [off["tp_kept"], off["tp_prev"] - off["tp_kept"]]])
            _, p = stats.fisher_exact(table)
            ptxt = "p<0.001" if p < 0.001 else f"p={p:.3f}"
            ax.text(i, max(on_vals[i], off_vals[i]) + 0.02, ptxt, ha="center", fontsize=7.5)
        table = np.array([[k1, n1 - k1], [k2, n2 - k2]])
        _, p = stats.fisher_exact(table)
        ax.text(len(labels) - 1, max(on_vals[-1], off_vals[-1]) + 0.02,
                "p<0.001", ha="center", fontsize=7.5)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=8.5)
        ax.set_ylim(0, 0.95)
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("TP stability  P(hit$_t$ | hit$_{t-1}$)")
    axes[0].legend(loc="upper right", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(OUT / "e2e_tp_stability.png", dpi=160)
    print("saved", OUT / "e2e_tp_stability.png")


if __name__ == "__main__":
    main()
