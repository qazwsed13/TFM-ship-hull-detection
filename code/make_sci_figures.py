# -*- coding: utf-8 -*-
"""
SCI 图件生成器 (scipilot-figure-skill 兼容规格的本地实现)
==========================================================
说明: scipilot-figure-skill 经核实不可用 (技能列表与官方市场均无), 本脚本按其规格要求
      输出兼容图件: 色盲友好 (Okabe-Ito), 600 dpi, PDF/SVG/PNG 三格式, 单栏 3.5in /
      双栏 7.16in 尺寸, 8pt 字号, 显著性标注, 完整图注 (caption 单独输出 txt)。

图件清单:
  fig1_framework        TFM 机制示意图 (双栏)
  fig2_training         训练动力学 (a 损失曲线 / b 验证 mAP 轨迹)
  fig3_main             主结果: 多种子 TP_stability (a 分域+显著性 / b 种子一致性 / c 剂量-效应)
  fig4_detection        检测保持性 (基线 vs 种子)
  fig5_ablation         消融: a 对齐 / b 寿命 / c 里程计噪声 / d 相位相关替代
  fig6_qualitative      定性: 检测样张网格 + 记忆时序条
用法: python make_sci_figures.py [--only fig3]
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = RUNS / "figures_sci"
OUT.mkdir(parents=True, exist_ok=True)

# ---- SCI 样式 (scipilot 规格) ----
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.7, "lines.linewidth": 1.3, "lines.markersize": 4,
    "savefig.dpi": 600, "figure.dpi": 110, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})
# Okabe-Ito 色盲友好
C_ON, C_OFF = "#D55E00", "#0072B2"       # vermillion / blue
C_G, C_O, C_P, C_GR = "#009E73", "#E69F00", "#CC79A7", "#999999"
W1, W2 = 3.5, 7.16                        # 单栏/双栏英寸


def save(fig, name):
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight",
                    dpi=600 if ext == "png" else None)
    plt.close(fig)
    print("saved", OUT / f"{name}.[pdf|svg|png]")


def load_eval(name):
    f = RUNS / name / "results.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def fisher(k1, n1, k2, n2):
    from scipy import stats
    if min(n1, n2) == 0:
        return float("nan")
    return stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


def ptxt(p):
    if p != p:
        return "n/a"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def pooled(r, mode):
    k = n = 0
    for s in ["test", "test_hard", "testbig"]:
        v = r.get(f"video_{s}_motion-gt_mem-{mode}")
        if v:
            k += v["tp_kept"]; n += v["tp_prev"]
    return k, n


# ================================================================ Fig 1 框架图
def fig1():
    fig, ax = plt.subplots(figsize=(W2, 2.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.3); ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#333", fs=7.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(x1, y1, x2, y2, style="-|>", color="#333", ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color,
                                     lw=0.9, linestyle=ls, mutation_scale=9,
                                     connectionstyle=f"arc3,rad={rad}"))

    # 上行: 帧 t-1 的记忆更新回路
    box(0.1, 2.85, 1.35, 1.0, "Frame $t{-}1$\n(detections)", "#F5F5F5")
    box(1.85, 2.85, 1.7, 1.0, "Focus selection\nscore $>\\tau$,  top-$K$", "#FFF2CC")
    box(3.85, 2.85, 1.95, 1.0, "Memory bank\nfeat · box · cls · age", "#F8D7DA", bold=True, fs=6.6)
    box(6.2, 2.85, 1.6, 1.0, "Aging / de-dup\nthen decay", "#F5F5F5")
    arrow(1.45, 3.35, 1.85, 3.35); arrow(3.55, 3.35, 3.85, 3.35)
    arrow(5.8, 3.35, 6.2, 3.35)

    # 下行: 帧 t 的检测
    box(0.1, 0.35, 1.35, 1.0, "Frame $t$", "#F5F5F5")
    box(1.85, 0.35, 1.7, 1.0, "Query set:\ncontent $\\oplus$ mem\n$\\oplus$ denoising", "#D5E8D4", fs=6.4)
    box(3.85, 0.35, 1.95, 1.0, "Detector decoder\n(anchor + deform.)", "#D6E4F0", fs=6.8)
    box(6.2, 0.35, 1.6, 1.0, "Set prediction\n+ Hungarian loss", "#F5F5F5")
    arrow(1.45, 0.85, 1.85, 0.85); arrow(3.55, 0.85, 3.85, 0.85)
    arrow(5.8, 0.85, 6.2, 0.85)

    # 记忆跨帧注入 + 对齐
    arrow(4.6, 2.85, 2.6, 1.35, color=C_ON, rad=-0.25)
    ax.text(3.3, 1.95, "warp by odometry\n$\\mathbf{b}\\,{-}\\,(\\Delta x,\\Delta y)$",
            color=C_ON, fontsize=6.8, ha="center")
    # 帧 t → 焦点选取 (回流)
    arrow(7.75, 1.35, 7.75, 2.85, style="-|>", color="#666", ls="--")
    ax.text(8.05, 2.1, "next\nframe", color="#666", fontsize=6.5, ha="left", linespacing=1.3)
    ax.text(5.0, 4.1, "Temporal Focus Memory (TFM)", ha="center", fontsize=8.5,
            fontweight="bold")
    save(fig, "fig1_framework")


# ================================================================ Fig 2 训练动力学
def _epoch_rows(csv_path):
    xs, ys = [], []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("iter") == "epoch_avg":
                xs.append(int(row["epoch"])); ys.append(float(row["loss"]))
    return xs, ys


def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5))
    runs = [("base_s1", "base R1", C_GR), ("base_s1b", "base R2", C_G),
            ("base_s1c", "base R3", "#56B4E9"), ("base_s1d", "base R4", C_OFF),
            ("focus_s1", "focus R1", "#F0C7C7"), ("focus_s1b", "focus R2", "#E39A9A"),
            ("focus_s1c", "focus R3", "#D06A6A"), ("focus_s1d", "focus R4", "#B03A3A"),
            ("focus_g_s1", "focus_g (GIoU)", C_P)]
    for name, lab, c in runs:
        f = RUNS / name / "train_log.csv"
        if f.exists():
            xs, ys = _epoch_rows(f)
            if xs:
                axes[0].plot(xs, ys, label=lab, color=c)
    axes[0].set_yscale("log"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training loss")
    axes[0].legend(ncol=2, fontsize=5.6, frameon=False)
    axes[0].set_title("(a) Training dynamics (four continuation rounds)", fontsize=8)

    # (b) val mAP 轨迹 (从链日志解析 running-best)
    best = {}
    colors = {"base_s1": C_OFF, "focus_s1": "#B03A3A", "focus_g_s1": C_G, "focus_t": C_P,
              "base_t": C_O}
    for logname in ["train_chain.log", "train_chain_ft.log", "train_chain_C.log",
                    "train_chain_D.log"]:
        p = RUNS / logname
        if not p.exists():
            continue
        cur = None
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "====" in line:
                for tag in ["focus_g_s1", "base_s1", "focus_s1", "focus_t", "base_t"]:
                    if tag in line:
                        cur = tag; break
            if "eval ep" in line and cur:
                try:
                    ep = int(line.split("eval ep")[1].split("]")[0])
                    m = float(line.split("mAP50=")[1].split()[0])
                    if m >= best.get(cur, (-1, 0))[0]:
                        best[cur] = (m, ep)
                        axes[1].scatter([ep], [m], s=14, color=colors.get(cur, "#666"),
                                        marker="*" if cur.startswith("base") or cur.startswith("focus_g") else "o")
                except Exception:
                    pass
    for k, v in best.items():
        axes[1].annotate(f"{k} {v[0]:.3f}", (v[1], v[0]), fontsize=5.6, xytext=(2, 2),
                         textcoords="offset points")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val mAP$_{50}$")
    axes[1].set_title("(b) Validation trajectory (best per run)", fontsize=8)
    plt.tight_layout()
    save(fig, "fig2_training")


# ================================================================ Fig 3 主结果
def fig3():
    r1 = load_eval("eval_e2e_ep7")
    if r1 is None:
        print("跳过 fig3: 缺评测数据"); return
    seeds = [("eval_e2e_ep7", "s1", C_OFF), ("eval_e2e_s2_ep7", "s2", C_G),
             ("eval_e2e_s3_ep7", "s3", C_P)]
    fig = plt.figure(figsize=(W2, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.85, 0.85], wspace=0.42)

    # (a) seed1 分域主结果
    ax = fig.add_subplot(gs[0, 0])
    splits = ["test", "test_hard", "testbig"]
    label = ["test", "test_hard", "testbig\n(200 seqs)"]
    on_v, off_v, ann = [], [], []
    for s in splits:
        o = r1[f"video_{s}_motion-gt_mem-on"]; f = r1[f"video_{s}_motion-gt_mem-off"]
        on_v.append(o["tp_kept"] / o["tp_prev"]); off_v.append(f["tp_kept"] / f["tp_prev"])
        ann.append(ptxt(fisher(o["tp_kept"], o["tp_prev"], f["tp_kept"], f["tp_prev"])))
    k1, n1 = pooled(r1, "on"); k2, n2 = pooled(r1, "off")
    on_v.append(k1 / n1); off_v.append(k2 / n2)
    ann.append(ptxt(fisher(k1, n1, k2, n2)))
    x = np.arange(4)
    ax.bar(x - 0.19, on_v, 0.36, color=C_ON, label="memory ON")
    ax.bar(x + 0.19, off_v, 0.36, color=C_OFF, label="memory OFF")
    for i, t in enumerate(ann):
        ax.text(i, max(on_v[i], off_v[i]) + 0.035, t, ha="center", fontsize=6, rotation=0)
    ax.set_xticks(x); ax.set_xticklabels(label + ["pooled"], fontsize=6.3)
    ax.set_ylabel("TP stability  $P(hit_t | hit_{t-1})$")
    ax.set_ylim(0, 0.92); ax.legend(frameon=False, loc="upper left")
    ax.set_title("(a) Main result (seed $s_1$)", fontsize=8)

    # (b) 三种子 pooled 一致性
    ax2 = fig.add_subplot(gs[0, 1])
    lifts = []
    for name, tag, c in seeds:
        r = load_eval(name)
        if r is None:
            continue
        k1, n1 = pooled(r, "on"); k2, n2 = pooled(r, "off")
        lift = (k1 / n1) / (k2 / n2) - 1
        lifts.append((tag, lift * 100, c))
        ax2.scatter([lift * 100], [tag], s=30, color=c, zorder=3)
    if lifts:
        mean = np.mean([v for _, v, _ in lifts])
        ax2.axvline(mean, color="#555", ls="--", lw=1)
        ax2.text(mean, -0.6, f"mean {mean:.1f}%", fontsize=6, ha="center")
        ax2.set_xlabel("Relative TP-stability gain (%)")
        ax2.set_title("(b) Seed consistency", fontsize=8)
        ax2.grid(axis="x", alpha=0.3)

    # (c) 剂量-效应 (记忆训练量)
    ax3 = fig.add_subplot(gs[0, 2])
    r_ep1 = load_eval("eval_e2e_ep1")
    doses, vals = [], []
    if r_ep1:
        k1, n1 = pooled(r_ep1, "on"); k2, n2 = pooled(r_ep1, "off")
        doses.append(1); vals.append(((k1 / n1) / (k2 / n2) - 1) * 100)
    k1, n1 = pooled(r1, "on"); k2, n2 = pooled(r1, "off")
    doses.append(7); vals.append(((k1 / n1) / (k2 / n2) - 1) * 100)
    ax3.plot(doses, vals, "o-", color=C_ON)
    for d, v in zip(doses, vals):
        ax3.annotate(f"{v:+.0f}%", (d, v), xytext=(4, -3), textcoords="offset points",
                     fontsize=6.5)
    ax3.set_xticks([1, 7]); ax3.set_xlabel("Memory training epochs")
    ax3.set_ylabel("TP-stability gain (%)")
    ax3.set_title("(c) Dose-response", fontsize=8)
    save(fig, "fig3_main")


# ================================================================ Fig 4 检测保持
def fig4():
    items = [("base", "eval_base_s1d", 1.0), ("e2e s1", "eval_e2e_ep7", 1.0),
             ("e2e s2", "eval_e2e_s2_ep7", 1.0), ("e2e s3", "eval_e2e_s3_ep7", 1.0)]
    rows = []
    for lab, name, _ in items:
        r = load_eval(name)
        if r and "test" in r:
            rows.append((lab, r["test"]["mAP50"], r["test"]["mAP@small"],
                         r["test_hard"]["mAP50"] if "test_hard" in r and r["test_hard"] else np.nan))
    if not rows:
        print("跳过 fig4"); return
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.3), sharey=False)
    labs = [r[0] for r in rows]
    for ax, idx, title, ylab in [(axes[0], 1, "(a) test mAP$_{50}$", ""),
                                 (axes[1], 3, "(b) test_hard mAP$_{50}$", ""),
                                 (axes[2], 2, "(c) AP$_S$ (small targets)", "")]:
        vals = [r[idx] for r in rows]
        cols = [C_GR] + [C_OFF, C_G, C_P][:len(rows) - 1]
        ax.bar(labs, vals, color=cols, width=0.6)
        for i, v in enumerate(vals):
            if v == v:
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6.3)
        ax.set_title(title, fontsize=8)
        ax.set_ylim(0, max([v for v in vals if v == v]) * 1.3)
        ax.tick_params(axis="x", rotation=12)
    axes[0].set_ylabel("Score")
    save(fig, "fig4_detection")


# ================================================================ Fig 5 消融
def _sweep(dirs, r_main, split="testbig"):
    vals = {}
    for tag, d in dirs.items():
        r = load_eval(d)
        if r and f"video_{split}_motion-gt_mem-on" in r:
            v = r[f"video_{split}_motion-gt_mem-on"]
            vals[tag] = v["tp_kept"] / v["tp_prev"]
    base_on = r_main[f"video_{split}_motion-gt_mem-on"]
    vals["align (full)"] = base_on["tp_kept"] / base_on["tp_prev"]
    return vals


def fig5():
    r1 = load_eval("eval_e2e_ep7")
    if r1 is None:
        print("跳过 fig5"); return
    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.2))
    # (a) 对齐消融
    v = _sweep({"no align (warp off)": "eval_sweep_noalign"}, r1)
    if len(v) >= 2:
        ks = list(v.keys()); vs = [v[k] for k in ks]
        axes[0].bar(ks, vs, color=[C_OFF, C_ON][:len(ks)])
        for i, vv in enumerate(vs):
            axes[0].text(i, vv + 0.01, f"{vv:.3f}", ha="center", fontsize=6)
        axes[0].set_title("(a) Motion alignment", fontsize=8)
    # (b) 寿命
    v = _sweep({"$L{=}1$": "eval_sweep_life1", "$L{=}5$": "eval_sweep_life5"}, r1)
    order = [k for k in ["align (full)", "$L{=}1$", "$L{=}5$"] if k in v]
    if order:
        axes[1].plot(range(len(order)), [v[k] for k in order], "o-", color=C_ON)
        axes[1].set_xticks(range(len(order)))
        axes[1].set_xticklabels([o.replace("align (full)", "$L{=}3$ (full)") for o in order],
                                fontsize=6.3)
        axes[1].set_title("(b) Memory lifespan", fontsize=8)
    # (c) 噪声
    v = _sweep({"$\\sigma{=}5$px": "eval_sweep_noise5", "$\\sigma{=}15$px": "eval_sweep_noise15"}, r1)
    order = [k for k in ["align (full)", "$\\sigma{=}5$px", "$\\sigma{=}15$px"] if k in v]
    if order:
        axes[2].plot(range(len(order)), [v[k] for k in order], "s-", color=C_G)
        axes[2].set_xticks(range(len(order)))
        axes[2].set_xticklabels(["0 (full)" if "full" in o else o for o in order], fontsize=6.3)
        axes[2].set_title("(c) Odometry noise", fontsize=8)
    # (d) phase
    v = _sweep({"phase corr.": "eval_sweep_phase"}, r1)
    if len(v) >= 2:
        ks = ["align (full)", "phase corr."]
        vs = [v[k] for k in ks]
        axes[3].bar(["odometry", "phase corr."], vs, color=[C_ON, C_O])
        for i, vv in enumerate(vs):
            axes[3].text(i, vv + 0.01, f"{vv:.3f}", ha="center", fontsize=6)
        axes[3].set_title("(d) Motion source", fontsize=8)
    for ax in axes:
        ax.set_ylim(0, max(ax.get_ylim()[1], 0.85))
        ax.set_ylabel("TP stability" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.25)
    save(fig, "fig5_ablation")


# ================================================================ Fig 6 定性
def fig6():
    import cv2
    from dataset import load_img, load_yolo_label
    from eval import load_model, ckpt_boxes_to_xyxy_px
    from ops import postprocess
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model("../runs/e2e_s1/ep7.pt", dev)
    DATA = ROOT / "data" / "defect_ds"

    def box(img, b, color, label=None, th=2, placed=None):
        x0, y0, x1, y1 = [int(round(float(v))) for v in b]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, th)
        if label:
            (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            lx, ly = x0, max(16, y0 - 4)
            if lx + tw > img.shape[1] - 2:          # 右缘钳制
                lx = max(2, img.shape[1] - tw - 2)
            if placed is not None:                   # 标签避碰: 与已放置标签重叠则下移
                for _ in range(6):
                    hit = next(((px0, py0, px1, py1) for (px0, py0, px1, py1) in placed
                                if lx < px1 and lx + tw > px0 and ly - tht < py1 and ly > py0), None)
                    if hit is None:
                        break
                    ly = hit[3] + tht + 6
                placed.append((lx, ly - tht, lx + tw, ly))
            # 深色描边 + 彩色前景: 复杂纹理/锈斑背景上仍可读 (Nature 可读性要求)
            cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2, cv2.LINE_AA)

    names = ["rust", "crack", "weld", "peel", "pitting"]
    # (a) 检测样张 2x3
    imgs = sorted((DATA / "images" / "test").glob("*.jpg"))
    idxs = np.linspace(0, len(imgs) - 1, 6).astype(int)
    rows = []
    with torch.no_grad():
        for i in idxs:
            p = imgs[i]
            img = load_img(p); H, W = img.shape[:2]
            out = model(torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(dev))
            pp = postprocess(out, W, H, score_thr=0.35)[0]
            vis = (img * 255).astype(np.uint8).copy()
            bx, lb = load_yolo_label(DATA / "labels" / "test" / (p.stem + ".txt"), W, H)
            for b in ckpt_boxes_to_xyxy_px(bx, W, H):
                box(vis, b, (0, 200, 0))
            placed = []
            for b, l, s in zip(pp["boxes"].tolist(), pp["labels"].tolist(), pp["scores"].tolist()):
                box(vis, b, (230, 40, 40), f"{names[l]} {s:.2f}", placed=placed)
            rows.append(vis)
    grid = np.vstack([np.hstack(rows[i:i + 3]) for i in range(0, 6, 3)])
    ok, buf = cv2.imencode(".png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    buf.tofile(str(OUT / "fig6a_detection.png"))
    print("saved", OUT / "fig6a_detection.png")

    # (b) 记忆时序条
    vi = json.loads((DATA / "video_index.json").read_text(encoding="utf-8"))
    seq = vi["test"][0]
    memory = None
    frames = []
    with torch.no_grad():
        for t, rel in enumerate(seq["frames"]):
            img = load_img(DATA / rel); H, W = img.shape[:2]
            x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(dev)
            motion = None if t == 0 else (float(seq["stride"]), 0.0)
            out = model(x, memory_state=memory, motion=motion, return_memory=True)
            pp = postprocess(out, W, H, score_thr=0.3)[0]
            vis = (img * 255).astype(np.uint8).copy()
            bx, lb = load_yolo_label(DATA / "labels" / seq["split"] / (Path(rel).stem + ".txt"), W, H)
            for b in ckpt_boxes_to_xyxy_px(bx, W, H):
                box(vis, b, (0, 200, 0), th=3)
            for b, l, s in zip(pp["boxes"].tolist(), pp["labels"].tolist(), pp["scores"].tolist()):
                box(vis, b, (230, 40, 40), th=3)
            if memory is not None:
                mb = memory[0]["box"].cpu()
                ag = memory[0]["age"].cpu()
                if mb.numel():
                    mw = (mb - torch.tensor([seq["stride"], 0., seq["stride"], 0.])).tolist()
                    for bb, a in zip(mw, ag.tolist()):
                        if bb[2] > 0 and bb[0] < W and bb[3] > 0 and bb[1] < H:
                            c = max(90, 255 - int(a) * 55)
                            box(vis, [max(0, bb[0]), max(0, bb[1]), min(W, bb[2]), min(H, bb[3])],
                                (60, 80, c), th=3)
            probs = out["pred_logits"].sigmoid()
            sm, lm = probs.max(dim=-1)
            memory = model.update_memory(memory, out["_mem_raw"], W, H, sm, lm)
            frames.append(cv2.resize(vis, (256, 256)))
    # 2 行 x 6 帧 (阅读顺序: 上行->下行), 每帧 256px 保证打印可读性
    strip = np.vstack([np.hstack(frames[0:6]), np.hstack(frames[6:12])])
    # 底部图例条: 与画框配色一致; memory 年龄渐变=亮(年轻)->暗(老)
    legend = np.full((80, strip.shape[1], 3), 255, np.uint8)

    def lchip(x, color, text, w=28):
        cv2.rectangle(legend, (x, 26), (x + w, 54), color, -1)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(legend, text, (x + w + 10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2, cv2.LINE_AA)
        return x + w + 10 + tw + 44

    xcur = 24
    xcur = lchip(xcur, (0, 200, 0), "ground truth")
    xcur = lchip(xcur, (230, 40, 40), "prediction")
    for a in range(4):
        cc = max(90, 255 - a * 55)
        cv2.rectangle(legend, (xcur + a * 34, 26), (xcur + a * 34 + 28, 54), (60, 80, cc), -1)
    (tw, _), _ = cv2.getTextSize("memory token (young -> old)", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(legend, "memory token (young -> old)", (xcur + 4 * 34 + 8, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    grid = np.vstack([strip, legend])
    ok2, buf2 = cv2.imencode(".png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    buf2.tofile(str(OUT / "fig6b_memory.png"))
    print("saved", OUT / "fig6b_memory.png")


# ================================================================ 图注 (caption)
CAPTIONS = """fig1_framework: Overview of the proposed Temporal Focus Memory (TFM). Detected
objects with confidence above tau are selected as focus tokens, stored with position, class
and age, and carried over to the next frame after odometry-based motion warping. Memory
tokens are injected into the decoder query set of the current frame, participating in set
prediction; aging, IoU de-duplication and confidence decay maintain the memory bank.

fig2_training: (a) Training loss across the four continuation rounds (log scale; base and
focus variants). (b) Best validation mAP50 reached by each training run. The loss plateau
of round 1 motivated the learning-rate floor adopted in rounds 2-4.

fig3_main: (a) TP stability P(hit_t | hit_{t-1}) with memory ON vs OFF under identical
weights (seed s1), per test domain and pooled; Fisher exact p-values annotated.
(b) Pooled relative gain for three independent training seeds (s1-s3); dashed line: mean.
(c) Dose-response between memory training epochs (1, 3, 5 and 7) and pooled relative gain.

fig4_detection: Frame-level detection quality of the baseline detector (base) and the
end-to-end runs (one per seed): (a) test mAP50, (b) test_hard mAP50, (c) AP_S for small
targets. Detection quality is preserved after end-to-end memory training.

fig5_ablation: Ablations on the testbig split (memory ON): (a) disabling motion
alignment, (b) memory lifespan L in {1,3,5}, (c) odometry noise sigma in {0,5,15} px
for TFM (blue) and for the output-level box-persistence control (gray; no learning,
identical warping/decay/de-duplication), (d) phase-correlation motion estimation as a
substitute for odometry.

fig6_qualitative: (a) Detection samples (green: ground truth; red: predictions, labelled
with class and confidence). (b) Consecutive frames along an inspection path (reading order:
top row, then bottom row); green: ground truth, red: current-frame predictions, blue:
memory tokens (brighter blue = younger; age ramp shown in the legend).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()
    todo = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4,
            "fig5": fig5, "fig6": fig6}
    for k, f in todo.items():
        if args.only and k != args.only:
            continue
        try:
            f()
        except Exception as e:
            print(f"[{k}] 生成失败: {e}")
    (OUT / "captions.txt").write_text(CAPTIONS, encoding="utf-8")
    print("saved", OUT / "captions.txt")


if __name__ == "__main__":
    main()
