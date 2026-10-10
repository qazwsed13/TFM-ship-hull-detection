# -*- coding: utf-8 -*-
"""
SCI 图件生成器 v2 —— 使用 scientific-figure-pro 技能 (figures4papers house style)
=================================================================================
工作流遵循技能规范:
  1) 识别图族  2) 按路径加载 scripts/scientific_figure_pro.py
  3) apply_publication_style(...)  4) make_* 构建  5) finalize_figure 导出

语义色板 (技能 Style Policy): 蓝=主张方法 (memory ON) / 红=基线 (memory OFF) /
绿=改进 (种子复现) / 中性=支撑类别。
输出: runs/figures_final/  (PDF + SVG + PNG@300-600dpi)
"""
import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---- 按技能规范加载 helper (需注册 sys.modules 以支持 dataclass) ----
import sys
SKILL_PATH = Path.home() / ".qoder" / "skills" / "scientific-figure-pro" / "scripts" / "scientific_figure_pro.py"
_spec = importlib.util.spec_from_file_location("scientific_figure_pro", SKILL_PATH)
mod = importlib.util.module_from_spec(_spec)
sys.modules["scientific_figure_pro"] = mod
_spec.loader.exec_module(mod)

FigureStyle = mod.FigureStyle
apply_publication_style = mod.apply_publication_style
create_subplots = mod.create_subplots
make_grouped_bar = mod.make_grouped_bar
make_trend = mod.make_trend
make_scatter = mod.make_scatter
annotate_bars = mod.annotate_bars
finalize_figure = mod.finalize_figure
PAL = mod.PALETTE

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = RUNS / "figures_final"
OUT.mkdir(parents=True, exist_ok=True)

# 语义映射
C_ON, C_OFF = PAL["blue_main"], PAL["red_strong"]        # 主张 / 基线
C_SEEDS = [PAL["blue_main"], PAL["teal"], PAL["violet"]]  # 种子复现
C_IMPROVE = PAL["green_3"]
C_NEUTRAL = PAL["neutral"]

# Nature 合规: Arial/Helvetica 优先 + 刻度朝内 (Nature 期刊规范)
NATURE_FONTS = ("Arial", "Helvetica", "DejaVu Sans", "sans-serif")


def pubstyle(font_size=13, axes_linewidth=1.6):
    """应用技能样式后叠加 Nature 规范微调 (字体族/刻度方向)."""
    apply_publication_style(FigureStyle(font_size=font_size,
                                        axes_linewidth=axes_linewidth,
                                        font_family=NATURE_FONTS))
    plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in"})


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
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def pooled(r, mode):
    k = n = 0
    for s in ["test", "test_hard", "testbig"]:
        v = r.get(f"video_{s}_motion-gt_mem-{mode}")
        if v:
            k += v["tp_kept"]; n += v["tp_prev"]
    return k, n


# ================================================================ Fig 1 框架示意
def fig1():
    pubstyle(13, 1.6)
    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.3); ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#333333", fs=10.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.4)

    def arrow(x1, y1, x2, y2, color="#333333", ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", color=color,
                                     lw=1.2, linestyle=ls, mutation_scale=11,
                                     connectionstyle=f"arc3,rad={rad}"))

    box(0.1, 2.85, 1.35, 1.0, "Frame $t{-}1$\n(detections)", "#F5F5F5")
    box(1.85, 2.85, 1.7, 1.0, "Focus selection\nscore $>\\tau$,  top-$K$", "#E8F0FA")
    box(3.85, 2.85, 1.95, 1.0, "Memory bank\nfeat · box · cls · age", "#DCE7F5", bold=True, fs=10)
    box(6.2, 2.85, 1.6, 1.0, "Aging / de-dup\nthen decay", "#F5F5F5")
    arrow(1.45, 3.35, 1.85, 3.35); arrow(3.55, 3.35, 3.85, 3.35); arrow(5.8, 3.35, 6.2, 3.35)

    box(0.1, 0.35, 1.35, 1.0, "Frame $t$", "#F5F5F5")
    box(1.85, 0.35, 1.7, 1.0, "Query set:\ncontent $\\oplus$ mem\n$\\oplus$ denoising", "#E8F0FA", fs=9.5)
    box(3.85, 0.35, 1.95, 1.0, "Detector decoder\n(anchor + deform.)", "#DCE7F5", fs=10)
    box(6.2, 0.35, 1.6, 1.0, "Set prediction\n+ Hungarian loss", "#F5F5F5")
    arrow(1.45, 0.85, 1.85, 0.85); arrow(3.55, 0.85, 3.85, 0.85); arrow(5.8, 0.85, 6.2, 0.85)

    arrow(4.6, 2.85, 2.6, 1.35, color=C_ON, rad=-0.25)
    ax.text(3.3, 1.95, "warp by odometry\n$\\mathbf{b}\\,{-}\\,(\\Delta x,\\Delta y)$",
            color=C_ON, fontsize=10, ha="center")
    arrow(7.75, 1.35, 7.75, 2.85, color="#666666", ls="--")
    ax.text(8.05, 2.1, "next\nframe", color="#666666", fontsize=9.5, ha="left", linespacing=1.3)
    ax.text(5.0, 4.12, "Temporal Focus Memory (TFM)", ha="center", fontsize=13.5,
            fontweight="bold")
    finalize_figure(fig, OUT / "fig1_framework", formats=["pdf", "svg", "png"], dpi=600)
    print("saved fig1")


# ================================================================ Fig 2 训练动力学
def _epoch_rows(csv_path):
    xs, ys = [], []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("iter") == "epoch_avg":
                xs.append(int(row["epoch"])); ys.append(float(row["loss"]))
    return xs, ys


def fig2():
    pubstyle(13, 1.8)
    fig, axes = create_subplots(1, 2, figsize=(13.5, 4.4))

    # (a) 损失曲线 —— make_trend (逐轮, 无阴影以保清晰)
    series, labels, colors = [], [], []
    runs = [("base_s1", "base R1", PAL["neutral"]), ("base_s1b", "base R2", PAL["blue_secondary"]),
            ("base_s1c", "base R3", PAL["teal"]), ("base_s1d", "base R4", C_ON),
            ("focus_s1", "focus R1", "#F6CFCB"), ("focus_s1b", "focus R2", "#E9A6A1"),
            ("focus_s1c", "focus R3", "#D88C88"), ("focus_s1d", "focus R4", "#C06A66"),
            ("focus_g_s1", "focus_g", PAL["violet"])]
    xs_ref = None
    for name, lab, c in runs:
        f = RUNS / name / "train_log.csv"
        if f.exists():
            xs, ys = _epoch_rows(f)
            if xs:
                axes[0].plot(xs, ys, label=lab, color=c, linewidth=2.2, alpha=0.92)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training loss")
    axes[0].legend(ncol=2, fontsize=10)
    axes[0].set_title("(a) Training dynamics", fontsize=14)

    # (b) 验证轨迹 (每轮 running-best)
    best = {}
    tag_color = {"base_s1": C_ON, "focus_s1": PAL["red_strong"],
                 "focus_g_s1": PAL["violet"], "focus_t": PAL["teal"], "base_t": PAL["green_3"]}
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
                    if m >= best.get(cur, (-1,))[0]:
                        best[cur] = (m, ep)
                except Exception:
                    pass
    for k, (m, ep) in best.items():
        axes[1].scatter([ep], [m], s=48, color=tag_color.get(k, "#777777"),
                        edgecolors="white", linewidth=0.6, zorder=3)
        axes[1].annotate(f"{k} {m:.3f}", (ep, m), fontsize=10, xytext=(4, 3),
                         textcoords="offset points")
    # base_t (stage-2 探针) 验证轨迹: 早峰后单调衰减 —— 回应"为何不用 base_t"
    dlog = RUNS / "train_chain_D.log"
    if dlog.exists():
        _txt = dlog.read_text(encoding="utf-8", errors="ignore")
        _i4, _i5 = _txt.find("[D4]"), _txt.find("[D5]")
        _sec = _txt[_i4:_i5] if 0 <= _i4 < _i5 else ""
        _bx, _by = [], []
        for _m in re.finditer(r"eval ep(\d+)\][^\n]*mAP50=([0-9.]+)", _sec):
            _bx.append(int(_m.group(1))); _by.append(float(_m.group(2)))
        if _bx:
            axes[1].plot(_bx, _by, "--o", color=tag_color["base_t"], linewidth=1.8,
                         markersize=5, zorder=2, alpha=0.9)
            axes[1].annotate("base_t: peak$\\rightarrow$decay", (_bx[-1], _by[-1]),
                             fontsize=9, xytext=(6, -3), textcoords="offset points",
                             color="#2f6b3f")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val mAP$_{50}$")
    axes[1].set_title("(b) Validation trajectory", fontsize=14)
    finalize_figure(fig, OUT / "fig2_training", formats=["pdf", "svg", "png"], dpi=600)
    print("saved fig2")


# ================================================================ Fig 3 主结果
def fig3():
    r1 = load_eval("eval_e2e_ep7")
    if r1 is None:
        print("跳过 fig3"); return
    pubstyle(16, 2.2)  # 密集柱状面板
    fig, axes = create_subplots(1, 3, figsize=(16.5, 4.8))

    # (a) 分域主结果 (grouped bars + p 值)
    splits = ["test", "test_hard", "testbig"]
    cats = ["test", "test_hard", "testbig\n(200 seqs)", "pooled"]
    on_v, off_v, ann = [], [], []
    for s in splits:
        o = r1[f"video_{s}_motion-gt_mem-on"]; f = r1[f"video_{s}_motion-gt_mem-off"]
        on_v.append(o["tp_kept"] / o["tp_prev"]); off_v.append(f["tp_kept"] / f["tp_prev"])
        ann.append(ptxt(fisher(o["tp_kept"], o["tp_prev"], f["tp_kept"], f["tp_prev"])))
    k1, n1 = pooled(r1, "on"); k2, n2 = pooled(r1, "off")
    on_v.append(k1 / n1); off_v.append(k2 / n2)
    ann.append(ptxt(fisher(k1, n1, k2, n2)))
    make_grouped_bar(axes[0], cats, [on_v, off_v], ["memory ON (proposed)", "memory OFF (baseline)"],
                     ylabel="TP stability  $P(hit_t|hit_{t-1})$", colors=[C_ON, C_OFF])
    for i, t in enumerate(ann):
        axes[0].text(i, max(on_v[i], off_v[i]) + 0.035, t, ha="center", fontsize=10)
    axes[0].set_ylim(0, 0.95)
    axes[0].tick_params(axis="x", labelsize=11)
    plt.setp(axes[0].get_xticklabels(), rotation=14, ha="right", rotation_mode="anchor")
    axes[0].set_title("(a) Main result (seed $s_1$)", fontsize=16)

    # (b) 种子一致性
    lifts = []
    for i, (name, tag) in enumerate([("eval_e2e_ep7", "$s_1$"), ("eval_e2e_s2_ep7", "$s_2$"),
                                     ("eval_e2e_s3_ep7", "$s_3$")]):
        r = load_eval(name)
        if r is None:
            continue
        k1, n1 = pooled(r, "on"); k2, n2 = pooled(r, "off")
        lifts.append(((k1 / n1) / (k2 / n2) - 1) * 100)
        axes[1].scatter([lifts[-1]], [tag], s=120, color=C_SEEDS[i],
                        edgecolors="white", linewidth=0.8, zorder=3)
    if lifts:
        mean = float(np.mean(lifts))
        axes[1].axvline(mean, color="#555555", ls="--", linewidth=1.6)
        axes[1].text(mean, 2.68, f"mean {mean:.1f}%", fontsize=11, ha="center")
    axes[1].set_xlabel("Relative TP-stability gain (%)")
    axes[1].set_title("(b) Seed consistency", fontsize=16)
    axes[1].grid(axis="x", alpha=0.3)
    if lifts:
        lo, hi = min(lifts), max(lifts)
        pad = max(3.0, (hi - lo) * 0.25)
        axes[1].set_xlim(lo - pad, hi + pad)
        axes[1].set_ylim(-0.6, 2.8)

    # (c) 剂量-效应 (四点: ep1/3/5/7, 各自独立取自对应检查点的 pooled 统计)
    doses, vals = [], []
    for ep, tag in [(1, "eval_e2e_ep1"), (3, "eval_e2e_ep3"), (5, "eval_e2e_ep5"), (7, None)]:
        r_ep = r1 if tag is None else load_eval(tag)
        if r_ep is None:
            continue
        k1, n1 = pooled(r_ep, "on"); k2, n2 = pooled(r_ep, "off")
        doses.append(ep); vals.append(((k1 / n1) / (k2 / n2) - 1) * 100)
    axes[2].plot(doses, vals, "o-", color=C_ON, linewidth=2.4, markersize=9)
    for d, v in zip(doses, vals):
        axes[2].annotate(f"{v:+.0f}%", (d, v), xytext=(8, -9), textcoords="offset points",
                         fontsize=11)
    axes[2].set_xticks([1, 3, 5, 7]); axes[2].set_xlim(0.5, 7.6)
    axes[2].set_ylim(0, max(vals) * 1.25)
    axes[2].set_xlabel("Memory training epochs")
    axes[2].set_ylabel("TP-stability gain (%)")
    axes[2].set_title("(c) Dose-response", fontsize=16)
    finalize_figure(fig, OUT / "fig3_main", formats=["pdf", "svg", "png"], dpi=600)
    print("saved fig3")


# ================================================================ Fig 4 检测保持
def fig4():
    pubstyle(15, 2.0)
    items = [("base\n(baseline)", "eval_base_s1d", C_OFF), ("e2e\n$s_1$", "eval_e2e_ep7", C_SEEDS[0]),
             ("e2e\n$s_2$", "eval_e2e_s2_ep7", C_SEEDS[1]), ("e2e\n$s_3$", "eval_e2e_s3_ep7", C_SEEDS[2])]
    rows = []
    for lab, name, c in items:
        r = load_eval(name)
        if r and "test" in r and r["test"]:
            rows.append((lab, r["test"]["mAP50"],
                         (r.get("test_hard") or {}).get("mAP50", np.nan),
                         r["test"]["mAP@small"], c))
    if not rows:
        print("跳过 fig4"); return
    fig, axes = create_subplots(1, 3, figsize=(13.5, 4.0))
    for ax, idx, title in [(axes[0], 1, "(a) test mAP$_{50}$"),
                           (axes[1], 2, "(b) test_hard mAP$_{50}$"),
                           (axes[2], 3, "(c) AP$_S$ (small targets)")]:
        labs = [r[0] for r in rows]
        vals = [r[idx] for r in rows]
        cols = [r[4] for r in rows]
        bars = ax.bar(labs, vals, color=cols, width=0.62, edgecolor="black", linewidth=1.0)
        annotate_bars(ax, bars, fmt="{:.3f}", fontsize=10.5)
        ax.set_title(title, fontsize=15)
        ymax = np.nanmax(vals)
        ax.set_ylim(0, ymax * 1.32)
    axes[0].set_ylabel("Score")
    finalize_figure(fig, OUT / "fig4_detection", formats=["pdf", "svg", "png"], dpi=600)
    print("saved fig4")


# ================================================================ Fig 5 消融
def _sweep_r(r_main, dir_name, split="testbig"):
    r = load_eval(dir_name)
    if r:
        for tag in [f"video_{split}_motion-gt_mem-on",
                    f"video_{split}_motion-phase_mem-on"]:
            if tag in r:
                v = r[tag]
                return v["tp_kept"] / v["tp_prev"], v
    return None, None


def fig5():
    r1 = load_eval("eval_e2e_ep7")
    if r1 is None:
        print("跳过 fig5"); return
    pubstyle(14, 1.9)
    fig, axes = create_subplots(1, 5, figsize=(19.2, 3.7))
    base_on = r1["video_testbig_motion-gt_mem-on"]
    base_v = base_on["tp_kept"] / base_on["tp_prev"]

    # (a) 对齐消融
    v, _ = _sweep_r(r1, "eval_sweep_noalign")
    if v is not None:
        bars = axes[0].bar(["no align\n(warp off)", "align\n(full)"], [v, base_v],
                           color=[PAL["red_strong"], C_ON], width=0.55,
                           edgecolor="black", linewidth=1.0)
        annotate_bars(axes[0], bars, fmt="{:.3f}", fontsize=10)
    axes[0].set_title("(a) Motion alignment", fontsize=14)

    # (b) 寿命
    v1, _ = _sweep_r(r1, "eval_sweep_life1")
    v5, _ = _sweep_r(r1, "eval_sweep_life5")
    lx, ly, ll = [3], [base_v], ["$L$=3 (full)"]
    if v1 is not None:
        lx.insert(0, 1); ly.insert(0, v1); ll.insert(0, "$L$=1")
    if v5 is not None:
        lx.append(5); ly.append(v5); ll.append("$L$=5")
    axes[1].plot(lx, ly, "o-", color=C_ON, linewidth=2.2, markersize=8)
    offs = {1: (14, 7), 3: (13, 8), 5: (0, -17)}   # 交叉错位: 左缘标签右移避免压脊线
    for d, v in zip(lx, ly):
        axes[1].annotate(f"{v:.3f}", (d, v), xytext=offs.get(d, (0, 10)),
                         textcoords="offset points", fontsize=10, ha="center")
    axes[1].set_xticks(lx); axes[1].set_xticklabels(ll, fontsize=10)
    axes[1].set_title("(b) Memory lifespan", fontsize=14)

    # (c) 噪声: TFM vs 输出级持久化对照 (同一噪声实现)
    n5, _ = _sweep_r(r1, "eval_sweep_noise5")
    n15, _ = _sweep_r(r1, "eval_sweep_noise15")
    xs, ys, xl = [0], [base_v], ["0"]
    if n5 is not None:
        xs.append(5); ys.append(n5); xl.append("5")
    if n15 is not None:
        xs.append(15); ys.append(n15); xl.append("15")
    axes[2].plot(xs, ys, "s-", color=C_ON, linewidth=2.2, markersize=8,
                 label="TFM (ours)")
    for d, v in zip(xs, ys):
        off = {0: (13, 7), 5: (0, 10), 15: (-10, 9)}.get(d, (0, 10))
        axes[2].annotate(f"{v:.3f}", (d, v), xytext=off, textcoords="offset points",
                         fontsize=10, ha="center")
    # 持久化对照 (runs/eval_persistence/results.json)
    pf = RUNS / "eval_persistence" / "results.json"
    if pf.exists():
        _p = json.loads(pf.read_text(encoding="utf-8"))
        px = [0, 5, 15]
        pv = [_p["persist_inf"]["TP_stability"], _p["persist_inf_sig5"]["TP_stability"],
              _p["persist_inf_sig15"]["TP_stability"]]
        axes[2].plot(px, pv, "o--", color="#5a5a5a", linewidth=2.0, markersize=7,
                     label="box persistence (no learning)")
        for d, v in zip(px, pv):
            axes[2].annotate(f"{v:.3f}", (d, v), xytext=(2, -17),
                             textcoords="offset points", fontsize=10, ha="center")
        axes[2].legend(loc="upper right", fontsize=9, frameon=False)
    axes[2].set_xticks(xs); axes[2].set_xticklabels(xl, fontsize=10)
    axes[2].set_xlabel("Odometry noise $\\sigma$ (px)", fontsize=12)
    axes[2].set_title("(c) Noise robustness", fontsize=14)

    # (d) phase
    ph, _ = _sweep_r(r1, "eval_sweep_phase")
    if ph is not None:
        bars = axes[3].bar(["odometry", "phase\ncorr."], [base_v, ph],
                           color=[C_ON, PAL["teal"]], width=0.55,
                           edgecolor="black", linewidth=1.0)
        annotate_bars(axes[3], bars, fmt="{:.3f}", fontsize=10)
    axes[3].set_title("(d) Motion source", fontsize=14)

    # (e) 密度对照: 阈值扫描的 精度–TP稳定度 前沿 (M2)
    dcj = RUNS / "density_control" / "summary.json"
    if dcj.exists():
        dc = json.loads(dcj.read_text(encoding="utf-8"))
        ax = axes[4]
        for mode, col, ls, lab in [("off", "#5a5a5a", "--o", "memory OFF"),
                                   ("on", C_ON, "-o", "memory ON")]:
            pts = [(p["precision@0.5"], p["TP_stability"]) for p in dc["frontier"][mode]
                   if p["thr"] <= 0.45]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=col,
                    linewidth=2.0, markersize=4.5, label=lab)
        rm = dc["recall_matched_off"]
        ax.scatter([rm["precision@0.5"]], [rm["TP_stability"]], s=110, marker="*",
                   color=PAL["red_strong"], zorder=5, edgecolors="white",
                   linewidths=0.6, label="OFF at matched recall")
        ax.annotate("OFF,\nmatched recall", (rm["precision@0.5"], rm["TP_stability"]),
                    fontsize=9, xytext=(-58, 2), textcoords="offset points")
        ax.set_xlabel("Precision@0.5", fontsize=12)
        ax.set_title("(e) Density control", fontsize=14)
        ax.legend(loc="lower right", fontsize=8.5, frameon=False)

    for ax in axes:
        y0, y1 = ax.get_ylim()
        ax.set_ylim(0, max(y1, 0.85))
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("TP stability (testbig)")
    finalize_figure(fig, OUT / "fig5_ablation", formats=["pdf", "svg", "png"], dpi=600)
    print("saved fig5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()
    todo = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5}
    for k, f in todo.items():
        if args.only and k != args.only:
            continue
        try:
            f()
        except Exception as e:
            import traceback
            print(f"[{k}] 失败: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
