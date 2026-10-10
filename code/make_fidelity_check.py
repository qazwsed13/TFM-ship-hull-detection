# -*- coding: utf-8 -*-
"""仿真保真度对比 (审稿意见 M1): 真实船体图像 vs 模拟器图像
- 真实源: 袁小芳2026 [4] 视频帧网格 (yuan_p8_71) 中裁剪 3 格
- 仿真源: data/defect_ds/images/test 中选 3 帧
- 统计: 亮度标准差 / 边缘密度(Sobel) / GLCM 对比度 / GLCM 同质性 (32级, d=1, 4方向均值)
- 输出: runs/figures_final/fig7_fidelity.{png,pdf,svg} + runs/fidelity_stats.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
REAL_SRC = ROOT / "data" / "fidelity" / "real" / "yuan_p8_71_1654x1251.png"
SIM_DIR = ROOT / "data" / "defect_ds" / "images" / "test"
OUT_DIR = ROOT / "runs" / "figures_final"
SEL = ROOT / "data" / "fidelity" / "selected"
SEL.mkdir(parents=True, exist_ok=True)


def imread_u(p):
    """中文路径安全的 imread."""
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(p, img):
    """中文路径安全的 imwrite."""
    ok, buf = cv2.imencode(Path(p).suffix, img)
    buf.tofile(str(p))


def crop_real():
    """从视频帧网格中裁剪 3 个锈蚀/钢面主导的子区域 (避开机器人/标注/彩色涂装块),
    保证对比针对表面纹理而非场景构图."""
    img = imread_u(REAL_SRC)
    H, W = img.shape[:2]
    cw, ch = W // 4, H // 4
    # (行, 列, 子区域相对坐标 fx0, fy0, fx1, fy1): 重锈 / 混合 / 较清洁钢面
    cells = [(1, 0, 0.12, 0.42, 0.95, 0.95),
             (2, 0, 0.28, 0.04, 0.96, 0.62),
             (0, 1, 0.06, 0.04, 0.94, 0.36)]
    outs = []
    for k, (r, c, fx0, fy0, fx1, fy1) in enumerate(cells):
        x0, y0 = c * cw + int(fx0 * cw), r * ch + int(fy0 * ch)
        x1, y1 = c * cw + int(fx1 * cw), r * ch + int(fy1 * ch)
        crop = img[y0:y1, x0:x1]
        p = SEL / f"real_{k}.png"
        imwrite_u(p, crop)
        outs.append(p)
    return outs


def pick_sim():
    imgs = sorted(SIM_DIR.glob("*.jpg"))
    idxs = [17, 214, 391]                      # 待目检后如需可调
    outs = []
    for k, i in enumerate(idxs):
        img = imread_u(imgs[i])
        p = SEL / f"sim_{k}.png"
        imwrite_u(p, img)
        outs.append(p)
    return outs


def glcm_feats(gray32, d=1):
    h, w = gray32.shape
    cts = []
    for di, dj in [(0, d), (d, 0), (d, d), (d, -d)]:
        a = gray32[max(0, -di):h - max(0, di), max(0, -dj):w - max(0, dj)]
        b = gray32[max(0, di):h - max(0, -di), max(0, dj):w - max(0, -dj)]
        m = np.zeros((32, 32), np.float64)
        np.add.at(m, (a.ravel(), b.ravel()), 1)
        m = m + m.T                                # 对称化
        m /= max(m.sum(), 1)
        cts.append(m)
    contrast, homog = [], []
    ii, jj = np.mgrid[0:32, 0:32]
    for m in cts:
        contrast.append(float((m * (ii - jj) ** 2).sum()))
        homog.append(float((m / (1 + np.abs(ii - jj))).sum()))
    return float(np.mean(contrast)), float(np.mean(homog))


def stats(path):
    g = cv2.cvtColor(imread_u(path), cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA)
    lum_std = float(g.std())
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    edge_density = float((mag > 40).mean() * 100)
    g32 = (g // 8).astype(np.int32)
    contrast, homog = glcm_feats(g32)
    return {"lum_std": lum_std, "edge_density_pct": edge_density,
            "glcm_contrast": contrast, "glcm_homogeneity": homog}


def main():
    reals = crop_real()
    sims = pick_sim()
    rs = [stats(p) for p in reals]
    ss = [stats(p) for p in sims]
    agg = {}
    for name, arr in [("real", rs), ("sim", ss)]:
        agg[name] = {k: float(np.mean([r[k] for r in arr])) for k in rs[0]}
    # 相对偏差
    agg["rel_diff_pct"] = {k: (agg["sim"][k] / agg["real"][k] - 1) * 100
                           for k in agg["real"]}
    (ROOT / "runs" / "fidelity_stats.json").write_text(
        json.dumps({"per_image": {"real": rs, "sim": ss}, "mean": agg},
                   indent=2), encoding="utf-8")
    print(json.dumps(agg, indent=2))

    # ---- 绘图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from make_sci_figures_v2 import pubstyle, finalize_figure, C_ON

    pubstyle(12, 1.6)
    real_imgs = [cv2.cvtColor(imread_u(p), cv2.COLOR_BGR2RGB) for p in reals]
    sim_imgs = [cv2.cvtColor(imread_u(p), cv2.COLOR_BGR2RGB) for p in sims]
    fig = plt.figure(figsize=(11.5, 7.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.92], hspace=0.28, wspace=0.10)
    for k in range(3):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(real_imgs[k]); ax.set_xticks([]); ax.set_yticks([])
        if k == 0:
            ax.set_ylabel("Real hull surface [4]", fontsize=12)
    for k in range(3):
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(sim_imgs[k]); ax.set_xticks([]); ax.set_yticks([])
        if k == 0:
            ax.set_ylabel("Simulated (ours)", fontsize=12)
    ax1 = fig.add_subplot(gs[2, 0])
    ax2 = fig.add_subplot(gs[2, 1])
    ax3 = fig.add_subplot(gs[2, 2]); ax3.axis("off")
    for ax, key, ttl in [(ax1, "glcm_contrast", "GLCM contrast"),
                         (ax2, "edge_density_pct", "Edge density (%)")]:
        rv = [r[key] for r in rs]
        sv = [s[key] for s in ss]
        bars = ax.bar(["real\n(range)", "sim\n(mean)"],
                      [np.mean(rv), np.mean(sv)],
                      color=["#9a9a9a", C_ON], width=0.55, edgecolor="black", linewidth=0.8)
        ax.errorbar([0], [np.mean(rv)], yerr=[[np.mean(rv) - min(rv)], [max(rv) - np.mean(rv)]],
                    fmt="none", ecolor="black", capsize=4, linewidth=1.2)
        ax.errorbar([1], [np.mean(sv)], yerr=[[0], [np.std(sv, ddof=1)]],
                    fmt="none", ecolor="black", capsize=4, linewidth=1.2)
        ax.set_title(ttl, fontsize=12)
        for b in bars:
            ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10.5)
        ax.grid(axis="y", alpha=0.25)
    con_r = [r["glcm_contrast"] for r in rs]
    edge_r = [r["edge_density_pct"] for r in rs]
    txt = (f"3 images/group; real spans heavy rust -> cleaner steel\n\n"
           f"GLCM contrast   sim {agg['sim']['glcm_contrast']:.2f} | real "
           f"{min(con_r):.2f}--{max(con_r):.2f}\n"
           f"Edge density    sim {agg['sim']['edge_density_pct']:.1f}% | real "
           f"{min(edge_r):.1f}--{max(edge_r):.1f}%\n"
           f"Lum. std        sim {agg['sim']['lum_std']:.1f} | real "
           f"{min(r['lum_std'] for r in rs):.1f}--{max(r['lum_std'] for r in rs):.1f}\n"
           f"GLCM homog.     sim {agg['sim']['glcm_homogeneity']:.3f} | real "
           f"{min(r['glcm_homogeneity'] for r in rs):.3f}--{max(r['glcm_homogeneity'] for r in rs):.3f}")
    ax3.text(0.0, 0.82, txt, fontsize=10.5, va="top", family="monospace")
    finalize_figure(fig, OUT_DIR / "fig7_fidelity", formats=["pdf", "svg", "png"], dpi=600)
    print("saved fig7_fidelity")


if __name__ == "__main__":
    main()
