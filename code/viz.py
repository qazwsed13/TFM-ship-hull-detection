# -*- coding: utf-8 -*-
"""
可视化工具:
  python viz.py curves                 # 训练损失曲线 (各变体)
  python viz.py detect --ckpt X.pt     # 检测结果网格 (GT绿 / 预测红)
  python viz.py memory --ckpt X.pt     # 时序记忆可视化 (检测红 / 记忆token蓝)
  python viz.py bars                   # 变体指标对比柱状图 (读 eval results.json)
  python viz.py video --ckpt X.pt      # 生成演示视频 (记忆开 vs 关 上下对照)
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import load_img, load_yolo_label
from eval import ckpt_boxes_to_xyxy_px, load_model
from ops import postprocess

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "defect_ds"
OUT = ROOT / "runs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = ["rust", "crack", "weld_defect", "peel", "pitting"]
# 中文字体兜底: 图中用英文类名, 避免字体缺失问题

def imwrite_u(path, img_bgr, quality=92):
    """兼容中文路径的图像写入 (cv2.imwrite 在非 ASCII 绝对路径上会静默失败)."""
    ext = str(path).rsplit(".", 1)[-1].lower()
    ext = "jpg" if ext in ("jpg", "jpeg") else ext
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == "jpg" else []
    ok, buf = cv2.imencode(f".{ext}", img_bgr, params)
    if not ok:
        raise RuntimeError(f"imencode 失败: {path}")
    buf.tofile(str(path))

def _draw_box(img, box, color, label=None, thickness=2):
    if torch.is_tensor(box):
        box = box.tolist()
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
    if label:
        cv2.putText(img, label, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)

# ---------------------------------------------------------------- 曲线

def curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    runs = [("base_s1", "base round1"), ("base_s1b", "base round2"),
            ("base_s1c", "base round3"), ("base_s1d", "base round4"),
            ("focus_s1", "focus round1"), ("focus_s1b", "focus round2"),
            ("focus_s1c", "focus round3"), ("focus_s1d", "focus round4"),
            ("focus_g_s1", "focus_g (GIoU)")]
    for name, label in runs:
        f = ROOT / "runs" / name / "train_log.csv"
        if not f.exists():
            continue
        xs, ys = [], []
        with open(f, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("iter") != "epoch_avg":
                    continue
                xs.append(int(row["epoch"]))
                ys.append(float(row["loss"]))
        if xs:
            axes[0].plot(xs, ys, label=label, linewidth=1.6)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].set_yscale("log")
    axes[0].set_title("Training loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    # 从链式日志解析 eval mAP (扫描全部轮次; 每个变体只标注历史最优)
    cur = None
    best_seen = {}
    colors = {"base_s1": "#3b7dd8", "focus_s1": "#d84341", "focus_g_s1": "#2e9e5b",
              "focus_t": "#8e44ad", "base_t": "#e07b12"}
    for logname in ["train_chain.log", "train_chain_ft.log", "train_chain_C.log",
                    "train_chain_D.log"]:
        chain = ROOT / "runs" / logname
        if not chain.exists():
            continue
        for line in chain.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "====" in line:
                for tag in ["focus_g_s1", "base_s1", "focus_s1", "focus_t", "base_t"]:
                    if tag in line:
                        cur = tag
                        break
            if "eval ep" in line and cur:
                try:
                    ep = int(line.split("eval ep")[1].split("]")[0])
                    m50 = float(line.split("mAP50=")[1].split()[0])
                    c = colors.get(cur, "#666")
                    is_best = m50 >= best_seen.get(cur, -1)
                    axes[1].scatter([ep], [m50], s=18, color=c,
                                    marker="*" if is_best else "o")
                    if is_best:
                        best_seen[cur] = m50
                        axes[1].annotate(f"{cur} {m50:.3f}", (ep, m50), fontsize=6.5)
                except Exception:
                    pass
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val mAP50")
    axes[1].set_title("Validation mAP50 (val canvas)"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "training_curves.png", dpi=150)
    print("saved", OUT / "training_curves.png")

# ---------------------------------------------------------------- 检测网格

@torch.no_grad()
def detect_grid(ckpt, split="test", n=6, thr=0.3):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(ckpt, dev)
    imgs = sorted((DATA / "images" / split).glob("*.jpg"))
    idxs = np.linspace(0, len(imgs) - 1, n).astype(int)
    rows = []
    for i in idxs:
        p = imgs[i]
        img = load_img(p)
        H, W = img.shape[:2]
        x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(dev)
        out = model(x)
        pp = postprocess(out, W, H, score_thr=thr)[0]
        vis = (img * 255).astype(np.uint8).copy()
        boxes, labels = load_yolo_label(DATA / "labels" / split / (p.stem + ".txt"), W, H)
        for b, l in zip(ckpt_boxes_to_xyxy_px(boxes, W, H), labels.tolist()):
            _draw_box(vis, b, (0, 220, 0), CLASS_NAMES[l][:4])
        for b, l, s in zip(pp["boxes"].tolist(), pp["labels"].tolist(), pp["scores"].tolist()):
            _draw_box(vis, b, (230, 30, 30), f"{CLASS_NAMES[l][:4]} {s:.2f}")
        rows.append(vis)
    grid = np.vstack([np.hstack(rows[i:i + 3]) for i in range(0, len(rows), 3)])
    out_p = OUT / f"detect_grid_{Path(ckpt).parent.name}_{split}.jpg"
    imwrite_u(out_p, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), 90)
    print("saved", out_p, "| 绿=GT 红=预测")

# ---------------------------------------------------------------- 记忆可视化

@torch.no_grad()
def memory_viz(ckpt, seq_idx=0, thr=0.3):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(ckpt, dev)
    if model.memory is None:
        print("该 checkpoint 无记忆模块")
        return
    idx = json.loads((DATA / "video_index.json").read_text(encoding="utf-8"))
    seq = idx["test"][seq_idx]
    memory = None
    frames = []
    for t, rel in enumerate(seq["frames"]):
        img = load_img(DATA / rel)
        H, W = img.shape[:2]
        x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(dev)
        motion = None if t == 0 else (float(seq["stride"]), 0.0)
        out = model(x, memory_state=memory, motion=motion, return_memory=True)
        pp = postprocess(out, W, H, score_thr=thr)[0]
        vis = (img * 255).astype(np.uint8).copy()
        boxes, labels = load_yolo_label(DATA / "labels" / seq["split"] /
                                        (Path(rel).stem + ".txt"), W, H)
        for b in ckpt_boxes_to_xyxy_px(boxes, W, H):
            _draw_box(vis, b, (0, 200, 0))
        for b, l, s in zip(pp["boxes"].tolist(), pp["labels"].tolist(), pp["scores"].tolist()):
            _draw_box(vis, b, (230, 30, 30), f"{s:.2f}")
        # 记忆 token (对齐到当前帧坐标; CPU 上计算避免设备混用)
        if memory is not None:
            m_b = memory[0]["box"].cpu()
            m_age = memory[0]["age"].cpu()
            if m_b.numel():
                mb = (m_b - torch.tensor([seq["stride"], 0., seq["stride"], 0.])).tolist()
                for bb, ag in zip(mb, m_age.tolist()):
                    if bb[2] > 0 and bb[0] < W and bb[3] > 0 and bb[1] < H:
                        c = max(80, 255 - int(ag) * 60)
                        _draw_box(vis, [max(0, bb[0]), max(0, bb[1]), min(W, bb[2]), min(H, bb[3])],
                                  (60, 80, c), "mem", 1)
        probs = out["pred_logits"].sigmoid()
        scores_m, labels_m = probs.max(dim=-1)
        memory = model.update_memory(memory, out["_mem_raw"], W, H, scores_m, labels_m)
        frames.append(vis)
    row = np.hstack([cv2.resize(f, (256, 256)) for f in frames[:8]])
    out_p = OUT / f"memory_seq_{Path(ckpt).parent.name}.jpg"
    imwrite_u(out_p, cv2.cvtColor(row, cv2.COLOR_RGB2BGR), 92)
    print("saved", out_p, "| 绿=GT 红=预测 蓝=记忆token(越浅越老)")

# ---------------------------------------------------------------- 指标柱状图

def bars():
    names = ["base_s1d", "focus_s1d", "focus_g_s1", "base_t_s2e", "base_t_s2d",
             "base_t_s2c", "focus_t_s2c"]
    labels = ["base", "focus", "focus(GIoU)", "base+T (s2e-ep11)", "base+T (s2d)",
              "base+T (s2c)", "focus+T (s2c)"]
    data = {}
    for name in names:
        f = ROOT / "runs" / f"eval_{name}" / "results.json"
        if f.exists():
            data[name] = json.loads(f.read_text(encoding="utf-8"))
    if not data:
        print("暂无 eval 结果 json, 先运行 eval.py")
        return
    def get_metric(r, key, source="test"):
        if source == "test":
            m = r.get("test", {})
            return m.get(key, np.nan)
        # 视频类: 优先 test/mem-on, 其次 test/mem-off, 再 hard/mem-on
        for tag in [f"video_test_motion-gt_mem-on", f"video_test_motion-gt_mem-off",
                    f"video_test_hard_motion-gt_mem-on"]:
            if tag in r and key in r[tag]:
                return r[tag][key]
        return np.nan

    fig, axes = plt.subplots(1, 5, figsize=(19, 3.6))
    metrics = [("mAP50", "mAP50 (test)", "test"), ("mAP@small", "AP_S (test)", "test"),
               ("TCM", "TCM (video)", "video"),
               ("TP_stability", "TP stability (video)", "video"),
               ("recovery_delay_avg", "recovery delay (video)", "video")]
    palette = ["#3b7dd8", "#d84341", "#2e9e5b", "#e07b12", "#8e44ad", "#6c8ea0", "#b06c5a"]
    for ax, (key, title, src) in zip(axes, metrics):
        vals, labs, cols = [], [], []
        for i, (name, lab) in enumerate(zip(names, labels)):
            if name in data:
                v = get_metric(data[name], key, src)
                if v is None or (isinstance(v, float) and v != v):
                    continue
                vals.append(v); labs.append(lab); cols.append(palette[i % len(palette)])
        ax.bar(labs, vals, color=cols)
        ax.set_title(title); ax.tick_params(axis="x", rotation=25)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}" if v < 10 else f"{v:.1f}", ha="center", va="bottom",
                    fontsize=8)
        ax.set_ylim(0, (max(vals) * 1.25) if vals else 1)
    plt.tight_layout()
    plt.savefig(OUT / "variant_bars.png", dpi=150)
    print("saved", OUT / "variant_bars.png")

# ---------------------------------------------------------------- 演示视频

@torch.no_grad()
def demo_video(ckpt, seq_idx=0, thr=0.3, fps=6):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(ckpt, dev)
    idx = json.loads((DATA / "video_index.json").read_text(encoding="utf-8"))
    seq = idx["test"][seq_idx]
    writer = None
    memory = None
    for t, rel in enumerate(seq["frames"]):
        img = load_img(DATA / rel)
        H, W = img.shape[:2]
        x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(dev)
        # 记忆开
        motion = None if t == 0 else (float(seq["stride"]), 0.0)
        out = model(x, memory_state=memory, motion=motion, return_memory=True)
        pp = postprocess(out, W, H, score_thr=thr)[0]
        top = (img * 255).astype(np.uint8).copy()
        for b, l, s in zip(pp["boxes"].tolist(), pp["labels"].tolist(), pp["scores"].tolist()):
            _draw_box(top, b, (230, 30, 30), f"{CLASS_NAMES[l][:4]} {s:.2f}")
        cv2.putText(top, f"with memory (T-frame {t})", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        if model.memory is not None:
            probs = out["pred_logits"].sigmoid()
            scores_m, labels_m = probs.max(dim=-1)
            memory = model.update_memory(memory, out["_mem_raw"], W, H, scores_m, labels_m)
        # 记忆关
        out0 = model(x)
        pp0 = postprocess(out0, W, H, score_thr=thr)[0]
        bot = (img * 255).astype(np.uint8).copy()
        for b, l, s in zip(pp0["boxes"].tolist(), pp0["labels"].tolist(), pp0["scores"].tolist()):
            _draw_box(bot, b, (30, 30, 230), f"{CLASS_NAMES[l][:4]} {s:.2f}")
        cv2.putText(bot, f"without memory (frame {t})", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        frame = np.vstack([top, bot])
        if writer is None:
            h, w = frame.shape[:2]
            outp = OUT / f"demo_memory_{Path(ckpt).parent.name}.mp4"
            writer = cv2.VideoWriter(str(outp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if writer:
        writer.release()
        print("saved", OUT / f"demo_memory_{Path(ckpt).parent.name}.mp4")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["curves", "detect", "memory", "bars", "video"])
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seq", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "curves":
        curves()
    elif args.cmd == "detect":
        detect_grid(args.ckpt, args.split, args.n)
    elif args.cmd == "memory":
        memory_viz(args.ckpt, args.seq)
    elif args.cmd == "bars":
        bars()
    elif args.cmd == "video":
        demo_video(args.ckpt, args.seq)

if __name__ == "__main__":
    main()
