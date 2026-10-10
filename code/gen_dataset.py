# -*- coding: utf-8 -*-
"""
合成船舶壁面缺陷数据集生成器（视频序列版）
================================================
为「Focus DETR 复现 + 时序视频焦点记忆」仿真实验提供数据。

设计要点:
  1. 程序化生成大尺寸船体壁面画布: 钢板底材 + 板缝/焊缝 + 5 类缺陷 + 未标注干扰物 + 光照场
  2. 帧渲染: 沿巡检路径滑动窗口裁剪 512x512, 叠加逐帧光照抖动/噪声/运动模糊 (视频序列帧间连续)
  3. 标签自动生成: YOLO 格式 (cls cx cy w h, 归一化); 裂纹按链式盒标注薄长目标
  4. 类别: 0 rust 锈蚀 | 1 crack 裂纹 | 2 weld_defect 焊缝缺陷 | 3 peel 涂层剥落 | 4 pitting 点蚀(小目标)
  5. 划分: train/val/test/test_hard, 画布级隔离 (不同种子), test_hard 为强光照/强干扰/强模糊变体

输出目录 (--out):
  images/{split}/*.jpg         帧图像
  labels/{split}/*.txt         YOLO 标签
  meta/{split}.jsonl           帧元数据 (含视频序列 id/时间步/滑窗偏移/画布种子 -> 帧间运动已知)
  video_index.json             视频序列清单 (时序模型用)
  debug/canvas_*.jpg           画布预览 (带标注框, 用于文档配图)
"""
import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = ["rust", "crack", "weld_defect", "peel", "pitting"]
CLASS_NAMES_CN = ["锈蚀", "裂纹", "焊缝缺陷", "涂层剥落", "点蚀"]

# ----------------------------- 基础工具 -----------------------------

def fractal_noise(shape, rng, octaves=6, base=4, persistence=0.55, interp=cv2.INTER_CUBIC):
    """分形噪声, 返回 [0,1] float32. 低分辨率白噪声逐级上采样叠加."""
    h, w = shape
    total = np.zeros((h, w), np.float32)
    amp, res = 1.0, base
    for _ in range(octaves):
        g = rng.random((res, max(2, int(res * w / h)))).astype(np.float32)
        g = cv2.resize(g, (w, h), interpolation=interp)
        total += amp * g
        amp *= persistence
        res = min(int(res * 2), 256)
    total -= total.min()
    total /= (total.max() + 1e-8)
    return total

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)

def clip_box(x0, y0, x1, y1, W, H):
    return [float(max(0, min(x0, W))), float(max(0, min(y0, H))),
            float(max(0, min(x1, W))), float(max(0, min(y1, H)))]

# ----------------------------- 画布: 底材与焊缝 -----------------------------

def make_base_canvas(H, W, rng):
    """钢板底材: 蓝灰基色 + 分形起伏 + 拉丝纹理 + 板缝暗线. 返回 float32 BGR."""
    base_col = np.array([128.0, 127.0, 124.0], np.float32) + rng.normal(0, 7, 3).astype(np.float32)
    canvas = np.ones((H, W, 3), np.float32) * base_col

    n1 = fractal_noise((H, W), rng, octaves=7, base=3)          # 大尺度起伏
    canvas += ((n1 - 0.5) * 22.0)[..., None]

    # 拉丝: 横向拉长噪声 (低分辨率行向量->上采样)
    streak = rng.random((max(4, H // 24), 6)).astype(np.float32)
    streak = cv2.resize(streak, (W, H), interpolation=cv2.INTER_LINEAR)
    canvas += ((streak - 0.5) * 14.0)[..., None]

    n2 = fractal_noise((H, W), rng, octaves=8, base=6)          # 细粒度
    canvas += ((n2 - 0.5) * 8.0)[..., None]

    # 板缝 (每隔 550~800px 一条竖缝, 600~800 一条横缝)
    seams_v, x = [], int(rng.integers(450, 700))
    while x < W - 80:
        seams_v.append(x)
        cv2.line(canvas, (x, 0), (x, H), (86, 86, 88), 3, cv2.LINE_AA)
        cv2.line(canvas, (x + 2, 0), (x + 2, H), (168, 166, 162), 1, cv2.LINE_AA)
        x += int(rng.integers(550, 800))
    seams_h, y = [], int(rng.integers(420, 620))
    while y < H - 80:
        seams_h.append(y)
        cv2.line(canvas, (0, y), (W, y), (86, 86, 88), 3, cv2.LINE_AA)
        cv2.line(canvas, (0, y + 2), (W, y + 2), (168, 166, 162), 1, cv2.LINE_AA)
        y += int(rng.integers(600, 800))

    canvas = np.clip(canvas, 0, 255)
    return canvas, seams_v, seams_h

def draw_weld_band(canvas, p0, p1, rng, width=None):
    """在板缝上绘制焊缝带 (波纹纹理). 返回焊缝中心线的稠密点列表 (供焊缝缺陷放置)."""
    H, W = canvas.shape[:2]
    width = width or int(rng.integers(26, 42))
    mask_line = np.zeros((H, W), np.uint8)
    cv2.line(mask_line, p0, p1, 255, 1, cv2.LINE_AA)
    dist = cv2.distanceTransform(255 - mask_line, cv2.DIST_L2, 3)
    alpha = smoothstep((width * 0.5 - dist) / 4.0) * 0.88
    ripple = 0.5 + 0.5 * np.cos(dist * (2 * np.pi / rng.uniform(7.5, 10.5)))
    col = 150.0 + 20.0 * ripple
    noise = fractal_noise((H, W), rng, octaves=6, base=8)
    col = col + (noise - 0.5) * 14.0
    for c in range(3):
        canvas[..., c] = canvas[..., c] * (1 - alpha) + (col + rng.normal(0, 2)) * alpha
    # 飞溅小点
    for _ in range(int(rng.integers(0, 26))):
        t = rng.random()
        px, py = int(p0[0] + (p1[0] - p0[0]) * t), int(p0[1] + (p1[1] - p0[1]) * t)
        px += int(rng.normal(0, width))
        py += int(rng.normal(0, width))
        if 0 <= px < W and 0 <= py < H:
            cv2.circle(canvas, (px, py), int(rng.integers(1, 3)), (95, 95, 96), -1, cv2.LINE_AA)
    # 中心线采样点
    L = int(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    pts = [(int(p0[0] + (p1[0] - p0[0]) * t / L), int(p0[1] + (p1[1] - p0[1]) * t / L))
           for t in range(0, L, 12)]
    return pts

# ----------------------------- 缺陷绘制 (局部 ROI 操作) -----------------------------

def _patch_slice(H, W, cx, cy, rx, ry, margin=48):
    x0, x1 = max(0, int(cx - rx - margin)), min(W, int(cx + rx + margin))
    y0, y1 = max(0, int(cy - ry - margin)), min(H, int(cy + ry + margin))
    return x0, y0, x1, y1

def add_rust(canvas, rng, noise_field, rx=None, ry=None, cx=None, cy=None):
    """锈蚀斑块: 分形噪声调制椭圆掩码, 橙棕色, 深色核心. 返回 bbox list."""
    H, W = canvas.shape[:2]
    rx = rx or float(rng.integers(40, 260))
    ry = ry or rx * float(rng.uniform(0.55, 1.0))
    cx = cx if cx is not None else int(rng.integers(int(rx) + 10, W - int(rx) - 10))
    cy = cy if cy is not None else int(rng.integers(int(ry) + 10, H - int(ry) - 10))
    x0, y0, x1, y1 = _patch_slice(H, W, cx, cy, rx, ry)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    r = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    # 从全局噪声场随机取样局部块 (保证形状多样且便宜)
    nh, nw = y1 - y0, x1 - x0
    sy = int(rng.integers(0, max(1, noise_field.shape[0] - nh)))
    sx = int(rng.integers(0, max(1, noise_field.shape[1] - nw)))
    nz = noise_field[sy:sy + nh, sx:sx + nw]
    if nz.shape[:2] != (nh, nw):  # 边界裁剪兜底
        nz = cv2.resize(nz, (nw, nh))
    mask = smoothstep(1.35 - r * 1.35 + (nz - 0.5) * 1.1)
    mask = np.clip(mask, 0, 1) * float(rng.uniform(0.75, 0.95))
    # 颜色: 橙棕, 核心更深
    base = np.array([58.0, 96.0, 148.0], np.float32) + rng.normal(0, 14, 3).astype(np.float32)  # BGR
    shade = np.clip(0.62 + 0.55 * (1 - r), 0.5, 1.15)[..., None]
    npx = fractal_noise((nh, nw), rng, octaves=6, base=16)
    col = base * shade * (0.8 + 0.4 * npx[..., None])
    # 叠加
    seg = canvas[y0:y1, x0:x1]
    m3 = mask[..., None]
    canvas[y0:y1, x0:x1] = seg * (1 - m3) + col * m3
    # bbox (掩码 > 0.35 区域)
    ys, xs = np.where(mask > 0.35)
    if len(xs) == 0:
        return []
    bx, by = x0 + xs.min(), y0 + ys.min()
    bx1, by1 = x0 + xs.max(), y0 + ys.max()
    if (bx1 - bx) < 6 or (by1 - by) < 6:
        return []
    return [[float(bx), float(by), float(bx1), float(by1)]]

def add_pitting(canvas, rng, cx=None, cy=None, n=None, on_rust=False):
    """点蚀: 小暗圆点 (10~16px 直径, 小目标). 返回逐点 bbox."""
    H, W = canvas.shape[:2]
    n = n or int(rng.integers(3, 9))
    boxes = []
    cx = cx if cx is not None else int(rng.integers(40, W - 40))
    cy = cy if cy is not None else int(rng.integers(40, H - 40))
    spread = float(rng.uniform(20, 90))
    for _ in range(n):
        px = int(np.clip(cx + rng.normal(0, spread), 12, W - 12))
        py = int(np.clip(cy + rng.normal(0, spread), 12, H - 12))
        rad = int(rng.integers(5, 9))
        col = (34, 36, 40) if not on_rust else (30, 52, 84)
        cv2.circle(canvas, (px, py), rad, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px + 1, py - 1), max(1, rad - 2), tuple(int(c * 0.7) for c in col), -1, cv2.LINE_AA)
        # 高光弧
        cv2.ellipse(canvas, (px, py), (rad, rad), 0, 200, 320, (170, 168, 164), 1, cv2.LINE_AA)
        boxes.append([float(px - rad), float(py - rad), float(px + rad), float(py + rad)])
    return boxes

def add_crack(canvas, rng, x0=None, y0=None, length=None, vertical=None):
    """裂纹: 随机游走折线, 带高光边缘; 链式盒标注 (薄长目标). 返回 bbox list."""
    H, W = canvas.shape[:2]
    vertical = rng.random() < 0.5 if vertical is None else vertical
    length = length or float(rng.integers(70, 360))
    x = x0 if x0 is not None else int(rng.integers(60, W - 60))
    y = y0 if y0 is not None else int(rng.integers(60, H - 60))
    theta = rng.uniform(-0.5, 0.5) + (math.pi / 2 if vertical else 0)
    pts, total = [(x, y)], 0.0
    while total < length:
        theta += rng.normal(0, 0.22)
        step = float(rng.uniform(4, 9))
        x += step * math.cos(theta)
        y += step * math.sin(theta)
        if not (4 < x < W - 4 and 4 < y < H - 4):
            break
        pts.append((int(x), int(y)))
        total += step
    if len(pts) < 4:
        return []
    # 高光边 (先画, 偏移 1px)
    hi = [(p[0] + 1, p[1] - 1) for p in pts]
    cv2.polylines(canvas, [np.array(hi)], False, (168, 166, 162), 1, cv2.LINE_AA)
    # 主体: 宽度渐变
    nseg = len(pts) - 1
    for i in range(nseg):
        t = i / max(1, nseg - 1)
        th = max(1, int(round(2 + 3 * (1 - t) * rng.uniform(0.7, 1.0))))
        cv2.line(canvas, pts[i], pts[i + 1], (26, 27, 30), th, cv2.LINE_AA)
    # 分支
    if len(pts) > 8 and rng.random() < 0.3:
        bi = int(rng.integers(2, len(pts) - 2))
        bth = theta + rng.uniform(-1.2, 1.2)
        bx, by = pts[bi]
        bpts = [(bx, by)]
        for _ in range(int(rng.integers(4, 14))):
            bth += rng.normal(0, 0.3)
            bx += 6 * math.cos(bth)
            by += 6 * math.sin(bth)
            bpts.append((int(bx), int(by)))
        cv2.polylines(canvas, [np.array(bpts)], False, (30, 31, 34), 1, cv2.LINE_AA)
    # 链式盒: 每 ~55px 一段
    boxes, chunk = [], []
    acc = 0.0
    last = pts[0]
    for p in pts[1:]:
        chunk.append(p)
        acc += math.hypot(p[0] - last[0], p[1] - last[1])
        last = p
        if acc >= 55:
            arr = np.array(chunk)
            pad = 4
            boxes.append([float(arr[:, 0].min() - pad), float(arr[:, 1].min() - pad),
                          float(arr[:, 0].max() + pad), float(arr[:, 1].max() + pad)])
            chunk, acc = [], 0.0
    if len(chunk) >= 2:
        arr = np.array(chunk)
        pad = 4
        boxes.append([float(arr[:, 0].min() - pad), float(arr[:, 1].min() - pad),
                      float(arr[:, 0].max() + pad), float(arr[:, 1].max() + pad)])
    return boxes

def add_peel(canvas, rng, noise_field, rx=None, ry=None, cx=None, cy=None):
    """涂层剥落: 不规则多边形区域, 色相偏移 + 亮边缘. 返回 bbox list."""
    H, W = canvas.shape[:2]
    rx = rx or float(rng.integers(70, 240))
    ry = ry or rx * float(rng.uniform(0.6, 1.0))
    cx = cx if cx is not None else int(rng.integers(int(rx) + 10, W - int(rx) - 10))
    cy = cy if cy is not None else int(rng.integers(int(ry) + 10, H - int(ry) - 10))
    x0, y0, x1, y1 = _patch_slice(H, W, cx, cy, rx, ry)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    r = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    nh, nw = y1 - y0, x1 - x0
    sy = int(rng.integers(0, max(1, noise_field.shape[0] - nh)))
    sx = int(rng.integers(0, max(1, noise_field.shape[1] - nw)))
    nz = noise_field[sy:sy + nh, sx:sx + nw]
    if nz.shape[:2] != (nh, nw):
        nz = cv2.resize(nz, (nw, nh))
    mask = smoothstep(1.3 - r * 1.3 + (nz - 0.5) * 1.0)
    alpha = mask * 0.92
    # 颜色: 底材去饱和偏暗 + 微绿底漆倾向
    shift = np.array([rng.uniform(-16, 6), rng.uniform(-12, 10), rng.uniform(-8, 14)], np.float32)
    seg = canvas[y0:y1, x0:x1]
    col = np.clip(seg * float(rng.uniform(0.72, 0.9)) + shift, 0, 255)
    m3 = alpha[..., None]
    canvas[y0:y1, x0:x1] = seg * (1 - m3) + col * m3
    # 亮边缘 (形态学梯度, 轻量)
    mb = (mask > 0.5).astype(np.uint8)
    edge = cv2.morphologyEx(mb, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edge = cv2.GaussianBlur(edge.astype(np.float32), (3, 3), 0) * 0.38
    e3 = edge[..., None]
    seg = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = seg * (1 - e3) + 180.0 * e3
    ys, xs = np.where(mask > 0.4)
    if len(xs) == 0:
        return []
    return [[float(x0 + xs.min()), float(y0 + ys.min()), float(x0 + xs.max()), float(y0 + ys.max())]]

def add_weld_defect(canvas, rng, weld_pts):
    """焊缝缺陷: 在焊缝中心线上的短暗色锯齿痕迹 (气孔/夹渣). 返回 bbox list."""
    if len(weld_pts) < 6:
        return []
    i = int(rng.integers(2, len(weld_pts) - 4))
    px, py = weld_pts[i]
    H, W = canvas.shape[:2]
    L = int(rng.integers(14, 42))
    theta = rng.uniform(0, 2 * math.pi)
    pts, x, y = [(px, py)], float(px), float(py)
    for _ in range(max(3, L // 5)):
        theta += rng.normal(0, 0.55)
        x += 5 * math.cos(theta)
        y += 5 * math.sin(theta)
        pts.append((int(x), int(y)))
    arr = np.array(pts)
    if arr[:, 0].min() < 2 or arr[:, 0].max() > W - 2 or arr[:, 1].min() < 2 or arr[:, 1].max() > H - 2:
        return []
    # 暗色痕迹 + 锈色晕
    cv2.polylines(canvas, [arr], False, (70, 66, 60), 6, cv2.LINE_AA)
    cv2.polylines(canvas, [arr], False, (34, 34, 38), 3, cv2.LINE_AA)
    pad = 6
    return [[float(arr[:, 0].min() - pad), float(arr[:, 1].min() - pad),
             float(arr[:, 0].max() + pad), float(arr[:, 1].max() + pad)]]

# ----------------------------- 干扰物 (不标注) -----------------------------

def add_distractors(canvas, rng, seams_v, seams_h, amount=1.0):
    H, W = canvas.shape[:2]
    # 水流痕 (自上而下半透明带)
    for _ in range(int(rng.integers(2, 7) * amount)):
        x = int(rng.integers(20, W - 20))
        wdt = int(rng.integers(24, 130))
        depth = int(rng.integers(H // 4, H))
        alpha = float(rng.uniform(0.04, 0.13))
        col = np.array([205.0, 200.0, 188.0], np.float32)
        y1c, y2c = 0, depth
        seg = canvas[y1c:y2c, max(0, x - wdt):min(W, x + wdt)]
        if seg.size == 0:
            continue
        wseg = seg.shape[1]
        gt = np.linspace(1.0, 0.15, wseg, dtype=np.float32)[None, :, None]
        canvas[y1c:y2c, max(0, x - wdt):min(W, x + wdt)] = seg * (1 - alpha * gt) + col * alpha * gt
    # 污渍 (低透明度棕色斑)
    for _ in range(int(rng.integers(2, 6) * amount)):
        cx, cy = int(rng.integers(40, W - 40)), int(rng.integers(40, H - 40))
        rx, ry = int(rng.integers(40, 160)), int(rng.integers(30, 120))
        x0, y0, x1, y1 = _patch_slice(H, W, cx, cy, rx, ry, margin=8)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        r = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
        m = smoothstep(1.2 - r * 1.2) * float(rng.uniform(0.05, 0.11))
        seg = canvas[y0:y1, x0:x1]
        canvas[y0:y1, x0:x1] = seg * (1 - m[..., None]) + np.array([86.0, 104.0, 122.0]) * m[..., None]
    # 铆钉/螺栓 (沿板缝, 潜在误检源)
    if rng.random() < 0.6:
        for sx in seams_v[:2]:
            for yy in range(int(rng.integers(80, 200)), H - 60, int(rng.integers(90, 150))):
                rad = int(rng.integers(4, 7))
                cv2.circle(canvas, (sx + int(rng.integers(-18, 18)), yy), rad, (96, 96, 98), -1, cv2.LINE_AA)
                cv2.circle(canvas, (sx + int(rng.integers(-18, 18)), yy - 1), max(2, rad - 3), (150, 148, 144), -1, cv2.LINE_AA)
    # 模板喷字 (白/红淡色)
    if rng.random() < 0.5:
        txt = f"{chr(65 + int(rng.integers(0, 26)))}-{int(rng.integers(1, 30))}"
        col = (215, 215, 210) if rng.random() < 0.7 else (60, 60, 190)
        pos = (int(rng.integers(60, W - 200)), int(rng.integers(60, H - 60)))
        cv2.putText(canvas, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(0.8, 1.6)), col, 2, cv2.LINE_AA)

# ----------------------------- 画布总装 -----------------------------

def build_canvas(seed, H, W, hard=False):
    rng = np.random.default_rng(seed)
    canvas, seams_v, seams_h = make_base_canvas(H, W, rng)
    noise_field = fractal_noise((H // 2, W // 2), rng, octaves=7, base=5)  # 供掩码取块

    # 焊缝 (选取部分板缝)
    weld_lines = []
    for x in seams_v[:max(1, len(seams_v) // 2)]:
        y0 = int(rng.integers(0, H // 3))
        y1 = int(rng.integers(2 * H // 3, H))
        weld_lines.append(draw_weld_band(canvas, (x, y0), (x, y1), rng))
    for y in seams_h[:max(1, len(seams_h) // 2)]:
        x0 = int(rng.integers(0, W // 3))
        x1 = int(rng.integers(2 * W // 3, W))
        weld_lines.append(draw_weld_band(canvas, (x0, y), (x1, y), rng))

    defects = []  # dict(cls, box)
    # 锈蚀
    for _ in range(int(rng.integers(10, 18)) + (4 if hard else 0)):
        for b in add_rust(canvas, rng, noise_field):
            defects.append({"cls": 0, "box": b})
    # 点蚀簇 (小目标)
    for _ in range(int(rng.integers(4, 8)) + (2 if hard else 0)):
        for b in add_pitting(canvas, rng):
            defects.append({"cls": 4, "box": b})
    # 裂纹
    for _ in range(int(rng.integers(4, 9)) + (2 if hard else 0)):
        for b in add_crack(canvas, rng):
            defects.append({"cls": 1, "box": b})
    # 涂层剥落
    for _ in range(int(rng.integers(2, 6))):
        for b in add_peel(canvas, rng, noise_field):
            defects.append({"cls": 3, "box": b})
    # 焊缝缺陷
    for wl in weld_lines:
        for _ in range(int(rng.integers(0, 3))):
            for b in add_weld_defect(canvas, rng, wl):
                defects.append({"cls": 2, "box": b})

    add_distractors(canvas, rng, seams_v, seams_h, amount=1.6 if hard else 1.0)

    # 画布外框裁掉越界 box
    final = []
    for d in defects:
        x0, y0, x1, y1 = d["box"]
        x0, y0, x1, y1 = clip_box(x0, y0, x1, y1, W, H)
        if x1 - x0 >= 5 and y1 - y0 >= 5:
            final.append({"cls": d["cls"], "box": [x0, y0, x1, y1]})
    return np.clip(canvas, 0, 255).astype(np.float32), final

# ----------------------------- 帧渲染 -----------------------------

def apply_light(f, rng, hard=False):
    H, W = f.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    gx, gy = rng.uniform(-1, 1), rng.uniform(-1, 1)
    s = float(rng.uniform(0.10, 0.22)) * (1.8 if hard else 1.0)
    gain = 1.0 + s * ((xx / W - 0.5) * gx + (yy / H - 0.5) * gy) * 2.0
    # 晕影
    r2 = ((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2) / 0.5
    gain *= 1.0 - float(rng.uniform(0.06, 0.16)) * r2 * (1.6 if hard else 1.0)
    f = f * gain[..., None] * float(rng.uniform(0.86, 1.12))
    # 伽马
    g = float(rng.uniform(0.85, 1.15)) * (0.75 if hard else 1.0)
    f = 255.0 * np.power(np.clip(f / 255.0, 0, 1), g)
    return f

def motion_blur(f, rng, motion=(0.0, 0.0), hard=False):
    ang = math.atan2(motion[1], motion[0]) if (abs(motion[0]) + abs(motion[1])) > 1e-3 else rng.uniform(0, math.pi)
    ang += rng.normal(0, 0.12) if (abs(motion[0]) + abs(motion[1])) > 1e-3 else 0.0
    k = int(rng.integers(5, 15) if hard else rng.integers(3, 9))
    kern = np.zeros((k, k), np.float32)
    c = (k - 1) / 2
    for t in np.linspace(-c, c, k * 2):
        x = int(round(c + t * math.cos(ang)))
        y = int(round(c + t * math.sin(ang)))
        if 0 <= x < k and 0 <= y < k:
            kern[y, x] = 1.0
    s = kern.sum()
    if s == 0:
        return f
    return cv2.filter2D(f, -1, kern / s)

def render_frame(canvas, ox, oy, W, H, rng, hard=False, motion=(0.0, 0.0), illum_scale=0.35):
    f = np.array(canvas[oy:oy + H, ox:ox + W], np.float32)
    # 光照: 画布级基础 + 帧级抖动
    f = apply_light(f, rng, hard)
    if rng.random() < (0.8 if hard else 0.45):
        f = motion_blur(f, rng, motion, hard)
    f += rng.normal(0, 4.5 if hard else 2.8, f.shape).astype(np.float32)
    return np.clip(f, 0, 255)

def boxes_in_window(defects, ox, oy, W, H, min_vis=0.35, min_size=4.0):
    """收集窗口中可见的缺陷框, 裁剪到窗口并过滤可见性过低的框."""
    out = []
    for d in defects:
        x0, y0, x1, y1 = d["box"]
        ix0, iy0 = max(x0, ox), max(y0, oy)
        ix1, iy1 = min(x1, ox + W), min(y1, oy + H)
        if ix1 - ix0 < 1 or iy1 - iy0 < 1:
            continue
        vis = (ix1 - ix0) * (iy1 - iy0) / max(1e-6, (x1 - x0) * (y1 - y0))
        if vis < min_vis or (ix1 - ix0) < min_size or (iy1 - iy0) < min_size:
            continue
        out.append((d["cls"], [ix0 - ox, iy0 - oy, ix1 - ox, iy1 - oy], vis))
    return out

def write_label(path, boxes, W, H):
    lines = []
    for cls, (x0, y0, x1, y1), _ in boxes:
        cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
        bw, bh = (x1 - x0) / W, (y1 - y0) / H
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")

def save_frame(f, path, quality=90):
    """兼容中文/绝对路径的 JPEG 写入 (cv2.imwrite 对非 ASCII 路径静默失败)."""
    ok, buf = cv2.imencode(".jpg", f.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"imencode 失败: {path}")
    buf.tofile(str(path))

# ----------------------------- 采样与主流程 -----------------------------

def sample_positions(rng, W, H, crop, n, margin=0):
    xs = rng.integers(0, W - crop - margin + 1, n)
    ys = rng.integers(0, H - crop - margin + 1, n)
    return list(zip(xs.tolist(), ys.tolist()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent.parent / "data" / "defect_ds"))
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--canvas", type=int, nargs=2, default=[3072, 1536])
    ap.add_argument("--seed0", type=int, default=20261002)
    ap.add_argument("--quick", action="store_true", help="冒烟测试: 小规模")
    ap.add_argument("--extra_split", type=str, default=None,
                    help="仅生成附加视频测试集 (如 testbig): 只产视频帧, 追加到 video_index")
    ap.add_argument("--extra_canvases", type=int, default=20)
    ap.add_argument("--extra_seqs", type=int, default=10)
    args = ap.parse_args()

    out = Path(args.out)
    for sub in ["images", "labels", "meta", "debug"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    CW, CH = args.canvas
    R = args.crop
    rng = np.random.default_rng(args.seed0 + 7)

    # 划分: (split名, 画布数, 每画布静帧数, 每画布视频序列数, hard)
    if args.extra_split:
        # 仅生成附加视频测试集 (0 静帧), 用于扩大时序指标样本量
        plan = [(args.extra_split, args.extra_canvases, 0, args.extra_seqs, False)]
    elif args.quick:
        plan = [("train", 1, 12, 2, False), ("val", 1, 6, 1, False),
                ("test", 1, 8, 1, False), ("test_hard", 1, 6, 1, True)]
    else:
        plan = [("train", 6, 620, 8, False), ("val", 1, 420, 6, False),
                ("test", 2, 420, 6, False), ("test_hard", 2, 420, 6, True)]

    video_index = {}
    used_seeds = set()
    for split, n_canvas, n_still, n_seq, hard in plan:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        meta_f = open(out / "meta" / f"{split}.jsonl", "w", encoding="utf-8")
        seq_list = []
        for ci in range(n_canvas):
            while True:
                seed = int(rng.integers(1, 10 ** 9))
                if seed not in used_seeds:
                    used_seeds.add(seed)
                    break
            canvas, defects = build_canvas(seed, CH, CW, hard=hard)
            canvas8 = canvas.astype(np.uint8)
            # 调试预览: 画布 + 标注框
            dbg = canvas8.copy()
            for d in defects:
                x0, y0, x1, y1 = [int(v) for v in d["box"]]
                cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(dbg, CLASS_NAMES[d["cls"]][:2], (x0, max(12, y0 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            save_frame(cv2.resize(dbg, (CW // 2, CH // 2)).astype(np.float32),
                       out / "debug" / f"canvas_{split}_{ci}_seed{seed}.jpg", quality=88)
            print(f"[{split}] canvas {ci+1}/{n_canvas} seed={seed} 缺陷数={len(defects)}", flush=True)

            # --- 静帧采样 ---
            n_ok, tries = 0, 0
            while n_ok < n_still and tries < n_still * 6:
                tries += 1
                ox = int(rng.integers(0, CW - R))
                oy = int(rng.integers(0, CH - R))
                boxes = boxes_in_window(defects, ox, oy, R, R)
                if len(boxes) == 0:
                    continue
                f = render_frame(canvas, ox, oy, R, R, rng, hard=hard)
                name = f"{split}_{ci:02d}_{n_ok:04d}"
                save_frame(f, out / "images" / split / f"{name}.jpg")
                write_label(out / "labels" / split / f"{name}.txt", boxes, R, R)
                meta_f.write(json.dumps({"img": f"images/{split}/{name}.jpg", "seq": None,
                                         "t": -1, "ox": ox, "oy": oy, "canvas": seed, "hard": hard}) + "\n")
                n_ok += 1

            # --- 视频序列 (沿巡检路径: 横向扫掠, 行间纵移) ---
            seqs = 0
            step = 128
            n_pos = (CW - R) // step          # 每行可用位姿数 (位姿索引 0..n_pos)
            n_rows = max(1, (CH - R) // 260 + 1)
            for row in range(n_rows):
                if seqs >= n_seq:
                    break
                oy = min(CH - R, row * 260)
                for s in range(4):
                    if seqs >= n_seq:
                        break
                    base = int(rng.integers(0, max(1, n_pos - 11)))  # 随机起点
                    if base + 11 > n_pos:
                        continue
                    sx = base * step
                    sid = f"{split}_c{ci}_r{row}_s{s}"
                    frames = []
                    for t in range(12):
                        ox = min(CW - R, sx + t * step)
                        boxes = boxes_in_window(defects, ox, oy, R, R)
                        fn = f"{split}_{ci:02d}_seq_{row}_{s}_{t:02d}"
                        f = render_frame(canvas, ox, oy, R, R, rng, hard=hard,
                                         motion=(step, 0.0))
                        save_frame(f, out / "images" / split / f"{fn}.jpg")
                        write_label(out / "labels" / split / f"{fn}.txt", boxes, R, R)
                        meta_f.write(json.dumps({"img": f"images/{split}/{fn}.jpg", "seq": sid,
                                                 "t": t, "ox": ox, "oy": oy, "canvas": seed, "hard": hard}) + "\n")
                        frames.append(f"images/{split}/{fn}.jpg")
                    seq_list.append({"id": sid, "split": split, "frames": frames,
                                     "stride": step, "canvas": seed})
                    seqs += 1
        video_index[split] = seq_list
        meta_f.close()
        print(f"[{split}] 完成: {n_canvas} 画布, 序列 {len(seq_list)} 段", flush=True)

    vi_path = out / "video_index.json"
    if vi_path.exists():                      # 追加模式: 合并既有索引
        try:
            old = json.loads(vi_path.read_text(encoding="utf-8"))
            old.update(video_index)
            video_index = old
        except Exception:
            pass
    vi_path.write_text(json.dumps(video_index, ensure_ascii=False), encoding="utf-8")
    (out / "classes.json").write_text(json.dumps({"names": CLASS_NAMES, "names_cn": CLASS_NAMES_CN},
                                                 ensure_ascii=False), encoding="utf-8")
    print("数据集生成完成 ->", out, flush=True)

if __name__ == "__main__":
    main()
