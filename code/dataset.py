# -*- coding: utf-8 -*-
"""数据集加载与增强: 帧级 (静帧+视频帧) 与 视频片段级 (时序训练)."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = None  # 本仿真不采用 ImageNet 预训练, 仅做 /255 归一化

def load_img(path):
    """兼容中文路径的图像读取 (Windows 下 cv2.imread 对非 ASCII 路径失效)."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"无法读取图像: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img  # H,W,3

def load_yolo_label(path, W, H):
    boxes, labels = [], []
    p = Path(path)
    if not p.exists():
        return torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)
    for line in p.read_text(encoding="utf-8").strip().split("\n"):
        if not line.strip():
            continue
        c, cx, cy, w, h = line.split()
        labels.append(int(c))
        boxes.append([float(cx), float(cy), float(w), float(h)])
    if not boxes:
        return torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)
    return (torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            torch.tensor(labels, dtype=torch.long))

# ---------------------------------------------------------------- 增强

def _rot90_boxes(boxes):
    """cxcywh 归一化: (xn,yn)->(yn,1-xn), swap w,h."""
    xn, yn, w, h = boxes.unbind(-1)
    return torch.stack([yn, 1 - xn, h, w], -1)

def _hflip_boxes(boxes):
    xn, yn, w, h = boxes.unbind(-1)
    return torch.stack([1 - xn, yn, w, h], -1)

def _vflip_boxes(boxes):
    xn, yn, w, h = boxes.unbind(-1)
    return torch.stack([xn, 1 - yn, w, h], -1)

def augment_frame(img, boxes, labels, rng):
    """img: HWC float [0,1]; boxes: cxcywh 归一化."""
    k = int(rng.integers(0, 4))
    if k:
        img = np.rot90(img, k=k).copy()
        for _ in range(k):
            boxes = _rot90_boxes(boxes)
    if rng.random() < 0.5:
        img = img[:, ::-1].copy()
        boxes = _hflip_boxes(boxes)
    if rng.random() < 0.5:
        img = img[::-1, :].copy()
        boxes = _vflip_boxes(boxes)
    # 光度扰动
    if rng.random() < 0.8:
        b = rng.uniform(0.78, 1.28)
        c = rng.uniform(0.82, 1.22)
        img = (img * 255.0)
        img = (img - 128.0) * c + 128.0
        img = img * b
        img = np.clip(img, 0, 255) / 255.0
    if rng.random() < 0.5:
        g = rng.uniform(0.85, 1.15)
        img = np.power(np.clip(img, 0, 1), g)
    if rng.random() < 0.5:
        img = np.clip(img + rng.normal(0, 0.012, img.shape), 0, 1)
    return img.astype(np.float32), boxes, labels

# ---------------------------------------------------------------- 帧级数据集

class DefectDataset(Dataset):
    """帧级目标检测数据集 (静帧 + 视频帧)."""

    def __init__(self, root, split, img_size=512, augment=False, limit=None, seed=0):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        imgs = sorted((self.root / "images" / split).glob("*.jpg"))
        if limit:
            imgs = imgs[:limit]
        self.items = imgs

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p = self.items[idx]
        lbl = self.root / "labels" / self.split / (p.stem + ".txt")
        img = load_img(p)
        H, W = img.shape[:2]
        boxes, labels = load_yolo_label(lbl, W, H)
        if self.augment:
            img, boxes, labels = augment_frame(img, boxes, labels, self.rng)
        img = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()
        return {"img": img, "target": {"labels": labels, "boxes": boxes, "img_id": p.stem}}

# ---------------------------------------------------------------- 视频片段数据集

class VideoClipDataset(Dataset):
    """视频序列数据集: 每项为一段连续帧 (时序焦点记忆训练/评估用)."""

    def __init__(self, root, split, limit=None, augment=False):
        self.root = Path(root)
        idx = json.loads((self.root / "video_index.json").read_text(encoding="utf-8"))
        self.seqs = idx.get(split, [])
        if limit:
            self.seqs = self.seqs[:limit]
        self.augment = augment
        self.rng = np.random.default_rng(1234)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        frames, motions = [], []
        for i, rel in enumerate(seq["frames"]):
            p = self.root / rel
            img = load_img(p)
            H, W = img.shape[:2]
            boxes, labels = load_yolo_label(
                self.root / "labels" / seq["split"] / (p.stem + ".txt"), W, H)
            frames.append({"img": torch.from_numpy(img.transpose(2, 0, 1)).contiguous(),
                           "target": {"labels": labels, "boxes": boxes, "img_id": p.stem}})
            if i == 0:
                motions.append([0.0, 0.0])
            else:
                motions.append([float(seq["stride"]), 0.0])
        return {"frames": frames, "motions": torch.tensor(motions, dtype=torch.float32),
                "seq_id": seq["id"]}

def collate_seqs(batch):
    B = len(batch)
    T = len(batch[0]["frames"])
    imgs = torch.stack([torch.stack([f["img"] for f in s["frames"]]) for s in batch])  # B,T,3,H,W
    motions = torch.stack([s["motions"] for s in batch])                               # B,T,2
    targets = [[f["target"] for f in s["frames"]] for s in batch]                      # B,T,dict
    return {"imgs": imgs, "motions": motions, "targets": targets,
            "seq_id": [s["seq_id"] for s in batch]}

def collate_frames(batch):
    imgs = torch.stack([b["img"] for b in batch])
    targets = [b["target"] for b in batch]
    return {"imgs": imgs, "targets": targets}
