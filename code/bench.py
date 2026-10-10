# -*- coding: utf-8 -*-
"""训练吞吐/显存基准 (fp32, 文件形式运行以支持 DataLoader 多进程)."""
import time

import torch
from torch.utils.data import DataLoader

from dataset import DefectDataset, collate_frames
from model import FocusDETRLite
from ops import HungarianMatcher, SetCriterion
from train import to_dev_targets


def bench(bs, workers, n_iter=25):
    dev = "cuda"
    torch.manual_seed(0)
    ds = DefectDataset("../data/defect_ds", "train", augment=True, seed=0)
    tl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=workers,
                    collate_fn=collate_frames, drop_last=True, pin_memory=True)
    model = FocusDETRLite(num_classes=5, variant="focus").to(dev)
    crit = SetCriterion(5, HungarianMatcher(), loss_type="focal_eiou").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    it = iter(tl)
    t0 = time.time()
    n = 0
    for batch in it:
        imgs = batch["imgs"].to(dev, non_blocking=True)
        targets = to_dev_targets(batch["targets"], dev)
        out = model(imgs, targets=targets)
        losses = crit(out, targets)
        total = sum(losses.values())
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        opt.step()
        n += 1
        if n >= n_iter:
            break
    torch.cuda.synchronize()
    dt = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 2 ** 30
    per_epoch = 4296 / bs
    print(f"bs={bs} workers={workers}: {dt/n*1000:.0f} ms/iter | 每epoch({per_epoch:.0f} it) "
          f"≈ {dt/n*per_epoch/60:.1f} min | 显存峰值 {mem:.2f} GB", flush=True)


if __name__ == "__main__":
    for bs in [8, 10]:
        for w in [2]:
            try:
                bench(bs, w)
            except RuntimeError as e:
                print(f"bs={bs} 失败: {str(e)[:80]}")
