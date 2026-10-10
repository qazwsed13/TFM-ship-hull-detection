# -*- coding: utf-8 -*-
"""
训练脚本:
  stage1: 单帧集预测训练 (variant: base / focus)
  stage2: 时序焦点记忆微调 (variant: focus_t / base_t, 由对应无记忆权重初始化, 冻结骨干+编码器)
用法:
  python train.py --stage 1 --variant focus --out ../runs/focus_s1
  python train.py --stage 2 --variant focus_t --init ../runs/focus_s1/best.pt --out ../runs/focus_t_s2
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import DefectDataset, VideoClipDataset, collate_frames, collate_seqs
from model import VARIANTS, FocusDETRLite, count_params
from ops import HungarianMatcher, SetCriterion, compute_map, postprocess

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "defect_ds"


def to_dev_targets(targets, dev):
    return [{k: t[k].to(dev) for k in ("labels", "boxes")} for t in targets]


@torch.no_grad()
def evaluate_frame_map(model, loader, dev, num_classes=5, max_batches=None, score_thr=0.05):
    model.eval()
    preds, gts = [], []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        imgs = batch["imgs"].to(dev)
        out = model(imgs)
        pp = postprocess(out, imgs.shape[-1], imgs.shape[-2], score_thr=score_thr)
        for b, t in enumerate(batch["targets"]):
            preds.append({"scores": pp[b]["scores"].cpu(), "labels": pp[b]["labels"].cpu(),
                          "boxes": pp[b]["boxes"].cpu()})
            W, H = imgs.shape[-1], imgs.shape[-2]
            tb = t["boxes"]
            x0 = (tb[:, 0] - tb[:, 2] / 2) * W
            y0 = (tb[:, 1] - tb[:, 3] / 2) * H
            x1 = (tb[:, 0] + tb[:, 2] / 2) * W
            y1 = (tb[:, 1] + tb[:, 3] / 2) * H
            gts.append({"labels": t["labels"], "boxes": torch.stack([x0, y0, x1, y1], -1)})
    m = compute_map(preds, gts, num_classes=num_classes)
    model.train()
    return m


def build_optimizer(model, lr, wd=1e-4):
    backbone, head = model.get_param_groups()
    return torch.optim.AdamW([{"params": backbone, "lr": lr * 0.1},
                              {"params": head, "lr": lr}], weight_decay=wd)


def lr_scale(step, total, warmup, floor=0.0):
    if step < warmup:
        return step / max(1, warmup) * (1 - floor) + floor
    p = (step - warmup) / max(1, total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * p))


def save_ckpt(path, model, args, epoch, best):
    torch.save({"model": model.state_dict(), "variant": args.variant, "epoch": epoch,
                "best": best, "args": vars(args)}, path)


def train_stage1(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = FocusDETRLite(num_classes=5, variant=args.variant).to(dev)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"[stage1] 续训加载 {args.init}: missing={len(missing)} "
              f"unexpected={len(unexpected)} (之前best={ck.get('best')})", flush=True)
    n_param = count_params(model)
    print(f"[stage1] variant={args.variant} params={n_param/1e6:.2f}M", flush=True)
    train_ds = DefectDataset(DATA, "train", augment=True, limit=args.limit, seed=args.seed)
    val_ds = DefectDataset(DATA, "val", augment=False, limit=args.limit or 200)
    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    collate_fn=collate_frames, drop_last=True, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                    collate_fn=collate_frames, pin_memory=True)
    crit = SetCriterion(5, HungarianMatcher(), loss_type=VARIANTS[args.variant]["loss"]).to(dev)
    opt = build_optimizer(model, args.lr)
    amp_on = (dev.type == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)
    total_steps = max(1, len(tl)) * args.epochs
    warmup = min(args.warmup, total_steps // 10 + 1)

    log_f = open(out_dir / "train_log.csv", "w", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    log.writerow(["epoch", "iter", "lr", "loss", "loss_cls", "loss_l1", "loss_box", "loss_enc",
                  "loss_dn"])
    best = -1.0
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        ep_loss, n_it = 0.0, 0
        for it, batch in enumerate(tl):
            imgs = batch["imgs"].to(dev, non_blocking=True)
            targets = to_dev_targets(batch["targets"], dev)
            lr_now = args.lr * lr_scale(step, total_steps, warmup, args.min_lr_ratio)
            for g in opt.param_groups:
                g["lr"] = lr_now * (0.1 if g is opt.param_groups[0] else 1.0)
            with torch.amp.autocast("cuda", enabled=amp_on):
                out = model(imgs, targets=targets)
                losses = crit(out, targets)
                total = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(opt)
            scaler.update()
            step += 1
            ep_loss += float(total)
            n_it += 1
            if (it + 1) % 50 == 0:
                dn = sum(float(v) for k, v in losses.items() if "dn" in k)
                log.writerow([epoch, it, f"{lr_now:.2e}", f"{float(total):.4f}",
                              f"{float(losses['loss_cls']):.4f}", f"{float(losses['loss_l1']):.4f}",
                              f"{float(losses['loss_box']):.4f}",
                              f"{float(losses.get('loss_enc', torch.tensor(0))):.4f}",
                              f"{dn:.4f}"])
                log_f.flush()
                print(f"[ep{epoch} it{it}] loss={float(total):.4f} "
                      f"cls={float(losses['loss_cls']):.3f} box={float(losses['loss_box']):.3f} "
                      f"dn={dn:.2f} ({time.time()-t0:.0f}s)", flush=True)
        dn_sum = sum(float(v) for k, v in losses.items() if "dn" in k)
        log.writerow([epoch, "epoch_avg", "", f"{ep_loss/max(n_it,1):.4f}",
                      f"{float(losses['loss_cls']):.4f}", f"{float(losses['loss_l1']):.4f}",
                      f"{float(losses['loss_box']):.4f}",
                      f"{float(losses.get('loss_enc', torch.tensor(0))):.4f}", f"{dn_sum:.4f}"])
        log_f.flush()
        print(f"[epoch {epoch} done] avg_loss={ep_loss/max(n_it,1):.4f} "
              f"(cls={float(losses['loss_cls']):.2f} l1={float(losses['loss_l1']):.2f} "
              f"box={float(losses['loss_box']):.2f} dn={dn_sum:.2f}) ({time.time()-t0:.0f}s)",
              flush=True)
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            m = evaluate_frame_map(model, vl, dev, max_batches=args.val_batches)
            print(f"[eval ep{epoch}] mAP50={m['mAP50']:.4f} mAP50-95={m['mAP50_95']:.4f} "
                  f"AP_S={m['mAP@small']:.4f}", flush=True)
            if m["mAP50"] > best:
                best = m["mAP50"]
                save_ckpt(out_dir / "best.pt", model, args, epoch, best)
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            save_ckpt(out_dir / "last.pt", model, args, epoch, best)
    log_f.close()
    print(f"[stage1 完成] best mAP50={best:.4f} -> {out_dir}", flush=True)


def train_stage2(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = FocusDETRLite(num_classes=5, variant=args.variant,
                          mem_tokens=args.mem_tokens, mem_lifespan=args.mem_lifespan).to(dev)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"[stage2] 加载 {args.init}: missing={len(missing)} unexpected={len(unexpected)}",
              flush=True)
    # 冻结骨干 + 编码器 + 查询选择头
    for m in [model.backbone, model.encoder, model.enc_score, model.enc_bbox]:
        for p in m.parameters():
            p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[stage2] 可训练参数: {sum(p.numel() for p in trainable)/1e6:.2f}M / "
          f"总参数 {count_params(model)/1e6:.2f}M", flush=True)

    train_ds = VideoClipDataset(DATA, "train", limit=args.limit)
    val_ds = DefectDataset(DATA, "val", augment=False, limit=args.limit or 150)
    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    collate_fn=collate_seqs)
    vl = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=args.workers,
                    collate_fn=collate_frames)
    crit = SetCriterion(5, HungarianMatcher(), loss_type=VARIANTS[args.variant]["loss"]).to(dev)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    amp_on = (dev.type == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)

    log_f = open(out_dir / "train_log.csv", "w", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    log.writerow(["epoch", "seq_step", "t", "loss"])
    best = -1.0
    t0 = time.time()
    for epoch in range(args.epochs):
        ep_loss, n_it = 0.0, 0
        for bi, batch in enumerate(tl):
            imgs_all = batch["imgs"].to(dev)                 # B,T,3,H,W
            motions = batch["motions"]                       # B,T,2
            B, T = imgs_all.shape[0], imgs_all.shape[1]
            memory = None
            seq_loss = 0.0
            for t in range(T):
                imgs = imgs_all[:, t]
                targets = to_dev_targets([batch["targets"][b][t] for b in range(B)], dev)
                motion = None
                # 记忆随机丢弃 (p=mem_dropout): 保留单帧检测能力, 防止 stage2 漂移
                use_mem = (t > 0) and (np.random.rand() > args.mem_dropout)
                if use_mem:
                    dx = float(motions[:, t, 0].mean())
                    dy = float(motions[:, t, 1].mean())
                    motion = (dx, dy)
                with torch.amp.autocast("cuda", enabled=amp_on):
                    out = model(imgs, memory_state=(memory if use_mem else None),
                                motion=motion, return_memory=True, targets=targets)
                    losses = crit(out, targets)
                    total = sum(losses.values())
                opt.zero_grad(set_to_none=True)
                scaler.scale(total).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, 0.1)
                scaler.step(opt)
                scaler.update()
                seq_loss += float(total)
                # 帧末更新记忆 (无梯度)
                probs = out["pred_logits"].sigmoid()
                scores, labels = probs.max(dim=-1)
                memory = model.update_memory(memory, out["_mem_raw"], imgs.shape[-1],
                                             imgs.shape[-2], scores, labels)
            log.writerow([epoch, bi, T, f"{seq_loss/T:.4f}"])
            log_f.flush()
            ep_loss += seq_loss / T
            n_it += 1
            if (bi + 1) % 10 == 0:
                print(f"[ep{epoch} seq{bi}] avg_frame_loss={seq_loss/T:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        print(f"[stage2 epoch {epoch} done] avg_loss={ep_loss/max(n_it,1):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            m = evaluate_frame_map(model, vl, dev, max_batches=args.val_batches)
            print(f"[eval ep{epoch}] 单帧模式 mAP50={m['mAP50']:.4f}", flush=True)
            # 每个评估点单独存档 (时序质量以最终视频评估为准选优)
            save_ckpt(out_dir / f"ep{epoch}.pt", model, args, epoch, m["mAP50"])
            if m["mAP50"] > best:
                best = m["mAP50"]
                save_ckpt(out_dir / "best.pt", model, args, epoch, best)
        save_ckpt(out_dir / "last.pt", model, args, epoch, best)
    log_f.close()
    print(f"[stage2 完成] best 单帧 mAP50={best:.4f} -> {out_dir}", flush=True)


def train_stage3(args):
    """端到端强化 (方向2 证据升级): 从 stage1 强检测器出发, 全网络可训练 + 记忆激活,
    静帧与视频序列混合训练——检验能否消除"检测保持 vs 记忆增益"的两阶段权衡."""
    from itertools import cycle
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = FocusDETRLite(num_classes=5, variant=args.variant,
                          mem_tokens=args.mem_tokens, mem_lifespan=args.mem_lifespan).to(dev)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"[stage3] 加载 {args.init}: missing={len(missing)} unexpected={len(unexpected)}",
              flush=True)
    n_all = count_params(model)
    still_ds = DefectDataset(DATA, "train", augment=True, seed=args.seed)
    seq_ds = VideoClipDataset(DATA, "train")
    still_loader = DataLoader(still_ds, batch_size=args.batch_still, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_frames,
                              drop_last=True, pin_memory=True)
    seq_loader = DataLoader(seq_ds, batch_size=args.batch_seq, shuffle=True,
                            num_workers=args.workers, collate_fn=collate_seqs)
    crit = SetCriterion(5, HungarianMatcher(), loss_type=VARIANTS[args.variant]["loss"]).to(dev)
    bb, head = model.get_param_groups()
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr * 0.1},
                             {"params": head, "lr": args.lr}], weight_decay=1e-4)
    amp_on = (dev.type == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)
    total_steps = args.steps_per_epoch * args.epochs
    warmup = min(args.warmup, total_steps // 10 + 1)
    val_ds = DefectDataset(DATA, "val", augment=False, limit=args.val_batches * args.batch_still)
    vl = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=args.workers,
                    collate_fn=collate_frames)

    log_f = open(out_dir / "train_log.csv", "w", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    log.writerow(["epoch", "step", "kind", "loss", "lr"])
    best = -1.0
    gstep = 0
    t0 = time.time()

    def train_still(batch):
        imgs = batch["imgs"].to(dev, non_blocking=True)
        targets = to_dev_targets(batch["targets"], dev)
        with torch.amp.autocast("cuda", enabled=amp_on):
            out = model(imgs, targets=targets)
            losses = crit(out, targets)
            total = sum(losses.values())
        return total

    def train_seq(batch):
        imgs_all = batch["imgs"].to(dev)
        motions = batch["motions"]
        B, T = imgs_all.shape[0], imgs_all.shape[1]
        L = min(args.clip_len, T)
        t0i = int(np.random.randint(0, T - L + 1))          # 随机窗口起点
        memory = None
        seq_loss = 0.0
        for t in range(t0i, t0i + L):
            imgs = imgs_all[:, t]
            targets = to_dev_targets([batch["targets"][b][t] for b in range(B)], dev)
            motion = None
            use_mem = (t > t0i) and (np.random.rand() > args.mem_dropout)
            if use_mem:
                motion = (float(motions[:, t, 0].mean()), float(motions[:, t, 1].mean()))
            with torch.amp.autocast("cuda", enabled=amp_on):
                out = model(imgs, memory_state=(memory if use_mem else None),
                            motion=motion, return_memory=True, targets=targets)
                out_l = out
                if args.mem_unsup and use_mem and out["n_main"] > model.n_queries:
                    nm = model.n_queries
                    out_l = {**out,
                             "pred_logits": out["pred_logits"][:, :nm],
                             "pred_boxes": out["pred_boxes"][:, :nm],
                             "aux_outputs": [{k: v[:, :nm] for k, v in a.items()}
                                             for a in out.get("aux_outputs", [])]}
                losses = crit(out_l, targets)
                total = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(opt)
            scaler.update()
            seq_loss += float(total)
            probs = out["pred_logits"].sigmoid()
            scores, labels = probs.max(dim=-1)
            memory = model.update_memory(memory, out["_mem_raw"], imgs.shape[-1],
                                         imgs.shape[-2], scores, labels)
        return seq_loss / L

    for epoch in range(args.epochs):
        si = iter(still_loader)
        sc = cycle(seq_loader)
        ep_loss, n_it, n_seq = 0.0, 0, 0
        for step in range(args.steps_per_epoch):
            lr_now = args.lr * lr_scale(gstep, total_steps, warmup, args.min_lr_ratio)
            for gi, g in enumerate(opt.param_groups):
                g["lr"] = lr_now * (0.1 if gi == 0 else 1.0)
            if step % args.seq_every == args.seq_every - 1:   # 序列步
                batch = next(sc)
                loss_v = train_seq(batch)
                n_seq += 1
                log.writerow([epoch, step, "seq", f"{loss_v:.4f}", f"{lr_now:.2e}"])
            else:                                             # 静帧步
                try:
                    batch = next(si)
                except StopIteration:
                    si = iter(still_loader)
                    batch = next(si)
                total = train_still(batch)
                opt.zero_grad(set_to_none=True)
                scaler.scale(total).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                scaler.step(opt)
                scaler.update()
                loss_v = float(total)
                log.writerow([epoch, step, "still", f"{loss_v:.4f}", f"{lr_now:.2e}"])
            gstep += 1
            ep_loss += loss_v
            n_it += 1
            if (step + 1) % 60 == 0:
                print(f"[ep{epoch} step{step}] loss={loss_v:.3f} (seq步{n_seq}) "
                      f"({time.time()-t0:.0f}s)", flush=True)
        log_f.flush()
        print(f"[stage3 epoch {epoch} done] avg_loss={ep_loss/max(n_it,1):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            m = evaluate_frame_map(model, vl, dev, max_batches=args.val_batches)
            print(f"[eval ep{epoch}] 单帧模式 mAP50={m['mAP50']:.4f} "
                  f"mAP50-95={m['mAP50_95']:.4f}", flush=True)
            save_ckpt(out_dir / f"ep{epoch}.pt", model, args, epoch, m["mAP50"])
            if m["mAP50"] > best:
                best = m["mAP50"]
                save_ckpt(out_dir / "best.pt", model, args, epoch, best)
        save_ckpt(out_dir / "last.pt", model, args, epoch, best)
    log_f.close()
    print(f"[stage3 完成] best 单帧 mAP50={best:.4f} -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--variant", type=str, default="focus")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--init", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="调试: 限制样本数")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--val_batches", type=int, default=40)
    ap.add_argument("--mem_tokens", type=int, default=48)
    ap.add_argument("--mem_lifespan", type=int, default=3)
    ap.add_argument("--mem_dropout", type=float, default=0.0, help="stage2 记忆随机丢弃比例")
    ap.add_argument("--mem_unsup", action="store_true",
                    help="消融: 记忆 token 不参与损失/匹配(仅作解码器上下文)")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--min_lr_ratio", type=float, default=0.0, help="余弦下限比例")
    # stage3 (端到端混合训练) 专用
    ap.add_argument("--batch_still", type=int, default=8)
    ap.add_argument("--batch_seq", type=int, default=4)
    ap.add_argument("--steps_per_epoch", type=int, default=240)
    ap.add_argument("--seq_every", type=int, default=5, help="每 N 步插一个序列步")
    ap.add_argument("--clip_len", type=int, default=8)
    args = ap.parse_args()
    assert args.variant in VARIANTS, f"未知 variant: {args.variant}"
    if args.stage == 1:
        train_stage1(args)
    elif args.stage == 2:
        train_stage2(args)
    else:
        train_stage3(args)


if __name__ == "__main__":
    main()
