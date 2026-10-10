# -*- coding: utf-8 -*-
"""
损失与评价: 匈牙利匹配 / Focal-EIoU / GIoU / 集预测损失 / 编码器辅助损失 / mAP 评估
===================================================================================
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------- 框变换

def cxcywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)

def box_area(box):
    return (box[..., 2] - box[..., 0]).clamp(min=0) * (box[..., 3] - box[..., 1]).clamp(min=0)

def box_iou_matrix(b1, b2, eps=1e-7):
    """[N,4] x [M,4] xyxy -> [N,M]."""
    lt = torch.max(b1[:, None, :2], b2[None, :, :2])
    rb = torch.min(b1[:, None, 2:], b2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = (box_area(b1)[:, None] + box_area(b2)[None, :] - inter).clamp(min=eps)
    return inter / union

# ---------------------------------------------------------------- IoU 损失

def giou_loss(pred_xyxy, tgt_xyxy, eps=1e-7):
    """标准 GIoU loss (1 - GIoU), 返回逐对损失. fp32."""
    pred_xyxy = pred_xyxy.float()
    tgt_xyxy = tgt_xyxy.float()
    lt = torch.max(pred_xyxy[:, :2], tgt_xyxy[:, :2])
    rb = torch.min(pred_xyxy[:, 2:], tgt_xyxy[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = (box_area(pred_xyxy) + box_area(tgt_xyxy) - inter).clamp(min=eps)
    iou = inter / union
    enclose_lt = torch.min(pred_xyxy[:, :2], tgt_xyxy[:, :2])
    enclose_rb = torch.max(pred_xyxy[:, 2:], tgt_xyxy[:, 2:])
    enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
    enclose_area = (enclose_wh[:, 0] * enclose_wh[:, 1]).clamp(min=eps)
    giou = iou - (enclose_area - union) / enclose_area
    return 1 - giou

def eiou_loss(pred_xyxy, tgt_xyxy, eps=1e-7):
    """EIoU loss: 1 - IoU + 中心距惩罚 + 宽高惩罚 (arXiv 2101.08158). fp32."""
    pred_xyxy = pred_xyxy.float()
    tgt_xyxy = tgt_xyxy.float()
    lt = torch.max(pred_xyxy[:, :2], tgt_xyxy[:, :2])
    rb = torch.min(pred_xyxy[:, 2:], tgt_xyxy[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = (box_area(pred_xyxy) + box_area(tgt_xyxy) - inter).clamp(min=eps)
    iou = inter / union
    # 中心距离 / 外接框对角线^2
    c_pred = (pred_xyxy[:, :2] + pred_xyxy[:, 2:]) / 2
    c_tgt = (tgt_xyxy[:, :2] + tgt_xyxy[:, 2:]) / 2
    rho2 = ((c_pred - c_tgt) ** 2).sum(-1)
    enclose_lt = torch.min(pred_xyxy[:, :2], tgt_xyxy[:, :2])
    enclose_rb = torch.max(pred_xyxy[:, 2:], tgt_xyxy[:, 2:])
    c2 = ((enclose_rb - enclose_lt) ** 2).sum(-1).clamp(min=eps)
    # 宽高差 / 外接框宽高^2
    wp, hp = pred_xyxy[:, 2] - pred_xyxy[:, 0], pred_xyxy[:, 3] - pred_xyxy[:, 1]
    wt, ht = tgt_xyxy[:, 2] - tgt_xyxy[:, 0], tgt_xyxy[:, 3] - tgt_xyxy[:, 1]
    cw = torch.max(wp, wt).clamp(min=eps) ** 2
    ch = torch.max(hp, ht).clamp(min=eps) ** 2
    l = 1 - iou + rho2 / c2 + (wp - wt) ** 2 / cw + (hp - ht) ** 2 / ch
    return l, iou

def focal_eiou(pred_xyxy, tgt_xyxy, gamma=0.5):
    """Focal-EIoU: IoU^γ * L_EIoU (γ=0.5, 同原文; 权重项 detach 作为纯重加权)."""
    l, iou = eiou_loss(pred_xyxy, tgt_xyxy)
    return (iou.detach().clamp(min=0) ** gamma) * l

def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """逐元素 focal loss (sigmoid 版). 强制 fp32 防止 AMP 数值溢出."""
    logits = logits.float()
    targets = targets.float()
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return a_t * loss

# ---------------------------------------------------------------- 匈牙利匹配

class HungarianMatcher(nn.Module):
    """集预测一对一匹配: 代价 = 2*分类 + 5*L1 + 2*IoU."""

    def __init__(self, cost_class=2.0, cost_l1=5.0, cost_iou=2.0):
        super().__init__()
        self.w_cls, self.w_l1, self.w_iou = cost_class, cost_l1, cost_iou

    @torch.no_grad()
    def forward(self, logits, boxes, targets):
        """logits [B,Q,C]; boxes [B,Q,4] cxcywh 归一化; targets: list[{labels,boxes(cxcywh归一化)}]."""
        B, Q, C = logits.shape
        logits = logits.float()
        boxes = boxes.float()
        probs = logits.sigmoid()
        results = []
        for b in range(B):
            tgt_l, tgt_b = targets[b]["labels"], targets[b]["boxes"]
            if tgt_l.numel() == 0:
                e = torch.empty(0, dtype=torch.long)
                results.append((e, e))
                continue
            p_b = boxes[b]
            cost_cls = -probs[b][:, tgt_l]                                    # Q,G
            pb_xyxy = cxcywh_to_xyxy(p_b)
            tb_xyxy = cxcywh_to_xyxy(tgt_b)
            cost_l1 = torch.cdist(pb_xyxy, tb_xyxy, p=1)
            iou = box_iou_matrix(pb_xyxy, tb_xyxy)
            cost = self.w_cls * cost_cls + self.w_l1 * cost_l1 + self.w_iou * (-iou)
            src, dst = linear_sum_assignment(cost.detach().cpu().numpy())
            results.append((torch.as_tensor(src, dtype=torch.long),
                            torch.as_tensor(dst, dtype=torch.long)))
        return results

# ---------------------------------------------------------------- 集预测损失

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, loss_type="focal_eiou",
                 w_cls=2.0, w_l1=5.0, w_box=2.0, aux_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_type = loss_type
        self.w_cls, self.w_l1, self.w_box = w_cls, w_l1, w_box
        self.aux_weight = aux_weight

    def _loss_single(self, logits, boxes, targets, log=None, prefix=""):
        """logits [B,Q,C], boxes [B,Q,4] cxcywh 归一化. 返回 (losses, matched_pairs)."""
        B, Q, C = logits.shape
        indices = self.matcher(logits.detach(), boxes.detach(), targets)
        num_pos = sum(len(i[0]) for i in indices)
        dev = logits.device

        # ----- 分类: 一对一二值 focal -----
        tgt_cls = torch.zeros(B, Q, C, device=dev)
        for b, (src, dst) in enumerate(indices):
            if len(src):
                tgt_cls[b, src, targets[b]["labels"][dst]] = 1.0
        l_cls_map = sigmoid_focal_loss(logits, tgt_cls)
        l_cls = l_cls_map.sum() / max(num_pos, 1)

        # ----- 框回归 -----
        pairs_pred, pairs_tgt = [], []
        for b, (src, dst) in enumerate(indices):
            if len(src):
                pairs_pred.append(boxes[b][src])
                pairs_tgt.append(targets[b]["boxes"][dst])
        if pairs_pred:
            pp = cxcywh_to_xyxy(torch.cat(pairs_pred))
            tt = cxcywh_to_xyxy(torch.cat(pairs_tgt))
            l_l1 = F.l1_loss(pp, tt, reduction="sum") / max(num_pos, 1)
            if self.loss_type == "focal_eiou":
                l_box = focal_eiou(pp, tt).sum() / max(num_pos, 1)
            else:
                l_box = giou_loss(pp, tt).sum() / max(num_pos, 1)
        else:
            l_l1 = logits.sum() * 0.0
            l_box = logits.sum() * 0.0

        losses = {"loss_cls": self.w_cls * l_cls,
                  "loss_l1": self.w_l1 * l_l1,
                  "loss_box": self.w_box * l_box}
        if log is not None:
            log[prefix + "loss_cls"] = float(l_cls)
            log[prefix + "loss_l1"] = float(l_l1)
            log[prefix + "loss_box"] = float(l_box)
        return losses, indices, num_pos

    def forward(self, outputs, targets, return_log=False):
        log = {} if return_log else None
        losses, _, num_pos = self._loss_single(outputs["pred_logits"], outputs["pred_boxes"],
                                               targets, log, prefix="")
        for i, aux in enumerate(outputs.get("aux_outputs", [])):
            al, _, _ = self._loss_single(aux["logits"], aux["boxes"], targets, log,
                                         prefix=f"aux{i}.")
            for k, v in al.items():
                losses[k] = losses[k] + self.aux_weight * v
        # ----- 编码器辅助损失 (监督 enc_score / enc_bbox, 使 top-K 查询选择有效) -----
        l_enc = self._enc_aux_loss(outputs.get("enc_scores"), outputs.get("enc_boxes"), targets)
        if l_enc is not None:
            losses["loss_enc"] = l_enc
        # ----- DN 去噪损失 (DN-DETR 风格) -----
        if "dn" in outputs:
            for k, v in self._dn_loss(outputs["dn"]).items():
                losses[k] = losses.get(k, 0.0) + v
        if return_log:
            return losses, log, num_pos
        return losses

    def _dn_loss(self, dn):
        """DN 查询损失: 正组(focal cls 到真标签 + 框回归到干净 GT) + 负组(背景)."""
        logits, boxes = dn["logits"], dn["boxes"]
        tgt, valid, pos = dn["tgt_labels"], dn["valid"], dn["pos"]
        clean = dn["clean"]
        num_pos = int(pos.sum().item())
        onehot = torch.zeros_like(logits)
        if num_pos:
            bi, ii = pos.nonzero(as_tuple=True)
            onehot[bi, ii, tgt[bi, ii]] = 1.0
        l_cls = (sigmoid_focal_loss(logits, onehot) * valid.unsqueeze(-1)).sum() / max(num_pos, 1)
        if num_pos:
            pp = cxcywh_to_xyxy(boxes[pos])
            tt = cxcywh_to_xyxy(clean[pos])
            l_l1 = F.l1_loss(pp, tt, reduction="sum") / max(num_pos, 1)
            if self.loss_type == "focal_eiou":
                l_box = focal_eiou(pp, tt).sum() / max(num_pos, 1)
            else:
                l_box = giou_loss(pp, tt).sum() / max(num_pos, 1)
        else:
            l_l1 = logits.sum() * 0.0
            l_box = logits.sum() * 0.0
        return {"loss_dn_cls": self.w_cls * l_cls,
                "loss_dn_l1": self.w_l1 * l_l1,
                "loss_dn_box": self.w_box * l_box}

    def _enc_aux_loss(self, enc_scores, enc_boxes, targets):
        """对 top-K 编码器预测做软标签监督 (仿 RT-DETR): 分类目标=最大IoU, 框回归到最优GT."""
        if enc_scores is None or enc_boxes is None:
            return None
        K = 100
        B, N = enc_scores.shape
        total_cls, total_box, num_gts = 0.0, 0.0, 0
        for b in range(B):
            tgt_b = targets[b]["boxes"]
            if tgt_b.numel() == 0:
                continue
            num_gts += 1
            topk = enc_scores[b].topk(min(K, N)).indices
            s = enc_scores[b][topk]
            pb = cxcywh_to_xyxy(enc_boxes[b][topk])
            tb = cxcywh_to_xyxy(tgt_b)
            iou = box_iou_matrix(pb, tb)                       # K,G
            max_iou, best = iou.max(dim=1)                     # K
            # 软分类目标 = max_iou (VFL 风格, 用 focal 近似)
            total_cls = total_cls + sigmoid_focal_loss(s, max_iou.detach()).sum()
            # 框回归 (仅对 iou>0.2 的样本)
            sel = max_iou > 0.2
            if sel.any():
                pp = pb[sel]
                tt = tb[best[sel]]
                if self.loss_type == "focal_eiou":
                    total_box = total_box + focal_eiou(pp, tt).sum()
                else:
                    total_box = total_box + giou_loss(pp, tt).sum()
        if num_gts == 0:
            return None
        return (total_cls + self.w_box * total_box / 2.0) / (num_gts * 2.0)

# ---------------------------------------------------------------- 推理后处理

@torch.no_grad()
def postprocess(outputs, img_w, img_h, score_thr=0.05, topk=100, nms=False, nms_iou=0.6):
    """logits [B,Q,C] + boxes [B,Q,4](cxcywh 归一化) -> list[{scores,labels,boxes(xyxy px)}]."""
    logits = outputs["pred_logits"]
    boxes = outputs["pred_boxes"]
    B, Q, C = logits.shape
    probs = logits.sigmoid()
    scores, labels = probs.max(dim=-1)                        # B,Q
    boxes_px = cxcywh_to_xyxy(boxes) * torch.tensor([img_w, img_h, img_w, img_h],
                                                    dtype=boxes.dtype, device=boxes.device)
    out = []
    for b in range(B):
        s, l, bb = scores[b], labels[b], boxes_px[b]
        keep = s > score_thr
        s, l, bb = s[keep], l[keep], bb[keep]
        if s.numel() > topk:
            t = s.topk(topk).indices
            s, l, bb = s[t], l[t], bb[t]
        if nms and s.numel():
            from torchvision.ops import nms as tv_nms
            offsets = l.float().unsqueeze(1) * (img_w * img_h + 1)
            idx = tv_nms(bb + offsets, s, nms_iou)
            s, l, bb = s[idx], l[idx], bb[idx]
        out.append({"scores": s, "labels": l, "boxes": bb})
    return out

# ---------------------------------------------------------------- mAP 评估

def compute_map(preds, gts, iou_thresholds=None, area_ranges=None, num_classes=5):
    """
    COCO 风格 mAP (101 点插值).
    preds: list[dict(scores, labels, boxes xyxy px)]
    gts:   list[dict(labels, boxes xyxy px)]
    返回 dict: mAP50, mAP50_95, 每类 AP50, AP_small 等
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 0.96, 0.05)
    if area_ranges is None:
        area_ranges = {"all": (0, np.inf), "small": (0, 32 ** 2), "medium": (32 ** 2, 96 ** 2),
                       "large": (96 ** 2, np.inf)}
    rec_thrs = np.linspace(0, 1, 101)

    results = {}
    for area_name, (amin, amax) in area_ranges.items():
        ap_per_iou = []
        for it in iou_thresholds:
            ap_cls = []
            for c in range(num_classes):
                # 汇总全部图像的预测与GT
                dt_scores, dt_match = [], []
                npos = 0
                for pi, (p, g) in enumerate(zip(preds, gts)):
                    gm = (g["labels"] == c)
                    gb = g["boxes"][gm]
                    g_area = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1]) if gb.numel() else gb
                    if gb.numel():
                        valid = (g_area >= amin) & (g_area < amax)
                    else:
                        valid = torch.zeros(0, dtype=torch.bool)
                    gb_valid = gb[valid]
                    npos += gb_valid.shape[0]
                    pm = (p["labels"] == c)
                    pb = p["boxes"][pm]
                    ps = p["scores"][pm]
                    if pb.numel():
                        p_area = (pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1])
                        pvalid = (p_area >= amin) & (p_area < amax)
                        pb, ps = pb[pvalid], ps[pvalid]
                    used = np.zeros(gb_valid.shape[0], dtype=bool)
                    order = torch.argsort(ps, descending=True)
                    for j in order.tolist():
                        if gb_valid.shape[0] == 0:
                            dt_scores.append(float(ps[j]))
                            dt_match.append(0)
                            continue
                        ious = box_iou_matrix(pb[j:j + 1].cpu(), gb_valid.cpu())[0].numpy()
                        best = int(np.argmax(ious)) if len(ious) else -1
                        if best >= 0 and ious[best] >= it and not used[best]:
                            used[best] = True
                            dt_scores.append(float(ps[j]))
                            dt_match.append(1)
                        else:
                            dt_scores.append(float(ps[j]))
                            dt_match.append(0)
                if npos == 0:
                    ap_cls.append(np.nan)
                    continue
                if len(dt_scores) == 0:
                    ap_cls.append(0.0)
                    continue
                order = np.argsort(-np.array(dt_scores))
                dm = np.array(dt_match)[order]
                tp = np.cumsum(dm)
                fp = np.cumsum(1 - dm)
                rc = tp / npos
                pr = tp / np.maximum(tp + fp, 1e-9)
                ap = 0.0
                for r in rec_thrs:
                    pr_at = pr[rc >= r]
                    ap += (pr_at.max() if pr_at.size else 0.0)
                ap_cls.append(ap / len(rec_thrs))
            ap_per_iou.append(np.nanmean(ap_cls) if len(ap_cls) else np.nan)
        results[f"mAP@{area_name}"] = float(np.nanmean(ap_per_iou)) if ap_per_iou else 0.0
        if area_name == "all":
            results["mAP50"] = float(np.nanmean([ap_per_iou[0]]))
            results["mAP50_95"] = float(np.nanmean(ap_per_iou))
    # 每类 AP50 (仅 all 区间)
    per_class = []
    for c in range(num_classes):
        dt_scores, dt_match, npos = [], [], 0
        for p, g in zip(preds, gts):
            gb = g["boxes"][g["labels"] == c]
            npos += gb.shape[0]
            pm = p["labels"] == c
            pb, ps = p["boxes"][pm], p["scores"][pm]
            used = np.zeros(gb.shape[0], dtype=bool)
            order = torch.argsort(ps, descending=True)
            for j in order.tolist():
                if gb.shape[0] == 0:
                    dt_scores.append(float(ps[j])); dt_match.append(0); continue
                ious = box_iou_matrix(pb[j:j + 1].cpu(), gb.cpu())[0].numpy()
                best = int(np.argmax(ious))
                if ious[best] >= 0.5 and not used[best]:
                    used[best] = True
                    dt_scores.append(float(ps[j])); dt_match.append(1)
                else:
                    dt_scores.append(float(ps[j])); dt_match.append(0)
        if npos == 0 or len(dt_scores) == 0:
            per_class.append(np.nan)
            continue
        order = np.argsort(-np.array(dt_scores))
        dm = np.array(dt_match)[order]
        tp = np.cumsum(dm); fp = np.cumsum(1 - dm)
        rc = tp / npos; pr = tp / np.maximum(tp + fp, 1e-9)
        ap = sum((pr[rc >= r].max() if (rc >= r).any() else 0.0) for r in rec_thrs) / len(rec_thrs)
        per_class.append(float(ap))
    results["AP50_per_class"] = per_class
    return results
