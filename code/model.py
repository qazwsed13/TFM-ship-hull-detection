# -*- coding: utf-8 -*-
"""
FocusDETR-lite (复现版) + 时序焦点记忆 (Temporal Focus Memory)
================================================================
复现 Focus DETR (Neurocomputing 2025) 公开信息中的核心要素:
  1. 混合 CNN-ViT 架构: 轻量 CNN 骨干 + AIFI 单尺度自注意力 + CCFM 跨尺度卷积融合 (hybrid encoder)
  2. Sa-Gn 模块: 递归门控卷积 (gnConv, HorNet NeurIPS 2022) + 空间注意力 (CBAM-SAM), 置于骨干末端
  3. Focal-EIoU 损失 (EIoU: arXiv 2101.08158)
  4. 轻量化设计 (面向移动端/边缘部署)

创新模块 (方向2 主创新):
  TemporalFocusMemory (TFM): 跨帧"焦点 token"记忆
    - 帧末从解码器输出选取高置信检测作为焦点 token (特征+框+类别+年龄)
    - 帧间利用相对运动 (滑窗偏移真值/视觉里程计) 对齐, 注入下一帧解码器查询集
    - 记忆 token 参与集预测: 可恢复瞬时漏检/抑制跨帧闪烁

说明: 原文全文未公开获取, 本实现依据公开摘要/关键词/同组工作的"风格复现";
      通道数与结构细节为工程选择 (见 W0 研读报告的核实程度声明).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 基础工具
# =====================================================================

def pos_embed_2d(h, w, d, device, temperature=10000.0):
    """标准 DETR 2D 正弦位置编码, 返回 [h*w, d]."""
    y_embed = torch.arange(h, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, w)
    x_embed = torch.arange(w, dtype=torch.float32, device=device).unsqueeze(0).repeat(h, 1)
    dim_t = torch.arange(d // 2, dtype=torch.float32, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / (d // 2))
    pe_x = x_embed[:, :, None] / dim_t
    pe_y = y_embed[:, :, None] / dim_t
    pe = torch.zeros(h, w, d, device=device)
    pe[:, :, 0:d // 2:2] = pe_x[:, :, 0::2].sin()
    pe[:, :, 1:d // 2:2] = pe_x[:, :, 1::2].cos()
    pe[:, :, d // 2::2] = pe_y[:, :, 0::2].sin()
    pe[:, :, d // 2 + 1::2] = pe_y[:, :, 1::2].cos()
    return pe.view(h * w, d)

def sine_embed_boxes(boxes, dim, temperature=10000.0):
    """cxcywh (归一化) -> 正弦嵌入 [..., dim] (4 个坐标各 dim/4), dim 为 8 的倍数."""
    assert dim % 8 == 0
    d_each = dim // 4
    dim_t = torch.arange(d_each, dtype=torch.float32, device=boxes.device)
    dim_t = temperature ** (2 * (dim_t // 2) / d_each)
    out = []
    for i in range(4):
        e = boxes[..., i:i + 1] / dim_t
        out.append(torch.cat([e[..., 0::2].sin(), e[..., 1::2].cos()], dim=-1))
    return torch.cat(out, dim=-1)

def xyxy_px_to_cxcywh_norm(box, W, H):
    """像素 xyxy -> 归一化 cxcywh."""
    x0, y0, x1, y1 = box.unbind(-1)
    return torch.stack([(x0 + x1) / 2 / W, (y0 + y1) / 2 / H,
                        (x1 - x0) / W, (y1 - y0) / H], -1)

def cxcywh_norm_to_xyxy_px(box, W, H):
    """归一化 cxcywh -> 像素 xyxy."""
    cx, cy, w, h = box[..., 0] * W, box[..., 1] * H, box[..., 2] * W, box[..., 3] * H
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)

def bbox_iou_px(boxes1, boxes2):
    """像素 xyxy 框 IoU: [N,4] x [M,4] -> [N,M]."""
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    a1 = ((boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]))[:, None]
    a2 = ((boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]))[None, :]
    return inter / (a1 + a2 - inter + 1e-6)

# =====================================================================
# Sa-Gn: 递归门控卷积 (gnConv) + 空间注意力
# =====================================================================

class gnConv(nn.Module):
    """HorNet 递归门控卷积 (忠实复现核心结构)."""

    def __init__(self, dim, order=3, s=0.7071):
        super().__init__()
        self.order = order
        self.dims = [dim // (2 ** i) for i in range(order)]
        self.dims.reverse()                      # [dim/2^(o-1), ..., dim]
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)
        self.dwconv = nn.Conv2d(sum(self.dims), sum(self.dims), 7, 1, 3,
                                groups=sum(self.dims))
        self.pws = nn.ModuleList([nn.Conv2d(self.dims[i], self.dims[i + 1], 1)
                                  for i in range(order - 1)])
        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.scale = s
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        fused = self.proj_in(x)
        pwa, abc = torch.split(fused, (self.dims[0], sum(self.dims)), dim=1)
        dw = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw, self.dims, dim=1)
        y = pwa * dw_list[0]
        for i in range(self.order - 1):
            y = self.pws[i](y) * dw_list[i + 1]
        return self.proj_out(y)

class SpatialAttention(nn.Module):
    """CBAM 空间注意力 (7x7 conv on [mean,max])."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, 1, 3)

    def forward(self, x):
        m = torch.cat([x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True)[0]], dim=1)
        return x * torch.sigmoid(self.conv(m))

class SaGnBlock(nn.Module):
    """Sa-Gn 块: DWConv7x7 -> gnConv -> 空间注意力, LayerScale 残差."""

    def __init__(self, dim, order=3):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.gn = gnConv(dim, order=order)
        self.sam = SpatialAttention()
        self.gamma = nn.Parameter(torch.full((1,), 1e-4))

    def forward(self, x):
        y = self.sam(self.gn(self.dw(x)))
        return x + self.gamma * y

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or c_in != c_out:
            self.down = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                                      nn.BatchNorm2d(c_out))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(y + idt)

class ConvBNSiLU(nn.Module):
    def __init__(self, c_in, c_out, k=1, s=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def fuse_bn_conv(conv, bn):
    """将 Conv(无bias)+BN 折叠为等价的 1x1/3x3 卷积 (重参化演示)."""
    k = conv.weight.clone()
    bk = bn.running_var.clamp(min=1e-6).sqrt()
    scale = (bn.weight / bk)
    wk = k * scale.view(-1, 1, conv.weight.shape[2], conv.weight.shape[3])
    b = bn.bias - bn.running_mean * bn.weight / bk
    return wk, b

# =====================================================================
# 轻量骨干 (末端 Sa-Gn)
# =====================================================================

class ShipBackbone(nn.Module):
    """C3(s8,128), C4(s16,256), C5(s32,384)."""

    def __init__(self, use_sagn=False, c3=128, c4=256, c5=384):
        super().__init__()
        self.stem = nn.Sequential(ConvBNSiLU(3, 32, 3, 2))
        self.s1 = BasicBlock(32, 64, 2)                                        # s4
        self.s2 = nn.Sequential(BasicBlock(64, c3, 2), BasicBlock(c3, c3, 1))  # s8
        self.s3 = nn.Sequential(BasicBlock(c3, c4, 2), BasicBlock(c4, c4, 1))  # s16
        s4 = [BasicBlock(c4, c5, 2), BasicBlock(c5, c5, 1)]
        if use_sagn:
            s4[-1] = SaGnBlock(c5, order=3)
        self.s4 = nn.Sequential(*s4)                                           # s32

    def forward(self, x):
        x = self.s1(self.stem(x))
        c3 = self.s2(x)
        c4 = self.s3(c3)
        c5 = self.s4(c4)
        return c3, c4, c5

# =====================================================================
# 混合编码器: AIFI + CCFM
# =====================================================================

class ResConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = ConvBNSiLU(dim, dim, 3)
        self.pw = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim))

    def forward(self, x):
        return F.silu(x + self.pw(self.dw(x)))

class FusionBlock(nn.Module):
    """跨尺度融合: 尺寸对齐 (up=最近邻上采样 / down=最大池化) + 1x1 融合 + ResConvBlock."""

    def __init__(self, dim, scale="up"):
        super().__init__()
        self.scale = scale
        self.fuse = ConvBNSiLU(2 * dim, dim, 1)
        self.b1 = ResConvBlock(dim)

    def forward(self, x_high, x_low):
        if self.scale == "up":
            x_high = F.interpolate(x_high, size=x_low.shape[-2:], mode="nearest")
        else:
            x_high = F.max_pool2d(x_high, 2)
        return self.b1(self.fuse(torch.cat([x_high, x_low], dim=1)))

class AIFI(nn.Module):
    """Intra-scale 自注意力 (RT-DETR 风格), 作用于 C5."""

    def __init__(self, dim=256, n_heads=8, n_layers=1, dim_ffn=1024):
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, n_heads, dim_ffn, dropout=0.0,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x, pos):
        return self.enc(x + pos.unsqueeze(0))

class HybridEncoder(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        self.proj3 = ConvBNSiLU(128, dim, 1)
        self.proj4 = ConvBNSiLU(256, dim, 1)
        self.proj5 = ConvBNSiLU(384, dim, 1)
        self.aifi = AIFI(dim)
        self.fuse54 = FusionBlock(dim, scale="up")
        self.fuse43 = FusionBlock(dim, scale="up")
        self.fuse34 = FusionBlock(dim, scale="down")
        self.fuse45 = FusionBlock(dim, scale="down")

    def forward(self, feats):
        c3, c4, c5 = feats
        p3, p4, p5 = self.proj3(c3), self.proj4(c4), self.proj5(c5)
        B, C, h5, w5 = p5.shape
        pos5 = pos_embed_2d(h5, w5, C, p5.device)
        p5 = self.aifi(p5.flatten(2).transpose(1, 2).contiguous(), pos5)
        p5 = p5.transpose(1, 2).view(B, C, h5, w5).contiguous()
        p4 = self.fuse54(p5, p4)
        p3 = self.fuse43(p4, p3)
        p4 = self.fuse34(p3, p4)
        p5 = self.fuse45(p4, p5)
        return {"p3": p3, "p4": p4, "p5": p5}

# =====================================================================
# 多尺度可变形注意力
# =====================================================================

class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=4):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.level_embed = nn.Parameter(torch.zeros(n_levels, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.xavier_uniform_(self.sampling_offsets.weight)
        # 结构化采样偏移初始 (官方实现风格): 各头朝不同方向, 点数依次扩展
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= (i + 1)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.view(-1) * 0.5)

    def forward(self, query, reference_points, value_list, shapes):
        """query [B,Nq,C]; reference_points [B,Nq,L,2]∈[0,1]; value_list 多尺度 [B,h*w,C]."""
        B, Nq, C = query.shape
        L, H, P = self.n_levels, self.n_heads, self.n_points

        offsets = self.sampling_offsets(query).view(B, Nq, H, L, P, 2)         # B,Nq,H,L,P,2
        attn_w = self.attention_weights(query).view(B, Nq, H, L * P)
        attn_w = attn_w.softmax(dim=-1).view(B, Nq, H, L, P)                   # 在 (L,P) 上归一化

        out = 0
        hd = C // H
        for l, (h, w) in enumerate(shapes):
            v_l = self.value_proj(value_list[l]) + self.level_embed[l]         # B,h*w,C
            v_l = v_l.transpose(1, 2).view(B, C, h, w).contiguous()
            ref_l = reference_points[:, :, l]                                  # B,Nq,2
            off_l = offsets[:, :, :, l]                                        # B,Nq,H,P,2
            norm = torch.tensor([w, h], dtype=query.dtype, device=query.device)
            loc = ref_l[:, :, None, None, :] + off_l / norm.view(1, 1, 1, 1, 2)
            grid = (2.0 * loc - 1.0).view(B, Nq * H, P, 2)
            sampled = F.grid_sample(v_l, grid, mode="bilinear",
                                    padding_mode="border", align_corners=True)  # B,C,Nq*H,P
            # 拆分头: C=H*hd; i 维 = nq*H+head; diagonal 取 head 对角, 尾部附上 head 维
            sampled = sampled.view(B, H, hd, Nq, H, P).diagonal(dim1=1, dim2=4)  # B,hd,Nq,P,H
            sampled = sampled.permute(0, 2, 4, 3, 1)                             # B,Nq,H,P,hd
            out = out + (sampled * attn_w[:, :, :, l, :].unsqueeze(-1)).sum(dim=3)
        return self.output_proj(out.reshape(B, Nq, C))

# =====================================================================
# 解码器
# =====================================================================

def mha_with_bias(q, k, v, bias, n_heads, allow=None):
    """带加性偏置 + 可选可见性掩码的多头注意力. allow: [B,Lq,Lk] bool."""
    B, Lq, C = q.shape
    Lk = k.shape[1]
    hd = C // n_heads
    q = q.view(B, Lq, n_heads, hd).permute(0, 2, 1, 3)
    k = k.view(B, Lk, n_heads, hd).permute(0, 2, 1, 3)
    v = v.view(B, Lk, n_heads, hd).permute(0, 2, 1, 3)
    attn = q @ k.transpose(-2, -1) / math.sqrt(hd)
    if bias is not None:
        attn = attn + bias[:, :, :Lq, :Lk]
    if allow is not None:
        attn = attn.masked_fill(~allow[:, None, :Lq, :Lk], -1e4)
    attn = attn.softmax(dim=-1)
    return (attn @ v).permute(0, 2, 1, 3).reshape(B, Lq, C)

class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dim_ffn=1024, n_levels=3, n_points=4):
        super().__init__()
        self.n_heads = n_heads
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_ffn), nn.ReLU(inplace=True),
                                 nn.Linear(dim_ffn, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.bias_mlp = nn.Sequential(nn.Linear(d_model // 2, d_model // 2), nn.ReLU(inplace=True),
                                      nn.Linear(d_model // 2, n_heads))

    def forward(self, q, pos_q, ref_boxes, value_list, shapes, q_type=None, type_embed=None,
                allow=None):
        B, Nq, C = q.shape
        x = q + pos_q
        if q_type is not None and type_embed is not None:
            x = x + type_embed(q_type)
        bias = self.bias_mlp(sine_embed_boxes(ref_boxes, C // 2))     # B,Nq,H
        bias = bias.permute(0, 2, 1).unsqueeze(-1)                    # B,H,Nq,1
        bias = bias.expand(B, self.n_heads, Nq, Nq)
        attn = mha_with_bias(x, x, x, bias, self.n_heads, allow=allow)
        q = self.norm1(q + attn)
        ref_pts = ref_boxes[..., :2].unsqueeze(2).expand(-1, -1, len(shapes), -1)
        cross = self.cross_attn(q + pos_q, ref_pts, value_list, shapes)
        q = self.norm2(q + cross)
        q = self.norm3(q + self.ffn(q))
        return q

class Decoder(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dim_ffn=1024, n_levels=3, n_points=4, n_layers=3,
                 num_classes=5):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, dim_ffn, n_levels, n_points)
                                     for _ in range(n_layers)])
        self.box_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
                                      nn.Linear(d_model, 4))
        self.cls_head = nn.Linear(d_model, num_classes)
        self.type_embed = nn.Embedding(3, d_model)   # 0 内容查询 | 1 记忆 token | 2 DN 查询

    def forward(self, q, ref_boxes, value_list, shapes, q_type=None, allow=None):
        outputs = []
        cur = ref_boxes
        for layer in self.layers:
            pos_q = sine_embed_boxes(cur, q.shape[-1])
            q = layer(q, pos_q, cur, value_list, shapes, q_type, self.type_embed, allow)
            delta = self.box_head(q)
            ref_inv = torch.logit(cur.clamp(1e-4, 1 - 1e-4))
            cur = torch.sigmoid(ref_inv + delta)
            outputs.append({"boxes": cur, "logits": self.cls_head(q)})
        return outputs, q

# =====================================================================
# 时序焦点记忆 (创新模块)
# =====================================================================

class TemporalFocusMemory(nn.Module):
    """跨帧焦点 token 记忆 (特征+框+类别+年龄; 帧间平移对齐; 参与集预测)."""

    def __init__(self, d_model=256, max_tokens=48, lifespan=3, score_thr=0.3, iou_dedup=0.6):
        super().__init__()
        self.max_tokens = max_tokens
        self.lifespan = lifespan
        self.score_thr = score_thr
        self.iou_dedup = iou_dedup
        self.age_embed = nn.Embedding(lifespan + 2, d_model)
        self.mem_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
                                      nn.Linear(d_model, d_model))
        self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())

    @torch.no_grad()
    def select(self, feats, boxes_px, scores, labels):
        """从当前帧解码输出选取焦点 token. 返回 list[dict] (逐 batch)."""
        results = []
        for b in range(feats.shape[0]):
            keep = scores[b] > self.score_thr
            f, bx, sc, lb = feats[b][keep], boxes_px[b][keep], scores[b][keep], labels[b][keep]
            if f.shape[0] > self.max_tokens:
                top = sc.topk(self.max_tokens).indices
                f, bx, sc, lb = f[top], bx[top], sc[top], lb[top]
            results.append({"feat": f, "box": bx, "score": sc, "label": lb,
                            "age": torch.zeros_like(sc)})
        return results

    @torch.no_grad()
    def update(self, prev_mem, new_sel):
        """合并上一帧记忆 (年龄+1, 超龄丢弃) 与当前选中, 跨类去重保留高置信."""
        merged = []
        for pm, ns in zip(prev_mem, new_sel):
            cat = {k: torch.cat([pm[k], ns[k]], dim=0)
                   for k in ["feat", "box", "score", "label"]}
            age = torch.cat([pm["age"] + 1.0, torch.zeros_like(ns["score"])], dim=0)
            cat["age"] = age
            ok = cat["age"] <= self.lifespan
            cat = {k: v[ok] for k, v in cat.items()}
            if cat["score"].numel() > 1:
                order = cat["score"].argsort(descending=True)
                cat = {k: v[order] for k, v in cat.items()}
                ious = bbox_iou_px(cat["box"], cat["box"])
                used = torch.zeros_like(cat["score"], dtype=torch.bool)
                keep_idx = []
                for i in range(cat["score"].numel()):
                    if used[i]:
                        continue
                    keep_idx.append(i)
                    dup = (ious[i] > self.iou_dedup) & (cat["label"] == cat["label"][i])
                    used = used | dup
                idx = torch.tensor(keep_idx, dtype=torch.long, device=cat["score"].device)
                cat = {k: v[idx] for k, v in cat.items()}
            cat["score"] = cat["score"] * 0.9        # 未复现目标置信度衰减
            merged.append(cat)
        return merged

    @torch.no_grad()
    def get_tokens(self, mem, dx, dy, img_w, img_h):
        """按帧间平移对齐记忆框, 返回 (feat, box_px, age, keep_mask) 或 (None,...)*4."""
        B = len(mem)
        feats, boxes, ages = [], [], []
        max_m = 0
        dev = None
        for m in mem:
            b = m["box"] - torch.tensor([dx, dy, dx, dy], dtype=m["box"].dtype,
                                        device=m["box"].device)
            ok = ((b[:, 2] > 0) & (b[:, 0] < img_w) & (b[:, 3] > 0) & (b[:, 1] < img_h)
                  & ((b[:, 2] - b[:, 0]) > 2) & ((b[:, 3] - b[:, 1]) > 2))
            bb = b.clamp(0, max(img_w, img_h))
            feats.append(m["feat"][ok])
            boxes.append(bb[ok])
            # 注意: age_embed 表在构造时固定 (lifespan+2 行); 推理期改 lifespan 时
            # 必须钳制到表尺寸以内, 否则嵌入索引越界 -> CUDA device-side assert
            max_age = self.age_embed.num_embeddings - 1
            ages.append(m["age"][ok].long().clamp(min=0, max=max_age))
            max_m = max(max_m, int(ok.sum()))
            dev = m["feat"].device if m["feat"].numel() > 0 else dev
        if max_m == 0:
            return None, None, None, None
        C = self.mem_proj[0].in_features
        pad_f = torch.zeros(B, max_m, C, device=dev)
        pad_b = torch.zeros(B, max_m, 4, device=dev)
        pad_a = torch.zeros(B, max_m, dtype=torch.long, device=dev)
        keep = torch.zeros(B, max_m, dtype=torch.bool, device=dev)
        for i, (f, b, a) in enumerate(zip(feats, boxes, ages)):
            n = f.shape[0]
            if n:
                pad_f[i, :n], pad_b[i, :n], pad_a[i, :n] = f, b, a
                keep[i, :n] = True
        return pad_f, pad_b, pad_a, keep

# =====================================================================
# FocusDETR-lite 主模型
# =====================================================================

VARIANTS = {
    "base":    dict(use_sagn=False, loss="giou",       use_memory=False),
    "focus":   dict(use_sagn=True,  loss="focal_eiou", use_memory=False),
    "focus_t": dict(use_sagn=True,  loss="focal_eiou", use_memory=True),
    "base_t":  dict(use_sagn=False, loss="giou",       use_memory=True),
    # 隔离实验: Sa-Gn + GIoU (分离 Focal-EIoU 因素)
    "focus_g": dict(use_sagn=True,  loss="giou",       use_memory=False),
}

class FocusDETRLite(nn.Module):
    def __init__(self, num_classes=5, variant="focus", dim=256, n_heads=8,
                 n_queries=100, n_dec_layers=3, n_points=4,
                 mem_tokens=48, mem_lifespan=3):
        super().__init__()
        cfg = VARIANTS[variant]
        self.variant = variant
        self.cfg = cfg
        self.num_classes = num_classes
        self.n_queries = n_queries
        self.dim = dim

        self.backbone = ShipBackbone(use_sagn=cfg["use_sagn"])
        self.encoder = HybridEncoder(dim=dim)
        self.enc_score = nn.Linear(dim, 1)
        self.enc_bbox = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True),
                                      nn.Linear(dim, 4))
        self.decoder = Decoder(dim, n_heads, dim_ffn=dim * 4, n_levels=3, n_points=n_points,
                               n_layers=n_dec_layers, num_classes=num_classes)
        self.memory = TemporalFocusMemory(dim, max_tokens=mem_tokens,
                                          lifespan=mem_lifespan) if cfg["use_memory"] else None
        # DN 去噪查询 (DN-DETR 风格, 稳定匹配): 标签嵌入 + 框正弦嵌入 -> 投影
        self.dn_label_emb = nn.Embedding(num_classes, dim // 2)
        self.dn_proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True),
                                     nn.Linear(dim, dim))

    def get_param_groups(self):
        backbone = list(self.backbone.parameters()) + list(self.encoder.parameters())
        head = (list(self.decoder.parameters()) + list(self.enc_score.parameters())
                + list(self.enc_bbox.parameters()))
        if self.memory is not None:
            head += list(self.memory.parameters())
        return backbone, head

    def extract(self, x):
        feats = self.encoder(self.backbone(x))
        value_list = [feats[k].flatten(2).transpose(1, 2).contiguous() for k in ["p3", "p4", "p5"]]
        shapes = [(feats[k].shape[2], feats[k].shape[3]) for k in ["p3", "p4", "p5"]]
        return value_list, shapes

    def select_queries(self, value_list, shapes):
        scores, feats_all, boxes_all = [], [], []
        for v in value_list:
            scores.append(self.enc_score(v).squeeze(-1))
            boxes_all.append(self.enc_bbox(v).sigmoid())
            feats_all.append(v)
        scores = torch.cat(scores, dim=1)
        feats_all = torch.cat(feats_all, dim=1)
        boxes_all = torch.cat(boxes_all, dim=1)
        topk = scores.topk(self.n_queries, dim=1).indices
        q0 = feats_all.gather(1, topk.unsqueeze(-1).expand(-1, -1, feats_all.shape[-1]))
        ref = boxes_all.gather(1, topk.unsqueeze(-1).expand(-1, -1, 4))
        return q0, ref, boxes_all, scores

    def build_dn_queries(self, targets, device, n_pos=4, n_neg=1):
        """DN-DETR 风格去噪查询: 正样本组(标签真, 框加噪 -> 预测干净框) + 负样本组(乱标签 -> 背景)."""
        B = len(targets)
        S = n_pos + n_neg
        G_max = max(1, max(len(t["labels"]) for t in targets))
        D = G_max * S
        boxes = torch.zeros(B, D, 4, device=device)          # 加噪框 (输入/参考)
        clean = torch.zeros(B, D, 4, device=device)          # 干净 GT (回归目标)
        labels_in = torch.zeros(B, D, dtype=torch.long, device=device)   # 输入标签特征用
        tgt_labels = torch.full((B, D), -1, dtype=torch.long, device=device)  # 分类目标 (-1=背景)
        valid = torch.zeros(B, D, dtype=torch.bool, device=device)
        pos_mask = torch.zeros(B, D, dtype=torch.bool, device=device)
        for b, t in enumerate(targets):
            G = len(t["labels"])
            for g in range(G):
                gt = t["boxes"][g]
                lab = int(t["labels"][g])
                for s in range(S):
                    idx = g * S + s
                    c = gt[:2]
                    wh = gt[2:]
                    nc = c + (torch.rand(2, device=device) * 2 - 1) * 0.35 * wh
                    nwh = wh * torch.exp(torch.randn(2, device=device) * 0.3).clamp(0.5, 2.0)
                    nc = nc.clamp(0.05, 0.95)
                    nwh = nwh.clamp(0.02, 0.9)
                    noisy = torch.cat([nc, nwh])
                    boxes[b, idx] = noisy
                    clean[b, idx] = gt
                    valid[b, idx] = True
                    if s < n_pos:
                        labels_in[b, idx] = lab
                        tgt_labels[b, idx] = lab
                        pos_mask[b, idx] = True
                    else:
                        rand_lab = int(torch.randint(1, self.num_classes, (1,)).item())
                        rand_lab = (lab + rand_lab) % self.num_classes
                        labels_in[b, idx] = rand_lab     # 只是输入特征; 分类目标=背景(-1)
            # 填充组: 复制 GT0 的形态但 invalid
            for g in range(G, G_max):
                src = 0 if G > 0 else 0
                for s in range(S):
                    idx = g * S + s
                    if G > 0:
                        boxes[b, idx] = boxes[b, src * S + s]
                        clean[b, idx] = clean[b, src * S + s]
                        labels_in[b, idx] = labels_in[b, src * S + s]
        return {"boxes": boxes, "clean": clean, "labels_in": labels_in,
                "tgt_labels": tgt_labels, "valid": valid, "pos": pos_mask, "S": S}

    def forward(self, x, memory_state=None, motion=None, return_memory=False, targets=None):
        B, _, H, W = x.shape
        value_list, shapes = self.extract(x)
        q0, ref, boxes_all, scores_all = self.select_queries(value_list, shapes)
        q_type = torch.zeros(B, self.n_queries, dtype=torch.long, device=x.device)
        mem_keep = None

        if self.memory is not None and memory_state is not None and motion is not None:
            dx, dy = motion
            mem_f, mem_b, mem_age, mem_keep = self.memory.get_tokens(memory_state, dx, dy, W, H)
            if mem_f is not None and mem_f.shape[1] > 0:
                mem_f = self.memory.mem_proj(mem_f)
                if getattr(self, "mem_featureless", False):
                    mem_f = torch.zeros_like(mem_f)      # 消融: 无特征记忆(仅框+年龄接力)
                age_e = self.memory.age_embed(mem_age)
                g = self.memory.gate(torch.cat([mem_f, age_e], dim=-1))
                mem_f = g * mem_f + (1 - g) * age_e
                mem_f = mem_f * mem_keep.unsqueeze(-1).to(mem_f.dtype)
                q0 = torch.cat([q0, mem_f], dim=1)
                ref = torch.cat([ref, xyxy_px_to_cxcywh_norm(mem_b, W, H)], dim=1)
                q_type = torch.cat([q_type, mem_keep.long()], dim=1)

        n_main = q0.shape[1]                     # 主查询数 (内容 + 记忆)
        allow, dn_info = None, None
        if self.training and targets is not None and any(len(t["labels"]) for t in targets):
            dn_info = self.build_dn_queries(targets, x.device)
            D = dn_info["boxes"].shape[1]
            S = dn_info["S"]
            dn_feat = self.dn_proj(torch.cat([
                self.dn_label_emb(dn_info["labels_in"]),
                sine_embed_boxes(dn_info["boxes"], self.dim // 2)], dim=-1))
            q0 = torch.cat([q0, dn_feat], dim=1)
            ref = torch.cat([ref, dn_info["boxes"]], dim=1)
            q_type = torch.cat([q_type, torch.full((B, D), 2, dtype=torch.long,
                                                   device=x.device)], dim=1)
            # 可见性: 主查询块全互通; DN 各分组内部互通, 与主查询互相隔离
            Q = q0.shape[1]
            allow = torch.zeros(B, Q, Q, dtype=torch.bool, device=x.device)
            allow[:, :n_main, :n_main] = True
            for gidx in range(D // S):
                s0 = n_main + gidx * S
                allow[:, s0:s0 + S, s0:s0 + S] = True

        outputs, q_last = self.decoder(q0, ref, value_list, shapes, q_type=q_type, allow=allow)
        last = outputs[-1]
        out = {"pred_logits": last["logits"][:, :n_main],
               "pred_boxes": last["boxes"][:, :n_main],
               "aux_outputs": [{k: v[:, :n_main] for k, v in aux.items()}
                               for aux in outputs[:-1]],
               "enc_boxes": boxes_all, "enc_scores": scores_all,
               "n_main": n_main}
        if dn_info is not None:
            out["dn"] = {"logits": last["logits"][:, n_main:],
                         "boxes": last["boxes"][:, n_main:],
                         "clean": dn_info["clean"], "tgt_labels": dn_info["tgt_labels"],
                         "valid": dn_info["valid"], "pos": dn_info["pos"]}
        if return_memory:
            out["_mem_raw"] = {"feats": q_last[:, :n_main], "boxes": last["boxes"][:, :n_main],
                               "q_type": q_type[:, :n_main]}
        return out

    @torch.no_grad()
    def update_memory(self, prev_memory, raw, W, H, scores, labels):
        """帧末调用: scores [B,Q] 前景置信度, labels [B,Q] 预测类别 (argmax)."""
        if prev_memory is None:
            prev_memory = [{"feat": raw["feats"][b][:0], "box": raw["boxes"][b][:0],
                            "score": scores[b][:0], "label": labels[b][:0],
                            "age": scores[b][:0]} for b in range(raw["feats"].shape[0])]
        boxes_px = cxcywh_norm_to_xyxy_px(raw["boxes"], W, H)
        sel = self.memory.select(raw["feats"], boxes_px, scores, labels)
        return self.memory.update(prev_memory, sel)

def count_params(model):
    return sum(p.numel() for p in model.parameters())
