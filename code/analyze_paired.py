# -*- coding: utf-8 -*-
"""配对统计补强 (审稿意见 #3): McNemar + 轨迹级 Wilcoxon
========================================================
设计: GT 轨迹结构仅由真值决定 (与预测无关), 因此 ON/OFF 在同一批转移上
天然配对. 主检验 = 共同支撑集 (t-1 在 ON 与 OFF 下均命中) 上的 kept 配对:
  b = ON kept & OFF not;  c = OFF kept & ON not  -> 精确二项 McNemar.
次检验 = 全体转移上的 "sustained" (hit_{t-1} & hit_t) 配对.
轨迹级: 每轨迹共同支撑转移的 kept 率 (ON vs OFF) -> Wilcoxon 符号秩.
数据: s1 用已缓存流; s2/s3 现场采集并缓存 (e2e_s2/s3 ep7).
输出: runs/paired_tests.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import load_model, match_boxes_iou, pred_hits_gt              # noqa: E402
from eval_persistence import collect                                    # noqa: E402

RUNS = Path(__file__).resolve().parent.parent / "runs"
CACHE = RUNS / "eval_persistence"


def get_streams(seed, dev):
    """返回 (off_streams, on_streams); 优先用缓存, 否则现场采集. 无 ckpt 返回 None."""
    po = CACHE / f"stream_memoff_{seed}.pt"
    pn = CACHE / f"stream_memon_{seed}.pt"
    if po.exists() and pn.exists():
        return torch.load(po), torch.load(pn)
    if seed == "s1":
        po = CACHE / "stream_memoff_testbig.pt"
        pn = CACHE / "stream_memon_testbig.pt"
        if po.exists() and pn.exists():
            return torch.load(po), torch.load(pn)
        return None
    ck = None
    for cand in [RUNS / f"e2e_{seed}" / "ep7.pt", RUNS / f"e2e_{seed}" / "best.pt",
                 RUNS / f"e2e_{seed}" / "last.pt"]:
        if cand.exists():
            ck = cand
            break
    if ck is None:
        return None
    model, _ = load_model(str(ck), dev)
    off = collect(model, dev, "testbig", use_memory=False)
    torch.save(off, po)
    on = collect(model, dev, "testbig", use_memory=True)
    torch.save(on, pn)
    return off, on


def analyze(off_streams, on_streams):
    acc = {"b": 0, "c": 0,          # 共同支撑 McNemar
           "b_all": 0, "c_all": 0,  # 全体转移 sustained 配对
           "common_trans": 0, "all_trans": 0,
           "kept_on_common": 0, "kept_off_common": 0}
    track_pairs = []

    def finalize(tr):
        ho = [bool(x) for x in tr["hits_on"]]
        hf = [bool(x) for x in tr["hits_off"]]
        common = []
        for t in range(1, len(ho)):
            acc["all_trans"] += 1
            so = ho[t - 1] and ho[t]
            sf = hf[t - 1] and hf[t]
            if so != sf:
                if so:
                    acc["b_all"] += 1
                else:
                    acc["c_all"] += 1
            if ho[t - 1] and hf[t - 1]:      # 共同支撑
                common.append(t)
                acc["common_trans"] += 1
                if ho[t]:
                    acc["kept_on_common"] += 1
                if hf[t]:
                    acc["kept_off_common"] += 1
                if ho[t] and not hf[t]:
                    acc["b"] += 1
                if hf[t] and not ho[t]:
                    acc["c"] += 1
        if common:
            track_pairs.append((float(np.mean([ho[t] for t in common])),
                                float(np.mean([hf[t] for t in common]))))

    for seq_off, seq_on in zip(off_streams, on_streams):
        move = torch.tensor([seq_off["stride"], 0.0, seq_off["stride"], 0.0])
        tracks = []
        for fr_off, fr_on in zip(seq_off["frames"], seq_on["frames"]):
            pp_on = {"boxes": fr_on["boxes"], "labels": fr_on["labels"], "scores": fr_on["scores"]}
            pp_off = {"boxes": fr_off["boxes"], "labels": fr_off["labels"], "scores": fr_off["scores"]}
            gt_xyxy, labels = fr_off["gt_boxes"], fr_off["gt_labels"]
            matched = set()
            if tracks:
                prev_boxes = torch.stack([tr["box"] for tr in tracks]) - move
                prev_labels = torch.tensor([tr["label"] for tr in tracks])
                m = match_boxes_iou(prev_boxes, gt_xyxy, 0.4, (prev_labels, labels))
            else:
                m = torch.full((0,), -1, dtype=torch.long)
            new_tracks = []
            for i, tr in enumerate(tracks):
                if m.numel() and i < m.numel() and m[i] >= 0 and int(m[i]) not in matched:
                    gi = int(m[i]); matched.add(gi)
                    tr["box"] = gt_xyxy[gi]; tr["label"] = int(labels[gi])
                    tr["hits_on"].append(pred_hits_gt(pp_on, gt_xyxy[gi], int(labels[gi])))
                    tr["hits_off"].append(pred_hits_gt(pp_off, gt_xyxy[gi], int(labels[gi])))
                    new_tracks.append(tr)
                else:
                    finalize(tr)
            for gi in range(gt_xyxy.shape[0]):
                if gi not in matched:
                    new_tracks.append({
                        "box": gt_xyxy[gi], "label": int(labels[gi]),
                        "hits_on": [pred_hits_gt(pp_on, gt_xyxy[gi], int(labels[gi]))],
                        "hits_off": [pred_hits_gt(pp_off, gt_xyxy[gi], int(labels[gi]))]})
            tracks = new_tracks
        for tr in tracks:
            finalize(tr)

    b, c = acc["b"], acc["c"]
    n = b + c
    p_mcnemar = stats.binomtest(b, n, 0.5).pvalue if n else 1.0
    or_ = (b + 0.5) / (c + 0.5)
    se = np.sqrt(1 / (b + 0.5) + 1 / (c + 0.5))
    ci = (float(np.exp(np.log(or_) - 1.96 * se)), float(np.exp(np.log(or_) + 1.96 * se)))
    ba, ca = acc["b_all"], acc["c_all"]
    p_all = stats.binomtest(ba, ba + ca, 0.5).pvalue if (ba + ca) else 1.0
    if len(track_pairs) >= 10:
        ron = np.array([x[0] for x in track_pairs]); roff = np.array([x[1] for x in track_pairs])
        diff = ron - roff
        if np.all(diff == 0):
            p_w, med_diff = 1.0, 0.0
        else:
            try:
                p_w = float(stats.wilcoxon(ron, roff).pvalue)
            except Exception:
                p_w = None
            med_diff = float(np.median(diff))
    else:
        p_w, med_diff = None, None
    res = {
        "common_transitions": acc["common_trans"], "all_transitions": acc["all_trans"],
        "P_kept_on_common": acc["kept_on_common"] / max(acc["common_trans"], 1),
        "P_kept_off_common": acc["kept_off_common"] / max(acc["common_trans"], 1),
        "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p_two_sided": float(p_mcnemar),
        "mcnemar_odds_ratio": float(or_), "odds_ratio_ci95": ci,
        "sustained_pair_b": ba, "sustained_pair_c": ca, "sustained_pair_p": float(p_all),
        "n_tracks": len(track_pairs), "wilcoxon_p": p_w, "wilcoxon_median_diff": med_diff,
    }
    return res


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for seed in ["s1", "s2", "s3"]:
        got = get_streams(seed, dev)
        if got is None:
            print(f"[{seed}] 缺 checkpoint/缓存, 跳过")
            continue
        off, on = got
        r = analyze(off, on)
        out[seed] = r
        print(f"[{seed}] common={r['common_transitions']} all={r['all_transitions']} | "
              f"P_kept ON={r['P_kept_on_common']:.4f} OFF={r['P_kept_off_common']:.4f} | "
              f"McNemar b={r['mcnemar_b']} c={r['mcnemar_c']} p={r['mcnemar_p_two_sided']:.2e} "
              f"OR={r['mcnemar_odds_ratio']:.2f} CI={r['odds_ratio_ci95'][0]:.2f}-{r['odds_ratio_ci95'][1]:.2f} | "
              f"sustained p={r['sustained_pair_p']:.2e} | "
              f"Wilcoxon n={r['n_tracks']} p={r['wilcoxon_p']} medΔ={r['wilcoxon_median_diff']}")
    (RUNS / "paired_tests.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved ->", RUNS / "paired_tests.json")


if __name__ == "__main__":
    main()
