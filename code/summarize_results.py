# -*- coding: utf-8 -*-
"""结果汇总: 读取各变体 eval results.json -> 打印 Markdown 表格 (供 W5 报告)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ["rust", "crack", "weld_defect", "peel", "pitting"]

RUNS = [("base_s1b", "base (V0)"), ("focus_s1b", "focus (V1)"),
        ("focus_t_s2b", "focus+T (V2)"), ("base_t_s2b", "base+T (V3)")]

def load(name):
    f = ROOT / "runs" / f"eval_{name}" / "results.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) and v == v else "—"


def main():
    print("## 帧级检测 (test / test_hard)\n")
    print("| 变体 | test mAP50 | test mAP50-95 | test AP_S | hard mAP50 | hard AP_S | FPS(bs1) |")
    print("|---|---|---|---|---|---|---|")
    for name, label in RUNS:
        r = load(name)
        if r is None:
            print(f"| {label} | (未评估) | | | | | |")
            continue
        t, h = r.get("test", {}), r.get("test_hard", {})
        print(f"| {label} | {fmt(t.get('mAP50'))} | {fmt(t.get('mAP50_95'))} | "
              f"{fmt(t.get('mAP@small'))} | {fmt(h.get('mAP50'))} | {fmt(h.get('mAP@small'))} | "
              f"{fmt(r.get('fps_bs1', float('nan')), 1)} |")

    print("\n## 视频时序指标 (test 序列, 运动=真值)\n")
    print("| 变体/模式 | 视频 mAP50 | TCM↑ | flicker↓ | 恢复延迟↓ | miss_runs |")
    print("|---|---|---|---|---|---|")
    for name, label in RUNS:
        r = load(name)
        if r is None:
            continue
        for tag in [k for k in r.keys() if k.startswith("video_test_motion-gt")]:
            v = r[tag]
            mode = "mem-on" if "mem-on" in tag else "mem-off"
            print(f"| {label} {mode} | {fmt(v.get('mAP50'))} | {fmt(v.get('TCM'), 3)} | "
                  f"{v.get('flicker_1frame', '—')} | {fmt(v.get('recovery_delay_avg'), 2)} | "
                  f"{v.get('miss_runs', '—')} |")
    print("\n## 视频时序指标 (test_hard 序列)\n")
    print("| 变体/模式 | 视频 mAP50 | TCM↑ | flicker↓ | 恢复延迟↓ |")
    print("|---|---|---|---|---|")
    for name, label in RUNS:
        r = load(name)
        if r is None:
            continue
        for tag in [k for k in r.keys() if k.startswith("video_test_hard_motion-gt")]:
            v = r[tag]
            mode = "mem-on" if "mem-on" in tag else "mem-off"
            print(f"| {label} {mode} | {fmt(v.get('mAP50'))} | {fmt(v.get('TCM'), 3)} | "
                  f"{v.get('flicker_1frame', '—')} | {fmt(v.get('recovery_delay_avg'), 2)} |")

    print("\n## 每类 AP50 (test)\n")
    print("| 变体 | " + " | ".join(CLASS_NAMES) + " |")
    print("|---|" + "---|" * len(CLASS_NAMES))
    for name, label in RUNS:
        r = load(name)
        if r is None:
            continue
        ap = r.get("test", {}).get("AP50_per_class", [])
        print(f"| {label} | " + " | ".join(fmt(v, 3) for v in ap) + " |")
    print("\n## 运动估计消融 (相位相关, focus+T)\n")
    r = load("focus_t_s2")
    if r:
        for tag in [k for k in r.keys() if "phase" in k]:
            v = r[tag]
            print(f"- {tag}: mAP50={fmt(v.get('mAP50'))} TCM={fmt(v.get('TCM'),3)} "
                  f"flicker={v.get('flicker_1frame')} 恢复延迟={fmt(v.get('recovery_delay_avg'),2)}")


if __name__ == "__main__":
    main()
