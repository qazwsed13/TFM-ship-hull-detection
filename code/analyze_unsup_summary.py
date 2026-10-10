# -*- coding: utf-8 -*-
"""mem_unsup 消融汇总: 从 runs/eval_e2e_unsup/results.json 复算
每划分 ON/OFF TP 稳定度、pooled、相对增益与 Fisher p, 落盘 summary.json。
论文 §5.3/§6.1 的每个数字必须与本文件一致 (sync_v8c.py 带断言)。
"""
import json
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "runs" / "eval_e2e_unsup" / "results.json"
res = json.loads(RES.read_text(encoding="utf-8"))


def tb(split, mode):
    v = res[f"video_{split}_motion-gt_mem-{mode}"]
    return v["tp_kept"], v["tp_prev"], v["TP_stability"]


def pooled(mode):
    k = n = 0
    for s in ["test", "test_hard", "testbig"]:
        kk, nn, _ = tb(s, mode)
        k += kk
        n += nn
    return k, n, k / n


out = {"source": "runs/eval_e2e_unsup/results.json",
       "ckpt": res["ckpt"], "per_split": {}, "pooled": {}}

for s in ["test", "test_hard", "testbig"]:
    on = tb(s, "on")
    off = tb(s, "off")
    p = stats.fisher_exact([[on[0], on[1] - on[0]],
                            [off[0], off[1] - off[0]]])[1]
    out["per_split"][s] = {
        "on": {"kept": on[0], "prev": on[1], "stability": on[2]},
        "off": {"kept": off[0], "prev": off[1], "stability": off[2]},
        "gain_pct": (on[2] / off[2] - 1) * 100,
        "fisher_p": float(p)}

pon = pooled("on")
poff = pooled("off")
p_pool = stats.fisher_exact([[pon[0], pon[1] - pon[0]],
                             [poff[0], poff[1] - poff[0]]])[1]
out["pooled"] = {
    "on": {"kept": pon[0], "prev": pon[1], "stability": pon[2]},
    "off": {"kept": poff[0], "prev": poff[1], "stability": poff[2]},
    "gain_pct": (pon[2] / poff[2] - 1) * 100,
    "fisher_p": float(p_pool)}

(ROOT / "runs" / "eval_e2e_unsup" / "summary.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")

for s in ["test", "test_hard", "testbig"]:
    d = out["per_split"][s]
    print(f"[{s:9s}] ON {d['on']['stability']:.4f} ({d['on']['kept']}/{d['on']['prev']}) "
          f"OFF {d['off']['stability']:.4f} ({d['off']['kept']}/{d['off']['prev']}) "
          f"gain {d['gain_pct']:+.2f}% p={d['fisher_p']:.3g}")
d = out["pooled"]
print(f"[pooled   ] ON {d['on']['stability']:.4f} ({d['on']['kept']}/{d['on']['prev']}) "
      f"OFF {d['off']['stability']:.4f} ({d['off']['kept']}/{d['off']['prev']}) "
      f"gain {d['gain_pct']:+.2f}% p={d['fisher_p']:.3g}")

# 断言: 与论文将写入口径一致
assert abs(out["per_split"]["testbig"]["on"]["stability"] - 0.5820) < 0.001
assert abs(out["per_split"]["testbig"]["gain_pct"]) < 0.3
assert abs(out["pooled"]["on"]["stability"] - 0.5755) < 0.001
assert abs(out["pooled"]["gain_pct"]) < 0.3
assert out["pooled"]["fisher_p"] > 0.5 and out["per_split"]["testbig"]["fisher_p"] > 0.5
print("all asserts passed; saved ->", ROOT / "runs" / "eval_e2e_unsup" / "summary.json")
