"""Clustering-discount robustness check for TP-stability (direction-2 paper).

Canonical method (must match analyze_tp.py):
  - two-sided Fisher exact test on the 2x2 hit-transition table
    [[kept_on, prev_on-kept_on], [kept_off, prev_off-kept_off]]
  - conservative effective-sample discount: floor-divide all four cells by k
    (k = 10 and 20), then re-test.

Inputs : runs/eval_e2e_ep7/results.json        (seed s1 = 42)
         runs/eval_e2e_s2_ep7/results.json     (seed s2 = 2027)
         runs/eval_e2e_s3_ep7/results.json     (seed s3 = 2028)
Output : runs/analyze_tp_discount.json  (+ console table)
"""
import json
from pathlib import Path
from scipy import stats

RUNS = Path(__file__).resolve().parent.parent / "runs"

SEEDS = {
    "s1": RUNS / "eval_e2e_ep7" / "results.json",
    "s2": RUNS / "eval_e2e_s2_ep7" / "results.json",
    "s3": RUNS / "eval_e2e_s3_ep7" / "results.json",
}


def pooled_table(res):
    """Pool the three domains' hit-transition tables (mem-on vs mem-off)."""
    on = off = None
    for dom in ("test", "test_hard", "testbig"):
        k_on = f"video_{dom}_motion-gt_mem-on"
        k_off = f"video_{dom}_motion-gt_mem-off"
        if k_on not in res or k_off not in res:
            continue
        o, f = res[k_on], res[k_off]
        o_row = [o["tp_kept"], o["tp_prev"] - o["tp_kept"]]
        f_row = [f["tp_kept"], f["tp_prev"] - f["tp_kept"]]
        if on is None:
            on, off = o_row, f_row
        else:
            on = [a + b for a, b in zip(on, o_row)]
            off = [a + b for a, b in zip(off, f_row)]
    return on, off


def test_table(on, off, k=1):
    t = [[on[0] // k, on[1] // k], [off[0] // k, off[1] // k]]
    p = stats.fisher_exact(t)[1]  # two-sided, matches analyze_tp.py
    return p, t


def main():
    out = {}
    for seed, path in SEEDS.items():
        res = json.load(open(path))
        for split in ("testbig", "pooled"):
            if split == "testbig":
                on = [res["video_testbig_motion-gt_mem-on"]["tp_kept"]]
                on.append(res["video_testbig_motion-gt_mem-on"]["tp_prev"] - on[0])
                off = [res["video_testbig_motion-gt_mem-off"]["tp_kept"]]
                off.append(res["video_testbig_motion-gt_mem-off"]["tp_prev"] - off[0])
            else:
                on, off = pooled_table(res)
            rec = {}
            for k in (1, 10, 20):
                p, t = test_table(on, off, k)
                rec[f"{k}x"] = {"p_two_sided": p,
                                "table": t,
                                "ratio_on": round(t[0][0] / max(sum(t[0]), 1), 4),
                                "ratio_off": round(t[1][0] / max(sum(t[1]), 1), 4)}
            out[seed] = {**out.get(seed, {}), split: rec}
            print(f"{seed:3s} {split:6s} | full p={rec['1x']['p_two_sided']:.3e} | "
                  f"10x p={rec['10x']['p_two_sided']:.4f} {rec['10x']['table']} | "
                  f"20x p={rec['20x']['p_two_sided']:.4f} {rec['20x']['table']}")
    with open(RUNS / "analyze_tp_discount.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("saved ->", RUNS / "analyze_tp_discount.json")


if __name__ == "__main__":
    main()
