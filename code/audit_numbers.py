# -*- coding: utf-8 -*-
"""P4 全稿数字审计: 校验论文中每个关键数字 (1) 可从落盘数据复算;
(2) 在 markdown 草稿与 LaTeX 全稿中一致出现.
用法: python audit_numbers.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
MD = ROOT / "manuscript" / "draft_en_v0.1.md"
TEX_FILES = sorted((ROOT / "manuscript" / "submission" / "english" / "sections").glob("*.tex")) + \
            [ROOT / "manuscript" / "submission" / "english" / "main.tex"] + \
            sorted((ROOT / "manuscript" / "submission" / "english").glob("t*.tex"))

fails = []


def load(d):
    return json.loads((RUNS / d / "results.json").read_text(encoding="utf-8"))


def tb(res, split, mode):
    v = res[f"video_{split}_motion-gt_mem-{mode}"]
    return v["tp_kept"], v["tp_prev"]


def pooled(res, mode):
    k = n = 0
    for s in ["test", "test_hard", "testbig"]:
        kk, nn = tb(res, s, mode)
        k += kk; n += nn
    return k, n


# ---- 1) 从数据复算关键值 ----
r1 = load("eval_e2e_ep7")
p = json.loads((RUNS / "eval_persistence" / "results.json").read_text(encoding="utf-8"))
disc = json.loads((RUNS / "analyze_tp_discount.json").read_text(encoding="utf-8"))
yolo = json.loads((RUNS / "eval_yolo_baseline" / "results.json").read_text(encoding="utf-8"))
recs = {}
po, pn = pooled(r1, "on"); qo, qn = pooled(r1, "off")
recs["s1 pooled on"] = po / pn
recs["s1 pooled off"] = qo / qn
recs["s1 pooled gain%"] = (po / pn) / (qo / qn) * 100 - 100
to, tn = tb(r1, "testbig", "on"); uo, un = tb(r1, "testbig", "off")
recs["testbig on"] = to / tn
recs["testbig off"] = uo / un
recs["testbig gain%"] = (to / tn) / (uo / un) * 100 - 100
recs["tfm vs persist clean p"] = stats.fisher_exact(
    [[p["sanity_memon"]["tp_kept"], p["sanity_memon"]["tp_prev"] - p["sanity_memon"]["tp_kept"]],
     [p["persist_inf"]["tp_kept"], p["persist_inf"]["tp_prev"] - p["persist_inf"]["tp_kept"]]])[1]
recs["persist clean"] = p["persist_inf"]["TP_stability"]
recs["persist sig15"] = p["persist_inf_sig15"]["TP_stability"]
recs["persist sig5"] = p["persist_inf_sig5"]["TP_stability"]
r5 = load("eval_sweep_noise5"); r15 = load("eval_sweep_noise15")
recs["tfm sig5"] = tb(r5, "testbig", "on")[0] / tb(r5, "testbig", "on")[1]
recs["tfm sig15"] = tb(r15, "testbig", "on")[0] / tb(r15, "testbig", "on")[1]
recs["persist sig5 p vs tfm"] = stats.fisher_exact(
    [[tb(r5, "testbig", "on")[0], tb(r5, "testbig", "on")[1] - tb(r5, "testbig", "on")[0]],
     [p["persist_inf_sig5"]["tp_kept"], p["persist_inf_sig5"]["tp_prev"] - p["persist_inf_sig5"]["tp_kept"]]])[1]
recs["persist sig15 p vs tfm"] = stats.fisher_exact(
    [[tb(r15, "testbig", "on")[0], tb(r15, "testbig", "on")[1] - tb(r15, "testbig", "on")[0]],
     [p["persist_inf_sig15"]["tp_kept"], p["persist_inf_sig15"]["tp_prev"] - p["persist_inf_sig15"]["tp_kept"]]])[1]
recs["yolo test mAP50"] = yolo["test"]["mAP50"]
recs["yolo tp"] = yolo["video_testbig"]["TP_stability"]
recs["s1 20x discount p"] = disc["s1"]["testbig"]["20x"]["p_two_sided"]
recs["featureless"] = load("eval_sweep_featureless")["video_testbig_motion-gt_mem-on"]["TP_stability"]
recs["noalign"] = load("eval_sweep_noalign")["video_testbig_motion-gt_mem-on"]["TP_stability"]
recs["phase"] = (load("eval_sweep_phase").get("video_testbig_motion-phase_mem-on")
                 or load("eval_sweep_phase")["video_testbig_motion-gt_mem-on"])["TP_stability"]
su = json.loads((RUNS / "eval_e2e_unsup" / "summary.json").read_text(encoding="utf-8"))
recs["unsup pooled on"] = su["pooled"]["on"]["stability"]
recs["unsup pooled off"] = su["pooled"]["off"]["stability"]
recs["unsup testbig on"] = su["per_split"]["testbig"]["on"]["stability"]
recs["unsup testbig off"] = su["per_split"]["testbig"]["off"]["stability"]
dcj = json.loads((RUNS / "density_control" / "summary.json").read_text(encoding="utf-8"))
recs["density rm off"] = dcj["recall_matched_off"]["TP_stability"]
recs["density dm off"] = dcj["density_matched_off"]["TP_stability"]
drj = json.loads((RUNS / "eval_drift" / "results.json").read_text(encoding="utf-8"))
recs["drift4 tfm"] = drj["drift_sd4"]["on_tfm"]["TP_stability"]
recs["drift4 wrapper"] = drj["drift_sd4"]["wrapper"]["TP_stability"]
recs["drift8 tfm"] = drj["drift_sd8"]["on_tfm"]["TP_stability"]
wpj = json.loads((RUNS / "eval_wrapper_phase" / "results.json").read_text(encoding="utf-8"))
recs["wrapper phase"] = wpj["wrapper_phase"]["TP_stability"]
recs["tfm phase"] = wpj["tfm_phase"]["TP_stability"]
tkj = json.loads((RUNS / "sweep_tauK" / "results.json").read_text(encoding="utf-8"))
recs["tauK 0.3/48"] = tkj["tau0.3_K48"]["TP_stability"]
recs["tauK 0.2/48"] = tkj["tau0.2_K48"]["TP_stability"]
recs["tauK 0.4/48"] = tkj["tau0.4_K48"]["TP_stability"]
recs["tauK 0.5/48"] = tkj["tau0.5_K48"]["TP_stability"]
fpj = json.loads((RUNS / "fps_stats2.json").read_text(encoding="utf-8"))
recs["fps2 on mean"] = fpj["on_mean"]
recs["fps2 off mean"] = fpj["off_mean"]

print("== recomputed canonical values ==")
for k, v in recs.items():
    print(f"  {k:28s} {v:.4f}" if isinstance(v, float) else f"  {k:28s} {v}")

# ---- 2) 关键数字在 md 与 tex 中的出现检查 ----
CRITICAL = [
    ("0.724", r"0\.724"), ("0.506", r"0\.506"), (r"+43.8%", r"\+43\.8"),
    (r"+34.6%", r"\+34\.6"), (r"+21.3%", r"21\.3"), (r"+38.7%", r"38\.7"),
    ("0.734", r"0\.734"), ("0.545", r"0\.545"), ("0.594", r"0\.594"),
    ("0.673", r"0\.673"), ("0.603", r"0\.603"), ("p=0.44", r"0\.44"),
    ("0.0006/0.041/0.0016", r"0\.0006"), ("0.691", r"0\.691"), ("0.508", r"0\.508"),
    ("0.722", r"0\.722"), ("0.735", r"0\.735"), ("0.714", r"0\.714"),
    ("0.172", r"0\.172"), ("0.147", r"0\.147"), ("0.132", r"0\.132"),
    ("0.966", r"0\.966"), ("0.906", r"0\.906"), ("0.980", r"0\.980"),
    ("0.152", r"0\.152"), ("0.146", r"0\.146"), ("0.184", r"0\.184"), ("0.221", r"0\.221"),
    ("0.218", r"0\.218"), ("0.228", r"0\.228"),
    ("14.31", r"14\.31"), ("14.58", r"14\.58"), ("0.26M", r"0\.26"),
    ("70-85 FPS 口径改 68-85", r"68(?:--|–)85"), ("3.3 ms", r"3\.3"), ("p=0.45", r"0\.45"),
    ("FPS 81.6", r"81\.6"), ("FPS 85\.1", r"85\.1"),
    ("recall-matched 0.576", r"0\.576"), ("density-matched 0.566", r"0\.566"),
    ("drift 0.629", r"0\.629"), ("drift 0.591", r"0\.591"), ("drift wrapper 0.540", r"0\.540"),
    ("Holm crack 0.085", r"0\.085"), ("src decomp 0.79", r"0\.79"),
    ("phase tie 0.727", r"0\.727"), ("phase p 0\.67", r"p = 0\.67|p=0\.67"),
    ("phase wrapper P 0\.462", r"0\.462"), ("tau sweep 0.71-0.72", r"0\.71.{0,3}0\.72"),
    ("starve 0\.578", r"0\.578"), ("starve 0\.509", r"0\.509"),
    ("continuity-corrected", r"continuity-corrected"), ("McNemar b/c 506/5", r"506/5"), ("8.9 boxes", r"8\.9"),
    ("0.977", r"0\.977"), ("0.702", r"0\.702"), ("0.585", r"0\.585"),
    ("dose 37.5", r"37\.5"), ("dose 40.9", r"40\.9"),
    ("McNemar OR 71-92", r"71(?:--|–)92"), ("McNemar p", r"10\^\{-87\}|10\^-87"),
    ("Wilcoxon p", r"10\^\{-53\}|10\^-53"),
    ("fidelity sim contrast 0.29", r"0\.29"), ("fidelity real contrast range", r"1\.23(?:--|–)2\.25"),
    ("fidelity edge range", r"26\.7(?:--|–)46\.4"),
    ("clutter gain 48.8", r"48\.8"), ("clutter heavy edge 32.8", r"32\.8"),
    ("clutter heavy on 0.408", r"0\.408"), ("clutter heavy off 0.149", r"0\.149"),
    ("rotation gain 55.1", r"55\.1"), ("per-class pitting 63.3", r"63\.3"),
    ("unsup pooled 0.575/0.576", r"0\.575.{0,90}0\.576|0\.576.{0,90}0\.575"),
    ("unsup testbig 0.582/0.583", r"0\.582.{0,90}0\.583|0\.583.{0,90}0\.582"),
    ("unsup p 0.96", r"p = 0\.96"),
    ("ever-hit low 48.6", r"48\.6"), ("ever-hit high 49.4", r"49\.4"), ("rust 36.7", r"36\.7"),
    ("200 sequences", r"200.sequence|200 sequences"), ("675", r"675"), ("613", r"613"),
    ("3811", r"3[,{}]*811"), ("3393", r"3[,{}]*393"),
]
md_text = MD.read_text(encoding="utf-8")
tex_text = "\n".join(f.read_text(encoding="utf-8") for f in TEX_FILES)
print("\n== presence check (md / tex) ==")
for label, pat in CRITICAL:
    in_md = bool(re.search(pat, md_text))
    in_tex = bool(re.search(pat, tex_text))
    status = "OK" if (in_md and in_tex) else "MISSING"
    if status != "OK":
        fails.append(f"{label}: md={in_md} tex={in_tex}")
    print(f"  [{status}] {label:22s} md={in_md} tex={in_tex}")

# ---- 3) 复算值 vs 论文值 抽查 ----
def close(a, b, tol):
    ok = abs(a - b) <= tol
    if not ok:
        fails.append(f"value mismatch: {a} vs {b}")
    return ok

checks = [
    ("s1 pooled gain +43.8", close(recs["s1 pooled gain%"], 43.8, 0.5)),
    ("testbig +43.0", close(recs["testbig gain%"], 43.0, 0.5)),
    ("persist clean 0.734", close(recs["persist clean"], 0.734, 0.002)),
    ("tfm vs persist clean p~0.44", close(recs["tfm vs persist clean p"], 0.44, 0.05)),
    ("sig5 p ~2e-7", recs["persist sig5 p vs tfm"] < 1e-5),
    ("sig15 p ~6e-4", 1e-5 < recs["persist sig15 p vs tfm"] < 0.005),
    ("yolo tp 0.966", close(recs["yolo tp"], 0.966, 0.002)),
    ("yolo test mAP50 0.906", close(recs["yolo test mAP50"], 0.906, 0.002)),
    ("unsup pooled on 0.5755", close(recs["unsup pooled on"], 0.5755, 0.001)),
    ("unsup pooled off 0.5762", close(recs["unsup pooled off"], 0.5762, 0.001)),
    ("unsup testbig on 0.5820", close(recs["unsup testbig on"], 0.5820, 0.001)),
    ("unsup testbig off 0.5826", close(recs["unsup testbig off"], 0.5826, 0.001)),
    ("density recall-matched off 0.5756", close(recs["density rm off"], 0.5756, 0.002)),
    ("density density-matched off 0.5661", close(recs["density dm off"], 0.5661, 0.002)),
    ("drift4 tfm 0.6293", close(recs["drift4 tfm"], 0.6293, 0.002)),
    ("drift4 wrapper 0.5674", close(recs["drift4 wrapper"], 0.5674, 0.002)),
    ("drift8 tfm 0.5907", close(recs["drift8 tfm"], 0.5907, 0.002)),
    ("fps2 on 81.6", close(recs["fps2 on mean"], 81.6, 0.5)),
    ("wrapper phase 0.7273", close(recs["wrapper phase"], 0.7273, 0.002)),
    ("tfm phase 0.7218", close(recs["tfm phase"], 0.7218, 0.002)),
    ("tauK 0.3/48 sanity 0.7244", close(recs["tauK 0.3/48"], 0.7244, 0.001)),
    ("tauK 0.2/48 0.7106", close(recs["tauK 0.2/48"], 0.7106, 0.002)),
    ("tauK 0.4/48 0.5776", close(recs["tauK 0.4/48"], 0.5776, 0.002)),
    ("tauK 0.5/48 0.5094", close(recs["tauK 0.5/48"], 0.5094, 0.002)),
    ("fps2 off 85.1", close(recs["fps2 off mean"], 85.1, 0.5)),
]
print("\n== consistency spot checks ==")
for name, ok in checks:
    print(f"  [{'OK' if ok else 'FAIL'}] {name}")

# ---- 4) 负检查: 旧措辞必须清零; fm 修复必须就位 ----
neg = [("pre-registered", "pre-registered"), ("预注册", None), ("study record", "study record")]
for label, pat in neg:
    if pat is None:
        continue
    hit = (pat in md_text) or (pat in tex_text)
    if hit:
        fails.append(f"stale wording remains: {label}")
    print(f"  [{'FAIL' if hit else 'OK'}] no '{label}' in md/tex")
fm_ok = re.search(r"\\tfm\{\} gain", tex_text) is not None
if not fm_ok:
    fails.append("04_setup: fm gain escape not restored")
print(f"  [{'OK' if fm_ok else 'FAIL'}] '\\tfm{{}} gain' restored (fm fix)")

print(f"\n===== AUDIT RESULT: {'PASS' if not fails else 'ISSUES: ' + str(len(fails))} =====")
for f in fails:
    print(" -", f)
