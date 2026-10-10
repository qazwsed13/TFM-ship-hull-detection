# Temporal Focus Memory — code, raw results, and evaluation protocol

Reference implementation, raw result files and the analysis protocol for the manuscript
**"Temporal Focus Memory for Stable Real-Time Object Detection of Ship Wall-Climbing Robots"**
(under review at *Intelligent Marine Technology and Systems*).

TFM is an instance-level focus-token memory for real-time DETR-style detection: detections above a
confidence threshold are distilled into at most 48 tokens (feature, box, class, age), warped
between frames by measured motion, and injected into the decoder query set, where they are
supervised by the same set-prediction matching as every other query; aging, IoU de-duplication
and confidence decay bound the bank. The study is a fully procedural ship-hull inspection
simulation (five defect classes; a 200-sequence high-power split among the test sets), in which
every claim is measured under identical weights with only the memory pathway toggled.

## What is in this repository

| Path | Content |
|---|---|
| `code/` | detector + TFM implementation, training and evaluation pipeline, all statistical analyses, and the number audit of the manuscript |
| `runs/` | **raw result files** — one `results.json` (or summary JSON) per experiment, plus training logs (`train_log.csv`). Every number reported in the paper traces to a file here |
| `docs/PROTOCOL.md` | the evaluation protocol (fixed in advance of the confirmatory runs) |
| `docs/REPRODUCTION.md` | environment, step-by-step reproduction, external dependencies |

## Environment

- Python 3.8.15; PyTorch 2.4.1+cu121; torchvision 0.19.1; OpenCV 4.13; SciPy 1.10.1;
  NumPy 1.24.4; Matplotlib 3.7.5
- fp32 throughout; a single RTX 4060 Laptop GPU (8 GB); total study compute ≈14–16 GPU-hours
- `pip install -r requirements.txt`

## Reproducing the study (summary — full commands in `docs/REPRODUCTION.md`)

1. **Benchmark** (deterministic, fixed seed): `python code/gen_dataset.py` (then
   `python code/gen_dataset.py --extra_split testbig` for the high-power split)
2. **Detector**: stage-1 single-frame training chain — `bash code/run_all_train.sh`
   (or individual `train.py --stage 1 ...` commands)
3. **TFM (flagship)**: end-to-end temporal training — see `docs/REPRODUCTION.md` §E
   (8 epochs from the stage-1 checkpoint; seed 42)
4. **Replicates and ablations**: `bash code/run_sci_chain.sh`
5. **Evaluation and statistics**: `bash code/run_all_eval.sh` and the `eval_*.py` /
   `analyze_*.py` scripts; `python code/audit_numbers.py` re-checks every number
   reported in the manuscript against `runs/`

## Paper-to-file map (headline numbers)

| Manuscript item | Raw result file |
|---|---|
| Main result, seed s1 (pooled +43.8%) | `runs/eval_e2e_ep7/results.json` |
| Replicates s2 / s3 | `runs/eval_e2e_s2_ep7/results.json`, `runs/eval_e2e_s3_ep7/results.json` |
| Cluster-discounted tests (10× / 20×) | `runs/analyze_tp_discount.json` |
| Paired checks (McNemar / Wilcoxon) | `runs/paired_tests.json` |
| Per-class and practical relevance | `runs/byclass_practical.json`, `runs/byclass_holm.json` |
| Ablations (alignment, lifespan, noise, phase) | `runs/eval_sweep_*/results.json` |
| Supervision ablation (tokens excluded from matching) | `runs/eval_e2e_unsup/summary.json` |
| Admission-threshold × capacity sweep | `runs/sweep_tauK/results.json` |
| Density control (recall/density-matched, frontier) | `runs/density_control/summary.json` |
| Drift (correlated motion error) experiment | `runs/eval_drift/results.json` |
| Zero-learning wrapper (clean, noisy, phase motion) | `runs/eval_persistence/results.json`, `runs/eval_wrapper_phase/results.json`, `runs/eval_phase_pr/results.json` |
| ByteTrack-style tracker baseline | `runs/eval_tracker_baseline/results.json` |
| YOLOv5s reference (and + wrapper) | `runs/eval_yolo_baseline/results.json`, `runs/eval_yolo_baseline/persistence_results.json` |
| Texture-clutter probe | `runs/eval_texture_robustness/results.json` |
| Rotation probe | `runs/eval_rotation/results.json` |
| Fidelity statistics vs real imagery | `runs/fidelity_stats.json` |
| Throughput (FPS) | `runs/fps_stats.json`, `runs/fps_stats2.json` |
| False-positive persistence ("ghost") proxy | `runs/fp_ghost.json` |
| Dose–response checkpoints (epochs 1/3/5/7) | `runs/eval_e2e_ep1|ep3|ep5/results.json` |

## What is *not* included (and how to obtain it)

- **Model checkpoints** (≈5 GB total): not bundled; the full training chains are reproducible
  from the released recipes. Checkpoints will be archived on Zenodo upon acceptance.
- **The dataset** (`data/defect_ds`): regenerated deterministically by `code/gen_dataset.py`
  from a fixed seed.
- **YOLOv5s reference**: requires the third-party `ultralytics/yolov5` code (AGPL-3.0, not
  redistributed here; clone into `third_party/yolov5`) and a checkpoint fine-tuned with the
  standard recipe described in the paper (80 epochs on the identical train split).
  `eval_yolo_baseline.py` reads the weights from `yolo_runs/weights/best.pt`.
- **Real hull-surface imagery** for the fidelity check (Fig. 7 of the manuscript): reproduced
  from the published literature and not redistributed; `make_fidelity_check.py` documents the
  exact measurement procedure.

## License

Our code is released under the MIT License (see `LICENSE`). Third-party components and imagery
are referenced, not redistributed. (YOLOv5 is AGPL-3.0; respect its license when reproducing
the reference.)

## Citation

> Yimeng Chai, Xiaoyi Wang, Linrui Fu, Xin Li. *Temporal Focus Memory for Stable
> Real-Time Object Detection of Ship Wall-Climbing Robots.* Under review, Intelligent Marine
> Technology and Systems.

*Raw result files are archived exactly as produced by the released code; the manuscript's
number audit (`code/audit_numbers.py`) recomputes every reported value from these files.*
