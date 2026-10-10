# Reproduction guide

Full pipeline: dataset generation → detector training (stage 1) → two-stage probes → end-to-end
TFM training → evaluation → statistics. Commands are given relative to the repository root;
`code/` scripts resolve all paths relative to their own location (`../runs`, `../data`).

## 0. Environment

```
pip install -r requirements.txt
python code/env_check.py       # optional environment probe
```

Tested on: Python 3.8.15, PyTorch 2.4.1+cu121, torchvision 0.19.1, OpenCV 4.13,
SciPy 1.10.1, NumPy 1.24.4, Matplotlib 3.7.5; single RTX 4060 Laptop GPU (8 GB), fp32
(mixed precision was disabled after intermittent fp16 gradient overflow).

Shell scripts use `PY="${PY:-python}"`; set `PY` to your interpreter if needed.

## A. Benchmark (deterministic)

```
python code/gen_dataset.py                       # train/val/test/test_hard splits
python code/gen_dataset.py --extra_split testbig # high-power split (200 sequences)
```

The generator is seeded (`--seed0`, default fixed); regenerating produces identical data. Output
defaults to `data/defect_ds`.

## B. Stage-1 detector (single-frame, four continuation rounds)

```
bash code/run_all_train.sh
```

This trains the `base` and `focus` variants across rounds 1–4; the flagship detector is
`runs/base_s1d/best.pt` (validation mAP50 0.228). Individual rounds are plain `train.py`
invocations (see the script for exact flags: batch 10, lr 1e-4, warmup 500, cosine with a
nonzero floor, fp32).

## C. Two-stage memory probes (development history)

```
bash code/run_final_stage2.sh    # stage-2 memory-on-frozen-recipe probe (see paper Sec. 3.3/6.2)
```

These probes document why the frozen-backbone route was rejected; they are not needed for the
main results.

## D. Replicates and ablations

```
bash code/run_sci_chain.sh       # seeds 2027/2028 + alignment / lifespan / noise / phase ablations
bash code/run_sci_chain_rest.sh  # remaining sweep items
```

## E. End-to-end flagship (the model all reported numbers use)

```
python -u code/train.py --stage 3 --variant base_t \
  --init runs/base_s1d/best.pt --out runs/e2e_s1 \
  --epochs 8 --steps_per_epoch 220 --seq_every 4 \
  --lr 5e-5 --warmup 100 --min_lr_ratio 0.1 --workers 2 --no_amp \
  --eval_every 2 --val_batches 40
```

(seed 42; ≈50 min on the reference GPU; the terminal checkpoint `ep7.pt` is the fixed analysis
checkpoint.) The token-supervision ablation retrains the identical recipe with `--mem_unsup`
(see `runs/e2e_unsup/`).

## F. Evaluation

```
python -u code/eval.py --ckpt runs/e2e_s1/ep7.pt --splits test,test_hard --video \
  --video_splits testbig,test,test_hard --motion gt --out runs/eval_e2e_ep7
```

Additional evaluations (each writes its own `runs/<name>/results.json`):

| Experiment | Script |
|---|---|
| Zero-learning wrapper + streams cache | `eval_persistence.py` |
| Drift (correlated) motion | `eval_drift.py` |
| Wrapper under phase motion / TFM+phase precision | `eval_wrapper_phase.py`, `eval_phase_pr.py` |
| Threshold × capacity sweep (inference only) | `eval_tauK_sweep.py` |
| Density control (low-threshold streams) | `collect_streams_lowthr.py`, `analyze_density_control.py` |
| ByteTrack-style tracker baseline | `eval_tracker_baseline.py` |
| YOLOv5s reference (and wrapper) | `eval_yolo_baseline.py`, `eval_yolo_persistence.py` |
| Texture-clutter / rotation probes | `eval_texture_robustness.py`, `eval_rotation.py` |
| Noise σ rows precision/recall | `eval_noise_pr.py` |
| Supervision ablation summary | `analyze_unsup_summary.py` |
| Throughput | `fps_stats.py`, `fps_stats2.py` |
| FP-persistence proxy | `fp_ghost.py` |

## G. Statistics and number audit

```
python code/analyze_tp.py runs/eval_e2e_ep7/results.json
python code/analyze_tp_discount.py
python code/analyze_paired.py
python code/analyze_byclass_practical.py
python code/audit_numbers.py      # recomputes every manuscript number from runs/
```

## External dependencies (not bundled)

- **YOLOv5s reference**: clone `ultralytics/yolov5` into `third_party/yolov5` (AGPL-3.0) and
  fine-tune with the standard recipe on the identical split (80 epochs); place the checkpoint at
  `yolo_runs/weights/best.pt`.
- **Real hull imagery** (fidelity check): imagery reproduced from the published literature; not
  redistributed. `make_fidelity_check.py` documents the measurement procedure (GLCM contrast and
  edge density on 256×256 patches).
