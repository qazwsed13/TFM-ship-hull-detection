# Evaluation protocol (fixed in advance of the confirmatory runs)

This document states the analysis protocol under which all confirmatory numbers of the manuscript
were produced. It was fixed before the multi-seed runs; the released code and raw result files
implement it exactly.

## 1. Primary metric: TP stability

For every ground-truth object track and every consecutive frame pair `(t-1, t)` in which the
track's detection was a hit at `t-1`, TP stability counts whether the track is still a hit at `t`:

```
TP stability = P(hit_t | hit_{t-1})
```

Definitions used throughout:

- A *hit* is a prediction matched to the ground-truth box with IoU > 0.5 and identical class, at
  the operating confidence threshold `tau_eval = 0.3`.
- Ground-truth tracks are maintained by motion-compensated IoU chaining (linking threshold 0.4,
  same class): the previous box is shifted by the recorded inter-frame stride before linking.
- The metric conditions on the previous-frame hit and counts only true detections; it is not
  normalized by the number of predictions. Because recall and output density can still influence
  a conditional probability, the metric is always reported together with precision/recall, and a
  recall-matched / density-matched control plus a full threshold sweep are included
  (`runs/density_control/summary.json`).

## 2. The same-weights toggle

The memory ON/OFF contrast holds model weights fixed and toggles only the memory pathway (memory
state disabled vs. enabled). All headline comparisons are of this form.

## 3. Checkpoint and seeds

- Primary analysis uses the **fixed terminal checkpoint** (epoch 7) of the end-to-end schedule.
- Three end-to-end replicates (seeds 42, 2027, 2028) are initialized from a **common stage-1
  detector**; the seed varies only the end-to-end continuation stage.

## 4. Statistics

- **Primary test**: Fisher's exact test (two-sided) on the 2×2 hit-transition table, per test
  domain and pooled over the three test domains.
- **Cluster conservatism**: because hit/miss transitions cluster within tracks and sequences, the
  effective sample is additionally discounted by factors of 10 and 20 (all four transition cells
  floor-divided by the factor; two-sided test). See `analyze_tp_discount.py`.
- **Paired robustness checks** (secondary, post-hoc): common-support McNemar (exact binomial) and
  track-level Wilcoxon signed-rank. See `analyze_paired.py`.
- No multiple-comparison correction is applied across the three pre-specified domains; the
  protocol relies instead on the advance specification, replicate consistency and the
  conservative discounts.

## 5. Post-hoc probes (added during revision; always labeled as such)

The following were added at the request of reviewers and are reported as post-hoc:
zero-learning output-persistence wrapper (clean / noisy / drift / phase motion), ByteTrack-style
tracking baseline, YOLOv5s pretrained reference, texture-clutter probe, rotation probe,
admission-threshold × capacity sweep (inference only), token-supervision (no-matching) ablation,
and the recall-matched density control. All are implemented in the released scripts and their
raw outputs are in `runs/` as listed in the README table.

## 6. Detection-quality reporting

Frame-level detection quality is reported as COCO-style mAP50 / mAP50–95 / AP_S, under the same
data and thresholds for every system compared. Video-mode mAP, flicker (isolated single-frame
miss runs over tracks) and miss-run counts are secondary temporal metrics; throughput is measured
at batch size 1.
