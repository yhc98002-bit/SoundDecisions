# Arc-4 B-1 frozen protocol V2

Frozen on 2026-07-15 before any B-1 completeness gate completed and before any
B-1 evaluation metric or prediction was produced or read.

## V1 disposition

`B1_PROTOCOL.md` is retained unchanged as audit history but is void for B-1
evaluation. It incorrectly treated `abstain` as a class, reported only raw
accuracy, omitted per-candidate predictions, and selected architectures on the
outer evaluation set. The V1 probe session was interrupted before its gate
completed: `gate.log` was zero bytes; `bundle_gate.json`, `b1_probe.json`,
`b1_probe.md`, `probe.log`, and the runner exit-code file did not exist. No V1
metric or prediction existed or was inspected.

V2 is append-only and supersedes V1 for all B-1 evaluation. It resolves the
explicit confident-only, balanced-accuracy, prediction-retention, nested
selection, float32, and streaming requirements without changing the model,
population, representation families, progress grid, or readout thresholds.

## Question and population

Test whether the final confident self-target class of each cfg=1.0 Phase-1
independent is readable from MMAudio internals before the external class
readout window.

- Model: MMAudio `small_16k`, full precision, video conditioning enabled.
- Kernel: cfg=1.0, `sqrt_down`, 20 Euler steps, 8.0 s audio, seed 0.
- Collection population: 200 frozen `single_event` clips, 16 independents per
  clip, including complete accounting for abstentions.
- Evaluation population: only trajectories whose retained final class label is
  not `abstain`. Abstaining rows are dropped before standardization, fitting,
  inner validation, candidate selection, outer evaluation, baselines, and
  bootstrap resampling.
- Progress grid: `0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90`.
- Trajectory identity: `<clip>__p1cfg1_ind<j>` with the existing CRC32-based
  `rng_for(0, clip, "ind", j)` lineage.

The frozen manifest is `data/manifests/phase1_manifest_frozen.json` (SHA256
`64a7a3d1a194edffc69506bf7baddc85e03a3ab102298f61782d3be0fe4a595b`).
Its outer split remains fixed at 126 `probe_train` clips and 74 `eval` clips.
No outer-eval row may enter standardization, fitting, early stopping, candidate
selection, or tie-breaking.

## Schema interpretation

The pooled cache has `pooled: (12, 448) float16`. The per-token collector has:

- `token_mean: (12, 448) float16`;
- `token_mean_max: (12, 896) float16`;
- `tokens_sub: (12, 64, 448) float16`;
- `xattn_clip: (4, 64) float16`;
- `xattn_frac: (4,) float32`.

The small model has 12 transformer blocks total: four joint blocks followed by
eight fused blocks. `pooled` and `token_mean_max` sweep all 12 block outputs;
`xattn_clip` sweeps the four joint blocks with latent-query to clip-key
attention. `token_mean`, `tokens_sub`, and `xattn_frac` are audit fields, not
additional searched families.

## Candidate families and computation

At each progress value, the registered candidate set is exactly:

1. `pooled` x 12 blocks x ridge;
2. `pooled` x 12 blocks x MLP;
3. `token_mean_max` x 12 blocks x ridge;
4. `token_mean_max` x 12 blocks x MLP;
5. `xattn_clip` x 4 joint blocks x ridge;
6. `xattn_clip` x 4 joint blocks x MLP.

Feature arrays are read and processed one progress/family block at a time; no
cross-family feature cube or all-candidate model set is retained. Float16 cache
values are converted to float32. Means, standard deviations, standardized
features, ridge targets, Gram matrices, solves, and predictions use float32.
Zero-variance scales are replaced by float32 one. No float64 feature or solve
path is permitted.

Ridge is one-vs-all least squares with intercept and lambda 1.0. The MLP has one
256-unit ReLU hidden layer, Adam learning rate 1e-3, L2 weight decay 1e-3,
batch size 64, at most 300 epochs, and seed 0.

## Nested candidate selection

The 126 outer-training clips have a fixed inner split: sort their IDs by
`SHA256("arc4-b1-inner-v1:" + clip)` and assign the first 26 to inner validation
and the remaining 100 to inner fit. The clip sets are defined before filtering;
only confident rows within each set are used.

For every candidate:

- Ridge fits and standardizes on the 100 inner-fit clips, then predicts the 26
  inner-validation clips.
- MLP fits and standardizes on the 100 inner-fit clips. The 26 inner-validation
  clips select the checkpoint with patience 15 and minimum balanced-accuracy
  improvement 1e-4. Ties retain the earlier epoch. That checkpoint's inner
  predictions are the candidate-selection predictions.

At each progress value, select one candidate by higher inner-validation
balanced accuracy, then higher inner-validation raw accuracy, then the fixed
tie order `pooled < token_mean_max < xattn_clip`, `ridge < MLP`, lower layer.
Outer-eval labels and predictions are not inputs to this selection.

After selection, evaluate exactly one specification at that progress value:

- The selected ridge is re-standardized and refit on all confident rows from
  the 126 outer-training clips, then predicts the outer-eval rows once.
- The selected MLP's inner-selected `best_epoch` is frozen. A fresh MLP and
  standardizer then fit all confident rows from the 126 outer-training clips
  for exactly that many epochs, with no early stopping or outer access, and
  predict outer eval once.

No outer-eval metric is computed for an unselected candidate. There is no
global outer-eval winner across progress values.

## Metrics, predictions, and uncertainty

For every inner candidate, retain one record per confident inner-validation
trajectory with `gen_id`, `clip`, `true_label`, and `predicted_label`, keyed by
progress, family, probe, and layer. Also retain its inner raw accuracy, balanced
accuracy (unweighted mean recall over true classes present in that split),
majority baseline, and raw margin over majority.

For each progress value's inner-selected candidate, retain the same fields for
every confident outer-eval trajectory. Report:

- raw accuracy;
- balanced accuracy;
- majority baseline, the outer confident-label mode frequency;
- `margin_over_majority = raw_accuracy - majority_baseline`;
- 95% CIs for raw and balanced accuracy from 1,000 bootstrap draws with seed 0.

The bootstrap unit is outer-eval clip. A sampled clip carries all of its
confident trajectories, and repeated clips repeat those rows. This preserves
variable confident-row counts after abstention filtering.

## Completeness and integrity gate

Evaluation is forbidden until all of these checks pass:

- 200 completed per-clip collection journals;
- exactly 25,600 valid per-token bundles and 25,600 matching pooled bundles;
- exactly 3,200 retained final class rows before abstention filtering;
- unique trajectory/progress keys, finite arrays, consistent shapes, and
  matching cfg/schedule/seed/model/clip/independent/progress metadata;
- disjoint and complete frozen 126/74 outer clip split;
- at least one confident row in inner fit, inner validation, and outer eval,
  with all confident/abstain counts reported by split.

Failure yields `B1_INCOMPLETE` with no readability token. Partial data are never
used to estimate a metric.

## Confirmatory decision

For each progress value, use only its inner-selected candidate's outer result.
Define `s_read_internal_class` as the earliest progress at which:

- outer balanced accuracy is at least 0.70; and
- outer raw margin over majority is at least 0.15.

If `s_read_internal_class <= 0.45`, emit `CLASS_INTERNAL_READOUT_FOUND` and the
selected specification. Otherwise, after all eight nested evaluations complete,
emit `R2_CLASS_CONFIRMED`. Gate or required-candidate failure emits only
`B1_INCOMPLETE` and no scientific token.

## Execution and outputs

Collection remains under `results/arc4_b1/`. Evaluation runs on the same node
after the full gate passes. GPU collection exposes exactly physical GPUs 4-7 as
`CUDA_VISIBLE_DEVICES=4,5,6,7`, uses four independent TP1 replicas, requires at
least 70 GiB free per selected GPU, uses local HF weights only, and sets
`PYTHONHASHSEED=0`.

The evaluator writes `results/arc4_b1/b1_probe_v2.json` and
`results/arc4_b1/b1_probe_v2.md`. V1 output names are never used. B-2 and B-6
raw outputs remain quarantined and may not be measured or summarized during
B-1 evaluation.
