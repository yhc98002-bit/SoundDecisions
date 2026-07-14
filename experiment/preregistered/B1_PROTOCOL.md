# Arc-4 B-1 frozen protocol

Frozen on 2026-07-14 before reading any B-1 evaluation metric or running any B-1
probe. This protocol resolves implementation details left open by
`arc3_tierB_preregistration.md`; it does not change its scientific question,
thresholds, or decision tokens.

## Question and population

Test whether the final self-target class of each cfg=1.0 Phase-1 independent is
readable from MMAudio internals before the external class readout window.

- Model: MMAudio `small_16k`, full precision, video conditioning enabled.
- Kernel: cfg=1.0, `sqrt_down`, 20 Euler steps, 8.0 s audio, seed 0.
- Population: the 200 frozen `single_event` clips, 16 independents per clip.
- Progress grid: `0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90`.
- Trajectory identity: `<clip>__p1cfg1_ind<j>` with the existing CRC32-based
  `rng_for(0, clip, "ind", j)` lineage.
- Target: that trajectory's own measured final class, retained verbatim,
  including `abstain` as a class.

The frozen manifest is `data/manifests/phase1_manifest_frozen.json` (SHA256
`64a7a3d1a194edffc69506bf7baddc85e03a3ab102298f61782d3be0fe4a595b`).
Within the single-event population it assigns 126 clips to `probe_train` and 74
to `eval`, with no overlap or unassigned clip. No trajectory from an eval clip
may enter fitting, standardization, early stopping, or hyperparameter selection.

## Schema interpretation

Schema inspection was limited to array names, shapes, and dtypes. The surviving
pooled cache has `pooled: (12, 448) float16`. The per-token collector contract is:

- `token_mean: (12, 448) float16`;
- `token_mean_max: (12, 896) float16`;
- `tokens_sub: (12, 64, 448) float16`;
- `xattn_clip: (4, N_clip) float16`;
- `xattn_frac: (4,) float32`.

The old registration called the small model's layers "12 joint blocks". The
audited MMAudio architecture has 12 transformer blocks total: four joint blocks
followed by eight fused blocks. Therefore `token_mean_max` and `pooled` sweep all
12 block outputs, while `xattn_clip` sweeps the four joint blocks on which the
latent-query to clip-key attention submatrix exists. This is an architecture
schema correction made before evaluation.

The old completion manifest reports 200 collection units, but no per-token NPZ
payload survives in project storage. It is not evidence of data completeness;
Arc-4 recollects the payloads before probing.

## Feature and probe families

The confirmatory sweep contains exactly these combinations:

1. `pooled` block features with ridge.
2. `pooled` block features with the MLP below.
3. `token_mean_max` block features with ridge.
4. `token_mean_max` block features with the MLP below.
5. `xattn_clip` joint-block maps with ridge.
6. `xattn_clip` joint-block maps with the MLP below.

`token_mean`, `tokens_sub`, and `xattn_frac` are retained as audit fields but are
not additional searched families. In particular, no post-result choice between
flattened token residuals and token-mean-max is permitted.

Ridge is the existing one-vs-all least-squares classifier with intercept and
lambda 1.0. Every dimension is standardized using outer-training statistics
only; zero-variance scales are replaced by one.

The MLP has one 256-unit ReLU hidden layer, Adam learning rate 1e-3, L2 weight
decay 1e-3, batch size 64, at most 300 epochs, and seed 0. Early stopping uses
patience 15 and minimum validation-accuracy improvement 1e-4. Its inner
validation set is clip-grouped: sort the 126 outer-training clip IDs by
`SHA256("arc4-b1-inner-v1:" + clip)` and take the first 26 as validation. The
remaining 100 clips fit the model and its standardizer. The checkpoint with the
highest inner-validation accuracy is evaluated once on the untouched outer
eval clips. Ties keep the earlier epoch.

There is no tuning of lambda, width, seed, layer range, feature family,
abstention handling, or progress grid after evaluation.

## Completeness and integrity gate

Evaluation is forbidden until all of these checks pass:

- 200 completed per-clip journal units;
- exactly 25,600 per-token bundles (200 clips x 16 independents x 8 progress
  points), with no duplicate trajectory/progress key;
- exactly 25,600 matching pooled bundles and 3,200 final class labels;
- all required arrays finite, with consistent shapes and the architecture counts
  above;
- collection cfg, schedule, seed, model variant, clip ID, independent index, and
  progress value match this protocol;
- the frozen train/eval split is disjoint and complete.

Failure yields `B1_INCOMPLETE` with no readability token. Partial data are never
used to estimate accuracy.

## Estimation and uncertainty

For each family, probe type, eligible layer, and progress value, fit on the
frozen outer-training clips and report trajectory-level held-out accuracy on the
74 eval clips. Chance is the majority-class frequency among those eval
trajectories and is reported alongside accuracy.

For each fixed specification, form a 95% percentile confidence interval from
1,000 bootstrap draws with seed 0, resampling eval clips with replacement and
carrying all 16 trajectories of each sampled clip. The point estimate, not a CI
endpoint, drives the frozen decision. Per-progress winners and the global winner
are selected by held-out point accuracy; deterministic ties are broken in this
order: lower progress, `pooled` before `token_mean_max` before `xattn_clip`, ridge
before MLP, then lower layer index.

## Decision rule

At each progress value, take the best point accuracy over the six registered
family/probe combinations and their eligible layers. Define
`s_read_internal_class` as the earliest progress whose best accuracy is both at
least 0.70 and at least `chance + 0.15`.

- If `s_read_internal_class <= 0.45`, emit
  `CLASS_INTERNAL_READOUT_FOUND` and record the winning fixed specification.
- Otherwise, after all six combinations complete, emit `R2_CLASS_CONFIRMED`.
- If the completeness gate or any required family fails, emit only
  `B1_INCOMPLETE` and no scientific token.

All searched specifications, not only winners, remain in the machine-readable
result. Bootstrap unit, search multiplicity, self-target labels, and the
single-model scope must be stated in the report.

## Execution and outputs

Collection is resumable and journaled under `results/arc4_b1/`. The four workers
run on one node as independent TP1 replicas with physical GPUs 4-7 exposed by
exactly `CUDA_VISIBLE_DEVICES=4,5,6,7`; logical devices `cuda:0` through `cuda:3`
receive shards `0/4` through `3/4`. Each selected physical device must show at
least 70 GiB free immediately before launch. Set `FOLEY_CW_WEIGHTS_SOURCE=hf`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `PYTHONHASHSEED=0`; a missing
local weight is a hard failure, never a download.

The run manifest records node, physical GPU IDs and UUIDs, logical device, TP
width 1, replica count 4, command, git commit, protocol SHA256, config hashes,
seed, start/end time, journal path, artifact path, and deviations. Final B-1
outputs are `results/arc4_b1/b1_probe.json` and `results/arc4_b1/b1_probe.md`.
No B-2 or B-6 value may be inspected while executing this protocol; their raw
outputs remain quarantined.
