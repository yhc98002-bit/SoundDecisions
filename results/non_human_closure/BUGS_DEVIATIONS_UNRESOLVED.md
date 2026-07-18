# Bugs, deviations, failed attempts, and unresolved issues

This ledger distinguishes engineering attempts from scientific results. No
failed process below was interpreted as negative scientific evidence, and no
threshold, taxonomy, probe family, target rule, or Material matching rule was
weakened after held-out inspection.

## Preserved engineering attempts and fixes

- A read-only full-bank inventory on the login node exhausted memory before a
  completion artifact. Four bounded workers later produced the canonical exact
  inventory.
- Class smoke v1 correctly failed before output because deterministic PyTorch
  required `CUBLAS_WORKSPACE_CONFIG`. The retry froze `:4096:8`; the traceback
  SHA-256 is `c48f6f83566c82fd8d80ca8212dfda8f0a7a4f764f5a6266505a29e7ec13a65a`.
- Class smoke v2 was quarantined because its normalized coarse posterior was
  not bitwise derived from the persisted fp32 coarse sums. Completion SHA-256:
  `3f2016b147d428e4e0a9831426f1500f1b286cd72757cefe013b03b0929d2426`.
  `c631a33` corrected lineage without changing taxonomy or abstention.
- The first Class merge rejected heterogeneous execution batch sizes as if
  they were a scientific invariant. `ecf930a` retained per-shard batch
  provenance while enforcing all scientific invariants.
- The first B-1 packet attempt encountered a non-JSON PyTorch `_CUuuid` and
  stopped without a completed unit. `5df15c7` serializes the canonical UUID.
- Two calibration replay attempts were interrupted at 30/32 and 31/32 units
  when the source-hash guard detected on-disk code edits. Neither has a root
  completion and neither contributed to tolerance calibration.
- Two held-out launch attempts failed before model load because
  `HF_HUB_DISABLE_XET=1` was omitted. Clip `1002` was not evaluated, and the
  frozen tolerance did not change.
- A serial full-feature merge was intentionally terminated before output
  because deep immutable-shard validation was unnecessarily sequential.
  `b2a8687` parallelized read-only validation while preserving order and hashes.
- Readout `merged_v1` mapped the ODE-final pattern to the wrong permitted
  conclusion sentence. `c276668` fixed only the label; `merged_v2` predictions
  and metrics are byte-identical to v1. A transient SSH UID lookup failure
  occurred before the successful v2 merge wrote output.
- The first post-hoc video-determined sensitivity implementation repeated full
  527-way posterior validation and was interrupted as impractically slow. A
  superseded asynchronous v1 output lacking the final CSV lineage binding is
  preserved at `failed_attempts/CLASS_VIDEO_DETERMINED_SENSITIVITY_v1.json`;
  it is not canonical evidence. The canonical create-only v2 report binds the
  merged posterior and the exact 6,528-cell CSV.
- Fixed-capacity MLP fits reached the frozen 120-iteration cap and emitted
  convergence warnings. The cap was not raised after results were inspected.
- The required Claude/Opus cross-model audit launched with a path-only prompt
  but returned `Not logged in` with zero input/output tokens. Prompt, raw JSON,
  and debug trace are preserved under `audit/`. This is an audit-process
  `ENGINEERING_FAILURE`, not a scientific failure or external review verdict.
- A multiline SSH wrapper used for the final create-only materialization smoke
  test returned without launching the remote commands and created no scientific
  output. Its log is preserved as `ENGINEERING_FAILURE`; an explicit remote
  directory creation followed by a single-line launch produced and validated
  the complete 37-output bundle.
- The first exact bundle-validator pass used a `1e-8` probability-sum tolerance,
  tighter than the float32 single-query softmax producer. A full read-only scan
  found maximum sum error `1.606e-7` across 113,212 vectors and none above
  `1e-6`. The validator now uses the outcome-independent bound
  `min(1e-6 producer gate, class_count × float32_epsilon)`, while also enforcing
  the 15-way width and `[0,1]` range; acceptance and corruption tests are
  retained. No prediction or scientific metric changed.

## Recorded deviations

- ModelScope mirrors were not present. The run used the existing hash-pinned
  local Hugging Face cache with `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1`, and `HF_HUB_DISABLE_XET=1`; no download occurred.
- Posterior shards used all safely available GPUs on both nodes. At feature and
  readout launch time all `an29` GPUs were occupied by normal neighboring jobs,
  so every safely free `an12` A800 (4–7) was used instead.
- Class shard 0 used batch 32 and colocated shards used batch 8. Batch size is
  execution provenance, not a scientific parameter; all invariant hashes match.
- Readout shard completions retain protocol, feature, target, projection, and
  implementation-freeze hashes but omit an executable source/git hash. The
  launch commit is retained in the progress log and Git history; this is a
  provenance limitation, not evidence of split leakage.
- The Material insufficiency artifact uses protocol SHA-256 `a9f2a846...`; the
  final protocol is `5c4fc402...`. Commit `86f18c5` added only the B-1 asset and
  offline-environment contract. Material rules are identical across revisions.

## Scientific and evidential limitations

- The historical `s_commit=0.34593` is a crossers-only mean of unsustained
  individual first crossings; the frozen B2 decision is an all-cell pooled
  sustained crossing. The registered `NOT_SUPPORTED` label is a frozen-rule
  result, not proof that individual early commitment vanished.
- Nine registered video-determined cases have `A_ind=1` and zero normalized
  gain in the all-cell pool. Their post-hoc exclusion moves the point estimate
  from `s=0.90` to `s=0.60` but still misses the frozen `s<=0.45` cutoff.
- The zero additive seed component is boundary-clipped after negative raw
  method-of-moments estimates. Video×seed interaction is large. Finite
  uncertainty in the 17-final video baseline is held fixed and not decomposed.
- Each WAV was measured once, so measurer repeatability remains `UNRESOLVED`.
- Probe targets are model-output-derived continuity proxies, not human semantic
  truth. Fork-majority target coverage is 82.7–89.3%; ODE-final coverage is
  70.6%. Missing targets are explicit and not imputed.
- Readout intervals are pointwise and unadjusted across 11 families and eight
  progress points. Pooled-MLP support occurs at `s=0.45` and `s=0.60` but is
  not sustained later. Fresh-video replication is required.
- The same-progress external baseline is a trained PANNs representation probe,
  not the historical direct preview-label readout. Its `s=0.45` information
  point is not the historical direct preview-label reliability/readout point
  near `s≈0.75`.
- No representation meets the frozen action-readout criterion; this result is
  `NOT_SUPPORTED`, not `UNRESOLVED`.
- Existing metadata supports only 3 complete Material videos / 96 cells. Valid
  2AFC, AUROC, margins, indifference, and a Material curve are therefore not
  measured. Material remains `INCOMPLETE_ARTIFACTS`; a distinct window is
  `UNRESOLVED`.
- Presence, Timing, event-centered Class/Material, Binding, B6, interventions,
  map scheduling, sealed confirmation, and a second backbone are `NOT_TESTED`.
- Cross-model audit independence remains unavailable until Claude CLI
  authentication is supplied. The completed Class, readout, and deliverable
  audits were independent agent passes but not a substitute external model.
