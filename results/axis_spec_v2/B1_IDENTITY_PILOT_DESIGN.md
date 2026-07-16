# B-1 feature-lineage identity pilot design

Status: **DRAFT FOR GOAL-1 PI REVIEW**. This document designs an engineering
identity pilot. It does not freeze a tolerance, authorize a GPU run, fit a
probe, inspect a target distribution, or support a scientific interpretation.
The pilot may run only after the Axis Specification v2 is signed and its
protocol is registered by amendment.

## 1. Starting evidence and question

The previous B-1 collection is `B1_INCOMPLETE`, not a negative internal-readout
result. Its binding gate stopped before probe fitting on the 31st bundle in the
sorted scan:

- bundle: `1002__p1cfg1_ind12__s0.75.npz`;
- whole-bundle relative L2: `0.0002585371085837806`;
- then-binding threshold: `0.0002`;
- labels were used only to validate IDs and the class universe;
- no probe metric, candidate prediction, or scientific token was produced.

One known implementation mismatch is operation order. The original pooled tap
evaluated the token reduction on the GPU and only then converted the result to
float16. The retap converted and cached tokens in memory as float16, converted
them back to float32, and took a CPU NumPy mean before writing the bundle.
Those are different numerical operations. In addition, the old artifacts came
from separately regenerated trajectories and separate velocity calls. The
operation-order mismatch is a proven confound, not an isolated explanation of
bundle 31. Comparing the old artifacts cannot establish same-forward trajectory
identity, regardless of a post-hoc tolerance.

The pilot asks one engineering question: **can all representations needed by
the internal-readout study be captured from the same model evaluation with
unambiguous IDs, and can equivalent capture paths be joined under a tolerance
derived from measured benign nondeterminism?**

It does not ask whether any axis is readable. Labels, fork outcomes, probe
predictions, and scientific thresholds are forbidden inputs to the pilot.

## 2. Pilot population and replay layout

### 2.1 Five-video set

The five videos are selected without outcomes:

1. Include clip `1002` so the exact historical failure case
   `1002__p1cfg1_ind12__s0.75` is a held-out identity check.
2. From the signed input manifest, remove `1002`, sort the remaining eligible
   clip IDs by `SHA256("b1-identity-pilot-v1:" + clip_id)`, and take the first
   four.
3. Use independent index `12` and the registered eight-point progress grid
   `0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90` for all five clips.

The four hash-selected videos form the tolerance-estimation subset. Clip
`1002`, including its `s=0.75` case, is held out until the tolerance amendment
has been written. No substitution based on a feature, label, or outcome is
allowed. A missing deterministic input makes the pilot `ENGINEERING_FAILURE`;
it is not replaced silently.

### 2.2 Canonical replay packets

For each of the 40 `(video, independent, progress)` cases, create one immutable
replay packet before cross-device comparison. A packet contains the exact
intermediate state `x_s`, model time `t`, every conditioning tensor consumed by
the network, and the complete generation/configuration identity. Packet tensor
bytes, dtype, shape, and SHA256 are canonicalized and hashed.

Every device replays the same packet. Independently regenerating a trajectory
on each GPU is not an identity comparison because upstream numerical drift
could change `x_s` or conditioning. Deterministic regeneration is used only to
create the canonical packet and is checked against the frozen seed/config
contract.

### 2.3 Nodes, devices, and repetitions

The intended topology is two eligible physical GPUs on `an12` and two on
`an29`, with three fresh process replays per packet and device. Preferred
physical IDs are `4` and `5`; any replacement must be in `4,5,6,7`, have at
least 70 GiB free immediately before launch, and be recorded before results are
read. Physical GPUs `0-3` are never eligible.

For every device, repeats 1 and 2 use `PYTHONHASHSEED=0`; repeat 3 uses
`PYTHONHASHSEED=1`. All use `FOLEY_CW_WEIGHTS_SOURCE=hf`, pinned local-cache
revisions, and offline/local-files-only loading. The weight-source decision is
recorded once in the signed pilot manifest and inherited by every unit.

Each replay is TP1 and one process. A launch guard records memory occupancy and
refuses an ineligible device. If a co-tenant expands, the worker completes its
current atomic unit if safe, checkpoints, and exits. No worker competes for
memory. The pilot does not span one model invocation across nodes.

Within each exact replay packet, the reducer forms all registered unordered
pairs of the 12 process/device replays. This exposes same-process-contract,
same-device, different-device, and different-node effects. Quantile uncertainty
is clustered by video and replay packet rather than treating layers or pairwise
comparisons as independent. Actual placement is an input to the reducer, not an
assumption.

## 3. Single-evaluation capture contract

The atomic unit is one instrumented velocity evaluation at one replay packet.
At cfg 1.0 the backend may execute conditional and empty-conditioning network
passes inside that evaluation. Both pass ordinals and roles are retained, the
expected pass count is asserted, and the conditional pass is the registered
probe representation. No second model evaluation may be used to produce a
field that is claimed to share the same forward lineage.

All hooks write through one capture coordinator. Every block event carries the
same `capture_nonce`, evaluation ordinal, pass role, stable block ID, and token
stream role. Hook call counts and order must match the registered architecture.

### 3.1 Required tensor fields

For each atomic unit, persist the following before any reduction or storage
quantization unless a different dtype is explicitly named:

- `state_x_s_fp32`: the replayed intermediate latent;
- `device_state_x_s`: the actual latent after host-to-device transfer and dtype
  conversion, exactly as passed to the instrumented velocity evaluation;
- `device_time_tensor`: the actual broadcast time tensor passed to
  `predict_flow`, including dtype, shape, and canonical bytes;
- `velocity_fp32`: the velocity returned by this instrumented evaluation;
- `tweedie_latent_fp32`: `x_s + (1 - t) * velocity`, with the exact operation
  and accumulation dtype recorded;
- `block_tokens_fp32`: latent/audio token activations for every block admitted
  by the registered probe family, keyed by architecture-native block ID and an
  enumerated post-block hook site (`joint_output_latent` or
  `fused_output_latent`); persist the canonical hash of the exact native
  post-block parent tensor used by both pooled capture paths;
- `pooled_original_prequant_fp32`: the original tap operation, evaluated as
  `latent.detach().mean(dim=1)[0].float()` on the GPU and captured before the
  host conversion and float16 storage step;
- `pooled_repaired_prequant_fp32`: a second capture path applying that exact
  expression to the same native activation in the same pass;
- `token_mean_fp32`: mean of `block_tokens_fp32`, retained as a distinct
  representation with its own reduction contract;
- `block_tokens_quantized`: the exact stored/quantized token tensor, with
  quantizer, source dtype, stored dtype, rounding mode, scale/zero-point where
  applicable, and saturation count;
- `block_tokens_dequantized_fp32` and `mean_after_quantization_fp32`, for
  storage-distortion diagnosis only;
- `joint_attention_latent_output_fp32`: the actual latent-query attention
  output for every selected joint block before its output projection;
- `xattn_clip_map_fp32`: the latent-query to clip-key probability block when it
  can be captured from the same Q/K tensors and pass, plus the exact scaling,
  mask, softmax dtype, and reduction contract;
- `xattn_clip_summary_fp32`: the registered compact reduction of that map;
- `conditioning_tokens_fp32`: all network-consumed conditional inputs,
  including clip/video tokens, text tokens, global conditioning, extended
  conditioning, and any Synchformer-derived inputs, each with a semantic role;
- `empty_conditioning_tokens_fp32`: the corresponding empty-branch inputs when
  the cfg implementation executes that branch;
- `external_preview_wav`: the decode of this evaluation's Tweedie latent, with
  sample rate and waveform SHA256;
- `external_preview_representation_fp32`: the exact representation consumed by
  the frozen external-readout measurer, with measurer name and model revision.

If the production attention kernel does not expose probabilities, the pilot
must capture its actual Q, K, V operands and actual latent attention output in
the same pass. An explicitly recomputed softmax map may be stored as a derived
diagnostic, but it cannot be described as the model's actual attention map or
used as an identity reference without a separate equivalence test.

The pilot stores full token and attention fields locally because it has only
five videos. Full recollection may retain a preregistered bounded subset or
lossless sufficient representation only after the pilot demonstrates that the
reduction is correct. No architecture or layer selection may use scientific
outcomes.

### 3.2 Required identity and provenance fields

Every bundle and journal records at least:

- pilot, replay-packet, attempt, and `capture_nonce` IDs;
- video ID and source SHA256;
- generation ID, independent index, base seed, seed derivation rule, progress,
  model time, cfg, schedule, alpha, step count, and duration;
- model variant, exact weights revision/snapshot, weights SHA256 where
  available, and the signed weights-source decision;
- code commit, dirty-worktree status, collector/schema/protocol hashes, and
  dependency-lock hash;
- node, physical GPU ID, visible-device mapping, GPU model/UUID, driver, CUDA,
  cuDNN, PyTorch, NumPy, dtype/autocast/TF32/determinism settings, and
  `PYTHONHASHSEED`;
- ordered block inventory, pass count, pass roles, hook order, enumerated hook
  sites/tensor stages, parent-native-tensor hashes, and tensor names;
- for every tensor: semantic role, dtype, shape, byte count, finite-value check,
  canonical-byte SHA256, and parent tensor/capture nonce;
- replay-packet SHA256, output-bundle SHA256, journal SHA256, start/end times,
  exit status, peak allocated memory, and NFE accounting.

The external-preview waveform and large tensors remain in immutable local run
storage. Git receives schemas, manifests, hashes, reducers, tests, and small
derived engineering evidence only.

## 4. Valid and invalid comparisons

### 4.1 Gate comparisons

Only numerically equivalent operations enter the identity gate:

1. `pooled_original_prequant_fp32` versus
   `pooled_repaired_prequant_fp32`, block by block, from the same native tensor,
   pass, and capture nonce.
2. In-memory tensors versus atomic readback of the same stored tensor, which is
   bitwise and hash exact.
3. In-memory quantized tokens versus readback quantized tokens, also bitwise
   exact.
4. Stored Tweedie latent versus a reducer recomputation using the registered
   operand dtype and operation order.
5. Stored cross-attention summary versus a reducer applying the registered
   reduction to the stored same-pass map.
6. Corresponding outputs from exact replay packets across process, GPU, and
   node replicas, grouped by tensor family and block.

For numerical comparisons, retain relative L2, absolute L2, maximum absolute
error, cosine in float64 for diagnosis, and exact-equality fraction. Relative
L2 is computed in float64 as
`||a-b||_2 / max(||reference||_2, 1e-12)` over the complete registered tensor.
Small-denominator per-layer ratios are diagnostic and cannot silently replace
the registered complete-tensor statistic.

### 4.2 Comparisons forbidden as identity evidence

The following may be reported side by side as storage diagnostics but never
enter the lineage pass/fail decision:

- `mean(float16(tokens))` versus `float16(mean(native_tokens))`;
- mean-after-quantization versus any pre-quantization pooled reference;
- a recomputed attention map versus a fused kernel output with different
  accumulation or softmax semantics;
- independently regenerated states or conditioning tensors whose hashes differ;
- different progress values, trajectories, branches, blocks, or model calls;
- a decoded or embedded preview versus an internal tensor.

This rule removes the known operation-order confound. Same-forward capture and
device-input hashes separately test the trajectory/velocity-call confound in
the historical B-1 failure.

## 5. Tolerance estimation and freeze

No numerical tolerance is asserted in Goal 1. Goal 2 derives one from the four
hash-selected calibration videos before opening clip `1002`.

For each gate comparator and tensor family:

1. Form benign-delta distributions from exact-packet replays, stratified by
   same GPU, different GPU on one node, and different node. Use all registered
   unordered replay pairs, but retain video and packet cluster IDs.
2. Use the complete registered tensor as the primary unit. Layerwise values
   diagnose localization but do not create a more permissive denominator.
3. Compute `q_0.999`, the empirical 99.9th percentile of relative L2, with the
   quantile convention and sample count recorded.
4. Set the proposed gate tolerance to `tau = q_0.999 * F`, where the proposed
   safety factor is `F = 2.0`. The factor provides one multiplicative margin
   beyond the observed benign tail without importing the failed historical
   threshold. If all benign deltas are bitwise zero, the comparator remains
   exact; no arbitrary epsilon floor is added.
5. Report video/packet-cluster bootstrap uncertainty on `q_0.999` and
   GPU/node-stratified maxima. Pairwise and layerwise observations are never
   represented as independent experimental units. A
   multimodal distribution, a clear node/device effect not covered by the same
   process contract, nonfinite values, or too few valid comparisons prevents a
   tolerance freeze. The response is investigation and rerun, not widening.

Before reading the held-out clip, append an amendment that records the signed
Axis Specification hash, pilot protocol and code hashes, input-packet manifest,
sample counts, quantile method, every `q_0.999`, `F`, resulting `tau`, device
strata, and justification. Register its SHA256. That amendment is the only
authority for the held-out and full-recollection gates. It cannot be edited
after clip `1002` is opened.

The held-out test then applies the frozen tolerances to all clip-`1002` progress
values and explicitly reports the historical `ind12, s=0.75` case. The old
`2e-4` value is shown only as audit context.

## 6. Completeness, join, and resume gates

### 6.1 Immutable write protocol

Each `(packet, node, physical GPU, repeat)` is one atomic unit. A worker writes
to an attempt-specific temporary directory, fsyncs payloads, writes hashes and
the completion journal last, then atomically renames the unit. Existing units
are never overwritten. Failed attempts remain audit evidence.

Resume behavior is fail-closed:

- validate protocol, packet, model, code, environment, placement, tensor
  inventory, dtype, shape, finite checks, and hashes before accepting a unit;
- accept a complete matching unit without rerunning it;
- write a new attempt ID for an absent, partial, or corrupt unit;
- never mix attempts in the reducer;
- select the first fully valid attempt by a registered deterministic rule;
- reject duplicate successful attempts whose payload hashes disagree.

A forced interruption test must stop workers after payload creation but before
the journal, then prove resume neither accepts the partial unit nor overwrites
it. A corruption test flips one payload byte and must fail validation before
reduction.

### 6.2 Deterministic reducer

The reducer validates the exact Cartesian product of 40 packets, registered
eligible devices, and three repeats. It verifies packet-byte identity across
all replicas, unique capture nonces, hook counts/order, conditional/empty pass
roles, all tensor parent links, and every payload hash before computing a
delta. Missing, extra, mismatched, nonfinite, or ambiguous records stop the
gate. A partial corpus never yields a tolerance.

The calibration reducer is physically unable to open clip `1002`. The held-out
reducer requires the registered tolerance-amendment hash and is physically
unable to rewrite it. Neither reducer imports labels or probe code.

## 7. Decision rules

### 7.1 `PASS`

The identity pilot is `PASS` only if all of the following hold:

- all expected units and fields validate, including exact replay-packet hashes;
- hook counts, pass roles, block order, tensor parentage, and IDs are exact;
- all storage readbacks and hashes are exact;
- calibration distributions support finite, stable, preregistered tolerances
  without unexplained node/GPU modes;
- the tolerance amendment is registered before held-out access;
- every held-out same-operation comparison passes, including
  `1002__p1cfg1_ind12__s0.75`;
- the forced-interruption, partial-shard, duplicate, and corruption tests pass;
- no scientific label, metric, prediction, or token was read or emitted.

A pass certifies the collector's feature lineage only. It says nothing about
whether internal states contain axis information.

### 7.2 `ENGINEERING_FAILURE`

Emit `ENGINEERING_FAILURE`, with exact localization and no scientific token,
for any of these conditions:

- the frozen video/seed/config/model trajectory cannot be regenerated;
- exact replay packets differ across replicas or cannot be serialized safely;
- a required representation cannot be captured in the same evaluation;
- hook counts/order or conditional-pass identity are ambiguous;
- equivalent same-pass operations exceed the frozen held-out tolerance;
- benign deltas require outcome-aware, bundle-specific, layer-specific, or
  repeatedly widened thresholds;
- device/node effects make one registered collection contract nonstationary;
- any required tensor, dtype, shape, hash, environment field, or shard is
  missing, corrupt, nonfinite, or inconsistent;
- resume or deterministic reduction accepts partial or conflicting evidence.

Engineering failure ends Stage D. No full collection and no internal-readout
experiment may launch. The report names the failing packet, block, pass,
representation, node/device, hashes, metrics, and cheapest repair.

## 8. Historical bundles and Goal-2 recollection decision

The default is a new collection under the unified same-pass schema. The old
25,600 retap bundles do not contain the required pre-quantization reference,
and their token mean was derived after float16 token storage. Therefore their
current schema cannot prove an equivalent-operation join.

The old bundles may be called "trivially certifiable" only if the identity
pilot reveals already-persisted, hash-linked native tensors and an exact
pre-quantization reference that were absent from the audited inventory. A
small numerical delta from a non-equivalent mean-after-quantization path is not
certification. Optional comparison with the historical path is forensic only
and cannot change `PASS`, a tolerance, or the recollection decision.

After pilot `PASS`, recollect the Stage-E feature population with the unified
collector and its immutable schema. Full recollection may begin only after a
five-video dry run passes the frozen identity gate and the crash/resume tests.
It records no labels during feature capture. Target joins and probe fitting are
separate downstream gates under the signed axis protocol.

If the pilot is `ENGINEERING_FAILURE`, preserve all attempted artifacts and
write `FEATURE_LINEAGE_REPORT.{json,md}` with that status. Do not fall back to
the historical cache, relax the tolerance, or interpret the failure
scientifically.

## 9. Goal-2 engineering outputs

The identity workstream produces:

- a signed pilot protocol and amendment registration;
- canonical replay-packet manifest and per-packet hashes;
- immutable per-device unit journals and completion manifests;
- a machine-readable tensor schema;
- calibration-only delta distributions and tolerance amendment;
- held-out identity-gate report with the bundle-31 case explicit;
- resume, corruption, no-data, and partial-shard test logs;
- `FEATURE_LINEAGE_REPORT.{json,md}` with `PASS` or
  `ENGINEERING_FAILURE` and no scientific token;
- exact launch, resume, validation, reduction, and reproduction commands;
- small Git-tracked manifests/checksums only, with large tensors kept in local
  immutable run storage.

This workstream stops after its engineering verdict. Probe selection and
internal-readout evaluation are separately authorized Stage-E work and may
start only after a lineage `PASS`.
