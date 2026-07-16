# Cohort and Goal-2 compute plan

Status: **GOAL-1 DRAFT - NOT FROZEN, NOT AUTHORIZED FOR EXECUTION**

Prepared on 2026-07-16 (Asia/Shanghai) from branch
`axis-spec-v2-goal1` at base `f793878`. This document designs cohorts and
estimates Goal-2 work. It does not release quarantined audio, ratify an axis
definition, select a threshold, launch a job, or inspect a B2/B6 measurement
outcome.

## 1. Evidence basis

The following are inventory facts, not proposed scientific results. GPU-track
files were read from the local `arc4-gpu` worktree at commit
`92b2f462bb3c69e82030433fec24b28fd5843352`; their small manifests should be
carried into the Goal-2 provenance graph by commit and SHA256, not copied over
the raw evidence.

| Evidence | SHA256 | What it establishes | What it does not establish |
|---|---|---|---|
| `arc4-gpu:results/arc4_quarantine/QUARANTINE_MANIFEST.json` | `200900e998ba30aa8592d9213b48c3c3f7ff69817017959e1393e517d76cd2b5` | B2 has 17 seeds, 48 clips, 8 progress points, 12 forks per cell, 79,152 retained WAVs, and completed cardinality/metadata/hash validation; B6 has 128 pairs at each cfg. | No B2/B6 measurement or outcome was inspected. |
| `arc4-gpu:results/arc4_quarantine/b2/generation_manifest.json` | `5c3a334ecfcfb3e91504354c14c8e8dbae71b3bade088b21bec26fb06fd68ed3` | The first five-seed arm fixes the 48 clip IDs, cfg 4.5, `sqrt_down`, alpha 0.8, the eight-point s-grid, and K=12. Extension manifests add seeds 5-16 without changing the clip set or cell design. | It does not provide event-level anchors, semantic controls, preview representations, or v2 measurements. |
| `arc4-gpu:results/arc4_quarantine/CAPACITY_LEDGER.md` | `15d9c9159bc4c178956e173d82afb2119b029d191a9f6f902bc82a349fd39000` | B2 raw generation used about 11.6 A800 device-hours in total; the prior B-1 collection used 31m29s wall time on four A800s. | These timings do not benchmark the new measurers, full-final regeneration, or probe training. |
| `arc4-gpu:results/arc4_quarantine/RETAG_ASSESSMENT.md` | `b6ade04670ab1e1026b8f7d29e28a3035336f5db16df51fa18b14aab0e1c1600` | Historical final WAVs and categorical posterior vectors were not retained; old categorical rows cannot be retagged from labels alone. | It does not prevent tagging the retained B2 WAVs or deterministically regenerating a journaled final. |
| `arc4-gpu:results/arc4_b1/B1_CLOSEOUT.md` | `0a7ffb48a78ce7f925506b69858aab0f88b2acdabe018fbb4af0c65187932012` | B-1 is `B1_INCOMPLETE`; the sorted bundle-31 join failed at relative L2 `2.585371085837806e-4` against `2e-4`, before probe fitting. | It is no evidence for or against internal class readability. |
| `results/arc4_wpA2/FLAGS.json` | `23d1528a6bf3d05d546d9a76a69ca800c1206bc948ee7e7611f80affd69f823a` | A2-4 lacks Phase-2 candidate, same-video reference, and matched other-video embeddings; cached rows retain only own-target scalar cosine. | It does not justify estimating the missing 2AFC result from legacy cosine. |
| `data/manifests/phase1_manifest_frozen.json` | `64a7a3d1a194edffc69506bf7baddc85e03a3ab102298f61782d3be0fe4a595b` | The historical source pool contains the 200 single-event clip IDs from which B2 was hash-selected. | Its clip-level axis summaries are not v2 event annotations. |
| `data/manifests/two_event_manifest.json` | `9331f384e51c6cdfbb559e5c11e24ceb5ab9cf464c4636825247fe5fddf28ea0` | There are 60 candidate clip IDs from the FoleyBench Multi-source+Discrete pool for binding. | It contains no event pair identities, two anchor intervals, assignment labels, or curation verdicts. It is a candidate list, not a ready binding cohort. |
| Node-only `data/FoleyBench/clips_index.csv` | `695b5a93e3d52ea8fcb406a95aa7b6891c60f09fff93f6299f5359e739e5216f` | All 48 B2 clips join to retained metadata as Single-source+Discrete and span 33 UCS categories. | UCS breadth alone does not prove class, material, timing, loudness, anchor-quality, or scene balance. |

The B2 WAV directories contain only base finals and fork finals. A sampled cell
journal lists 12 `role=fork` WAVs and no external-preview representation. Thus
banked audio can supply v2 final/fork targets wherever a measurer accepts WAV
input, but same-progress external previews and internal states require a
deterministic trajectory replay.

## 2. Cohort roles and leakage boundaries

Three roles stay disjoint by source-video/scene group, not merely by row:

1. **Development cohort:** may be used to fit fusion rules, select abstention or
   indifference procedures, choose matching tolerances, debug measurers, and
   set model-selection rules. It may never contribute a confirmatory estimate.
2. **Banked B2 pilot cohort:** the existing 48 clips and 17 seeds. It is neither
   a threshold-development set nor a confirmatory set. After signed release it
   supports exploratory multi-seed variance/transition measurement under the
   already-frozen v2 procedures.
3. **Reserved confirmatory cohort:** manifest-only in this phase. Its identities
   are frozen and sealed before development outcomes are inspected. Goal 2 does
   not measure it and does not launch the prohibited 200-400-video confirmatory
   study.

All generations, seeds, forks, progress points, cfgs, negative references, and
representations for one video remain in one role and one split. Near-duplicate
videos or clips sharing a source video/scene are grouped before splitting.
Material negatives and binding event pairs inherit the role of every source
clip; a confirmatory clip can never be a development negative.

## 3. Proposed development cohort

The proposed development cohort has 120 unique video groups. This is a design
quota for PI review, not a frozen sample size. The rows below are mutually
exclusive primary strata; diversity constraints apply across all rows.

| Primary stratum | Proposed n | Required evidence before admission | Purpose |
|---|---:|---|---|
| Clear single-event positive | 30 | One specified visible event, an anchor interval with provenance/uncertainty, and confident corresponding audio evidence | Positive support, timing tolerance, class/posterior and material sanity |
| Weak event | 20 | Specified low-SNR, brief, partly occluded, or anchor-uncertain event; uncertainty reason retained | Abstention, undefined timing, robustness, and coverage behavior |
| Event-absent or silent control | 20 | Ten visually event-absent/background-only cases and ten deterministic silence replacements of visually anchored positives, stored as separate control types | The first subtype tests prerequisite/undefined handling; the second tests the Presence null. Synthetic silence is never used to estimate natural prevalence |
| Hard unrelated-background negative | 20 | Non-target audio energy near the target anchor, matched on gross loudness/duration but independently judged unrelated to the visible event | Proves Presence is not an energy detector; provides hard material/reference exclusions |
| Separable two-event clip | 30 | Two event IDs, two non-overlapping or adjudicably ordered anchor intervals, both event-presence states, and assignment evidence | Binding presence/order/identity decomposition |

Required cross-stratification is coarse class, material family, early/middle/late
anchor location, anchor-width band, loudness band, scene type, repeated versus
transient event, and source-video group. No stratum is filled by reading a v2
model score. Candidate admission is based on source metadata plus blinded human
or frozen sidecar annotation. Missing or ambiguous annotations remain explicit;
they are not filled with model predictions.

The two-event component should be selected from the existing 60-ID candidate
manifest only after event-pair curation. A deterministic, metadata-only split
assigns 30 usable candidates to development and reserves up to 30 for the
confirmatory manifest. If curation removes candidates, replacements must be
selected from the remaining FoleyBench Multi-source+Discrete pool using the
same frozen rule before any generated-audio outcome is viewed. Counts are not
silently backfilled from the other role.

## 4. Reserved confirmatory cohort

Reserve a second 120-video-group cohort with the same five primary quotas and
cross-strata as the development cohort. The confirmatory manifest records only
selection provenance, group IDs, intended strata, and immutable hashes during
Goal 2. Semantic-label collection for this reserve belongs only to the later
independent confirmatory phase; it is not authorized in Goal 2. Those labels
remain inaccessible to threshold selection, architecture selection, negative
mining, and the banked-pilot analysis.

Before reservation, exclude:

- every B2 clip and source-video/scene near-duplicate;
- every development clip and all of its reference/negative partners;
- the 30 historical anchor-check clips if their labels were used to change a
  v2 procedure;
- every clip used in the B-1 identity pilot or probe development; and
- any clip whose semantic status was chosen after seeing a v2 score.

This reserved set is for the later independent confirmatory phase only. Its
existence does not authorize measurement in Goal 2.

## 5. How the banked B2 cohort maps to v2

Verified mapping:

- 48 clip IDs were selected without a measurement outcome by an ascending
  SHA256 rule over the frozen 200-clip single-event pool.
- The retained metadata classifies all 48 as Single-source+Discrete and spans
  33 UCS categories. None is in the 60-ID two-event manifest.
- Only four B2 IDs occur in the historical 30-clip manual anchor-check list.
  Therefore the old check cannot stand in for event-level anchor validation on
  B2.
- The bank contains 17 base seeds, eight s-points, K=12 forks, 816 base WAVs,
  and 78,336 fork WAVs. It exceeds the requested seed replication but remains a
  48-video pilot.

Role under v2: after sign-off and release, B2 is the primary exploratory
multi-seed cohort for commitment targets and variance decomposition. It is not
used to choose thresholds, fusion weights, matching rules, probe capacity, or
confirmatory claims.

Required supplements before B2 can answer the intended questions:

1. Curate one or more event records per B2 clip with stable event IDs, visible
   anchor intervals, uncertainty, and source provenance. The four historical
   check overlaps do not waive this requirement.
2. Freeze material positive/reference IDs and same-class, similar-timing,
   similar-loudness, different-material negatives without using a measured
   margin. Persist the matching covariates and relaxation path.
3. Add development-only absent/silent and unrelated-background controls; the
   B2 selection contract does not supply them.
4. Use the separately curated 60-ID candidate manifest for Binding. B2 contains
   no designed two-event coverage.
5. Deterministically replay the 816 base trajectories only for representations
   not banked as WAVs: same-progress external previews and lineage-valid
   internal features. Fork target audio must be measured in place, not
   regenerated.
6. Generate a second fork-noise strength or event-diverse supplement only if a
   frozen precondition says the banked alpha/design cannot identify a required
   variance component. The reason and delta require an amendment.

## 6. B6 asset boundary

B6 contributes 128 donor-not-source pairs per cfg at cfg 1.0 and 4.5, with
atomic WAVs. It is recorded in the asset matrix so later work can apply the
frozen v2 Presence, Timing, Class, Material, and Binding output schemas to each
source/donor/swap event. It remains quarantined here: Goal 1 does not inspect
its audio, Goal 2 does not analyze or measure it, and no causal or mechanism
claim may use it. A later causal-phase amendment must explicitly release it.

## 7. Goal-2 dependency order

Nothing below starts until both PIs sign the Axis Specification v2 freeze.

1. **Freeze and release ledger:** hash the canonical spec, schemas, semantic
   edge-case tests, development-selection rule, confirmatory-reserve rule, and
   negative-construction manifests. Append the dual-PI sign-off and a B2
   measurement-only quarantine release amendment. Keep B6 closed.
2. **Priority measurement repair:** implement and calibrate Material 2AFC and
   Class posterior persistence first. Resolve A2-4 by replaying the existing
   Phase-2 preview path and persisting candidate, positive, and frozen-negative
   embeddings. Regenerate only journaled legacy finals that need selectable
   posteriors.
3. **B-1 identity pilot:** run the five-video identity pilot on both nodes and
   multiple GPUs. Set a join tolerance only from measured benign
   nondeterminism and record it by amendment. If it fails, emit
   `ENGINEERING_FAILURE`; do not fit a probe.
4. **Parallel axis calibration:** complete Presence/Timing/Binding development
   and semantic gates while Class/Material jobs run. CPU joins, schemas, and
   tests overlap independent GPU inference.
5. **Day-7 calibration checkpoint:** publish PASS/UNRESOLVED/INVALID_MEASUREMENT
   per axis. Only PASS axes enter pilot measurement or internal readout.
6. **Banked multi-seed measurement:** measure all 79,152 B2 WAVs in immutable
   shards for eligible axes; reduce only after shard validation. Replay the 816
   base trajectories for external previews and internal states, preserving the
   original seed/config/progress IDs.
7. **Lineage-valid recollection and probes:** after the identity gate passes,
   recollect features rather than salvaging the gated cache. Train the bounded,
   development-selected probe family and retain candidate predictions. ODE
   final and fork-majority targets remain separate.
8. **Conditional supplements:** generate the curated two-event binding pilot or
   a second alpha only after its axis gate passes and its amendment/manifest is
   frozen. No confirmatory study begins.
9. **Closeout:** three-process reproduction, numbers index, immutable test logs,
   reports, and a decision memo by 2026-08-07.

## 8. Provisional compute budget

These are planning ranges, not allocations. They are A800 **device-hours**; four
GPUs running for one wall-clock hour consume four device-hours. The only local
throughput anchor is prior raw B2 generation: about 11.6 device-hours for
79,152 WAV artifacts. Measurement throughput and pure-final generation have
not been benchmarked, so the ranges include startup and validation but can be
revised after a 1% throughput-only smoke test run only after dual-PI sign-off.
A throughput revision changes the operations ledger, not a scientific
threshold or cohort.

| Work package | Reuse / compute path | Provisional A800 device-hours | CPU work that overlaps |
|---|---|---:|---|
| Development measurer calibration and robustness | Original/development audio plus small, frozen synthetic controls; no B2 outcome for tuning | 4-12 | 40-100 CPU-hours for schemas, perturbations, joins, bootstrap, and blinded package assembly |
| A2-4 Phase-2 material continuity repair | Deterministic replay of 200 clips x 8 progress points; persist candidate/positive/negative embeddings | 2-6 | Negative-manifest validation and 2AFC/bootstrap reducer |
| Legacy Class posterior preservation | Deterministically regenerate only journaled finals whose audio/posterior is absent; run pinned tagger and store full 527-way/coarse posterior | 4-12 | Journal-ID audit, abstention development, exact legacy reconstruction |
| B2 commitment-target measurement | No generation: batch v2 measurers over 79,152 retained WAVs | 8-24 | Audio integrity checks, per-event joins, variance-model preparation |
| B2 external-readout representations | Deterministic replay of 816 base trajectories at eight s-points because previews were not banked | 3-10 | Preview/reference joins and calibration summaries |
| B-1 five-video identity pilot | New lineage bundle, repeated across an12/an29 and multiple GPUs | 0.5-1.5 | Delta distribution, tolerance amendment, join/resume tests |
| B2 lineage-valid feature recollection | Deterministic replay/taps for 816 base trajectories after identity PASS | 2-5 | Hash validation, immutable-shard reducer, target joins |
| Bounded internal-readout fitting/evaluation | Eligible axes only; development-selected fixed family | 2-8 | Split validation, bootstrap CIs, prediction completeness |
| Two-event Binding supplement | New generation only after Binding calibration PASS; proposed 60 clips x 5 seeds gives 300 base finals plus 60 x 5 x 8 x 12 = 28,800 forks, then v2 measurement | 6-12 | Event-pair joins and assignment scoring |

All-axis planning envelope: approximately **32-91 A800 device-hours**. The lower
bound assumes fused batched inference, few eligible axes, and no failed-shard
reruns. The upper bound assumes all axes pass, separate model passes, the
Binding supplement runs, and deterministic replays are slower than raw-fork
generation. CPU and human annotation/review time are not included in this
device-hour total. Do not convert the envelope into a promise before the 1%
smoke benchmark and occupancy check.

The priority path is Material 2AFC, Class posterior preservation, the B-1
identity/recollection chain, and banked-cohort Class measurement. Presence,
Timing, and Binding development runs in parallel but cannot delay that path.

## 9. Operations and overlap

- At every launch, query actual occupancy. Physical GPUs 0-3 were resident at
  the last snapshot and remain prohibited for this phase; a low-utilization
  reading does not release them. Among the permitted devices, expose only GPUs
  with at least 70 GB free; yield and checkpoint if a co-tenant expands.
- Use TP1 replicas for these small-model/measurer and MMAudio jobs. Keep each
  job within one node; shard independent jobs over free devices on an12/an29.
- Use the pinned local Hugging Face cache as the recorded weights source until
  the ModelScope mirror is populated. No silent downloads.
- Each node writes immutable per-shard journals and completion manifests to its
  own RunStore root. A CUDA-hidden CPU reducer starts only after hashes,
  cardinality, IDs, dtypes, and shapes validate.
- While GPUs perform independent replay/inference, CPUs build schemas, check
  semantic edge cases, validate negative/reference joins, run bootstraps, and
  assemble human-ready packages. Do not hold a GPU allocation for CPU-only
  reduction.
- Every launch records node, physical/logical GPU IDs, TP width, replica count,
  command, branch commit, config/spec/input hashes, weights revision/source,
  seed policy, output root, and deviations. Resume is append-only and
  idempotent.

## 10. Schedule against PI anchors

| Date / relative day | Required state |
|---|---|
| 2026-07-16 to 2026-07-20 | Goal-1 audit/spec/cohort/identity/compute drafts delivered; no Goal-2 execution |
| By 2026-07-22 | Target dual-PI review and sign-off; if not signed, Goal 2 remains blocked and the later dates are re-risked |
| Goal-2 days 0-2 | Release amendment, A2-4 replay, Class posterior path, and B-1 identity pilot start |
| Goal-2 days 2-7 | Per-axis calibration gates; priority-path work continues; publish the mid-goal checkpoint around day 7 |
| Goal-2 days 5-12 | Eligible B2 WAV measurement, external preview replay, lineage-valid recollection |
| Goal-2 days 8-16 | Internal probes and any gate-authorized two-event supplement; deterministic reproduction and closeout overlap |
| By 2026-08-07 | Goal-2 report complete; no recommended next phase launched |
| 2026-08-10 | PI venue decision; executor supplies evidence but does not make the venue decision |

## 11. Required quarantine-release amendment

The proposed sign-off amendment must record, at minimum:

- both PI identities, signatures/approvals, and timestamps;
- SHA256 of `AXIS_SPEC_V2.md`, every machine schema, semantic edge-case test
  manifest, cohort-selection rule, and material-negative manifest;
- all four B2 generation/completion manifest-hash pairs listed by the
  quarantine manifest, plus the quarantine manifest hash itself;
- the statement that B2/B6 measurement outcomes were not inspected before
  freeze;
- the exact B2 release scope: v2 measurement and deterministic representation
  replay only, with raw files immutable;
- an explicit statement that B6 remains quarantined and causal analysis remains
  out of scope; and
- the rule that any post-freeze semantic, cohort, negative, threshold-selection,
  or metric change requires a separately visible amendment.

Until that amendment exists, B2 remains generation-only evidence and every
Goal-2 row above is `NOT_AUTHORIZED`.

## 12. Assumptions and PI decisions still required

1. **Proposed cohort size:** 120 development and 120 sealed confirmatory video
   groups are planning values, not ratified sample sizes. PIs must approve or
   revise them before manifest construction.
2. **Control construction:** the 10 natural/10 deterministic-silence split is a
   proposal. PIs must confirm whether synthetic silence may calibrate only a
   failure mode, as specified here, or should be replaced with natural controls.
3. **Two-event yield:** the 60-ID manifest has not been event-pair curated. Its
   usable yield and the feasibility of a clean 30/30 split are unknown.
4. **B2 anchor adequacy:** only four clips overlap the old manual anchor check.
   Event-level anchor intervals for all 48 are required; their eventual quality
   is unknown.
5. **Material matching:** the acceptable timing/loudness matching bands and
   deterministic relaxation order belong in the signed Axis Spec. They must not
   be chosen from measured 2AFC margins.
6. **Replay cost:** B2 preview/internal-state replay and legacy-final
   regeneration throughput are unmeasured. After dual-PI sign-off, the proposed
   1% smoke benchmark may revise hours and sharding only.
7. **Confirmatory reservation:** this plan assumes the reserve is manifest-only
   through Goal 2. Measuring it would violate the stated phase boundary.

## STOP boundary

This is the end of the Goal-1 compute/cohort design. Do not construct the full
cohorts, read B2/B6 audio through a measurer, regenerate finals, run the
identity pilot, launch calibration, or record a freeze hash until both PIs sign
the canonical Axis Specification v2 and the sign-off amendment is committed.
