> **HISTORICAL CHECKPOINT.** The 2026-07-19 operational update retired electronic signatures and authorized human event curation only. No quarantine, measurement, generation, probe, model, or GPU work is released by that update.

# SoundDecisions Axis Specification v2: Goal-1 delivery

Date: 2026-07-16

Evidence base: remote `main` at `f793878c977844efcb4bdbec5877874d9aaea22e`

Branch: `axis-spec-v2-goal1`

Environment: `ln207`, Linux 5.15, Python 3.10.12, NumPy 2.0.2,
SciPy 1.15.3, pytest 9.0.3

Schema verification: isolated `/tmp` target with jsonschema 4.23.0; project
`.venv` unchanged and temporary target removed after validation

Execution: CPU-only document/schema/test work; zero GPU hours

## A. Execution status

| Goal-1 workstream | Status | Deliverable | Decisive outcome |
|---|---|---|---|
| Asset and gap audit | `PASS` with incomplete artifacts recorded | `ASSET_AND_GAP_AUDIT.{json,md}` | Banked and historical evidence is inventoried without measuring quarantined audio |
| Axis Specification v2 | `PASS` as a draft; not ratified | `experiment/axis_spec_v2/AXIS_SPEC_V2.md`, JSON contract, four schemas | Five event-centered definitions and fail-closed lifecycle are ready for PI decisions |
| Semantic edge contracts | `PASS` | `semantic_edge_cases.json`, `foley_cw/axis_spec_contract.py`, `tests/test_axis_spec_v2.py` | Missingness, interval, abstention, 2AFC, and Binding semantics execute deterministically |
| Cohort design | `PASS` as a proposal | `COHORT_AND_GOAL2_COMPUTE_PLAN.md` | Development, B2 pilot, and sealed confirmatory roles are disjoint by video/scene group |
| B-1 identity-pilot design | `PASS` as a design | `B1_IDENTITY_PILOT_DESIGN.md` | Same-forward, same-operation lineage and measured-tolerance procedure are specified |
| PI checkpoint | `UNRESOLVED` | `PI_OPEN_QUESTIONS.md` | Twelve scientific decisions and two PI signatures remain required |
| Goal-2 execution | `UNRESOLVED` / `NOT_AUTHORIZED` | No execution artifact | Correctly stopped at the mandatory gate |

The draft specification uses a separate post-signature freeze envelope. No
envelope instance or freeze hash exists in Goal 1. This avoids a self-hash and
prevents the draft itself from authorizing work. A valid future envelope must
bind exact reviewed files, both PI approvals, a sign-off amendment, and a
narrow B2 release while keeping B6, causal work, and confirmatory execution
closed.

## B. Asset and gap findings

- **B2 is intact but closed:** 17 seeds x 48 clips, 816 base finals and 78,336
  fork finals, for 79,152 retained WAVs. All 16 registered validators passed.
  The bank has final/fork audio, not same-progress external previews or internal
  representations.
- **B6 is intact and out of scope:** 128 donor-not-source pairs per cfg and
  2,560 WAVs total. No audio was measured and no swap outcome was analyzed.
  Later measurement first requires source/donor event IDs and anchor intervals,
  then a separate causal-phase release.
- **Historical audio is absent:** primary `results/` retains zero final,
  preview, or audit WAVs. Journaled seeds/configs and input video make required
  finals deterministically regenerable after sign-off; a label is never
  expanded into an invented posterior.
- **A2-4 has a precise repair scope:** 6,400 Phase-2 Material cells (200 clips x
  four subjects x eight progress points). Eight hundred subject-final 512-D
  embeddings survive. Candidate previews/embeddings and matched negative
  embeddings do not.
- **Class posteriors are missing:** old Class rows retain hard labels but not
  the full 527-way or coarse posterior. Reselectable abstention requires
  deterministic final regeneration or retained banked audio.
- **Binding is not ready:** the 60 IDs are a candidate pool with no paired
  event identities, two anchor intervals, per-event presence, or assignment
  truth.
- **Human anchor evidence is not row-auditable:** the 30-clip aggregate survives,
  but the raw light-human mark file is absent.
- **B-1 remains incomplete:** bundle
  `1002__p1cfg1_ind12__s0.75.npz` failed at relative L2
  `2.585371085837806e-4` versus the old `2e-4` gate before probe fitting. No
  internal-readout evidence exists in either direction.

## C. Axis status at the checkpoint

These are evidence/disposition statements, not Goal-2 calibration verdicts.
No axis receives a new scientific `PASS` from a specification exercise.

| Axis | Incoming evidence preserved | V2 decision | Status until Goal 2 |
|---|---|---|---|
| Presence | Old clip-level majority was high, but within-video agreement exceeded between-video agreement; that supports redesign, not scientific degeneracy | Target visible event near an interval anchor; off-target background is a hard negative; component fusion selected on development data | `UNRESOLVED` |
| Timing | Old absolute-bin definition was incapable of testing target-event timing despite a small positive video-heterogeneity signal | Continuous anchor-relative onset offset with propagated interval uncertainty, defined only under confident Presence | `UNRESOLVED` |
| Class | Legacy journal reconstruction was exact (`max delta=0.0`); confident `s_commit=0.346`, gap `0.404`, naive gap `0.287` | Preserve the legacy curve beside event-centered v2; persist the full posterior; abstention remains missing evidence | Legacy evidence preserved; v2 `UNRESOLVED` |
| Material | Tier-0 continuous cosine separation was informative, but absolute cosine was an invalid Bernoulli predicate | Matched, randomized 2AFC with disjoint positive and matched different-material negative; chance `0.5`; persist all embeddings and choices | `INCOMPLETE_ARTIFACTS` / `UNRESOLVED` |
| Binding | Never implemented | Separate event-set Presence, temporal order, and identity assignment; primary correct/swapped accuracy only when both distinguishable targets are present; missing is never swapped | `INCOMPLETE_ARTIFACTS` / `UNRESOLVED` |

All five machine contracts define scientific meaning, event unit, interval
anchor, prerequisites, target, missingness, references, nulls, commitment,
external/internal targets, metrics, robustness, semantic calibration, minimum
fields, deterministic development selection, and terminal dispositions. Old
numeric thresholds are not promoted to universal v2 thresholds.

## D. Cohorts and proposed Goal-2 compute

The proposed development and sealed confirmatory designs each contain 120
video groups across clear positives, weak events, natural/silent controls,
hard unrelated-background negatives, and separable two-event clips. Those are
PI-review quotas, not frozen sample sizes. Development may select procedures;
the 48-clip B2 bank is exploratory pilot data only; the confirmatory reserve is
manifest-only through Goal 2.

B2 supplies banked commitment-target audio after signed release. It still needs
event records, material reference manifests, controls, and deterministic replay
of 816 base trajectories for external previews/internal states. Fork audio is
measured in place, not regenerated. Binding uses separately curated two-event
clips and cannot be inferred from B2.

The provisional all-axis planning envelope is 32-91 A800 device-hours plus
80-220 CPU-hours. It is a capacity estimate, not an allocation or authorization.
The priority path is Material 2AFC, Class posterior preservation, B-1 identity
and recollection, and banked-cohort Class measurement. Physical GPUs 0-3 remain
categorically excluded; permitted devices must have at least 70 GiB free.

## E. B-1 lineage design

The five-video pilot holds clip 1002 out from tolerance estimation and compares
exact replay packets across eligible GPUs on `an12` and `an29`. One instrumented
evaluation persists fp32 post-block tokens, exact original-order fp32 pooling,
quantized tokens, device-side latent/time inputs, velocity, Tweedie latent,
selected attention outputs/QKV, conditioning tokens, and external-preview
representation with full IDs, dtypes, shapes, parent hashes, and hook sites.

Only numerically equivalent operations enter the gate. Mean-after-quantization
versus quantization-after-fp32-mean is diagnostic only. The old mismatch also
used separately regenerated trajectories and velocity calls, so reduction order
is one proven confound, not the isolated cause. Goal 2 proposes tolerance as a
measured benign-delta quantile times a PI-ratified safety factor, records it by
amendment before held-out access, and defaults to recollection after PASS.
Failure ends internal-readout execution as `ENGINEERING_FAILURE`.

## F. Independent review corrections

Two independent reviews were applied before verification:

1. Replaced mutable in-document signatures/self-hash with a separate signed
   envelope and made unsigned/frozen reference manifests invalid.
2. Replaced the generic measurement schema with axis-specific records and
   fail-closed ranges/states.
3. Added auditable Material candidate/positive/negative generation and audio
   lineage, randomized A/B orientation, correct choice, symmetric scores, and
   honest missing-reference records without invented IDs.
4. Added Binding composite event sets and source/donor/swap lineage while
   retaining the B6 analysis ban.
5. Made every axis's development selection preference deterministic and added
   all five terminal workstream statuses.
6. Fixed edge contracts so off-target background cannot imply Presence,
   missing events cannot score as swapped, and non-boolean/fractional Binding
   inputs fail.
7. Corrected B-1 causal wording and added native post-block hook-site hashes and
   actual device-side forward inputs.
8. Made the missing raw anchors, exact A2-4 scope, B6 curation prerequisite, and
   Material positive-reference decisions explicit for the PIs.

## G. Verification

- Artifact-free full suite, `PYTHONHASHSEED=0`:
  `1070 passed, 3 skipped, 16 warnings in 818.99s`; SHA256 of
  `GOAL1_TESTS.log` is
  `c91cb8aa8f16dada4cc904cb125c2fc6f628a9ae93bd45bb2cfe725c0bf27ac0`.
- Fresh semantic run, `PYTHONHASHSEED=1`: `12 passed in 0.21s`; SHA256 of
  `GOAL1_HASHSEED1_TESTS.log` is
  `6b68487be8cb7888dd484c4315bec7c57a83f116ca55d197780f04e35f0d7f77`.
- The worktree contained zero WAV, NPZ, PT, or safetensors files.
- Native Draft 2020-12 metaschema checks passed for all four schemas under an
  isolated jsonschema 4.23.0 target. The draft axis-spec instance, signed-
  envelope fixture, draft-reference fixture, and Presence-record fixture
  validated; an unsigned empty frozen reference and a Class record without a
  posterior were rejected. SHA256 of `GOAL1_SCHEMA_VALIDATION.log` is
  `c12a6fd9da03895bb304e378df7e409f727e1e292664e3819b11903e9bb1acf4`.
  Cross-record constraints JSON Schema cannot express remain explicitly listed
  as mandatory semantic-validator checks.
- `git diff --check` passed for every scoped commit.
- No model weights were loaded and no banked or historical audio was decoded or
  measured; no generation, GPU job, B2/B6 analysis, probe fit, threshold
  selection, freeze hash, amendment, or quarantine release occurred.

## PI checkpoint and STOP

The PIs must resolve `PI_OPEN_QUESTIONS.md`, revise or approve the draft, and
both sign before an envelope or freeze hash is created. The subsequent
amendment may release B2 only for v2 measurement and deterministic
representation replay. B6, causal analysis, and confirmatory execution stay
closed.

**Executor state: STOPPED AT THE GOAL-1 GATE.**
