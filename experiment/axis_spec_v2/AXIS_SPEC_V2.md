> **OPERATIONAL UPDATE (2026-07-19).** Electronic-signature and freeze-envelope gates are retired. Human event curation is authorized under versioned Git and manifest hashes. This does not release B2/B6 or authorize model inference, GPU work, or downstream measurement; the pre-release language below remains audit history.

# SoundDecisions Axis Specification v2

## Authority and scope

This document is the human-readable canonical draft for five event-centered
measurement axes: Presence, Timing, Class, Material, and Binding. Its machine
contract is `axis_spec_v2.json`. The schemas and semantic edge cases in this
directory are part of the review package.

The executor has not ratified this draft. The separate signed freeze envelope
does not exist and no freeze hash or amendment entry has been recorded. Keeping
the envelope separate avoids an impossible self-hash: it hashes the exact
reviewed content files and is itself registered by the sign-off amendment.
This draft defines what would be measured after sign-off; it reports no new
measurement outcome and does not release the banked B2 or B6 cohorts.

The incoming evidence is interpreted narrowly:

- The old clip-level Presence and Timing definitions were unable to test the
  intended event-level axes. Their small positive within-versus-between-video
  differences justify redesign, not a scientific degeneracy claim.
- Material has useful continuous between-video discrimination, but the old
  absolute-cosine decision predicate is invalid. Raw cosine is not accuracy.
- The historical Class curve reconstructed exactly under its confident-subset
  contract. Its `s_commit=0.346` and readout gap `0.404` remain continuity
  evidence unless an exact v2 reconstruction finds a real error.
- Binding is unimplemented. The 60-clip file is a candidate manifest, not a
  scored cohort.
- B-1 stopped at a lineage gate before probe fitting. It provides no internal
  readout evidence in either direction.

## Shared contract

### Event and anchor

The scientific unit is a specified visible event. A clip is only its container.
Every event has a stable `event_id`, description, and visual anchor represented
as a closed time interval `[lo_s, hi_s]` with source and provenance. An exact
frame without justified uncertainty is forbidden. Multiple events receive
separate IDs and intervals. A missing or unusable anchor makes every
anchor-dependent observation undefined; it is never replaced by a clip prior.

The current candidate provenance chain is FoleyBench metadata, visual onset
detection, and light human marks. FoleyBench metadata contains no event
timestamps, and the raw historical human mark file is absent even though its
aggregate survives. The PIs must ratify the chain's precedence, adjudication,
and replacement-annotation rule. Audio-only onset detection cannot define the
visual anchor.

### Missingness

Uncertain, abstained, indifferent, ambiguous, and undefined observations are
missing evidence, not semantic classes. They are excluded from the relevant
primary denominator while coverage and reason counts remain mandatory. Primary
scoring never imputes an undefined value. Every workstream ends as `PASS`,
`UNRESOLVED`, `INVALID_MEASUREMENT`, `INCOMPLETE_ARTIFACTS`, or
`ENGINEERING_FAILURE`.

### Targets and windows

The primary commitment/readout target is the fork-majority or fork distribution
at the current state. The deterministic ODE-final value is secondary. IDs,
selection, metrics, and reports keep the two targets separate.

Categorical commitment uses agreement over distinct scorable fork pairs,
compared with the same statistic over independent base-seed finals for that
event. Timing uses an interval-aware equivalence tolerance chosen only on
development data. An independent agreement at the numerical identity boundary
is registered separately and excluded from window estimation. Calling such an
axis *video-determined* additionally requires a passed between-video
discrimination gate.

The proposed window rule is the earliest supported sampled progress whose lower
video-bootstrap confidence bound passes the development-selected criterion and
remains passing at every later supported point. Unsupported points are reported
as unsupported; they are not interpolated into a claim. Information-readout and
action-readout windows are separate: the former must beat the correct null and
conditioning-only baseline, while the latter must also meet the frozen
axis-specific accuracy and calibration rule.

### Selection and calibration

No historical numeric threshold becomes universal by reuse. The freeze covers
the metric, null, candidate family/grid, support rule, model-selection rule, and
the procedure by which a threshold is selected on development data. Selection
uses video-disjoint nested development folds. Each axis declares one
deterministic preference ordering among candidates within one standard error of
the best primary metric; remaining ties resolve by calibration and stable ID.
Confirmatory labels and B2 pilot outcomes are unavailable during selection.

A claim of human semantic validity requires completed blinded ratings joined at
the v2 event/trial level. Existing aggregate sidecars are diagnostic until their
per-example joins are recovered. An assembled but unrated package is not human
validation.

## Presence

**Meaning and unit.** For one specified visible event, determine whether a
corresponding audio event occurs near its anchor: `present`, `absent`, or
`uncertain`. Salient unrelated background audio does not count. The unit is the
event.

**Prerequisites and target.** A usable anchor interval, event identity, and an
event-centered crop are required. Persist target-semantic, anchor-local temporal,
and acoustic-salience component scores separately. Select the fusion,
present/absent threshold, and uncertainty rule only on video-disjoint
development folds. `uncertain` is unscorable missing evidence.

**References, nulls, and metrics.** Positives contain a semantically matched
event overlapping the anchor. Natural event-absent and silent controls define
the negative, while loud off-target background is the hard negative. Baselines
are development prevalence/majority, controls, and conditioning-only. Commitment
uses confident present/absent fork agreement. External and internal readout use
balanced accuracy, state recall, Brier score, calibration error, coverage, and
margin over majority for the primary and secondary targets separately.

**Robustness and disposition.** Gain, EQ, codec, and resampling changes should
not change target-event semantics. Off-anchor background must not become target
presence; target removal must change the result. PASS requires semantic
calibration, correct controls, robustness, and support. Background-driven
responses are `INVALID_MEASUREMENT`; inadequate support or ratings are
`UNRESOLVED`; persistent failure demotes the axis from downstream claims.

## Timing

**Meaning and unit.** Timing is the continuous target-audio onset offset
`delta_t` from the visual-event anchor. It is defined only for a confidently
present event. Preserve both the center offset and the conservative interval
`[audio_lo-anchor_hi, audio_hi-anchor_lo]`. Missing onsets stay undefined and
are never forced to zero or a bin.

**Prerequisites and selection.** A usable visual interval, confident Presence,
and a target-event onset interval are required. Event identity and overlap
govern multi-onset association; choosing an arbitrary nearest clip onset is
forbidden. Development folds select onset association, confidence/undefined
handling, and the commitment-equivalence tolerance. The continuous target is
never replaced by that tolerance.

**Nulls and metrics.** Zero offset, a development-only video/anchor prior, and
conditioning-only are the baselines. Report MAE, median absolute error,
accuracy within 50/100/200 ms, interval coverage, and defined coverage. Known
time shifts must translate `delta_t`; irrelevant acoustic perturbations must
not. Wider anchor uncertainty widens the interval rather than silently moving
the point estimate. Failure to track the target event is
`INVALID_MEASUREMENT`; inadequate presence or uncertainty support is
`UNRESOLVED`.

## Class

**Meaning and unit.** Class is a confident coarse semantic class for the target
audio event. The primary v2 unit is an event-centered region. The historical
clip-level curve remains visible only as the continuity lens. Abstention is
missing evidence, never a class.

**Posterior and selection contract.** Persist the tagger's complete 527-way
`clipwise_output` and the derived coarse posterior, mapping hash, event crop,
top indices, margin, entropy, hard label, and rule ID. Calibrate the posterior
and select the abstention threshold on inner video-disjoint development folds
from a predeclared grid, requiring powered per-class support and applying the
shared one-standard-error rule toward coverage. Missing posterior artifacts are
unscorable even when an old argmax survives.

**Nulls, metrics, and preservation.** Use the development confident-label
majority, per-class prevalence, and conditioning-only baselines. Report balanced
accuracy, per-class recall, majority margin, selective risk, coverage, Brier
score, and calibration error. Reconstruct the legacy and v2 curves exactly and
publish them side by side. The current Class headline is preserved unless that
join exposes an actual error; semantic self-consistency alone is not a human
correctness claim.

## Material

**Meaning and unit.** Material is a matched event-level 2AFC trial. Compare the
candidate with a disjoint same-video/same-event positive reference and a
different-video, same-class, similar-timing/loudness, different-material
negative. Randomized A/B presentation, its seed, the correct choice, both
symmetric scores, and the decision are retained so accuracy and AUROC have an
auditable target. The semantic score is
`cos(candidate, positive) - cos(candidate, negative)`. Chance is `0.5`. Raw
cosine is never converted into Bernoulli correctness.

**References and selection.** Candidate, positive, and negative identities and
matching metadata are frozen before candidate measurement. A candidate never
contributes to its references. A reference bank uses leave-one-out construction.
Easy negatives are labeled sanity checks only. Development trials select
embedding preprocessing and an indifference margin from a predeclared grid
using repeat stability, blinded material judgments, coverage, 2AFC accuracy,
AUROC, and the shared one-standard-error rule. Indifferent and invalid-reference
trials are unscorable with explicit reasons.

**Continuity task and disposition.** Resolve A2-4 by replaying the existing
Phase-2 preview path under the same frozen negative manifest. Its exact scope is
6,400 cells (200 clips x four subjects x eight progress points); 800 subject
final embeddings survive, while candidate previews/embeddings and matched
negative embeddings do not. Freeze positive-reference composition,
aggregation, and disjointness as well as negative matching, then persist all
three embeddings. Keep the legacy cosine curve beside the corrected margin
result, but never feed it into policy recovery.
PASS requires semantic material calibration, chance/permutation separation,
and confound robustness. Class, timing, loudness, or reference leakage makes
the measurement invalid; missing embeddings or ratings leave it unresolved.

## Binding

**Meaning and unit.** Binding is the assignment of two target audio-event
identities to two visible-event anchors. Outcomes are `correct`, `swapped`,
`missing`, `extra`, and `ambiguous`. The primary assignment accuracy is
conditional on both target events being present. A missing event is never
scored as swapped.

**Decomposition and selection.** Event-set Presence, temporal order, and
identity assignment are separate outputs with separate support. Development
pairs select bounded detection, ambiguity, and assignment rules before pilot
outcomes. The assignment null is `0.5` only for decisive trials with exactly
two distinguishable target events, both present, and a binary
correct-versus-swapped assignment. Ambiguous trials are not forced into it.
Unconditional outcome majority, temporal-order-only, and conditioning-only are
additional baselines.

**Robustness and boundary.** Test event-ID and anchor-order permutation
equivariance, known swaps, missing events, extra events, and overlapping-anchor
ambiguity. PASS requires semantic assignment calibration and all decomposed
gates. Timing shortcuts or missing-as-swapped behavior are invalid. The banked
B6 cohort inherits these record semantics for a later causal phase, but its
source/donor clips first require curated event IDs and anchor intervals. It
remains quarantined and unanalyzed in this phase.

## Minimum persisted record

Every record carries stable measurement/audio/video/event IDs, input hashes,
anchor interval/source/provenance, generation role and seed/progress/cfg/alpha,
raw component scores, hard/continuous observation, confidence, abstention or
missing reason, reference IDs and metadata where applicable, exact measurer and
model revisions, code/input/spec hashes, command, node, and
`PYTHONHASHSEED`. Axis-specific required fields are enumerated in
`axis_spec_v2.json`; `measurement_record_v2.schema.json` and
`reference_manifest_v2.schema.json` define the transport contracts. The
separate `axis_spec_v2.freeze.schema.json` defines a valid post-signature
envelope; no envelope instance belongs in Goal 1.

## PI freeze gate

Before Goal 2, both PIs must resolve the open questions, ratify or revise this
draft, sign the exact files, and record their SHA256 values and the B2 release
for v2 measurement plus deterministic representation replay in a new
amendment. B6 must remain closed. Any later change to semantics, cohorts,
references, selection, metrics, or thresholds is an explicit amendment. Until
then, this package is `DRAFT` and execution stops.
