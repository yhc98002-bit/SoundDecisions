# Asset and gap audit for Axis Specification v2

Status: **PASS (audit complete); INCOMPLETE ARTIFACTS RECORDED; GOAL-1 DRAFT; NOT FROZEN**

This is an inventory and recovery plan, not a measurement result. No audio was
opened or measured, no model was run, and no B2/B6 outcome was inspected. Goal
2 remains blocked on dual-PI sign-off and a recorded release amendment.

## Evidence boundary

The audit used the Goal-1 worktree and the primary full-storage checkout at
`f793878`, plus small manifests and reports from the `arc4-gpu` worktree at
`92b2f46`. The authoritative GPU manifest is
`results/arc4_quarantine/QUARANTINE_MANIFEST.json` at SHA256
`200900e998ba30aa8592d9213b48c3c3f7ff69817017959e1393e517d76cd2b5`.
Raw B2 and B6 WAVs were inventoried only through file metadata, journals, and
registered counts.

## Anchor sources

| Source | What exists | Reuse | Decisive gap |
|---|---|---|---|
| FoleyBench metadata/UCS | 5,000 metadata rows; 785 extracted Discrete MP4s, of which 590 are Single-source and 195 Multi-source; `clips_index.csv` binds key, SHA256, duration, caption, UCS/AudioSet category, and source type | Candidate selection, grouping, diversity, and matching covariates | No event ID, timestamp/interval, event presence state, material identity, or assignment truth |
| Visual detector in `foley_cw/visual_anchors.py` | Up to four salience-ranked frame-difference/Farneback candidates with half-FWHM uncertainty for all 785 clips | Candidate anchor evidence | Candidates are not attached to a specified semantic event; top salience must not become the event anchor automatically |
| Original-audio spectral-flux onset | Candidates for all 785 clips; median audio-vs-visual disagreement 0.757 s | Diagnostic only | `AUDIO_ANCHOR_NOT_ADOPTED`; it is not a visible-event ground truth source |
| Light human check | Aggregate says 27 marked and 3 no-event among 30 clips | Aggregate audit history | `results/labeling/labels_anchor_v1.jsonl` is absent and `anchor_check_30.csv` has zero filled human values. Raw marks cannot be row-audited or converted to intervals |
| Two-event manifest | 60 unique Multi-source+Discrete clip IDs; all join to the anchor table | Binding candidate pool | No two event IDs, two anchor intervals, presence states, assignment labels, or ambiguity verdicts |

The old `EventAnchor` representation stores point timestamps with uncertainty.
V2 needs a named visual event and an interval. Existing detector points can seed
annotation, but cannot settle which event is being measured.

## Asset matrix

| Asset | Verified inventory | Disposition | Required work |
|---|---:|---|---|
| Legacy measurer code | Old Presence energy/eventness gate, absolute Timing bin, Class tagger/abstention path, Material CLAP embedding; Binding raises `NotImplementedError` | **REUSE LOADERS / CLASS AUDIT PATH, REDESIGN DECISIONS** | Build event-centered Presence/Timing and 2AFC Material; implement Binding for the first time |
| FoleyBench extracted inputs | 785 MP4s with SHA256 index | **REUSABLE** | Curate event IDs, anchor intervals, and v2 development/confirmatory roles |
| Frozen Phase-1 population | 200 single-event and 60 two-event IDs | **AUDIT HISTORY / ID SOURCE** | Do not reuse old clip-level axis semantics as v2 annotations |
| Primary measurement JSONL | 350,556 rows: Presence 87,063; Timing 87,063; Class 89,367; Material 87,063 | **REUSABLE FOR HISTORICAL RECONSTRUCTION** | Class has no posterior; Presence/Timing use old definitions; Material has embeddings but no 2AFC references |
| Primary historical audio | zero WAVs under `results/` (`finals`, `previews`, and audit directories empty) | **INCOMPLETE_ARTIFACTS** | Deterministically regenerate only required journaled identities into a new root |
| Pooled internal features | 111,040 NPZs, `pooled[12,448]` float16; 25,600 are p1cfg1 independent bundles | **HISTORICAL BASELINE ONLY** | Missing full tokens, state, velocity, Tweedie, complete conditioning lineage, and fp32 same-operation reference |
| Conditioning features | 200 clip-level NPZs | **HISTORICAL BASELINE ONLY** | Not the complete network-consumed conditioning-token lineage |
| Gate-A caches | 400 cfg-specific probability bundles plus 144 dial-noise bundles | **GATE-SCOPE REUSE ONLY** | Not a complete posterior store for every required historical or banked cohort |
| Stage-M / rerun caches | 2,720 plus 3,744 pooled-feature NPZs per retained rerun copy, gate bundles, and JSONLs; no WAVs | **LEGACY DIAGNOSTICS** | Old semantics and absent source audio prevent direct v2 remeasurement |
| Phase-2 readout rows | 200 journals; 25,600 rows; 6,400 Material rows for 200 clips x 4 subjects x 8 progress points | **LEGACY CONTINUITY ONLY** | Material stores only scalar own-target cosine; no candidate preview or reference embeddings |
| B2 multi-seed bank | 17 seeds x 48 clips; 816 base finals + 78,336 fork finals = **79,152 WAVs** and 40,532,156,160 bytes; 16/16 validators passed | **REUSABLE AFTER SIGNED RELEASE** | Measure banked WAVs in place; replay only missing previews/internal representations; add event annotations and controls |
| B6 condition-swap bank | 128 pairs at each cfg; 1,280 WAVs per cfg, 2,560 total | **OUT OF SCOPE, PRESERVE UNMEASURED** | Freeze inherited event-level semantics now; a later causal phase must separately release it |
| B1 retap collection | 25,600 bundles plus 200 journals | **DIAGNOSIS ONLY** | Bundle 31 failed relative L2 `2.585e-4 > 2e-4`; run identity pilot and recollect, no probe on this cache |
| Legacy semantic sidecars | 300 MLLM rows over 100 clips; a 50-clip human summary | **CONDITIONAL REUSE** | Use only when event and axis records join exactly; otherwise build a blinded v2 package |

The banked B2 WAVs are final and fork audio, not external Tweedie previews.
They are sufficient for frozen WAV-in/WAV-out v2 measurement and commitment
targets. They are not sufficient for external-readout or internal-readout maps,
which require deterministic replay of the 816 base trajectories.

## Deterministic final regeneration

Historical Phase-1 identities are reconstructible from the retained input video,
journal/config, local model weights, and the runner's seed contract:

```text
SeedSequence([seed, crc32(str(part_1)), ...])
independent: (clip, "ind", j)
base:        (clip, "base")
fork:        (clip, "fork", s)
```

For `p1cfg1`, the recorded contract is `small_16k`, cfg 1.0, alpha 0 for
independents, 20 steps, 8 seconds, fp32, video CLIP + Synchformer conditioning,
seed 0, and progress `{0.05,0.15,0.25,0.35,0.45,0.60,0.75,0.90}`. The local
small MMAudio weight SHA256 is
`61987bcbd6fc689af063075d7efaef29425f65df155dac589c07fa8173a03c1c`;
the PANNs tagger SHA256 is
`e2ee543a27919542c2ea03eabaa70b24dcd4e6c8e05621de6b67a94e4c5058e6`.

Goal 2 must regenerate into a new immutable RunStore, persist the audio hash and
the full 527-way `clipwise_output` plus coarse posterior/diagnostics, and bind
every output to the input video, source journal, code/config/model hashes, RNG
parts, dtype, device, and environment. A missing or mismatched join fails the
unit; the old label is never expanded into an invented posterior.

## Exact A2-4 repair inputs

The continuity cohort is exactly 6,400 Phase-2 Material cells: 200 clips, four
subjects (`j=0..3`), and eight progress points. The current row has only
`clip`, `j`, `s`, and scalar own-final cosine. The corresponding 800 subject
final embeddings do join from the Phase-1 JSONL and are 512-D. The same JSONL
also contains 16 final Material embeddings per clip, which can supply a
disjoint same-video reference pool if the frozen reference rule admits them.

The repair still needs:

1. deterministic replay and decode of all 6,400 intermediate previews;
2. a persisted 512-D candidate embedding for every preview;
3. a frozen, disjoint same-video/same-event positive-reference assignment;
4. a frozen different-video negative assignment matched on coarse class,
   anchor timing, and loudness, with independently established different
   material;
5. positive/negative embeddings and IDs, event IDs and intervals, matching
   covariates, relaxation level, audio hashes, model revision, both raw cosines,
   their margin, and the indifference decision; and
6. the legacy scalar cosine beside the new 2AFC result, never as a Bernoulli
   probability.

The negative manifest must be fixed without reading candidate margins. It must
record every candidate/positive/negative clip, event, and generation ID,
coarse class, material stratum, timing/loudness distance, scene group, matching
relaxation, rule version, and manifest hash. A candidate must never contribute
to its own positive reference.

## B6 semantic inheritance

B6 remains closed. Stage A should only ensure that a later release can measure
each source, donor, and swap WAV with the signed v2 schemas: pair/cfg and audio
hash; source/donor event IDs and anchor intervals; Presence components and
abstention; defined/undefined Timing offset; full Class posterior; Material
embedding and frozen source/donor nearest-reference margins; and Binding
event-set presence, order, identity assignment, missing/extra, and ambiguity.

Before any later measurement, both clips in every pair need curated event
identities/anchors and all reference rules must be frozen. That later causal
phase requires its own quarantine-release amendment. This phase computes no B6
statistic.

## Fail-closed gaps

| Gap | Status | Cheapest valid resolution |
|---|---|---|
| Event identities and interval anchors | `INCOMPLETE_ARTIFACTS` | Curate named events and uncertain visible intervals for development, B2, and two-event candidates |
| Raw 30-clip human anchor marks | `INCOMPLETE_ARTIFACTS` | Recover `labels_anchor_v1.jsonl` or repeat blinded interval annotation |
| Binding assignments | `INCOMPLETE_ARTIFACTS` | Curate two events, two anchors, per-event presence, assignment, and ambiguity |
| Phase-2 Material 2AFC inputs | `INCOMPLETE_ARTIFACTS` | Freeze references and replay 6,400 preview cells with three persisted embeddings |
| Reselectable Class posteriors | `INCOMPLETE_ARTIFACTS` | Regenerate only required journaled finals and persist full posteriors |
| B1 feature lineage | `INCOMPLETE_ARTIFACTS` (upstream `B1_INCOMPLETE`) | Signed same-forward identity pilot, then recollect after PASS; only emit `ENGINEERING_FAILURE` if the repaired gate cannot pass |
| B2 measurement authorization | `UNRESOLVED` (`BLOCKED_ON_PI_FREEZE`) | Dual-PI v2 sign-off and measurement-only release amendment |

## Audit conclusion

B2 is the primary reusable Goal-2 measurement asset after signed release.
Historical journals preserve the Class result's audit trail and many Material
final embeddings, but they do not supply event-centered anchors, selectable
Class posteriors, the Phase-2 2AFC triplets, or lineage-valid internal features.
Those items must be newly materialized under the signed v2 spec. B6 remains
preserved and unmeasured.
