> **SUPERSEDED FOR HUMAN CURATION (2026-07-19).** No PI signature or electronic approval artifact is required. The current user directive authorizes the Round-1 human-curation release; unresolved scientific questions remain constraints on later measurement and claims.

# Historical PI checkpoint: Axis Specification v2

Status: **DECISIONS REQUIRED; GOAL 2 BLOCKED.** These are scientific freeze
questions, not routine engineering blockers. The executor will not choose them
implicitly.

## Decisions required for sign-off

1. **Anchor authority.** FoleyBench metadata has no event timestamps and the raw
   historical human mark file is missing. Ratify or revise the proposed
   FoleyBench metadata -> visual detector -> light-human-mark provenance chain,
   including whether new blinded interval marks replace the absent raw file.
   Specify precedence, conflict adjudication, and when an interval is too
   uncertain to use. An audio-derived onset cannot define the visual anchor.
2. **Event identity and crop.** Approve the rule that every observation names a
   visible event and uses an event-centered crop. Decide how repeated events,
   partly occluded events, and multiple plausible instances receive stable IDs.
3. **Presence semantics.** Approve the development annotation rubric separating
   target event, unrelated background, absence, and uncertainty. Decide the
   minimum blinded-rater evidence needed before a Presence PASS can support a
   semantic claim.
4. **Timing association.** Approve interval-valued visual and audio onsets and
   the rule that Timing is undefined unless Presence is confidently positive.
   Decide how a repeated or sustained event selects its target onset without a
   nearest-onset shortcut.
5. **Class continuity.** Approve event-centered Class as the v2 primary unit and
   clip-level confident-subset Class as a side-by-side continuity lens. Confirm
   that the frozen coarse map remains the taxonomy and that a full 527-way or
   coarse posterior is mandatory for selectable abstention.
6. **Material references and matching.** Ratify positive-reference generation
   eligibility, aggregation, leave-one-out/disjointness, and same-event proof.
   Define the independent evidence that establishes different material, and
   ratify the negative timing/loudness matching variables, candidate grids, and
   deterministic relaxation order. None may be chosen from observed 2AFC
   margins.
7. **Binding eligibility.** Confirm that conditional identity-assignment
   accuracy remains eligible when both target events are present but an extra
   event is also detected, while the event-set outcome remains `extra`. If not,
   revise the decomposition before freeze.
8. **Selection priorities.** Ratify each axis's deterministic preference order
   among candidates within one standard error of the best development metric.
   In particular, confirm the proposed coverage/support tradeoffs and require a
   power analysis before numeric support minima are selected.
9. **Cohort quotas.** Ratify or revise the proposed 120 development and 120
   sealed confirmatory video-group designs, including the natural versus
   synthetic silent controls. Confirm that the 48-clip B2 bank is exploratory
   only and cannot tune a measurer.
10. **Two-event yield.** Decide whether the existing 60 candidate IDs may be
    divided 30 development / up to 30 reserve after blinded event-pair curation,
    and approve a metadata-only replacement rule if usable yield is lower.
11. **Semantic sidecars.** Decide whether the missing per-example historical
    human anchor/validity records must be recovered or whether new blinded v2
    packages will replace them. Aggregate summaries alone cannot pass the v2
    semantic gate.
12. **Freeze and release scope.** Confirm that the post-signature amendment may
    release B2 for v2 measurement and deterministic representation replay only,
    while B6, causal analysis, and confirmatory execution remain closed. Before
    any later B6 release, require event IDs and anchor intervals for every
    source/donor clip plus separately frozen swap-reference semantics.

## Sign-off mechanics

After both approvals, create `axis_spec_v2.freeze.json` according to
`axis_spec_v2.freeze.schema.json`. It records exact content-file hashes, the
ordered bundle hash, both approvals, authorization scope, and the sign-off
amendment. The envelope is deliberately absent from this Goal-1 branch. Any
semantic revision before signature changes the reviewed content; any revision
after signature requires a visible amendment.

## Executor stop

Until all required decisions are resolved and both signatures are recorded,
do not measure quarantined WAVs, regenerate finals, run calibration, launch the
B-1 identity pilot, fit a probe, or construct the confirmatory cohort.
