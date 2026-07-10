You are editing two markdown research-planning documents for the `foley-cw` project:

1. `EXPERIMENT_PLAN.md`
2. `FINAL_PROPOSAL_SHORT.md`

You must revise them according to `PI_review.txt`.

This is not a cosmetic rewrite. It is a strategic revision. The current documents are already mostly correct on the core measurement protocol: self-target maps, `A_independent` normalization, Phase-0 reliability gate, smallest-valid-α, ODE-target vs fork-majority readout, bootstrap-by-video, and no premature policy claims. Preserve that backbone. The task is to incorporate the PI review’s higher-level corrections and make the two markdown files internally consistent, more reviewer-proof, and more aligned with a serious ICLR-style project.

Do not run experiments. Do not invent empirical results. Do not claim anything has been validated. Do not silently upgrade claim tier. Edit markdown only.

## Overall revision goal

Move the project framing from:

> “Axis-gated pruning for V2A” / “policy method with maps as support”

to:

> “Sound Decisions: a trajectory-level study of when cross-modal Foley decisions become committed, when they become readable, and how that map schedules inference-time compute.”

The paper’s center of gravity should become:

1. **Decision / commitment map** across Foley axes.
2. **Readout map**, including internal-feature readout as a major scientific pillar.
3. **Commitment–readout gap** and R1/R2 interpretation.
4. **Causal validation of commitment as irreversibility**, via a small intervention.
5. **Policy as downstream scheduler**, not as the main novelty and not as a head-to-head fight against SMC-ITA.

The revised documents should make clear that the policy layer is now crowded and therefore table-stakes. The distinctive paper is the map, internal readout, and causal validation.

---

# Part A — Revisions to `FINAL_PROPOSAL_SHORT.md`

## A1. Rename / retitle to reduce overemphasis on pruning

Current subtitle overweights “Axis-Gated Candidate Pruning.” Revise title/subtitle to foreground the map.

Use something like:

```markdown
# Final Proposal — Sound Decisions (`foley-cw`)

**Title:** Sound Decisions: When Are Foley Decisions Made in Video-to-Audio Flow Generation?
**Subtitle:** Commitment and Readout Maps for Cross-Modal Foley Generation
```

Keep `foley-cw` as project handle.

Do not make pruning the subtitle. Pruning should appear as a downstream use case.

## A2. Add a short abstract-style paragraph near the top

After the status paragraph or after “The pivot,” add a concise abstract-style framing derived from the PI review:

```markdown
**Working abstract.** We map when a cross-modal generator decides each aspect of its output, show that some decisions may be readable from the model’s internal states before they are audible in `x0(s)` previews, and use the resulting commitment–readout map to schedule inference-time compute.
```

Make it clear that this is a planned claim, not an empirical result yet.

## A3. Unify notation: use `s`, not raw `t`

The experiment plan declares generation progress `s ∈ [0,1]` as the canonical reporting axis. The proposal still uses `t_commit`, `t_read`, and `x0(t)` in several places. Revise the proposal to use:

* `s_commit`
* `s_read`
* `x0(s)`
* `gap = s_read - s_commit`

If you retain `t` at all, define it only as MMAudio’s internal integration time and say it is never the reported axis unless mapped to `s`.

Update the glossary, hypotheses, contributions, claim tiers, and next gate accordingly.

## A4. Rewrite the novelty sentence

Current novelty is close but under-specified relative to the PI review. Replace the novelty sentence with a sharper one:

```markdown
> **Novelty:** per-axis commitment **and** readout maps for cross-modally conditioned Foley axes, including timing and temporal binding, measured in a real V2A flow system; the commitment–readout gap is treated as a first-class object and used to schedule inference-time compute.
```

Also state the negative boundary:

```markdown
If the axes do not separate, this is not a failed method paper; it becomes a diagnostic / negative result about the absence of useful decision-window separation in the tested V2A system.
```

## A5. Expand “Background Lineage & Novelty” with the missing speciation / dynamical-regimes line

The proposal currently cites Critical Windows, DiffRS/Restart, and V2A. Add a fourth lineage or revise the first lineage to include:

* speciation / dynamical regimes in diffusion;
* broad structure such as class getting sealed in a narrow transition;
* spectral or geometry-based accounts of speciation time;
* the recent coupled multimodal diffusion theory from PI review, where cross-modal coupling strength shifts speciation times and creates tunable synchronization.

Do not invent exact bibliography if not available in the local files. Use citation placeholders if needed, e.g.:

```markdown
*Biroli et al.; Bonnaire/de Bortoli/Mézard line; Raya & Ambrogioni; Georgiev et al.; Sclocchi et al.; coupled multimodal speciation theory, 2026 — TODO: verify exact BibTeX.*
```

The point is not to overclaim that we are first to observe critical windows. The point is to position our novelty as:

* real V2A / Foley, not image-only;
* cross-modal conditioning;
* axis decomposition beyond class;
* readout map, not just commitment/speciation;
* commitment–readout gap;
* inference scheduler.

## A6. Add a “Why policy alone is no longer enough” paragraph

In the pivot or background section, add a paragraph explaining the June 2026 landscape shift from PI review:

* SMC-ITA uses sequential Monte Carlo inference-time alignment for V2A with intermediate rewards and lookahead;
* another concurrent line studies inference-time scaling for joint AV generation using multi-verifier frameworks;
* this does not kill the project because they do not measure commitment, do not decompose axes, and do not ask when decisions are made;
* consequence: policy is no longer the main novelty; maps and readout gaps are the paper.

Use careful wording. Do not assert bibliographic details beyond the PI review unless present elsewhere. Include TODO citation placeholders if exact references are absent.

Suggested wording:

```markdown
Recent V2A inference-time alignment work makes a pure policy paper crowded: scalar or multi-verifier rewards can already guide population search. Our claim is orthogonal. We ask when each Foley axis becomes decided and when it becomes readable. SMC-style search can then be treated as one downstream consumer of the map, not as the object we are trying to beat by brute force.
```

## A7. Reorganize the contributions

Current C1–C5 are usable but need reweighting. Revise contributions so that internal readout and causal validation are not buried.

Recommended structure:

```markdown
### C1 — Per-axis decision maps
Commitment maps normalized over video-conditioned priors; `A(axis, s, α)` surfaces; α-robust ordering.

### C2 — Readout maps and the commitment–readout gap
External `x0(s)` probes, ODE-target and fork-majority targets, R1/R2 interpretation.

### C3 — Internal readout: “the generator knows before the audio shows”
Always cache pooled DiT / latent / AV cross-attention features during Phase 1/2 generation. Linear probes on cached features test whether `s_read(internal) ≈ s_commit ≪ s_read(x0-probes)`. This is the cleanest test of probe limitation and the strongest path to a cheap process reward.

### C4 — Causal irreversibility / editing-window validation
A small intervention around measured `s_commit`: push one axis toward an alternative value just before vs. just after the commitment boundary. If flip rate collapses after the boundary, commitment is not merely observational fork agreement but causal irreversibility.

### C5 — Map-scheduled inference-time search
Axis-gated pruning / rollback and SMC-style resampling are downstream schedulers. Present the map as a scheduler for population search, not as a direct competitor to scalar SMC-ITA.
```

You can keep C2 measurement/calibration as a separate contribution if the document reads better, but do not let measurement protocol crowd out internal readout and causal validation.

## A8. Revise hypotheses

Add or revise hypotheses to reflect PI review:

* H1: axes commit at different `s`.
* H2: commitment and readout differ; R1/R2 gap exists.
* H3: internal features can read some commitments earlier than `x0(s)` previews.
* H4: stronger cross-modal conditioning, e.g. CFG scale or sync-feature conditioning, shifts commitment earlier.
* H5: commitment boundary has causal irreversibility: interventions before `s_commit` can flip an axis more often than interventions after.
* H6: the map schedules population search, including cascaded BoN and SMC-style resampling.
* H7: some phenomena are seed-determined rather than trajectory-window-determined.

Keep all as untested.

## A9. Add correctness factorization explicitly

Insert a subsection, probably before the policy contribution or compute story:

```markdown
## Correctness factorization

The maps target the model’s self-target, not human correctness. Policy uses correctness only after a value is readable. For an axis whose value can be read, correctness can be factored as:

`axis correctness = match(readable axis value, video/event anchor)`

The matching relation is time-independent; the time-dependent part is readout fidelity. Once “class = door slam” is readable at `s = 0.3`, checking whether it matches the video anchor does not require waiting until the final audio, except insofar as the readout itself is unreliable. This factorization is the load-bearing logic for Phase 4 and also defines the boundary of the framework: holistic qualities that do not factor through a readable axis value are out of scope for the v1 map.
```

This must also be reflected in the experiment plan’s Phase 4 preflight.

## A10. Revise claim tiers

Current METHOD tier requires policy improvement. That is now too restrictive. Revise claim tiers to allow a strong science paper even before policy dominates.

Suggested claim tiers:

```markdown
| Tier | Required evidence |
|---|---|
| **METHOD / full** | Separated α-robust commitment ordering, early readout, internal probes closing the R2 gap for at least one important axis, causal irreversibility evidence, and map-scheduled search improving fixed-budget correctness over strong baselines. |
| **SCIENCE / diagnostic strong** | Separated α-robust commitment windows and a clear commitment–readout gap; internal probes show whether the gap is probe-limited. Policy may be weak, but the map is publishable. |
| **CAUSAL diagnostic** | Fork agreement windows are supported by intervention asymmetry around `s_commit`, even if downstream pruning is limited. |
| **NEGATIVE** | No useful axis separation or all axes commit only near final audio; method degenerates to scalar rejection. |
| **STOP** | No trajectory access, invalid SDE/fork kernel, unreliable measurements, or unusable anchors. |
```

## A11. Update anti-overclaim rules

Add these anti-overclaim rules:

* Do not claim absolute `s_commit` is model-intrinsic; it is relative to model, α, discretization, and fork kernel.
* Do not claim policy beats SMC-ITA unless actually tested under matched NFE and scoring-call budgets.
* Do not claim internal probes are free in storage; they are zero extra forward passes but have storage and fitting costs.
* Do not claim coupled multimodal speciation theory proves our results; it motivates our ablations.
* Do not claim holistic quality can be handled unless it factors into readable axes.

## A12. Add expected Figure 1 taxonomy

Add a planned figure concept:

```markdown
**Planned Figure 1:** not a simple early/mid/late timeline. Use a decision taxonomy:
1. video-determined axes: `A_independent` is already near 1;
2. seed-determined phenomena: predictable from initial noise / very early state;
3. trajectory-early axes: low `A_independent`, early `s_commit`;
4. trajectory-late axes: low `A_independent`, late `s_commit`;
5. committed-but-unreadable axes: R2 gap, candidates for internal readout.
```

This follows the PI review’s suggestion and helps avoid overclaiming.

---

# Part B — Revisions to `EXPERIMENT_PLAN.md`

## B1. Preserve the existing backbone

Do not delete the current core protocol. Preserve these elements:

* maps target self-target, not correctness-vs-video;
* maps are measurement-dependent, not measurement-free;
* Phase 0 reliability gate;
* velocity-to-score SDE validation at α=0 and nonzero α;
* `A_independent` normalization;
* smallest-valid-α rule and full `A(axis, s, α)` surface;
* ODE-target and fork-majority readout;
* bootstrap over videos;
* no `GO_POLICY` before maps pass `GO_MAP` + `GO_READOUT`;
* no generator fine-tuning;
* no delayed callback / memory architecture.

The revision should add missing safeguards and new planned analyses, not destabilize the protocol.

## B2. Add a “PI Review Amendments” section near the top

After Project Summary or Evidence Status, add a short section listing what changed after PI review:

1. Add speciation / dynamical-regimes lineage to proposal.
2. Add micro-map sanity stage before full Phase 0.
3. Add internal-feature logging now, not after the fact.
4. Add `A_independent` cap / video-determined exclusion rule.
5. Add α-robust ordering requirement to `GO_MAP`.
6. Add SMC-ITA and AV inference-time scaling as Phase-4 baselines / positioning.
7. Add correctness factorization for Phase 4.
8. Add optional second-checkpoint commitment-only sanity.
9. Add causal intervention validation.
10. Add coupling-strength ablation via CFG scale or sync-feature drop.

This helps future agents understand why the plan changed.

## B3. Insert a Phase 0A micro-map stage

The PI review says to insert a micro-map before the full Phase-0 build. This should not replace the strict feasibility/reliability gate. It is an end-to-end convention and pipeline sanity test.

Add a new phase before full Phase 0, perhaps:

```markdown
### Phase 0A — Video-conditioned micro-map sanity test

Purpose: exercise the full measurement chain before expensive FoleyBench curation and full reliability gating.

Inputs:
- 12–16 available VGGSound or equivalent clips; do not block on FoleyBench.
- coarse-class axis only.
- off-the-shelf audio tagger.
- `s ∈ {0.25, 0.5, 0.75}`.
- `K = 8` forks per `x_s`.
- `N = 8` independent full generations per video for `A_independent`.
- one provisional α from a quick valid-diversity probe.

Outputs:
- `micro_map_sanity_report.md`
- `micro_map_sanity.csv`
- cached audio previews and generated completions sufficient to debug decode/tagger/agreement/normalization.

Required sanity endpoints:
- fork agreement should approach ~1 as `s → 1`;
- fork agreement near `s → 0` should approximately match `A_independent`;
- no denominator explosion from `A_independent≈1`;
- tagger outputs and agreement metrics are non-degenerate;
- failures route to `FIX_PIPELINE_CONVENTION`, `FIX_SCORE_CONVERSION`, or `FIX_MEASUREMENT_CHAIN`, not to paper-level conclusions.

Important: Phase 0A is not publishable evidence and does not license `GO_MAP`.
```

Add decision tokens:

* `GO_FULL_PHASE0`
* `FIX_PIPELINE_CONVENTION`
* `FIX_MEASUREMENT_CHAIN`
* `FIX_SCORE_CONVERSION`

The agent should run 0A only after minimal trajectory access and provisional nonzero-α sanity exist. Do not let it bypass the full Phase 0 reliability gate.

## B4. Add internal-feature logging as mandatory instrumentation

Currently internal probes are Phase 7 conditional. The PI review says this is too late because features cannot be retrofitted without rerunning.

Revise the plan as follows:

* In Phase 0.1 trajectory access, require hooks to optionally dump pooled hidden states / DiT block features / AV cross-attention summaries per saved `s`.
* Add `internal_feature_cache_manifest.md` to Files to Produce.
* Add a hard instruction: during Phase 1/2 generation, cache internal features even if internal probes are analyzed later.
* Make internal analysis non-blocking for `GO_MAP`, but no longer an afterthought.

Suggested wording:

```markdown
Internal probes are not required to pass Phase 0 or `GO_MAP`, but internal feature logging is required before any large Phase 1/2 generation. If features are not cached, R2 cannot be cleanly tested without rerunning the expensive generation. Therefore log now, analyze later.
```

Update Phase 7 title to something like:

```markdown
Phase 3b / Phase 7 — Internal-feature readout analysis
```

It can remain conditional for analysis, but the logging must be mandatory.

## B5. Add cache requirements for `x_s` and `x0(s)`

The PI review says to cache the full `x_s` grid and decoded `x0(s)` previews so readout probes are post-hoc.

Add to Files to Produce:

* `trajectory_cache_manifest.md`
* `preview_cache_manifest.md`
* `internal_feature_cache_manifest.md`

Add to hard constraints:

```markdown
Do not design readout probes that require regenerating trajectories. Phase 1/2 must cache the full `x_s` grid, decoded `x0(s)` previews, final completions, fork completions, and internal feature summaries needed for post-hoc readout.
```

## B6. Fix the `A_independent≈1` normalization landmine

Current formula divides by `1 - A_independent`. Add a cap/exclusion rule.

In Phase 1, after the formula, add:

```markdown
**High-`A_independent` cap / video-determined rule.**
If `A_independent(video, axis) ≥ τ_video` (default pilot value: 0.90; sensitivity reported), the per-video denominator is too small for stable normalized gain. Do not compute per-video normalized commitment for that pair. Instead:
1. mark `(video, axis)` as `VIDEO_DETERMINED`;
2. exclude it from the reducible-diversity commitment curve for that axis;
3. report the excluded fraction as a result;
4. run sensitivity over `τ_video`;
5. never let denominator blow-up create artificial early commitment.

Axis-level aggregation before normalization is allowed only if explicitly documented and compared.
```

Add `VIDEO_DETERMINED` to taxonomy and reporting.

## B7. Add richer decision taxonomy

Add a section in Phase 3 or reporting:

```markdown
### Decision taxonomy

Each axis/video or axis-level result should be assigned to one of:
- `VIDEO_DETERMINED`: `A_independent` is near 1; the video alone pins the value.
- `SEED_DETERMINED`: value or failure predictable from initial noise / very early state, before meaningful trajectory commitment.
- `TRAJECTORY_EARLY`: reducible diversity resolves early.
- `TRAJECTORY_LATE`: reducible diversity resolves late.
- `COMMITTED_BUT_UNREADABLE`: commitment exists but available external probes lag; R2.
- `UNRELIABLE_MEASUREMENT`: axis failed determinism/robustness/validity.
```

Use this taxonomy in `commitment_readout_gap_report.md` and in the proposal’s planned Figure 1.

## B8. Strengthen `GO_MAP`: require α-robust ordering

Current `GO_MAP` checks separated commitment windows beyond CIs. Add the PI review requirement:

```markdown
`GO_MAP` requires not only separation at the primary α, but also stable axis ordering across the α pilot grid. Absolute `s_commit` values are kernel-relative; the robust claim is the rank/order and R1/R2 gap structure.
```

Operationalize it:

* Define an `ordering_stability_score`.
* Require the pairwise axis ordering to agree across most valid α values, e.g. pre-register threshold during pilot.
* Report disagreements as sensitivity, not hidden failure.
* Add `ordering_invariance_report.md` or include it inside `commitment_readout_gap_report.md`.

Suggested metric:

```markdown
For each valid α, compute rank order of `s_commit(axis)`. Report Kendall τ or pairwise order agreement against the primary α ordering. `GO_MAP` requires ordering stability above pre-registered threshold and no major reversal among Tier-1 axes.
```

Do not invent numeric threshold unless the current plan already defines one; say it must be frozen before headline curves.

## B9. Add cross-modal coupling ablation

Add a non-blocking but high-priority ablation after primary map:

```markdown
### Coupling-strength ablation

Purpose: make “cross-modal” a variable rather than a fixed setting.

Candidate manipulations:
- sweep CFG scale / conditioning strength;
- drop or weaken Synchformer sync features if MMAudio exposes them;
- compare normal video conditioning vs degraded / weaker conditioning.

Pre-registered prediction:
stronger cross-modal conditioning should pull relevant `s_commit` earlier and may reduce reducible diversity for video-determined axes.

Output:
`coupling_ablation_report.md`
```

This should be P1 after the micro-map and primary map, not a blocker for initial feasibility.

## B10. Add second-checkpoint commitment-only sanity

Current plan says one white-box generator and second generator only later. PI review says the predictable ICLR concern is single-checkpoint specificity. Modify hard constraints:

Old idea:
“One white-box generator for maps; second generator only later transfer sanity.”

New wording:
“MMAudio remains the v1 white-box backbone. After the primary commitment map is working, run a commitment-only sanity check on a second MMAudio size variant / checkpoint if available. Do not run readout or policy on the second checkpoint unless cheap. This tests whether the ordering is stable across scale/checkpoints.”

Add output:

* `checkpoint_commitment_sanity_report.md`

Make it P1 / robustness, not Phase-0 blocker.

## B11. Add causal intervention validation

Add a conditional phase after Phase 3, before or alongside policy:

```markdown
### Phase 3b — Causal commitment intervention

Purpose: test whether observed fork-agreement commitment corresponds to causal irreversibility.

Design:
- choose one reliable axis, preferably coarse class or timing;
- choose ~50 clips;
- choose three intervention points: before measured `s_commit`, near boundary, after measured `s_commit`;
- apply an intervention that pushes `x0(s)` / guidance toward an alternative axis value;
- measure flip rate of the final self-target.

Pre-registered prediction:
flip rate should be higher before `s_commit` and collapse after the boundary.

Output:
`causal_irreversibility_report.md`

Token:
`GO_CAUSAL_COMMITMENT` if intervention asymmetry supports the boundary.
```

Make clear this is small and method-flavored, useful even if full policy is weak.

## B12. Update Phase 4 policy positioning and baselines

Add SMC-ITA as a mandatory baseline / comparator in Phase 4. Also update the policy framing:

* Do not present cascaded BoN as fighting SMC-ITA.
* Present the map as a scheduler for population search.
* Instantiate at least:

  1. cascaded BoN / pruning;
  2. axis-aware resampling schedule inside SMC-style search, if feasible.
* Baselines should include:

  * full BoN;
  * same-compute BoN;
  * random pruning;
  * scalar DiffRS-style rejection;
  * final-score reranking;
  * seed restart;
  * oracle axis-gated pruning;
  * SMC-ITA or SMC-ITA-style intermediate reward baseline;
  * multi-verifier inference-time scaling baseline if implementable or cite as nearest neighbor.

Add warning:

```markdown
Do not claim superiority over SMC-ITA unless matched NFE, matched scoring-call budget, and same candidate pool / comparable pool accounting are satisfied.
```

## B13. Add correctness factorization to Phase 4

In Phase 4 pre-flight, add:

```markdown
Correctness enters through a sidecar only after a value is readable. For factorable axes, correctness is `match(axis value, video/event anchor)`. This matching relation is time-independent; time-dependence lives in readout fidelity. Holistic quality axes that cannot be decomposed into readable values are out of scope for axis-gated correctness pruning.
```

This must be consistent with the proposal.

## B14. Update launch commands

Replace or extend launch commands to include:

1. **Phase 0A micro-map / instrumentation command**
2. **Full Phase 0 command**
3. **Phases 1–3 command with α-robust ordering and high-`A_independent` cap**
4. **Phase 3b causal/internal readout command**
5. **Phase 4 policy command with SMC-ITA baseline**

Suggested first launch command:

```text
/run-experiment "foley-cw Phase 0A (instrumented micro-map sanity): first confirm video-conditioned MMAudio end-to-end trajectory access on one real video, including extract x_s, resume from x_s, compute x0(s), audited s<->t mapping, and provisional nonzero-alpha fork sanity. Add mandatory logging hooks for trajectory cache, decoded x0(s) previews, final/fork completions, and pooled internal features / AV attention summaries. Then run a micro-map sanity check on 12-16 available VGGSound or equivalent clips, coarse-class axis only, s={0.25,0.5,0.75}, K=8 forks, N=8 independent generations for A_independent, one provisional smallest-valid alpha. Check that fork agreement approaches ~1 near s=1, matches A_independent near s=0, tagger/agreement/normalization are non-degenerate, and high-A_independent cases are marked VIDEO_DETERMINED rather than normalized with unstable denominators. This is pipeline sanity only, not publishable evidence and not GO_MAP. Output feasibility_report.md, trajectory_cache_manifest.md, preview_cache_manifest.md, internal_feature_cache_manifest.md, micro_map_sanity_report.md, micro_map_sanity.csv. Emit GO_FULL_PHASE0 / FIX_PIPELINE_CONVENTION / FIX_MEASUREMENT_CHAIN / FIX_SCORE_CONVERSION / NO_TRAJECTORY_ACCESS." — priority: P0
```

Then update the existing Phase 0 command to explicitly say full reliability gate still must run and micro-map does not replace it.

Update Phases 1–3 command to include:

* high-`A_independent` cap rule;
* α-robust ordering / Kendall τ or pairwise order stability;
* full taxonomy;
* internal-feature cache used for post-hoc probes if available;
* no correctness labels.

Add a separate command for internal probes and causal intervention if maps show promise:

```text
/run-experiment "foley-cw Phase 3b (internal readout + causal commitment validation): using cached Phase 1/2 trajectories, train/evaluate simple linear probes on pooled internal features / AV attention summaries to test whether s_read(internal) approaches s_commit and beats x0(s)-probe readout. Then run a small causal intervention on one reliable axis around measured s_commit: before, near, and after boundary; measure final self-target flip rate. Output internal_probe_report.md and causal_irreversibility_report.md. Emit GO_INTERNAL_READOUT / R2_CONFIRMED / GO_CAUSAL_COMMITMENT / CAUSAL_INCONCLUSIVE." — priority: P1
```

Update Phase 4 command to include SMC-ITA:

```text
/run-experiment "foley-cw Phase 4 (conditional map-scheduled search): after GO_MAP + GO_READOUT and policy_preregistration.md, instantiate the map as a scheduler for population search, including cascaded BoN pruning and, if feasible, axis-aware SMC-style resampling. Compare against full BoN, same-compute BoN, random pruning, scalar DiffRS-style rejection, final-score reranking, seed restart, oracle axis-gated pruning, and SMC-ITA / SMC-ITA-style intermediate reward baseline under matched NFE and matched scoring-call budgets. Correctness-vs-video labels are used only in the sidecar for policy evaluation. Output policy_preregistration.md and policy_pruning_report.md. Emit GO_POLICY / GO_RESTRICTED / DIAGNOSTIC_ONLY." — priority: P1
```

## B15. Update downstream agent instructions

The first agent response currently requires Phase 0 outputs. Revise it so the first agent response may be Phase 0A if no experiments have run:

Required first response after Phase 0A:

1. one real-video trajectory access result;
2. audited `s↔t` mapping;
3. provisional α fork sanity;
4. cache hooks installed and what is cached;
5. micro-map sanity endpoints;
6. denominator / `VIDEO_DETERMINED` behavior;
7. emitted token: `GO_FULL_PHASE0` or fix token.

Then full Phase 0 still requires:

1. trajectory access;
2. α=0 and nonzero-α SDE validation;
3. dataset manifest;
4. event anchors;
5. reliability gate;
6. emitted Phase-0 token.

## B16. Update implementation watch-list

Add items:

* Do not normalize per-video commitment when `A_independent` is too close to 1.
* Do not report absolute `s_commit` as universal; report α-robust ordering.
* Do not forget internal-feature logging before large generation.
* Micro-map is pipeline sanity, not evidence.
* Policy must include SMC-ITA or explicitly mark it unimplemented with reason.
* Phase 4 correctness relies on axis-value matching to anchors; holistic quality is out of scope.
* Second checkpoint commitment-only sanity is recommended before a serious conference submission.

---

# Part C — Consistency requirements across both files

After editing, run a self-review pass for consistency.

## C1. Terminology

Use consistently:

* `s`, `s_commit`, `s_read`, `x_s`, `x0(s)`
* “commitment map”
* “readout map”
* “commitment–readout gap”
* “self-target”
* “video-conditioned prior”
* “high-`A_independent` / video-determined”
* “α-robust ordering”
* “internal readout”
* “causal irreversibility”
* “map-scheduled search”

Avoid:

* using `t_commit/t_read` as reported variables;
* saying simply “label-free” without “measurement-dependent”;
* making “axis-gated pruning” sound like the headline;
* claiming policy results before Phase 4;
* claiming internal probes are blocking for Phase 0;
* claiming the method beats SMC-ITA without experiments.

## C2. Claim discipline

Every planned or speculative claim must remain explicitly planned/untested.

Words like “will show,” “demonstrates,” “proves,” “beats,” “is superior” should be replaced by “tests whether,” “is designed to measure,” “would support,” or “if validated.”

## C3. File output lists

Make sure any new reports added in the plan are included in the Files to Produce table:

* `micro_map_sanity_report.md`
* `micro_map_sanity.csv`
* `trajectory_cache_manifest.md`
* `preview_cache_manifest.md`
* `internal_feature_cache_manifest.md`
* `ordering_invariance_report.md` or included in gap report
* `coupling_ablation_report.md`
* `internal_probe_report.md`
* `causal_irreversibility_report.md`
* `checkpoint_commitment_sanity_report.md`

Do not remove existing required files unless truly redundant.

## C4. Decision tokens

Add new tokens where appropriate:

* `GO_FULL_PHASE0`
* `FIX_PIPELINE_CONVENTION`
* `FIX_MEASUREMENT_CHAIN`
* `VIDEO_DETERMINED`
* `GO_INTERNAL_READOUT`
* `R2_CONFIRMED`
* `GO_CAUSAL_COMMITMENT`
* `CAUSAL_INCONCLUSIVE`
* `GO_CHECKPOINT_SANITY`

Do not remove existing tokens unless replaced with compatible aliases.

## C5. Final output from you

After editing, output:

1. a concise summary of what changed in each file;
2. the most important conceptual change;
3. any assumptions made;
4. any TODO citations that need manual BibTeX verification;
5. a quick consistency checklist showing:

   * `s` notation unified;
   * Phase 0A added;
   * internal logging mandatory;
   * `A_independent` cap added;
   * α-robust ordering added;
   * SMC-ITA baseline added;
   * correctness factorization added;
   * policy demoted from headline to downstream scheduler.

Do not run experiments. Do not fabricate results.
