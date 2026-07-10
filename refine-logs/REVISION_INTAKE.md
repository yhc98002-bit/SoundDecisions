# Revision Intake — Round 1

- Target: both (`refine-logs/FINAL_PROPOSAL_SHORT.md` + `refine-logs/EXPERIMENT_PLAN.md`)
- Patch mode: both (explicit user target arg; critique file explicitly instructs editing both documents)
- Decision log used: no (`orbit-research/RESEARCH_DECISION_LOG.md` does not exist; the only diagnostic on record — Phase-0 crux, `diag_20260609_161203_985ce5eccd17` — PASSED, so the failed-diagnostic "both forbidden" rule does not apply)
- Critique source: file — `critic.md` (PI-review-derived revision spec)
- Reviewer backend: Claude CLI (project-local `codex-precondition.md` replaces Codex with Claude CLI reviewer; precondition `claude --version` → 2.1.170 PASS)
- Timestamp: 2026-06-10T03:33:03Z

## Strategic frame (binding, from critique preamble)

Move framing from "axis-gated pruning for V2A / policy method with maps as support" to
"Sound Decisions: a trajectory-level study of when cross-modal Foley decisions become
committed, when they become readable, and how that map schedules inference-time compute."
Preserve the existing measurement backbone (self-target maps, `A_independent`
normalization, Phase-0 reliability gate, smallest-valid-α, ODE-target vs fork-majority
readout, bootstrap-by-video, no premature policy claims). Edit markdown only; no
experiments; no claim-tier upgrades; no invented results.

## Critique Items

| ID | Owning Stage | Affected Artifact(s) | Raw Text (abbrev.) | Suggested Direction |
|----|--------------|----------------------|--------------------|---------------------|
| A1 | 21 (claim/framing) | FINAL_PROPOSAL_SHORT.md | Subtitle overweights pruning | Retitle: "Sound Decisions …"; subtitle = Commitment and Readout Maps; pruning = downstream use case |
| A2 | 21 | FINAL_PROPOSAL_SHORT.md | Missing abstract-style paragraph | Add working abstract near top, marked planned-not-empirical |
| A3 | 21 / consistency | FINAL_PROPOSAL_SHORT.md | Proposal uses raw `t` | Unify to `s`, `s_commit`, `s_read`, `x0(s)`, `gap = s_read − s_commit`; `t` only as MMAudio internal time |
| A4 | 10/21 | FINAL_PROPOSAL_SHORT.md | Novelty under-specified | Sharper novelty sentence + negative-boundary sentence |
| A5 | 9 (lineage) | FINAL_PROPOSAL_SHORT.md | Missing speciation/dynamical-regimes line | Add speciation lineage incl. coupled multimodal speciation theory; TODO citation placeholders |
| A6 | 21 | FINAL_PROPOSAL_SHORT.md | June 2026 landscape shift absent | Add "why policy alone is no longer enough" paragraph (SMC-ITA, multi-verifier AV scaling); careful wording, TODO citations |
| A7 | 21 | FINAL_PROPOSAL_SHORT.md | Internal readout + causal validation buried | Reorganize contributions C1–C5 per PI structure |
| A8 | 5/21 | FINAL_PROPOSAL_SHORT.md | Hypotheses incomplete | H1–H7 incl. internal readout, coupling strength, causal irreversibility, seed-determined; all untested |
| A9 | 14/21 | FINAL_PROPOSAL_SHORT.md | Correctness factorization implicit | Add explicit "Correctness factorization" subsection |
| A10 | 21 | FINAL_PROPOSAL_SHORT.md | METHOD tier requires policy win — too restrictive | New 5-tier table: METHOD/full, SCIENCE/diagnostic strong, CAUSAL diagnostic, NEGATIVE, STOP |
| A11 | 23 | FINAL_PROPOSAL_SHORT.md | Anti-overclaim rules incomplete | Add 5 new rules (s_commit kernel-relative; no SMC-ITA superiority w/o matched budgets; internal probes not storage-free; theory motivates not proves; holistic quality boundary) |
| A12 | 21 | FINAL_PROPOSAL_SHORT.md | No figure plan | Add planned Figure 1 decision taxonomy (5 categories) |
| B2 | 16 | EXPERIMENT_PLAN.md | Plan-change provenance missing | Add "PI Review Amendments" section near top (10 items) |
| B3 | 16 | EXPERIMENT_PLAN.md | No cheap end-to-end sanity before full Phase 0 | Insert Phase 0A micro-map (12–16 clips, coarse class, s∈{0.25,0.5,0.75}, K=8, N=8); new fix tokens; does not replace reliability gate |
| B4 | 16 | EXPERIMENT_PLAN.md | Internal probes Phase-7-only = too late | Mandatory internal-feature logging before any large Phase 1/2 generation; analysis non-blocking; retitle Phase 7 → Phase 3b/7 |
| B5 | 16 | EXPERIMENT_PLAN.md | No cache requirements | Cache full `x_s` grid, `x0(s)` previews, completions, fork completions, internal features; 3 new manifests; hard constraint: probes must be post-hoc |
| B6 | 14/16 | EXPERIMENT_PLAN.md | `1 − A_independent` denominator landmine | High-`A_independent` cap τ_video (pilot default 0.90, sensitivity); VIDEO_DETERMINED exclusion rule |
| B7 | 16 | EXPERIMENT_PLAN.md | Taxonomy too coarse | 6-category decision taxonomy (VIDEO_DETERMINED, SEED_DETERMINED, TRAJECTORY_EARLY/LATE, COMMITTED_BUT_UNREADABLE, UNRELIABLE_MEASUREMENT) |
| B8 | 16 | EXPERIMENT_PLAN.md | GO_MAP ignores α-dependence | Require α-robust ordering: ordering_stability_score (Kendall τ / pairwise agreement), pre-registered threshold, ordering report |
| B9 | 11/16 | EXPERIMENT_PLAN.md | Cross-modal is fixed, not a variable | Coupling-strength ablation (CFG sweep, sync-feature drop); P1 non-blocking; pre-registered prediction |
| B10 | 11/16 | EXPERIMENT_PLAN.md | Single-checkpoint specificity | Second MMAudio checkpoint commitment-only sanity, P1 robustness; new report |
| B11 | 16 | EXPERIMENT_PLAN.md | Commitment only observational | Phase 3b causal intervention (~50 clips, 3 intervention points, flip-rate asymmetry); GO_CAUSAL_COMMITMENT |
| B12 | 7/11 | EXPERIMENT_PLAN.md | Phase 4 baselines stale vs June 2026 | Add SMC-ITA + multi-verifier baselines; map = scheduler for population search; matched-NFE/scoring-budget warning |
| B13 | 16 | EXPERIMENT_PLAN.md | Phase 4 correctness logic implicit | Correctness factorization in Phase 4 preflight, consistent with proposal A9 |
| B14 | 16 | EXPERIMENT_PLAN.md | Launch commands stale | 0A / full-0 / 1–3 / 3b / 4 command set per critique |
| B15 | 16 | EXPERIMENT_PLAN.md | First-response contract assumes Phase 0 | First response may be Phase 0A (7-point checklist); full Phase 0 contract unchanged |
| B16 | 16 | EXPERIMENT_PLAN.md | Watch-list missing new landmines | 7 new watch-list items |
| C1–C5 | 23 (consistency) | both | Terminology, claim discipline, file lists, tokens, final checklist | Cross-file consistency pass after edits |

Constraint item (not an edit): **B1** — preserve the existing protocol backbone verbatim in
spirit; add safeguards, do not destabilize.

## Stages To Re-run (Phase 1 plan)

The project's v1.3 artifact set is partial (no ASSUMPTION_LEDGER / ALGORITHM_TOURNAMENT /
BASELINE_CEILING files exist); every critique item's affected artifact IS one of the two
target documents. Phase 1 therefore performs direct in-place targeted revision of:

- `refine-logs/FINAL_PROPOSAL_SHORT.md` (A1–A12, C1–C5)
- `refine-logs/EXPERIMENT_PLAN.md` (B2–B16, C1–C5)

with `.prerevise.1` snapshots written first. No upstream orbit-research stage artifacts
require regeneration. Canonical packs (`proposal/proposal_pack.json`,
`experiment/experiment_pack.json`, `experiment/EXPERIMENT_PLAN_EXEC.md`) are NOT modified
by this skill; if the user accepts the revision, re-sync belongs to `/experiment-bridge`.

## Ambiguities logged (AUTO_PROCEED defaults applied)

1. Critique names `FINAL_PROPOSAL.md`-style proposal target; only `FINAL_PROPOSAL_SHORT.md`
   exists → treated as the proposal target (matches critique Part A title and
   ORBIT_STATE legacy_artifacts list).
2. Critique B4 retitles Phase 7 "Phase 3b / Phase 7" while B11 adds a causal "Phase 3b" →
   resolved as one Phase 3b with two subsections (3b.1 causal intervention, 3b.2 internal
   readout analysis = former Phase 7), since critique C5/B14 treat "Phase 3b" as one
   command covering both.
3. Evidence-status sections say "fresh-start / nothing empirical"; ORBIT_STATE records a
   PASSED Phase-0 crux diagnostic (2026-06-09). Critique forbids claiming validation and
   does not request evidence-status edits → evidence-status text left unchanged; staleness
   flagged for the human checkpoint instead of edited.

## Anchor (for Phase 2)

No PROBLEM_SELECTION.md exists. Anchor = pre-revision proposal §2 Core Scientific Question
+ §1 one-sentence thesis ("when does the generator commit to each Foley output axis, and
when can that commitment be read out"), cross-checked against `proposal/proposal_pack.json`.

## Phase 2 Check Results (independent Claude CLI reviewer, opus/effort-max, fresh session)

- Raw output: `orbit-research/codex-imports/proposal-revise-r1-phase2.response.json`
- `anchor_check`: **ANCHOR_PASS** — revised pair still targets the anchored problem; SMC-ITA
  positioning, internal readout (C3), and causal validation (C4) judged on-anchor (C3 =
  readout half made sharper; C4 = validation of the commitment half). No factual→working
  assumption downgrades without tests; mechanism family unchanged.
- `simplicity_check`: **SIMPLICITY_VIOLATION (controlled, PI-mandated; remediated by
  re-ranking, not revert)** —
  - Trainable components: PASS (exactly 2: internal linear probes + conditional Phase-5
    verifier; unchanged from pre-revision).
  - Primary claims: revised docs read as 3 co-equal pillars (map / internal readout /
    causal). Remediation applied in place of revert (reverting would reject binding
    critique items A7/A10/B11): explicit 2-primary-claim ranking added to proposal §0;
    C3/C4 re-labeled as subordinate evidence; tier-ladder note added to §8 (fallback
    ladder, not co-equal claims).
  - Removable mechanisms (causal 3b.2, coupling 3c, second-checkpoint 3d): kept — each is
    explicitly conditional/P1, adds no trainable mass, and is directly mandated by the
    binding critique (B9/B10/B11). Recorded as accepted, PI-mandated additions.
- Consistency audit fixes applied: H3 "beats"→"precedes" (proposal + plan 3b.1 prose;
  launch-command wording kept verbatim per critique B14, hedged by "test whether");
  dangling standalone `ordering_invariance_report.md` removed (analysis lives inside
  `commitment_readout_gap_report.md`, allowed by critique C3); `NO_TRAJECTORY_ACCESS`
  added to proposal §13 token list; causal-validation-not-definition note added to plan
  3b.2 and proposal C4.

Final per-item status: all critique items **addressed = yes** (A7/A10/B11 with the
simplicity re-ranking remediation noted above).
