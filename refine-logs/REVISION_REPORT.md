# Revision Report — Round 1

- Target: both (`refine-logs/FINAL_PROPOSAL_SHORT.md` + `refine-logs/EXPERIMENT_PLAN.md`)
- Critique source: file — `critic.md` (PI-review-derived revision spec, Parts A/B/C)
- Round: 1 of MAX_ROUNDS=2 (no second round needed; Phase-3 re-eval returned SCORE_DELTA: POSITIVE)
- Started: 2026-06-10T03:33:03Z · Completed: 2026-06-10T03:59:30Z
- Reviewer backend: Claude CLI (`claude -p --model opus --effort max`, fresh sessions), per
  project-local `codex-precondition.md` (Claude-CLI-reviewer override of the Codex contract)

## Critique Resolution

| ID | Owning Stage | Addressed | Re-eval (d1/d2/d3) | Notes |
|----|--------------|-----------|--------------------|-------|
| A1/A2 | 21 | yes | 5/5/5 | Retitled "Sound Decisions"; subtitle = Commitment and Readout Maps; working abstract added, marked planned-not-empirical |
| A3 | 21 | yes | 5/5/5 | `s` notation unified everywhere (15 raw-`t` instances → 0); `t` defined as MMAudio-internal only |
| A4/A5 | 9/21 | yes | 5/5/5 | New novelty sentence + negative boundary; 4th lineage (speciation / coupled multimodal theory) with TODO-flagged citations |
| A6 | 21 | yes | 5/5/5 | "Why policy alone is no longer enough" (SMC-ITA, multi-verifier) — policy = table-stakes, claim orthogonal |
| A7 | 21 | yes | 5/5/5 | C1–C5 restructured; internal readout (C3) + causal validation (C4) first-class; measurement protocol folded into C1 |
| A8 | 5 | yes | 5/5/5 | H1–H7 (internal readout, coupling, causal irreversibility, seed-determined); all marked untested |
| A9 | 14 | yes | 5/5/5 | New proposal §7 correctness factorization; mirrored in plan Phase-4 preflight |
| A10/A11/A12 | 21/23 | yes | 5/5/5 | 5-tier claim ladder (METHOD/SCIENCE/CAUSAL/NEGATIVE/STOP); 5 new anti-overclaim rules; planned Figure-1 taxonomy |
| B2/B3 | 16 | yes | 5/5/4 | PI-amendments section (10 items); Phase 0A micro-map with GO_FULL_PHASE0 / FIX_* tokens; sanity-only, never licenses GO_MAP |
| B4/B5 | 16 | yes | 5/5/5 | Internal-feature logging mandatory before large Phase 1/2 generation; 3 cache manifests; "no probe may require regeneration" hard constraint |
| B6/B7/B8 | 14/16 | yes | 5/5/5 | τ_video=0.90 cap + VIDEO_DETERMINED exclusion; 6-category decision taxonomy; α-robust ordering (Kendall τ, pre-registered θ_order) required for GO_MAP |
| B9/B10/B11 | 11/16 | yes | 5/5/5 | Phase 3c coupling ablation, Phase 3d second-checkpoint sanity, Phase 3b.2 causal intervention — all conditional/P1, pre-registered predictions |
| B12/B13 | 7/11/16 | yes | 5/5/5 | SMC-ITA + multi-verifier baselines mandatory; map-as-scheduler framing; matched-NFE/scoring-budget no-superiority warning; Phase-4 correctness factorization |
| B14/B15/B16 | 16 | yes | 5/5/5 | 5 launch commands (0A → 4); Phase-0A first-response contract (7 points); watch-list 10 → 17 items |
| C1–C5 | 23 | yes | all 8 checklist items YES | Independent reviewer confirmed full consistency checklist (see below) |

No critique item was rejected. No reverts were applied (anchor check passed; the simplicity
finding was remediated by re-ranking, not revert — see below).

## Phase 2 — Anchor + Simplicity (independent reviewer, fresh session)

- **ANCHOR_PASS** — revised pair still targets the anchored problem (when do Foley axes
  commit / become readable; policy strictly downstream). Internal readout judged "the
  readout half made sharper"; causal intervention judged "validation of the commitment
  half". Mechanism family unchanged; no untested assumption downgrades.
- **SIMPLICITY_VIOLATION (controlled; remediated in place rather than reverted)** —
  trainable-component budget PASSES (exactly 2, unchanged: internal linear probes +
  conditional Phase-5 verifier). The violation was presentational: the docs read as 3
  co-equal headline pillars (map / internal readout / causal). Reverting would have
  rejected binding critique items A7/A10/B11, so the reviewer's recommended re-ranking was
  applied instead: explicit 2-primary-claim budget in proposal §0 (claim 1 = map + gap,
  evidenced by C3/C4; claim 2 = map-scheduled compute), C3/C4 re-labeled subordinate
  evidence, tier-ladder note in §8 (fallback ladder ≠ co-equal claims), causal framed as
  validating — never redefining — fork-agreement commitment.
- Consistency fixes from the same review: H3 "beats" → "precedes"; dangling standalone
  `ordering_invariance_report.md` removed (lives inside `commitment_readout_gap_report.md`,
  allowed by critique C3); `NO_TRAJECTORY_ACCESS` added to proposal §13 token list.
- Raw output: `orbit-research/codex-imports/proposal-revise-r1-phase2.response.json`

## Phase 3 — Independent Re-evaluation (fresh session, reviewer-independence protocol)

- **SCORE_DELTA: POSITIVE** — every critique group d1/d2/d3 ≥ 4; no group ≤ 3; no gap
  analysis triggered; Part C5 consistency checklist all 8 items YES (s-notation unified;
  Phase 0A added; internal logging mandatory; A_independent cap; α-robust ordering;
  SMC-ITA baseline; correctness factorization; policy demoted to downstream scheduler).
- Reviewer also confirmed: backbone fully preserved; no fabricated/asserted-verified
  citations (all new ones TODO-flagged); the revision repaired a pre-existing broken
  cross-reference (plan pointed to nonexistent `FINAL_PROPOSAL.md`).
- Raw output: `orbit-research/codex-imports/proposal-revise-r1-phase3.response.json`

## Re-integration scope note

Phase 3 re-integration was performed as direct in-place revision instead of delegating to
`/research-refine` + `/experiment-plan` wholesale regeneration: the critique is itself a
fully-specified patch spec that orders "edit markdown only / preserve the backbone", and
this project's v1.3 artifact set is partial (the affected artifacts ARE the two target
documents). Canonical packs (`proposal/proposal_pack.json`, `experiment/experiment_pack.json`,
`experiment/EXPERIMENT_PLAN_EXEC.md`, `proposal/PROPOSAL.md`, `proposal/METHOD_SPEC.md`)
were **not** modified; if you accept this revision, re-sync them via `/experiment-bridge`.

## Artifact Diffs (lines added / removed)

- `refine-logs/FINAL_PROPOSAL_SHORT.md`: +107 / −50 (159 → ~216 lines)
- `refine-logs/EXPERIMENT_PLAN.md`: +256 / −60 (332 → ~528 lines)
- Milestone snapshots: `FINAL_PROPOSAL_SHORT_20260610_115914.md`, `EXPERIMENT_PLAN_20260610_115914.md`
- Pre-revision snapshots (for diff/revert): `*.prerevise.1`

## TODO citations needing manual BibTeX verification

1. Speciation / dynamical-regimes line: *Biroli et al.; Bonnaire / de Bortoli / Mézard;
   Raya & Ambrogioni; Georgiev et al.; Sclocchi et al.* (proposal §4.1)
2. *Coupled multimodal speciation theory, 2026* (proposal §4.1)
3. *SMC-ITA* — SMC inference-time alignment for V2A (proposal §0/§4.2; plan Phase 4)
4. *Multi-verifier inference-time scaling for joint AV generation* (proposal §0/§4.2; plan Phase 4)

## Items flagged for the human (not critique gaps)

1. **Stale evidence-status lines.** Both documents still say "fresh-start / no
   project-specific experiment has been run / nothing empirical", but
   `orbit-research/ORBIT_STATE.json` + `results/` record a PASSED Phase-0 crux diagnostic
   (trajectory access, α=0 + nonzero-α SDE validation on real MMAudio, 2026-06-09/10).
   The critique ordered "do not claim anything has been validated", so the status lines
   were deliberately left frozen (under-claiming, not over-claiming); plan §1b carries a
   pointer to the diagnostic record. Decide whether to reconcile these lines before
   circulating the documents.
2. **`experiment/EXPERIMENT_PLAN.md` (canonical pack view, 60 lines) now lags
   `refine-logs/EXPERIMENT_PLAN.md`.** Re-sync via `/experiment-bridge` after acceptance.

## Next steps

- Review the regenerated `FINAL_PROPOSAL_SHORT.md` and `EXPERIMENT_PLAN.md` (diff against
  `*.prerevise.1` if desired).
- If satisfied, invoke `/experiment-bridge "refine-logs/EXPERIMENT_PLAN.md"` (STOP B) to
  re-sync the experiment pack and proceed; the Phase-0A micro-map command in plan §10 is
  the first runnable step (GPU work still requires your explicit approval).
- If still dissatisfied, invoke `/proposal-revise` again with new critique points
  (Round 2 will be triggered; already-addressed items are idempotently skipped).
