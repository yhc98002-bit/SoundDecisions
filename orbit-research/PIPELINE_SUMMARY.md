# Pipeline Summary — foley-cw (SoundDecisions)

- **Date:** 2026-06-09
- **Action:** Froze the human-authored proposal + experiment plan and backfilled the
  canonical ORBIT artifacts so STOP B / experiment execution can proceed. This was a
  **faithful backfill**, not a skill rerun: no `/idea-to-proposal`, `/experiment-bridge`,
  or `/experiment-plan` was run, no external reviewer was called, and **no content was
  invented** — every pack field traces to a line in the frozen Markdown.

## Frozen sources (authoritative; record-only freeze, files left at 0644)

- `refine-logs/FINAL_PROPOSAL_SHORT.md` — the Final Proposal
- `refine-logs/EXPERIMENT_PLAN.md` — the operational experiment plan
- Freeze record: `orbit-research/PROPOSAL_STABILITY.md`

## Canonical artifacts produced (structured index of the frozen sources)

- `proposal/proposal_pack.json` (status `ready`) + `proposal/PROPOSAL.md`, `proposal/METHOD_SPEC.md` (generated views)
- `experiment/experiment_pack.json` (status `ready`) + `experiment/EXPERIMENT_PLAN.md`, `experiment/EXPERIMENT_PLAN_EXEC.md` (generated views)
- `orbit-research/CONTROL_DESIGN.md`, `NULL_RESULT_CONTRACT.md`, `COMPONENT_BUNDLE_LADDER.md`, `ALGORITHMIC_FORMALIZATION.md` (STOP-C gate docs)
- `orbit-research/ORBIT_STATE.json` (pinned state)
- Tooling provisioned: `tools/{orbit_pack,validate_orbit_pack,orbit_status,orbit_state,orbit_verdicts,check_stop_c_approval}.py` + `schemas/*.json`

## Current state

- STOP: **STOP_B**, paused (`missing_prereq`).
- `experiment_pack.formal_diagnostics`: two registered diagnostics, both `status=pending`,
  `owner=human` — `diag_phase0_feasibility` (Phase 0 STRICT gate) and
  `diag_maps_phases123` (Phases 1–3 make-or-break maps).
- **No code implemented and no results exist yet.**

## Next steps (for a later agent / human)

1. `/experiment-bridge "experiment/experiment_pack.json" — mode: audit-only`
   → implement the planned code and produce `orbit-research/PLAN_CODE_AUDIT.md`
   (verdict `MATCHES_PLAN`).
2. After STOP B review: `/diagnostic-to-review "experiment/experiment_pack.json"`
   → run `diag_phase0_feasibility` first (must emit `GO_MAPS_PHASE`), then
   `diag_maps_phases123` for the `GO_MAP` + `GO_READOUT` make-or-break, writing
   `claims/claim_ledger.json`.

Check state anytime with: `python3 tools/orbit_status.py --repo . --pretty`.

---

# /experiment-bridge Summary (STOP B — 2026-06-09)

- Input: `experiment/experiment_pack.json` — Mode: **audit-only**
- Proposal: `proposal/proposal_pack.json`; Experiment pack: `experiment/experiment_pack.json` (`ready`)
- Plan views: `experiment/EXPERIMENT_PLAN.md`, `experiment/EXPERIMENT_PLAN_EXEC.md`
- **Implementation: `foley_cw/` package implemented** (numpy-only import; heavy/MMAudio deps lazy).
  - Core/crux authored + CPU-verified: `types`, `time_map` (s↔t seam), `model_adapter`
    (`MMAudioBackend` raises), `synthetic_backend` (analytic Gaussian oracle), `score_sde`
    (velocity↔score, Tweedie, marginal-preserving fork), `config` + `configs/*.json`.
  - Modules: `agreement, axes, stats, dataset, probes, validation, reporting, commitment,
    readout, reliability, gap` + `cli/{phase0_feasibility, phases123_maps}` + `tests/` (515 pass).
- **Audit: `orbit-research/PLAN_CODE_AUDIT.md` — verdict `MATCHES_PLAN`** (Codex GPT xhigh,
  5 rounds; 18 findings fixed). Mirrored into `experiment_pack.plan_code_audit`.
- Probes: NONE (audit-only). Formal diagnostics: NOT RUN BY THIS SKILL.
- Scope: planned code validated on the synthetic backend; **MMAudio seam UNVERIFIED — pin in
  the Phase-0 GPU diagnostic (owner=human)**. No GPU job, no MMAudio call, no fabricated results.

## STOP B

Review: `experiment/experiment_pack.json`, `experiment/EXPERIMENT_PLAN.md`,
`experiment/EXPERIMENT_PLAN_EXEC.md`, `orbit-research/PLAN_CODE_AUDIT.md`, `foley_cw/`.

Human question: Is this code/plan good enough to authorize the Phase-0 GPU feasibility diagnostic?

## Next

`/diagnostic-to-review "experiment/experiment_pack.json"` — runs `diag_phase0_feasibility`
first (must emit `GO_MAPS_PHASE`), then `diag_maps_phases123`. Requires explicit GPU approval
(SSH nodes `an17`/`an22`) and MMAudio wiring.
