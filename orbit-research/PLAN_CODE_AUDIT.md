# Plan-Code Audit — foley-cw (STOP B, audit-only)

## VERDICT: MATCHES_PLAN

Semantic plan↔code audit of the `foley_cw` implementation against the frozen plan
(`refine-logs/EXPERIMENT_PLAN.md`, `experiment/experiment_pack.json`). Required cross-model
reviewer: **Codex (GPT, `model_reasoning_effort=xhigh`)**, run via `codex exec` per
`CLAUDE.md`. The final (round 5) Codex verdict is **MATCHES_PLAN** — "No critical or major
plan-fidelity issues remain."

- Mode: `audit-only` (no GPU, no MMAudio, no formal diagnostics).
- Tests at finalization: `PYTHONPATH=$PWD python -m pytest tests/ -q` → **515 passed**.
- Synthetic CPU dry-runs (no GPU/MMAudio) run end-to-end and emit consistent tokens:
  Phase 0 → `GO_MAPS_PHASE`; Phases 1-3 (default, config min-n, n=6 → underpowered) →
  `STOP_PROJECT`; Phases 1-3 (`--min-n 2`, exercising the full path) →
  `STOP_ADSR, GO_READOUT, GO_DIAGNOSTIC` (internally consistent).

## Scope and audit boundary (what is VERIFIED vs UNVERIFIED)

This is planned code validated on an **analytic synthetic flow backend**
(`synthetic_backend.SyntheticGaussianFlow`), under which the highest silent-bug-risk math is
checked to machine precision on CPU. Everything MMAudio-specific is isolated behind a seam and
is **UNVERIFIED — to be pinned in the Phase-0 GPU diagnostic (owner=human)**, NOT in this
audit-only bridge:

- s↔t integration direction — `time_map.MMAUDIO_S_TO_T` (marked unverified).
- `v_theta`, latent decode — `model_adapter.MMAudioBackend` (raises `MMAudioNotWired`; does
  not fabricate).
- velocity→score sign/parameterization for MMAudio — `score_sde.score_from_velocity` (the
  rectified-linear branch is exact vs the analytic backend; the MMAudio branch must be audited
  in Phase 0.2).

## Verified correct (Codex "what is correct", confirmed across rounds)

- velocity→score conversion `score=(t·v−x)/(1−t)` is exact vs the analytic score; Tweedie
  `x0=x+(1−t)·v`; marginal-preserving SDE drift `v+½σ²·score` (α=0 ⇒ ODE).
- Commitment normalized as gain over `A_independent` (never raw fork agreement); `A_independent`
  is the alpha-independent video prior, computed once per (video,axis) and stable across the α
  surface; Restart re-noising is absent from the commitment kernel.
- Readout uses decoded Tweedie `x0(s)` and reports both `ode` and `fork_majority` self-targets;
  probe ladder labels CLAP/SyncNet/ImageBind legacy and MLLM-on-preview headline; heavy probes
  are raising stubs.
- Maps target the model's own self-target, NOT correctness-vs-video (no model-output-as-ground-
  truth anywhere).
- Bootstrap unit = video for s_commit, s_read, and gaps; pre-registered thresholds carried
  (with an unfrozen-placeholder caveat); maps are reliability-gated; underpowered axes are
  reported as underpowered, not results.
- Phase-0 gate (`decide_phase0`) emits `GO_MAPS_PHASE` only when trajectory access, full SDE
  validation, manifest, and ≥3 reliable axes all hold. Phase-3 tokens are internally consistent
  (`STOP_ADSR` ⟂ `GO_MAP`; `GO_READOUT`/`GO_RESTRICTED` require an early-committed, readable
  axis).

## Audit history (cross-model, iterative; full responses under `orbit-research/codex-imports/`)

| Round | Verdict | Findings | Disposition |
|------|---------|----------|-------------|
| 1 | CRITICAL_MISMATCH | 6 (2 critical, 4 major) | all fixed |
| 2 | CRITICAL_MISMATCH | round-1 resolved; 4 new (1 crit, 2 med, 1 low) | all fixed |
| 3 | CRITICAL_MISMATCH | round-2 resolved; 3 (1 crit + 2 major) | all fixed |
| 4 | PARTIAL_MISMATCH | round-3 resolved; 3 major (new reporting code) | all fixed |
| 5 | **MATCHES_PLAN** | round-4 resolved; 2 minor | both fixed post-round-5 |

Representative fixes: SDE-validation token now gates on all nonzero-α checks (was α=0 +
continuity only); Phases 1-3 consume the reliability gate; `decide_phase3` excludes
underpowered windows and recomputes separation over result windows only (no underpowered
outlier can drive `GO_MAP`); `GO_MAP`⟂`STOP_ADSR`; `GO_READOUT`/`GO_RESTRICTED` require an
early-committed, readable axis; commitment-map CSV carries real `A_fork`/`A_independent` and
`s_commit`+CIs; readout CSV carries `s_read`+CIs; gap CIs are bootstrap-over-videos with the
same crossing definition as the windows; threshold sensitivity re-reports separation under a
θ_commit sweep (result-filtered); empty calibration sidecar → validity NaN → gate fails.

### Round-5 minor notes (resolved after the MATCHES_PLAN verdict)

1. The legacy single-axis `threshold_sweep` section in `cli/phases123_maps.py` did not apply
   `min_n`; it was redundant with the result-filtered separation sensitivity and has been
   removed from the report. (Resolved.)
2. `gap.decide_phase3` `GO_RESTRICTED` now requires the presence/timing axes to be readable
   early (actionable), not merely committed early. (Resolved; added a negative test.)

## Next

This clears blocker G11. STOP B is awaiting human review of the plan/code. Safe next command
(human-gated; runs the GPU Phase-0 feasibility diagnostic first):

```
/diagnostic-to-review "experiment/experiment_pack.json"
```

Formal diagnostics (`diag_phase0_feasibility`, then `diag_maps_phases123`) require MMAudio
wiring + GPU and are owner=human; they are NOT run by this audit-only bridge.
