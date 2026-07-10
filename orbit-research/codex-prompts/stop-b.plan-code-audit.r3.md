ROLE: Semantic plan-code audit reviewer (STOP B), ROUND 3. Independent cross-model reviewer.
Be skeptical and specific; re-derive from the code, do not trust this summary. Do NOT improve
the code; JUDGE plan-fidelity.

REPO: /XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions

CONTEXT: `foley-cw` STOP-B audit-only implementation (planned code only; CPU-validated vs an
analytic synthetic backend; MMAudio/GPU intentionally absent; MMAudio adapter is a raising
stub). Round 1 = CRITICAL_MISMATCH (6 findings, all fixed). Round 2 = CRITICAL_MISMATCH: round-1
items 1,2,4,5,6 RESOLVED, item 3 PARTIAL, plus 4 new findings. The 4 round-2 findings were
addressed — VERIFY each from the code and check for regressions:

R2-1 (was CRITICAL): foley_cw/gap.py decide_phase3 now excludes underpowered windows from
  BOTH valid_commit and valid_read (`not np.isnan(w.s_hat) and not w.underpowered`), so
  underpowered axes can no longer drive GO_MAP/GO_READOUT; all-underpowered -> STOP_PROJECT.
R2-2 (was MEDIUM): foley_cw/commitment.py build_commitment_map now precomputes the base alpha=0
  trajectory states AND A_independent ONCE per (video,axis) and reuses them across the whole
  alpha sweep, so A_independent no longer varies by alpha (verified: one distinct a_independent
  value per axis across the commitment_map.csv surface).
R2-3 (was MEDIUM): foley_cw/reliability.py validity() returns NaN (not 1.0) for an empty
  sidecar, and reliability_gate treats a non-finite det/rob/val score as a FAILURE
  (`not np.isfinite(...) or ... < theta`).
R2-4 (was LOW): foley_cw/cli/phases123_maps.py excludes axes with requires=="two_event_clips"
  (Tier-3 binding) from the single-event synthetic dry-run.

Also note: phases123_maps.py gained a --min-n override (default None -> per-axis config
minimums); with the default a small dry-run correctly reports underpowered -> STOP_PROJECT.

AUTHORITATIVE SPEC: refine-logs/EXPERIMENT_PLAN.md, experiment/experiment_pack.json,
experiment/EXPERIMENT_PLAN_EXEC.md.

RE-AUDIT the whole implementation (foley_cw/ modules + cli/ + configs/ + tests/). For each of
R2-1..R2-4 state RESOLVED/PARTIAL/NOT-RESOLVED with file:line. Re-confirm the round-1 items
stayed resolved. Then sweep the full plan-fidelity surface again for ANY remaining or newly
introduced divergence (score conversion sign/convention; s<->t seam + MMAudio unverified;
commitment normalized over A_independent and Restart-free; readout x0(s) with ode+fork_majority;
probe ladder legacy/headline + raising heavy stubs; self-target not correctness-vs-video;
bootstrap unit = video; pre-registered thresholds; reliability-gated maps; MMAudio raises).

OUTPUT FORMAT (exactly):
VERDICT: <MATCHES_PLAN | PARTIAL_MISMATCH | CRITICAL_MISMATCH | ERROR>
RESOLUTION OF ROUND-2 FINDINGS: R2-1..R2-4 each RESOLVED/PARTIAL/NOT-RESOLVED with file:line.
REMAINING OR NEW FINDINGS: numbered (file:line, plan requirement, code behavior, severity), or
"none".
WHAT IS CORRECT: short balanced list.
Decision rule: MATCHES_PLAN = faithful with at most MINOR notes (no critical/major open);
PARTIAL_MISMATCH = only scoped non-blocking issues remain; CRITICAL_MISMATCH = a real
method/control/metric divergence remains.
