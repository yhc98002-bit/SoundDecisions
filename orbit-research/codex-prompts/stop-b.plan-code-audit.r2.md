ROLE: Semantic plan-code audit reviewer (STOP B), ROUND 2. Independent cross-model reviewer.
Be skeptical and specific. Do NOT improve the code; JUDGE plan-fidelity. Re-derive from the
code; do not assume the fixes are correct just because they are described below.

REPO: /XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions

CONTEXT: `foley-cw` STOP-B audit-only implementation (planned code only; validated on CPU vs
an analytic synthetic backend; MMAudio/GPU intentionally absent, MMAudio adapter is a
documented raising stub). Round 1 returned CRITICAL_MISMATCH with 6 findings. They were
addressed as follows — VERIFY each is genuinely fixed and that no regression or NEW
divergence was introduced:

1. foley_cw/validation.py run_sde_validation: token now "OK" only if alpha=0 reproduction,
   small-alpha continuity, exact score (synthetic), marginal preservation (synthetic), fork
   validity AND nontrivial diversity all pass; score-conversion failures -> FIX_SCORE_CONVERSION,
   nonzero-alpha operating-point failures -> FORK_ALPHA_NO_VALID_OPERATING_POINT. foley_cw/gap.py
   decide_phase0 now blocks GO_MAPS_PHASE on ANY non-"OK" token.
2. foley_cw/cli/phases123_maps.py now runs the three-part reliability gate and builds maps
   ONLY for axes that pass (demoted axes dropped, recorded in the gap report).
3. foley_cw/commitment.py build_commitment_map and foley_cw/readout.py build_readout_map take
   min_n_per_axis (from configs/dataset.json min_usable_n_per_axis) and pass it to
   window_with_ci, so underpowered axes are flagged.
4. foley_cw/cli/phases123_maps.py warns when thresholds.frozen is False and records the
   non-binding status in go_no_go_decision.md / the gap report.
5. foley_cw/gap.py decide_phase3 GO_READOUT now requires the axis to commit early
   (s_commit < near-s1), read before the end, and read a committed (not uncommitted) axis with
   a bounded gap; it is mutually exclusive with STOP_ADSR.
6. foley_cw/reporting.py write_commitment_map_csv / write_readout_map_csv now append s_commit /
   s_read with CIs (the CLI passes the window dicts), per plan §5.

AUTHORITATIVE SPEC: refine-logs/EXPERIMENT_PLAN.md, experiment/experiment_pack.json,
experiment/EXPERIMENT_PLAN_EXEC.md.

RE-AUDIT the full implementation in foley_cw/ (all modules + cli/ + configs/ + tests/). For
EACH of the 6 items above, state RESOLVED / PARTIAL / NOT-RESOLVED with file:line evidence.
Then re-check the rest of the plan-fidelity surface (velocity->score convention and sign; s<->t
seam and MMAudio direction unverified; commitment normalized over A_independent; Restart
re-noising absent from the commitment kernel; readout uses x0(s) with ODE + fork-majority
targets; probe ladder legacy/headline labels and heavy probes as raising stubs; maps target the
model self-target NOT correctness-vs-video; bootstrap unit = video; MMAudioBackend raises rather
than fabricates; no model-output-as-ground-truth). Report any NEW issue introduced by the fixes.

OUTPUT FORMAT (exactly):
VERDICT: <MATCHES_PLAN | PARTIAL_MISMATCH | CRITICAL_MISMATCH | ERROR>
RESOLUTION OF ROUND-1 FINDINGS: items 1-6 each RESOLVED/PARTIAL/NOT-RESOLVED with file:line.
NEW OR REMAINING FINDINGS: numbered (file:line, plan requirement, code behavior, severity).
WHAT IS CORRECT: short balanced list.
MATCHES_PLAN = faithful with at most minor notes; PARTIAL_MISMATCH = scoped issues not blocking
the synthetic-validated path; CRITICAL_MISMATCH = a real method/control/metric divergence;
ERROR = could not audit.
