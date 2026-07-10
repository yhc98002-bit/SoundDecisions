ROLE: Semantic plan-code audit reviewer (STOP B), ROUND 5. Independent cross-model reviewer.
Re-derive from code; do not trust this summary. JUDGE plan-fidelity; do not improve code.

REPO: /XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions

CONTEXT: `foley-cw` STOP-B audit-only implementation (planned code only; CPU-validated vs an
analytic synthetic backend; MMAudio/GPU absent; MMAudio adapter raises). History: R1 CRITICAL
(6 fixed), R2 CRITICAL (4 fixed), R3 CRITICAL (3 fixed), R4 PARTIAL_MISMATCH with 3 MAJOR
reporting findings, now addressed — VERIFY from code:

R4-1 (gap definition mismatch): foley_cw/stats.py bootstrap_gap_ci now computes per-video
  crossings first and averages over videos (mean-of-per-video-crossings), matching
  window_with_ci, so gap_point == s_read.s_hat - s_commit.s_hat (verified numerically: 4/4
  result windows matched).
R4-2 (sensitivity ignored underpowered): foley_cw/stats.py separation_under_thresholds now
  takes min_n_per_axis, builds windows with the per-axis min_n, and computes separation /
  ordered_non_overlapping over RESULT windows only (non-NaN, not underpowered); the CLI passes
  min_n_per_axis. With config minimums a small run yields NaN separation across the sweep.
R4-3 (gap CIs for underpowered windows): the CLI computes gap CIs only for result windows
  (skips underpowered / no-crossing), and foley_cw/reporting.py excludes underpowered/undefined
  windows from the Gap section (showing an "excluded as underpowered — not results" note),
  while still listing them with the underpowered flag in the window tables.

AUTHORITATIVE SPEC: refine-logs/EXPERIMENT_PLAN.md, experiment/experiment_pack.json,
experiment/EXPERIMENT_PLAN_EXEC.md.

RE-AUDIT the whole implementation (foley_cw/ + cli/ + configs/ + tests/). For R4-1..R4-3 state
RESOLVED/PARTIAL/NOT-RESOLVED with file:line. Confirm earlier fixes stayed resolved. Then do a
final full plan-fidelity sweep. Be explicit about whether anything critical or major remains.
Remember this is AUDIT-ONLY planned code validated on a synthetic backend (MMAudio wiring is a
later GPU phase); do not penalize the intentional MMAudio stub.

OUTPUT FORMAT (exactly):
VERDICT: <MATCHES_PLAN | PARTIAL_MISMATCH | CRITICAL_MISMATCH | ERROR>
RESOLUTION OF ROUND-4 FINDINGS: R4-1..R4-3 each RESOLVED/PARTIAL/NOT-RESOLVED with file:line.
REMAINING OR NEW FINDINGS: numbered (file:line, requirement, behavior, severity), or "none".
WHAT IS CORRECT: short balanced list.
Decision rule: MATCHES_PLAN = faithful with at most MINOR notes (NO critical/major open);
PARTIAL_MISMATCH = only minor/scoped non-blocking issues; CRITICAL_MISMATCH = a real
method/control/metric divergence. If only minor notes remain, return MATCHES_PLAN.
