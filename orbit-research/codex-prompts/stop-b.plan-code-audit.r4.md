ROLE: Semantic plan-code audit reviewer (STOP B), ROUND 4. Independent cross-model reviewer.
Re-derive from the code; do not trust this summary. JUDGE plan-fidelity; do not improve code.

REPO: /XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions

CONTEXT: `foley-cw` STOP-B audit-only implementation (planned code only; CPU-validated vs an
analytic synthetic backend; MMAudio/GPU absent; MMAudio adapter raises). History: R1
CRITICAL (6 findings fixed); R2 CRITICAL (4 fixed); R3 CRITICAL with one PARTIAL + two MAJOR.
The R3 findings were addressed — VERIFY from code:

R3-1 (was CRITICAL: separation could be inflated by an underpowered outlier): decide_phase3 now
  recomputes separation over RESULT windows only via `separation = separation_score(valid_commit)`
  (valid_commit excludes NaN AND underpowered) at foley_cw/gap.py; the CLI also reports separation
  over result-only windows (foley_cw/cli/phases123_maps.py result_commit_windows).
R3-2 (was MAJOR: no gap CIs): foley_cw/stats.py bootstrap_gap_ci bootstraps s_read - s_commit over
  videos jointly; build_commitment_map/build_readout_map stash per-video curves in window.extra;
  the CLI computes gap CIs and the gap report renders "gap = point (95% CI [lo,hi], n)".
R3-3 (was MAJOR: sensitivity only first-axis s_hat): foley_cw/stats.py separation_under_thresholds
  recomputes per-axis windows + separation across a theta_commit sweep; the report renders a
  "Separation under theta_commit sweep" table.

AUTHORITATIVE SPEC: refine-logs/EXPERIMENT_PLAN.md, experiment/experiment_pack.json,
experiment/EXPERIMENT_PLAN_EXEC.md.

RE-AUDIT the whole implementation (foley_cw/ + cli/ + configs/ + tests/). For R3-1..R3-3 state
RESOLVED/PARTIAL/NOT-RESOLVED with file:line. Confirm earlier-round fixes stayed resolved. Then
sweep the full plan-fidelity surface once more for ANY remaining or newly introduced divergence
that would block the planned diagnostic. Be explicit if nothing critical/major remains.

OUTPUT FORMAT (exactly):
VERDICT: <MATCHES_PLAN | PARTIAL_MISMATCH | CRITICAL_MISMATCH | ERROR>
RESOLUTION OF ROUND-3 FINDINGS: R3-1..R3-3 each RESOLVED/PARTIAL/NOT-RESOLVED with file:line.
REMAINING OR NEW FINDINGS: numbered (file:line, plan requirement, code behavior, severity), or
"none".
WHAT IS CORRECT: short balanced list.
Decision rule: MATCHES_PLAN = faithful with at most MINOR notes (NO critical/major open);
PARTIAL_MISMATCH = only scoped non-blocking (minor/major-but-not-blocking) issues remain;
CRITICAL_MISMATCH = a real method/control/metric divergence remains. Use MATCHES_PLAN if the
remaining items are genuinely minor.
