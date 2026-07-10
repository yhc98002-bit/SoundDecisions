VERDICT: MATCHES_PLAN

RESOLUTION OF ROUND-4 FINDINGS: R4-1 RESOLVED — `foley_cw/stats.py:335`/`foley_cw/stats.py:338` compute per-video crossings, and `foley_cw/stats.py:349` computes mean read crossing minus mean commit crossing, matching `window_with_ci` at `foley_cw/stats.py:196`/`foley_cw/stats.py:208`. R4-2 RESOLVED — sensitivity accepts and applies `min_n_per_axis` at `foley_cw/stats.py:371`/`foley_cw/stats.py:399`, filters to result windows at `foley_cw/stats.py:404`, and the CLI passes min-n at `foley_cw/cli/phases123_maps.py:301`. R4-3 RESOLVED — gap CIs are computed only for result windows at `foley_cw/cli/phases123_maps.py:276`/`foley_cw/cli/phases123_maps.py:282`; reporting excludes underpowered/undefined gap entries at `foley_cw/reporting.py:416`/`foley_cw/reporting.py:419` while still listing window underpowered flags at `foley_cw/reporting.py:377`/`foley_cw/reporting.py:392`.

REMAINING OR NEW FINDINGS:
1. `foley_cw/cli/phases123_maps.py:261`, `foley_cw/reporting.py:471` — Requirement: underpowered axes should not be presented as result evidence. Behavior: the legacy “first-axis s_commit” threshold-sweep section still calls `threshold_sweep` without `min_n`, so small underpowered dry-runs can show non-NaN first-axis sensitivity values. Severity: MINOR; main separation sensitivity and decisions are correctly result-filtered.
2. `foley_cw/gap.py:450` — Requirement: `GO_RESTRICTED` implies only presence/gross timing have early actionable windows. Behavior: the token currently checks early commitment axes only, not readout/actionability, and can emit with no read window. Severity: MINOR; `GO_READOUT` and `GO_DIAGNOSTIC` remain separately gated, so no critical/major decision path is open.

WHAT IS CORRECT:
- No critical or major plan-fidelity issues remain.
- Earlier fixes stayed resolved: SDE token gating, reliability-gated axes, min-n propagation, committed-axis readout gating, stable `A_independent`, CSV window CIs, binding exclusion, and underpowered decision filtering are present.
- Core map semantics match the plan: normalized commitment over video prior, both ODE and fork-majority readout targets, decoded `x0(s)` probes, no Restart re-noising in commitment, and no correctness-vs-video leakage.
- MMAudio/heavy real paths remain intentional stubs and do not fabricate GPU/model results.
- Verification run: `PYTHONPATH=$PWD pytest -q` passed `514` tests with `16` NumPy warnings.
