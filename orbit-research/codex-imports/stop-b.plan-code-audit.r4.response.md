VERDICT: PARTIAL_MISMATCH

RESOLUTION OF ROUND-3 FINDINGS: R3-1 RESOLVED — `decide_phase3` filters to non-NaN, non-underpowered commitment windows and recomputes separation internally (`foley_cw/gap.py:328`, `foley_cw/gap.py:340`); CLI main separation also uses result-only windows (`foley_cw/cli/phases123_maps.py:193`). R3-2 PARTIAL — gap CIs are implemented and rendered (`foley_cw/stats.py:303`, `foley_cw/reporting.py:417`), but the CI helper’s point statistic is not the planned/report gap definition. R3-3 PARTIAL — separation is recomputed across axes under a `theta_commit` sweep (`foley_cw/stats.py:357`, `foley_cw/reporting.py:451`), but underpowered axes are not excluded from that sweep.

REMAINING OR NEW FINDINGS:
1. `foley_cw/stats.py:330`, `foley_cw/reporting.py:420`: plan requirement: `gap(axis, probe) = s_read(axis, probe) - s_commit(axis)` with bootstrap-over-video CIs. Code behavior: `bootstrap_gap_ci` computes `first_crossing(mean read curve) - first_crossing(mean commit curve)`, while `s_read`/`s_commit` elsewhere are mean per-video crossings (`foley_cw/stats.py:196`, `foley_cw/stats.py:208`). The rendered gap point can therefore disagree with the defined `s_read - s_commit`. Severity: MAJOR.
2. `foley_cw/cli/phases123_maps.py:293`, `foley_cw/stats.py:380`, `foley_cw/reporting.py:440`: plan requirement: underpowered axes are reported as underpowered, not as results, including sensitivity. Code behavior: separation sensitivity includes every axis with stored curves and recomputes windows with default `min_n=1`; a default 3-video run correctly emits `STOP_PROJECT` for underpowered main results but still renders non-NaN separation sensitivity. Severity: MAJOR.
3. `foley_cw/cli/phases123_maps.py:276`, `foley_cw/reporting.py:417`: plan requirement: underpowered axes are not result evidence. Code behavior: gap CIs are computed/rendered for underpowered or undefined windows, producing lines such as `gap = nan (95% CI [...])`. Severity: MAJOR.

WHAT IS CORRECT:
- Earlier R1/R2 fixes remain resolved: SDE token gating, reliability-gated axes, min-n propagation to main windows, threshold caveats, committed-axis readout gating, CSV window columns, stable `A_independent`, empty-sidecar failure, and two-event binding exclusion are present.
- Core map math is faithful: normalized commitment over `A_independent`, both `ode` and `fork_majority` readout targets, decoded `x0(s)` probe input, no Restart re-noising in commitment, and MMAudio/real probes raise instead of fabricating results.
- Main Phase-3 GO/STOP decisions are now protected from underpowered outlier separation.
- `PYTHONPATH=$PWD pytest -q` passed: 514 tests, 16 NumPy warnings.
