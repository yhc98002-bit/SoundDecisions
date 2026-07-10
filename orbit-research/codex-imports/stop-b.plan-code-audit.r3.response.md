VERDICT: CRITICAL_MISMATCH

RESOLUTION OF ROUND-2 FINDINGS:
R2-1 PARTIAL — `foley_cw/gap.py:326-329` filters underpowered commitment windows and `foley_cw/gap.py:391-394` filters underpowered readout windows; all-underpowered maps emit `STOP_PROJECT` at `foley_cw/gap.py:348-357`. However `GO_MAP` can still be driven by underpowered windows through the unfiltered separation score: `foley_cw/cli/phases123_maps.py:191` computes `separation_score(commit_windows)`, `foley_cw/stats.py:242-247` includes non-NaN underpowered windows, and `foley_cw/gap.py:376-379` trusts that score.
R2-2 RESOLVED — `foley_cw/commitment.py:353-367` precomputes base states and `A_independent` once per `(video, axis)`, then `foley_cw/commitment.py:372-385` reuses them across alphas; rows report one per-axis mean baseline at `foley_cw/commitment.py:389-400`.
R2-3 RESOLVED — empty sidecar validity returns `NaN` at `foley_cw/reliability.py:284-285`, and non-finite determinism/robustness/validity fail the gate at `foley_cw/reliability.py:431-435`.
R2-4 RESOLVED — synthetic Phase 1-3 excludes `requires == "two_event_clips"` at `foley_cw/cli/phases123_maps.py:119-123`; binding is marked that way in `configs/axes.json:8`.

REMAINING OR NEW FINDINGS:
1. `foley_cw/cli/phases123_maps.py:191`, `foley_cw/stats.py:242-247`, `foley_cw/gap.py:376-379`: plan requirement: underpowered axes are not results and cannot support `GO_MAP`. Code behavior: separation is computed over all non-NaN commitment windows, including underpowered ones, then used for `GO_MAP`. I reproduced a case where adding one underpowered extreme window changes tokens from no `GO_MAP` to `GO_MAP`. Severity: critical.
2. `foley_cw/reporting.py:405-413`: plan requirement: bootstrap-over-video CIs for `s_commit`, `s_read`, and gaps. Code behavior: gap report only subtracts point estimates (`s_read - s_commit`) and emits no gap CI. Severity: major.
3. `foley_cw/cli/phases123_maps.py:237-263`, `foley_cw/reporting.py:435-437`: plan requirement: threshold sensitivity, including separation under threshold sweeps. Code behavior: sensitivity is only a first-active-axis commitment sweep and reports only `s_hat`; it does not recompute separation/readout sensitivity. Severity: major.

WHAT IS CORRECT:
- Round-1 fixes remain resolved: SDE token gating, reliability-gated map axes, min-n propagation, threshold caveats, committed-axis readout gating, and CSV window columns are present.
- Score conversion, `s<->t` seam, self-target convention, `x0(s)` readout, ODE/fork-majority targets, and no-Restart commitment are implemented as planned.
- MMAudio and heavy real probes/measurers raise instead of fabricating results.
- `PYTHONPATH=$PWD pytest -q` passed: 514 tests. Plain `pytest -q` fails import collection because the package is not installed/importable without `PYTHONPATH`.
