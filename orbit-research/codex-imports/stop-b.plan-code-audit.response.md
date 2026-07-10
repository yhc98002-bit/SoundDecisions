VERDICT: CRITICAL_MISMATCH

1. `refine-logs/EXPERIMENT_PLAN.md:103-109`, `refine-logs/EXPERIMENT_PLAN.md:128`: the plan requires Phase-0 SDE validation to include alpha=0 plus nonzero-alpha continuity, fork validity, and nontrivial diversity before `GO_MAPS_PHASE`. `foley_cw/validation.py:600-610` runs fork-validity/diversity/marginal checks, but `foley_cw/validation.py:612-615` emits `OK` based only on alpha=0 and small-alpha continuity. `foley_cw/cli/phase0_feasibility.py:100` passes that token to `decide_phase0`, and `foley_cw/gap.py:165-169` / `foley_cw/gap.py:232-234` can therefore approve maps with failed nonzero-alpha validity/diversity. Severity: critical.

2. `refine-logs/EXPERIMENT_PLAN.md:41`, `refine-logs/EXPERIMENT_PLAN.md:115-119`: the plan says map windows are valid only for axes that pass the three-part reliability gate, with demotion/drop on failure. `foley_cw/cli/phases123_maps.py:83-87` selects active axes only by tier, not reliability; `foley_cw/commitment.py:333-337` filters only `EXCLUDED`/`SEPARATE`; `foley_cw/readout.py:331-333` runs every supplied axis. Phase 1-3 can run without consuming Phase-0 reliability outputs. Severity: critical.

3. `refine-logs/EXPERIMENT_PLAN.md:39-40`, `configs/dataset.json:5-11`: the plan requires bootstrap over videos and axes below minimum usable `n` to be reported underpowered, not as results. `foley_cw/stats.py:151-183` supports `min_n`, but `foley_cw/commitment.py:404-413` and `foley_cw/readout.py:363-374` hardcode `min_n=1`; the synthetic CLI defaults to far below the declared 30-40 minimums (`foley_cw/cli/phases123_maps.py:286-290`). Severity: major.

4. `refine-logs/EXPERIMENT_PLAN.md:38`: thresholds must be frozen before headline maps. `configs/thresholds.json:2-10` explicitly marks them `UNFROZEN_PLACEHOLDER`, and `foley_cw/config.py:64-74` loads `frozen=False` without enforcement. `foley_cw/cli/phases123_maps.py:96-112` writes a pre-map placeholder, but `foley_cw/cli/phases123_maps.py:244-248` overwrites it after maps; the code never blocks on unfrozen thresholds. Severity: major.

5. `refine-logs/EXPERIMENT_PLAN.md:178`, `refine-logs/EXPERIMENT_PLAN.md:207-216`: the plan says readout of an uncommitted path is not a decided-axis readout, and `GO_MAP + GO_READOUT` is the make-or-break gate. `foley_cw/gap.py:383-397` emits `GO_READOUT` whenever `s_read < 0.8` and the axis has any commitment window; it assigns `c_w` but never checks that commitment is high/early. This can co-fire with `STOP_ADSR` from `foley_cw/gap.py:342-347` for late/coincident commitment windows. Severity: major.

6. `refine-logs/EXPERIMENT_PLAN.md:64-65`: planned CSV artifacts include `s_commit`/`s_read` with CIs, not only surfaces. `foley_cw/types.py:160-181` and `foley_cw/reporting.py:64-97` define `commitment_map.csv` and `readout_map.csv` without `s_commit`, `s_read`, `ci_low`, or `ci_high`; those windows are only rendered later in Markdown (`foley_cw/reporting.py:354-380`). Severity: major.

what is correct

- `foley_cw/score_sde.py:47-59` implements the rectified-linear score formula `score=(t*v-x)/(1-t)`, and `foley_cw/synthetic_backend.py:20-26` gives the matching analytic score.
- `foley_cw/score_sde.py:104-120` uses the planned SDE drift `v + 0.5*sigma^2*score`; `foley_cw/score_sde.py:152-159` makes `alpha=0` the deterministic ODE path.
- `foley_cw/time_map.py:1-15` isolates the `s<->t` seam and marks MMAudio direction unverified; `foley_cw/model_adapter.py:68-103` raises instead of fabricating MMAudio behavior.
- Commitment uses independent full alpha=0 generations for `A_independent` (`foley_cw/commitment.py:92-118`) and normalized gain (`foley_cw/commitment.py:71-85`).
- Readout includes both `ode` and `fork_majority` targets (`foley_cw/readout.py:326-375`) and probes `x0(s)` (`foley_cw/readout.py:265-277`).
- With `PYTHONPATH=.`, the local test suite passes: 514 tests passed, 16 warnings.
