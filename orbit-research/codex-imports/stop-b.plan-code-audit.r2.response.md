VERDICT: CRITICAL_MISMATCH

RESOLUTION OF ROUND-1 FINDINGS:
1. RESOLVED — `run_sde_validation` returns `OK` only after alpha=0, exact-score, continuity, fork-validity, diversity, and marginal checks pass; failures route to `FIX_SCORE_CONVERSION` or `FORK_ALPHA_NO_VALID_OPERATING_POINT` (`foley_cw/validation.py:563`, `foley_cw/validation.py:627`). `decide_phase0` blocks any non-`OK` token (`foley_cw/gap.py:200`).
2. RESOLVED — `phases123_maps.py` runs reliability and maps only `active_axes`; demoted axes are recorded (`foley_cw/cli/phases123_maps.py:99`, `foley_cw/cli/phases123_maps.py:117`, `foley_cw/cli/phases123_maps.py:258`).
3. PARTIAL — `min_n_per_axis` is passed into both map builders and `window_with_ci` (`foley_cw/commitment.py:314`, `foley_cw/commitment.py:414`, `foley_cw/readout.py:297`, `foley_cw/readout.py:374`), but Phase-3 decisions still treat underpowered windows as valid results; see finding 1.
4. RESOLVED — unfrozen thresholds warn on stderr and are recorded as non-binding in reports (`foley_cw/cli/phases123_maps.py:89`, `foley_cw/cli/phases123_maps.py:274`, `foley_cw/cli/phases123_maps.py:277`).
5. RESOLVED — `GO_READOUT` now requires early commitment, early readout, and bounded gap from the committed axis (`foley_cw/gap.py:391`, `foley_cw/gap.py:400`, `foley_cw/gap.py:404`). `GO_MAP` is mutually exclusive with `STOP_ADSR` (`foley_cw/gap.py:349`, `foley_cw/gap.py:365`).
6. RESOLVED — CSV writers append `s_commit` / `s_read` and CIs when windows are passed, and the CLI passes them (`foley_cw/reporting.py:64`, `foley_cw/reporting.py:96`, `foley_cw/reporting.py:125`, `foley_cw/cli/phases123_maps.py:153`, `foley_cw/cli/phases123_maps.py:170`).

NEW OR REMAINING FINDINGS:
1. `foley_cw/gap.py:323`, `foley_cw/gap.py:382` — Plan requirement: axes below minimum usable n are underpowered, not results. Code behavior: Phase-3 validity filters only on non-NaN `s_hat`, ignoring `WindowEstimate.underpowered`; direct audit call with all windows `underpowered=True` still emitted `GO_MAP` and `GO_READOUT`. Severity: CRITICAL.
2. `foley_cw/commitment.py:357`, `foley_cw/commitment.py:363`, `foley_cw/commitment.py:190`, `foley_cw/commitment.py:377` — Plan requirement: `A_independent(video, axis)` is the alpha-independent video-prior baseline. Code behavior: it is recomputed with fresh RNG inside every alpha pass, so the reported baseline can vary by alpha and contaminates the full `A(axis,s,alpha)` surface. Severity: MEDIUM.
3. `foley_cw/reliability.py:249`, `foley_cw/reliability.py:281` — Plan requirement: validity must be agreement with a calibration sidecar. Code behavior: an explicitly empty sidecar returns validity `1.0`, allowing a real caller to pass validity without calibration. Severity: MEDIUM.
4. `foley_cw/cli/phases123_maps.py:103`, `foley_cw/dataset.py:232`, `configs/axes.json:8` — Plan requirement: Tier-3 binding runs only on clean two-event clips. Code behavior: synthetic Phase 1-3 includes `binding` as a mapped axis even though the synthetic dataset creates single-event anchors. Severity: LOW.

WHAT IS CORRECT:
- Synthetic score/SDE path is analytic and well tested; `PYTHONPATH=$PWD pytest -q` passed 514 tests.
- MMAudio is not fabricated: backend and real measurers/probes raise stubs, and `--no-synthetic` exits nonzero.
- Commitment uses marginal-preserving SDE only; Restart re-noising is absent from the commitment kernel.
- Readout uses decoded `x0(s)` and reports both ODE and fork-majority self-targets.
- Maps target model self-targets, not correctness-vs-video.
