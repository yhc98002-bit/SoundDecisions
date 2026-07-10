# SoundDecisions — PI Review Guide

**Prepared:** 2026-07-10
**Purpose:** A short index for reviewing the implementation, the current evidence record, and the few places where interpretation or follow-up choices matter.

## Start here

1. [`results/PI_REPORT_arc3.md`](results/PI_REPORT_arc3.md) is the latest consolidated status report. It should be read before older progress reports; it describes the final Arc-3 token ledger, the evidence tier, corrections made after adversarial checking, and deferred follow-ups.
2. [`experiment/preregistered/arc3_tierB_preregistration.md`](experiment/preregistered/arc3_tierB_preregistration.md) records the Tier-B plan that was frozen before that analysis. [`experiment/preregistered/SHA256SUMS.json`](experiment/preregistered/SHA256SUMS.json) records its freeze hashes.
3. [`proposal/METHOD_SPEC.md`](proposal/METHOD_SPEC.md) and [`experiment/EXPERIMENT_PLAN.md`](experiment/EXPERIMENT_PLAN.md) define the intended method and decision gates.
4. [`foley_cw/README.md`](foley_cw/README.md) is the shortest technical introduction to the package and its verified-versus-model-specific boundaries.

The latest report is a project-authored evidence summary, not a new independent re-analysis performed for this guide. In particular, no GPU job or external model run was launched during repository preparation.

## Current implementation and evidence status

The repository implements a video-to-audio flow-generation analysis framework asking **when a perceptual audio decision is committed during denoising, and when it is externally readable from an intermediate state**. The implementation has four layers:

1. **Flow mechanics:** Convert a rectified-flow velocity to a score, form marginal-preserving stochastic tail forks, and obtain Tweedie audio previews.
2. **Measurements:** For each axis (presence, timing, class, material), measure fork agreement and probe predictability without defining the completed model output as ground truth.
3. **Statistical decisions:** Estimate commitment/readout windows with video-level bootstrap CIs, test separation and robustness, and emit pre-specified tokens.
4. **Follow-on analyses:** Test guidance effects, condition swaps, class readability, seed effects, and an offline axis-gated policy.

According to the latest Arc-3 report, the implementation and reported analyses are complete through the Tier-B program. The reported current interpretation is:

- The cfg=1.0 and cfg=4.5 kernel checks were accepted, with an explicit caveat for cfg=4.5.
- The commitment/readout map gate reported `GO_MAP` and `GO_READOUT`.
- The original F-1 seed-migration prediction was **refuted**; the alternative explanation is guidance-driven entropy reduction rather than a causal move into either seed or video conditioning.
- The oracle-to-non-oracle policy bridge reported `BRIDGE_PARTIAL`: timing is recovered well, while class readout is the limiting factor. The report therefore calls the overall result **DIAGNOSTIC-strong**, not a fully robust joint non-oracle result.
- The report records 1,023 passing tests at the end of Arc 3. This is a historical test result; repository preparation should rerun the local suite before release changes.

### Important chronology note

Some earlier reports are intentionally retained and can look inconsistent if read out of order. For example, [`PROGRESS_SUMMARY_2026-06-10.md`](PROGRESS_SUMMARY_2026-06-10.md) describes a much earlier Phase-0 checkpoint, and [`results/stage0/reliability_report.md`](results/stage0/reliability_report.md) is an older standalone report that demotes all axes. Treat [`results/PI_REPORT_arc3.md`](results/PI_REPORT_arc3.md), together with the Arc-3 aggregate artifacts below, as the latest account; review the older files as audit history rather than current conclusions.

## Critical code to review

| Review priority | Files | Why they matter |
|---|---|---|
| 1 — core mathematical claim | [`foley_cw/score_sde.py`](foley_cw/score_sde.py), [`foley_cw/time_map.py`](foley_cw/time_map.py), [`foley_cw/validation.py`](foley_cw/validation.py) | Implements velocity-to-score conversion, Tweedie reconstruction, Euler–Maruyama tail forks, time conventions, and the α=0/nonzero-α validation checks. This is the load-bearing causal/mechanical layer. |
| 1 — real-model seam | [`foley_cw/mmaudio_backend.py`](foley_cw/mmaudio_backend.py), [`foley_cw/model_adapter.py`](foley_cw/model_adapter.py), [`scripts/phase0_mmaudio_validate.py`](scripts/phase0_mmaudio_validate.py) | Connects the generic mechanism to MMAudio: conditioning construction, latent access, velocity calls, decoding, and Phase-0 checks. Confirm the MMAudio conventions and dependency assumptions match the reported runs. |
| 1 — primary maps | [`foley_cw/commitment.py`](foley_cw/commitment.py), [`foley_cw/readout.py`](foley_cw/readout.py), [`foley_cw/determination.py`](foley_cw/determination.py), [`foley_cw/stats.py`](foley_cw/stats.py) | Defines the agreement decomposition, window crossings, bootstrap CIs, and the commitment/readout quantities underlying Figs. 1–2 and the map tokens. |
| 1 — decision controls | [`foley_cw/gate_a.py`](foley_cw/gate_a.py), [`foley_cw/gap.py`](foley_cw/gap.py), [`foley_cw/reliability.py`](foley_cw/reliability.py), [`configs/thresholds.json`](configs/thresholds.json) | Contains the kernel, reliability, and GO/NO-GO logic. Review the ratified full-pool Gate-A exposure-cap correction especially carefully. |
| 2 — measurement validity | [`foley_cw/real_measurer.py`](foley_cw/real_measurer.py), [`foley_cw/measurers_panns_cnn14.py`](foley_cw/measurers_panns_cnn14.py), [`foley_cw/visual_anchors.py`](foley_cw/visual_anchors.py), [`foley_cw/sidecar.py`](foley_cw/sidecar.py) | Defines real perceptual axes, PANNs features, anchors, and sidecar/correctness checks. These choices determine what the reported axes mean. |
| 2 — Arc-3 interpretation | [`foley_cw/class_probes.py`](foley_cw/class_probes.py), [`foley_cw/cond_features.py`](foley_cw/cond_features.py), [`foley_cw/seed_floor.py`](foley_cw/seed_floor.py), [`foley_cw/condition_swap.py`](foley_cw/condition_swap.py) | Tests the F-1 alternative mechanism: whether class is readable from internal/conditioning features, whether seed predicts class, and whether a swapped condition causally changes an axis. |
| 2 — policy conclusion | [`foley_cw/bridge.py`](foley_cw/bridge.py), [`foley_cw/policy_offline.py`](foley_cw/policy_offline.py), [`scripts/b4_bridge.py`](scripts/b4_bridge.py), [`scripts/phase4_policy.py`](scripts/phase4_policy.py) | Implements the corrected oracle-to-non-oracle bridge and the offline policy comparison. The Arc-3 report notes that an adversarial audit found and fixed an earlier inflation bug here. |
| 3 — reproducibility/operations | [`foley_cw/run_store.py`](foley_cw/run_store.py), [`foley_cw/storage_budget.py`](foley_cw/storage_budget.py), [`scripts/phase1_commitment.py`](scripts/phase1_commitment.py), [`scripts/phase2_readout.py`](scripts/phase2_readout.py), [`scripts/phase3_decision.py`](scripts/phase3_decision.py) | Controls persistent artifacts, resume/journal behavior, storage limits, and the sequential execution path for the main maps. |

The tests in [`tests/`](tests/) mirror these modules. Start with `test_validation.py`, `test_commitment.py`, `test_readout.py`, `test_gate_a.py`, `test_bridge.py`, `test_condition_swap.py`, `test_seed_floor.py`, and `test_policy_offline.py` when auditing the claims above.

## Code path in plain language

```text
frozen manifests/configs
        ↓
MMAudio backend produces a conditioned flow trajectory
        ↓
score_sde makes stochastic tail forks + Tweedie previews at progress s
        ↓
real measurers/probes score presence, timing, class, and material
        ↓
commitment/readout + bootstrap statistics estimate windows and shares
        ↓
Gate-A/reliability/gap logic emits bounded decision tokens
        ↓
Arc-3 scripts test guidance, condition swaps, readability, seed floor,
and the oracle-to-non-oracle policy bridge
```

The key safeguard is that deterministic continuation (`α=0`) must reproduce the base ODE, while nonzero-α checks test score conversion, continuity, diversity, and marginal preservation. The per-axis result is normalized against independent generations so agreement from the video-conditioned prior is not incorrectly credited to late trajectory commitment.

## Select result files for PI review

| File | What it answers |
|---|---|
| [`results/PI_REPORT_arc3.md`](results/PI_REPORT_arc3.md) | Current overall narrative, token ledger, caveats, adversarial correction, and deferred work. |
| [`results/stage0/phase1/phase3_decision.md`](results/stage0/phase1/phase3_decision.md) | Main `GO_MAP`/`GO_READOUT` separation decision and reported commitment/readout windows. |
| [`results/stage0/gate_a_fullpool_report.md`](results/stage0/gate_a_fullpool_report.md) and [`results/stage0/gate_a_fullpool_interpretation.md`](results/stage0/gate_a_fullpool_interpretation.md) | The full-pool Gate-A calculation and why the exposure-scaled cap was ratified. |
| [`results/stage0/arc3/two_budgets.md`](results/stage0/arc3/two_budgets.md) and [`results/stage0/arc3/b3_seed_floor_dial.json`](results/stage0/arc3/b3_seed_floor_dial.json) | Evidence behind the F-1 refutation and entropy-reduction interpretation. |
| [`results/stage0/arc3/b4_bridge.json`](results/stage0/arc3/b4_bridge.json), [`results/stage0/arc3/b2_cond_audit_report.md`](results/stage0/arc3/b2_cond_audit_report.md), and [`results/figures/`](results/figures/) | The `BRIDGE_PARTIAL` result, the conditioning-channel audit, and the publication figures. |

Raw per-clip journals and feature tensors remain in the original run storage for auditability. They are supporting evidence, not the recommended first read; the high-volume tensor/audio artifacts are intentionally excluded from the GitHub upload.

## PI review points and open work

1. **Evidence tier and framing:** Decide whether “DIAGNOSTIC-strong, class-readout-limited” is the desired top-level framing. The evidence does not support a strong joint non-oracle recovery claim.
2. **Gate-A ratification:** Review the full-pool exposure-cap correction in the two Gate-A files above and confirm its governance/freeze treatment is acceptable.
3. **Axis semantics:** Review the self-target, anchor, tagger, and material-embedding choices before treating the four axes as perceptual conclusions.
4. **Class-readout follow-up:** The latest report identifies a robust nonlinear/learned class head as the concrete unresolved technical lever. The collected per-token/cross-attention features need a numerically safe, streamed follow-up probe before a new claim is made.
5. **Deferred scale checks:** The full-pool B3 seed test and MMAudio large_44k scale-insurance check were reported as deferred and not story-critical; they are not completed evidence.
6. **Document hygiene:** Retain older reports for traceability, but establish a clearly labeled current release summary before external scientific circulation so superseded snapshots cannot be mistaken for current status.

## Repository scope and intentionally untracked assets

This GitHub-ready repository includes first-party source, tests, configurations, frozen plans, manifests, aggregate results, figures, and compact derived artifacts. It deliberately excludes local environments/caches, execution logs, the 12-GB FoleyBench checkout, model checkpoints (including the 343-MB PANNs checkpoint), the local MMAudio checkout, and raw run tensors/audio/journals (`results/**/*.npz`, `*.wav`, `pertoken`, `features`, `previews`, `finals`, `measurements`, and `journal`).

To reproduce model-backed runs, obtain the data and weights through their authorized channels and install MMAudio separately. The local checkout inspected for this guide was `https://github.com/hkchengrex/MMAudio` at commit `974010a026c731054592d8f777218bd9d85a6c24`; it is not redistributed here. The lightweight CPU package dependencies are in [`pyproject.toml`](pyproject.toml) and [`requirements.txt`](requirements.txt); heavy model dependencies remain deliberately unpinned in `requirements.txt` and must be pinned for a fresh real-model environment.
