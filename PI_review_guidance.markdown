# SoundDecisions — PI Review Guide

**Prepared:** 2026-07-10
**Purpose:** A short index for reviewing the implementation, the current evidence record, and the few places where interpretation or follow-up choices matter.

## Start here

1. [`results/CURRENT_STATUS.md`](results/CURRENT_STATUS.md) is the authoritative current status, evidence tier, and citation boundary.
2. [`results/PI_REPORT_arc3.md`](results/PI_REPORT_arc3.md) is a superseded Arc-3 audit record. Read it only after the current status and amendments; its conclusions are retained for chronology, not current citation.
3. [`experiment/preregistered/amendments_arc4.md`](experiment/preregistered/amendments_arc4.md) records the post-result governance amendments. [`experiment/preregistered/SHA256SUMS.json`](experiment/preregistered/SHA256SUMS.json) records the freeze hashes.
4. [`experiment/preregistered/arc3_tierB_preregistration.md`](experiment/preregistered/arc3_tierB_preregistration.md) records the Tier-B plan frozen before that analysis.
5. [`proposal/METHOD_SPEC.md`](proposal/METHOD_SPEC.md), [`experiment/EXPERIMENT_PLAN.md`](experiment/EXPERIMENT_PLAN.md), and [`foley_cw/README.md`](foley_cw/README.md) define the method, decision gates, and package boundaries.

The current status and amendments are project-authored governance records, not a new independent re-analysis performed for this guide. In particular, no GPU job or external model run was launched during repository preparation.

## Current implementation and evidence status

The repository implements a video-to-audio flow-generation analysis framework asking **when a perceptual audio decision is committed during denoising, and when it is externally readable from an intermediate state**. The implementation has four layers:

1. **Flow mechanics:** Convert a rectified-flow velocity to a score, form marginal-preserving stochastic tail forks, and obtain Tweedie audio previews.
2. **Measurements:** For each axis (presence, timing, class, material), measure fork agreement and probe predictability without defining the completed model output as ground truth.
3. **Statistical decisions:** Estimate commitment/readout windows with video-level bootstrap CIs, test separation and robustness, and emit pre-specified tokens.
4. **Follow-on analyses:** Test guidance effects, condition swaps, class readability, seed effects, and an offline axis-gated policy.

WP-A corrected code and re-analysed cached Arc-1..3 artifacts. The current interpretation is:

- The evidence tier is **diagnostic/exploratory**. Arcs 1-3 are exploratory; Arc 4 is the frozen confirmatory pass.
- The cfg=1.0 and cfg=4.5 kernel checks were accepted, with the AMD-13 near-exchangeability caveat on cfg=4.5 and every derived result.
- The legacy commitment/readout rules report `GO_MAP` and `GO_READOUT`, under the self-target charter and human-alignment limits recorded by AMD-14.
- The F-1 guidance-to-entropy mechanism is **UNRESOLVED** pending Arc-4 B-3/B-6. The Arc-3 condition-swap interpretation is withdrawn.
- `R2_CLASS_PENDING_PERTOKEN` replaces `R2_CLASS_CONFIRMED`; `NO_GLOBAL_SEED_DECODER` replaces the broader `NO_SEED_FLOOR`. Per-token and video-conditional seed claims remain pending.
- `BRIDGE_PARTIAL` is provisional. Joint recovery is zero under the simulated symmetric keep-flip error model; per-axis means are sensitivity results, not a confirmed joint scheduler result.
- Arc 3 recorded 1,023 passing tests in the full checkout; WP-A recorded 1,041. These are historical environment-specific test results, not scientific findings.

### Important chronology note

Earlier reports are intentionally retained and can look inconsistent if read out of order. For example, [`PROGRESS_SUMMARY_2026-06-10.md`](PROGRESS_SUMMARY_2026-06-10.md) describes an early Phase-0 checkpoint, [`results/stage0/reliability_report.md`](results/stage0/reliability_report.md) is an older standalone report, and [`results/PI_REPORT_arc3.md`](results/PI_REPORT_arc3.md) preserves withdrawn Arc-3 conclusions. Treat all of them as audit history; [`results/CURRENT_STATUS.md`](results/CURRENT_STATUS.md) and the Arc-4 amendments govern current interpretation.

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
| [`results/CURRENT_STATUS.md`](results/CURRENT_STATUS.md) | Current evidence tier, citable bridge result, unresolved mechanism, and citation boundary. |
| [`results/PI_REPORT_arc3.md`](results/PI_REPORT_arc3.md) | Superseded Arc-3 narrative retained as audit history. |
| [`results/stage0/phase1/phase3_decision.md`](results/stage0/phase1/phase3_decision.md) | Main `GO_MAP`/`GO_READOUT` separation decision and reported commitment/readout windows. |
| [`results/stage0/gate_a_fullpool_report.md`](results/stage0/gate_a_fullpool_report.md) and [`results/stage0/gate_a_fullpool_interpretation.md`](results/stage0/gate_a_fullpool_interpretation.md) | The full-pool Gate-A calculation and why the exposure-scaled cap was ratified. |
| [`results/stage0/arc3/two_budgets.md`](results/stage0/arc3/two_budgets.md) and [`results/stage0/arc3/b3_seed_floor_dial.json`](results/stage0/arc3/b3_seed_floor_dial.json) | Historical evidence behind the now-withdrawn Arc-3 mechanism interpretation. |
| [`results/stage0/arc3/b4_bridge.json`](results/stage0/arc3/b4_bridge.json), [`results/stage0/arc3/b2_cond_audit_report.md`](results/stage0/arc3/b2_cond_audit_report.md), and [`results/figures/`](results/figures/) | The `BRIDGE_PARTIAL` result, the conditioning-channel audit, and the publication figures. |

Raw per-clip journals and feature tensors remain in the original run storage for auditability. They are supporting evidence, not the recommended first read; the high-volume tensor/audio artifacts are intentionally excluded from the GitHub upload.

## PI review points and open work

1. **Evidence tier and framing:** Keep the top-level tier diagnostic/exploratory. The evidence does not support a strong joint non-oracle recovery or resolved-mechanism claim.
2. **Gate-A ratification:** Review the full-pool exposure-cap correction in the two Gate-A files above and confirm its governance/freeze treatment is acceptable.
3. **Axis semantics:** Review the self-target, anchor, tagger, and material-embedding choices before treating the four axes as perceptual conclusions.
4. **Class-readout follow-up:** The collected per-token/cross-attention features need a numerically safe, streamed follow-up probe before `R2_CLASS_PENDING_PERTOKEN` can be reconsidered.
5. **Deferred scale checks:** The full-pool B3 seed test and MMAudio large_44k scale-insurance check were reported as deferred and not story-critical; they are not completed evidence.
6. **Document hygiene:** Retain older reports for traceability, but establish a clearly labeled current release summary before external scientific circulation so superseded snapshots cannot be mistaken for current status.

## Repository scope and intentionally untracked assets

This GitHub-ready repository includes first-party source, tests, configurations, frozen plans, manifests, aggregate results, figures, and compact derived artifacts. It deliberately excludes local environments/caches, execution logs, the 12-GB FoleyBench checkout, model checkpoints (including the 343-MB PANNs checkpoint), the local MMAudio checkout, and raw run tensors/audio/journals (`results/**/*.npz`, `*.wav`, `pertoken`, `features`, `previews`, `finals`, `measurements`, and `journal`).

To reproduce model-backed runs, obtain the data and weights through their authorized channels and install MMAudio separately. The local checkout inspected for this guide was `https://github.com/hkchengrex/MMAudio` at commit `974010a026c731054592d8f777218bd9d85a6c24`; it is not redistributed here. The lightweight CPU package dependencies are in [`pyproject.toml`](pyproject.toml) and [`requirements.txt`](requirements.txt); heavy model dependencies remain deliberately unpinned in `requirements.txt` and must be pinned for a fresh real-model environment.
