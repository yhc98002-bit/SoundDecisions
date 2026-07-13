Audit-era description; predates GPU wiring; see CURRENT_STATUS

# foley_cw — implementation of the `foley-cw` experiment plan

This package implements **When Are Foley Decisions Made? Commitment and Readout Windows in
V2A Flow Generation**. It is the STOP-B (`/experiment-bridge`, `audit-only`) implementation
of `experiment/experiment_pack.json` / `refine-logs/EXPERIMENT_PLAN.md`. It contains the
planned code and is validated by a semantic plan-code audit; it does **not** run formal
diagnostics (those are GPU, owner=human, via `/diagnostic-to-review`).

## What this measures (per Foley correctness axis)

- **Commitment** `s_commit(axis)`: the smallest progress `s` at which the model's own
  *self-target* (e.g. tagger top-1 class of the completed audio) stabilizes under
  marginal-preserving **stochastic tail-forks**, normalized as a **gain over the
  video-conditioned prior** `A_independent`.
- **Readout** `s_read(axis, probe)`: the smallest `s` at which a probe predicts that same
  self-target from the running Tweedie preview `x0(s)`.
- **Make-or-break**: do commitment windows *separate* across axes, and are early axes
  *readable early*? (`GO_MAP` + `GO_READOUT`) — both **without correctness labels**.

## The audit boundary (what is VERIFIED vs UNVERIFIED)

MMAudio source/weights are **not vendored** in this repo, so anything model-specific is
isolated behind a seam and marked UNVERIFIED — it must be pinned in the **Phase 0** GPU
diagnostic (owner=human), not in this audit-only bridge:

| Concern | Where | Status |
|---|---|---|
| s ↔ t integration direction | `time_map.MMAUDIO_S_TO_T` | **UNVERIFIED** (Phase 0.1) |
| `v_theta(x, t, video_cond)` | `model_adapter.MMAudioBackend.velocity` | **STUB → MMAudioNotWired** (Phase 0.1/0.2) |
| latent decode → audio | `model_adapter.MMAudioBackend.decode` | **STUB** (Phase 0.1) |
| velocity → score sign/parameterization | `score_sde.score_from_velocity` | rectified-linear form; **MMAudio branch must be audited** (Phase 0.2) |

Everything **model-agnostic** is implemented and CPU-testable against an analytic oracle:

| Concern | Where | Status |
|---|---|---|
| analytic Gaussian flow (closed-form score/velocity/marginal) | `synthetic_backend.SyntheticGaussianFlow` | **VERIFIED** (test oracle) |
| velocity→score conversion, Tweedie x0 | `score_sde` | **VERIFIED to machine precision** vs analytic score |
| marginal-preserving SDE step (drift `v + ½σ²·score`) | `score_sde.euler_maruyama_step` | **VERIFIED** (α=0 ⇒ ODE; marginal preserved; FP-derived ½) |
| commitment / readout maps, agreement, bootstrap-over-videos | `commitment`, `readout`, `agreement`, `stats` | model-agnostic, synthetic-tested |
| reliability gate, dataset/anchor manifest, GO/NO-GO logic | `reliability`, `dataset`, `gap`, `validation` | model-agnostic |

### The crux, in one paragraph

The highest silent-bug risk is the velocity→score conversion. Under the rectified-flow
interpolant `x_t = t·x1 + (1−t)·ε`, the exact identity is `score = (t·v − x)/(1−t)` and
`x0 = x + (1−t)·v`; the marginal-preserving fork SDE is `dx = (v + ½σ²·score)·dt + σ·dW`
with the **½ coefficient unique** by Fokker–Planck. `synthetic_backend` provides these in
closed form so the conversion is checked exactly. **α=0 reproducing the ODE is necessary
but does NOT exercise the score term** (it is multiplied by 0) — the real tests are
nonzero-α (small-α continuity, fork validity, nontrivial diversity, marginal preservation),
all in `foley_cw.validation` / `tests/`.

## Layout

```
foley_cw/
  types.py            data contract (axes, thresholds, schedules, result cells)
  time_map.py         s<->t seam (MMAudio direction UNVERIFIED)
  model_adapter.py    FlowModelBackend ABC + MMAudioBackend stub
  synthetic_backend.py analytic Gaussian flow (CPU oracle)
  score_sde.py        velocity<->score, Tweedie x0, Euler-Maruyama, fork_tail, ode_complete
  config.py           load configs/*.json
  agreement.py        per-axis agreement metrics (exact-match / Krippendorff / cosine)
  commitment.py       A_independent, A_fork, normalized gain, s_commit
  readout.py          probe ladder over x0(s); ODE-target & fork-majority; s_read
  probes.py           EnergyOnsetProbe (CPU) + CLAP/SyncNet/ImageBind/tagger/MLLM stubs
  reliability.py      three-part reliability gate (determinism/robustness/validity)
  dataset.py          FoleyBench manifest + event-anchor protocol (+ synthetic CI dataset)
  stats.py            bootstrap-over-videos CIs, separation score, threshold sensitivity
  gap.py              gap(axis,probe), R1/R2 cross-tab, GO/NO-GO token logic
  validation.py       Phase-0.2 SDE validation (alpha=0, continuity, diversity, marginal)
  reporting.py        CSV/Markdown writers for results/*
  cli/                phase0_feasibility.py, phases123_maps.py entrypoints
configs/   *.json (axes, thresholds[UNFROZEN], schedule, alpha_grid, dataset)
tests/     pytest unit tests (numpy-only, synthetic backend)
```

## Running (audit-only / CPU)

```bash
python -m pytest tests/ -q                              # validates the crux math on CPU
python -m foley_cw.cli.phase0_feasibility --synthetic   # dry-run Phase 0 on synthetic data
python -m foley_cw.cli.phases123_maps   --synthetic     # dry-run maps -> results/*.csv
```

The `--synthetic` dry-runs exercise the full pipeline end-to-end on the analytic backend
(no GPU, no MMAudio) and emit parseable CSV/Markdown with the same schema the real Phase-0
GPU diagnostic will produce. Real runs require Phase-0 MMAudio wiring (owner=human).
