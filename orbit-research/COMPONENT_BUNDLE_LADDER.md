# Component Bundle Ladder

> Generated view of `experiment/experiment_pack.json` (`component_ladder`), faithfully
> transcribed from the frozen `refine-logs/EXPERIMENT_PLAN.md` (§6, §7) and the proposal
> contributions C1–C5. The pack is the source of truth.

Minimal-mechanism rungs, in dependency order. Each rung is gated by the rung(s) below it;
everything from rung 4 onward is **conditional** on the make-or-break passing.

| Rung | Component | Gate to enter |
|------|-----------|---------------|
| 0 | **Feasibility + reliability** — trajectory access; validated velocity→score SDE (α=0 AND nonzero-α); dataset/anchor manifest; per-axis reliability gate | start here (STRICT) |
| 1 | **C1a Commitment map** — marginal-preserving stochastic tail-forks → `s_commit(axis)`, normalized over `A_independent` | `GO_MAPS_PHASE` |
| 2 | **C1b Readout map** — probe ladder on `x0(s)` for ODE-target and fork-majority targets → `s_read(axis, probe)` | `COMMITMENT_MAP_DONE` |
| 3 | **C1c Gap + separation** — gap, R1/R2 cross-tab, axis separation, GO/NO-GO (**make-or-break ends here**) | `READOUT_MAP_DONE` |
| 4 | **C3 Axis-gated population pruning** vs baselines incl. oracle upper bound | `GO_MAP` + `GO_READOUT` (+ correctness sidecar + `policy_preregistration.md`) |
| 5 | **C4 Cheap process-aware verifier** | maps show headroom |
| 6 | **C5 Axis-gated rollback** (SDE / Restart re-noising) | forking is stable |
| 7 | **Internal-feature probes** (latent / AV cross-attention) | external-probe gap is large; **non-blocking** |

The pilot's make-or-break is rungs 0→3 (`GO_MAP` + `GO_READOUT`), all correctness-label-free.
Do not build rung 4+ before the maps exist.
