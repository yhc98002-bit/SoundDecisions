# Score / SDE Validation Report — Phase 0.2 (foley-cw)

**Diagnostic:** `diag_phase0_feasibility` (partial — crux components).
**Run:** real MMAudio `small_16k` on an17 (1× A800), float32, 2026-06-09.
**Backend:** `foley_cw.mmaudio_backend.MMAudioBackend` driving MMAudio's flow network via
foley_cw's model-agnostic `score_sde` integrator. Conditioning: **unconditional/empty**
(the velocity→score conversion tests the flow-network mechanics, not the conditioning
content — no FoleyBench video needed for this crux). cfg_strength=1.0 (pure conditional
velocity), 25 Euler steps, 4 s @ 16 kHz (latent (125, 20)).

## VERDICT (Phase 0.2 token): `OK`

The plan's stated **highest silent-bug risk — the velocity→score conversion — is validated
on the real model.**

| Check | Pass | Evidence |
|---|---|---|
| α=0 reproduces the ODE | ✅ | max L2(fork@α=0, ode_complete) = **0.000e+00** (thr 1e-6) — deterministic fork ≡ ODE |
| small-α continuity (tests the score term) | ✅ | α=0.02: mean dist 1.71 vs ode_norm 14.26 → ratio **0.12** (thr 2.0), monotone in α |
| fork validity | ✅ | α=0.2: **8/8** forks finite & non-trivially large |
| nontrivial diversity | ✅ | α=0.2: mean per-dim std of forks = **0.085** (thr 1e-6) |
| exact score vs analytic | n/a | skipped — MMAudio has no closed-form score (synthetic-only check) |
| marginal preservation | n/a | skipped — synthetic-only check |

## Why this is conclusive for the conversion

The convention was audited directly against MMAudio source (`flow_matching.py`,
`networks.ode_wrapper`): MMAudio uses the rectified-flow linear interpolant
`x_t=(1−t)x0+t·x1` with x0=noise, x1=data, `v=x1−x0`, `min_sigma=0`, integrating t:0→1.
This is **identical** to the convention foley_cw's `score_from_velocity=(t·v−x)/(1−t)` and
the marginal-preserving fork drift `v+½σ²·score` were derived for — so the score identity
holds with `IdentitySToT` and **no sign/direction change**. α=0 reproducing the ODE
*exactly* (necessary), plus small-α continuity + fork validity + diversity (the real
nonzero-α tests of the score term), confirms the conversion empirically. No
`FIX_SCORE_CONVERSION`.

## Scope / what this does NOT yet cover

This validates Phase 0.1 (trajectory access) + Phase 0.2 (SDE conversion) only. The full
`GO_MAPS_PHASE` gate additionally requires Phase 0.3 dataset/anchor manifest, Phase 0.4
event-anchor validation, and Phase 0.5 reliability gate (≥3 axes pass
determinism+robustness+validity) — all of which need FoleyBench data and the per-axis
audio measurements (taggers/onset), not yet run. Forks here used cfg=1.0 unconditional
generation; video-conditioned behavior (Phases 1-3) is a later, separate step.

Raw: `results/phase0_mmaudio_validation.json`.
