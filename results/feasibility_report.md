# Feasibility Report — Phase 0.1 Trajectory Access (foley-cw)

**Diagnostic:** `diag_phase0_feasibility` (partial — crux components).
**Run:** real MMAudio `small_16k` on an17 (1× A800), float32, 2026-06-09.

## VERDICT: trajectory access **PASS**

MMAudio's flow trajectory is fully accessible through
`foley_cw.mmaudio_backend.MMAudioBackend` + foley_cw's `score_sde` integrator:

- **extract `x_s`**: intermediate latent recorded at scan points, shape **(125, 20)** (4 s
  @ 16 kHz, latent_seq_len=125, latent_dim=20). Finite. ✅
- **resume from `x_s` to s=1** (`ode_complete`): produces a decoded waveform of **64000
  samples** (= 4.0 s @ 16 kHz). Finite. ✅
- **compute `x0(s)`** (Tweedie best-guess of final audio, the readout input): decoded
  waveform 64000 samples, finite. ✅

Model load 43.9 s (cold weights from Lustre); full check < 65 s end-to-end.

## s↔t mapping audit

Audited against MMAudio source: integration is **ascending t:0(noise)→1(audio)** (`to_data`,
Euler `x += dt·v`, `min_sigma=0`), so generation progress `s` maps to MMAudio's native time
by the **identity** (`time_map.IdentitySToT`, now verified). No reversed-time bug. See
`results/score_sde_validation_report.md` for the velocity→score consequence.

## Environment (documented for reproduction)

- GPU: an17, NVIDIA A800 80 GB, driver 535.104.12; torch 2.5.1+cu121.
- Env: shared venv on `/XYFS02/.../SoundDecisions/.venv`; for GPU runs it is staged into
  an17 `/dev/shm/foley_venv` per-session (Lustre cold-import is otherwise pathologically
  slow; `/dev/shm` import ≈ 1 s vs > 400 s on Lustre). RemoveIPC wipes `/dev/shm` when the
  user's last session ends, so each GPU job re-stages (≈ 22 s) and runs in one SSH session.
- MMAudio vendored at `third_party/MMAudio`; small_16k weights (model/VAE/BigVGAN/synchformer)
  under `third_party/MMAudio/{weights,ext_weights}` (md5-verified). FoleyBench NOT yet present.

## Scope

Phase 0.1 trajectory access + Phase 0.2 SDE conversion are validated. Phase 0.3 (dataset
manifest), 0.4 (event anchors), 0.5 (reliability gate) remain — they require FoleyBench and
per-axis audio measurements. The full `GO_MAPS_PHASE` token is therefore NOT yet emitted;
the load-bearing mechanism, however, is confirmed feasible.
