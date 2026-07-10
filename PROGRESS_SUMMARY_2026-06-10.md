# foley-cw — Research & Progress Summary (as of 2026-06-10)

Status snapshot of the SoundDecisions project. Obtained results and plans are kept strictly
separate; sources are cited inline.

## 1. What the research is

**Project:** *When Are Foley Decisions Made? Commitment and Readout Windows in V2A Flow
Generation.* Target model: **MMAudio** (CVPR 2025), a video-to-audio rectified-flow model.

**Core question:** as a flow model generates audio for a video, *when along the trajectory
does each perceptual decision get made* (is a sound present? when does it hit? what class of
sound? what material/timbre?), and *when does that decision become readable* from the
intermediate state — well before generation finishes?

**Method:** for each generation-progress point s, two measurements per perceptual axis,
neither requiring correctness labels:

- **Commitment window (s_commit):** fork the trajectory at s into K stochastic continuations
  using a *marginal-preserving* SDE — drift `v + ½σ²·score`, with the score recovered from
  the model's own velocity via `score = (t·v − x)/(1 − t)` (exact for MMAudio's rectified-flow
  convention). If forks from s already agree on an axis (relative to independent
  regenerations), that decision is committed by s. The noise scale α controls fork strength;
  α = 0 reproduces the deterministic ODE exactly, which doubles as a built-in correctness
  check.
- **Readout window (s_read):** probe the Tweedie best-guess `x̂₀(s) = x + (1 − t)·v` of the
  final audio to see when each axis becomes predictable from the intermediate state.

**Make-or-break outcome:** `GO_MAP` (different axes have *separated* commitment windows,
beyond bootstrap CIs over videos) together with `GO_READOUT` (at least one axis is readable
by a cheap probe well before the end of generation, s_read ≪ 1). Pre-registered failure tokens exist for
every other outcome (e.g. `STOP_ADSR` if all commitment points coincide, or all lie only near
s = 1 — degenerating to a scalar; `STOP_PROJECT` on reliability failure).

## 2. Pipeline position (ORBIT stops)

| Stop | Meaning | Status |
|---|---|---|
| STOP A | Frozen proposal (`proposal/proposal_pack.json`) | done |
| STOP B | Implementation + cross-model plan-fidelity audit (`experiment/experiment_pack.json`) | **done — verdict `MATCHES_PLAN`** |
| STOP C | Formal diagnostic (Phase 0 feasibility on the real model) | **in progress — crux validated, checkpointed** |

## 3. Completed — implementation & audit (STOP B, 2026-06-09)

- **`foley_cw` package implemented:** numpy-only core (fork/trajectory SDE machinery,
  s↔t time maps, per-axis agreement, commitment/readout estimation, reliability gate,
  bootstrap statistics, decision logic, two CLIs). A model-agnostic integrator drives any
  backend; an **analytic synthetic Gaussian flow** serves as an exact oracle, so the
  highest-risk math (velocity→score, marginal preservation) is verified to machine precision
  on CPU.
- **515 tests pass.** Synthetic end-to-end dry-runs emit internally consistent decision
  tokens (these are *code checks*, not scientific evidence).
- **Cross-model audit:** 5 rounds of Codex review (`xhigh`); 16 findings fixed across rounds
  1–4, plus 2 minor notes in round 5 fixed after the verdict (18 total); final verdict
  **`MATCHES_PLAN`** (`orbit-research/PLAN_CODE_AUDIT.md`). Everything
  MMAudio-specific was left behind an explicitly UNVERIFIED seam, to be pinned on the real
  model in Phase 0.

## 4. Completed — Phase-0 crux on the real model (STOP C partial, 2026-06-09/10)

All on node an17 (1× A800, fp32), real MMAudio `small_16k`, user-approved scope.
Sources: `results/feasibility_report.md`, `results/score_sde_validation_report.md`,
`results/video_conditioning_derisk_report.md`.

- **Environment built and documented:** shared venv on /XYFS02 (torch 2.5.1+cu121); MMAudio
  vendored with md5-verified weights; CLIP cached for offline compute nodes. A `/dev/shm`
  staging recipe fixes Lustre's pathological cold import (> 400 s → ≈ 1 s).
- **Convention audit (against MMAudio source):** MMAudio integrates t: 0(noise) → 1(audio)
  with `v = x₁ − x₀`, `min_sigma = 0` — *identical* to the convention foley_cw's score formula
  was derived for. Identity s↔t map, no sign flip.
- **Phase 0.1 — trajectory access: PASS.** Intermediate latents extractable, resumable to
  completion, and Tweedie-decodable; all finite.
- **Phase 0.2 — velocity→score SDE validation: token `OK`** (the plan's stated highest
  silent-bug risk). α = 0 fork reproduces the ODE **exactly** (max L2 = 0.0); small-α
  continuity ratio 0.12 (threshold 2.0), monotone in α; 8/8 forks finite and non-trivial;
  fork diversity 0.085. Run unconditionally (cfg = 1.0) — the conversion tests network
  mechanics, not conditioning.
- **Video-conditioning de-risk: PASS.** CLIP + synchformer load from the offline cache;
  a real example clip encodes and generates valid audio end-to-end (8.0 s @ 16 kHz,
  rms 0.060, `results/vidcond_0B4dYTMsgHA_000130.wav`); and the SDE crux re-validated
  **under video conditioning** (token `OK`). CFG handling is principled: forks use
  cfg = 1.0 (where the score identity is exact); the deployed cfg ≈ 4.5 is used only for
  headline generation.

**Honest status:** `GO_MAPS_PHASE` has *not* been emitted and no scientific claim is made.
The generated wav is a smoke artifact. What is established is that the load-bearing mechanism
— fork-based commitment measurement on the real MMAudio — is feasible and the math transfers
exactly, in both unconditional and video-conditioned regimes.

## 5. Remaining work (planned, not started — checkpointed by user decision)

To finish Phase 0 and unlock the maps phase (`GO_MAPS_PHASE` requires all of these):

1. **Phase 0.3 — FoleyBench subset:** download the dataset, build
   `dataset_subset_manifest.md`.
2. **Phase 0.4 — event-anchor validation** on the selected clips.
3. **Phase 0.5 — reliability gate:** implement the real per-axis measurements
   (`foley_cw.axes.RealMeasurer`, currently stubs): presence, onset timing,
   audio-tagger class, material/timbre embedding — model choices for the tagger/embedder
   still to be decided — then require ≥ 3 axes to pass determinism + robustness + validity
   on real video-conditioned generations.

Then **Phases 1–3:** measure per-axis commitment and readout maps with
bootstrap-over-videos CIs and evaluate the pre-registered `GO_MAP` / `GO_READOUT` decision.
Generation fan-out can parallelize across an17 + an29 (both 8×A800, no time limit).

## 6. Notes

- Hard rule kept throughout: synthetic dry-runs are never presented as diagnostic evidence;
  diagnostic evidence comes only from the real-model runs in §4.
- Operational recipes (venv staging, offline HF env vars, the `HF_HUB_DISABLE_XET=1` fix for
  the huggingface_hub 1.18.0 download bug) are recorded in the run reports and in session
  memory for reproducibility.
- `orbit-research/ORBIT_STATE.json` (2026-06-09T16:12Z) predates the video-conditioning
  result; its "CLIP blocked" note is superseded by §4. Authoritative latest state:
  `orbit-research/diagnostics/diag_20260609_161203_985ce5eccd17/DIAGNOSTIC_CONTEXT.json`.
