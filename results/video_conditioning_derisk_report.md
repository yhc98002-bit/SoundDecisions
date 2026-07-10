# Video-Conditioning De-risk Report (foley-cw)

**Run:** real MMAudio `small_16k`, enable_conditions=True, an17 (1× A800), fp32, 2026-06-09/10
(artifacts written 2026-06-10).
**Goal:** confirm the video-conditioned generation path works end-to-end before building the
full FoleyBench / measurement / reliability pipeline.

## VERDICT: PASS

| Step | Result |
|---|---|
| Load net + CLIP (DFN5B-CLIP-ViT-H-14, 987M) + synchformer (offline HF cache) | ✅ 66.5 s |
| Read VGGSound clip (`training/example_videos/0B4dYTMsgHA_000130.mp4`) via PyAV; encode CLIP + sync conditioning | ✅ 5.5 s |
| Generate video-conditioned audio (cfg=4.5, 25 steps) via foley_cw integrator → VAE → BigVGAN | ✅ 8.00 s @ 16 kHz, finite, rms=0.060 → `results/vidcond_0B4dYTMsgHA_000130.wav` |
| Re-check velocity→score SDE crux UNDER video conditioning (cfg=1) | ✅ token=`OK` (α=0 exact 0.0; small-α continuity ratio 0.10 monotone; fork validity 4/4; diversity 0.032) |

The score-conversion crux therefore holds in the **conditioned** regime, not only
unconditionally — the foley_cw rectified-flow math + `IdentitySToT` transfer to MMAudio in
both regimes.

## Environment notes (reproduction)

- CLIP download required `HF_HUB_DISABLE_XET=1` (the hf_xet backend in huggingface_hub 1.18.0
  raised "client has been closed"); model cached under `.hf_cache` on shared `/XYFS02`.
- an17 runs offline: `HF_HOME=.../.hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1`.
- Per-GPU-job pattern: one SSH session that rsyncs the venv to `/dev/shm/foley_venv` (≈22 s)
  then runs (Lustre cold-import is otherwise > 400 s; `/dev/shm` is wiped by RemoveIPC between
  sessions, so stage+run in a single session).

## Scope / what remains (NOT done; checkpoint here per user)

Full `GO_MAPS_PHASE` still needs Phase 0.3 FoleyBench dataset manifest, 0.4 event anchors,
0.5 reliability gate (≥3 reliable axes) — i.e. a FoleyBench download + real per-axis audio
measurements (`foley_cw.axes.RealMeasurer`: presence/onset-timing/audio-tagger-class/material-
embedding) + a calibration sidecar. These are the Phases-1-3 data+measurement infrastructure.
No GO_MAPS_PHASE emitted; no claims; the generated wav is a smoke artifact, not evidence for a
map.
