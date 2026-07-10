# Control Design

> Generated view of `experiment/experiment_pack.json` (`controls`), faithfully transcribed
> from the frozen `refine-logs/EXPERIMENT_PLAN.md` (§1, §3, §9). The pack is the source of
> truth.

All baselines score the **same** candidate pools. Controls:

- **Video-prior normalization (mandatory).** Normalize commitment against
  `A_independent(video, axis)` = self-target agreement across N independent full generations
  of the same video. Report commitment as *gain* over this video-conditioned prior; never
  report raw fork agreement as commitment.
- **Pre-registered thresholds.** Freeze `θ_commit, θ_read, θ_rel, θ_robust, θ_cal` from
  pilot/anchor data and record in `go_no_go_decision.md` **before** computing the headline
  maps; report threshold sensitivity (sweep) afterward.
- **Bootstrap over videos.** Bootstrap unit = video (resample videos, not individual
  measurements) for all CIs on `s_commit`, `s_read`, and gaps. Declare minimum usable n per
  axis; underpowered axes are reported as such, not as results.
- **Reliability gate** (per axis, on generated audio, before any map): determinism
  (test-retest) + robustness (event-window shift / loudness-norm / resample / compression /
  small noise) + validity (vs a small human/MLLM calibration sidecar). Demote or drop on
  failure; material/fine-class is the prime demotion risk.
- **Same candidate pools.** Readout-side ladder: energy/onset; CLAP; SyncNet/AV-sync;
  ImageBind; audio tagger; MLLM-on-preview; (Phase 7) internal-feature probe. Policy-side:
  full BoN; same-compute BoN; random pruning; scalar DiffRS-style rejection; final-score
  reranking; seed restart; **oracle axis-gated pruning (upper bound)**. Rollback-side:
  Restart without axis gating; scalar rejection + rollback; full seed restart.
- **Legacy vs headline probes.** CLAP / SyncNet / ImageBind are **legacy** baselines, not
  the headline probe; the main probe-side comparison is a strong MLLM-on-preview.
