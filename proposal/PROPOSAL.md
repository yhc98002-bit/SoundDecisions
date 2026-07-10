# Proposal

> Generated view of `proposal/proposal_pack.json`. Do not treat this Markdown file as the source of truth.

## Status

- Pack status: `ready`
- Updated at: `2026-06-09T00:00:00Z`
- Safe next command: `/experiment-bridge "proposal/proposal_pack.json"`

## Problem Selection

- core_question: For a video-conditioned audio flow model, WHEN does the generator commit to each Foley output axis (presence, timing, class, material, multi-event binding), and WHEN can that commitment be read out by available probes? The inference policy (pruning/rollback) is strictly downstream of this map.
- handle: foley-cw
- pivot: Deliberate replacement of the earlier 'EventWISE-AV verifier + reranking' idea, now treated as crowded/weak as a headline (FoleyBench audits real V2A errors; MultiSoundGen does multi-event AV preference + DPO; VideoReward/VisionReward make multi-dimensional reward modeling standard; DiffRS already does scalar intermediate-step rejection). The scientific object moves from final-output scoring to trajectory structure.
- primary_task: Video-to-audio (V2A) / Foley generation, analyzed as a flow trajectory, not as a final-output scoring problem.
- status: fresh-start; no project-specific experiment has been run; make-or-break is the MMAudio commitment-readout probe (refine-logs/EXPERIMENT_PLAN.md).
- subtitle: Axis-Gated Candidate Pruning for Cross-Modal Audio-Visual Binding
- thesis: Video-to-audio models do not decide all audio-visual output axes at once: different Foley axes have distinct commitment windows (when the generator locks the value) and readout windows (when a probe can read it), and the gap between them determines when a system should prune, defer, or roll back candidate trajectories.
- title: When Are Foley Decisions Made? Commitment and Readout Windows in Video-to-Audio Flow Generation

## Abstract Task

- foley_axes: ['presence', 'gross timing', 'coarse event class', 'fine class / material', 'multi-event binding', 'offscreen hallucination']
- generator: MMAudio v1 — a public video-conditioned audio flow-matching model (Cheng et al., CVPR 2025) whose sampler integrates a learned velocity field; the v1 white-box backbone.
- glossary: {'self_target': "The model's own final axis value; the maps' target.", 'commitment': "Robustness of an axis's final value to downstream sampling stochasticity.", 'commitment_window_t_commit': 'Earliest trajectory time at which an axis self-target is stable across stochastic tail-forks.', 'readout_window_t_read': 'Earliest time a given probe predicts the axis self-target from x0(t) / a partial preview.', 'gap': 't_read(axis, probe) - t_commit(axis); a first-class result (R1/R2).', 'R1_uncommitted': 'Probe fails because the value is not yet locked -> defer is genuinely necessary.', 'R2_committed_but_unreadable': 'Value is locked but the probe cannot read it -> build a stronger probe / use internals; the gap is reducible.', 'x0_t_tweedie': 'The denoised best-guess of the final audio at step t; the readout probe input. Blurry early; this blur penalty is part of the gap.', 'probe_ladder': 'Ordered scorers of increasing power on x0(t): heuristics -> CLAP/sync -> audio tagger -> cheap learned verifier -> MLLM-on-preview -> (non-blocking) internal-feature probes.', 'fork_sampler': 'Marginal-preserving stochastic tail sampler: the ODE->SDE conversion (Flow-GRPO-style) with a single noise knob alpha; alpha=0 recovers the deterministic ODE. NOT Restart re-noising.', 'event_anchor': 'Timestamp(s) of the visible sound-producing event(s); an owned dependency.', 'calibration_sidecar': "A small human/MLLM subset used only to confirm the objective self-measurements are perceptually meaningful Foley correctness; never the maps' per-instance target."}

## Baseline Headroom

- compute_story: Report two quantities, not single-generation wall-clock: (1) scoring-call budget — a cheap verifier scores every step where MLLM-on-x0 per-step-per-candidate is prohibitive (the verifier's edge); (2) generator-completions / total NFE saved by axis-gated early pruning (the schedule's edge). Headline: capture most of the oracle (MLLM-on-x0) pruning gain at near-zero scoring cost, with advantage scaling on expensive generators.
- lineage: ['Critical windows in diffusion: image attributes emerge in narrow sampling intervals (Li et al., Critical Windows, ICML 2024, arXiv:2403.01633).', 'Intermediate-step rejection / restart: DiffRS uses a single scalar discriminator and re-noises low-quality samples (arXiv:2405.17880); Restart Sampling provides noise-and-re-denoise (Xu et al., NeurIPS 2023, arXiv:2306.14878).', 'V2A flow generation: MMAudio (Cheng et al., CVPR 2025) backbone; FoleyBench (Dixit et al., 2025, arXiv:2511.13219) data source; MultiSoundGen (arXiv:2509.19999) nearest neighbor for multi-event binding; PAVRM (arXiv:2511.21541) process-reward on noisy latents for video.', 'Commitment kernel reuses the ODE->SDE conversion standard in RL-for-flow (Flow-GRPO line): the score is closed-form from the velocity.']
- music_adsr_relation: The music-generation early-decidability finding is convergent evidence for the general principle only; it is NOT proof for V2A and must not be cited as such. V2A ordering is re-established from scratch.
- novelty_one_sentence: Critical windows for cross-modal Foley correctness, decomposed by axis, with separately measured commitment and readout maps, used to schedule axis-gated inference.

## Candidate Mechanisms

- C1-maps: C1a commitment map (generator-side, label-free): per-axis agreement of the self-target across stochastic tail-forks -> t_commit(axis) and surface A(axis,t,alpha). C1b readout map (probe-side): per-axis, per-probe prediction of the same self-target from x0(t) -> t_read(axis,probe). C1c commitment-readout gap = t_read - t_commit, attributed to R1/R2.
- C2-measurement: A reproducible objective per-axis self-target measurement, an event-anchor protocol, a reliability gate (test-retest on generated audio + a small perceptual-calibration sidecar), and explicit demotion rules. Unreliable axes are demoted, not forced.
- C3-pruning: A cascaded best-of-N policy that prunes a candidate population only on axes whose windows have closed and whose early-action precision is high; survivors continue; later axes evaluated at later windows; finish and rerank.
- C4-verifier: A small axis-conditioned verifier approximating the useful readout signal cheaply. Value claim: high-frequency low-cost intermediate-state scoring; NOT 'beats MLLM on completed audio'.
- C5-rollback: Use the stochastic tail mechanism (or Restart re-noising) to repair after an in-window axis failure. Not a v1 headline; runs only if forking is stable.

## Selected Sketch

- core: C1 commitment & readout maps of cross-modal Foley correctness.
- downstream: The inference policy (axis-gated pruning C3 / rollback C5) is strictly conditional and downstream of the maps; a failed first policy does not kill a good map.
- floor: DIAGNOSTIC (separated commitment but probe lags, R2-dominated) and NEGATIVE (no useful window separation) are genuinely publishable outcomes.
- make_or_break: Window separation across axes + early readability of early axes — both measurable WITHOUT correctness labels (GO_MAP + GO_READOUT).

## Assumptions

- A1-self-target: The maps target the model's OWN final axis value (the self-target, e.g. the tagger top-1 class of the completed audio), NOT human/MLLM correctness-vs-video. This is the central scoping decision; correctness enters only downstream via a calibration sidecar.
- A2-measurement-dependent: Maps are correctness-/human-label-free but NOT measurement-free: they depend on objective/model-based per-axis measurements (taggers, onset detectors). Unreliable measurements poison both maps, so the Phase-0 reliability gate is load-bearing. Never call them simply 'label-free'.
- A3-commitment-meaning: Under a deterministic ODE the trajectory is fixed by the initial noise, so 'commitment' is only meaningful as invariance under stochastic re-completion, measured with a marginal-preserving SDE fork kernel (alpha=0 recovers the ODE).
- A4-event-anchors: Visible-event timestamps (event anchors) are an owned dependency; timing and multi-event binding are uninterpretable without reliable anchors and reported anchor uncertainty.
- A5-single-backbone: One white-box flow V2A generator (MMAudio v1) is used for the maps; a second generator is only a later transfer sanity check.

## Open Risks

- H1-axis-commitment: Axes commit at different t; expected ordering presence/gross-timing/coarse-class (early) -> fine class/material (mid) -> multi-event binding (late). Tested by the commitment map (C1a). PLANNED/UNTESTED.
- H2-commitment-ne-readout: Some axes are committed before cheap probes can read them; the gap separates R1 (defer) from R2 (better probe). Tested by the gap (C1c). PLANNED/UNTESTED.
- H3-separation-enables-gating: If early axes commit and become readable before late axes, axis-gated pruning/defer is safer than scalar rejection. Tested by policy analysis (C3). PLANNED/UNTESTED.
- H4-scalar-insufficient: If windows differ across axes, a DiffRS-style single-scalar criterion cannot exploit them. Tested vs the scalar-DiffRS baseline. PLANNED/UNTESTED.
- H5-seed-determined: Offscreen hallucination / generic ambience may be predictable from the initial noise; if so the fix is reseed, not defer. Tested by seed-predictability analysis (separate). PLANNED/UNTESTED.
- claim-tiers: METHOD (separated windows + early readability + axis-gated pruning beats strongest baseline beyond CIs); DIAGNOSTIC-strong (separated commitment but cheap probes lag, R2-dominated); NEGATIVE-publishable (all axes commit together / only near final step -> collapses to scalar rejection); STOP (no trajectory access / no reliable axis measurement / forking not meaningful).
- anti-overclaim: Do NOT claim (unless directly proven): verifier is a general AV reward model; constructed/scalar negatives prove transfer; CLAP/SyncNet/ImageBind are useless (they are legacy baselines, not the main comparison); music-ADSR proves V2A ADSR; restart/rollback helps before window evidence; delayed-callback failures are common; a new foundation model / memory architecture / generator fine-tuning is needed.
- surprises-are-findings: Expect surprises and treat them as findings (e.g. coarse class splitting coarse-early/fine-late, or hallucination being a seed lottery rather than a window). The data, not H1's ordering, is the result.

## Source Markdown

- refine-logs/FINAL_PROPOSAL_SHORT.md
