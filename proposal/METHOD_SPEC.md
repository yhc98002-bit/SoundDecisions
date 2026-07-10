# Method Spec

> Generated view of `proposal/proposal_pack.json`. Detailed experiment planning still belongs to STOP B.

## Selected Sketch

- core: C1 commitment & readout maps of cross-modal Foley correctness.
- downstream: The inference policy (axis-gated pruning C3 / rollback C5) is strictly conditional and downstream of the maps; a failed first policy does not kill a good map.
- floor: DIAGNOSTIC (separated commitment but probe lags, R2-dominated) and NEGATIVE (no useful window separation) are genuinely publishable outcomes.
- make_or_break: Window separation across axes + early readability of early axes — both measurable WITHOUT correctness labels (GO_MAP + GO_READOUT).

## Candidate Mechanisms

- C1-maps: C1a commitment map (generator-side, label-free): per-axis agreement of the self-target across stochastic tail-forks -> t_commit(axis) and surface A(axis,t,alpha). C1b readout map (probe-side): per-axis, per-probe prediction of the same self-target from x0(t) -> t_read(axis,probe). C1c commitment-readout gap = t_read - t_commit, attributed to R1/R2.
- C2-measurement: A reproducible objective per-axis self-target measurement, an event-anchor protocol, a reliability gate (test-retest on generated audio + a small perceptual-calibration sidecar), and explicit demotion rules. Unreliable axes are demoted, not forced.
- C3-pruning: A cascaded best-of-N policy that prunes a candidate population only on axes whose windows have closed and whose early-action precision is high; survivors continue; later axes evaluated at later windows; finish and rerank.
- C4-verifier: A small axis-conditioned verifier approximating the useful readout signal cheaply. Value claim: high-frequency low-cost intermediate-state scoring; NOT 'beats MLLM on completed audio'.
- C5-rollback: Use the stochastic tail mechanism (or Restart re-noising) to repair after an in-window axis failure. Not a v1 headline; runs only if forking is stable.

## Assumptions

- A1-self-target: The maps target the model's OWN final axis value (the self-target, e.g. the tagger top-1 class of the completed audio), NOT human/MLLM correctness-vs-video. This is the central scoping decision; correctness enters only downstream via a calibration sidecar.
- A2-measurement-dependent: Maps are correctness-/human-label-free but NOT measurement-free: they depend on objective/model-based per-axis measurements (taggers, onset detectors). Unreliable measurements poison both maps, so the Phase-0 reliability gate is load-bearing. Never call them simply 'label-free'.
- A3-commitment-meaning: Under a deterministic ODE the trajectory is fixed by the initial noise, so 'commitment' is only meaningful as invariance under stochastic re-completion, measured with a marginal-preserving SDE fork kernel (alpha=0 recovers the ODE).
- A4-event-anchors: Visible-event timestamps (event anchors) are an owned dependency; timing and multi-event binding are uninterpretable without reliable anchors and reported anchor uncertainty.
- A5-single-backbone: One white-box flow V2A generator (MMAudio v1) is used for the maps; a second generator is only a later transfer sanity check.

## Baseline Headroom

- compute_story: Report two quantities, not single-generation wall-clock: (1) scoring-call budget — a cheap verifier scores every step where MLLM-on-x0 per-step-per-candidate is prohibitive (the verifier's edge); (2) generator-completions / total NFE saved by axis-gated early pruning (the schedule's edge). Headline: capture most of the oracle (MLLM-on-x0) pruning gain at near-zero scoring cost, with advantage scaling on expensive generators.
- lineage: ['Critical windows in diffusion: image attributes emerge in narrow sampling intervals (Li et al., Critical Windows, ICML 2024, arXiv:2403.01633).', 'Intermediate-step rejection / restart: DiffRS uses a single scalar discriminator and re-noises low-quality samples (arXiv:2405.17880); Restart Sampling provides noise-and-re-denoise (Xu et al., NeurIPS 2023, arXiv:2306.14878).', 'V2A flow generation: MMAudio (Cheng et al., CVPR 2025) backbone; FoleyBench (Dixit et al., 2025, arXiv:2511.13219) data source; MultiSoundGen (arXiv:2509.19999) nearest neighbor for multi-event binding; PAVRM (arXiv:2511.21541) process-reward on noisy latents for video.', 'Commitment kernel reuses the ODE->SDE conversion standard in RL-for-flow (Flow-GRPO line): the score is closed-form from the velocity.']
- music_adsr_relation: The music-generation early-decidability finding is convergent evidence for the general principle only; it is NOT proof for V2A and must not be cited as such. V2A ordering is re-established from scratch.
- novelty_one_sentence: Critical windows for cross-modal Foley correctness, decomposed by axis, with separately measured commitment and readout maps, used to schedule axis-gated inference.
