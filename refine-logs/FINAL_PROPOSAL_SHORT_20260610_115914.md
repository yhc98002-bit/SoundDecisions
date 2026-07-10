# Final Proposal — Sound Decisions (`foley-cw`)

**Title:** Sound Decisions: When Are Foley Decisions Made in Video-to-Audio Flow Generation?
**Subtitle:** Commitment and Readout Maps for Cross-Modal Foley Generation
**Project handle:** `foley-cw`
**Primary task:** video-to-audio (V2A) / Foley generation, analyzed as a *flow trajectory*, not as a final-output scoring problem.
**Status:** fresh-start. No project-specific experiment has been run. The make-or-break test is the MMAudio commitment–readout probe (see `EXPERIMENT_PLAN.md`).

> This document is written for autonomous research agents and collaborators. It states *what* we are claiming and *why*. The *how* (procedures, gates, decision tokens) lives in `EXPERIMENT_PLAN.md`. Read both before acting.

**Working abstract.** We map when a cross-modal generator decides each aspect of its output, show that some decisions may be readable from the model's internal states before they are audible in `x0(s)` previews, and use the resulting commitment–readout map to schedule inference-time compute. *Every sentence above is a planned claim — nothing in it is an empirical result yet.*

---

## 0. The pivot (read first)

This project is a **deliberate replacement** of the earlier "EventWISE-AV verifier + reranking" idea. That earlier framing — "train an event-level audio-visual verifier and rerank V2A candidates" — is now treated as **crowded and weak as a headline** (FoleyBench already audits real V2A errors; MultiSoundGen already does multi-event AV preference + DPO; VideoReward/VisionReward make multi-dimensional reward modeling standard; DiffRS already does intermediate-step rejection with a scalar discriminator).

The scientific object has moved from **final-output scoring** to **trajectory structure**:

> Old question: *Is the final generated audio good?*
> New question: **During generation, when does the model commit to each kind of audio-visual correctness, and when can we read enough to act?**

The verifier is now only a tool. The contribution is the **commitment–readout map** of cross-modal Foley correctness. The paper's center of gravity is: (1) the per-axis decision/commitment map; (2) the readout map, including internal-feature readout; (3) the commitment–readout gap and its R1/R2 interpretation; (4) causal validation of commitment as irreversibility; (5) policy as a downstream scheduler — not the main novelty and not a head-to-head fight against SMC-ITA.

### Why policy alone is no longer enough (June 2026 landscape)

The inference-time-alignment layer for V2A has become crowded. SMC-ITA applies sequential Monte Carlo inference-time alignment to V2A with intermediate rewards and lookahead; a concurrent line studies inference-time scaling for joint audio-video generation with multi-verifier frameworks. *(Citations per PI review — TODO: verify exact BibTeX; do not assert bibliographic details beyond this until verified.)* This does not kill the project: neither line measures commitment, decomposes axes, or asks *when* decisions are made. The consequence is a reweighting, not a retreat:

> Recent V2A inference-time alignment work makes a pure policy paper crowded: scalar or multi-verifier rewards can already guide population search. Our claim is orthogonal. We ask when each Foley axis becomes decided and when it becomes readable. SMC-style search can then be treated as one downstream consumer of the map, not as the object we are trying to beat by brute force.

The policy layer is therefore table-stakes. What makes the paper distinctive is the map, with internal readout and causal validation as its strongest supporting evidence. In claim-budget terms (max 2 primary claims): **primary claim 1** is the per-axis commitment–readout map and its gap structure (C1+C2 — with C3 internal readout as evidence for the readout half and C4 causal validation as evidence for the commitment half); **primary claim 2** is that the map schedules inference-time compute (C5, conditional). C3 and C4 are subordinate evidence, not co-equal headlines.

---

## 1. One-Sentence Thesis

**Video-to-audio models do not decide all audio-visual output axes at once: different Foley axes have distinct *commitment windows* (`s_commit`, when the generator locks the value) and *readout windows* (`s_read`, when a probe can read it), and the gap between them (`gap = s_read − s_commit`) determines when a system should prune, defer, or roll back candidate trajectories.**

---

## 2. Core Scientific Question

> For a video-conditioned audio **flow** model, **when** (in generation progress `s`) does the generator commit to each Foley output axis (presence, timing, class, material, multi-event binding), and **when** can that commitment be read out by available probes — external `x0(s)` previews or internal features?

The inference policy (map-scheduled pruning / rollback / resampling) is strictly downstream of this map.

---

## 3. Key Terms (agent glossary)

| Term | Meaning |
|---|---|
| **Generation progress `s`** | The canonical reported time axis: `s = 0` = pure noise, `s = 1` = final audio. MMAudio's internal integration time `t` is **never** a reported axis; it is mapped to `s` once, in audited code (see `EXPERIMENT_PLAN.md` §2). |
| **V2A flow model** | A video-conditioned audio generator whose sampler integrates a learned velocity field (flow matching). MMAudio is the v1 backbone. |
| **Foley correctness axis** | A single correctness dimension: presence, gross timing, coarse class, fine class/material, multi-event binding, offscreen hallucination. |
| **Self-target** | The **model's own final axis value** (e.g., the tagger top-1 class of the *completed* audio). The maps target this, **NOT** human correctness-vs-video. This is the central scoping decision. |
| **Commitment** | Robustness of an axis's final value to **downstream sampling stochasticity**. Under a deterministic ODE the whole trajectory is fixed by the initial noise, so "commitment" is only meaningful as *invariance under stochastic re-completion*. |
| **Commitment window / `s_commit`** | Earliest generation progress at which an axis's self-target is stable across stochastic tail-forks, normalized over the video-conditioned prior. |
| **Readout window / `s_read`** | Earliest progress at which a given probe predicts the axis self-target from `x0(s)` / a partial preview / internal features. |
| **Commitment–readout gap** | `gap(axis, probe) = s_read(axis, probe) − s_commit(axis)`. A first-class result (see R1/R2). |
| **R1 (uncommitted)** | Probe fails because the value is not yet locked → **defer is genuinely necessary**. |
| **R2 (committed-but-unreadable)** | Value is locked but the probe can't read it → **build a stronger probe / use internals**; the gap is reducible. |
| **`x0(s)` / Tweedie estimate** | The denoised "best guess of the final audio" at progress `s`. The external readout probe input. Blurry early; this blur penalty is part of the gap. |
| **Internal readout** | Linear probes on **cached** internal features (pooled DiT block features, latents, AV cross-attention summaries) testing whether a commitment is readable before `x0(s)` previews show it — "the generator knows before the audio shows". |
| **Probe ladder** | Ordered scorers of increasing power applied to `x0(s)`: heuristics → CLAP/sync → audio tagger → cheap learned verifier → MLLM-on-preview → internal-feature probes. |
| **Marginal-preserving stochastic tail sampler** | The commitment fork kernel: the ODE→SDE conversion (Flow-GRPO-style) with a single noise knob α; α=0 recovers the deterministic ODE. **Not** Restart re-noising. |
| **α-robust ordering** | Absolute `s_commit` values are relative to model, α, discretization, and fork kernel. The robust claim is the **rank order** of `s_commit` across axes, which must be stable across the valid-α grid. |
| **Video-determined / high-`A_independent`** | `A_independent(video, axis) ≥ τ_video`: the video alone pins the value; excluded from normalized commitment (denominator instability), reported as its own category. |
| **Seed-determined** | A value or failure predictable from the initial noise / very early state, before meaningful trajectory commitment (e.g., possibly offscreen hallucination). |
| **Causal irreversibility** | Intervention-based validation of commitment: pushing an axis toward an alternative value should flip the final self-target more often *before* `s_commit` than *after*. |
| **Map-scheduled search** | Using the maps to schedule population search: cascaded best-of-N pruning, axis-aware SMC-style resampling, rollback. |
| **Event anchor** | The timestamp(s) of the visible sound-producing event(s) in the video. Required for timing/binding. An owned dependency. |
| **Correctness-calibration sidecar** | A small human/MLLM subset used only to confirm the objective self-measurements are perceptually meaningful Foley correctness — never used as the maps' per-instance target. |

---

## 4. Background Lineage & Novelty

This sits at the intersection of four lines; the novelty is none of them alone.

1. **Critical windows & speciation / dynamical regimes in diffusion** — final image features (class, color) emerge in narrow sampling intervals; models progressively "decide" attributes coarse-to-fine (*Li et al., Critical Windows, ICML 2024, arXiv:2403.01633*). The statistical-physics line describes **speciation**: broad structure such as class is sealed in a narrow transition, with spectral / geometry-based accounts of when speciation happens; recent coupled multimodal diffusion theory argues cross-modal coupling strength shifts speciation times and creates tunable synchronization between modalities. *(Biroli et al.; Bonnaire / de Bortoli / Mézard line; Raya & Ambrogioni; Georgiev et al.; Sclocchi et al.; coupled multimodal speciation theory, 2026 — TODO: verify exact BibTeX.)*
2. **Intermediate-step rejection / restart / inference-time alignment** — DiffRS evaluates intermediate samples with a **single scalar** discriminator and re-noises low-quality ones (arXiv:2405.17880); Restart Sampling provides the noise-and-re-denoise mechanism (Xu et al., NeurIPS 2023, arXiv:2306.14878); SMC-ITA brings sequential-Monte-Carlo inference-time alignment with intermediate rewards to V2A, and multi-verifier frameworks study inference-time scaling for joint AV generation *(TODO: verify exact BibTeX)*.
3. **V2A flow generation** — MMAudio is a public flow-matching V2A model (Cheng et al., CVPR 2025), a natural white-box backbone. FoleyBench (Dixit et al., 2025, arXiv:2511.13219) is a Foley-focused data source. MultiSoundGen (arXiv:2509.19999) is the nearest neighbor for multi-event binding. PAVRM ("Video Generation Models Are Good Latent Reward Models", arXiv:2511.21541) does process-reward on noisy latents for **video** (our analog is audio + axis-deferred).
4. **RL-for-flow machinery** — the commitment kernel reuses the **ODE→SDE conversion** that is now standard in the Flow-GRPO line: a deterministic flow ODE has an equivalent reverse SDE with the same marginals; the score is closed-form from the velocity.

> **Novelty:** per-axis commitment **and** readout maps for cross-modally conditioned Foley axes, including timing and temporal binding, measured in a real V2A flow system; the commitment–readout gap is treated as a first-class object and used to schedule inference-time compute.

> If the axes do not separate, this is not a failed method paper; it becomes a diagnostic / negative result about the absence of useful decision-window separation in the tested V2A system.

We do **not** claim to be first to observe critical windows. The positioning is: real V2A / Foley, not image-only; cross-modal conditioning; axis decomposition beyond class; a readout map, not just commitment/speciation; the commitment–readout gap; and an inference scheduler built on the map.

**Relation to music-ADSR (sibling project):** the music-generation "early-decidability / bad-stays-bad" finding is **convergent evidence** for the general principle that generative trajectories decide different axes at different times. It is **NOT** proof for V2A and must not be cited as such. V2A is strongly cross-modally conditioned, so the early/late ordering here is re-established from scratch.

---

## 5. Hypotheses (all currently *planned / untested*)

| ID | Hypothesis | Tested by |
|---|---|---|
| **H1 — axis-specific commitment** | Axes commit at different `s`. Expected ordering: `presence / gross timing / coarse class` (early) → `fine class / material` (mid) → `multi-event binding` (late). | Commitment map (C1). |
| **H2 — commitment ≠ readout** | Some axes are committed before external probes can read them; the gap separates R1 (defer) from R2 (better probe). | Commitment–readout gap (C2). |
| **H3 — internal readout precedes previews** | Internal features can read some commitments earlier than `x0(s)` previews: `s_read(internal) ≈ s_commit ≪ s_read(x0-probes)` for at least one important axis. | Internal readout (C3). |
| **H4 — coupling strength shifts commitment** | Stronger cross-modal conditioning (e.g., CFG scale or sync-feature conditioning) shifts commitment earlier. | Coupling-strength ablation. |
| **H5 — causal irreversibility** | The commitment boundary is causal: interventions before `s_commit` can flip an axis's final self-target more often than interventions after. | Causal intervention (C4). |
| **H6 — the map schedules search** | The map schedules population search (cascaded BoN, axis-aware SMC-style resampling) better than unscheduled use of the same budget; a single-scalar criterion cannot exploit per-axis window structure. | Map-scheduled search (C5). |
| **H7 — seed-determined phenomena exist** | Some phenomena (e.g., offscreen hallucination / generic ambience) are seed-determined rather than trajectory-window-determined; the fix is **reseed**, not defer. | Seed-predictability analysis. |

**Expect surprises and treat them as findings**, e.g. coarse class may split into coarse-early / fine-late, or hallucination may be a seed lottery rather than a window. The data, not H1's ordering, is the result.

---

## 6. Contributions

### C1 — Per-axis decision maps
Commitment maps normalized over video-conditioned priors; `A(axis, s, α)` surfaces; **α-robust ordering** as the headline claim form. Includes the axis measurement & calibration protocol: reproducible self-target measurement per axis, the **event-anchor** protocol, the three-part **reliability gate** (determinism + robustness + validity, with a small perceptual-calibration sidecar), and explicit demotion rules — unreliable axes are demoted, not forced.

### C2 — Readout maps and the commitment–readout gap
External `x0(s)` probes, ODE-target and fork-majority targets, R1/R2 interpretation. The gap is a first-class object, not a nuisance.

### C3 — Internal readout: "the generator knows before the audio shows" *(supporting evidence for the readout half of primary claim 1)*
Always cache pooled DiT / latent / AV cross-attention features during Phase 1/2 generation. Linear probes on cached features test whether `s_read(internal) ≈ s_commit ≪ s_read(x0-probes)`. This is the cleanest test of probe limitation (R2) and the strongest path to a cheap process reward.

### C4 — Causal irreversibility / editing-window validation *(validation of the commitment half of primary claim 1 — validates the Phase-1 fork-agreement definition of commitment, never replaces it as the operating definition)*
A small intervention around measured `s_commit`: push one axis toward an alternative value just before vs. just after the commitment boundary. If flip rate collapses after the boundary, commitment is not merely observational fork agreement but causal irreversibility.

### C5 — Map-scheduled inference-time search
Axis-gated pruning / rollback and SMC-style resampling are downstream schedulers. The map is presented as a **scheduler for population search**, not as a direct competitor to scalar SMC-ITA. The cheap process-aware verifier (plan Phase 5) is retained inside C5 as a conditional cost-reduction component — high-frequency, low-cost intermediate-state scoring — not as a headline contribution.

---

## 7. Correctness factorization

The maps target the model's self-target, not human correctness. Policy uses correctness only after a value is readable. For an axis whose value can be read, correctness can be factored as:

`axis correctness = match(readable axis value, video/event anchor)`

The matching relation is time-independent; the time-dependent part is readout fidelity. Once "class = door slam" is readable at `s = 0.3`, checking whether it matches the video anchor does not require waiting until the final audio, except insofar as the readout itself is unreliable. This factorization is the load-bearing logic for Phase 4 and also defines the boundary of the framework: holistic qualities that do not factor through a readable axis value are out of scope for the v1 map.

---

## 8. Claim Tiers / Minimum Publishable Outcomes

| Tier | Required evidence |
|---|---|
| **METHOD / full** | Separated α-robust commitment ordering, early readout, internal probes closing the R2 gap for at least one important axis, causal irreversibility evidence, and map-scheduled search improving fixed-budget correctness over strong baselines. |
| **SCIENCE / diagnostic strong** | Separated α-robust commitment windows and a clear commitment–readout gap; internal probes show whether the gap is probe-limited. Policy may be weak, but the map is publishable. |
| **CAUSAL diagnostic** | Fork agreement windows are supported by intervention asymmetry around `s_commit`, even if downstream pruning is limited. |
| **NEGATIVE** | No useful axis separation or all axes commit only near final audio; method degenerates to scalar rejection. Honest diagnostic, still publishable. |
| **STOP** | No trajectory access, invalid SDE/fork kernel, unreliable measurements, or unusable anchors. |

The make-or-break for the science is **window separation + early readability**, both measurable **without correctness labels**. The floor (SCIENCE / CAUSAL / NEGATIVE) is genuinely good — a strong science paper does not require the policy to dominate.

> The tier ladder is a **fallback/degradation ladder for publishable outcomes**, not a list of co-equal primary claims; the primary-claim budget stays at 2 (see §0 and §6).

---

## 9. Compute Story (how the method is justified)

MMAudio is small and fast, so do **not** sell single-generation wall-clock savings. Report **two** quantities:
1. **Scoring-call budget** — a cheap verifier scores every step; MLLM-on-`x0(s)` per-step-per-candidate is prohibitive. This is the verifier's edge.
2. **Generator-completions / total NFE saved** by map-scheduled early pruning. This is the schedule's edge.

Headline: *capture most of the oracle (MLLM-on-`x0(s)`) pruning gain at near-zero scoring cost, and the advantage scales on expensive generators where generator FLOPs dominate.* Any comparison against SMC-ITA-style search is valid only under matched NFE **and** matched scoring-call budgets with comparable candidate-pool accounting.

---

## 10. Anti-Overclaim Rules (hard)

Do **not** claim, unless directly proven by this project's evidence:
- the verifier is a general reward model for all AV generation;
- constructed/scalar negatives prove real transfer;
- global metrics (CLAP/SyncNet/ImageBind) are useless — they are **legacy baselines**, not the main comparison (the main probe-side comparison is a strong MLLM-on-preview);
- music-ADSR proves V2A ADSR;
- restart/rollback helps before window evidence exists;
- delayed-callback failures are common;
- a new foundation model, memory architecture, or generator fine-tuning is needed.

Additional rules (PI review, 2026-06):
- Do **not** claim absolute `s_commit` is model-intrinsic; it is relative to model, α, discretization, and fork kernel. The robust object is the α-robust ordering and the gap structure.
- Do **not** claim the policy beats SMC-ITA unless actually tested under matched NFE and matched scoring-call budgets.
- Do **not** claim internal probes are free in storage; they are zero extra forward passes but have storage and probe-fitting costs.
- Do **not** claim coupled multimodal speciation theory proves our results; it motivates our ablations.
- Do **not** claim holistic quality can be handled unless it factors into readable axes (see §7).

The narrow honest claim:
> *For video-conditioned audio flow generation, different Foley correctness axes are committed and become readable at different points along the trajectory; we map this structure, test whether commitment is causally irreversible, and test whether the map licenses axis-gated compute allocation.*

---

## 11. Planned Figure 1 (decision taxonomy)

**Planned Figure 1:** not a simple early/mid/late timeline. Use a decision taxonomy:
1. **video-determined** axes: `A_independent` already near 1;
2. **seed-determined** phenomena: predictable from initial noise / very early state;
3. **trajectory-early** axes: low `A_independent`, early `s_commit`;
4. **trajectory-late** axes: low `A_independent`, late `s_commit`;
5. **committed-but-unreadable** axes: R2 gap, candidates for internal readout.

This taxonomy is also the reporting scheme in `EXPERIMENT_PLAN.md` (gap report) and guards against overclaiming a single "decision timeline".

---

## 12. Not in v1

Delayed callback; long-horizon world memory; final-audio-only verifier as headline; "music proves V2A"; CLAP/SyncNet as main baselines; generator fine-tuning; large human labeling before commitment curves exist; beating SMC-ITA head-to-head as a headline; internal-feature probes as a *Phase-0 blocking* requirement (their **logging** during Phase 1/2 generation is mandatory; their **analysis** is non-blocking for `GO_MAP`).

---

## 13. Next Gate

Run the **MMAudio commitment–readout probe** (`EXPERIMENT_PLAN.md`): first **Phase 0A** (video-conditioned micro-map sanity test — pipeline sanity only, not evidence), then the full **Phase 0** feasibility + reliability gate, then Phases 1–3.

Required first deliverables: `micro_map_sanity_report.md` (Phase 0A), then `axis_reliability_report.md`, `commitment_map.csv`, `readout_map.csv`, `commitment_readout_gap_report.md`, `go_no_go_decision.md`.

Allowed decision tokens: `GO_FULL_PHASE0`, `FIX_PIPELINE_CONVENTION`, `FIX_MEASUREMENT_CHAIN`, `FIX_SCORE_CONVERSION`, `NO_TRAJECTORY_ACCESS`, `GO_MAPS_PHASE`, `GO_MAP`, `GO_READOUT`, `GO_INTERNAL_READOUT`, `R2_CONFIRMED`, `GO_CAUSAL_COMMITMENT`, `CAUSAL_INCONCLUSIVE`, `GO_CHECKPOINT_SANITY`, `GO_POLICY`, `GO_RESTRICTED`, `GO_DIAGNOSTIC`, `STOP_ADSR`, `STOP_PROJECT`.
