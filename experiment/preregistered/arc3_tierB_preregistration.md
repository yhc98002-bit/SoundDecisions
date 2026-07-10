# Arc-3 Tier-B Pre-registration (frozen BEFORE running; §1.5)

Pre-registers every NEW analysis: probe family, layers, metric, and the per-outcome decision
rule + token. Frozen by SHA256 (SHA256SUMS.json) before any Tier-B result is inspected.
Bootstrap unit = video throughout. Stage-0/M diagnostics are never evidence. Frozen quantities
(θ_cal, certified tuple, frozen interpretations) UNCHANGED.

Governing ruling: **F-1 is REFUTED** (Arc-2). METHOD tier no longer requires `F1_SUPPORTED`;
the METHOD path is the **B4 oracle→non-oracle bridge** (substantial non-oracle headroom recovery).

---

## B1 — Class readability, fairly tested

**Question:** is the final class linearly/nonlinearly readable from the generator's internal
features EARLY (≪ s_read_external = 0.75)? Current "R2/unreadable" rests on ONE linear pooled
probe; this tests it fairly.

**Probe families (all on the frozen 60/40 clip split; labels = each trajectory's own final class
self-target; cfg=1.0 independents pool):**
1. Linear ridge on POOLED per-layer features (the existing baseline — re-run for reference).
2. **MLP** (1 hidden layer, width 256, ReLU, weight-decay 1e-3, early-stopped on a train-internal
   split) on POOLED per-layer features.
3. Linear ridge + MLP on **UN-POOLED per-token features** (mean removed → kept per token; probe
   sees the full (T, D) flattened or token-mean-max concat) — requires a GPU re-tap of un-pooled
   features (token-level activations, manual §1.4) on the 200-clip pool.
4. **Cross-attention-map probe**: pool the video→audio cross-attention weights per layer as the
   probe input (where the model routes conditioning) — GPU re-tap.

**Layers:** swept over all 12 joint blocks; **s-grid:** the Phase-1 grid (0.05…0.90).
**Metric:** held-out class accuracy on the eval split; chance = eval majority-class prior
(reported alongside). Bootstrap by video for CIs.

**Decision rule (per the best probe family × layer):**
- `s_read_internal_class` = min s with best-probe eval accuracy ≥ θ_read (0.70) AND ≥ chance+0.15.
- If `s_read_internal_class ≤ 0.45` (early, well below the 0.75 external) → emit
  **`CLASS_INTERNAL_READOUT_FOUND`** (TOP result: strengthens "generator knows before the sound
  shows"; the winning probe becomes the class feature-head candidate for B4). CONTINUE.
- Else (never reaches θ_read early, on ≥2 probe families incl. MLP + per-token) → **class = R2 on
  strong evidence** (R2_CLASS_CONFIRMED). CONTINUE. No pause either way.

## B2 — Conditioning-channel audit (MMAudio bottleneck)

**Question:** do the RAW video-conditioning features (CLIP + Synchformer, pre-DiT) carry the
class at all? Tests whether class non-readability is a *conditioning bottleneck* vs a DiT-internal
property.

**Probe:** linear ridge + MLP (as B1) predicting the clip's measured class from the raw CLIP and
Synchformer conditioning tensors (pooled), frozen split. **Metric:** eval accuracy vs chance.
**Decision rule:** if conditioning-feature class accuracy is low (≤ chance+0.15) AND substantially
below the DiT-internal best (B1) → emit **`COND_BOTTLENECK`** and frame as a sharp single-model
MMAudio conditioning-bottleneck mechanism (mechanistic finding, NOT a scope collapse); validate
the direction at large_44k (D). CONTINUE regardless.

## B3 — Seed-floor direct test (well-powered, reduced-dim)

**Question:** does the initial noise seed predict the final class above chance (a seed floor), and
does that grip grow with cfg? Fixes the Arc-2 dial's underpower (5000-dim, ~230 samples).

**Reduced-dim:** project the s=0 noise latent to **d=256 via a fixed Gaussian random projection**
(seed 0, frozen) before probing (Johnson–Lindenstrauss; documented). **Probe:** ridge + MLP.
**Pools:** (i) full cfg=1.0 pool (needs the 200-clip s=0 noise — GPU re-extract or reuse stored
priors); (ii) the cfg-dial clips across cfg∈{1,1.5,2,2.5,3,4.5}. **Metric:** eval accuracy vs
chance, bootstrap by video; trend = OLS slope of accuracy vs cfg with a bootstrap CI.
**Decision rule:**
- accuracy > chance+0.10 at cfg=1.0 → **seed floor EXISTS** (`SEED_FLOOR_CONFIRMED`); does NOT
  resurrect F-1 by itself.
- slope across cfg ≈ 0 (CI includes 0) or < 0 → F-1 stays refuted (seed grip flat/shrinking).
- slope > 0 beyond CI → revise narrative to **"mixed entropy-reduction + seed-amplification"**
  (`SEED_AMPLIFICATION`). CONTINUE either way; no pause.

## B4 — Oracle→Non-oracle bridge (METHOD make-or-break; offline)

**Question:** how much of the oracle axis-gated headroom (final-corr 0.785 − scalar 0.37 = 0.415)
does a REALISTIC non-oracle scorer recover? This REPLACES `F1_SUPPORTED` as the METHOD path.

**Non-oracle scorer (frozen split, per axis):** the best available readout per axis —
- class: B1's winning internal probe if `CLASS_INTERNAL_READOUT_FOUND`, else the Phase-2 external
  audio-tagger-on-preview readout;
- presence/timing/material: the Phase-2 external readout probes (per-axis).
**Replay:** Phase-4 axis-gated pruning using the NON-oracle per-axis scores at each axis's
s_commit, vs BoN, same-compute BoN, scalar DiffRS, SMC-scalar, final-rerank, and the oracle, on the
SAME cached cfg=4.5 pool, matched generator-NFE AND scoring-calls. **Metric:** final + per-axis
correctness (proxy = per-clip majority self-target, documented), winner-retention, false-prune,
regret, the Pareto; **headroom recovery per axis** = (non_oracle − scalar_baseline)/(oracle −
scalar_baseline), clipped to [0,1], bootstrap by video.
**Decision rule (sets the tier, OFFLINE — never pauses; the online run is a later PI call):**
- mean per-axis recovery ≥ 0.5 (substantial) → **METHOD tier** (`BRIDGE_METHOD`).
- 0.2–0.5 → **DIAGNOSTIC-strong** (`BRIDGE_PARTIAL`).
- < 0.2 → **DIAGNOSTIC** (`BRIDGE_WEAK`); recovery is bounded by readout quality (axis-dependent),
  reported honestly per axis.

## C — Two budgets + entropy lens (assembly, no new generation)
Observational determination budget (i) and **causal conditioning responsiveness** (ii, the
cond-swap follow/retention, explicitly NOT the conditioning share) reported SIDE BY SIDE, with the
distinct-class-count vs cfg (iii, 4.83→3.62) as the explainer for their divergence. Pre-registered
as descriptive (no token).

## D — Replication / sanity (pre-registered rules)
- large_44k commitment-only (100 clips) + scaled Gate-A: **catastrophic contradiction** (ordering
  inverts or separation vanishes, not merely shifts) → PAUSE (trigger b); else CONTINUE.
- Phase-4 scalar-baseline tie sanity: confirm the 0.370 BoN/DiffRS/SMC tie is real (no wiring bug);
  if genuinely no-help, that supports the axis-gating thesis.
- Material 2nd-embedder validity (CLAP vs PANNs already ρ=0.49; add BEATs if wired) — elevated.

---
**Freeze:** this file's SHA256 is recorded in `experiment/preregistered/SHA256SUMS.json` before any
Tier-B result is read. Any deviation requires an explicit amendment + recorded rationale (§15.7).
