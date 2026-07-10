# Experiment Plan — Sound Decisions (`foley-cw`)

**Project:** Sound Decisions: When Are Foley Decisions Made in Video-to-Audio Flow Generation? (Commitment and Readout Maps for Cross-Modal Foley Generation.)
**Purpose:** execution index for autonomous research agents. Defines what to run, what *not* to run, what evidence each gate requires, and which decision token to emit.
**Fresh-start status:** no project-specific experiment has been run. **Start from Phase 0A, then the full Phase 0. Do not skip the feasibility/reliability gate.**

> Read `FINAL_PROPOSAL_SHORT.md` first for the conceptual frame and glossary. This file is the operational contract.

---

## 1. Project Summary

We measure, for a video-conditioned audio **flow** model (MMAudio, v1), two things per Foley correctness axis:
- **Commitment** — when the axis's *self-target* (the model's own final value) becomes stable under stochastic re-completion, **above the video-conditioned prior** (`s_commit`).
- **Readout** — when a probe can predict that same self-target from the running `x0(s)` (`s_read`), and — via cached internal features — whether the generator's internal states are readable **before** the `x0(s)` previews are (internal readout).

The **make-or-break test** is whether these windows **separate across axes** (with α-robust ordering) and whether early axes are **readable early**. The inference policy (map-scheduled pruning / rollback / resampling) is conditional and downstream — a scheduler consuming the map, not the headline.

**Two standing rules an agent will be tempted to break:**
1. **The maps target the model's own final axis value, NOT human/MLLM correctness-vs-video.** Correctness enters only in the conditional policy phase, via a calibration sidecar.
2. **The maps are human-label-free and correctness-label-free, but NOT measurement-free.** They depend on objective / model-based per-axis measurements (taggers, onset detectors). Unreliable measurements poison both maps. Never call them simply "label-free." The reliability gate (Phase 0) is therefore load-bearing.

> **Novelty boundary (enforce repeatedly):** this is *not* a final-audio verifier paper and *not* a generic DiffRS clone. The novelty is **per-axis commitment/readout maps for cross-modal Foley decisions**, the commitment–readout gap, internal readout, and causal validation. If axes do **not** separate, method novelty collapses to scalar rejection → route to DIAGNOSTIC / NEGATIVE framing; do **not** force a method claim.

---

## 1b. PI Review Amendments (2026-06-10)

Changes applied to this plan after PI review, so future agents understand why the plan moved:

1. Speciation / dynamical-regimes lineage added to the proposal (`FINAL_PROPOSAL_SHORT.md` §4).
2. **Phase 0A micro-map sanity stage** inserted before the full Phase 0 (pipeline sanity, not evidence).
3. **Internal-feature logging is now mandatory instrumentation** before any large Phase 1/2 generation (analysis stays conditional/non-blocking).
4. **High-`A_independent` cap / video-determined exclusion rule** added to Phase 1 normalization.
5. **α-robust ordering requirement** added to `GO_MAP`.
6. **SMC-ITA and multi-verifier AV inference-time scaling** added as Phase-4 baselines / positioning.
7. **Correctness factorization** made explicit in the Phase-4 preflight (and proposal §7).
8. **Optional second-checkpoint commitment-only sanity** added (P1 robustness).
9. **Causal intervention validation** added (Phase 3b).
10. **Coupling-strength ablation** (CFG scale / sync-feature drop) added (P1, non-blocking).

*Scope note:* this revision edits planning documents only. Evidence-status sections below are intentionally unchanged by this revision; the current diagnostic record lives in `orbit-research/ORBIT_STATE.json` and `results/`.

---

## 2. Notation & Conventions

- **Generation progress `s ∈ [0,1]`** is the canonical time axis for **all** reported windows: `s = 0` = start of generation (pure noise), `s = 1` = final audio.
- Internally, map `s` to MMAudio's actual integration time `t` **once**, in code, after auditing MMAudio's direction (some flow models integrate `t: 1→0`). **Never expose a raw `t` without stating the convention.** This avoids silent time-direction bugs.
- `x_s` = intermediate state at progress `s`. `x0(s)` = the Tweedie/denoised "best guess of final audio" at `s` (the readout probe input).
- `α` = stochasticity knob of the marginal-preserving SDE fork kernel; `α = 0` = deterministic ODE.

---

## 3. Statistical Protocol & Pre-registration (applies to all phases)

- **Thresholds are frozen before inspecting the main curves.** `θ_commit`, `θ_read`, `θ_rel` (determinism), `θ_robust`, `θ_cal` (validity) are set from pilot/anchor data only, recorded in `go_no_go_decision.md` *before* the headline maps are computed. **Sensitivity analysis** (sweep each threshold) is reported afterward.
- **The ordering-stability threshold (`θ_order`, for α-robust ordering) and the video-determined cap `τ_video` are likewise frozen from pilot data before the headline curves are computed.** Pilot default for `τ_video` is 0.90 with sensitivity reported; `θ_order` has no pre-set numeric value — it must be set during the pilot and recorded in `go_no_go_decision.md` before Phase 1–3 headline analysis.
- **Bootstrap unit = video** (resample videos, not individual measurements) for all CIs on `s_commit`, `s_read`, and gaps.
- **Minimum usable `n` per axis** is declared in `dataset_subset_manifest.md`; an axis below its minimum is reported as underpowered, not as a result.
- A window (`s_commit`/`s_read`) is only valid if its axis passed the reliability gate.

---

## 4. Evidence Status

**Available now:** the design; the lineage (Critical Windows; speciation/dynamical regimes; DiffRS; Restart Sampling; SMC-ITA-style inference-time alignment; MMAudio; FoleyBench; Flow-GRPO ODE→SDE machinery). Nothing empirical.

**Not yet available:** MMAudio trajectory access; a *validated* velocity→score SDE; a dataset/event-anchor manifest; per-axis measurement reliability; the commitment map; the readout map; the gap; internal-probe results; causal-intervention results; any policy result.

**Hard rule:** emit no METHOD-tier or `GO_POLICY` claim until both maps exist and pass `GO_MAP` + `GO_READOUT`.

---

## 5. Files to Produce

| File | Purpose | Phase |
|---|---|---|
| `micro_map_sanity_report.md` | Phase 0A end-to-end convention/pipeline sanity endpoints; emitted 0A token. | 0A |
| `micro_map_sanity.csv` | Phase 0A raw agreement / `A_independent` numbers for the micro-map. | 0A |
| `feasibility_report.md` | Trajectory access: extract `x_s`, resume integration from `x_s`, compute `x0(s)`; `s↔t` mapping audit. | 0A/0 |
| `trajectory_cache_manifest.md` | What `x_s` grid is cached, where, at what resolution. | 0A/0 |
| `preview_cache_manifest.md` | Cached decoded `x0(s)` previews (post-hoc readout depends on these). | 0A/0 |
| `internal_feature_cache_manifest.md` | Cached pooled DiT / latent / AV cross-attention summaries per saved `s`. | 0A/0 |
| `score_sde_validation_report.md` | velocity→score derivation under MMAudio's convention; **α=0 unit test** AND **nonzero-α checks** (small-α continuity, fork audio validity, nontrivial diversity). | 0 |
| `dataset_subset_manifest.md` | FoleyBench-based (or equivalent) subset: single-event and optional two-event sets, class balance, clip duration, event-timestamp source, **usable n per axis**, anchor uncertainty. | 0 |
| `event_anchor_validation_report.md` | Event-timestamp source, coverage, error/uncertainty, validation against a small check set. | 0 |
| `axis_reliability_report.md` | Per-axis **determinism + robustness + validity** + demotion decisions. | 0 |
| `commitment_map.csv` | `A_fork(x_s,axis,α)`, `A_independent(video,axis)`, normalized commitment gain, `s_commit` with CIs; full `A(axis,s,α)` surface; `VIDEO_DETERMINED` exclusions. | 1 |
| `readout_map.csv` | `acc(axis, probe, s, target)` for **ODE-target** and **fork-majority** targets; `s_read` with CIs. | 2 |
| `commitment_readout_gap_report.md` | `gap(axis, probe)`; R1/R2 cross-tab; axis-separation stats; **α-robust ordering / ordering-stability analysis**; decision-taxonomy assignment; threshold sensitivity. | 3 |
| `go_no_go_decision.md` | Pre-registered thresholds (incl. `θ_order`, `τ_video`); emitted token(s) + justification. | 3 |
| `internal_probe_report.md` | Linear probes on cached internal features; whether `s_read(internal)` approaches `s_commit`. | 3b |
| `causal_irreversibility_report.md` | Intervention flip-rate asymmetry around `s_commit`. | 3b |
| `coupling_ablation_report.md` | Commitment-map shift under conditioning-strength manipulations. | 3c |
| `checkpoint_commitment_sanity_report.md` | Commitment-only ordering sanity on a second MMAudio size variant / checkpoint. | 3d |
| `policy_preregistration.md` | Definitions of scalar-DiffRS rejection, **oracle axis-gated pruning**, SMC-ITA-style baseline, same-compute accounting (matched NFE + matched scoring calls), offline-sim vs online — required *before* Phase 4 runs. | pre-4 |
| `policy_pruning_report.md` | Map-scheduled population search vs baselines; two-axis compute accounting. | 4 (cond.) |
| `process_verifier_report.md` | Cheap process-aware verifier: training, cost, readout-window match. | 5 (cond.) |
| `rollback_report.md` | Axis-gated rollback variant. | 6 (cond.) |

---

## 6. Phased Flow

```text
Phase 0A Video-conditioned micro-map sanity test (pipeline sanity ONLY; not evidence).
         Gate: end-to-end measurement chain works; conventions audited; cache hooks installed.

Phase 0  Feasibility + reliability gate (STRICT; micro-map does NOT replace it).
         Gate: trajectory access; SDE validated at alpha=0 AND nonzero alpha;
               dataset+anchor manifest; >=3 axes pass determinism+robustness+validity;
               cache hooks (x_s grid, x0(s) previews, internal features) verified.

Phase 1  Commitment map (C1).    -> s_commit(axis), normalized vs A_independent,
                                    with high-A_independent cap / VIDEO_DETERMINED rule
Phase 2  Readout map (C2).       -> s_read(axis, probe) for ODE-target & fork-majority
Phase 3  Gap + separation + alpha-robust ordering + GO/NO-GO. -> R1/R2, taxonomy, token
         === MAKE-OR-BREAK ENDS HERE. Everything below is conditional. ===

Phase 3b Internal readout analysis + causal commitment intervention (P1).
Phase 3c Coupling-strength ablation (P1, non-blocking).
Phase 3d Second-checkpoint commitment-only sanity (P1 robustness).

Phase 4  Map-scheduled population search (needs correctness sidecar + pre-registration;
         baselines include SMC-ITA-style search under matched budgets).
Phase 5  Cheap process-aware verifier (only if maps show headroom).
Phase 6  Axis-gated rollback (only if forking is stable).
Phase 7  = Phase 3b internal readout (kept as alias; logging happens in Phases 1-2).
```

---

## 7. Phase Details

### Phase 0A — Video-Conditioned Micro-Map Sanity Test (DO THIS BEFORE THE FULL PHASE 0 BUILD)

**Purpose:** exercise the full measurement chain end-to-end before expensive FoleyBench curation and full reliability gating. This is a convention and pipeline sanity test, **not** a replacement for the strict Phase 0 gate.

**Preconditions:** minimal trajectory access (extract `x_s`, resume, compute `x0(s)`) and provisional nonzero-α fork sanity must exist first. Phase 0A must not bypass or weaken the full Phase 0 reliability gate.

**Inputs:**
- 12–16 available VGGSound or equivalent clips; do not block on FoleyBench.
- coarse-class axis only.
- off-the-shelf audio tagger.
- `s ∈ {0.25, 0.5, 0.75}`.
- `K = 8` forks per `x_s`.
- `N = 8` independent full generations per video for `A_independent`.
- one provisional α from a quick valid-diversity probe.

**Outputs:**
- `micro_map_sanity_report.md`
- `micro_map_sanity.csv`
- cached audio previews and generated completions sufficient to debug decode/tagger/agreement/normalization
- cache manifests (`trajectory_cache_manifest.md`, `preview_cache_manifest.md`, `internal_feature_cache_manifest.md`) initialized.

**Required sanity endpoints:**
- fork agreement should approach ~1 as `s → 1`;
- fork agreement near `s → 0` should approximately match `A_independent`;
- no denominator explosion from `A_independent ≈ 1` (high-`A_independent` cases marked `VIDEO_DETERMINED`, not normalized);
- tagger outputs and agreement metrics are non-degenerate;
- failures route to `FIX_PIPELINE_CONVENTION`, `FIX_SCORE_CONVERSION`, or `FIX_MEASUREMENT_CHAIN`, **not** to paper-level conclusions.

**Important: Phase 0A is not publishable evidence and does not license `GO_MAP`.**

**Tokens:** `GO_FULL_PHASE0` · `FIX_PIPELINE_CONVENTION` · `FIX_MEASUREMENT_CHAIN` · `FIX_SCORE_CONVERSION` · `NO_TRAJECTORY_ACCESS`.

---

### Phase 0 — Feasibility + Reliability Gate (STRICT; the micro-map does NOT replace this)

**0.1 Trajectory access + mandatory instrumentation.** Hook MMAudio's sampler to (a) record `x_s`, (b) resume integration from a given `x_s` to `s = 1`, (c) compute `x0(s)`, and (d) **optionally dump pooled hidden states / DiT block features / AV cross-attention summaries per saved `s`** (internal-feature hooks). Audit and record the `s↔t` mapping. Run a tiny smoke generation. → `feasibility_report.md`, cache manifests.

> **Internal probes are not required to pass Phase 0 or `GO_MAP`, but internal feature logging is required before any large Phase 1/2 generation.** If features are not cached, R2 cannot be cleanly tested without rerunning the expensive generation. Therefore **log now, analyze later** (Phase 3b).

**0.2 Velocity→score SDE validation (highest silent-bug risk).** Derive the closed-form score from MMAudio's **actual** velocity/time parameterization. A common rectified-flow form is `score(x,t) ≈ (t·v − x)/(1−t)` (equivalently `∇log p_t = −(x_t + (1−t)v)/t`) — **audit sign and direction against code; do not copy blindly.**
Validation (all required, → `score_sde_validation_report.md`):
- **α=0 unit test (necessary, not sufficient):** integrating the SDE at `α=0` must reproduce the deterministic ODE completion. *At α=0 the score term is multiplied by 0, so this does NOT test the score conversion.*
- **Nonzero-α checks (these actually test the score term):**
  - *small-α continuity:* as `α → 0⁺`, fork outputs converge continuously to the ODE output (no jumps → score term is consistent);
  - *fork audio validity:* forks at the test α are valid Foley audio (presence/quality measure passes);
  - *nontrivial diversity:* at the test α, forks show measurable diversity (else α is too small to probe commitment).
- Emit `FIX_SCORE_CONVERSION` and halt if α=0 fails or small-α continuity is violated.

**0.3 Dataset subset manifest.** Do **not** rebuild a Foley benchmark. Use FoleyBench (or equivalent). Specify: single-event subset, optional clean two-event subset (for binding), class balance, clip duration, event-timestamp source, **usable n per axis**, and anchor uncertainty. → `dataset_subset_manifest.md`.

**0.4 Event-anchor protocol + validation.** Obtain visible-event timestamps. Priority: FoleyBench metadata → off-the-shelf visual onset/event detector → light human marks. Report coverage and error/uncertainty, validated on a small check set. Timing and binding are uninterpretable without reliable anchors. → `event_anchor_validation_report.md`.

**0.5 Per-axis reliability gate (three parts; → `axis_reliability_report.md`).** On **generated** audio, an axis passes only if:
- **Determinism:** repeated measurement on identical audio is stable (`≥ θ_rel`). Most taggers are deterministic → fork-disagreement is then genuine generator variance.
- **Robustness:** the measurement survives small nuisance perturbations — event-window shift, loudness normalization, resampling, light compression, small added noise (`≥ θ_robust`).
- **Validity:** agreement with a small human/frontier-MLLM calibration subset that the measurement is perceptually meaningful Foley correctness (`≥ θ_cal`). *A measurement can be deterministic yet invalid.*
- **Demotion rule:** fails any → demote (smaller-scale MLLM/human labeling) or drop; emit `AXIS_DEMOTED:<axis>`. **Material / fine class is demoted unless reliability is strong on all three.**

**Axis tiers for v1:**
- **Tier 1 (run):** event-sound presence; gross timing; coarse event class.
- **Tier 2 (run iff reliability strong):** material / fine class.
- **Tier 3 (stretch, clean 2-event clips only):** multi-event binding.
- **Separate analysis:** offscreen hallucination → seed-predictability (predict from initial noise / very early `x_s`, **not** a window).
- **Excluded:** delayed callback.

**Gate / tokens:** `GO_MAPS_PHASE` (access OK, SDE validated at α=0 *and* nonzero-α, manifest+anchors ready, ≥3 axes pass all three reliability parts, cache hooks verified) · `FIX_SCORE_CONVERSION` · `NO_TRAJECTORY_ACCESS` → `STOP_PROJECT` · `STOP_PROJECT` (<3 reliable axes, or no usable anchors).

---

### Phase 1 — Commitment Map (C1)

**Goal:** per-axis `s_commit`, normalized against the video-conditioned prior. Human-label-free, but measurement-dependent.

**Caching requirement (hard):** Phase 1/2 generation must cache the full `x_s` grid, decoded `x0(s)` previews, final completions, fork completions, and pooled internal-feature summaries (per `internal_feature_cache_manifest.md`). Readout probes (Phase 2) and internal probes (Phase 3b) are **post-hoc** on these caches — never design a probe that requires regenerating trajectories.

**Fork kernel (marginal-preserving SDE; α is the only knob):**
```text
fork_tail(x_s, s, alpha, K):           # integrate progress s -> 1 (final audio)
  comps = []
  for k in 1..K:
    x = x_s
    for (s_i -> s_next) in progress_schedule(s -> 1):
      t_i   = s_to_t(s_i)                          # audited mapping (Phase 0.1)
      v     = v_theta(x, t_i, video_cond)
      score = score_from_velocity(v, x, t_i)       # VALIDATED in Phase 0.2
      sigma = alpha * g(s_i)                         # alpha=0 => deterministic ODE
      x     = step_euler_maruyama(x, v, score, sigma, ds)
    comps.append(decode(x))                          # final audio
  return comps
```
**Reserve Restart re-noising for rollback (Phase 6); do NOT use it here.**

**α handling (not a hidden knob — point of failure if abused):**
- Predefine an **α pilot grid**.
- **Selection rule:** primary operating α = the **smallest** α that produces measurable tail diversity while preserving valid generated audio (the audio-validity guard = presence/quality measure on forks).
- Report the **full `A(axis, s, α)` surface** as secondary evidence; use the single primary α for headline `s_commit`.
- **α-robust ordering is part of the headline claim** (see Phase 3): absolute `s_commit` is kernel-relative; the robust object is the cross-axis ordering.

**Video-prior normalization (critical — avoids confusing video-prior agreement with trajectory commitment):**
- Compute `A_independent(video, axis)` = agreement of the self-target across **N independent full generations** of the same video (independent initial noise). High when the video tightly implies the sound.
- Compute `A_fork(x_s, axis, α)` = agreement across K stochastic tail-forks from `x_s`.
- **Normalized commitment gain:**
  `commit(s, axis) = clip( (A_fork(x_s,axis,α) − A_independent(video,axis)) / (1 − A_independent(video,axis)), 0, 1 )`
  = fraction of the *reducible* (video-conditioned) diversity that `x_s` has resolved. At `s=0`, `A_fork ≈ A_independent ⇒ commit ≈ 0`; at `s=1`, `commit = 1`.
- `s_commit(axis) = min s with commit(s,axis) ≥ θ_commit`, bootstrapped over videos.

**High-`A_independent` cap / video-determined rule.**
If `A_independent(video, axis) ≥ τ_video` (default pilot value: 0.90; sensitivity reported), the per-video denominator is too small for stable normalized gain. Do not compute per-video normalized commitment for that pair. Instead:
1. mark `(video, axis)` as `VIDEO_DETERMINED`;
2. exclude it from the reducible-diversity commitment curve for that axis;
3. report the excluded fraction as a result;
4. run sensitivity over `τ_video`;
5. never let denominator blow-up create artificial early commitment.

Axis-level aggregation before normalization is allowed only if explicitly documented and compared.

**Agreement metric per axis:** categorical (presence / timing-bin / class / binding) → exact-match rate or Krippendorff's α across forks; embedding (material) → mean pairwise cosine.

**Tokens:** `COMMITMENT_MAP_DONE` · `FORK_ALPHA_NO_VALID_OPERATING_POINT` — small α gives no diversity AND large α destroys audio (no usable operating point). This does **not** auto-kill the project; route to a different commitment kernel or to `GO_DIAGNOSTIC`.

---

### Phase 2 — Readout Map (C2)

**Goal:** per-axis, per-probe `s_read`, for **two targets** (report both; interpret only alongside commitment status):
- **ODE-target readout:** predict the self-target of the deterministic (`α=0`) completion of this `x_s` — the original path this candidate would realize.
- **Fork-majority readout:** predict the majority/typical self-target across the Phase-1 forks from `x_s` — the typical future under stochastic completion.

> **Interpretation rule:** predicting an *uncommitted* deterministic path (low `commit(s,axis)`) is **not** the same as reading a decided axis — the probe is reading one path among many, not a commitment. Only readout where `commit(s,axis)` is high licenses early action.

All probes run **post-hoc on the Phase-1/2 caches** (decoded `x0(s)` previews per `preview_cache_manifest.md`); no regeneration.

**Probe ladder (apply to `x0(s)` / partial preview / cached internal features):**
1. energy / onset heuristics;
2. CLAP / SyncNet / ImageBind — **legacy** baselines, not the headline probe;
3. audio tagger on `x0(s)`;
4. *(optional in pilot)* cheap learned process verifier — defer to Phase 5 if it needs training;
5. frontier MLLM-on-`x0(s)`;
6. **(Phase 3b; analysis non-blocking, logging mandatory)** internal-feature probe (pooled DiT / latent / AV cross-attention).

`s_read(axis, probe, target) = min s with accuracy/AUROC ≥ θ_read`. Bootstrap over videos.

**Blur note:** commitment uses **clean** final completions; readout uses **blurry** `x0(s)`. The early gap is partly this blur penalty — expected, and disentangled by R1/R2.

**Tokens:** `READOUT_MAP_DONE`.

---

### Phase 3 — Gap, Separation, α-Robust Ordering, GO/NO-GO (make-or-break ends here)

**Compute per axis / probe:**
- `gap(axis, probe) = s_read(axis, probe) − s_commit(axis)`.
- **R1/R2 cross-tab** per (axis, s): `committed? = (commit(s,axis) ≥ θ_commit)` × `readable_by_cheap_probe?`:
  - `~committed` → **R1** (defer genuinely necessary);
  - `committed & ~readable(cheap)` → **R2** (probe-limited; gap reducible — flag for Phase 3b internal probes);
  - `committed & readable` → early-action candidate.
- **Separation test:** bootstrap CIs (over videos) on `s_commit` per axis; separation = ordered, non-overlapping CIs (or a rank-order test). `separation_score = spread(s_commit across axes) / mean(within-axis CI width)`.
- **α-robust ordering (required for `GO_MAP`):** `GO_MAP` requires not only separation at the primary α, but also **stable axis ordering across the α pilot grid**. Absolute `s_commit` values are kernel-relative; the robust claim is the rank/order and R1/R2 gap structure. Operationalization:
  - For each valid α, compute the rank order of `s_commit(axis)`.
  - Report Kendall τ (or pairwise order agreement) against the primary-α ordering → `ordering_stability_score`.
  - `GO_MAP` requires ordering stability above the pre-registered threshold `θ_order` (frozen during the pilot, before headline curves — see §3) and **no major reversal among Tier-1 axes**.
  - Report disagreements as sensitivity results, not hidden failures. → ordering-stability analysis reported inside `commitment_readout_gap_report.md`.

**Decision taxonomy (assign every axis/video or axis-level result):**
- `VIDEO_DETERMINED`: `A_independent` is near 1; the video alone pins the value.
- `SEED_DETERMINED`: value or failure predictable from initial noise / very early state, before meaningful trajectory commitment.
- `TRAJECTORY_EARLY`: reducible diversity resolves early.
- `TRAJECTORY_LATE`: reducible diversity resolves late.
- `COMMITTED_BUT_UNREADABLE`: commitment exists but available external probes lag; R2.
- `UNRELIABLE_MEASUREMENT`: axis failed determinism/robustness/validity.

Use this taxonomy in `commitment_readout_gap_report.md`; it is also the proposal's planned Figure 1.

- **Threshold sensitivity:** re-report separation under the θ sweep from Section 3.

**Decision tokens (emit all that apply — a good map is NOT killed by a failed first policy):**
- `GO_MAP` — axes show separated commitment windows beyond CIs **with α-robust ordering** (`ordering_stability_score ≥ θ_order`, no Tier-1 reversal).
- `GO_READOUT` — at least one feasible probe reads the early axes well before the end.
- `GO_POLICY` — *(requires Phase 4)* early-action precision / false-prune / winner-retention / regret support pruning or rollback.
- `GO_RESTRICTED` — only presence / gross timing show early actionable windows → restricted policy.
- `GO_DIAGNOSTIC` — commitment exists but cheap readout lags far behind (R2-dominated) → publish gap + probe-limitation; motivate internals (Phase 3b).
- `STOP_ADSR` — all `s_commit` coincide or only near `s=1` → degenerates to scalar DiffRS; **publishable NEGATIVE result**, route to diagnostic framing.
- `STOP_PROJECT` — reliability/feasibility failure.

**`GO_MAP` + `GO_READOUT` is the scientific make-or-break and needs NO correctness labels.** Only proceed to Phase 4 if both fire.

---

### Phase 3b — Internal Readout Analysis + Causal Commitment Intervention (conditional analysis; P1)

Logging happened in Phases 1–2 (mandatory); the analyses below are conditional but high priority. Small, method-flavored, and useful even if full policy is weak.

**3b.1 Internal-feature readout analysis** *(formerly "Phase 7"; analysis conditional, logging mandatory)*
Train/evaluate simple **linear probes** on the cached pooled DiT / latent / AV cross-attention features to test whether `s_read(internal)` approaches `s_commit` and precedes `x0(s)`-probe readout.
- A strong internal probe pulling `s_read` toward `s_commit` → **R2 confirmed** (gap is probe-limited / reducible) → `GO_INTERNAL_READOUT`, `R2_CONFIRMED`.
- If even internals cannot beat `s_commit` → **R1** (irreducible; defer genuinely necessary).
- Until internals run, report **"gap under available external probes"**, never "irreducible uncommitted information".
- Costs are not free: zero extra forward passes, but real storage and probe-fitting costs (report them).
→ `internal_probe_report.md`

**3b.2 Causal commitment intervention**
Purpose: test whether observed fork-agreement commitment corresponds to **causal irreversibility**. This **validates** the Phase-1 fork-agreement definition of commitment; it never replaces it as the operating definition.
Design:
- choose one reliable axis, preferably coarse class or timing;
- choose ~50 clips;
- choose three intervention points: before measured `s_commit`, near boundary, after measured `s_commit`;
- apply an intervention that pushes `x0(s)` / guidance toward an alternative axis value;
- measure flip rate of the final self-target.
Pre-registered prediction: flip rate should be higher before `s_commit` and collapse after the boundary.
→ `causal_irreversibility_report.md`

**Tokens:** `GO_INTERNAL_READOUT` · `R2_CONFIRMED` · `GO_CAUSAL_COMMITMENT` (intervention asymmetry supports the boundary) · `CAUSAL_INCONCLUSIVE`.

---

### Phase 3c — Coupling-Strength Ablation (non-blocking; P1 after micro-map + primary map)

**Purpose:** make "cross-modal" a **variable** rather than a fixed setting.

**Candidate manipulations:**
- sweep CFG scale / conditioning strength;
- drop or weaken Synchformer sync features if MMAudio exposes them;
- compare normal video conditioning vs degraded / weaker conditioning.

**Pre-registered prediction:** stronger cross-modal conditioning should pull relevant `s_commit` earlier and may reduce reducible diversity for video-determined axes. (Motivated by — not proven by — coupled multimodal speciation theory; see proposal §4.)

**Output:** `coupling_ablation_report.md`. Not a blocker for initial feasibility.

---

### Phase 3d — Second-Checkpoint Commitment-Only Sanity (P1 robustness; not a Phase-0 blocker)

MMAudio remains the v1 white-box backbone. After the primary commitment map is working, run a **commitment-only** sanity check on a second MMAudio size variant / checkpoint if available. Do not run readout or policy on the second checkpoint unless cheap. This tests whether the **ordering** is stable across scale/checkpoints — the predictable reviewer concern is single-checkpoint specificity.

**Output:** `checkpoint_commitment_sanity_report.md`. **Token:** `GO_CHECKPOINT_SANITY`. Recommended before a serious conference submission.

---

### Phase 4 — Map-Scheduled Population Search (conditional; NOT YET RUNNABLE)

**Positioning (PI review):** the map is a **scheduler for population search**, not a competitor trying to beat SMC-ITA by brute force. Do not present cascaded BoN as fighting SMC-ITA. Instantiate at least:
1. cascaded BoN / axis-gated pruning;
2. axis-aware resampling schedule inside SMC-style search, if feasible.

**Pre-flight (required before running; → `policy_preregistration.md`):** define
- **scalar DiffRS-style rejection** (single-discriminator reject+rollback) precisely;
- **oracle axis-gated pruning** (upper bound: prune using true self-targets / correctness);
- **SMC-ITA-style intermediate-reward baseline** (or document precisely why it is not implementable and cite as nearest neighbor);
- **same-compute accounting** (matched NFE and matched scoring-call budgets across methods);
- whether pruning is **offline-simulated** (score a fixed candidate pool, replay decisions) or **online** (decisions alter generation).

**Correctness factorization (consistent with proposal §7):** correctness enters through a sidecar only after a value is readable. For factorable axes, correctness is `match(axis value, video/event anchor)`. This matching relation is time-independent; time-dependence lives in readout fidelity. Holistic quality axes that cannot be decomposed into readable values are out of scope for axis-gated correctness pruning.

Only after this, and after `GO_MAP` + `GO_READOUT`, scale the correctness-vs-video sidecar (human/MLLM) — pruning decisions must be judged against actual wrongness.

**Method:**
```text
generate N candidates to the first actionable window
prune only on axes whose windows have closed AND whose early-action precision is high
continue survivors
at later windows, evaluate later axes; prune/rerank
finish remaining candidates; final rerank
(optional) axis-aware resampling schedule inside SMC-style search
```
Operating progress per axis set by early-action precision / false-prune / winner-retention / regret — **not** raw correlation.

**Baselines (same candidate pools):** full BoN; same-compute BoN; random pruning; **scalar DiffRS-style rejection**; final-score reranking; seed restart; **oracle axis-gated pruning** (upper bound); **SMC-ITA or SMC-ITA-style intermediate-reward baseline**; **multi-verifier inference-time scaling baseline** if implementable, else cite as nearest neighbor.

> **Do not claim superiority over SMC-ITA unless matched NFE, matched scoring-call budget, and same candidate pool / comparable pool accounting are satisfied.**

**Metrics:** final + per-axis Foley correctness; total completed candidates; total NFE; **scoring-call budget**; winner retention; false-prune rate; regret; compute–quality Pareto.

**Tokens:** `GO_POLICY` / `GO_RESTRICTED` / `DIAGNOSTIC_ONLY`.

---

### Phase 5 — Cheap Process-Aware Verifier (conditional)

Train a small axis-conditioned verifier to approximate the **same-tier readout signal** at low cost (NOT to beat MLLM-on-final). Re-run Phase 4 with it as the per-step scorer; report scoring-call savings vs MLLM-on-`x0(s)`. **Train only after maps show headroom.**

### Phase 6 — Axis-Gated Rollback (conditional, aggressive)

Use Restart re-noising (or the SDE kernel) to re-noise to an axis-appropriate earlier progress and regenerate the tail after an in-window axis failure. Baselines: Restart without axis gating; scalar rejection + rollback; full seed restart. Run only if forking is stable.

### Phase 7 — Internal-Feature Probes

Merged into **Phase 3b.1** (analysis) with **mandatory logging during Phases 1–2**. Kept here as an alias so older references resolve.

---

## 8. Hard Constraints

- **Maps target the model's own final axis value, NOT correctness-vs-video.** Correctness enters only in Phase 4+, factored as `match(readable axis value, video/event anchor)` (proposal §7).
- **Maps are correctness-/human-label-free but measurement-dependent** — gate measurement reliability (determinism + robustness + validity) before any map.
- **Commitment kernel = marginal-preserving SDE; validate at α=0 AND nonzero-α** (Phase 0.2). Restart re-noising is reserved for Phase 6.
- **Normalize commitment against `A_independent`** (video-conditioned prior) — never report raw fork agreement as commitment; apply the **high-`A_independent` cap (`τ_video`) / `VIDEO_DETERMINED` exclusion**; never let denominator blow-up create artificial early commitment.
- **Report all windows in generation progress `s`** (s=0 noise, s=1 audio); map to MMAudio `t` only in audited code.
- **Do not design readout probes that require regenerating trajectories.** Phase 1/2 must cache the full `x_s` grid, decoded `x0(s)` previews, final completions, fork completions, and internal feature summaries needed for post-hoc readout.
- **Internal-feature logging is mandatory before any large Phase 1/2 generation**; internal-probe *analysis* is non-blocking for `GO_MAP`.
- **Report α-robust ordering, not absolute `s_commit`, as the headline claim form**; `GO_MAP` requires ordering stability (§7 Phase 3).
- **Pre-register thresholds** (incl. `θ_order`, `τ_video`); bootstrap over videos; declare minimum usable n per axis.
- **`GO_MAP` + `GO_READOUT` is the make-or-break;** do NOT gate the pilot on policy (Phase 4) or internal-probe analysis (Phase 3b).
- **MMAudio remains the v1 white-box backbone.** After the primary commitment map is working, run a commitment-only sanity check on a second MMAudio size variant / checkpoint if available (Phase 3d). Do not run readout or policy on the second checkpoint unless cheap.
- Do **not** train the verifier before maps show headroom. Do **not** fine-tune the generator. Do **not** add delayed callback / memory architectures. Do **not** cite music-ADSR as proof for V2A.
- CLAP / SyncNet / ImageBind are **legacy** baselines, not the headline probe.
- Do **not** claim superiority over SMC-ITA without matched NFE + matched scoring-call budgets + comparable pool accounting.
- **Novelty boundary:** if axes do not separate, route to DIAGNOSTIC / NEGATIVE — do not force a method claim. Prefer early stop. Do not infer missing results. Do not upgrade claim tier without gate evidence.

---

## 9. Baseline Requirements

All baselines score the **same** candidate pools.

**Readout-side (probe ladder):** energy/onset; CLAP; SyncNet/AV-sync; ImageBind; audio tagger; MLLM-on-preview; (Phase 3b) internal-feature probe.

**Policy-side (Phase 4):** full BoN; same-compute BoN; random pruning; scalar DiffRS-style rejection; final-score reranking; seed restart; **oracle axis-gated pruning (upper bound)**; **SMC-ITA / SMC-ITA-style intermediate-reward baseline**; **multi-verifier inference-time scaling baseline** (if implementable; else cite as nearest neighbor with reason).

**Rollback-side (Phase 6):** Restart without axis gating; scalar rejection + rollback; full seed restart.

---

## 10. Launch Commands

**Phase 0A (instrumented micro-map sanity) — first command:**
```text
/run-experiment "foley-cw Phase 0A (instrumented micro-map sanity): first confirm video-conditioned MMAudio end-to-end trajectory access on one real video, including extract x_s, resume from x_s, compute x0(s), audited s<->t mapping, and provisional nonzero-alpha fork sanity. Add mandatory logging hooks for trajectory cache, decoded x0(s) previews, final/fork completions, and pooled internal features / AV attention summaries. Then run a micro-map sanity check on 12-16 available VGGSound or equivalent clips, coarse-class axis only, s={0.25,0.5,0.75}, K=8 forks, N=8 independent generations for A_independent, one provisional smallest-valid alpha. Check that fork agreement approaches ~1 near s=1, matches A_independent near s=0, tagger/agreement/normalization are non-degenerate, and high-A_independent cases are marked VIDEO_DETERMINED rather than normalized with unstable denominators. This is pipeline sanity only, not publishable evidence and not GO_MAP. Output feasibility_report.md, trajectory_cache_manifest.md, preview_cache_manifest.md, internal_feature_cache_manifest.md, micro_map_sanity_report.md, micro_map_sanity.csv. Emit GO_FULL_PHASE0 / FIX_PIPELINE_CONVENTION / FIX_MEASUREMENT_CHAIN / FIX_SCORE_CONVERSION / NO_TRAJECTORY_ACCESS." — priority: P0
```

**Full Phase 0 (after `GO_FULL_PHASE0`; the micro-map does NOT replace this gate):**
```text
/run-experiment "foley-cw Phase 0 (STRICT; micro-map sanity does not replace this): confirm MMAudio trajectory access (extract x_s, resume from x_s, compute x0(s)) and audit the s<->t mapping; verify mandatory cache hooks (x_s grid, x0(s) previews, final/fork completions, internal features); derive and VALIDATE the velocity->score SDE (alpha=0 reproduces ODE AND nonzero-alpha checks: small-alpha continuity, fork audio validity, nontrivial diversity); build dataset_subset_manifest from FoleyBench (single-event + optional 2-event, class balance, duration, anchor source, usable n per axis); build+validate event-anchor protocol; run the three-part reliability gate (determinism + robustness + validity) on Tier-1 axes plus Tier-2 if strong. Do NOT build maps, train a verifier, or use correctness-vs-video. Output feasibility_report.md, score_sde_validation_report.md, dataset_subset_manifest.md, event_anchor_validation_report.md, axis_reliability_report.md. Emit GO_MAPS_PHASE / FIX_SCORE_CONVERSION / NO_TRAJECTORY_ACCESS / STOP_PROJECT." — priority: P0
```

**Phases 1–3 (after `GO_MAPS_PHASE`):**
```text
/run-experiment "foley-cw Phases 1-3 (make-or-break, no correctness labels): COMMITMENT map via marginal-preserving stochastic tail-forks (predefined alpha grid, smallest-valid-alpha rule + audio-validity guard, full A(axis,s,alpha) surface), normalized as commitment gain over A_independent(video,axis) per the video-prior baseline, WITH the high-A_independent cap: A_independent >= tau_video (pilot 0.90, sensitivity reported) is marked VIDEO_DETERMINED and excluded from normalized curves, excluded fraction reported. Cache the full x_s grid, decoded x0(s) previews, final/fork completions, and pooled internal features for post-hoc probes. READOUT map via the probe ladder on cached x0(s) for BOTH ODE-target and fork-majority targets. Report s_commit, s_read, gap, R1/R2 cross-tab, the full decision taxonomy (VIDEO_DETERMINED / SEED_DETERMINED / TRAJECTORY_EARLY / TRAJECTORY_LATE / COMMITTED_BUT_UNREADABLE / UNRELIABLE_MEASUREMENT), axis-separation with bootstrap-over-videos CIs, AND alpha-robust ordering stability (Kendall tau or pairwise order agreement vs the primary-alpha ordering, against the pre-registered theta_order, no Tier-1 reversal). Pre-registered thresholds + sensitivity. No correctness labels. Output commitment_map.csv, readout_map.csv, commitment_readout_gap_report.md, go_no_go_decision.md. Emit GO_MAP / GO_READOUT / GO_RESTRICTED / GO_DIAGNOSTIC / STOP_ADSR / STOP_PROJECT / FORK_ALPHA_NO_VALID_OPERATING_POINT." — priority: P0
```

**Phase 3b (if maps show promise):**
```text
/run-experiment "foley-cw Phase 3b (internal readout + causal commitment validation): using cached Phase 1/2 trajectories, train/evaluate simple linear probes on pooled internal features / AV attention summaries to test whether s_read(internal) approaches s_commit and beats x0(s)-probe readout. Then run a small causal intervention on one reliable axis around measured s_commit: before, near, and after boundary; measure final self-target flip rate. Output internal_probe_report.md and causal_irreversibility_report.md. Emit GO_INTERNAL_READOUT / R2_CONFIRMED / GO_CAUSAL_COMMITMENT / CAUSAL_INCONCLUSIVE." — priority: P1
```

**Phase 4 (after `GO_MAP` + `GO_READOUT` and `policy_preregistration.md`):**
```text
/run-experiment "foley-cw Phase 4 (conditional map-scheduled search): after GO_MAP + GO_READOUT and policy_preregistration.md, instantiate the map as a scheduler for population search, including cascaded BoN pruning and, if feasible, axis-aware SMC-style resampling. Compare against full BoN, same-compute BoN, random pruning, scalar DiffRS-style rejection, final-score reranking, seed restart, oracle axis-gated pruning, and SMC-ITA / SMC-ITA-style intermediate reward baseline under matched NFE and matched scoring-call budgets. Correctness-vs-video labels are used only in the sidecar for policy evaluation (correctness = match(readable axis value, video/event anchor)). Output policy_preregistration.md and policy_pruning_report.md. Emit GO_POLICY / GO_RESTRICTED / DIAGNOSTIC_ONLY." — priority: P1
```

---

## 11. Downstream Agent Instructions

Start with **Phase 0A** if no experiments have run; the micro-map never replaces the full Phase 0 gate. Do not infer missing results. Do not upgrade the claim tier without gate evidence. Do not add method components before the maps exist. Prefer early stop.

**The first agent response (after Phase 0A) must contain:**
1. one real-video trajectory-access result (extract/resume `x_s`, compute `x0(s)`);
2. the audited `s↔t` mapping;
3. provisional-α fork sanity;
4. cache hooks installed, and exactly what is cached (trajectory / previews / completions / internal features);
5. micro-map sanity endpoints (agreement → 1 near `s=1`; ≈ `A_independent` near `s=0`; non-degenerate tagger/agreement);
6. denominator / `VIDEO_DETERMINED` behavior;
7. the emitted token: `GO_FULL_PHASE0` or a fix token.

**Full Phase 0 then still requires:**
1. trajectory-access result and audited `s↔t` mapping;
2. the **α=0 unit-test AND nonzero-α validation** results for the velocity→score SDE (pass/fail each);
3. the dataset subset + event-anchor plan with estimated coverage and usable n per axis;
4. per-axis reliability (determinism + robustness + validity) and which axes survive;
5. whether Phase 1 can start, and the emitted Phase-0 token.

---

## 12. Implementation Watch-List (these bite first)

1. **Reliability gate is step 0**, before any map. Three parts: **determinism** (test-retest), **robustness** (survive event-window shift / loudness-norm / resample / compression / small noise), **validity** (vs human on the sidecar). A deterministic-but-invalid (or fragile) measure poisons both maps. Material/fine-class is the prime demotion risk.
2. **The pilot's make-or-break is `GO_MAP` + `GO_READOUT`, both correctness-free.** Do not compute early-action precision / regret (Phase 4) in the first pass.
3. **The most likely silent bug is the velocity→score conversion.** `α=0` reproducing the ODE is necessary but **does not test the score term** (it's multiplied by 0). The real test is nonzero-α: small-α continuity + fork validity + nontrivial diversity.
4. **`A_independent` normalization is mandatory.** High raw fork agreement can be pure video prior, not commitment. Always report commitment as gain over the video-conditioned independent-sample baseline.
5. **Do not normalize per-video commitment when `A_independent` is too close to 1** (`≥ τ_video`): mark `VIDEO_DETERMINED`, exclude, report the excluded fraction, run `τ_video` sensitivity. Denominator blow-up must never masquerade as early commitment.
6. **Use `s` (progress), never raw `t`.** MMAudio's internal time direction is a silent-bug source; map `s↔t` once, in audited code.
7. **Pick the operating α by the smallest-valid-α rule** with the audio-validity guard; if no α works, emit `FORK_ALPHA_NO_VALID_OPERATING_POINT` and consider a different kernel or a diagnostic outcome — do not silently push α up and call the broken audio "uncommitted."
8. **Do not report absolute `s_commit` as universal; report α-robust ordering.** Absolute values are kernel-relative; the claim is the ordering + gap structure.
9. **Do not forget internal-feature logging before large generation.** Features cannot be retrofitted without rerunning; log now, analyze later (Phase 3b).
10. **The micro-map (Phase 0A) is pipeline sanity, not evidence.** It never licenses `GO_MAP` and never replaces the full Phase 0 gate.
11. **Freeze thresholds (incl. `θ_order`, `τ_video`) before inspecting the headline curves;** report sensitivity after. Bootstrap over videos.
12. **Event anchors are an owned dependency** — timing and binding are uninterpretable without them; report anchor uncertainty.
13. **Internal-probe analysis is non-blocking** for the pilot. Until internals run, report "gap under available external probes," never "irreducible uncommitted information."
14. **Phase 4 must include SMC-ITA (or SMC-ITA-style) as a baseline, or explicitly mark it unimplemented with the reason.** No superiority claims without matched NFE + scoring-call budgets.
15. **Phase 4 correctness relies on axis-value matching to anchors; holistic quality is out of scope** unless it factors through a readable axis value.
16. **Second-checkpoint commitment-only sanity (Phase 3d) is recommended before a serious conference submission.**
17. **Novelty boundary holds throughout:** not a final-audio verifier, not a DiffRS clone. No separation ⇒ diagnostic / negative, never a forced method claim.
