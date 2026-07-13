# Long-Range Experiment Plan — `foley-cw` / Sound Decisions (v2)

**Project:** *When Are Foley Decisions Made? Commitment and Readout Windows in V2A Flow Generation.*
**Backbone:** MMAudio (rectified flow, `small_16k` primary; `large_44k` for scale insurance only).
**Scope of this document:** everything from the current checkpoint (Phase-0 crux validated on the real model; no `GO_MAPS_PHASE` yet) through paper assembly. It **amends and extends** `EXPERIMENT_PLAN.md`; where the two conflict, this document wins. `FINAL_PROPOSAL_SHORT.md` remains the conceptual frame.
**No calendar dates by design.** All sequencing is expressed as stages, gates, and dependencies. Stages begin when their preconditions hold, not on a date.
**Audience:** autonomous research agents and collaborators. Decision tokens are the only legitimate way to change claim tier or unlock a stage.

---

> **[EXECUTION STATUS 2026-06-11 — see Appendix E for the full record]**
> Stage M EXECUTED → `MICROMAP_FAIL(endpoints, kernel_cfg4.5)` + `GATE_A_UNDERPOWERED`
> → **HALTED awaiting PI review** (`results/stage_m/pi_checkpoint.md`, 6 decisions).
> Done: env+weights, thresholds/pre-registrations frozen, FoleyBench extraction
> (785), measurers + tagger gate, logging contract, Gate-A machinery, qwen judge,
> anchors (785, human 30-clip pending), α pilots (`PRIMARY_ALPHA=1.6` @ cfg=1.0;
> no valid point @ cfg=4.5), full Stage M, 4 Codex review rounds.
> Gated (not run): Stage-0 screening, MLLM sidecar at scale, reliability gate,
> manifest freeze, Phases 1–3+.

## 0. Current Position and What Changed

**Established (real-model evidence, citable as feasibility only):**
- Trajectory access on real MMAudio `small_16k`: extract `x_s`, resume integration, compute `x̂0(s)`; `s↔t` mapping audited (identity; `v = x1 − x0`, `t: 0→1`).
- Velocity→score SDE validated: `α = 0` reproduces the ODE exactly; small-α continuity, fork validity, and nontrivial diversity pass — both unconditionally and **under video conditioning** at `cfg = 1.0`.
- Offline environment, weights, CLIP/synchformer caches, and `/dev/shm` staging are operational on an12/an29.

**Not established (no scientific claim exists yet):** dataset manifest, event anchors, per-axis measurement reliability, any commitment or readout curve, any policy result.

**Amendments introduced by this plan (consensus of the PI review round):**
1. **CFG doctrine** — three-layer gate structure replacing the earlier ambiguous treatment (Section 1.2).
2. **Logging & storage contract** — internal-feature caching is mandatory before any large run, with a hard budget (Section 1.4).
3. **Stage M (micro-map)** — a quantitative engineering checkpoint inserted before Phase-0 completion; the full grid cannot launch until it passes human review (Section 2).
4. **`A_independent` screening and the four-way determination taxonomy** (Sections 3.1, 4.3).
5. **Amended `GO_MAP` gate language** — α-ordering stability at the primary cfg is the hard requirement; cfg-invariance is science, not a kill switch (Section 6).
6. **Track P (internal probes)** promoted from "Phase 7, conditional" to a parallel, non-blocking analysis track that never feeds decision tokens (Section 7).
7. **Stage R (causal & robustness layer)**: condition-swap intervention, `large_44k` scale insurance, cfg-dial analysis — all conditional on `GO_MAP` (Section 8).
8. **SMC-style sequential baseline** added to the Phase-4 pre-registration; written now, implemented only after `GO_MAP + GO_READOUT` (Section 9).
9. **Figure-driven deliverable rule**: an experiment that feeds no figure or table in Section 11 is cut.

**Stage map (dependencies, not dates):**

```text
Stage M (micro-map + cfg-kernel checkpoint)
   └─► Stage 0 (Phase 0.3–0.5: manifest, anchors, reliability)  ─► GO_MAPS_PHASE
            └─► Phase 1 (commitment maps) ─► Phase 2 (readout maps) ─► Phase 3 (gap + GO/NO-GO)
                     │                                                      │
                     └─(features cached)──► Track P (internal probes) ──────┤  [non-gating]
                                                                            ├─ GO_MAP ──────────────► Stage R (condition-swap, large_44k, cfg-dial)
                                                                            └─ GO_MAP + GO_READOUT ─► Phase 4 (policy) ─► Phase 5/6 (verifier head, rollback)
Track L (human-time long poles) starts immediately and runs in parallel throughout.
```

---

## 1. Doctrine — Hard Rules Governing All Stages

### 1.1 Scientific object (unchanged, restated because agents drift)
- The maps target the **model's own final axis value (self-target)**, never human/MLLM correctness-vs-video. Correctness enters only in Phase 4, via the calibration sidecar.
- The maps are correctness-label-free but **measurement-dependent**; the reliability gate (Phase 0.5) is load-bearing.
- The **correctness factorization** that licenses the policy: `correctness(axis) = match(readable axis value, video anchor)` — time-independent once the value is readable. Axes whose correctness does not factor through a readable value (holistic quality) are **out of scope** for gated action and may appear only as legacy baselines.

### 1.2 CFG doctrine (three layers)
cfg is simultaneously a property of the system under study (conditioning strength) **and** a contaminant of the measurement instrument (the velocity→score identity is exact only at `cfg = 1.0`; at `cfg > 1` the affine-mixed velocity yields a tilted pseudo-score with no exact marginal-preservation guarantee). Therefore:

- **Gate A — per-cfg kernel validity (HARD, per model and per cfg).** Any cfg value entering a headline map must first pass, at that cfg: (i) small-α continuity; (ii) fork audio validity; (iii) nontrivial diversity; (iv) **distributional match** of fork-finals vs. independent ODE-finals (embedding-space MMD + per-axis marginal match; thresholds calibrated from the `cfg = 1.0` reference run). Failure → `CFG_KERNEL_FAIL(cfg=x)`; that cfg's commitment curves are uninterpretable. Fallback: headline at `cfg = 1.0` + readout-transfer check at deployed cfg + explicit scope note.
- **Gate B — α-ordering stability (HARD, at primary cfg).** At the primary operating point `cfg = 4.5`, the axis ordering of `s_commit` must be stable across the α pilot grid. α is an instrument parameter; ordering instability across α means the measurement, not the model, is unstable.
- **Science, not gate — cross-cfg behavior.** Window shifts and any ordering changes across cfg are reported as robustness/findings. **Pre-registered predictions (frozen before curves):** primary — `s_commit` shifts earlier for all axes as cfg increases; secondary — inter-axis separation may compress; ordering changes across cfg carry no directional bet and are reported as findings (multimodal speciation theory permits coupling to move decision times in either direction; semantic-class axes are the most likely movers).

Operating points: **primary `cfg = 4.5`** (deployed; headline maps), **reference `cfg = 1.0`** (exact kernel), **coarse `cfg = 2.5`** (dial midpoint, reduced grid).

### 1.3 α doctrine (unchanged)
Predefined α pilot grid; primary α = smallest α with measurable tail diversity and valid fork audio; full `A(axis, s, α)` surface reported as secondary evidence; `FORK_ALPHA_NO_VALID_OPERATING_POINT` routes to kernel redesign or diagnostic framing, never to silently raising α.

### 1.4 Logging & storage contract (HARD; implement before any large run)
- Pooled (mean-over-tokens) per-layer hidden states at **grid s-points** for **all** generations (base, independents, forks).
- Every-step pooled features for **base trajectories only**.
- Full token-level activations for **≤ 8 debug clips** total.
- `x̂0(s)` previews decoded and stored at grid s-points for **base + independents**; fork finals are measured on the fly with wavs retained for a 10% audit sample; all per-axis measurements stored for every generation.
- **Hard cap 100 GB.** If projected usage exceeds the cap: halt and report; no silent expansion, downsampling, or format degradation.

### 1.5 Evidence and statistics discipline
- Stage-M and synthetic outputs are **never** citable as scientific evidence; diagnostic evidence comes only from Phase 1–3+ runs on the frozen manifest.
- Thresholds (`θ_commit`, `θ_read`, `θ_rel`, `θ_robust`, `θ_cal`) frozen before inspecting headline curves; sensitivity sweeps reported afterward.
- Bootstrap unit = video, for all CIs on `s_commit`, `s_read`, gaps, and separation statistics.
- Statistical power rule: CI width on `s_commit` is dominated by across-video variance — **add clips, not forks** (K = 12 is past diminishing returns).
- Per-axis minimum usable n declared in the manifest; underpowered axes are reported as underpowered, never as results.

### 1.6 Parallelism rule
Non-blocking analyses (Track P) may run as soon as their inputs are cached, but their outputs **never** enter decision tokens. Track L (human-time items) starts immediately; nothing in Track L waits for GPU stages.

### 1.7 Novelty boundary and anti-overclaim (updated)
- Not a final-audio verifier paper; not a DiffRS clone; not an SMC-for-V2A paper. The contribution is the **per-axis commitment/readout map for cross-modally conditioned Foley decisions**, its causal validation, and what it licenses.
- Landscape note: scalar/global intermediate-reward search for V2A exists (SMC-style inference-time alignment); multi-verifier ITS exists for joint AV generation. Neither measures commitment, decomposes axes, or measures readout. Cite both; claim neither superiority nor obsolescence without head-to-head evidence (Phase 4).
- Do not present `cfg = 1.0`-only results as deployment-relevant. Do not cite music-ADSR as proof for V2A. Do not claim "irreducible uncommitted information" before internal probes have run — until then the phrase is "gap under available external probes."
- No generator fine-tuning, no delayed callback, no memory architectures, no new foundation model. If axes do not separate: route to DIAGNOSTIC/NEGATIVE; never force a method claim.

---

## 2. Stage M — Micro-Map + CFG-Kernel Checkpoint (engineering sanity; gate to everything)

> **[STATUS: EXECUTED 2026-06-11 → FAILED → HALT.]** 16 clips × cfg {1.0, 4.5},
> α=1.6 (pilot-selected at cfg=1.0), n_steps=20, ≈973 FGE. Verdict
> `MICROMAP_FAIL(endpoints, kernel_cfg4.5)`: criteria 2/4/5 PASS, criterion 1
> late-endpoint FAIL (class), criterion 3 FAIL (`GATE_A_UNDERPOWERED`).
> Attribution analysis + decision requests: `results/stage_m/pi_checkpoint.md`.
> Appendix E has the run-by-run record.

**Purpose:** exercise the entire measurement chain (generate → decode → measure → agree → normalize) end-to-end at toy scale, and validate the fork kernel at the deployed cfg, **before** any dataset download or large grid. Stage-M outputs are diagnostics, never evidence.

**Spec:**
- 16 clips, any on-hand VGGSound-style clips (FoleyBench **not** required for this stage).
- Axes: presence + coarse class only (implement these two `RealMeasurer` entries first).
- `s ∈ {0.05, 0.30, 0.60, 0.90}` (endpoints included by design), K = 8 forks, `cfg ∈ {1.0, 4.5}`, `N_independent = 8` per clip per cfg.
- Budget ≈ 800–900 full-generation equivalents (hours on one node).
- Simultaneously: run the Gate-A kernel check at `cfg = 4.5`; verify the logging contract end-to-end (features, previews, measurements land where specified).

**Quantitative pass criteria (all five required; reviewed by a human before the full grid launches):**
1. **Endpoints:** `A_fork(s = 0.90) ≥ 0.90` on both axes; `|A_fork(s = 0.05) − A_independent| ≤ 0.10`.
2. **Monotonicity:** `commit(s)` non-decreasing within CI tolerance.
3. **Kernel at cfg = 4.5:** small-α continuity below threshold; 8/8 forks valid; diversity above floor; distributional-match check passes against the `cfg = 1.0` reference.
4. **Measurer determinism** = 1.0 on identical wavs.
5. **Informativeness warning:** if ≥ 12/16 clips show `A_independent > 0.9` on the class axis, the candidate pool for Stage 0 must be widened/re-stratified before proceeding (this outcome is itself the justification for the screening step).

**Tokens:** `MICROMAP_PASS` / `MICROMAP_FAIL(reason)` / `CFG_KERNEL_OK(cfg=4.5)` / `CFG_KERNEL_FAIL(cfg=4.5)`.
**Failure routing:** endpoint failure at `s → 1` → suspect terminal-time SDE numerics; early-endpoint failure → suspect normalization or `A_independent` estimation; kernel failure → Section 1.2 fallback path.

---

## 3. Stage 0 — Phase-0 Completion (manifest, anchors, reliability) → `GO_MAPS_PHASE`

> **[STATUS: GATED — NOT RUN (Stage-M halt).]** Preparation done: 400-clip
> screening manifest selected; §3.2 anchors computed for all 785 clips
> (dual-source, σ median 0.757 s; 30-clip human check set pending — PI item);
> runners ready (`scripts/stage0_screening.py`, `stage0_mllm_sidecar.py`,
> `stage0_reliability.py`). No screening generation, sidecar at scale, or
> reliability gate has run; `GO_MAPS_PHASE` not emitted.

### 3.1 Phase 0.3 — Dataset manifest with `A_independent` screening
- Candidate pool: ~400 FoleyBench single-event clips (FoleyBench is the Phase-1 source of record; Stage-M clips are not carried forward).
- **Screen:** `N = 8` independent generations per candidate at `cfg = 4.5`; compute per-axis `A_independent`.
- **Stratify** the final **200 single-event clips** by per-axis prior agreement so each axis retains estimable diversity; add a **60-clip two-event subset** (curated from VGGSound if FoleyBench is single-event-dominated) for the binding axis (Tier 3).
- **Per-axis exclusion:** clips with `A_independent > 0.9` on an axis are excluded from that axis's normalized curve (denominator blow-up) and logged in the **video-determined registry**. They remain usable for other axes. Important: at screening time only the video-determined set is identifiable; seed-determined vs. trajectory-early/late classification is a **Phase-1 output** (Section 4.3), not a screening output.
- Freeze in the manifest: ~12 coarse Foley classes mapped from the AudioSet ontology; 8 s clip duration; a 60/40 probe-train/eval split by clip (consumed only by learned probes — Track P and Phase 5 — but frozen now); per-axis usable n; anchor source per clip.

### 3.2 Phase 0.4 — Event anchors
Source chain: FoleyBench metadata → off-the-shelf visual onset/event detector → light human marks. Validate on a **30-clip manual check set**; report MAE and coverage; record `σ_anchor`. **Propagation rule:** gross-timing bins ≥ `2·σ_anchor` wide. Timing and binding axes are uninterpretable without this; anchor uncertainty is the quiet failure mode of the timing axis.

### 3.3 Phase 0.5 — Reliability gate (three parts, per axis, on generated audio)
- **Measurer choices (pre-registered):** presence = energy gate + tagger eventness; coarse class = 16 kHz-native tagger (BEATs or PANNs-16k) as primary, second tagger for cross-tagger agreement; onset = spectral-flux; material/timbre = CLAP/BEATs embedding cosine (Tier 2, demotion armed). MLLM probes at temperature 0 with a test-retest subset and versioned prompts.
- **Thresholds (frozen before any curve):** determinism `θ_rel ≥ 0.95`; robustness `θ_robust ≥ 0.85` (class) / ±1 bin (timing) under event-window shift, loudness normalization, resampling, light compression, small noise; validity `κ ≥ 0.6` against a sidecar of ~100 MLLM-judged + ~50 human-judged clips.
- **Demotion rule:** any failure → demote or drop; emit `AXIS_DEMOTED:<axis>`. ≥ 3 axes must survive.
- Also pre-register here: the cfg-sweep predictions (Section 1.2) and the `GO_MAP` gate language (Section 6), so Phase 3 evaluates against frozen text.

**`GO_MAPS_PHASE` preconditions (all):** `MICROMAP_PASS`; `CFG_KERNEL_OK(cfg=4.5)`; logging contract implemented and exercised; manifest + anchors + reliability reports complete; ≥ 3 axes pass; splits and thresholds frozen.

---

## 4. Phase 1 — Commitment Maps (full grid)

### 4.1 Grid
- 200 single-event clips (+ 60 two-event for binding, headline cfg only).
- `s ∈ {0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90}` (densified where action value lives).
- Primary α (smallest-valid rule), K = 12; `N_independent = 16` per clip.
- cfg: full grid at 4.5 and 1.0; coarse grid (4 s-points, K = 8) at 2.5.
- Secondary α × 2 at 4 s-points, K = 6, `cfg = 4.5` only (the surface).
- **Compute envelope:** ≈ 40–45k full-generation equivalents total (including Phase-2 reuse); `small_16k` at ~3–4 s/gen ⇒ tens of GPU-hours of pure generation, ×2–3 pipeline overhead — a small number of node-days on an12+an29, trivially parallel by clip.

### 4.2 Estimation (unchanged math, two amendments)
- `commit(s, axis) = clip((A_fork − A_independent)/(1 − A_independent), 0, 1)`, **with the per-axis `A_independent > 0.9` exclusion** applied (Section 3.1); aggregate at the axis level, bootstrap by video.
- Agreement metrics: exact-match / Krippendorff's α for categorical axes; mean pairwise cosine for embedding axes; report majority-share alongside raw agreement for class.

### 4.3 Determination taxonomy (Phase-1 deliverable; Figure 1)
Operational definitions, applied per (axis, clip) and aggregated:
- **Video-determined:** `A_independent ≥ 0.9` (from screening registry).
- **Seed-determined / very-early:** normalized `commit(s = 0.05) ≥ θ_commit` (the fork at `s = 0⁺` already contains the seed, so `A_fork(ε) ≫ A_independent` ⇔ the seed locks the value). Feeds the separate H5 seed-predictability analysis (classifier: initial noise + video → final axis value).
- **Trajectory-early / trajectory-late:** `s_commit ≤ 0.4` / `≥ 0.7` (descriptive labels; the continuum is the result).

**Outputs:** `commitment_map.csv` (with cfg and α dimensions), `taxonomy_report.md`, `A(axis, s, α)` surface. **Token:** `COMMITMENT_MAP_DONE`.

---

## 5. Phase 2 — Readout Maps (external probes)

- Probes applied to cached `x̂0(s)` previews of base + independents at grid s-points, at the cfg of the trajectory (headline = 4.5).
- **Dual targets, both reported:** ODE-target (the candidate's own deterministic completion) and fork-majority (typical future under stochastic completion). Interpretation rule unchanged: readout of an uncommitted axis is path-reading, not decision-reading; only readout where `commit(s, axis)` is high licenses early action.
- Probe ladder: energy/onset heuristics → CLAP / SyncNet / ImageBind (legacy) → audio tagger → frontier MLLM-on-preview (≈ 3.5k calls incl. retest; temperature 0). The cheap learned verifier stays in Phase 5; internal probes live in Track P and do **not** enter `s_read` for gating.
- `s_read(axis, probe, target) = min s with accuracy/AUROC ≥ θ_read`, bootstrap by video, evaluated on the frozen eval split where a learned component is involved.

**Outputs:** `readout_map.csv`. **Token:** `READOUT_MAP_DONE`.

---

## 6. Phase 3 — Gap, Separation, GO/NO-GO (make-or-break ends here)

- `gap(axis, probe) = s_read − s_commit`; R1/R2 cross-tab per (axis, s): uncommitted → R1 (defer necessary); committed-but-unreadable-by-cheap-probe → R2 (probe-limited; flag for Track P confirmation); committed & readable → early-action candidate.
- Separation: bootstrap CIs on `s_commit` per axis at `cfg = 4.5`; ordered non-overlapping CIs or rank-order test; `separation_score = spread / mean CI width`; threshold-sensitivity sweep.

**Frozen gate language:**
- `GO_MAP` — separated commitment windows beyond CIs at the primary cfg, **and ordering stable across the α grid at that cfg** (Gate B). Cross-cfg ordering changes do **not** block `GO_MAP`; they are reported under the pre-registered cfg-sweep analysis.
- `GO_READOUT` — at least one feasible **external** probe reads the early axes well before the end (`s_read ≪ 1`) where committed.
- `GO_RESTRICTED` / `GO_DIAGNOSTIC` / `STOP_ADSR` / `STOP_PROJECT` — unchanged semantics from `EXPERIMENT_PLAN.md`.

`GO_MAP + GO_READOUT` unlocks Phase 4; `GO_MAP` alone unlocks Stage R.

---

## 7. Track P — Internal-Feature Probes (parallel, non-blocking, non-gating)

**Status change:** formerly "Phase 7, conditional on a large external gap." Now a standing analysis track that starts the moment Phase-1/2 features are cached. It consumes CPU/single-GPU only, never the A800 generation budget, and its outputs **never** enter decision tokens.

- **Training data:** pooled per-layer features at each grid s-point from the independents pool (~3,200 trajectories per cfg) + base trajectories, labels = each trajectory's own final self-target; 60/40 frozen split by clip.
- **Models:** linear/logistic probes per (layer, s) per axis; report the best-layer curve `s_read_internal(axis)`.
- **Headline question:** does `s_read_internal ≈ s_commit ≪ s_read_external`? If yes → R2 confirmed (gap is probe-limited, reducible) and the figure *"the generator knows before the audio shows"* exists; this is also the direct foundation of the Phase-5 verifier-as-head. If even internal probes cannot approach `s_commit` → genuine R1 evidence (defer is necessary). Until Track P reports, all gap language remains "gap under available external probes."
- **Output:** `internal_probe_report.md` + the overlay figure (Fig. 4).

---

## 8. Stage R — Causal & Robustness Layer (conditional on `GO_MAP`)

### 8.1 Condition-swap intervention (causal validation of commitment; Fig. 5)
- **Mechanic:** at progress s, replace the source clip's conditioning (CLIP visual + synchformer sync features, and text if used) with a donor clip's; complete the deterministic ODE at `cfg = 4.5`; measure final axes. Donor pairing: different coarse class, offset anchors.
- **Quantities:** follow-rate (matches donor) vs. retention-rate (matches source) per axis per s; `s_cond(axis)` = earliest s where follow-rate < 0.5. Sanity controls: swap at `s = 0` → full follow; swap near `s = 1` → full retention (10 clips).
- **Scale:** 40 source clips × 3 s-points bracketing the measured `s_commit(axis)`, for class + gross timing.
- **Caveats (pre-registered):** mid-trajectory condition switching is off-distribution; interpret as steerability, not distribution sampling. Fallback if full swaps produce degenerate audio: embedding interpolation (γ-blend) toward the donor.
- **Payoff:** a 2-D commitment picture — robustness to tail noise (`s_commit`) vs. robustness to conditioning change (`s_cond`) — plus a streaming-V2A interpretation ("when is it too late for the video to change the sound"). Labeled exploratory; bounded; never a gate.

### 8.2 Scale insurance: `large_44k` commitment-only replication
- 100 clips, primary α, `cfg = 4.5`, commitment map only (no readout, no policy). **Gate A must be re-run at `large_44k`** — kernel validity is per-model as well as per-cfg.
- Outcome framing: ordering stable across scale (strengthens generality) or scale-dependent (a finding). Either answers the single-checkpoint review.

### 8.3 cfg-dial analysis
Evaluate the pre-registered predictions (Section 1.2) on the {1.0, 2.5, 4.5} maps: window shift direction and magnitude per axis; separation compression; any ordering changes, with the multimodal-speciation framing. This is the result that makes "cross-modal" a manipulated variable rather than a setting.

---

## 9. Phase 4 — Axis-Gated Population Pruning (conditional on `GO_MAP + GO_READOUT`)

**Pre-registration (`policy_preregistration.md`, written before any Phase-4 run; the SMC baseline definition is written NOW even though implementation waits):**
- Precise definitions of: scalar DiffRS-style rejection; **SMC-style sequential resampling with a scalar intermediate reward** (the strongest published V2A inference-time-alignment pattern); oracle axis-gated pruning (upper bound); same-compute accounting (matched generator NFE **and** matched scoring-call budgets); offline-simulated vs. online execution.
- **Offline simulation first:** candidate pools = the cached N = 16 independents per clip at `cfg = 4.5`; replay gating decisions from stored per-s measurements and probe scores; only run online if offline shows headroom.
- Correctness-vs-video sidecar scales only after both GO tokens; pruning decisions judged against actual wrongness via the correctness factorization (Section 1.1).

**Method (unchanged):** generate N candidates to the first actionable window; prune only on axes whose windows have closed and whose early-action precision is high; continue survivors; evaluate later axes at later windows; finish and rerank. Operating points set by early-action precision / false-prune / winner-retention / regret — never raw correlation.

**Baselines (same candidate pools):** full BoN; same-compute BoN; random pruning; scalar DiffRS-style rejection; **SMC-style scalar sequential resampling**; final-score reranking; seed restart; oracle axis-gated pruning.
**Metrics:** final + per-axis correctness; completed candidates; total NFE; scoring-call budget; winner retention; false-prune rate; regret; compute–quality Pareto (two-axis accounting: generator FLOPs and scoring calls, reported separately).
**Tokens:** `GO_POLICY` / `GO_RESTRICTED` / `DIAGNOSTIC_ONLY`.

**Optional bolt-on (only if H5 fired in Phase 1):** seed triage — a small classifier from (initial noise, video) rejecting doomed seeds at zero NFE; reported as a bonus, never as the headline.

---

## 10. Phases 5–6 — Verifier Head and Rollback (conditional)

- **Phase 5 (verifier), revised:** if Track P shows internal features carry the readout signal, the "cheap process-aware verifier" is implemented as a **frozen-generator feature head** (linear/MLP on cached pooled features) rather than a separately trained preview model — near-zero marginal scoring cost since features are computed during generation anyway. Re-run Phase 4 with it as the per-step scorer; report scoring-call savings vs. MLLM-on-preview at matched pruning quality. Train only after maps show headroom; respect the frozen split.
- **Phase 6 (rollback):** Restart re-noising or the SDE kernel to repair in-window axis failures. Run only if forking is stable and Phase 4 shows the failure mode it would fix. Baselines: Restart without axis gating; scalar rejection + rollback; full seed restart.

---

## 11. Figure-Driven Deliverable Map (the cut rule)

| Deliverable | Content | Producing stage |
|---|---|---|
| **Fig. 1** | Determination spectrum: video-determined / seed-determined / trajectory-early / trajectory-late, per axis | Phase 1 (§4.3) |
| **Fig. 2** | Commitment surfaces `A(axis, s, α)` + cfg shift overlay | Phase 1 + §8.3 |
| **Fig. 3** | Commitment–readout gap + R1/R2 cross-tab | Phase 3 |
| **Fig. 4** | `s_commit` vs. `s_read_internal` vs. `s_read_external` overlay ("the generator knows before the audio shows") | Track P |
| **Fig. 5** | Condition-swap follow/retention curves; `s_cond` vs. `s_commit` | Stage R §8.1 |
| **Fig. 6** | Policy Pareto at matched NFE and matched scoring calls, all baselines | Phase 4 |
| **Tab. 1** | Reliability gate results + demotions | Stage 0 |
| **Tab. 2** | Separation statistics + threshold sensitivity + α/cfg robustness | Phase 3 |
| **Tab. 3** | `large_44k` replication summary | Stage R §8.2 |

**Rule:** any proposed experiment that feeds none of the above is cut unless this document is first amended.

---

## 12. Claim Tiers and Outcome Routing (updated)

| Tier | Required evidence | Notes |
|---|---|---|
| **METHOD** | `GO_MAP + GO_READOUT + GO_POLICY`: maps separated, readable early, axis-gated policy beats the strongest matched-compute baseline (incl. the SMC-style baseline) beyond CIs | Figs. 1–6 all present |
| **DIAGNOSTIC (strong)** | `GO_MAP` + R2-dominated gap; internal probes close the gap that external probes cannot | Ships Figs. 1–5; the verifier-head result may still appear as analysis |
| **NEGATIVE (publishable)** | `STOP_ADSR`: windows coincide or sit at `s ≈ 1` | Honest diagnostic; cfg-dial + taxonomy + condition-swap still publishable as the characterization |
| **STOP** | Reliability/feasibility failure | `STOP_PROJECT` |

Even the floor tiers ship Track P and Stage R content — this is deliberate: the paper's identity is the map and its validation, not the policy.

---

## 13. Track L — Long Poles (start immediately; human-time, not GPU-time)

> **[STATUS 2026-06-11]** (2) 30-clip anchor check set: template READY
> (`data/manifests/anchor_check_30.csv`), human marks pending. (4) MLLM prompts
> v1 frozen + qwen judge live-validated; ~10 of the ≤500-call slice budget used.
> (5) storage monitoring ACTIVE (0.433 GB / 100 GB). (1) two-event subset:
> source identified (FoleyBench Multi-source+Discrete, 195 candidates), curation
> not started. (3) human sidecar protocol: pending PI checkpoint.

1. **Two-event subset curation** (60 clips) — the only real data-curation cost; begins before Stage M completes.
2. **30-clip anchor manual check set** — required for Phase 0.4; independent of GPUs.
3. **Human sidecar protocol** (~50 clips) — recruitment/instructions written early; executed at Phase 0.5 and scaled at Phase 4.
4. **MLLM probe budget & prompt versioning** — fixed prompts, temperature 0, response caching; total budget across Phases 2/0.5/4 ≈ 5–6k calls.
5. **Storage monitoring** against the 100 GB cap (Section 1.4).

---

## 14. Consolidated Decision-Token Registry

| Token | Emitted by | Meaning |
|---|---|---|
| `MICROMAP_PASS` / `MICROMAP_FAIL(reason)` | Stage M | Measurement chain healthy / failed criterion named |
| `CFG_KERNEL_OK(cfg=x)` / `CFG_KERNEL_FAIL(cfg=x)` | Stage M, §8.2 | Gate-A kernel validity per cfg (and per model) |
| `GO_MAPS_PHASE` | Stage 0 | All §3 preconditions hold |
| `AXIS_DEMOTED:<axis>` | Stage 0 | Reliability failure → demote/drop |
| `FORK_ALPHA_NO_VALID_OPERATING_POINT` | Phase 1 | No usable α; kernel redesign or diagnostic routing |
| `COMMITMENT_MAP_DONE` / `READOUT_MAP_DONE` | Phases 1/2 | Maps complete |
| `GO_MAP` / `GO_READOUT` | Phase 3 | Make-or-break passed (see frozen language, §6) |
| `GO_RESTRICTED` / `GO_DIAGNOSTIC` / `STOP_ADSR` / `STOP_PROJECT` | Phase 3 | Alternative routings |
| `GO_POLICY` / `DIAGNOSTIC_ONLY` | Phase 4 | Policy outcome |

---

## 15. Standing Agent Instructions

1. Start at the lowest unsatisfied gate; never skip a gate because a later stage "looks ready."
2. Do not infer missing results; do not upgrade claim tier without the named token; prefer early stop.
3. Stage-M and synthetic outputs are never evidence. Micro-map curves go to human review before the full grid.
4. Track P runs in parallel but never touches tokens. Track L items are launched immediately and tracked like experiments.
5. Every report states: which gate it serves, which token it justifies, and which figure/table it feeds. Anything that answers "none" to the last question is out of scope until this plan is amended.
