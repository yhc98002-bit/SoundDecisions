# Experiment Plan — `foley-cw` / Sound Decisions

**Project:** *When Are Foley Decisions Made? Commitment and Readout Windows in V2A Flow Generation.*
**Backbone:** MMAudio rectified flow (`small_16k` primary; `large_44k` for scale insurance). **Audience:** autonomous research agents and collaborators. Decision tokens are the only legitimate way to change claim tier or unlock a stage. No calendar dates: sequencing is by gates and dependencies only.

**History (one line):** the Phase-0 crux is validated on the real model (trajectory access; velocity→score exact at `cfg = 1.0`, α = 0 reproduces the ODE), and the first micro-map run halted on three instrument faults — an argmax knife-edge in the class measure, a kernel test that conflated seed-conditioning with kernel error, and no constant-α operating point at deployed cfg — all corrected in the spec below; full journals live in `results/` and are not duplicated here. **Next action: Stage-M re-run (§2).**

**Status (one line, maintained per §15.6):** 2026-06-12 — Stage-M run 3 at the pilot-selected tuple **(cfg=1.0, sqrt_down, α=0.8)**: 4/5 criteria PASS incl. **Gate-A OK at BOTH cfgs** (`CFG_KERNEL_OK(cfg=1)`; cfg=4.5 under sqrt_down passes the calibrated adjudication — candidate §1.2 re-entry token, labeling caveat noted); sole failure = early-endpoint class gap **0.153 vs the 0.10 band**, co-existing with passing marginal exchangeability — most consistent with a genuine label-space seed floor, i.e. the §4 determination-budget "seed share" (analysis, not a claim). **HALTED per the frozen iteration bound — PI checkpoint #2: `results/stage_m_rerun/pi_checkpoint2.md` (5 decisions; recommendation = re-scope the early rule to the budget framework, zero GPU cost).** Run-1/2 records: `results/stage_m/pi_checkpoint.md`, `results/stage_m_rerun/run2_archive/`. Stage-0 remains gated.

```text
Stage M (micro-map checkpoint)
   └─► Stage 0 (manifest, anchors, reliability) ─► GO_MAPS_PHASE
            └─► Phase 1 (commitment) ─► Phase 2 (readout) ─► Phase 3 (gap + GO/NO-GO)
                     │                                            │
                     └─(features cached)─► Track P (internal) ────┤  [non-gating]
                                                                  ├─ GO_MAP ───────────► Stage R (condition-swap, large_44k, cfg-dial + F-1)
                                                                  └─ GO_MAP+GO_READOUT ─► Phase 4 (policy) ─► Phases 5/6
Track L (human-time long poles) runs in parallel from the start.
```

---

## 1. Doctrine

### 1.1 Scientific object
The maps target the **model's own final axis value (self-target)**, never human correctness-vs-video; correctness enters only in Phase 4 via the sidecar. The maps are label-free but measurement-dependent — the reliability gate (§3.3) is load-bearing. The policy is licensed by the **correctness factorization**: `correctness(axis) = match(readable axis value, video anchor)`, time-independent once the value is readable. Axes whose correctness does not factor through a readable value (holistic quality) are out of scope for gated action.

### 1.2 CFG doctrine
The velocity→score identity is exact only at `cfg = 1.0`; at `cfg > 1` the CFG-mixed velocity yields a tilted pseudo-score with no exact marginal-preservation guarantee, and no constant-α fork operating point exists at deployed guidance. Operating points:

- **Headline `cfg = 1.0`** (exact kernel): commitment maps, readout maps for the gap, the α surface, separation tests, Gate B.
- **Deployed `cfg = 4.5`**: readout-transfer (probes on `x̂0(s)` need no fork kernel), the F-1 protocol (§8.3), condition-swap, and Phase-4 policy. Commitment maps at 4.5 only after re-entry.
- **Re-entry route:** a (cfg, g(s)-schedule) tuple that passes Gate A emits `CFG_KERNEL_OK(cfg=x, schedule=g)` and re-enters the headline commitment grid. `cfg = 2.5` is the schedule-pilot candidate.

**Gate A — kernel validity (HARD, per cfg × schedule × model).** Required before any commitment map at that tuple: (i) small-α continuity; (ii) fork-audio validity; (iii) nontrivial diversity; (iv) **seed-marginalized exchangeability** — fork one tail from each of the N independents at s, pool, permutation-test against fresh independents on tagger-probability-vector MMD + label-marginal TV. Per-seed-cell embedding tests are forbidden: the seed legitimately fingerprints fine texture, so they conflate seed-conditioning with kernel error. Commitment measurement itself keeps K forks per seed — per-seed posterior concentration *is* the signal. Guards never pass silently.

**Gate B — α-ordering stability (HARD, at the headline cfg).** The axis ordering of `s_commit` must be stable across the α pilot grid; α is an instrument parameter, so ordering instability across α indicts the measurement, not the model.

**Science, not gate — cross-cfg behavior.** Pre-registered predictions: `s_commit` shifts earlier as cfg increases; inter-axis separation may compress; ordering changes carry no directional bet. The F-1 predictions (§8.3) extend these to the limit where commitment collapses onto the initial noise.

### 1.3 α and noise schedule
- Primary α = smallest α with measurable tail diversity and valid fork audio; set value `PRIMARY_ALPHA = 1.6` at `cfg = 1.0`. Full `A(axis, s, α)` surface reported as secondary evidence.
- **Early-heavy g(s) schedules** (`linear_down`, `sqrt_down`) are the sanctioned kernel-redesign route — explore early where decisions are open, quiet the tail. Pilot at `cfg ∈ {2.5, 4.5}`; each tuple needs its own Gate-A pass.
- **Discretization arm:** `n_steps = 40` on a micro-map-scale sub-grid bounds integrator error vs. dynamics in the high-cfg locking. Production grids use `n_steps = 20` so every scan point lies on the integration grid.
- `FORK_ALPHA_NO_VALID_OPERATING_POINT` routes to the schedule pilots or diagnostic framing — never to silently raising α.

### 1.4 Logging & storage contract (HARD)
Pooled per-layer hidden states at grid s-points for **all** generations; every-step pooled features for base trajectories only; token-level activations for ≤ 8 debug clips; `x̂0(s)` previews stored for base + independents; fork finals measured on the fly with a 10% wav audit sample; all per-axis measurements stored. **Hard cap 100 GB** — halt and report if exceeded; no silent expansion or degradation.

### 1.5 Evidence & statistics
Stage-M and synthetic outputs are never citable as evidence; evidence comes only from Phase 1–3+ runs on the frozen manifest. All thresholds (`θ_commit`, `θ_read`, `θ_rel`, `θ_robust`, `θ_cal`, δ) are frozen before inspecting headline curves; sensitivity sweeps after. Bootstrap unit = video. Power rule: CI width is dominated by across-video variance — **add clips, not forks** (K = 12 is past diminishing returns). Per-axis minimum usable n is declared in the manifest; underpowered axes are reported as underpowered, never as results.

### 1.6 Parallelism
Non-gating analyses (Track P) start as soon as their inputs are cached but never feed decision tokens. Track L items launch immediately; nothing human-time waits on GPUs.

### 1.7 Novelty boundary & anti-overclaim
Not a final-audio verifier paper, not a DiffRS clone, not an SMC-for-V2A paper: the contribution is the per-axis commitment/readout map for cross-modally conditioned Foley decisions, its causal validation, and what it licenses. Scalar intermediate-reward search for V2A and multi-verifier ITS for joint AV generation exist — cite both; claim nothing without head-to-head evidence (Phase 4). Do not present headline-cfg results as deployment-relevant without the readout-transfer evidence and scope note. Do not cite music-ADSR as proof for V2A. Until Track P reports, the gap is "gap under available external probes," not "irreducible." No generator fine-tuning, no delayed callback, no new foundation model. If axes do not separate, route to DIAGNOSTIC/NEGATIVE; never force a method claim.

---

## 2. Stage M — Micro-Map Checkpoint (next action; diagnostics, never evidence)

Shakes down the full chain (generate → decode → measure → agree → normalize) at toy scale before any large grid.

**Spec:** 16 FoleyBench Single-source+Discrete clips, registered as exclusions from the Phase-1 manifest. Axes: presence + coarse class (event-restricted argmax + abstain margin δ, §3.3), with the non-gating entropy/margin instruments logged on the same runs. Grid: `s ∈ {0.05, 0.30, 0.60, 0.90}`, K = 8, `cfg ∈ {1.0, 4.5}`, `N_independent = 8` per clip per cfg; Gate A additionally takes one fork per independent at its test s-points. The 4.5 arm serves Gate-A adjudication and the g(s)-schedule pilots — it is not a pass requirement. Budget ≈ 1k full-generation equivalents (≈ 20 GPU-min); logging contract exercised end-to-end.

**Pass criteria (human review before the full grid):**
1. **Endpoints, granularity-aware.** Label level (confident subset): `A_fork(0.90) ≥ 0.90` and `|A_fork(0.05) − A_independent| ≤ 0.10`, abstain rate reported at every s. Embedding level: a stable, CI-bounded seed floor `A_fork_emb(0.05)` with `A_fork_emb(0.90) > A_fork_emb(0.05)` beyond CIs (the early-endpoint identity is not required — the seed floor is real).
2. **Monotonicity:** `commit(s)` non-decreasing within CI tolerance (label level, confident subset).
3. **Kernel:** Gate A passes at `cfg = 1.0` (hard). At 4.5, Gate A is adjudicated and reported; a 4.5 failure routes to §1.2 fallback, not a halt.
4. **Measurer determinism** = 1.0 on identical wavs.
5. **Informativeness:** < 12/16 clips video-pinned (`A_independent > 0.9`) on class, else widen/re-stratify the pool; abstains ≤ 30% of fork-final labels at s = 0.90, else revisit δ or trigger the BEATs contingency.

**Tokens:** `MICROMAP_PASS` / `MICROMAP_FAIL(reason)` / `CFG_KERNEL_OK|FAIL(cfg=x[, schedule=g])`.
**Routing:** late-endpoint failure on the confident subset → genuinely suspect kernel/terminal-time numerics; early-endpoint label failure → normalization or `A_independent` estimation; Gate-A failure at `cfg = 1.0` → STOP-level instrument review.

---

## 3. Stage 0 — Manifest, Anchors, Reliability → `GO_MAPS_PHASE`

### 3.1 Manifest with screening
~400 FoleyBench single-event candidates → `N = 8` independents per clip **at the headline cfg** (exclusions must share the conditional distribution of the curves they gate) → per-axis `A_independent` → stratified **200 single-event + 60 two-event** clips (two-event from the FoleyBench Multi-source pool; VGGSound fallback). Clips with `A_independent > 0.9` on an axis are excluded from that axis's normalized curve and logged in the **video-determined registry** (still usable for other axes); seed-vs-trajectory classification is a Phase-1 output, not a screening output. A 60-clip `cfg = 4.5` sub-screen is logged for Phase-4 pools and F-1, non-gating. Frozen in the manifest: ~12 event-restricted coarse classes, 8 s clip duration, 60/40 probe-train/eval split by clip, per-axis usable n, anchor source per clip.

### 3.2 Event anchors
FoleyBench ships no event timestamps, so **original-audio-track onsets (spectral flux) are the primary anchor source**, visual onset detector as cross-check, `σ_anchor` = cross-source disagreement, light human marks where both fail. The **30-clip manual check set** has two jobs: report MAE/coverage, and arbitrate whether validated audio-only anchors beat the current disagreement (median 0.757 s → 1.51 s bins, ≈ 5 bins per clip — marginal for timing power; if audio-only `σ_anchor ≤ 0.35 s`, adopt audio-only and re-derive bins). **Propagation rule:** timing bins ≥ `2·σ_anchor`. Anchor uncertainty is the quiet failure mode of the timing axis.

### 3.3 Reliability gate (per axis, on generated audio)
- **Class:** tagger argmax restricted to event classes (speech/music/ambient excluded; frozen coarse map) with abstain margin δ; agreement on the confident subset, abstain rate always reported. Primary tagger PANNs-16k with a **BEATs swap contingency** if the sanity score is not well clear of 0.65; second tagger for cross-tagger agreement; qwen judge triangulates at micro-map scale. Non-gating parallel instruments wherever class is measured: fork-ensemble label entropy / top-prob concentration, and top-1 margin trajectories `margin(s)`.
- **Other axes:** presence = energy gate + tagger eventness; onset = spectral flux vs. anchor bins; material/timbre = CLAP/BEATs embedding cosine (Tier 2, demotion armed; normalized against the seed floor, §4.2).
- **MLLM probes:** temperature 0, versioned prompts, response cache, test-retest subset.
- **Thresholds (frozen):** determinism `θ_rel ≥ 0.95`; robustness `θ_robust ≥ 0.85` (class) / ±1 bin (timing) under event-window shift, loudness normalization, resampling, light compression, small noise; validity `κ ≥ 0.6` against ~100 MLLM-judged + ~50 human-judged clips. Any failure → demote or drop (`AXIS_DEMOTED:<axis>`); ≥ 3 axes must survive.
- Pre-register here: cfg-sweep predictions (§1.2), F-1 predictions (§8.3), and the `GO_MAP` gate language (§6).

**`GO_MAPS_PHASE` preconditions (all):** `MICROMAP_PASS`; Gate A passed at the headline cfg; logging contract exercised; manifest + anchors + reliability complete; ≥ 3 axes pass; splits and thresholds frozen.

---

## 4. Phase 1 — Commitment Maps

**Grid:** 200 single-event clips (+ 60 two-event for binding); `s ∈ {0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90}`; primary α, K = 12; `N_independent = 16`. cfg: full grid at the headline `cfg = 1.0`; `cfg = 4.5` runs the commitment grid only under a re-entry token, otherwise its budget goes to F-1 and readout-transfer trajectories; `cfg = 2.5` coarse grid contingent on its own Gate-A pass. Secondary α × 2 at 4 s-points, K = 6, headline cfg only. Envelope ≈ 25k full-generation equivalents (≈ 40k with re-entry) — a small number of node-days, trivially parallel by clip.

**Estimation — the two-baseline determination budget.** There are two floors: the **conditioning floor** `A_independent` (what the video induces) and the **seed floor** `A_fork(s_min)` (what the initial noise additionally locks — real even at the exact kernel).
- **Label-granular axes** (presence, class confident subset, timing bins): `commit(s) = clip((A_fork − A_independent)/(1 − A_independent), 0, 1)` with the `A_independent > 0.9` exclusion; the seed floor should ≈ the conditioning floor here — report the difference as a check.
- **Embedding-granular axes** (material/timbre): `commit_traj(s) = clip((A_fork(s) − A_fork(s_min))/(1 − A_fork(s_min)), 0, 1)`, with `A_independent` reported alongside, never as denominator.
- **Determination budget (per axis × cfg; the Figure-1 quantity):** conditioning share = `A_independent`; seed share = `A_fork(s_min) − A_independent`; trajectory share resolved by s = `A_fork(s) − A_fork(s_min)`; residual = `1 − A_fork(s)`. Shares clipped at 0; bootstrap by video.
- Agreement metrics: exact-match / Krippendorff's α on the confident subset (abstain rate reported); pairwise cosine for embedding axes; majority-share alongside raw agreement for class.

**Taxonomy (Figure 1):** video-determined (`A_independent ≥ 0.9`, from the registry); seed-determined (`commit(0.05) ≥ θ_commit`, i.e. a dominant seed share — feeds the H5 noise→value probe); trajectory-early (`s_commit ≤ 0.4`) / trajectory-late (`≥ 0.7`) as descriptive labels on the continuum.

**Outputs:** `commitment_map.csv` (cfg, α, schedule dimensions), `determination_budget.csv`, `taxonomy_report.md`, the `A(axis, s, α)` surface. **Token:** `COMMITMENT_MAP_DONE`.

---

## 5. Phase 2 — Readout Maps (external probes)

Probes run on cached `x̂0(s)` previews of base + independents. cfg roles: headline previews feed the gap analysis; deployed-cfg previews are the readout-transfer set and feed Phase 4 directly. **Dual targets, both reported:** ODE-target and fork-majority; readout of an uncommitted axis is path-reading, not decision-reading — only readout where commitment is high licenses early action. **Probe ladder:** heuristics → CLAP/SyncNet/ImageBind (legacy) → audio tagger → MLLM-on-preview, two-tier: qwen3.5-omni-plus as the primary versioned probe (full grid, ≈ 3.5k cached calls, temperature 0) plus a frontier-class judge on a stratified subset (~100 clips × 4 s-points) to bound the external ceiling — R2 attribution cites the ceiling tier. `s_read(axis, probe, target) = min s with accuracy/AUROC ≥ θ_read`, bootstrap by video, learned components evaluated on the frozen split. **Output:** `readout_map.csv`. **Token:** `READOUT_MAP_DONE`.

---

## 6. Phase 3 — Gap, Separation, GO/NO-GO

`gap(axis, probe) = s_read − s_commit`, computed at the headline cfg. R1/R2 cross-tab per (axis, s): uncommitted → R1 (defer necessary); committed-but-unreadable → R2 (probe-limited; flag for Track P); committed & readable → early-action candidate. Separation: bootstrap CIs on `s_commit` per axis at the headline cfg; ordered non-overlapping CIs or rank-order test; `separation_score = spread / mean CI width`; threshold-sensitivity sweep.

**Gate language (frozen):** `GO_MAP` = separated windows beyond CIs at the headline cfg **and** ordering stable across the α grid (Gate B); cross-cfg ordering changes never block it. `GO_READOUT` = at least one feasible external probe reads the early axes well before the end where committed. `GO_RESTRICTED` / `GO_DIAGNOSTIC` / `STOP_ADSR` / `STOP_PROJECT` retain their standard semantics. `GO_MAP + GO_READOUT` unlocks Phase 4; `GO_MAP` alone unlocks Stage R.

---

## 7. Track P — Internal-Feature Probes (parallel, non-gating)

Starts the moment Phase-1/2 features are cached; CPU/single-GPU only; outputs never enter decision tokens. Training data: pooled per-layer features at each grid s-point from the independents pool (~3.2k trajectories per cfg) + base trajectories; labels = each trajectory's own final self-target; frozen 60/40 split. Linear/logistic probes per (layer, s) per axis; report the best-layer curve `s_read_internal(axis)`. Headline question: `s_read_internal ≈ s_commit ≪ s_read_external`? Yes → R2 confirmed, the *"generator knows before the audio shows"* figure exists, and the Phase-5 verifier-as-head is grounded. No (even internal probes lag commitment) → genuine R1 evidence. **Output:** `internal_probe_report.md` + Fig. 4.

---

## 8. Stage R — Causal & Robustness Layer (conditional on `GO_MAP`)

### 8.1 Condition-swap intervention (Fig. 5)
At progress s, replace the source clip's conditioning (CLIP + synchformer + text) with a donor's; complete the deterministic ODE at deployed cfg; measure final axes. Follow-rate vs. retention-rate per axis per s; `s_cond(axis)` = earliest s with follow-rate < 0.5. Sanity controls on 10 clips (swap at s = 0 → full follow; near s = 1 → full retention). Scale: 40 source clips × 3 s-points bracketing `s_commit`, for class + gross timing. Caveat (pre-registered): mid-trajectory switching is off-distribution — interpret as steerability; fallback to embedding interpolation if full swaps degenerate. Payoff: the 2-D picture (robust-to-tail-noise vs. robust-to-condition-change) plus the streaming-V2A reading. Exploratory; never a gate.

### 8.2 Scale insurance
`large_44k`, 100 clips, primary α, headline cfg, commitment map only. Gate A re-runs at `large_44k` — kernel validity is per-model as well as per-(cfg, schedule). Ordering stable across scale strengthens generality; scale-dependence is itself a finding.

### 8.3 cfg-dial + F-1 ("guidance moves the decision into the seed")
- **cfg-dial:** evaluate the §1.2 predictions on every cfg with a certified kernel — shift direction/magnitude, separation compression, ordering changes, speciation framing. This makes "cross-modal" a manipulated variable.
- **F-1 (registered candidate finding):** motivated by the mode-locking diagnostic (same seed → same class at every piloted α; different seeds → different classes), i.e. *class may be seed-determined at deployed guidance*. Protocol, instrument-light (no marginal-preserving kernel needed at high cfg): ~24 clips × `cfg ∈ {1.0, 1.5, 2.0, 2.5, 3.0, 4.5}`, measuring (a) seed-predictability of class — earliest-s fork agreement plus a small (noise, video) → class probe — and (b) the minimum unlocking α* per cfg. **Pre-registered predictions:** α*(cfg) increasing; commitment curves shift left and degenerate toward step functions at the seed; the seed share of the determination budget grows with cfg. **Tokens:** `F1_SUPPORTED` / `F1_REFUTED` / `F1_INCONCLUSIVE`. If supported, the conclusion upgrades to *guidance changes who decides — trajectory at low cfg, seed at deployed cfg* — and Phase 4 becomes guidance-aware.

---

## 9. Phase 4 — Axis-Gated Population Pruning (conditional on `GO_MAP + GO_READOUT`)

`policy_preregistration.md` is written before any run and defines: scalar DiffRS-style rejection; SMC-style sequential resampling with a scalar intermediate reward (the strongest published V2A pattern); oracle axis-gated pruning; matched generator-NFE **and** matched scoring-call accounting; offline-simulated vs. online execution. **Offline first:** pools = the cached deployed-cfg independents; replay gating from stored measurements and probe scores; go online only if offline shows headroom. The correctness sidecar scales only after both GO tokens; pruning is judged against actual wrongness via the correctness factorization.

**Method:** generate N candidates to the first actionable window; prune only on axes whose windows have closed with high early-action precision; continue survivors; evaluate later axes at later windows; finish and rerank. Operating points set by early-action precision / false-prune / winner-retention / regret — never raw correlation.
**Baselines (same pools):** full BoN; same-compute BoN; random pruning; scalar DiffRS rejection; SMC-style scalar resampling; final-score reranking; seed restart; oracle gating.
**Metrics:** final + per-axis correctness; completed candidates; total NFE; scoring calls; winner retention; false-prune rate; regret; the two-axis compute–quality Pareto.
**Guidance-aware framing (contingent on `F1_SUPPORTED`):** the optimal intervention point migrates with cfg — seed triage (zero-NFE selection on the initial noise) becomes the primary lever at deployed guidance, axis-gated mid-trajectory pruning governs moderate guidance and late-deciding axes; report the policy comparison per cfg regime, with seed triage promoted to a headline arm at deployed cfg.
**Tokens:** `GO_POLICY` / `GO_RESTRICTED` / `DIAGNOSTIC_ONLY`.

---

## 10. Phases 5–6 — Verifier Head and Rollback (conditional)

**Phase 5:** if Track P shows internal features carry the signal, the cheap verifier is a frozen-generator **feature head** (linear/MLP on cached pooled features) — near-zero marginal scoring cost. Re-run Phase 4 with it as the per-step scorer; report scoring-call savings vs. MLLM-on-preview at matched pruning quality; respect the frozen split. **Phase 6:** Restart re-noising or the SDE kernel to repair in-window axis failures — only if forking is stable and Phase 4 exhibits the failure mode it would fix; baselines: ungated Restart, scalar rejection + rollback, full seed restart.

---

## 11. Deliverable Map (the cut rule)

| Deliverable | Content | Stage |
|---|---|---|
| **Fig. 1** | Stacked determination budget (conditioning / seed / trajectory-by-s / residual) per axis × cfg; the taxonomy is its discretization | Phase 1 |
| **Fig. 2** | Commitment surfaces `A(axis, s, α)` + cfg shift + α*(cfg) unlocking curve | Phase 1, §8.3 |
| **Fig. 3** | Commitment–readout gap + R1/R2 cross-tab | Phase 3 |
| **Fig. 4** | `s_commit` vs. internal vs. external readout ("the generator knows before the audio shows") | Track P |
| **Fig. 5** | Condition-swap follow/retention; `s_cond` vs. `s_commit` | §8.1 |
| **Fig. 6** | Policy Pareto at matched NFE and matched scoring calls | Phase 4 |
| **Tab. 1** | Reliability gate results + demotions | Stage 0 |
| **Tab. 2** | Separation statistics + threshold/α/cfg sensitivity | Phase 3 |
| **Tab. 3** | `large_44k` replication summary | §8.2 |

Any experiment feeding none of the above is cut unless this plan is amended first.

---

## 12. Claim Tiers

| Tier | Required evidence |
|---|---|
| **METHOD** | `GO_MAP + GO_READOUT + GO_POLICY`: separated, readable-early maps; axis-gated policy beats the strongest matched-compute baseline (incl. SMC-style) beyond CIs |
| **DIAGNOSTIC (strong)** | `GO_MAP` + R2-dominated gap; internal probes close what external probes cannot |
| **NEGATIVE (publishable)** | `STOP_ADSR`: windows coincide or sit near s = 1 — honest characterization; budget, cfg-dial, F-1, condition-swap still publish |
| **STOP** | Reliability/feasibility failure (`STOP_PROJECT`) |

Floor tiers still ship Track P and Stage R content — the paper's identity is the map and its validation, not the policy.

---

## 13. Track L — Long Poles (human-time; start immediately)

Two-event subset curation (60 clips, FoleyBench Multi-source pool); 30-clip anchor manual marks (also arbitrating audio-only anchor sharpness, §3.2); ~50-clip human validity sidecar; MLLM prompt versioning + caching within a ≈ 5–6k total call budget; storage monitoring against the 100 GB cap.

---

## 14. Decision-Token Registry

| Token | Emitted by | Meaning |
|---|---|---|
| `MICROMAP_PASS` / `MICROMAP_FAIL(reason)` | Stage M | Measurement chain healthy / failed criterion named |
| `CFG_KERNEL_OK|FAIL(cfg=x[, schedule=g])` | Stage M, §1.3 pilots, §8.2 | Gate-A verdict per (cfg, schedule, model); OK is the re-entry token |
| `GO_MAPS_PHASE` | Stage 0 | All §3 preconditions hold |
| `AXIS_DEMOTED:<axis>` | Stage 0 | Reliability failure → demote/drop |
| `FORK_ALPHA_NO_VALID_OPERATING_POINT` | α pilots | No usable α; route to schedule pilots or diagnostic framing |
| `COMMITMENT_MAP_DONE` / `READOUT_MAP_DONE` | Phases 1/2 | Maps complete |
| `GO_MAP` / `GO_READOUT` | Phase 3 | Make-or-break passed (frozen language, §6) |
| `GO_RESTRICTED` / `GO_DIAGNOSTIC` / `STOP_ADSR` / `STOP_PROJECT` | Phase 3 | Alternative routings |
| `F1_SUPPORTED` / `F1_REFUTED` / `F1_INCONCLUSIVE` | §8.3 | F-1 protocol outcome |
| `GO_POLICY` / `DIAGNOSTIC_ONLY` | Phase 4 | Policy outcome |

---

## 15. Standing Agent Instructions

1. Start at the lowest unsatisfied gate; never skip a gate because a later stage looks ready.
2. Do not infer missing results; do not upgrade claim tier without the named token; prefer early stop.
3. Micro-map and synthetic outputs are never evidence; micro-map curves go to human review before the full grid.
4. Track P never touches tokens; Track L launches immediately and is tracked like experiments.
5. Every report states which gate it serves, which token it justifies, and which figure/table it feeds — "none" to the last means out of scope until this plan is amended.
6. Execution history belongs in `results/` journals, not in this document; keep the plan stateless apart from the one-line status at the top.
