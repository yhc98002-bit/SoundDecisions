# Experiment Plan — `foley-cw` / Sound Decisions

**Project:** *Where Do Foley Decisions Come From? Conditioning, Seed, and Trajectory Shares in Video-to-Audio Flow Generation.* (Commitment and readout windows are the mechanism by which the trajectory share is mapped and acted on.)
**Backbone:** MMAudio rectified flow (`small_16k` primary; `large_44k` for scale insurance). **Audience:** autonomous research agents and collaborators. Decision tokens are the only legitimate way to change claim tier or unlock a stage. No calendar dates: sequencing is by gates and dependencies only.

**Status (one line, maintained per §15.6).** 2026-06-20 — **`GO_MAPS_PHASE` EMITTED** (autonomous Stage-0 freeze; `results/stage0/go_maps_phase.md`). Self-target gate (det+rob, §3.3 split): presence/timing/material PASS → **3/4 ≥ 3**; class fails rob 0.833 → **kept DIAGNOSTIC** (in maps, not gating); validity is correctness-layer only. Manifest frozen: **200 single + 60 two-event**, 60/40 split, per-axis usable/non-pinned n, timing_bin_s **1.5147** (`AUDIO_ANCHOR_NOT_ADOPTED`: human-vs-audio σ=0.969s>0.35s → approved chain), class=diagnostic (`data/manifests/phase1_manifest_frozen.json`). Human-label suite done (correctness-layer, diagnostic): validity full suite (`validity_suite.md`) — human↔qwen agree (κ pres 0.66/class 0.62) but the measurer is the class-validity weak link; timing humans tap ~1s after the t≈0 machine onset; material CLAP-vs-PANNs RSA ρ=0.49 (moderate). Kernel tokens `CFG_KERNEL_OK(cfg=1, schedule=sqrt_down)` ratified + `(cfg=4.5,…)` **CANDIDATE** (full-Phase-1-pool Gate-A required). θ_cal + frozen interpretations + certified tuple UNCHANGED. Codex: gate reproduced, no blocking. **Phase 1–3 cfg=1.0 COMPLETE → `GO_MAP` + `GO_READOUT`** (the make-or-break passed; `results/stage0/phase1/phase3_decision.md`). Tokens: `COMMITMENT_MAP_DONE` (Fig 1 three-share budget: class carries the trajectory share 0.35 + seed floor 0.23; presence/timing conditioning-bound; material commits late s=0.64), `READOUT_MAP_DONE`, separation_score 7.24 + Gate B stable → `GO_MAP`, presence/timing/material readable-early → `GO_READOUT`. Track P (Fig 4): internal features read presence/timing from s=0.05 but class never (R2). class = R2 (commits 0.35, externally readable only 0.75). **Arc 2 (PI Decisions 1/2/3) COMPLETE → PAUSED for 2 PI rulings** (`results/PI_REPORT_arc2.md`).
**ARC 3 IN PROGRESS (PI rulings applied; autonomous).** **R1:** Gate-A scaled cap RATIFIED — cfg=1.0 `CFG_KERNEL_OK` CLEAN; **cfg=4.5 `CFG_KERNEL_OK(cfg=4.5, schedule=sqrt_down)` RATIFIED with caveat** "near-exchangeable on tagger-probs, not provably exact" (ledger updated; scaled cap = Binomial(200,0.05)=15, the frozen-16→200 exposure bug-fix; n_perm=1000). Fig 1b now on certified cfg=4.5 data. **R2:** **F-1 REFUTED** (final wording: guidance narrows the reachable-outcome set via mode collapse — distinct classes 4.83→3.62 — NOT seed-migration; §1.1/§8.3/§9/§12 amended; METHOD path = B4 bridge, not F1_SUPPORTED). Tier-B pre-registered (SHA256). **ARC 3 COMPLETE** (`results/PI_REPORT_arc3.md`). Tier-B tokens: **`R2_CLASS_CONFIRMED`** (class not internally readable: pooled 0.451, per-token, conditioning 0.419 — all ≪ θ_read 0.70), **`COND_NOT_BOTTLENECK`** (conditioning carries class as well as the internals — class is hard EVERYWHERE, not a conditioning failure), **`NO_SEED_FLOOR`** (seed→class below chance, F-1 refuted firmly), **`BRIDGE_PARTIAL`** (the METHOD make-or-break → **DIAGNOSTIC-strong tier**: timing bridges 0.94, class bottleneck 0.00; mean recovery 0.514 CI straddles 0.5, joint final recovery 0.000). C two-budgets: class's rising observational conditioning-share is non-causal (cond-swap follow 0.45); Phase-4 0.370 scalar tie genuine. Material CLAP-vs-PANNs RSA 0.494. Figs 1/1b/2/5/6 rendered. 1023 tests green. **Deferred:** large_44k (proxy download stalled; scale-insurance, not story-critical). **Headline (final): F-1 refuted → guidance narrows the reachable outcome set (entropy reduction), proven causally; the method is DIAGNOSTIC-strong, bounded by class readout quality (poor from all representations — the key open lever).** No pause trigger across Arc 3.

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

### 1.1 Scientific object — the three-share decomposition

> **AMENDED 2026-06-20 (Arc-3): F-1 is REFUTED.** The original "working thesis" below — that at
> deployed CFG the decision migrates into the seed — did NOT survive. The dial + condition-swap +
> distinct-class-count show that increasing guidance **narrows the set of reachable outcomes**
> (mode collapse: distinct classes among independents 4.83→3.62 over cfg 1.0→4.5), which raises
> apparent agreement WITHOUT making the decision either video-steered (condition-swap: class
> follow-rate 0.45 ≈ chance) or more seed-determined (dial: seed→class grip flat/down). So the
> "conditioning share grows with cfg" seen in the observational budget is an **entropy-reduction
> artifact** (A_independent rises mechanically as diversity falls), NOT real video-determination.
> The observational determination budget is kept; the **causal conditioning responsiveness**
> (condition-swap follow/retention) is reported beside it (§C), with the distinct-class count as the
> explainer. The METHOD path is the B4 bridge (§12), not F1_SUPPORTED. §8.3/§9/§12 are superseded
> accordingly.

Every audio-visual generative decision is apportioned among three sources, and the central object of the paper is this apportionment per perceptual axis: the **conditioning share** (what the video forces, `A_independent`), the **seed share** (what the initial noise fixes before dynamics, `A_fork(s_min) − A_independent`), and the **trajectory share** (what the sampling path resolves, `A_fork(s) − A_fork(s_min)`), with a residual. The decomposition is **guidance-dependent** (~~the working thesis (F-1, §8.3) is that at deployed CFG the decision migrates out of the trajectory and into the seed~~ — REFUTED, see the amendment above: guidance narrows the outcome set). Commitment and readout windows are the mechanism by which the **trajectory share** is mapped (when the path locks a value) and made actionable (when a probe can read it); the seed share is mapped by the determination budget and the F-1 protocol.

The maps target the **model's own final axis value (self-target)**, never human correctness-vs-video; correctness enters only in Phase 4 via the sidecar. They are label-free but measurement-dependent — the reliability gate (§3.3) is load-bearing. The policy is licensed by the **correctness factorization**: `correctness(axis) = match(readable axis value, video anchor)`, time-independent once the value is readable. Axes whose correctness does not factor through a readable value (holistic quality) are out of scope for gated action.

### 1.2 CFG doctrine
The velocity→score identity is exact only at `cfg = 1.0`; at `cfg > 1` the CFG-mixed velocity yields a tilted pseudo-score with no exact marginal-preservation guarantee, and no constant-α fork operating point exists at deployed guidance. Operating points:

- **Headline `cfg = 1.0`** (exact kernel): commitment maps, readout maps for the gap, the α surface, separation tests, Gate B.
- **Deployed `cfg = 4.5`**: readout-transfer (probes on `x̂0(s)` need no fork kernel), the F-1 protocol (§8.3), condition-swap, and Phase-4 policy. Commitment maps at 4.5 only after re-entry.
- **Re-entry route:** a (cfg, g(s)-schedule) tuple that passes Gate A emits `CFG_KERNEL_OK(cfg=x, schedule=g)` — **the token MUST carry the `schedule=` suffix and the per-(cfg, schedule) provenance MUST be asserted in code**, so a commitment grid is never run under a schedule different from the one that certified its kernel. `cfg = 2.5` is the schedule-pilot candidate.
- **`cfg = 4.5` is an ADDITIONAL headline, not a replacement.** `cfg = 1.0` remains the kernel-exact backbone for commitment + readout + gap. When `CFG_KERNEL_OK(cfg=4.5, schedule=sqrt_down)` is confirmed **on the full Phase-1 independent pool** (not only the pilot/precheck cells), the deployed-cfg commitment grid runs as a second headline arm, and the cfg=1.0↔4.5 **contrast** (share migration) becomes a primary result — especially important because F-1 predicts 4.5 is where the trajectory share may partially collapse into the seed.

**Gate A — kernel validity (HARD, per cfg × schedule × model).** Required before any commitment map at that tuple: (i) small-α continuity; (ii) fork-audio validity; (iii) nontrivial diversity; (iv) **seed-marginalized exchangeability** — fork one tail from each of the N independents at s, pool, permutation-test against fresh independents on tagger-probability-vector MMD + label-marginal TV. Per-seed-cell embedding tests are forbidden: the seed legitimately fingerprints fine texture, so they conflate seed-conditioning with kernel error. Commitment measurement itself keeps K forks per seed — per-seed posterior concentration *is* the signal. Guards never pass silently.

**Gate B — α-ordering stability (HARD, at the headline cfg).** The axis ordering of `s_commit` must be stable across the α pilot grid; α is an instrument parameter, so ordering instability across α indicts the measurement, not the model.

**Science, not gate — cross-cfg behavior.** Pre-registered predictions: `s_commit` shifts earlier as cfg increases; inter-axis separation may compress; ordering changes carry no directional bet. The F-1 predictions (§8.3) extend these to the limit where commitment collapses onto the initial noise.

### 1.3 α and noise schedule
- Primary α is **certified per `(cfg, schedule)` tuple** (smallest α with measurable tail diversity + valid fork audio). For the ratified backbone `(cfg = 1.0, schedule = sqrt_down)`, production **α = 0.8**. Do not use any legacy α (e.g. the old constant-schedule 1.6) for a certified schedule unless an explicit amendment records the change. Full `A(axis, s, α)` surface reported as secondary evidence.
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

## 2. Stage M — Micro-Map Checkpoint (PASSED; diagnostics, never evidence)

Shakes down the full chain (generate → decode → measure → agree → normalize) at toy scale before any large grid.

**Spec:** 16 FoleyBench Single-source+Discrete clips, registered as exclusions from the Phase-1 manifest. Axes: presence + coarse class (event-restricted argmax + abstain margin δ, §3.3), with the non-gating entropy/margin instruments logged on the same runs. Grid: `s ∈ {0.05, 0.30, 0.60, 0.90}`, K = 8, `cfg ∈ {1.0, 4.5}`, `N_independent = 8` per clip per cfg; Gate A additionally takes one fork per independent at its test s-points. The 4.5 arm serves Gate-A adjudication and the g(s)-schedule pilots — it is not a pass requirement. Budget ≈ 1k full-generation equivalents (≈ 20 GPU-min); logging contract exercised end-to-end.

**Pass criteria (human review before the full grid):**
1. **Endpoints, granularity-aware (seed-floor-consistent).** *Late, label level (confident subset):* `A_fork(0.90) ≥ 0.90`, abstain rate reported. *Early, label level — washout DIRECTION, not a fixed band:* the early gap `g₀ = A_fork(0.05) − A_independent` must be (i) **non-negative** (a floor, not anti-correlation), (ii) **Gate-A exchangeable at s = 0.05** (the kernel is marginally valid there — already the operative test), and (iii) **bounded, `g₀ ≤ 0.25`** — a floor that is present but not dominant. This replaces the old `≤ 0.10` distance-to-conditioning-floor band, which contradicted §4's treatment of the seed floor as a first-class positive quantity. **Falsifiability guard (why this is not circular):** the re-scoped rule still fails if `g₀ > 0.25` (the model is near-deterministic from noise — no trajectory phase to map) or if Gate-A rejects at s = 0.05 (marginal invalidity); the measured `g₀` is logged as *candidate* Fig-1 seed-share content, with the certified seed-share number established only in Phase 1 at scale. *Embedding level:* a stable, CI-bounded seed floor `A_fork_emb(0.05)` with `A_fork_emb(0.90) > A_fork_emb(0.05)` beyond CIs.
2. **Monotonicity:** `commit(s)` non-decreasing within CI tolerance (label level, confident subset).
3. **Kernel:** Gate A passes at `cfg = 1.0` (hard). At 4.5, Gate A is adjudicated and reported; a 4.5 failure routes to §1.2 fallback, not a halt.
4. **Measurer determinism** = 1.0 on identical wavs.
5. **Informativeness:** < 12/16 clips video-pinned (`A_independent > 0.9`) on class, else widen/re-stratify the pool; abstains ≤ 30% of fork-final labels at s = 0.90, else revisit δ or trigger the BEATs contingency.

**Tokens:** `MICROMAP_PASS` / `MICROMAP_FAIL(reason)` / `CFG_KERNEL_OK|FAIL(cfg=x[, schedule=g])`.
**Routing:** late-endpoint failure on the confident subset → genuinely suspect kernel/terminal-time numerics; early-endpoint label failure → normalization or `A_independent` estimation; Gate-A failure at `cfg = 1.0` → STOP-level instrument review.

---

## 3. Stage 0 — Manifest, Anchors, Reliability → `GO_MAPS_PHASE`

### 3.1 Manifest with screening
~400 FoleyBench single-event candidates → `N = 8` independents per clip **at each headline cfg** (1.0 backbone, and 4.5 once its full-pool Gate-A confirms; exclusions and the video-pinned registry are cfg-specific because `A_independent` differs by cfg) → per-axis `A_independent` → stratified **200 single-event + 60 two-event** clips (two-event from the FoleyBench Multi-source pool; VGGSound fallback). Clips with `A_independent > 0.9` on an axis are excluded from that axis's normalized curve and logged in the **video-determined registry** (still usable for other axes); seed-vs-trajectory classification is a Phase-1 output, not a screening output. A 60-clip `cfg = 4.5` sub-screen is logged for Phase-4 pools and F-1, non-gating. Frozen in the manifest: ~12 event-restricted coarse classes, 8 s clip duration, 60/40 probe-train/eval split by clip, per-axis usable n, anchor source per clip. **Headroom stratification (required):** presence/timing are conditioning-pinned on 51%/72% of clips, so the 200-clip manifest must guarantee a sufficient *non-pinned* (`A_independent < 0.9`) subset per axis — especially material and class, which carry the seed/trajectory-share story; report per-axis non-pinned n beside usable n.

### 3.2 Event anchors
FoleyBench ships no event timestamps, so **original-audio-track onsets (spectral flux) are the primary anchor source**, visual onset detector as cross-check, `σ_anchor` = cross-source disagreement, light human marks where both fail. The **30-clip manual check set** has two jobs: report MAE/coverage, and arbitrate whether validated audio-only anchors beat the current disagreement (median 0.757 s → 1.51 s bins, ≈ 5 bins per clip — marginal for timing power; if audio-only `σ_anchor ≤ 0.35 s`, adopt audio-only and re-derive bins). **Propagation rule:** timing bins ≥ `2·σ_anchor`. Anchor uncertainty is the quiet failure mode of the timing axis.

### 3.3 Reliability gate (per axis, on generated audio)
- **Class:** tagger argmax restricted to event classes (speech/music/ambient excluded; frozen coarse map) with abstain margin δ; agreement on the confident subset, abstain rate always reported. Primary tagger PANNs-16k with a **BEATs swap contingency** if the sanity score is not well clear of 0.65; second tagger for cross-tagger agreement; qwen judge triangulates at micro-map scale. Non-gating parallel instruments wherever class is measured: fork-ensemble label entropy / top-prob concentration, and top-1 margin trajectories `margin(s)`. **Class is a diagnostic/NEGATIVE-layer axis:** it fails the correctness gate (robustness 0.833 < 0.85; validity 0.459 de-confounded) but its self-target is reliable (determinism = 1.0), so it stays in the determination budget and carries the seed-floor (g₀) and F-1 mode-locking results; **do not spend MLLM budget re-eliciting its labels** — robustness fails independently, so a re-run cannot rescue it.
- **Other axes:** presence = energy gate + tagger eventness; onset = spectral flux vs. anchor bins; material/timbre = CLAP/BEATs embedding cosine (normalized against the seed floor, §4.2), **validity via a second embedder** (cross-embedder CLAP-vs-PANNs cosine consistency) since it has no MLLM-judgeable gold and — being ~0% video-pinned (largest seed/trajectory headroom) — is the axis most likely to carry the trajectory-share window story, so it is validated, not relegated.
- **MLLM probes:** temperature 0, versioned prompts, response cache, test-retest subset.
- **Thresholds (frozen), gate SPLIT BY CLAIM LAYER.** *Self-target gate = determinism + robustness* — gates the determination-budget / commitment / readout analysis (Phases 1–3, the §1.1 object): determinism `θ_rel ≥ 0.95`; robustness `θ_robust ≥ 0.85` (class) / ±1 bin (timing) under event-window shift, loudness norm, resampling, light compression, small noise. **≥ 3 axes must pass THIS gate for `GO_MAPS_PHASE`.** *Correctness gate = validity* — gates only the correctness sidecar + Phase-4 policy: validity ≥ 0.6 vs ~100 MLLM + ~50 human clips, **human sidecar is the arbiter**. Rationale: the maps depend on self-target reliability (determinism = 1.0 on all four axes), not on perceptual validity, which §1.1 needs only where correctness claims live. **Validity statistic:** report the full suite — raw agreement, marginals, confusion matrix, Cohen κ, Gwet AC1, PABAK — never a lone chance-corrected number (under the ~90% skew here κ is too harsh and AC1 too lenient; the confusion matrix is the truth). **The human validity sample MUST oversample minority cases** (gold-absent, off-bin-0 timing, varied classes); on conditioning-pinned screening marginals every statistic is uninformative, so the bundle is rebuilt stratified before the human pass. A self-target failure → `AXIS_DEMOTED:<axis>`; a validity-only failure demotes to diagnostic/correctness-excluded but **keeps the axis in the maps**.
- Pre-register here: cfg-sweep predictions (§1.2), F-1 predictions (§8.3), and the `GO_MAP` gate language (§6).

**`GO_MAPS_PHASE` preconditions (all):** `MICROMAP_PASS`; Gate A passed at the headline cfg; logging contract exercised; manifest + anchors + reliability complete; **≥ 3 axes pass the self-target gate (determinism + robustness)** — validity is a correctness-layer gate (§3.3), not a `GO_MAPS_PHASE` precondition; splits and thresholds frozen.

---

## 4. Phase 1 — Commitment Maps

**Grid:** 200 single-event clips (+ 60 two-event for binding); `s ∈ {0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90}`; α = 0.8 under `sqrt_down`, K = 12; `N_independent = 16`. cfg: **two headline arms — `cfg = 1.0` (kernel-exact backbone) and `cfg = 4.5` under `sqrt_down`** once its full-pool Gate-A is confirmed (§1.2); the cfg=1.0↔4.5 contrast is a primary deliverable. `cfg = 2.5` coarse grid contingent on its own Gate-A pass. Secondary α × 2 at 4 s-points, K = 6, per headline cfg. **Screening and the video-pinned exclusion registry are cfg-specific** — run `A_independent` screening at each headline cfg. Envelope ≈ 40k full-generation equivalents across both arms — a small number of node-days, trivially parallel by clip.

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

### 8.3 cfg-dial + F-1

> **AMENDED 2026-06-20 (Arc-3): F-1 REFUTED (see §1.1).** The F-1 protocol ran (dial 24×6 + condition-swap); the seed-migration prediction is refuted and replaced by the guidance→entropy-reduction mechanism. `F1_SUPPORTED` is no longer a METHOD requirement (§12). The cfg-dial and condition-swap remain as the causal instruments; tokens F1_REFUTED (recorded). The text below is retained for provenance only.
 ("guidance moves the decision into the seed")
- **cfg-dial:** evaluate the §1.2 predictions on every cfg with a certified kernel — shift direction/magnitude, separation compression, ordering changes, speciation framing. This makes "cross-modal" a manipulated variable.
- **F-1 — share migration under guidance (now a CENTRAL result, not a bolt-on; promoted because the seed floor is the paper's lead quantity, §1.1).** Motivated by the mode-locking diagnostic (same seed → same class at every piloted α; different seeds → different classes) and the Run-3 label-space seed floor. Two parts. **(a) The cfg=1.0↔4.5 contrast** is read directly off the two headline commitment grids (§1.2): the seed/trajectory share split per axis at each cfg, with the migration as the headline. **(b) The dial**, instrument-light (no marginal-preserving kernel needed at high cfg): ~24 clips × `cfg ∈ {1.0, 1.5, 2.0, 2.5, 3.0, 4.5}`, measuring seed-predictability of class (earliest-s fork agreement + a small (noise, video) → class probe) and the minimum unlocking α* per cfg. **Pre-registered predictions:** α*(cfg) increasing; commitment curves shift left and degenerate toward step functions at the seed; the **seed share grows monotonically with cfg**. **Tokens:** `F1_SUPPORTED` / `F1_REFUTED` / `F1_INCONCLUSIVE`. If supported, the conclusion is *guidance changes who decides — trajectory at low cfg, seed at deployed cfg* — connecting to multimodal-speciation theory and explaining why seed-selection methods help precisely in high-guidance regimes; Phase 4 becomes guidance-aware. Stage-M diagnostics are never themselves cited as F-1 evidence.

---

## 9. Phase 4 — Axis-Gated Population Pruning (conditional on `GO_MAP + GO_READOUT`)

> **AMENDED 2026-06-20 (Arc-3).** The METHOD make-or-break is the **B4 oracle→non-oracle bridge** (offline, this section's machinery): how much of the oracle axis-gated headroom a realistic non-oracle scorer recovers per axis. The 'guidance-aware framing contingent on F1_SUPPORTED' below is superseded — F-1 is refuted; the policy comparison is reported per cfg regime descriptively (guidance narrows outcomes, it does not relocate the decision into the seed). Online `GO_POLICY` with human gold is a later PI decision.


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
| **Fig. 1 (lead)** | **Three-share determination budget** (conditioning / seed / trajectory-by-s / residual) per axis × cfg — the paper's headline object; the taxonomy is its discretization | Phase 1 |
| **Fig. 1b (lead)** | **Share migration: cfg=1.0 ↔ 4.5 contrast** — seed/trajectory split per axis at each guidance level | Phase 1, §8.3 |
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

**AMENDED 2026-06-20 (Arc-3, supersedes the F1_SUPPORTED requirement).** F-1 is **REFUTED**:
increasing guidance reduces the diversity of reachable class outcomes (distinct classes among
independents 4.83→3.62 cfg 1.0→4.5), which raises apparent agreement WITHOUT making class either
video-steered or increasingly seed-determined — guidance **narrows the outcome set** rather than
moving the decision into the seed. The METHOD tier therefore no longer requires `F1_SUPPORTED`; the
METHOD path is the **B4 oracle→non-oracle bridge** (a realistic non-oracle scorer recovers a
substantial fraction of the oracle axis-gated headroom).

| Tier | Required evidence (amended) |
|---|---|
| **METHOD (full)** | The three-share decomposition (Fig. 1) + `GO_MAP + GO_READOUT`, **plus** the **B4 bridge**: an offline non-oracle axis-gated policy recovering ≥ 0.5 of the oracle headroom per axis vs the strongest matched-compute baseline (incl. SMC-style); the online `GO_POLICY` run with human gold is a separate later step |
| **DIAGNOSTIC (strong)** | The three-share decomposition + the guidance→entropy-reduction mechanism established (the headline stands on its own), with either an R2-dominated readout gap (internal probes close what external probes cannot) or a bridge recovering only partial headroom; the measurement + the entropy-reduction finding carry the paper |
| **NEGATIVE (publishable)** | `STOP_ADSR`: trajectory-share windows coincide or sit near s = 1 — but the conditioning/seed/trajectory decomposition and its guidance dependence remain a publishable characterization of where V2A decisions come from |
| **STOP** | Reliability/feasibility failure (`STOP_PROJECT`) |

The paper's identity is now the **three-share decomposition and its guidance dependence**; the commitment/readout windows and the policy are how the trajectory share is mapped and exploited. Even the NEGATIVE tier ships the headline, because the decomposition does not depend on the windows separating.

---

## 13. Tooling & Long Poles

**Agent-built labeling tool (standard task).** A web app — audio playback, 12-class forced choice, presence toggle, onset-tap, writing JSONL — that serves every human pass from one build: the ~50-clip validity sidecar, the ~100-clip MLLM-validity set, and Phase-4 correctness labeling. It records human and qwen labels on the same clips in one pass, so the validity check (§3.3) and the human readout-ceiling anchor come from one effort. **Rebuild the validity bundle on the stratified minority-oversampled sample and make both bundles delivery-sized (16k mono audio; split the anchor video bundle) before the human pass** — the first build sat on a degenerate sample and was too large to deliver.

**Genuine long poles (human-time — the only items that gate silently).** The ~50-clip validity labeling and the 30-clip anchor marks (the latter also arbitrates audio-only anchor sharpness, §3.2). Queue both as soon as the tool is ready; manifest freeze / `GO_MAPS_PHASE` cannot complete without them.

**Continuous agent bookkeeping.** Two-event subset curation (60 clips, FoleyBench Multi-source pool); MLLM prompt versioning + caching within a ≈ 5–6k total call budget; storage monitoring against the 100 GB cap.

---

## 14. Decision-Token Registry

| Token | Emitted by | Meaning |
|---|---|---|
| `MICROMAP_PASS` / `MICROMAP_FAIL(reason)` | Stage M | Measurement chain healthy / failed criterion named |
| `CFG_KERNEL_OK|FAIL(cfg=x, schedule=g)` | Stage M, §1.3 pilots, §8.2 | Gate-A verdict per (cfg, schedule, model); the `schedule=` suffix is mandatory; OK is the re-entry token, ratified only on the full Phase-1 pool |
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
7. The frozen Stage-M interpretations (`experiment/preregistered/stage_m_rerun_interpretations.md`: δ = 0.05, sqrt transform, fresh refs, exact-match gating, scorability ≥ 12/16, paired CIs, g3-p95 refinement, iteration bound, tagger-sanity confident reading, tuple-selection record) and the converged instrument tuple `(cfg = 1.0, sqrt_down, α = 0.8)` are **ratified read-only provenance** — do not silently re-tune them; any change requires an explicit amendment and a recorded rationale.
8. `CFG_KERNEL_OK` tokens are invalid without a `schedule=` suffix; a commitment grid may only run under the exact (cfg, schedule) its kernel certification used. The cfg=4.5 re-entry is promoted from candidate to ratified only after its Gate-A passes on the **full Phase-1 independent pool**, not the pilot cells.