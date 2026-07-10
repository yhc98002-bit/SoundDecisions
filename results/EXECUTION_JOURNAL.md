
## 2026-06-13 — Slice 3 (Stage-M release → Stage-0)

- **T1 Stage-M re-scope → MICROMAP_PASS** (0 GPU): June-13 §2.1 washout-direction
  rule (amendment #12); class g0=0.1526 passes (>=−0.02, <=0.25, Gate-A ok@s0.05).
  Tokens: MICROMAP_PASS, CFG_KERNEL_OK(cfg=1, schedule=sqrt_down),
  CFG_KERNEL_OK(cfg=4.5, schedule=sqrt_down) [CANDIDATE]. certified_kernels.json +
  foley_cw/kernel_provenance.py guard. Codex: 1 High + 3 lesser fixed. 852 tests green.
- **T2 labeling tool**: foley_cw/labeling_tool.py static-bundle generator;
  anchor_bundle.html (30 clips) DELIVERED; validity bundle pending finals.
- **T5**: two-event subset (60, Multi-source+Discrete); pre-registrations re-frozen
  from June-13 manual (F-1 promoted to headline).
- **T3 screening**: cfg=4.5 sub-screen 60/60 done (registry: presence 36, timing 37,
  class 18, material 9 video-pinned). cfg=1.0: first pass 297/400, 4 shards crashed
  on a CLIP frame-count off-by-one (63 vs 64) on a real FoleyBench clip → fixed in
  mmaudio_backend._fit_frames (pad-by-repeat/truncate; test_fit_frames green) →
  resumed (journals make it idempotent).
  **Correction (frame fix):** input-level `_fit_frames` fixed CLIP but the sync encoder
  DOWNSAMPLES, so a 184-vs-192 sync mismatch resurfaced; superseded by FEATURE-level
  `_fit_seq` (pad-by-repeat-last/truncate at dim=1 AFTER encoding) applied to both
  clip_features(→64) and sync_features(→192). All 400 cfg=1.0 + 60 cfg=4.5 finals
  completed, 0 errors. A_independent (cfg=1.0): presence mean 0.809 / 203-of-400
  video-pinned, timing 0.899 / 290, class abstain-heavy / 51 scorable+pinned, material
  0.646 / 3. cfg-specific aggregates written (tags screen / subscreen45).

- **T4 sidecar + reliability gate + κ-paradox diagnostic** (login CPU; 0 GPU, 0 new MLLM
  calls): MLLM sidecar = 100 finals × {presence,class,timing} (+20 retest: pres/class
  1.00, timing 0.95). Frozen gate (Cohen κ vs MLLM gold, seed-0 50-clip subsample,
  θ_cal=0.6) → ALL FOUR axes demoted, 0 survivors. **Diagnosed as the Cohen-κ paradox,
  not invalidity:** presence/timing raw agree 0.75/0.80 but ~90%-skewed marginals →
  κ −0.125/−0.028; skew-robust **Gwet AC1 = 0.674 / 0.793 (≥0.6)**. class is the genuine
  casualty (rob 0.833<0.85 AND AC1 0.459 even de-confounded). **Decision: did NOT spend
  the remaining qwen budget (431/500) re-eliciting the 45 label-space-confounded class
  labels** — class fails robustness independently, so a re-run cannot change the gate;
  θ_cal and frozen interpretations UNCHANGED. New: `foley_cw.sidecar.gwet_ac1` (+12 tests)
  and read-only `scripts/stage0_reliability_diagnostic.py` (re-measures on-disk finals,
  reproduces the gate's exact κ — pres −0.124567 / timing −0.028278 / class 0.145121 all
  MATCH — then reports AC1 + class confound decomposition + per-clip audit triples) →
  `results/stage0/reliability_diagnostic.{md,json}`. **Codex review: no BLOCKING; AC1
  formula confirmed; "AC1 as diagnostic, not a silent gate swap" endorsed; 3 SHOULD + 1
  NIT all applied** (soften "prove"→statistic-level + emit audit pairs; note AC1 q=union
  is a diagnostic choice; relabel both-confident as favorable upper bound; fix stale
  --n-clips help). 864 tests green.

## 2026-06-20 — ARC 3 (PI rulings applied; longest-stretch autonomous)

Directive supersedes the stale F-1-centric plan sections; core principle: story-changing results
UPDATE the narrative and CONTINUE, pause only on 5 named triggers. Pre-register every NEW analysis
with SHA256 (done: `arc3_tierB_preregistration.md`, sha 7678f89e…).

- **RULING 1 — Gate-A scaled cap RATIFIED.** Implemented `scaled_cap(n)=Binomial(n,0.05) 95pct`
  (=2 at n=16 recovering the frozen value; =15 at n=200) in `gate_a_fullpool_eval.py` — the
  frozen-16→200 exposure bug-fix, NOT a re-tune (frozen constants untouched; cap passed explicitly).
  Re-ran at **n_perm=1000**: cfg=1.0 `CFG_KERNEL_OK` CLEAN (low-p 11,9 ≤ cap 15); **cfg=4.5
  `CFG_KERNEL_OK(cfg=4.5, schedule=sqrt_down)` RATIFIED WITH CAVEAT** (MMD exceedance 15 = cap 15
  even at n_perm=1000 → "near-exchangeable on tagger-probs, not provably exact"; MW all pass, TV
  clears). **Ledger updated** (deployed.ratified=true + caveat). Fig 1b now rests on certified
  cfg=4.5 data → the entropy-reduction reframing is on certified ground.
- **RULING 2 — F-1 REFUTED, final wording adopted:** "Increasing guidance reduces the diversity of
  reachable class outcomes; this raises apparent agreement WITHOUT making class either video-steered
  or increasingly seed-determined — guidance narrows the outcome set rather than moving the decision
  into the seed." Plan §1.1/§8.3/§9/§12 amended (METHOD path = B4 bridge, not F1_SUPPORTED).
- **Tier-B COMPLETE (core)** (workflow build+run+adversarial-verify; report `PI_REPORT_arc3.md`):
  - **B1 `R2_CLASS_CONFIRMED`**: class not internally readable (MLP best 0.451 vs chance 0.319,
    never reaches θ_read=0.70); leakage-clean. Per-token/cross-attn GPU probe in flight (confirmatory).
  - **B3 `NO_SEED_FLOOR`**: seed→class (reduced-dim 256) acc 0.184 < chance 0.272 at cfg=1.0, slope
    CI∋0 → F-1 stays refuted, firmly.
  - **B4 `BRIDGE_PARTIAL` (make-or-break → DIAGNOSTIC-strong).** Adversarial verify CAUGHT the first
    result as an artifact (multiclass floor inflated survivor purity with K; empty-mask fallback
    leaked true labels). FIXED: symmetric keep-flip calibrated so keep-accuracy=readout quality
    (K-invariant); fallback→final_score (no leak); oracle 0.785 unchanged. Honest per-axis recovery:
    timing 0.94, presence 0.56, material 0.56, **class 0.00 (bottleneck, readout 0.345, no internal
    head)**; mean 0.514 CI[0.355,0.648] straddles 0.5, overall joint recovery 0.000 → BRIDGE_PARTIAL.
    Token made CI-aware (BRIDGE_METHOD needs CI-lo≥0.5; conservative, not a re-tune).
  - **C** two-budgets: class apparent cond-share rises 0.378→0.508 but causal cond-swap follow=0.45
    (FAILS) → not video-driven; entropy lens explains. Phase-4 0.370 scalar tie is GENUINE (scalar
    reward corr 0.179 w/ correctness) — argues FOR axis-gating.
  - Material CLAP-vs-PANNs RSA 0.494 [0.41,0.64]. Figures 1/1b/2/5/6 rendered.
  - 1023 tests green (hardened an order-fragile scipy-import guard to a clean-subprocess check).
- **Confirmatory GPU runs LANDED** (after fixing 2 workflow-script infra bugs — RunStore subdir
  allowlist + B2's np.savez `.npz`-suffix tmp-rename):
  - **B2 `COND_NOT_BOTTLENECK`** (genuinely-new token): raw video-conditioning (CLIP+Synchformer)
    predicts class at 0.419 vs chance 0.270 — essentially the SAME as the DiT-internal best (0.4375).
    Class non-readability is NOT a conditioning bottleneck; class is hard to read linearly
    EVERYWHERE (conditioning ≈ internals ≈ external preview, all ~0.42–0.44 ≪ θ_read 0.70). This
    STRENGTHENS the entropy-reduction story: class is dynamics-determined but not linearly encoded
    in any accessible representation. `best_cond_probe=clip_f/ridge`.
  - **B1 per-token** (families 3-4, token-mean-max + cross-attn): 25600 npz collected, but the quick
    probe over them was impractical as written (float16 max-pool overflow + 25600-file I/O) — NO
    trustworthy per-token number obtained (do not claim one). R2_CLASS_CONFIRMED stands firmly on
    families 1-2 (pooled 0.451) + B2 conditioning (0.419) + B3 seed (below chance) — class not
    linearly readable from any representation; per-token cannot plausibly flip it. Proper per-token
    probe (float32 + streamed) is a clean follow-up on the collected features.
- **Deferred:** B3 full-pool cfg=1.0 (dial-scale NO_SEED_FLOOR already firm); **large_44k** —
  3.9 GB checkpoint fetched 0 bytes in 15+ min over the login proxy (stalled); scale-insurance whose
  absence is NOT the catastrophic-contradiction pause trigger. Noted as deferred-pending-weights.
- **ARC 3 COMPLETE.** Tokens: `CFG_KERNEL_OK(cfg=1)` clean + `CFG_KERNEL_OK(cfg=4.5)` ratified+caveat,
  `F1_REFUTED`, `R2_CLASS_CONFIRMED`, `COND_NOT_BOTTLENECK`, `NO_SEED_FLOOR`, `BRIDGE_PARTIAL`
  (→ DIAGNOSTIC-strong tier). Headline: F-1 refuted → guidance narrows the reachable outcome set
  (entropy reduction), proven causally; method DIAGNOSTIC-strong, bounded by class readout quality
  (which is poor from ALL representations — the key open lever). No pause trigger across the arc.

## 2026-06-20 — F-1 RESOLVED (entropy-reduction) + Phase-4 → PAUSE for 2 PI rulings [Arc-2, RESOLVED by PI Arc-3]

Consolidated report: `results/PI_REPORT_arc2.md`. Two pause points (both PI-named): the final F-1
narrative call, and the Gate-A threshold-scaling ruling.

- **F-1 disambiguation COMPLETE (Decision 1) — suggested `F1_REFUTED`, reframed as entropy
  reduction.** Three converging measures: (a) **condition-swap (clean causal, Fig 5):** class does
  NOT follow a donor conditioning-swap (follow=0.45≈chance at s=0.05) while presence/timing/material
  do (0.85/0.90/0.80) → class is internally locked, NOT video-determined; (b) **dial (24×6):**
  seed's grip does NOT grow with cfg — noise→class probe 0.33→0.20 (underpowered), fork-agreement
  flat ~0.33, α* flat; (c) **entropy-reduction confirmed:** distinct classes among 16 independents
  drop 4.83→3.62 with cfg → mode collapse mechanically raises A_independent, inflating the budget's
  apparent "conditioning share" without real video-determination. **Verdict:** trajectory share
  collapses (✓ half of F-1) but migrates into NEITHER seed NOR video — into DEGENERACY. The
  conditioning confound is ruled out by the causal test; entropy-reduction is the finding. Cleaner
  than F-1; Fig-1 headline stands. Binding F-1 token withheld (PI's call). `cond_swap_map_cswap.csv`,
  `cfg_dial_f1.json`.
- **Decision 2/3 as recorded below.** 947 tests green. cond_swap kernel-guard fix (§8.1 exploratory →
  candidate kernel OK). No frozen quantity changed; ledger untouched. PAUSED for the 2 rulings.

## 2026-06-20 — Gate-A verdict (threshold artifact) + Phase-4 Pareto + dial/swap launched

- **Full-pool Gate-A ran → both `CFG_KERNEL_FAIL`, but both are a 16→200 cell-count THRESHOLD
  ARTIFACT, not real failures** (`gate_a_fullpool_interpretation.md`). Evidence: cfg=1.0 low-p
  11/200,9/200 (5.5%,4.5% = exact-kernel chance rate) vs the frozen cap=2 ("out of 16",
  Binomial(16,0.05)); under the SAME rule at n=200 the cap is ≈15 → PASS. cfg=4.5: Mann-Whitney
  ALL pass (MMD p=0.745,0.235; TV p≈1.0 — not stochastically larger than the null); only
  exceedance fails (MMD 15/200=7.5%, TV 6,5 vs cap=3) → under scaled cap≈15 it ratifies, MMD at
  the boundary. **Codex (MCP) reviewed: confirmed the scaling is the same pre-registered rule at
  the correct n (not a re-tune), cfg=4.5 "approximately exchangeable, gross failure ruled out,
  small non-exactness not ruled out", and that this is a PI ratification decision — do not rescale
  autonomously.** Ledger UNCHANGED (cfg=4.5 still ratified=false). **PAUSE item (Decision 2):** PI
  to ratify the per-16→per-200 threshold scaling. If yes → cfg=4.5 certified → the F-1 Fig-1b
  tension does NOT dissolve (data would be certified, not candidate-only). NB: `gate_a.py:48,50`
  fixed caps are the locus; scaled-cap is a 1-line change I'll make only on ratification.
- **Phase-4 offline policy (Fig 6) — strong, framing-agnostic (Decision 3 done):**
  `policy_pareto.csv` over 200 cfg=4.5 clips, 7 policies. oracle axis-gated pruning
  **final_corr=0.785, regret 0.175 at 49k NFE** vs all scalar baselines (full/same-compute BoN,
  random, DiffRS, SMC, rerank) clustered at **0.33–0.37, regret ~0.59** at equal-or-higher NFE.
  Axis-gated pruning (the paper's lever) beats every matched-compute scalar baseline by ~2× —
  METHOD-tier headroom (proxy-correctness, offline; licenses going online). DIAGNOSTIC_ONLY token
  (offline + proxy, not human gold).
- **F-1 dial + condition-swap LAUNCHED (Decision 1 disambiguation):** cfg_dial smoke already shows
  the seed-locking signal — fork diversity collapses to 0 at ALL α for cfg≥1.5 (vs unlocking at
  α≈0.4 for cfg=1.0): the seed locks class so hard no fork noise changes it. This points TOWARD
  the seed (opposite to the share-decomposition's conditioning reading) — the exact tension to
  resolve. cond_swap smoke: source→donor swaps produce finals cleanly (fixed an over-strict kernel
  guard — §8.1 is exploratory, candidate kernel OK). Full runs in flight (24×6 dial on an17, 40
  sources cond-swap on an29). On completion → consolidated report for the final F-1 narrative call.

## 2026-06-20 — PI Decisions 1/2/3 → resume: Gate-A ratification + Stage-R/Phase-4 modules

PI: proceed, don't reframe yet. Decision 2 first (cfg=4.5 full-pool Gate-A; re-examine Fig 1b
on certified data; report if it fails to ratify). Decision 1: F-1 dial + condition-swap TOGETHER
to disambiguate the seed/conditioning/entropy conflation. Decision 3: green-light Stage R +
Phase-4 offline (framing-agnostic compute). Pause only at: final F-1 narrative once dial+swap in,
or any bad token.

- **Gate-A full-pool collection (Decision 2)** built + launched: `scripts/gate_a_collect.py`
  (per clip: 16 fork-parent independents + 16 FRESH refs + one tail-fork/independent at s∈{0.05,
  0.90}; stores `probs_{ind,ref,gafork_s*}` npz mirroring the Stage-M contract) +
  `scripts/gate_a_fullpool_eval.py` (calibrate from cfg=1.0 internal null → adjudicate cfg=4.5;
  ratify→flip ledger ratified=true, or `CFG_KERNEL_FAIL`→pause). 1-clip GPU smoke confirmed npz
  shapes (16,527)×4. Running: cfg=1.0 on an17, cfg=4.5 on an29 (~40 min).
- **Stage-R + Phase-4 modules built via a parallel workflow** (3 agents build + 3 adversarially
  verify): `foley_cw/condition_swap.py` (§8.1 Fig 5 causal test) + `scripts/stage_r_cond_swap.py`;
  `foley_cw/cfg_dial.py` (§8.3 F-1 dial: seed-predictability + α*(cfg), suggests but never emits
  the F-1 token) + `scripts/stage_r_cfg_dial.py`; `foley_cw/policy_offline.py` (§9 Fig-6 offline
  policy sim: BoN/DiffRS/SMC/oracle-gating, matched NFE+scoring) + `scripts/phase4_policy.py` +
  `experiment/preregistered/policy_preregistration.md`. Verdicts: cfg-dial PASS, phase4-policy
  PASS (one latent-unreachable KeyError noted), condition-swap NEEDS_FIX → **FIXED** (the `neither`
  diagnostic was computed by subtraction, wrong for mixed categorical batches; now counted
  per-pair + dead branch removed + regression test). 947 tests green (+57). GPU runs for these
  await the Gate-A verdict (they share the nodes).

## 2026-06-20 — cfg=4.5 commitment + share migration (Fig 1b) → PAUSE (trigger c) [RESOLVED by PI]

cfg=4.5 commitment collection complete (200/200, 0 errors; +102400 Gate-A fork
measurements stored as labels). cfg=4.5 determination budget + **share migration Fig 1b**
(independently recomputed from raw CSVs — not an aggregation artifact):

| axis | cond 1.0→4.5 | seed 1.0→4.5 | traj 1.0→4.5 |
|---|---|---|---|
| class | 0.378→**0.508** | 0.231→**0.182** | 0.350→**0.285** |
| material | 0.637→**0.784** | 0.207→**0.119** | 0.144→**0.089** |
| presence | 0.813→0.833 | 0.099→0.083 | 0.082→0.072 |
| timing | 0.903→0.825 | 0.083→0.112 | 0.033→0.076 |

**F-1 tension (pre-registration contradiction → PAUSE trigger c/d):** F-1 predicts "the seed
share grows monotonically with cfg" and the decision "migrates into the seed." The contrast
supports the FIRST half — the **trajectory share collapses** with guidance (class −0.065,
material −0.055: the decision leaves the trajectory ✓) — but contradicts the DESTINATION:
the budget migrates into **conditioning** (class +0.13, material +0.15), and the **seed share
SHRINKS** (class −0.049, material −0.088), not grows. **Caveat (the crux):** at high cfg the
16 independents collapse toward similarity, so A_independent rises mechanically; since
conditioning≡A_independent and seed=A_fork(s_min)−A_independent, a rising A_independent
inflates conditioning and deflates seed EVEN IF the seed still determines the value. The
share decomposition cannot disambiguate seed vs conditioning here — that is exactly what the
F-1 **dial** (part b: same-seed-across-conditions test + α*(cfg)) was designed to resolve,
and it is a separate run not yet done. So this is NOT yet `F1_REFUTED`; it is a framing call
for the PI: (1) run the F-1 dial to disambiguate, (2) reframe F-1 as trajectory→conditioning
migration, or (3) treat the contrast as `F1_INCONCLUSIVE` pending the dial. **PAUSED for PI
with the consolidated report** (`results/PI_REPORT_arc1.md`). cfg=4.5 commitment data is in
hand; cfg=4.5 Gate-A ratification still needs a prob-vector collection run (labels-only were
stored). No frozen quantity changed.

## 2026-06-20 — Phase 3 MAKE-OR-BREAK: `GO_MAP` + `GO_READOUT` (cfg=1.0 headline)

`READOUT_MAP_DONE` (200/200, audio-tagger on x̂0(s), s_read: timing 0.05, presence 0.35,
material 0.60, class 0.75). **Phase 3 (§6) → `GO_MAP` + `GO_READOUT`** (`phase3_decision.md`,
Tab 2):
- **Separation**: self-target axes order timing<presence<material with non-overlapping CIs;
  **separation_score = 7.24** (windows separated at 7× the CI width). Gate B stable. → GO_MAP.
- **Readout**: presence/timing/material committed AND readable before s=1 → GO_READOUT.
- gap = s_read − s_commit: timing −0.06, material −0.04 (readable ~at commitment), presence
  +0.14, **class +0.40** — class commits at 0.35 but is externally readable only at 0.75 and
  (Track P) NEVER linearly readable internally: the **R2 axis** (decision made in the
  trajectory, hard to read early) — motivates Phase 5 feature-head and a class-blind policy.

**Scientific arc through the make-or-break is COMPLETE and positive (METHOD-trajectory):**
Fig 1 three-share budget (class carries trajectory share + seed floor; presence/timing
conditioning-bound; material late), Fig 2 surfaces, Fig 4 internal-vs-external ("generator
knows before the audio shows" for presence/timing), Tab 1 reliability, Tab 2 separation.
Tokens so far: `GO_MAPS_PHASE, CFG_KERNEL_OK(cfg=1,…), COMMITMENT_MAP_DONE, READOUT_MAP_DONE,
GO_MAP, GO_READOUT` (`arc_tokens.json`). `GO_MAP`→unlocks Stage R; `GO_MAP+GO_READOUT`→
unlocks Phase 4. No pause trigger.

**Remaining (conditional, net-new builds):** cfg=4.5 full-pool Gate-A → ratify or
`CFG_KERNEL_FAIL`(pause) → cfg=4.5 commitment arm + Fig 1b share-migration; Stage R
(large_44k scale, cfg-dial + F-1, condition-swap Fig 5); Phase 4 offline policy (Fig 6).

## 2026-06-20 — Gate B PASS + Track P (Fig 4) + Phase 2 in flight

- **Gate B (α-ordering stability): PASS.** Second-α grid α=0.4/K=6 (200 clips, 0 errors)
  → s_commit ordering timing<presence<material is IDENTICAL at α=0.8 and α=0.4 (magnitudes
  shift earlier at lower α, as expected; ordering stable). With the non-overlapping α=0.8
  CIs (timing[0.08,0.16]<presence[0.17,0.27]<material[0.62,0.66]), **separation + Gate B
  → GO_MAP conditions met** (pending Phase-3 emission).
- **Track P (Fig 4) — "the generator knows before the audio shows":** internal pooled
  features linearly predict the final self-target from **s=0.05** for presence (acc→0.87)
  and timing (0.93 flat) — s_read_internal=0.05 ≪ their s_commit. **class is NEVER linearly
  readable internally** (max 0.44 < θ_read 0.7) despite carrying the largest trajectory
  share — committed by the dynamics but not pre-encoded linearly (R2; flags Phase-5
  feature-head richness). `internal_probe_report.md` + `track_p_p1cfg1.json`. NON-GATING.
- **Phase 2 readout LAUNCHED** (audio-tagger probe on x̂0(s) previews, 200 clips × 4
  subjects, both nodes). On completion: aggregate → s_read → READOUT_MAP_DONE → Phase 3
  decision (GO_MAP/GO_READOUT). 1-clip smoke s_read: timing 0.05, presence 0.35, material
  0.6, class 0.9.

## 2026-06-20 — Phase 1 cfg=1.0 DONE → `COMMITMENT_MAP_DONE` (Fig 1 lead result)

200/200 clips, 16 shards, 0 errors. Three-share determination budget (bootstrap by video,
tight CIs) — **the paper's lead Fig 1, now on real evidence (not Stage-M diagnostic):**

| axis | conditioning | seed | trajectory | s_commit [CI] |
|---|---|---|---|---|
| presence | 0.81 | 0.10 | 0.08 | 0.21 [0.17,0.27] |
| timing | 0.90 | 0.08 | 0.03 | 0.11 [0.08,0.16] |
| class | 0.38 | **0.23** | **0.35** | 0.35 [0.31,0.39] |
| material(emb) | 0.64 | 0.21 | 0.14 | **0.64** [0.62,0.66] |

Reading: presence/timing are CONDITIONING-dominated (video fixes them); **class carries the
largest trajectory share (0.35) + a real seed floor (0.23)** — the dynamics resolve class,
connecting to the Stage-M g₀ and the F-1 mode-locking story; material commits LATEST
(s_commit 0.64), its ~36% seed+trajectory headroom carrying the readout-window story.
Taxonomy: class 41 seed-det / 107 traj-early / 29 traj-late; material 116 traj-mid / 78
traj-late. s_commit ordering timing<presence<class<material with mostly non-overlapping CIs
→ promising separation for GO_MAP (Gate B needs the secondary-α run). Fig 1 + Fig 2 PNGs
written (`results/figures/`). `results/stage0/phase1/{commitment_map,determination_budget,
commitment_surface,taxonomy_report}_p1cfg1.*`. Track P running. No pause trigger.

## 2026-06-20 — Phase 1 (commitment maps) IN FLIGHT

- **Built & verified (CPU smoke + unit tests):** `scripts/phase1_commitment.py` (sharded
  fork-grid runner: N=16 independents + 1 base + K=12 forks/s at the certified tuple
  cfg=1.0/sqrt_down/α=0.8; kernel-provenance guard refuses cfg=4.5 until ratified),
  `foley_cw/determination.py` (three-share budget + Fig-1 taxonomy, +8 tests),
  `scripts/phase1_aggregate.py`, `scripts/make_figures.py` (Fig 1/1b/2; matplotlib
  installed), `foley_cw/internal_probes.py` + `scripts/track_p.py` (Track P linear probes,
  +8 tests). GPU smoke (an29, 1 clip): generate→fork→measure→store→journal OK, exit 0;
  full aggregate→budget→taxonomy→figures pipeline confirmed.
- **LAUNCHED cfg=1.0 commitment grid:** 200 single-event clips, 16 shards (an17 0–7,
  an29 8–15), tag=p1cfg1 → results/stage0 (journaled, resumable). ~96s/clip (nfe≈1556),
  ~20 min wall. Storage well under the 100 GB cap. NEXT on completion: `--aggregate` →
  Fig 1/2 + `COMMITMENT_MAP_DONE`; Track P → Fig 4; then cfg=4.5 full-pool Gate-A (P1.4),
  Phase 2 readout, Phase 3 gap → GO_MAP/GO_READOUT. No pause trigger hit.

## 2026-06-20 — Autonomous arc, Stage-0 FREEZE → `GO_MAPS_PHASE`

PI delegated the full arc (Stage-0 freeze → P1→P2→P3 → Track P/Stage R/Phase 4) autonomously;
human labels arrived (anchor 30, validity 50 = 36 present/14 absent); §3.3 decisions resolved
(reliability split by claim layer; class diagnostic). Stage-0 frozen with NO pause trigger.

- **S0.1 human-label full suite** (CPU): added `sidecar.pabak` + `sidecar.confusion_matrix`
  (+10 tests). `compute_validity_kappa.py` → full suite (raw, marginals, confusion, κ, AC1,
  PABAK) → `validity_suite.{json,md}`. Findings (correctness-layer, NOT gating): human↔qwen
  agree on presence/class (κ 0.66/0.62) but the MEASURER disagrees with both on class
  (κ 0.23) — confirms class's measurer-validity weakness; **timing**: humans tap the salient
  event ~1s later than both machine onsets (which fire at t≈0) → weak timing validity (§3.2
  "quiet failure mode"), flagged for Phase-4 timing-correctness only. Fixed a latent timing
  bin bug (measurer/qwen labels already binned; only human seconds binned).
- **Anchor adoption** (`stage0_anchor_adoption.py`): human-vs-audio σ_anchor = 0.969s > 0.35s
  → **`AUDIO_ANCHOR_NOT_ADOPTED`** (pre-registered outcome, not a downgrade) → keep approved
  chain, **timing_bin_s = 1.5147** (the manual's anticipated "marginal" timing resolution).
- **Material 2nd-embedder** (`stage0_material_second_embedder.py`): CLAP-vs-PANNs RSA
  Spearman ρ = **0.494** (CI [0.41, 0.64]) — moderate, significant (CI excludes 0)
  cross-embedder agreement; material kept in maps with this caveat (correctness-layer).
- **S0.3 manifest freeze** (`freeze_manifest.py`): **200 single + 60 two-event**, 60/40 split
  (161/99), per-axis usable/non-pinned n (timing 56 non-pinned = scarce axis as expected;
  material 197, class 167 abundant), cfg-specific exclusions, class=diagnostic, timing_bin_s
  1.5147 → `data/manifests/phase1_manifest_frozen.json`.
- **S0.2 self-target gate** (`stage0_gate.py`): det+rob only (validity excluded). presence
  (1.0/0.983), timing (1.0/0.955), material (1.0/0.976) PASS; class (1.0/0.833) fails rob →
  kept DIAGNOSTIC. **3/4 ≥ 3, validation OK, manifest frozen → `GO_MAPS_PHASE`.** (Note: det/rob
  reused from the 0.5s-bin reliability run; timing robustness only improves at the coarser
  frozen bin, so the ≥3 decision is bin-robust.)
- **Codex review:** independently re-ran the gate, reproduced `GO_MAPS_PHASE` with frozen
  thresholds; **no BLOCKING**. One forward catch: `cli/phases123_maps.py` still filters axes by
  the OLD combined gate — must filter by the self-target gate (class diagnostic) in the Phase-1
  runner. 874 tests green. **Tokens: `GO_MAPS_PHASE`.** Proceeding to Phase 1 (no pause).

---

- **T6 PI checkpoint #3 → PAUSE.** `results/stage0/pi_checkpoint3.md` assembled: MICROMAP_PASS
  + dual schedule-suffixed kernel tokens (cfg=4.5 CANDIDATE); screening done; reliability
  κ-paradox finding (presence+timing pass under AC1, class genuine fail, material no-gold);
  validity_bundle.html (50 clips) + anchor_bundle.html (30) ready. **Under either statistic
  <3 axes survive the automated proxy** → GO_MAPS_PHASE stays blocked. Surfaced to PI:
  validity-statistic choice (κ vs AC1 vs both), material second-embedder, class accept-vs-
  rescue, plus the two human JSONL passes + manifest-freeze sign-off. **No manifest freeze,
  no GO_* token emitted. Paused for PI.**
