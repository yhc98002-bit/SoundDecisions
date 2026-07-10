# PI Checkpoint #3 — Stage-0 Ready, Pausing Before Manifest Freeze (foley-cw, 2026-06-13)

**Status: PAUSED at the planned boundary.** The largest agent-doable segment is
complete: Stage M passed under the §2.1 re-scope, the labeling tool is built, and
Stage-0 screening + the automated reliability computation are done. **The headline of
this checkpoint is a reliability finding that needs your call** (§4): the frozen Cohen-κ
validity gate demotes all four axes via the κ paradox; under the skew-robust diagnostic
presence+timing pass but class genuinely fails and material has no automated gold, so
**fewer than 3 axes survive the automated proxy either way** — and the human validity pass
is the manual's real arbiter. **Manifest freeze / `GO_MAPS_PHASE` await the two human long
poles, the §6 decisions, and your sign-off** (§13) — nothing downstream of this line was
run, θ_cal and the frozen interpretations are unchanged, and no `GO_*` map/policy token has
been emitted. Stage-M/Stage-0 diagnostics are never scientific evidence (§1.5).

## 1. Stage M — `MICROMAP_PASS` (your decisions 1–3)

Zero-GPU re-evaluation of the Run-3 data under the June-13 §2.1 washout-direction
rule (amendment #12, frozen pre-evaluation):
- class seed floor **g₀ = 0.1526** — non-negative, ≤ 0.25, and Gate-A exchangeable
  at s=0.05 → criterion 1 PASS; all five criteria pass → **`MICROMAP_PASS`**.
- Kernel tokens now carry the mandatory schedule suffix:
  **`CFG_KERNEL_OK(cfg=1, schedule=sqrt_down)`** (ratified backbone) and
  **`CFG_KERNEL_OK(cfg=4.5, schedule=sqrt_down)`** — **CANDIDATE only** (Stage-M
  pilot cells; a full-Phase-1-pool Gate-A must ratify it before the deployed-cfg
  commitment grid runs, §1.2/§15.8). Ledger: `certified_kernels.json`; the
  code-side guard `foley_cw/kernel_provenance.assert_certified_kernel` refuses any
  commitment grid under an uncertified or candidate-only (cfg, schedule).
- Amendment #12 frozen with rationale in
  `experiment/preregistered/stage_m_rerun_interpretations.md` (#12) + SHA256.
- Codex review: 1 High + 3 lesser findings, all fixed (multi-shard SDE merge,
  cfg=4.5 SDE override, token-text provenance validation, candidate marking,
  docstring). 852 tests green.

## 2. Labeling tool (your decision 4)

Self-contained static HTML bundles (`foley_cw/labeling_tool.py` +
`scripts/build_labeling_bundle.py`) — zero-setup, open in any browser, in-browser
JSONL export, localStorage resume; **audio-only** for validity (your choice). **Both
files live in `results/labeling/`** (NOT next to this checkpoint); they are large because
all media is base64-inlined, so download from the cluster path and open locally:
- **`results/labeling/anchor_bundle.html`** (42 MB, 30 clips, video+onset-tap) — fills
  `anchor_check_30.csv`'s human_onset_s and arbitrates audio-only anchor sharpness
  (adopt audio-only if σ_anchor ≤ 0.35 s, §3.2).
- **`results/labeling/validity_bundle.html`** (17 MB, ~50 generated-audio clips, presence
  + 12-class + abstain + onset) — built on the sidecar clips (§4 below); the same clips
  carry qwen labels so `scripts/compute_validity_kappa.py` reports human-vs-measurer κ
  (gate: ≥ 0.6), human-vs-qwen, qwen-vs-measurer in one pass.

## 3. Stage-0 screening (your decision 5)

**cfg=1.0, 400 clips** (`screening/a_independent_screen.csv`,
`video_determined_registry_screen.json`, tag=`screen`). Per-axis independent
self-agreement A_independent and the video-determined exclusions (A_ind > 0.9 → the
backbone fixes that attribute from conditioning alone, so 1−A_ind→0 leaves no seed/
trajectory headroom; dropped from THAT axis's normalized curve only, manual §3.1):

| axis | mean A_ind | median | video-determined (A_ind>0.9) | usable n (this axis) |
|---|---|---|---|---|
| presence | 0.809 | 1.000 | 203/400 (50.7%) | 197 |
| timing | 0.899 | 1.000 | 290/400 (72.5%) | 110 |
| class | — (abstain-heavy; many cells unscorable) | — | 51/400 scorable+det | small (abstain-limited) |
| material | 0.646 | 0.644 | 3/400 (0.7%) | 397 |

The high video-determined fraction on presence/timing is itself a result: for half to
three-quarters of clips the conditioning alone pins the attribute (A_ind→1), so there is
no headroom for a seed/trajectory share on that axis (those clips remain usable for the
other axes). This is what the manifest freeze must stratify per-axis usable-n on.

**cfg=4.5 sub-screen, 60 clips — NON-GATING** (for the Phase-4 deployed-cfg pool + F-1;
`screening/a_independent_subscreen45.csv`, tag=`subscreen45`): video-determined presence
36/60, timing 37/60, class 18/60, material 9/60 — a uniformly HIGHER determination
fraction than cfg=1.0, consistent with the cfg-sweep prediction that higher guidance
trades seed/trajectory share for conditioning share.

Storage stayed within the §1.4 contract (features + previews); the 100 GB cap was not
approached.

## 4. MLLM sidecar + provisional reliability gate — the κ-paradox finding

The MLLM sidecar judged 100 screening finals × {presence, class, timing} (+20-clip
test-retest self-stability: presence/class 1.00, timing 0.95). The frozen reliability gate
scores validity as Cohen's κ vs this MLLM gold over a seed-0 50-clip subsample
(θ_cal = 0.6, **FROZEN**). It **demoted all four axes → 0 survivors**:

| axis | determinism | robustness (θ=0.85) | Cohen κ (θ_cal=0.6) | gate |
|---|---|---|---|---|
| presence | 1.000 | 0.983 ✓ | **−0.125** | DEMOTED |
| timing | 1.000 | 0.955 ✓ | **−0.028** | DEMOTED |
| class | 1.000 | **0.833 ✗** | 0.145 | DEMOTED |
| material | 1.000 | 0.976 ✓ | NaN (no gold) | DEMOTED |

**This is the Cohen's-κ paradox, NOT measurement invalidity.** presence/timing have high
raw agreement (0.75 / 0.80) but ~90%-skewed marginals (present 92%/81%; timing-bin-0
92%/85%); Cohen's κ collapses (even negative) under that skew. The skew-robust **Gwet AC1**
(Gwet 2008) recovers them: **presence AC1 = 0.674, timing AC1 = 0.793 — both ≥ 0.6** (point
estimates vs the MLLM proxy under a skew-robust *diagnostic*, not the frozen gate). I
added `gwet_ac1` to `foley_cw/sidecar.py` (+12 hand-checked tests) and a zero-budget,
read-only diagnostic (`scripts/stage0_reliability_diagnostic.py` →
`reliability_diagnostic.{md,json}`) that **re-measures the on-disk finals and reproduces
the gate's exact κ** (presence −0.124567, timing −0.028278, class 0.145121 — all MATCH)
before reporting AC1, so the AC1 numbers are provably on the same pairs the gate scored.

**class is the genuine casualty — and re-eliciting qwen would NOT rescue it.** Decomposing
its κ over the full 100 clips:
- full: raw 0.230, κ 0.191, AC1 0.181
- gold-event-restricted (drop 45 clips whose gold is speech/music/ambient — classes the
  12-event-class measurer can never emit, because the gold was elicited under the OLD
  15-class qwen prompt) (n=55): AC1 0.373
- both-confident (also drop 9 measurer abstentions — a *favorable upper bound*, since it
  conditions on the measurer not abstaining) (n=46): AC1 **0.459 — fails even here.**

Class ALSO fails robustness (0.833 < 0.85) independently of validity. Because robustness
fails on its own, re-eliciting the 45 confounded class labels under the restricted prompt
(≈45 of the 69 remaining qwen calls) **cannot lift class to a pass** — so I **held the MLLM
budget (431/500)** rather than spend it on a re-run that can't change the outcome. θ_cal and
the frozen interpretations are **UNCHANGED**; AC1 is presented as a *diagnostic*, not a
silent swap of the gate statistic. **material** has no MLLM-judgeable gold (embedding) →
validity undefined; a second embedder would be needed to calibrate it.

**Bottom line for the gate:** under the frozen Cohen-κ statistic 0 axes survive; under the
skew-robust AC1, presence + timing survive (2). **Either way < 3**, so the GO_MAPS_PHASE
≥3-surviving-axes precondition is NOT met by the automated proxy alone. The manual's actual
validity arbiter is the **HUMAN sidecar (κ ≥ 0.6), still pending** — which is exactly why
this is your checkpoint. The decisions this raises are in §6.

## 5. Bookkeeping
- Two-event subset curated: 60 clips from the FoleyBench Multi-source+Discrete
  pool (`data/manifests/two_event_manifest.json`).
- Pre-registrations re-frozen from the June-13 manual (the three-share
  decomposition + F-1 were promoted to the headline; text changed):
  `cfg_sweep_predictions.md`, `go_map_gate_language.md`,
  `f1_protocol_predictions.md` + SHA256.

## 6. What remains gated (NOT run) — needs you

`GO_MAPS_PHASE` preconditions (§3) still open:
1. **30-clip human anchor marks** (anchor_bundle.html → JSONL) — also decides the
   audio-only anchor adoption (§3.2).
2. **~50-clip human validity** (validity_bundle.html → JSONL) — confirms the
   measurer validity κ ≥ 0.6 (the MLLM-validity in §4 is the automated proxy).
3. **Manifest freeze**: 200 single-event (stratified, video-pinned exclusions
   applied per the cfg-specific registry) + 60 two-event; 60/40 probe split;
   per-axis usable n; anchor source per clip; thresholds frozen — assembled and
   sign-off requested only after (1)+(2).
4. **cfg=4.5 ratification**: the deployed-cfg headline arm needs a full-Phase-1-pool
   Gate-A (a Phase-1 activity); the Stage-M token is candidate-only until then.

New decisions the §4 reliability finding raises (I changed nothing frozen; these are
yours):
5. **Validity statistic.** Cohen's κ is broken here by the marginal-skew paradox. I
   recommend reporting **both κ and the skew-robust AC1** and treating the **human
   sidecar (2) as the arbiter** of the frozen 0.6 gate — not silently swapping the gate
   statistic. Your call: keep κ-only, adopt AC1 (would need an amendment), or both.
6. **material axis.** No MLLM-judgeable gold → validity undefined. Validate via a second
   embedder (e.g. CLAP-vs-PANNs cosine), or accept it as a reported-only continuous axis?
7. **class axis.** Genuinely sub-threshold (robustness 0.833 < 0.85; AC1 0.459 even
   de-confounded). Accept it as a demoted/NEGATIVE-layer axis (the three-share headline
   survives a demoted axis, §12), or attempt a rescue? Note: re-eliciting the 45
   confounded qwen class labels (≈45 of 69 remaining calls) **cannot** fix robustness, so
   I did not spend it — tell me if you still want the restricted re-elicitation before
   Phase 1.

Because presence + timing are the only axes that pass even under the generous statistic,
**≥3 surviving axes is not yet met by the automated proxy** — the human sidecar and your
(5)–(7) decisions determine whether the manifest can freeze. Send the two JSONL exports,
your (5)–(7) calls, your manifest-freeze sign-off, and any cfg=4.5 Phase-1 ramp
preference, and I resume into the manifest freeze → `GO_MAPS_PHASE` → Phase 1.
