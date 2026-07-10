# PI Report — Autonomous Arc 1: Stage-0 → Make-or-Break PASSED → F-1 tension (foley-cw, 2026-06-20)

**Bottom line.** The autonomous run cleared the entire **make-or-break**: Stage-0 froze and
emitted `GO_MAPS_PHASE`, the cfg=1.0 commitment/readout/gap maps completed, and Phase 3
emitted **`GO_MAP` + `GO_READOUT`** — the best Phase-3 outcome, which unlocks both Stage R
and Phase 4. The paper's lead object (the three-share determination budget, Fig 1) now stands
on real evidence. I then computed the cfg=1.0↔4.5 share-migration contrast (Fig 1b) and hit a
genuine **pre-registration tension on F-1** that needs your framing call — that is the one
reason I paused (delegation trigger c/d), not routine review. **No frozen quantity was
changed; no negative gate token was forced.**

Tokens emitted, in order: `GO_MAPS_PHASE` → `CFG_KERNEL_OK(cfg=1, schedule=sqrt_down)` (ratified
backbone) → `COMMITMENT_MAP_DONE` → `READOUT_MAP_DONE` → **`GO_MAP` + `GO_READOUT`**.

---

## 1. Stage-0 freeze → `GO_MAPS_PHASE` (autonomous, §3.3 split)

- **Self-target gate** (determinism + robustness; validity is correctness-layer only):
  presence (1.00 / 0.98), timing (1.00 / 0.96), material (1.00 / 0.98) **PASS → 3/4 ≥ 3**;
  class (1.00 / 0.83) fails robustness → **kept DIAGNOSTIC** (in the maps, not gating).
- **Manifest frozen**: 200 single-event + 60 two-event, 60/40 split, per-axis usable/
  non-pinned n, timing_bin_s **1.5147** (`AUDIO_ANCHOR_NOT_ADOPTED`: human-vs-audio onset
  σ = 0.97 s > 0.35 s → keep the approved anchor chain). `data/manifests/phase1_manifest_frozen.json`.
- **Human-label suite** (correctness-layer diagnostics, never gating): validity full suite
  (κ / AC1 / PABAK / confusion) — human↔qwen agree (presence κ 0.66, class 0.62) but the
  *measurer* is the class-validity weak link; **timing**: humans tap the salient event ~1 s
  after the measurer's t≈0 onset (a Phase-4 timing-correctness caveat). Material CLAP-vs-PANNs
  RSA ρ = 0.49 (moderate). Codex independently reproduced the gate; no blocking. 874 tests green.

## 2. Phase 1 — Commitment maps → **Fig 1 (the lead result)**

200 clips, K=12 forks at the certified tuple (cfg=1.0, sqrt_down, α=0.8), N=16 independents,
bootstrap by video. The three-share determination budget:

| axis | conditioning | seed | trajectory | s_commit [95% CI] |
|---|---|---|---|---|
| presence | 0.81 | 0.10 | 0.08 | 0.21 [0.17, 0.27] |
| timing | 0.90 | 0.08 | 0.03 | 0.11 [0.08, 0.16] |
| **class** (diagnostic) | 0.38 | **0.23** | **0.35** | 0.35 [0.31, 0.39] |
| material (emb) | 0.64 | 0.21 | 0.14 | **0.64** [0.62, 0.66] |

**Reading:** presence/timing are **conditioning-dominated** (the video fixes them). **Class
carries the largest trajectory share (0.35) and a real seed floor (0.23)** — the dynamics
resolve the event class, connecting to the Stage-M seed floor g₀ and the F-1 mode-locking
story. **Material commits latest** (s_commit 0.64); its ~36 % seed+trajectory headroom carries
the readout-window story. Taxonomy (clip counts): class = 41 seed-determined / 107
trajectory-early / 29 trajectory-late; material = 116 trajectory-mid / 78 trajectory-late.

## 3. Phases 2–3 — Readout + the make-or-break → **`GO_MAP` + `GO_READOUT`**

- **Readout** (audio-tagger probe on the x̂0(s) previews): s_read = timing 0.05, presence
  0.35, material 0.60, class 0.75.
- **Separation**: self-target axes order **timing < presence < material** with
  **non-overlapping CIs**; **separation_score = 7.24** (windows separated at 7× the mean CI
  width). **Gate B PASS** — the ordering is identical at α = 0.8 and α = 0.4. → **`GO_MAP`**.
- **Readout feasibility**: presence/timing/material are committed *and* externally readable
  before s = 1 → **`GO_READOUT`**.
- **gap = s_read − s_commit**: timing −0.06, material −0.04 (readable ≈ at commitment),
  presence +0.14, **class +0.40**.

## 4. Track P — internal probes → **Fig 4** ("the generator knows before the audio shows")

Linear probes on the generator's own pooled features (frozen 60/40 split) read the final
self-target from **s = 0.05** for presence (acc → 0.87) and timing (0.93, flat) — far earlier
than the external probe. **Class is never linearly readable internally** (max 0.44 < θ_read
0.7) despite carrying the largest trajectory share. So **class is the R2 axis**: committed in
the trajectory, but hard to read early either internally or externally — this is the strongest
motivation for the Phase-5 feature-head and for a class-blind early-action policy.

## 5. cfg=4.5 share migration → **Fig 1b** → the F-1 tension (why I paused)

cfg=4.5 commitment grid complete (200/200). The cfg=1.0→4.5 contrast (independently
recomputed from raw per-clip CSVs — not an aggregation artifact):

| axis | conditioning 1.0→4.5 | seed 1.0→4.5 | trajectory 1.0→4.5 |
|---|---|---|---|
| class | 0.38 → **0.51** (+0.13) | 0.23 → **0.18** (−0.05) | 0.35 → **0.29** (−0.07) |
| material | 0.64 → **0.78** (+0.15) | 0.21 → **0.12** (−0.09) | 0.14 → **0.09** (−0.06) |

**F-1 predicted** (§8.3): *the seed share grows monotonically with cfg; the decision migrates
into the seed.* What the contrast shows:
- ✅ **Trajectory share collapses** with guidance — the decision *does* leave the trajectory.
- ❌ It migrates into **conditioning**, not the seed; the **seed share shrinks** for the two
  axes that carry the story (class, material).

**The crux (a real interpretive fork, not a bug):** at high cfg the 16 independents collapse
toward similarity, so `A_independent` rises mechanically. Because conditioning ≡ `A_independent`
and seed = `A_fork(s_min) − A_independent`, a rising `A_independent` **inflates the
conditioning share and deflates the seed share even if the seed still determines the value.**
The share decomposition *cannot* separate "the video now determines it" from "guidance collapsed
all fork diversity." Disambiguating is exactly the job of the F-1 **dial** (part b:
same-seed-across-conditions test + α*(cfg)), which is a separate small run I have **not** done.

This is therefore **not** `F1_REFUTED` — it is a pre-registration framing call for you.

### What I need from you (the only blocker)

1. **F-1 direction.** Pick one: **(a)** run the F-1 dial (~24 clips × 6 cfgs: same-seed→same-class
   predictability + α*(cfg)) to disambiguate seed vs conditioning — I recommend this, it is the
   pre-registered tie-breaker; **(b)** reframe F-1 as *trajectory→conditioning* migration (a
   defensible new framing the data directly supports); or **(c)** record the contrast as
   `F1_INCONCLUSIVE` and lean on the dial later.
2. **cfg=4.5 Gate-A ratification.** The commitment data is in hand, but ratifying the deployed-cfg
   kernel needs a **prob-vector collection run** (I stored labels, not the 527-dim PANNs probs
   Gate-A consumes). Want me to run it (≈ same cost as one commitment grid) so the cfg=4.5 arm
   becomes a ratified second headline?
3. **Proceed into Stage R + Phase 4?** `GO_MAP` unlocks Stage R (large_44k scale, condition-swap
   Fig 5); `GO_MAP + GO_READOUT` unlocks Phase 4 (offline policy Fig 6). These are net-new builds
   I can take on autonomously once you've set the F-1 framing (it changes how Phase 4 is framed —
   guidance-aware seed-triage is contingent on the F-1 verdict).

---

## Figures / tables populated

| Deliverable | Status | File |
|---|---|---|
| Fig 1 — three-share budget | ✅ | `results/figures/fig1_determination_budget.png` |
| Fig 1b — cfg share migration | ✅ (interpretation pending F-1 call) | `results/figures/fig1b_share_migration.png` |
| Fig 2 — A(axis,s) surface | ✅ | `results/figures/fig2_commitment_surface.png` |
| Fig 4 — internal vs external readout | ✅ (data) | `results/stage0/phase1/internal_probe_report.md` |
| Tab 1 — reliability + demotions | ✅ | `results/stage0/reliability_report.*` + `validity_suite.md` |
| Tab 2 — separation + sensitivity | ✅ | `results/stage0/phase1/phase3_decision.md` |
| Fig 3 / 5 / 6, Tab 3 | pending Stage R / Phase 4 | — |

Everything is journaled in `results/EXECUTION_JOURNAL.md`; the manual's one-line Status is
current; all artifacts are under `results/stage0/phase1/`. Paused for your §5 decisions.
