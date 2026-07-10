# PI Report — Autonomous Arc 2: Gate-A, F-1 disambiguation, Phase-4 (foley-cw, 2026-06-20)

**Bottom line.** I executed all three of your decisions and am pausing at exactly the two points
you named: **(1) the final F-1 narrative call** (the dial+swap data is in and it cleanly *refutes*
F-1's specific prediction while identifying a better mechanism), and **(2) a frozen-quantity
ruling** on the Gate-A threshold scaling (`CFG_KERNEL_FAIL`, but it's a 16→200-cell artifact, not
a real failure). Phase-4 (Decision 3) is done and strong. No frozen quantity was changed. 947
tests green. Two surprising results, both Codex-cross-checked.

---

## Decision 2 — cfg=4.5 full-pool Gate-A: a threshold-scaling artifact (your ruling needed)

Full-pool Gate-A (200 clips) emitted `CFG_KERNEL_FAIL` for **both** cfgs — but the underlying
statistics show **both behave at the null/chance level**; the "failures" come entirely from
applying the frozen **16-cell** count caps to a **200-cell** pool.

| | low-p frac | median MMD p | MMD exceedance | TV exceedance | Mann-Whitney vs null |
|---|---|---|---|---|---|
| cfg=1.0 s=0.05 / 0.90 | 5.5% / 4.5% | 0.43 / 0.53 | (is the null) | — | — |
| cfg=4.5 s=0.05 / 0.90 | 6.0% / 5.5% | 0.43 / 0.43 | 7.5% / 7.5% | 3.0% / 2.5% | p=0.75 / 0.24, TV p≈1.0 |

- **cfg=1.0 (the exact backbone) is sound**: low-p ≈ 5% = exact-kernel chance rate. The frozen cap
  `LOW_P_MAX_CELLS=2` is explicitly "out of 16" (Binomial(16,0.05)); the same Binomial(n,0.05) rule
  at n=200 gives a cap of **≈15**, not 2. 11 and 9 → PASS. Applying 2 to 200 cells is a ~1% cap.
- **cfg=4.5 is approximately exchangeable** (a real, surprising result vs the Stage-M underpowered
  failure — *not* "exact"). Mann-Whitney all pass (its MMD/TV aren't stochastically larger than the
  null); only the **exceedance count** trips the cap-of-3 (MMD 15/200, TV 6,5). Under the scaled cap
  (≈15): TV clears, MMD sits **right at the boundary**. So cfg=4.5 would ratify, with a mild
  left-shift (median p 0.43 vs 0.5) meaning small non-exactness isn't fully ruled out.
- **Codex (independent) confirmed**: scaling Binomial(16,·) → Binomial(200,·) is the *same*
  pre-registered rule at the correct exposure n (not a confound-hiding re-tune) **if documented as
  correcting n** — but because the frozen code encoded "16", changing it after seeing the result is
  a **ratification decision for you**, not something I apply silently. **Ledger unchanged.**

**Your ruling:** ratify the per-16 → per-200 threshold scaling? **If yes** → cfg=1.0 confirmed
sound + `CFG_KERNEL_OK(cfg=4.5, schedule=sqrt_down)` [ratified]; **and the F-1 contrast then rests
on CERTIFIED cfg=4.5 data — the tension does NOT dissolve** (contrary to the candidate-only hope).
Detail: `results/stage0/gate_a_fullpool_interpretation.md`, `gate_a_fullpool.json`. The fix is a
one-line scaled-cap change to `gate_a.py:48,50` I will make only on your ratification.

## Decision 1 — F-1 disambiguation: REFUTED in form, replaced by a cleaner mechanism

You asked for the dial + condition-swap *together* to separate seed / conditioning /
entropy-reduction. They converge decisively:

**(a) Condition-swap — the clean causal test (Fig 5).** Replace a clip's video conditioning with a
donor's at progress s, finish the ODE, measure follow-rate (matches donor) vs retention (matches
source). The swap mechanism is validated (presence/timing/material follow the donor when swapped
early). **Class is the exception: it does NOT follow a donor swap even at s=0.05 (follow = 0.45 ≈
chance).** → class is **internally locked, not video-determined**.

| axis | s_cond | follow@0.05 | reading |
|---|---|---|---|
| presence | 0.75 | 0.85 | video-determined (follows donor) |
| timing | never | 0.90 | video-determined throughout |
| material | 0.25 | 0.80 | video-determined early |
| **class** | 0.05 | **0.45** | **internally locked — NOT video-determined** |

**(b) The dial — does the seed's grip grow with cfg? NO** (24 clips × cfg∈{1,1.5,2,2.5,3,4.5}):
- noise→class probe accuracy: 0.33 → 0.20 (decreasing; underpowered, near the class prior — weak).
- earliest fork-agreement of class: flat ~0.33 (the cleaner measure — no growth).
- α*(cfg): flat at the 0.05 floor (uninformative after pooling).
- seed share (Fig 1b, the authoritative 2-point contrast): **shrinks** (class −0.05, material −0.09).

**(c) The actual mechanism — ENTROPY REDUCTION (mode collapse).** Distinct classes among the 16
independents drop monotonically with cfg: **4.83 → 4.12 → 3.75 → 3.71 → 3.75 → 3.62.** Guidance
narrows the output distribution; this **mechanically raises A_independent**, which is what inflated
the budget's apparent "conditioning share" — *without* the video actually determining class (the
condition-swap proves it doesn't).

**Synthesis (suggested token `F1_REFUTED`, your binding call).** F-1 said "guidance moves the
decision into the seed." The data: the **trajectory share does collapse** with cfg (✓ half of F-1),
but the decision migrates into **neither the seed (dial: no growth) nor the video (swap: class
doesn't follow)** — it migrates into **degeneracy**: the distribution simply tightens (entropy
reduction). The conditioning confound you flagged is **ruled out by the clean causal test**; the
entropy-reduction confound is the *finding*. This is a cleaner, more defensible result than F-1 —
the three-share headline (Fig 1) stands, and we now have a crisp causal story: *at deployed
guidance, class is internally locked and the outcome set mode-collapses, not video-steered.* I did
NOT emit a binding F-1 token (the module only suggests; the narrative is yours). Data:
`cond_swap_map_cswap.csv`, `cfg_dial_f1.json`.

## Decision 3 — Phase-4 offline policy (Fig 6): oracle axis-gating wins decisively

Offline replay over the 200 cfg=4.5 clips, matched generator-NFE + scoring-call accounting:

| policy | final corr | NFE | scoring | regret |
|---|---|---|---|---|
| full BoN | 0.370 | 80000 | 3200 | 0.590 |
| same-compute BoN | 0.350 | 46925 | 1877 | 0.610 |
| DiffRS scalar | 0.370 | 50512 | 4848 | 0.590 |
| SMC scalar | 0.370 | 58511 | 5269 | 0.590 |
| **oracle axis-gated** | **0.785** | 49151 | 8437 | **0.175** |

Axis-gated pruning (the paper's lever) **≈2× the correctness of every matched-compute scalar
baseline** (DiffRS, SMC, BoN) at comparable NFE — METHOD-tier headroom. Caveat: proxy-correctness
(agreement with the per-clip majority self-target, not human gold) and offline; token
`DIAGNOSTIC_ONLY`. It shows the headroom that licenses an online run + the human sidecar.

---

## Token ledger this arc
`COMMITMENT_MAP_DONE`, `READOUT_MAP_DONE`, `GO_MAP`, `GO_READOUT` (Arc 1) →
Gate-A `CFG_KERNEL_FAIL`×2 **(threshold artifact — pending your scaling ruling)**, suggested
`F1_REFUTED` **(entropy-reduction reframing — your narrative call)**, Phase-4 `DIAGNOSTIC_ONLY`.
Frozen quantities (θ_cal, certified tuple, interpretations) UNCHANGED; ledger untouched.

## Figures / tables
Fig 1 (budget), Fig 1b (share migration), Fig 2 (surface), Fig 4 (internal probes), Fig 5
(condition-swap, `cond_swap_map_cswap.csv`), Fig 6 (policy Pareto, `policy_pareto.csv`); Tab 1
(reliability), Tab 2 (separation). Remaining: large_44k scale insurance (Tab 3) — not yet run;
Fig 5/6 PNGs pending (data complete).

## The two rulings I need
1. **Gate-A threshold scaling** — ratify per-16 → per-200? (determines whether cfg=4.5 certifies and
   whether the F-1 contrast is on certified data).
2. **Final F-1 narrative** — accept `F1_REFUTED` + the entropy-reduction reframing as the headline,
   or hold for more evidence (e.g. a larger dial, or the seed→class probe on a reduced-dim latent)?

Everything is journaled (`results/EXECUTION_JOURNAL.md`); manual Status current. Standing by.
