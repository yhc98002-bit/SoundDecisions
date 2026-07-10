# Full-pool Gate-A — interpretation + the threshold-scaling ruling needed (Decision 2)

**Mechanical verdict (as the frozen code ran):** `CFG_KERNEL_FAIL(cfg=1, schedule=sqrt_down)`
AND `CFG_KERNEL_FAIL(cfg=4.5, schedule=sqrt_down)`. **But both "failures" are an artifact of
applying the frozen *16-cell* count caps to a *200-cell* pool**, not real kernel failures. This
is a frozen-quantity governance decision — I did NOT rescale anything; the PI must rule.

## The data (200 cells / s-point; per-cell permutation MMD on sqrt PANNs-probs, n_perm=200)

| | low-p frac (p<0.05) | median MMD p | MMD exceedance (>95th-pct) | TV exceedance | Mann-Whitney vs null |
|---|---|---|---|---|---|
| cfg=1.0 s=0.05 | 11/200 = 5.5% | 0.425 | — (it IS the null) | — | — |
| cfg=1.0 s=0.90 | 9/200 = 4.5% | 0.525 | — | — | — |
| cfg=4.5 s=0.05 | 12/200 = 6.0% | 0.435 | 15/200 = 7.5% | 6/200 | MMD p=0.745, TV p≈1.0 |
| cfg=4.5 s=0.90 | 11/200 = 5.5% | 0.430 | 15/200 = 7.5% | 5/200 | MMD p=0.235, TV p≈1.0 |

## What this means (Codex-reviewed)

1. **cfg=1.0 internal null behaves as an exact kernel should.** Low-p fraction ≈ 5% = the chance
   rate under a uniform-p null. The frozen cap is `LOW_P_MAX_CELLS=2`, explicitly "out of 16"
   (Binomial(16, 0.05)). At 200 cells the *same* Binomial(n, 0.05) rule gives a 95% one-sided cap
   of **≈ 15**, not 2. Observed 11 and 9 → **PASS**. The cfg=1.0 backbone is sound; its "FAIL" is
   purely the 16→200 exposure mismatch (applying `2` to 200 cells ≈ a 1% cap — far over-strict).

2. **cfg=4.5 is APPROXIMATELY exchangeable at full-pool scale** (a real, surprising result vs the
   Stage-M underpowered failure — but NOT "exact"). Its Mann-Whitney tests ALL pass (MMD p=0.745,
   0.235; TV p≈1.0): cfg=4.5's MMD/TV are *not* stochastically larger than the cfg=1.0 null. The
   only failing check is **exceedance count**: MMD 15/200 (7.5%) and TV 6,5 vs the frozen cap of 3
   (a 16-cell number). Under the scaled cap (Binomial(200, 0.05) ≈ 15): TV passes comfortably; MMD
   sits **right at the boundary** (15 ≈ 15). So cfg=4.5 would ratify under the scaled rule, but the
   MMD exceedance at 7.5% (vs the 5% null expectation) is a *mild* left-shift — **small
   non-exactness is not ruled out** (median p 0.43 vs 0.5 ideal echoes this). Honest framing:
   "empirically near-exchangeable on tagger-probs, not provably exact."

3. **Scaling the cap is the SAME pre-registered rule at the correct n, not a re-tune** —
   *provided it is documented as correcting the exposure n*. Per-cell power is low (16-vs-16
   permutation MMD); the 200-cell aggregate is where power lives, so the absence of a strong
   leftward p-shift at 200 cells is *stronger* evidence of exchangeability, not weaker. But because
   the frozen implementation encoded `16`, changing the operational threshold *after seeing the
   result* is a **ratification decision** — to be made by the PI, not applied silently by the agent.

## The ruling I need (Decision 2)

**Ratify the per-16 → per-200 threshold scaling?** (i.e. evaluate the frozen Binomial(n, 0.05)
caps at the actual cell count n=200: low-p cap ≈ 15, exceedance cap ≈ 15.)
- **If yes:** cfg=1.0 PASSES (backbone confirmed sound) and cfg=4.5 RATIFIES (MMD exceedance at the
  boundary, TV/MW clear) → emit `CFG_KERNEL_OK(cfg=4.5, schedule=sqrt_down)` [ratified]. **Then the
  F-1 share-migration contrast (Fig 1b) rests on CERTIFIED cfg=4.5 data — the tension does NOT
  dissolve** (contrary to the candidate-only hope in your Decision 2).
- **If no / re-collect:** keep the frozen 16-cell design by subsampling the pool into 16-clip
  blocks and applying the cap per block (a stricter, design-faithful reading), or hold cfg=4.5 at
  candidate.

I have NOT modified `certified_kernels.json` (ledger unchanged; cfg=4.5 still `ratified=false`).
Raw per-s detail: `results/stage0/gate_a_fullpool.json`. The evaluator's fixed caps
(`gate_a.py:48,50` = 2, 3, both "out of 16") are the code locus; a scaled-cap option is a one-line
change I will make only on your ratification.
