# Phase-4 scalar-baseline TIE sanity (pre-reg §D)

**Question (pre-reg §D):** the offline Phase-4 Pareto (`results/stage0/phase1/policy_pareto.csv`)
shows `full_bon`, `diffrs_scalar`, `smc_scalar`, `final_rerank` ALL at `final_correctness ≈ 0.370`
while `oracle_axis_gated = 0.785`. Is the **0.370 scalar tie REAL** (genuine no-help, which
supports the axis-gating thesis), or a **wiring bug** in `foley_cw/policy_offline.py` that forces
the scalars to collapse onto the same ranking?

**Verdict: the tie is GENUINE. No wiring bug. The no-help result is real and supports
axis-gating.** Code inspected: `foley_cw/policy_offline.py`, `scripts/phase4_policy.py`. Numbers
below were reproduced exactly from the cache (all 200 clips, seed 0, `smc_temp=0.1`,
`diffrs_tau = median(final_score) = 0.9976`), matching every digit of the committed CSV.

## Reproduced metrics (all 200 clips — matches policy_pareto.csv)

| policy | final_corr | presence | timing | class | material |
|---|---|---|---|---|---|
| full_bon | 0.370 | 0.865 | 0.880 | 0.630 | 0.655 |
| same_compute_bon | 0.350 | 0.875 | 0.870 | 0.605 | 0.660 |
| random_prune | 0.330 | 0.885 | 0.890 | 0.605 | 0.640 |
| diffrs_scalar | 0.370 | 0.865 | 0.880 | 0.630 | 0.655 |
| smc_scalar | 0.370 | 0.875 | 0.880 | 0.600 | 0.680 |
| final_rerank | 0.370 | 0.865 | 0.880 | 0.630 | 0.655 |
| oracle_axis_gated | **0.785** | 1.000 | 1.000 | 0.995 | 0.785 |

## Why the scalars tie — three checks, each rules out a bug

**(1) `full_bon` ≡ `final_rerank` is correct BY CONSTRUCTION, not a bug.**
`policy_offline.py:254-258` handles both in one branch: complete all candidates, score at the
final window, pick `argmax(final_score)`. They are the *same* algorithm (rerank with no earlier
pruning = full BoN), so identical per-axis numbers are expected. Winner-index agreement = **1.000**.

**(2) `diffrs_scalar` = `full_bon` because DiffRS never prunes the eventual winner — also correct.**
`diffrs_scalar` (`:291-305`) keeps candidates with `final_score ≥ τ` where `τ = median`. It does
prune real candidates (mean **8.3 of 16** survive; on 19 pools the max score is below τ and the
best is force-kept), so it is NOT a silent no-op. But the final winner is `argmax(final_score)`
over the survivors, and the global argmax is always `≥ median`, so it always survives → same winner
as BoN. Winner-index agreement with `full_bon` = **1.000**. This is the intended semantics of a
scalar rejection filter whose ranking key equals its acceptance key; it is not a wiring error.

**(3) `smc_scalar` genuinely DIVERGES — this rules out a hard-wired collapse.**
`smc_scalar` (`:307-324`) resamples the population by `softmax(score/T)` and can drop the argmax
candidate. Winner-index agreement with `full_bon` = **0.690** (not 1.000), and its per-axis numbers
differ (class 0.600 vs 0.630, material 0.680 vs 0.655). If the simulator forced all scalars onto
one ranking (the bug hypothesis), SMC could not differ. It does → no forced collapse.

## Root cause of the ~0.37 ceiling: the scalar reward is near-uninformative

All four scalar policies ultimately select via `_argmax_by_score(survivors, final_score)`
(`:166-176`) on the **same** scalar reward. That reward (`phase4_policy.py:141-148`) is the cosine
of a candidate's final-grid pooled feature to the per-clip **mean** independent feature — a
self-consistency reward, deliberately *distinct* from the per-clip-majority label proxy used for
correctness.

That reward barely tracks the proxy correctness:
- within-pool `corr(scalar reward, all-axis proxy-correct)` = **0.179** (mean over pools);
- the `argmax(scalar)` winner is all-axis proxy-correct on **0.370** of clips, versus
  **0.260** for a random candidate and an oracle ceiling of **0.960** (fraction of pools with ≥1
  correct candidate).

So picking the highest-scalar candidate is only marginally better than random and far below the
oracle. Every scalar policy ranks by this same weak signal, so they all land near the
argmax-of-weak-signal ceiling (~0.37). The oracle instead gates on the *true per-axis labels*
(`oracle_axis_gated`, `:326-355`), reaching 0.785. This is exactly the headroom the pre-reg's B4
bridge is about.

## The one bug that WOULD invalidate the reading — ruled out

A cross-model reviewer (Codex GPT-5.5) flagged the right failure mode: if the scalar reward had
silently fallen back to the **proxy labels** (`phase4_policy.py:149-151`, the feature-free
`final_score = sum of axis agreement` branch), then the scalars would be ranking on leaked
correctness labels and the comparison to the oracle would be contaminated — a real invalidating
bug. Direct check over all 200 clips:

```
pools=200  features=3200  missing=0  fallback_pools=0
```

Every pool has all 16 final-grid features present, so `final_score` is ALWAYS the feature-cosine
self-consistency reward; the proxy-label fallback is NEVER taken. The scalar channel is clean — it
never sees correctness labels. (This is also why the within-pool corr is only 0.179: the scalar
reward is a genuine, label-free feature statistic.)

## Conclusion

The 0.370 tie is a **genuine no-help finding**, not an artifact: the scalar policies behave
distinctly under the hood (SMC diverges; DiffRS prunes; BoN/rerank are defined to coincide), and
they converge in *outcome* only because the available scalar reward carries almost no information
about the per-clip correctness proxy. A scalar reward that cannot separate correct from incorrect
candidates cannot recover the oracle's axis-gated headroom — which is precisely the argument FOR
axis-aware gating over scalar BoN/DiffRS/SMC.

**Caveat (inherited):** correctness here is the ORACLE PROXY (agreement with the per-clip majority
self-target), not human/MLLM correctness-vs-video — as stated in `policy_offline.py:10-13` and the
policy preregistration. The tie's *genuineness* (no bug) is independent of that caveat; the
*magnitude* of the headroom inherits it.

_Independent reproduction: `results/stage0/arc3/_repro_all200.py` (CPU, all 200 clips). Cross-model
review: see `results/stage0/arc3/_codex_tie_review.txt`._
