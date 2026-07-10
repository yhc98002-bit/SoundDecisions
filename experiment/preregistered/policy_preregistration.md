# Pre-registered Phase-4 OFFLINE policy simulation (June-13 manual §9, Fig. 6)

Frozen 2026-06-20 from "LONG_RANGE_EXPERIMENT_PLAN _June_13th.md" §9, **before any policy
number was computed**, after `GO_MAP + GO_READOUT` (`results/stage0/phase1/phase3_decision.json`)
licensed Phase 4. This file fixes the baselines, the metric set, the accounting rules, and the
correctness PROXY caveat. It is framing-agnostic (the guidance-aware seed-triage arm of §9 is a
separate online study); this freeze covers ONLY the offline REPLAY simulation on cached pools.

## Scope and what is being simulated

- **Offline first (§9).** No new generation. Pools = the cached deployed-cfg (`cfg = 4.5`)
  Phase-1 independents already on disk: `measurements.jsonl` role `p1cfg45_independent`,
  200 clips × up to 16 candidates, per-axis final self-target labels
  (presence/timing/class categorical; material embedding).
- **Generate-N-candidates is REPLAYED** from the cached pool: drawing candidate `j` for a clip
  returns that clip's cached independent `ind{j}` and its already-measured final self-targets.
  Going online is gated on this offline pass showing headroom; it is out of scope here.

## CORRECTNESS PROXY caveat (load-bearing — this is NOT human correctness)

There is no human gold-vs-video at scale. The manual's correctness factorization
(`correctness(axis) = match(readable axis value, video anchor)`) needs the video anchor, which we
do not have for every clip. We therefore define an **ORACLE PROXY**:

> `proxy_correct(candidate j, axis a)` = the candidate's final self-target on axis `a` AGREES with
> the **per-clip majority self-target across that clip's independents** on axis `a`.

For categorical axes (presence/timing/class) agreement is exact label match to the plurality label.
For the embedding axis (material) a candidate is proxy-correct iff its self-target embedding is
closer (cosine) to the per-clip mean independent embedding than the per-clip median candidate
distance (i.e. in the consensus half). Final proxy-correctness of a candidate = proxy-correct on
**all** in-scope axes; per-axis proxy-correctness is also reported.

**This is a self-consistency proxy, NOT validated correctness.** It measures "does this candidate
match what the deployed model usually produces for this clip," not "is this audio right for the
video." Ties in the majority are resolved deterministically (sorted-label order) and flagged. Every
reported number inherits this caveat; the proxy is explicitly distinguished from the human/MLLM
correctness sidecar, which scales only after both GO tokens and is the only path to a correctness
claim. No headline correctness claim is made from the proxy alone.

## Policies / baselines (same pools, frozen list, §9)

All policies see the SAME per-clip pool of N=16 cached candidates and the SAME cached scalar/per-axis
reward signals (the scorer is an INPUT, identical across policies — see accounting). Implemented:

1. **full_bon** — generate all N to completion, rank by final scalar reward, take argmax. Compute
   ceiling reference (max scoring + max generator NFE).
2. **same_compute_bon** — BoN restricted to the SAME total generator-NFE budget as the gated policy
   (fewer completed candidates), then final-rerank. The matched-compute honest baseline.
3. **random_prune** — at the first window, prune a fraction of candidates uniformly at random
   (seeded), complete survivors, final-rerank. Control: pruning that ignores the signal.
4. **diffrs_scalar** — DiffRS-style rejection: at the gate, reject any candidate whose scalar reward
   is below a frozen threshold `τ_diffrs`; complete survivors; final-rerank.
5. **smc_scalar** — SMC-style sequential resampling: at the gate, resample the surviving population
   with replacement proportional to softmax(scalar reward / `T_smc`); complete the resampled set;
   final-rerank. Matched candidate count.
6. **final_rerank** — no pruning; complete all N; pick argmax final scalar reward. (Equals full_bon
   when the gate scalar = final scalar; kept distinct because its scoring-call accounting differs:
   it scores only at the final window.)
7. **oracle_axis_gated** — generate N to the first actionable window; prune on axes whose commitment
   window has CLOSED by that s (using `s_commit` from `determination_budget_p1cfg45.csv`), keeping
   only candidates whose in-window axis self-target matches the running plurality; continue
   survivors; evaluate later axes at later windows; finish and final-rerank.

The accounting fixes `same_compute_bon`'s budget to `oracle_axis_gated`'s realized generator NFE,
so #2 and #7 are read off the SAME compute x-coordinate of Fig. 6 (the matched-compute contrast).

## Accounting rules (frozen — both matched, §9)

- **Generator-NFE.** One full candidate costs `NFE_full = num_steps` (deployed `cfg=4.5`,
  `num_steps = 25`). A candidate pruned at progress `s` costs `round(s · num_steps)` NFE (the
  fraction of the trajectory actually integrated before the prune); a never-pruned candidate costs
  `NFE_full`. `total_nfe(policy) = Σ_clip Σ_candidate nfe(candidate)`. This must be EXACT and is
  asserted in the test, not approximated.
- **Scoring-calls.** One scoring call per (candidate, window) the scorer is actually invoked at.
  full_bon / final_rerank score once per candidate (final window only) → N per clip. Gated policies
  score every surviving candidate at every gate it reaches (early gates + final). The two axes of
  Fig. 6 are generator-NFE and scoring-calls; both are reported, never conflated.
- **Determinism.** Given a seed, the whole simulation is reproducible (numpy default_rng, per-clip
  sub-stream via SeedSequence). No wall-clock, no unordered-dict iteration, no I/O randomness.

## Metrics per policy (frozen, §9 / Fig. 6)

For each policy, over the eval clips (and reported with a video-unit bootstrap CI):
`final_correctness` (proxy), per-axis `correctness` (proxy), `completed_candidates`, `total_nfe`,
`scoring_calls`, `winner_retention` (P[the pool's best-final candidate survives to the end]),
`false_prune_rate` (fraction of pruned candidates that were proxy-correct AND would have been the
winner), `regret` (best-achievable proxy-correctness in the pool − achieved), and the two-axis
**compute–quality Pareto points** `(total_nfe, final_correctness)` and
`(scoring_calls, final_correctness)`. These are the only quantities Fig. 6 / `policy_pareto.csv`
report; anything else is out of pre-registered scope.

## Pre-registered predictions (directional, not gates)

- `oracle_axis_gated` reaches `same_compute_bon`'s final proxy-correctness at strictly fewer
  generator-NFE (early prune on closed-window axes is free quality), Pareto-dominating it.
- `random_prune` ≤ `same_compute_bon` ≤ `oracle_axis_gated` on the proxy-quality-at-matched-NFE
  axis; `random_prune`'s false-prune rate is highest.
- `full_bon` is the quality ceiling at the highest NFE; the offline "headroom" decision is whether
  `oracle_axis_gated` approaches that ceiling at materially lower NFE than the matched baselines.
- A NEGATIVE offline outcome (gated ≈ random at matched compute) routes to `GO_RESTRICTED` /
  `DIAGNOSTIC_ONLY` and does NOT license the online study. Tokens stay §9: `GO_POLICY` /
  `GO_RESTRICTED` / `DIAGNOSTIC_ONLY`.

## Anti-overclaim

This offline pass cannot establish a deployment policy claim by itself: it is REPLAY on a
self-consistency proxy. Per §1.7 the paper claims nothing about beating SMC/DiffRS "beyond CIs"
without the online head-to-head and the human/MLLM correctness sidecar. The offline Pareto is a
headroom screen and a method illustration, reported as such.
