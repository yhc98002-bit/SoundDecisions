# Frozen interpretive rules — Stage-M re-run (slice 2)

Frozen 2026-06-12, BEFORE the Stage-M re-run or any curve inspection. These
instantiate points the revised manual (`LONG_RANGE_EXPERIMENT_PLAN_revised.md`)
delegates to the executing agent or leaves ambiguous. Each is submitted for PI
ratification at checkpoint #2; none is presented as PI-approved before then.

1. **Gate-A reference pool = genuinely FRESH independents.** Spec 1.2(iv) says
   "permutation-test against fresh independents"; we generate 8 dedicated
   reference independents per (clip, cfg) (RNG stream `gateref`) rather than
   reusing the fork-parents' own finals — textbook permutation exchangeability,
   no prefix pairing with the reference set. Cost +~0.3k FGE, accepted.

2. **Class abstain margin δ = 0.05, cross-group.** Label = top-1 over EVENT
   classes (non_event_indices ∪ groups {speech_vocal, music, ambient_nature}
   excluded; coarse map v3); runner-up = best event class whose coarse group
   differs from the top-1's (within-group flips never abstain). Abstain iff
   margin < 0.05. Rationale: ~17× the observed knife-edge flip scale (0.003,
   clip 2322, first run) and an upper bound on the prob jitter of the
   pre-registered robustness perturbations (O(0.01–0.03)), coupling confidence
   to θ_robust. Non-gating sensitivity ladder δ ∈ {0.02, 0.05, 0.10} computed
   from logged margins, reported only.

3. **Scorability rule for criterion 1 (label level).** A cell is scorable iff
   n_confident ≥ 2; agreement = pairwise exact-match over confident labels
   (abstain–abstain pairs NEVER count as agreement; n_conf < 2 → NaN, not the
   n<2→1.0 convention). Per endpoint per axis, ≥ 12/16 clips must be scorable,
   else criterion 1 FAILS with the δ/BEATs routing.

4. **Seed = 1.** The first run's seed-0 generations informed the redesign
   (δ choice, event-restriction); re-running at seed 0 would test on the data
   that motivated the instrument. RNG streams otherwise unchanged.

5. **exact-match gates; Krippendorff reported-only (class axis).** At K = 8 a
   7–1 split yields Krippendorff α = 0 exactly (D_o = D_e) — one flip is a
   collapse; the spec's §4 listing of both metrics is resolved as exact-match
   gating + Krippendorff/majority-share reporting (`configs/axes.json` updated;
   change pre-registered here).

6. **"Beyond CIs" = paired-difference bootstrap CI excludes 0.** All
   criterion-1 embedding rules use per-clip paired differences, percentile
   bootstrap over the 16 clips (B = 1000), 95% CI lower bound > 0; "stable
   floor" = CI half-width of mean A_fork_emb(0.05) ≤ 0.05. Gate-A feature
   transform = elementwise sqrt of tagger prob vectors (logit rejected:
   variance blow-up on ~500 near-zero coordinates).

7. **'other' coarse group survives event-restriction** (77 heterogeneous
   classes; two different events both tagged 'other' count as agreement).
   Accepted at Stage-M scale; registry note required before the Phase-1
   manifest freeze.

8. **Budget restatement.** Spec §2 says ≈1k FGE (≈20 GPU-min); with the
   spec-mandated Gate-A forks ({0.05, 0.90}) and the fresh reference pool the
   measured plan is ≈1.55k FGE ≈ 70 GPU-min total (≈9 min wall on 8×A800).
   The binding cap remains the §1.4 100 GB storage contract. Criterion-5
   video-pinned expectations are NOT anchored to the first run's 7/16 (the
   measure changed).

Gate-A verdict rules (fixed pre-run): cfg=1.0 internal null — guards healthy
AND per test s-point at most 2/16 cells with p < 0.05 (Binomial(16, 0.05):
P(≥3) = 4.3%); cfg=4.5 adjudication — per s and per statistic (sqrt-prob MMD,
extended-alphabet label TV incl. abstain): Mann–Whitney vs null p ≥ 0.01,
≤ 3/16 cells above the null 95th percentile, no cell above 3× threshold
(TV additionally capped at 0.9). Test s-points = {0.05, 0.90}.

9. **Tagger sanity metric under abstention.** The W3 sanity gate is scored on
   the CONFIDENT subset (abstain ≠ miss; abstention is the instrument's designed
   behavior), with the abstain rate reported and capped analogously to
   criterion 5 (≤ 30%). Event-restricted result on original FoleyBench audio:
   confident-subset accuracy 0.733 (11/15), abstain rate 0.25, vs 0.55 under
   abstain=miss. BEATs contingency: ARMED, NOT TRIGGERED (0.733 is well clear
   of 0.65); the micro-map qwen triangulation adjudicates at checkpoint #2.

10. **Run-2 outcome, routed iteration, and the g3 refinement (frozen BEFORE
    run 3).** Stage-M re-run #2 (seed 1, revised instruments) failed with all
    proximate causes consistent with constant-g α=1.6 tail violence (confident
    A_fork(0.90)=0.735; commit dip 0.30→0.60; abstain@0.90=0.34>0.30) plus a g3
    guard misfire at s=0.05 (θ_mmd 0.34 vs cross-clip MEDIAN 0.32, while the
    p-based guards were healthy: g1 power 0.95, g2 KS 0.13). Per revised manual
    1.3 this routes to the early-heavy g(s) schedule pilots (now extended to the
    headline cfg=1.0), NOT to silent α changes. Frozen for run 3:
    (a) the (cfg=1.0, schedule, α) tuple is selected by the pilot rule (smallest
        α passing washout+anchoring+validity; schedule preference order
        constant > sqrt_down > linear_down at equal α);
    (b) g3 separation compares θ_mmd[s] to the cross-clip 95TH PERCENTILE (not
        median): raw 8v8 MMD magnitudes in 527-dim sqrt-prob space are noisy
        and discrimination is p-based (g1); both readings stay reported;
    (c) ITERATION BOUND: at most ONE further Stage-M attempt; any failure of
        run 3 → halt and full PI package (no further instrument iteration).
    cfg=4.5 adjudication recorded: CFG_KERNEL_FAIL(cfg=4.5) (tv:gross @0.05,
    mmd:mw+exceedance @0.90) — the deployed-cfg kernel failure is now
    quantified by the seed-marginalized instrument; non-gating per 1.2.

11. **Run-3 tuple selection record (per the frozen #10 rule).** Pilot grid (8
    tuples, seed 1, an29): cfg=1.0 constant — no valid α (confirms run-2 under
    the new measure); cfg=1.0 linear_down — α=0.8 valid; cfg=1.0 sqrt_down —
    α=0.8 valid; cfg∈{2.5, 4.5} × {linear_down, sqrt_down} — no valid point;
    cfg=4.5 constant at n_steps=40 — no valid point (the discretization arm
    bounds the integrator-error explanation: guided-cfg locking is dynamics).
    Selected by the frozen preference order: **(cfg=1.0, sqrt_down, α=0.8,
    n_steps=20)**. Gate-A PRECHECK on the selected tuple (re-using stored
    fresh-reference pools; forks regenerated): **CFG_KERNEL_OK(cfg=1,
    schedule=sqrt_down)** — 1/16 low-p cells at both test s-points (cap 2),
    median MMD² −0.001/0.003, guards healthy (power 0.98, KS 0.17, p95
    separation OK). Stage-M run 3 launched at this tuple; per #10 it is the
    final attempt before a PI halt.

12. **Criterion-1 early rule re-scoped to WASHOUT DIRECTION + cfg=4.5 candidate
    promotion (PI decision 1a, 2026-06-13; June-13 manual §2.1/§1.2/§15.8).**
    The old early band `|A_fork(0.05) − A_independent| ≤ 0.10` contradicted the
    manual's treatment of the seed floor as a first-class positive quantity (§4)
    and was the SOLE reason Run-3 (otherwise clean: Gate-A OK at both cfgs,
    monotonicity/determinism/informativeness all pass) read FAIL. Replaced by a
    signed seed-floor rule, frozen here:
    - `g0 = A_fork(0.05) − A_independent` (SIGNED) must satisfy, per axis at the
      headline cfg: (i) `g0 ≥ G0_MIN` (= −0.02, numerical slack on the 16-clip
      mean; below it = anti-correlation, a normalization/A_independent defect);
      (ii) the headline Gate-A is exchangeable at s=0.05 (`per_s["0.05"].ok`);
      (iii) `g0 ≤ G0_MAX` (= 0.25; above it the model is near-deterministic from
      noise, leaving no trajectory phase to map → route to NEGATIVE/F-1).
    - Falsifiability preserved: the rule still fails on g0 < −0.02, g0 > 0.25, or
      s=0.05 marginal invalidity. The measured g0 is logged as CANDIDATE Fig-1
      seed-share content; the certified seed-share number is a Phase-1 result at
      scale, never a Stage-M claim.
    - Re-evaluation of Run-3 (zero GPU, same on-disk data): class g0 = 0.1526
      (≥ −0.02, ≤ 0.25, Gate-A ok @ s=0.05) → criterion 1 PASS →
      **`MICROMAP_PASS`**.
    - **Schedule-suffix mandate (§15.8):** `CFG_KERNEL_OK` tokens now carry the
      `schedule=` suffix (`CFG_KERNEL_OK(cfg=1, schedule=sqrt_down)` /
      `(cfg=4.5, schedule=sqrt_down)`); the evaluator asserts a single
      (cfg, schedule) provenance per cfg and writes `certified_kernels.json`;
      `foley_cw/kernel_provenance.assert_certified_kernel` is the code-side guard
      for Phase-1 commitment runners.
    - **cfg=4.5 is CANDIDATE, not ratified (§1.2/§15.8):** its OK token is from
      Stage-M pilot cells; promotion to a Phase-1 headline arm requires a Gate-A
      pass on the FULL Phase-1 independent pool (a Phase-1 activity), recorded by
      flipping `certified_kernels.json["deployed"].ratified` to true.
    Constants `G0_MIN`, `G0_MAX` are ratified read-only provenance henceforth.
