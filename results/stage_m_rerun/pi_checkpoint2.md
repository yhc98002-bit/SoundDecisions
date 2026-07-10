# PI Checkpoint #2 — Stage-M Halt After Run 3 (foley-cw, 2026-06-12)

**Status: HALTED per the frozen iteration bound** (`stage_m_rerun_interpretations.md`
#10: at most one Stage-M attempt after the schedule pilots). Run 3 passed 4 of 5
criteria — including Gate-A at BOTH cfgs — and failed only the early-endpoint
label band by 0.053. Stage-0 screening and everything downstream remain gated on
your decisions below. Stage-M outputs are diagnostics, never evidence.

## 1. The three-run instrument trajectory (all journaled)

| Run | Setup | Verdict | What it established |
|---|---|---|---|
| 1 (2026-06-11) | old measure, constant g, α=1.6, seed 0 | FAIL | class argmax knife-edge; per-seed embedding Gate-A conflated seed-conditioning with kernel error |
| 2 (2026-06-12) | revised measure (event-restricted + abstain δ=0.05), seed-marginalized Gate-A, fresh refs, seed 1 | FAIL | the NEW Gate-A flagged a fork-marginal deviation AT THE EXACT KERNEL (16/16 internal-null rejections at s=0.05) — consistent with constant-g α=1.6 discretization bias (analysis); tail noise broke late endpoints/monotonicity/abstain cap |
| 3 (2026-06-12) | **(cfg=1.0, sqrt_down, α=0.8)** — pilot-selected (8 tuples), Gate-A PRECHECK passed first | **4/5 PASS**; endpoints FAIL by one number | instrument clean; see §2 |

Schedule pilots (8 tuples, seed 1): cfg=1.0 sqrt_down/linear_down valid at α=0.8;
cfg=1.0 constant — no valid α; all guided-cfg (2.5/4.5) pilots — no valid
constant-α point; the n_steps=40 arm did NOT unlock cfg=4.5 (locking is
dynamics, not integrator error). α-response on sqrt_down: gap 0.59 (α=.05) →
0.144 (α=.4) → **0.052 (α=.8, pilot 3 clips)** → 0.111 + late degradation
(α=1.6). Full pilot table: `results/stage_m_rerun/pilot_*.json`.

## 2. Run-3 verdict (16 clips × cfg {1.0, 4.5}, ≈1.55k FGE, journals complete)

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | Endpoints | **FAIL (early, class)** | clip-mean \|A_fork(0.05) − A_ind\| = **0.153** vs the 0.10 band. Late endpoint, scorability (≥12/16), and BOTH embedding rules (E1 seed floor, E2 growth) passed. |
| 2 | Monotonicity | PASS | confident-subset commit curves non-decreasing |
| 3 | Kernel | **PASS** | internal null @ cfg=1.0: `CFG_KERNEL_OK(cfg=1)` (1/16 low-p at both s-points in the precheck; full-run consistent). Adjudicated @ cfg=4.5: **`CFG_KERNEL_OK`** — under sqrt_down α=0.8 the seed-marginalized pools are exchangeable with fresh references EVEN AT DEPLOYED GUIDANCE (labeling caveat: the evaluator printed the token without the `schedule=sqrt_down` suffix; the cells were generated under sqrt_down). |
| 4 | Determinism | PASS | 1.0 / 1.0 |
| 5 | Informativeness | PASS | 2/16 video-pinned; abstain@0.90 under the 0.30 cap |

## 3. Reading of the one remaining number (analysis, not a claim)

The 0.153 early gap coexists with a PASSING marginal-exchangeability test at the
same s-point. These measure different things: Gate-A pools ONE fork per seed
(marginal validity — clean); criterion 1 measures WITHIN-seed K-fork agreement
(conditional concentration). The residual is consistent with a **label-space
seed floor ≈ 0.15 at the achievable α optimum** — a HYPOTHESIS, not an
established result: if ratified, it is exactly the quantity the spec's own
determination budget (§4) is designed to quantify as the "seed share" in
Phase 1, rather than an instrument defect. The pilot α-response suggests no α
does better than ≈0.8 on this population (1.6 over-noises the tail; 0.4
under-mixes). We did NOT run a 4th attempt (frozen bound).

## 4. Decisions requested

1. **The early-endpoint band vs the determination budget.** Options:
   (a) register the early-gap quantity as CANDIDATE Fig.-1 content (under the
   seed-share hypothesis, to be quantified properly in Phase 1) and re-scope
   criterion 1's early rule to washout DIRECTION + Gate-A exchangeability
   (both already pass) — Stage M then passes as-is (recommendation; zero GPU
   cost; one re-evaluation);
   (b) widen the band to ≤0.20 (passes; weaker); or
   (c) authorize ONE α micro-pilot (e.g. α ∈ {1.0, 1.2} sqrt_down, ~10 min) +
   a 4th run — the α-response curve suggests limited headroom.
2. **Ratify `CFG_KERNEL_OK(cfg=4.5, schedule=sqrt_down)` as a §1.2 re-entry
   token** (with the labeling caveat fixed in the evaluator). If ratified, the
   deployed-cfg commitment grid re-enters Phase 1 — a major scope upgrade the
   spec explicitly anticipated.
3. **Ratify the 11 frozen interpretations** in
   `experiment/preregistered/stage_m_rerun_interpretations.md` (δ=0.05, sqrt
   transform, fresh refs, exact-match gating, scorability ≥12/16, paired CIs,
   g3-p95 refinement, iteration bound, tagger-sanity confident reading, tuple
   selection record).
4. **Human-time items** (independent of 1–3): 30-clip anchor marks
   (`data/manifests/anchor_check_30.csv`; also arbitrates audio-only anchors at
   σ ≤ 0.35 s), ~50-clip human validity sidecar.
5. **Stage-0 launch** upon resolution of (1): screening at the headline cfg
   (runner ready, ~1.5 h on an17+an29), MLLM sidecar (≤500-call budget),
   reliability gate, manifest freeze — feeding the Stage-0 token gate, which
   itself awaits the human items in (4) plus your sign-off.

## 5. Artifacts

`results/stage_m_rerun/` — run-3 journals/curves/reports/tokens; `run2_archive/`
— run-2 full record; `pilot_*.json` (8 tuples) + `precheck_sqrtdown_a0.8.json`;
`tagger_sanity.json` (confident 0.733, BEATs armed-not-triggered);
`experiment/preregistered/` — all freezes with SHA256; 834 tests green; 4 Codex
review rounds this slice (logs `logs/codex_s2_*.log`, `logs/codex_review_*.log`).
Compute this slice: completed Stage-M runs 3,097.6 FGE (journal-verified: run 2
1,548.8 + run 3 1,548.8) plus pilots/prechecks/smoke ≈1.5k FGE (estimated);
storage 1.566 GB of the 100 GB cap, fully accounted.
