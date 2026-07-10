# PI Checkpoint — Stage-M Halt (foley-cw, 2026-06-11)

**Status: execution HALTED per the slice contract** (LONG_RANGE_EXPERIMENT_PLAN.md
Appendix E): Stage M emitted `MICROMAP_FAIL(endpoints, kernel_cfg4.5)` +
`GATE_A_UNDERPOWERED`. Stage-0 GPU screening, MLLM sidecar at scale, the
reliability gate and the manifest freeze are gated behind your decisions below.
Nothing in this document is scientific evidence (manual §1.5: Stage-M outputs are
diagnostics); no `GO_*` token has been emitted.

---

## 1. What was executed (all artifacts on disk, fully journaled/resumable)

| Step | Result | Artifact |
|---|---|---|
| Thresholds + pre-registrations frozen (pre-curve) | θ_rel .95 / θ_robust .85 / κ .6; §1.2 predictions + §6 gate language | `configs/thresholds.json`, `experiment/preregistered/` (SHA256) |
| FoleyBench extraction | 785/785 Discrete clips, 0 decode errors | `data/FoleyBench/clips/`, `clips_index.csv` |
| Stage-M selection | 16 clips (UCS-stratified) + exclusions + 400-clip screening pool | `data/manifests/*.json` |
| W3 tagger sanity gate | first pass FAILED at 0.45 (session log; artifact stores the final run) → reverted unauthorized group-sum to pre-registered top-1-then-map → 0.65 PASS; coarse map frozen v2 | `results/stage_m/tagger_sanity.json`, `configs/coarse_class_map.json` |
| Anchors (785 clips, dual-source) | coverage 1.00/1.00; σ_anchor median 0.757 s → bin 1.51 s | `results/stage0/anchor_report.md`, 30-clip template ready |
| qwen3.5-omni-plus judge | live-validated, 3 axes, temp 0, cached | `foley_cw/mllm_judge.py`, `configs/mllm_prompts/` |
| α pilot @ cfg=4.5 (manual §1.3) | **FORK_ALPHA_NO_VALID_OPERATING_POINT** | `results/stage_m/alpha_pilot.json` |
| α pilot @ cfg=1.0 | **PRIMARY_ALPHA=1.6** | `results/stage_m/alpha_pilot_cfg1.json` |
| **Full Stage M** (16 clips × cfg {1.0, 4.5}, α=1.6, n_steps=20, ≈973 FGE) | **MICROMAP_FAIL** (details §2) | `results/stage_m/{micromap_report.md, micromap_curves.csv, gate_a_report.md, tokens.json, logging_contract_audit.md}` |
| Codex cross-model reviews (4 passes) | 3 BLOCKING + 4 MAJOR + 3 MINOR found & fixed; final pass verified | `logs/codex_review_{1,2,3}.log` |
| Tests | 797+ passed, 0 failed | `tests/` |

Storage: 0.433 GB of the 100 GB cap, fully accounted (`logging_contract_audit.md`).

## 2. The five Stage-M criteria (manual §2)

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | Endpoints | **FAIL (late, class axis)** | clip-mean A_fork(0.90) on class = 0.625 (cfg 1.0) / 0.500 (cfg 4.5), need ≥ 0.90. Early endpoint passed at α=1.6. |
| 2 | Monotonicity | PASS | commit(s) non-decreasing within CI tolerance |
| 3 | Kernel @ cfg=4.5 | **FAIL** | SDE re-validation token=OK, but Gate-A = `GATE_A_UNDERPOWERED` (§4) |
| 4 | Measurer determinism | PASS | 1.0 / 1.0 on identical wavs (all 8 shards) |
| 5 | Informativeness | PASS (no warning) | 7/16 clips with A_independent>0.9 on class (< 12/16) |

## 3. Attribution analysis (why it failed — three separable causes)

All three attributions below are diagnostic ANALYSIS (hypotheses supported by,
but not certified by, the cited numbers — Gate A itself returned UNDERPOWERED and
certifies nothing).

**(a) The class measure has an argmax knife-edge (hypothesis, strongly supported).**
Label dump (clip 2322): PANNs top-1 vs top-2 probabilities differ by as little
as 0.003 (Speech 0.693 vs Music 0.690); tiny seed/tail differences flip the
label, and Krippendorff-α on K=8 amplifies one flip to a collapse. Embedding
cohesion is consistent with this reading: **fork ensembles at s=0.90 have mean
pairwise cosine 0.937/0.941 (cfg 1.0/4.5) versus 0.758/0.826 for independents**
— the late audio is highly similar while the label flickers. The criterion-1
late failure is therefore most consistent with a label-granularity artifact
rather than kernel divergence — pending confirmation by a redesigned measure.
(Also: Speech/Music as top classes on non-speech/non-music FoleyBench content is
tagger confusion by construction.)

**(b) A per-clip mode-locking pattern at cfg=4.5 (label space).** The α pilot
found NO valid operating point at the deployed cfg: e.g. clip 2322's forks stayed
on one class label at every piloted α∈{0.05..1.6} while its independent
generations are label-diverse; clip 1610 unlocked early at α≥0.4 but lost late
anchoring there. The pattern is consistent with the §1.2 "tilted pseudo-score"
risk Gate A exists to police, though (a)-type label noise contributes to it and
the redesigned measure must re-test it.

**(c) Embedding-level fork cohesion persisted at every piloted α.** Even at
cfg=1.0 with α=1.6 — where class-label marginals DO wash out — fork embeddings at
s=0.05 stayed more cohesive than independents (0.870 vs 0.758). If this holds up
under a redesigned instrument, it would preview the §4.3 determination taxonomy
(fine-grained texture ≈ seed-determined, coarse class ≈ trajectory-committed);
operationally it makes the §2 early-endpoint expectation unattainable for
embedding-sensitive measures within the piloted α grid, and it is the proximate
reason Gate-A's reference distribution broke (§4).

## 4. Why Gate-A returned UNDERPOWERED (instrument, not science)

Calibration found `fp_ref = 1.00`: at the EXACT kernel (cfg=1.0), 100% of
fork-vs-independent cells reject in raw-PANNs-embedding MMD — because of (c).
θ_MMD (0.472) then sits at the cross-clip MMD level (0.508), failing the
pre-registered separation guard: the threshold cannot distinguish "kernel broken"
from "different clip". The guards refused to certify (by design; manual §1.2:
never pass silently). The raw-embedding statistic appears mismatched in
granularity to the label-level science; the Gate-A STATISTIC needs redesign
before cfg=4.5 can be formally adjudicated: candidates = MMD on
tagger-probability vectors / label-marginal tests / larger K,N / per-clip pooling.

## 5. Decisions requested (in priority order)

1. **Class-axis measure redesign** (root cause (a); blocks everything downstream).
   Options: (i) restrict argmax to event classes (exclude speech/music/ambient
   groups — frozen-map amendment v3); (ii) abstain margin (label = "ambiguous"
   when top1−top2 < δ); (iii) make class an embedding axis (PANNs/CLAP cosine,
   sidestepping argmax); (iv) qwen-judge as the class measurer (semantic, slower).
   Agent recommendation: (i)+(ii) combined, then re-run Stage M (≈20 GPU-min).
2. **Gate-A statistic redesign** (§4): agent recommendation = MMD on the
   527-dim tagger-probability vectors + label-marginal TV, keeping the same
   calibration/guard scaffold.
3. **Ratify the §1.2 fallback direction** given (b): headline maps at cfg=1.0,
   deployed cfg=4.5 via readout-transfer + explicit scope note — or, if you
   prefer, direct a dedicated investigation of the cfg-4.5 mode-locking pattern
   as a CANDIDATE finding (it would need valid future evidence under a
   redesigned instrument; Stage-M diagnostics cannot support a paper claim).
4. **Instrument settings**: PRIMARY_ALPHA=1.6 at cfg=1.0 (token emitted); whether
   to also pilot early-heavy g(s) schedules (`linear_down`/`sqrt_down`, §1.3
   kernel-redesign route) to reduce tail noise while keeping early mixing.
5. **Amendments A-1..A-4** (Appendix E of the manual): Stage-M clips from
   FoleyBench (exclusions registered); audio-track anchor source; qwen as the
   MLLM; n_steps=20 grid alignment.
6. **Human-time items** (can start independently of 1-4): 30-clip anchor marks
   (`data/manifests/anchor_check_30.csv`); ~50-clip human validity sidecar.

## 6. What remains gated (not run, per contract)

Stage-0 A_independent screening (400 clips, runner ready: `scripts/stage0_screening.py`),
MLLM sidecar at scale (`scripts/stage0_mllm_sidecar.py`, ≤500-call budget),
reliability gate (`scripts/stage0_reliability.py`), manifest freeze, and all of
Phases 1–3. With decisions 1–2 made, the expected path to the next checkpoint is:
re-run Stage M (≈20 min) → if PASS → screening (≈1 h on an17+an29) → sidecar +
reliability → manifest freeze → `GO_MAPS_PHASE` sign-off.
