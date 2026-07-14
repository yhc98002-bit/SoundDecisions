# Arc-4 GPU appendix execution report

Status: **RUNNING**. This report records protocol, placement, artifact integrity,
and queue progress only. No B-2 or B-6 generated-audio value has been measured,
inspected, summarized, or used for a decision token.

## Isolation and frozen inputs

- Worktree: `SoundDecisions-arc4-gpu`, branch `arc4-gpu`.
- B-1 protocol: `experiment/preregistered/B1_PROTOCOL.md`, SHA256
  `b85eeece6f18ff7ce3ab254411d06f97cf2446d393f74eb81ad34048131cc03f`,
  frozen in commit `9586b13` before any B-1 probe evaluation.
- B-2 manifest: SHA256
  `5c3a334ecfcfb3e91504354c14c8e8dbae71b3bade088b21bec26fb06fd68ed3`.
- B-6 pair manifest: SHA256
  `640b00059e04e9db51cf5b4f267fc901612aa57eea0d59b12b8bb198dd8bdc4e`.
- Raw queue correction: commit `0b3f27f`; corrected launch ledger: commit
  `dbd40d9`.

## Placement and environment

| Queue | Node/session | Physical GPUs | Logical workers | Placement |
|---|---|---|---|---|
| B-1 recollection | `an12:arc4_b1_collect` | 4,5,6,7 | `cuda:0..3` | TP1 x 4 replicas |
| B-2 raw multi-seed | `an29:arc4_b2_generate` | 4,5,6,7 | `cuda:0..3` | TP1 x 4 replicas |
| B-6 raw swaps | `an21:arc4_b6_raw` | 4,5,6,7 | `cuda:0..3` | TP1 x 4 replicas |

Every launch queried only physical GPUs 4-7 and observed 81,226 MiB free per
selected GPU, above the 71,680 MiB guard. All runners expose exactly
`CUDA_VISIBLE_DEVICES=4,5,6,7`, use the shared primary `.venv/bin/python`, set
`FOLEY_CW_WEIGHTS_SOURCE=hf`, force offline HF/Transformers operation, and set
`PYTHONHASHSEED=0`. The shared environment replaced an abandoned `/dev/shm`
copy that had reached only 3.9/6.0 GiB after about 15 minutes; no model process
or GPU allocation occurred during that staging attempt.

## B-1

The collector regenerates 200 clips x 16 independents x 8 progress points. The
mandatory evaluation gate requires 200 valid journals, exactly 25,600 valid
per-token bundles, 25,600 matching pooled bundles, and 3,200 final class labels.
Partial evaluation is impossible through `scripts/arc4_b1_probe.py`.

Snapshot at 2026-07-15 01:03 +08:00: 80 completed clip journals and 10,568
per-token bundles, including the currently active partial clips. Four owned GPU
PIDs held about 11,276 MiB each and were productively executing. No B-1 probe
evaluation has started.

## B-2

The legacy conditioning-feature extractor was caught before Python execution
and produced no B-2 artifact. Its replacement is an axis-agnostic raw generator:
48 clips selected by a seeded SHA256 rank from the frozen 200-clip manifest,
base seeds 0-4, cfg 4.5, `sqrt_down`, alpha 0.8, the standard eight-point grid,
and K=12. Frozen cardinality is 240 base WAVs and 23,040 fork WAVs. Each shard
owns 12 clips, 60 clip/seed units, and 5,760 fork WAVs.

Artifacts are atomic mono 16 kHz IEEE-float WAVs with SHA256, byte, frame,
format, subtype, seed-lineage, kernel-ledger, command, node, GPU, and git
metadata. Completion is journaled per clip/seed/progress cell and rolled up per
clip/seed. Existing valid files are resumed; malformed files or journals abort
without replacement. There is no measurement or aggregate mode.

Snapshot at 2026-07-15 01:03 +08:00: all four workers passed the frozen manifest
and ratified-kernel guards, owned about 11,276 MiB per GPU, and completed their
first clip/seed unit in about 50 seconds. Four unit journals, 43 cell journals,
and 536 WAVs had landed including active partial units. No raw value was read.

## B-6

The frozen design contains 128 unique ordered cross-class source/donor pairs per
cfg in `{1.0, 4.5}`, with no self or same-cached-class pair, and the standard
eight-point grid. The initial implementation incorrectly instantiated a tagger
and would have written derived targets; it failed before generation on a missing
worktree PANNs path. The corrected queue removes all measurer and target code.

For every pair it atomically retains measurement-ready IEEE-float source, donor,
and eight swapped WAVs with hash-validated resume journals. Cached class labels
are retained only as frozen stratification provenance. There is no aggregate
mode and no B-6 decision token.

Snapshot at 2026-07-15 01:03 +08:00: the cfg4.5 wave had 86/128 completed pair
journals and 860 raw WAVs. Four owned GPU PIDs
held about 11,282 MiB each and were productively executing.

## Posterior retag assessment

The primary Stage-0 store has 350,556 measurement rows, including 89,367 class
rows, but zero retained WAVs anywhere under `results/`. Class rows retain only
`{axis_id, embedding, kind, label}`, not a 527-way posterior. Historical cached
finals therefore cannot be retagged without regeneration.

The cheapest forward resolution is now implemented for B-6: retain its raw
float WAVs, then run tagging later as a separate quarantined evaluation. This
does not add a tagger dependency to generation and does not recover historical
cohorts outside B-6.

## Verification

- Targeted queue tests: `7 passed` (`tests/test_arc4_gpu.py`).
- B-1 protocol, B-2 manifest, and B-6 pair-manifest hashes verified.
- B-1, B-2, and B-6 workers map one-to-one onto physical GPUs 4-7.
- B-2/B-6 quarantine boundary: raw audio plus integrity metadata only.
- Remaining gate: complete all collections; run B-1 full-bundle validation,
  then and only then run the frozen B-1 evaluation on `an12`.
