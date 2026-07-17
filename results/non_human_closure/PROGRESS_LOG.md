# Non-human experimental closure: progress log

## Checkpoint 0 — provenance and scope

- Status: complete.
- PR #1 is open/unmerged; reviewed head is exactly
  `a1e8f3ae324e8886379c19c5bc312d7ebc942946`.
- Created clean worktree and branch `agent/non-human-closure` from that head.
- Preserved the primary checkout's unrelated modifications to `AGENTS.md`,
  `CLAUDE.md`, and `arc4_wpA2_verify.py` without touching them.
- Read the canonical status, Axis Specification v2, WP-A/WP-A2 reports, Goal-1
  asset audit, B2 quarantine manifest, and B-1 closeout.
- Environment reused (not created):
  `/XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions/.venv`;
  Python 3.10.12, NumPy 2.0.2, SciPy 1.15.3, pytest 9.0.3,
  torch/torchaudio 2.5.1+cu121.
- Pending at this checkpoint: physical artifact validation, current GPU
  occupancy, implementation audit, and all scientific measurement.

## Checkpoint 1 — inventory and outcome-independent protocol

- Status: complete.
- Verified all four B2 roots: 79,152 immutable WAVs, 816 unit journals,
  6,528 cell journals, and 40,532,156,160 WAV bytes. No B2 posterior exists.
- Verified the legacy Material inventory: 6,400 scalar Phase-2 cells and 3,200
  retained final CLAP embeddings, but zero candidate preview WAVs/embeddings.
- Verified B-1 historical caches are diagnostic only and that no same-forward
  collector exists.
- Live safe GPU candidates at the inventory snapshot were `an12:4`,
  `an12:7`, and `an29:1`; every launch must re-query occupancy.
- `an29:/tmp` was full. Scientific outputs will use immutable XYFS roots;
  `/dev/shm` may be used only for ephemeral staging.
- ModelScope mirrors are absent. Pinned local HF assets are available; any use
  is offline and recorded as a deviation.
- Frozen `experiment/non_human_closure/PROTOCOL.json` before inspecting B2
  posteriors, Material candidate margins, or held-out clip 1002.
- Validation: JSON parse and `git diff --check` (see checkpoint commit).

## Checkpoint 2 — pre-launch implementation audits and Material gate

- Status: partial; Class and B-1 launches remain blocked on code corrections.
- An independent Class audit rejected the first implementation before posterior
  measurement. It found frozen-default drift, insufficient pinned-asset
  enforcement, incorrect unscorable crossing semantics, and missing required
  replication/variance summaries. The defects are being corrected with
  regression tests; no B2 posterior has been produced or inspected.
- An independent B-1 audit rejected the first pilot implementation before any
  GPU replay or held-out access. It found an in-place Tweedie mutation, a
  vacuous within-call comparator, missing fresh-device/repeat reducers, and
  incomplete held-out isolation and validation. The held-out clip `1002`
  remains unopened.
- The outcome-blind Material feasibility gate was reproduced into immutable
  storage. Exact inventory was 200 videos / 6,400 cells, but strict existing-
  metadata matching produced only 3 complete four-subject videos / 96 cells,
  below the frozen 20-video / 640-cell floor. Workstream C therefore stopped as
  `INCOMPLETE_ARTIFACTS` before candidate replay or margin measurement.
- The first read-only B2 inventory attempt on the login node exhausted memory
  after hashing part of the bank and exited without a completion artifact. This
  is an engineering attempt, not a scientific result. The retry will run on a
  compute node with substantially more available RAM after the audited code is
  committed.
- GPU occupancy snapshot at `2026-07-17T15:36:16+08:00`: only `an12:4`
  (`GPU-cc0f...`) and `an12:7` (`GPU-f76f...`) were idle with 81,226 MiB free.
  All `an29` GPUs and the other `an12` GPUs had active neighboring jobs.

## Checkpoint 3 — audited launch code and canonical B2 inventory

- Status: complete.
- Hardened B-1 lineage code is committed through `440fd96`. The final protocol
  SHA-256 is `5c4fc4025995c16e355feb8cc02fbb3627891d47f6df052becde4845eaa7bd09`.
  The focused B-1 suite passes 7/7; an integration-topology correction records
  fused blocks as latent self-attention rather than inventing a clip-key map.
- Hardened Class code is committed through `f0c4649`; focused Class tests pass
  19/19 after adding a fail-before-model deterministic-CuBLAS environment gate.
- Four immutable inventory workers independently hashed and header-validated
  all four B2 roots. The fail-closed merge validated the exact union: 79,152
  records = 816 base finals + 78,336 fork finals, 48 videos, seeds 0--16,
  eight progress points, and 12 forks per state. Merged manifest SHA-256:
  `a5c5e721650e486f09fe70231d96e61f85904e0f17dc793ff3088b67646c3df2`.
- The deterministic B-1 selection/asset gate passed without model replay.
  Selection completion SHA-256:
  `022f6157ecd58c48493f8260ce321061d7e2c8c8d33abee0604e8ef894e21e7a`.
- First Class smoke attempt: `ENGINEERING_FAILURE` before artifact creation.
  PyTorch deterministic mode correctly rejected CuBLAS without
  `CUBLAS_WORKSPACE_CONFIG`; the traceback is preserved with SHA-256
  `c48f6f83566c82fd8d80ca8212dfda8f0a7a4f764f5a6266505a29e7ec13a65a`.
  No threshold, taxonomy, or scientific rule changed. The retry uses the
  documented `:4096:8` setting and a new immutable output root.

## Checkpoint 4 — Class measurement launch and B-1 packet retry

- Status: active; no Class outcome analysis or held-out lineage replay has
  occurred at this checkpoint.
- The second Class smoke wrote a complete-looking shard, but independent
  validation rejected it because the persisted normalized coarse posterior
  was not bitwise derived from the persisted fp32 coarse sums. The invalid
  artifact remains quarantined; completion SHA-256 is
  `3f2016b147d428e4e0a9831426f1500f1b286cd72757cefe013b03b0929d2426`.
- The persisted-posterior lineage was corrected without changing the coarse
  map, abstention rule, or any scientific threshold. The third smoke passed
  full inventory-bound validation; completion SHA-256 is
  `8436b9e84067e28f80f36fa76a9be41562628c92226f10959b2c9dda22e5810d`.
- Seven immutable Class measurement shards were launched across every A800
  with sufficient observed memory on `an12` and `an29`. The first three
  shards completed with 11,308 records each and independently passed the
  inventory-bound validator; the four `an29` shards remain active.
- The first B-1 packet attempt stopped while writing the first unit because
  PyTorch exposed the device UUID as a non-JSON `_CUuuid` object. It produced
  no completed unit or attempt and did not run a same-forward replay. The
  partial immutable root and log (SHA-256
  `c904c21ca145f9246761bbfd49a8195a6e45b15cf66892d4891373a1b34a83b2`)
  are preserved. The provenance encoder now converts the UUID to its canonical
  string, with a regression test; this is an engineering retry, not a gate
  failure and not evidence about clip `1002`.
