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

## Checkpoint 5 — complete posterior shards and clean replay relaunch gate

- Status: active; Class posteriors are measured but not yet scientifically
  analysed, and the B-1 calibration/held-out gate has not been reduced.
- All seven Class shards completed and passed independent inventory-bound
  validation. Their exact union is 79,152 records; the seven completion hashes
  are recorded in the run ledger and will be rebound by the canonical merge.
- The corrected B-1 packet attempt passed recursive validation for the exact
  five clips, eight registered progress points, and 40 units. Completion
  SHA-256 is
  `9b8e2de82d62f69641e98809f67eb3cde69d9bbadb94a3229bccb1b4745a384a`.
- The first Class merge attempt failed before output creation because the
  reducer treated execution batch size as a cross-shard scientific invariant.
  The workers deliberately used batch 32 on the idle GPU and batch 8 on
  colocated GPUs. The reducer now keeps batch size per input shard while still
  requiring identical protocol, checkpoint, taxonomy, abstention, measurer,
  and inventory provenance. The complete Class suite passes 21/21.
- Two calibration replay attempts were interrupted and quarantined at 30/32
  and 31/32 completed units, respectively, after an on-disk source edit was
  detected while their Python processes were alive. This prevents a late
  provenance snapshot from hashing code different from the already-loaded
  module. Neither attempt has a root completion, neither will enter tolerance
  calibration, and clip `1002` remains unreplayed. Fresh attempts will launch
  only from the next clean commit.

## Checkpoint 6 — Class result and same-forward lineage gate

- Status: complete.
- The canonical Class merge revalidated the exact 79,152-record B2 union.
  Completion SHA-256 is
  `f2bfa420782444bc37b21f9379f535f9592b47889a625f81880abbebd620d3df`;
  posterior-data SHA-256 is
  `a381e0f8e662aae482959f0cd2e7fe46f3cb1e8ef2ef7040b4e65913ff86cd1e`.
- Under the registered pooled sustained commitment criterion (theta 0.70),
  the point crossing is `s=0.90`. Of 5,000 video-bootstrap draws, 3,767 cross
  and 1,233 are noncrossing; the conditional percentile range among crossing
  draws is `[0.75, 0.90]`. The historical `s≈0.35` result therefore did not
  reproduce and is assigned `NOT_SUPPORTED` in this exploratory multi-seed
  replication. Individual crossings remain heterogeneous and often early;
  they are not collapsed into a first-crossing mean.
- The registered variance decomposition attributes 42.19% to video, 0% to an
  additive base-seed component, 41.63% to video-by-seed interaction, 14.93% to
  fork Monte Carlo non-abstention variance, and 1.25% to the identifiable
  abstention subcomponent. Measurer repeatability is not identifiable from one
  measurement per WAV.
- The four calibration clips produced the tolerance artifact without access to
  clip `1002`; tolerance completion SHA-256 is
  `26a498c969c145405f548d84ee159f6718059668b2a3b5318a1934d1abd5bd19`.
  Two held-out launch attempts then failed before model load because the pinned
  environment omitted `HF_HUB_DISABLE_XET=1`; both logs are retained and the
  scientific tolerance was unchanged.
- Two fresh, independent held-out replays of clip `1002` passed all 1,960
  frozen same-forward equivalence checks. Held-out completion SHA-256 is
  `96fe3e6909b84597d97617acfa27f5912a5d6d4e643b63539b7c811a46827a5c`.
  The B-1 identity gate therefore passed and full feature recollection was
  authorized.

## Checkpoint 7 — full feature recollection and readout launch

- Status: active; full features and targets are complete, nested-CV fits are
  running without a scientific conclusion at this checkpoint.
- Eight immutable feature workers on `an12` GPUs 4--7 produced 816 base
  trajectories / 6,528 state units. The shards are disjoint, complete, and
  bound to collector SHA-256
  `4de307abb9b5241a1ca2dc7f83ef2f3bcd18730a16b23c79b9dfa04d83f2d5a7`.
  All `an29` GPUs were occupied at launch, so they were not used.
- A serial merge attempt was intentionally terminated before output creation
  because immutable-shard validation was needlessly sequential. The reducer
  now validates shards concurrently while preserving input order; the focused
  recollection/readout suites pass 15/15. No scientific rule changed.
- The canonical merge contains 6,528 units, 816 trajectories, 48 videos, and
  all eight progress points. Completion SHA-256 is
  `351e191a55d6d3eaa57ac0d1a1081bcf77d8df46ce331773a271275f906cf759`;
  manifest SHA-256 is
  `1aca68300a485e4831f73aae286026bf69a97d7545bca3ce9eb702e24b968c3c`.
- The immutable target manifest contains 6,528 candidates. Fork-majority
  observed counts by progress are 709, 729, 717, 723, 720, 712, 699, and 675;
  ODE-final coverage is 576/816 at every progress. Missing targets remain
  explicit and are never imputed. Target completion SHA-256 is
  `a633a27012b5683a29f6fae03798fea81a267f947364a00d4a7ccf4b7196fa69`.
- Occupancy was re-queried immediately before fitting: `an12` GPUs 4--7 each
  had 81,226 MiB free and no compute process. Eight independent TP1 progress
  shards were launched as two replicas per GPU, with offline mode,
  deterministic CuBLAS, and four CPU threads per worker. Launch source commit:
  `b2a8687a4ec831306332cb6e837b6b5b7dd7eb8d`.
