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
