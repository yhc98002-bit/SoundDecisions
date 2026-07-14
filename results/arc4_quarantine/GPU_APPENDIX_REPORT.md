# Arc-4 GPU appendix execution report

Status: **COMPLETE**. B1 is fail-closed; B2/B6 contain raw generation plus integrity metadata only. No B2/B6 generated-audio value was measured, inspected, summarized, or used for a token.

## Frozen authority

- Worktree/branch: `SoundDecisions-arc4-gpu` / `arc4-gpu`; generation and primary validation used `PYTHONHASHSEED=0`, local/offline weights, and the primary `.venv`. A deliberate B1 audit-test rerun used hash seed 1.
- `B1_PROTOCOL.md` (SHA `b85eee...`) is explicit V1 audit history. Its attempted gate log was zero bytes; no gate artifact, prediction, metric, or exit code existed.
- V2 protocol SHA `1386287a...` and binding Amendment-4 SHA `2283ae...` are authoritative. Earlier V2 amendments remain audit history.
- The raw inventory and exact ledger hashes are in `QUARANTINE_MANIFEST.json`; node/device timing is in `CAPACITY_LEDGER.md`.

## B1 boundary

Collection completed: 200 journals, 3,200 labels, 25,600 retap bundles, and 25,600 pooled bundles. The binding whole-bundle join stopped on sorted bundle 31 (`1002__p1cfg1_ind12__s0.75.npz`): relative L2 `0.0002585371085837806` exceeded `0.0002`.

Authoritative status is **B1_INCOMPLETE**, `scientific_token=null`. Labels were used only for exact-set/universe validation; no distribution, probe fit, prediction, metric, gate-pass artifact, or readability interpretation exists. Evidence: `results/arc4_b1/B1_CLOSEOUT.md` and `results/arc4_b1/job_manifests/arc4_b1_probe_v2_gate.json`.

## Capacity and inventory

| Queue | Node / physical GPUs | Recorded queue span | Integrity inventory |
|---|---|---:|---|
| B1 collection | an12 / 4-7 | 31m29s | complete counts above |
| B2 seeds 0-4 | an29 / 4-7 | 52m10s | 240 units; 23,280 WAVs |
| B2 seeds 5-8 | an12 / 4-7 | 38m03s | 192 units; 18,624 WAVs |
| B2 seeds 9-12 | an29 / 4-7 | 44m00s | 192 units; 18,624 WAVs |
| B2 seeds 13-16 | an12 / 4-7 | 39m15s | 192 units; 18,624 WAVs |
| B6 initial/resume | an21 then an12 / 4-7 | 12m19s productive in a 4h00m11s allocation + 5m04s resume | 128 pairs and 1,280 WAVs per cfg |

All launches used TP1 x 4 replicas and exposed only physical GPUs 4-7. Physical GPUs 0-3 had resident co-tenant memory and were deliberately excluded; they were never touched. Guards saw 81,226 MiB free/GPU. Active workers held about 11.3 GiB/GPU; sampled utilization was typically 43-91% outside initialization/clip transitions.

B2 totals are 17 disjoint seeds, 816 units, 6,528 cells, 816 base WAVs, and 78,336 fork WAVs (79,152 WAVs; 40,532,156,160 bytes). Every cohort's four fresh CUDA-hidden CPU validators reported `todo_units=0`. B6 has 128 pair journals/1,280 WAVs per cfg; all eight CPU validators reported `todo=0`.

## Quarantine and deviations

- B6's original allocation timed out after cfg4.5 and 53 cfg1 pairs; the an12 resume completed the remaining 75. Pending Slurm job 97896 never started. an21 later denied access because no allocation existed.
- an12's required 5m44s gap and an29's 19m40s gap, local-weight initialization, validation idle, and all placement/utilization samples are accounted in `CAPACITY_LEDGER.md`. No further wave ran beyond the deliberate 17-seed bound.
- A slow shared-memory staging attempt was canceled at 3.9/6.0 GiB before any model process or GPU allocation. One seeds 5-8 validation attempt was deferred to release I/O for an29 initialization, then rerun to PASS.
- Primary storage has zero retained final WAVs; categorical class embeddings are null and cannot reconstruct 527-way posteriors. Resolution is recover/rerun final audio and persist `clipwise_output`; see `RETAG_ASSESSMENT.md`.

## Verification

- Focused raw-queue tests: `7 passed` (`tests/test_arc4_gpu.py`). Independent B1 audit: `25 passed` with hash seed 0 and `18 passed` with hash seed 1.
- Completion records hash every launch ledger/log tree/journal tree. B2 validators: 16/16 PASS; B6 validators: 8/8 PASS; all runner exits are 0 except the recorded B1 fail-closed gate and an21 allocation timeout.
- No GPU tmux session or generation process remains; an12/an29 GPUs 4-7 were released at closeout.
