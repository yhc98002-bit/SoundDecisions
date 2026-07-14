# Arc-4 appendix capacity ledger

All times are 2026-07-15 Asia/Shanghai. Every GPU launch exposed exactly physical GPUs 4-7 as logical `cuda:0..3`, TP1 x 4 replicas. Physical GPUs 0-3 had resident co-tenant memory and were deliberately excluded; they were never queried or exposed.

| Node / devices | Assignment | Start-end | Recorded queue span | Observed active footprint | Produced inventory |
|---|---|---|---:|---|---|
| an12 / 4-7 | B1 collection | 00:48:03-01:19:32 | 31m29s | ~11.28 GiB/GPU, productive | 200 journals; 25,600 retap + 25,600 pooled bundles; 3,200 labels |
| an21 / 4-7 | B6 raw initial | 00:57:33-01:09:52 | 12m19s productive within a 4h00m11s allocation | ~11.28 GiB/GPU, productive | cfg4.5 128 pairs; cfg1.0 53 committed pairs before timeout |
| an29 / 4-7 | B2 seeds 0-4 | 00:57:37-01:49:47 | 52m10s | ~11.30 GiB/GPU; sampled 67-90% | 240 units; 23,280 WAVs |
| an12 / 4-7 | B6 cfg1 resume | 01:20:26-01:25:30 | 5m04s | ~11.28 GiB/GPU, productive | remaining 75 pairs; B6 final 128/cfg |
| an12 / 4-7 | B2 seeds 5-8 | 01:31:14-02:09:17 | 38m03s | ~11.29 GiB/GPU; sampled 50-90% | 192 units; 18,624 WAVs |
| an29 / 4-7 | B2 seeds 9-12 | 02:09:27-02:53:27 | 44m00s | ~11.29 GiB/GPU; sampled 54-91% typical | 192 units; 18,624 WAVs |
| an12 / 4-7 | B2 seeds 13-16 | 02:19:43-02:58:58 | 39m15s | ~11.29 GiB/GPU; sampled 43-80% typical | 192 units; 18,624 WAVs |

Idle accounting: an12 had 54s between B1 and B6 resume, the required 5m44s gap before seeds 5-8 (manifest/test/freeze), and 10m26s before seeds 13-16 (CPU validation plus authorization/freeze). an29 had 19m40s between primary and seeds 9-12: 7m47s full rehash plus 11m53s awaiting the binding B1/capacity decision and freezing the extension.

Initialization was inside launch wall time: seeds 9-12 took about 4m44s and seeds 13-16 about 2m04s to move from local-weight I/O to active generation. Post-generation GPU idle was intentional during CUDA-hidden rehash: primary 7m47s, seeds 9-12 4m54s, seeds 13-16 4m51s; seeds 5-8 completed validation at 02:17:33 after one deferred attempt released I/O for an29 initialization.

Slurm allocation 97747 ended `TIMEOUT` after 4:00:11; pending job 97896 was canceled before start and B6 resumed on an12. At 02:18, an21 had no active/queued allocation and direct SSH returned access denied, so no work was forced there. All guards observed 81,226 MiB free per selected GPU before launch; all weights were local/offline.
