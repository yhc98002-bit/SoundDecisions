# Phase-4 OFFLINE policy Pareto (manual §9, Fig. 6)

Split: **all** (200 clips). Seed 0. Deployed cfg=4.5, num_steps=25. DiffRS τ=0.9976, SMC T=0.1, random-prune frac=0.5.

**PROXY-CORRECTNESS CAVEAT.** `correctness` here is the ORACLE PROXY = agreement with the per-clip MAJORITY self-target across independents (preregistration). It is a self-consistency proxy, NOT human/MLLM correctness-vs-video. No correctness claim follows from these numbers alone; the offline pass is a headroom screen + method illustration (manual §9, §1.7).

Oracle gate windows from s_commit (determination_budget_p1cfg45.csv):
- window s=0.25: prune on presence, timing
- window s=0.35: prune on class
- window s=0.75: prune on material

| policy | final_corr | NFE | scoring | winner_ret | false_prune | regret |
|---|---|---|---|---|---|---|
| full_bon | 0.370 | 80000 | 3200 | 1.000 | 0.000 | 0.590 |
| same_compute_bon | 0.365 | 51525 | 2061 | 0.635 | 0.340 | 0.595 |
| random_prune | 0.330 | 49600 | 4800 | 0.485 | 0.475 | 0.630 |
| diffrs_scalar | 0.370 | 50512 | 4848 | 1.000 | 0.210 | 0.590 |
| smc_scalar | 0.370 | 58511 | 5269 | 0.690 | 0.350 | 0.590 |
| final_rerank | 0.370 | 80000 | 3200 | 1.000 | 0.000 | 0.590 |
| oracle_axis_gated | 0.785 | 49151 | 8437 | 0.475 | 0.000 | 0.175 |

**Two-axis Pareto points** `(generator-NFE, final proxy-correctness)` and `(scoring-calls, final proxy-correctness)`:

| policy | (NFE, final_corr) | (scoring, final_corr) |
|---|---|---|
| full_bon | (80000, 0.370) | (3200, 0.370) |
| same_compute_bon | (51525, 0.365) | (2061, 0.365) |
| random_prune | (49600, 0.330) | (4800, 0.330) |
| diffrs_scalar | (50512, 0.370) | (4848, 0.370) |
| smc_scalar | (58511, 0.370) | (5269, 0.370) |
| final_rerank | (80000, 0.370) | (3200, 0.370) |
| oracle_axis_gated | (49151, 0.785) | (8437, 0.785) |

## Conservatively rounded-up compute comparator (oracle_axis_gated vs same_compute_bon)
- same_compute_bon is allocated by ceiling per clip, so the baseline receives 4.8% more aggregate NFE (51,525 vs 49,151) than oracle_axis_gated.
- oracle_axis_gated: final_corr=0.785, NFE=49151, false_prune=0.000.
- same_compute_bon: final_corr=0.365, NFE=51525.
- full_bon ceiling: final_corr=0.370, NFE=80000.
- random_prune control: final_corr=0.330, false_prune=0.475.

**Offline headroom (proxy): YES** — gated_final=0.785, same_compute_bon_final=0.365, gated_nfe=49151, same_compute_bon_nfe=51525; criterion: gated_final >= same_compute_bon_final + 0.01 and gated_nfe <= same_compute_bon_nfe * 1.02. Token routing (§9) is a human STOP decision; this report does not emit GO_POLICY / GO_RESTRICTED / DIAGNOSTIC_ONLY on its own.
