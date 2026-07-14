# Arc-4 WP-A2 class reconstruction

Raw class measurements were streamed from the RunStore JSONL and joined using structured `extra` keys; `gen_id` was not parsed.

Cardinality: independent `3200` = `200 x 16`; forks `19200` = `200 x 8 x 12`. All join keys are unique.

Committed class CSV reproduction: **PASS**, 4800 numeric/NaN cells, maximum absolute delta `0` (tolerance `1e-09`).

Naive-vs-confident power check: **PASS**.

| lens | A_ind | crossing clips | s_commit | s_read | gap |
|---|---:|---:|---:|---:|---:|
| confident (abstains excluded) | 0.377680 | 172 | 0.345930 | 0.75 | 0.404070 |
| naive (abstain is a label) | 0.337000 | 135 | 0.462593 | 0.75 | 0.287407 |

Confident-subset unscorable cells: **85**.

| s | abstain rate | unscorable | A_fork confident | A_fork naive | commit confident | commit naive |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.290833 | 3 | 0.567132 | 0.472955 | 0.351625 | 0.247144 |
| 0.15 | 0.309583 | 6 | 0.595401 | 0.503030 | 0.397866 | 0.283798 |
| 0.25 | 0.305417 | 8 | 0.644015 | 0.548258 | 0.440853 | 0.341740 |
| 0.35 | 0.310833 | 8 | 0.708689 | 0.576970 | 0.529758 | 0.379023 |
| 0.45 | 0.297083 | 7 | 0.761728 | 0.615152 | 0.599839 | 0.427416 |
| 0.60 | 0.297500 | 12 | 0.808320 | 0.664318 | 0.668476 | 0.495647 |
| 0.75 | 0.292500 | 16 | 0.878489 | 0.711667 | 0.770541 | 0.560104 |
| 0.90 | 0.295000 | 25 | 0.917889 | 0.801515 | 0.826931 | 0.687697 |

Independent-label abstain rate: `0.323125`; independent unscorable cells: `0`.
