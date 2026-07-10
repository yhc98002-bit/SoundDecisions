# Phase-1 Determination Budget + Taxonomy (p1cfg1) — Fig 1

θ_commit = 0.7; bootstrap by video. Shares clipped at 0; s_min = seed floor. class is DIAGNOSTIC (kept).

| axis | n | conditioning | seed | trajectory | residual | s_commit |
|---|---|---|---|---|---|---|
| presence | 200 | 0.813 | 0.099 | 0.082 | 0.054 | 0.214 |
| timing | 200 | 0.903 | 0.083 | 0.033 | 0.007 | 0.114 |
| class | 200 | 0.378 | 0.231 | 0.350 | 0.082 | 0.346 |
| material (emb) | 200 | 0.637 | 0.207 | 0.144 | 0.013 | 0.638 |

## Taxonomy (clip counts per axis)
| axis | video-det | seed-det | traj-early | traj-mid | traj-late | never |
|---|---|---|---|---|---|---|
| presence | 70 | 67 | 91 | 9 | 13 | 87 |
| timing | 121 | 68 | 71 | 1 | 5 | 123 |
| class | 10 | 41 | 107 | 36 | 29 | 28 |
| material | 3 | 0 | 6 | 116 | 78 | 0 |
