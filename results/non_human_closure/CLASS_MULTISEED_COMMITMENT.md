# Class multi-seed commitment

Scientific status: `NOT_SUPPORTED`. This is an exploratory multi-seed continuity replication, not event-centered v2 confirmation.

At the registered all-cell pooled sustained threshold theta=0.70, the point crossing is `s=0.90`. In the 5,000-draw video bootstrap, 3,767 draws cross and 1,233 are noncrossing; among crossing draws, the conditional percentile range is `[0.75, 0.90]`. This range is not an unconditional confidence interval. The frozen classification is `not_reproduced`.

The historical estimate `s=0.346` is a crossers-only mean of unsustained individual first crossings, whereas the frozen B2 decision is an all-cell pooled sustained crossing. They are different estimands. The historical value lies outside the conditional pooled range and is more than one sampled step from the pooled point, but the individual B2 evidence must be read separately below.

Individual video-seed units are heterogeneous: 582/634 scorable nondetermined units cross, 52 do not, and 29 remain unscorable. Among crossers only, the mean first crossing is 0.300 and median is 0.25; noncrossers are right-censored and never imputed.

## Pooled curve

| s | mean commitment gain (95% video CI) | confident fork agreement | fork abstention | scorable cells |
|---:|---:|---:|---:|---:|
| 0.05 | 0.340 [0.280, 0.420] | 0.682 | 0.315 | 765/816 |
| 0.15 | 0.391 [0.327, 0.471] | 0.713 | 0.314 | 766/816 |
| 0.25 | 0.448 [0.381, 0.535] | 0.753 | 0.306 | 757/816 |
| 0.35 | 0.485 [0.411, 0.568] | 0.790 | 0.304 | 749/816 |
| 0.45 | 0.522 [0.448, 0.610] | 0.812 | 0.303 | 748/816 |
| 0.60 | 0.587 [0.504, 0.680] | 0.856 | 0.298 | 733/816 |
| 0.75 | 0.645 [0.561, 0.750] | 0.897 | 0.306 | 710/816 |
| 0.90 | 0.718 [0.628, 0.835] | 0.952 | 0.300 | 683/816 |

## Registered threshold sensitivity

| theta | point estimate | bootstrap crossing / noncrossing draws | conditional crossing percentile range |
|---:|---:|---:|---:|
| 0.50 | 0.45 | 5000 / 0 | [0.25, 0.60] |
| 0.60 | 0.75 | 4957 / 43 | [0.45, 0.90] |
| 0.70 | 0.90 | 3767 / 1233 | [0.75, 0.90] |
| 0.80 | noncrossing | 567 / 4433 | [0.90, 0.90] |
| 0.90 | noncrossing | 2 / 4998 | [0.90, 0.90] |

Nine of 48 videos are video-determined under the registered A_ind >= 0.90 rule. The registered pooled curve includes these cases, whose A_ind=1 commitment gain is fixed to zero; `CLASS_VIDEO_DETERMINED_SENSITIVITY.json` reports the separately labeled post-hoc exclusion sensitivity. Detailed per-video, per-seed, crossing, noncrossing, and baseline records are included as CSV files.
