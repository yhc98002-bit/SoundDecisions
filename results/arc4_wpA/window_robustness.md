# Window-estimator robustness (Arc-4 WP-A)

Diagnostic re-analysis of cached per-video Phase-1 commitment curves. The legacy estimator averages only clips that cross the threshold; the censored estimator assigns never-crossers s=1 before taking the cohort median.

Source: `results/stage0/phase1/commitment_map_p1cfg1.csv`; cfg=1.0, alpha=0.8, schedule=sqrt_down. Crossing uses the existing discrete earliest-grid-point rule (no interpolation).

## Theta_commit = 0.70

| axis | crossing fraction | legacy mean of crossers | censored median |
|---|---:|---:|---:|
| timing | 77/200 (0.385000) | 0.113636 | 1.000000 |
| presence | 113/200 (0.565000) | 0.214159 | 0.675000 |
| material | 200/200 (1.000000) | 0.638250 | 0.600000 |

## Threshold sweep and ordering stability

Stability is an exact comparison to the estimator's ordering at theta=0.70; `=` denotes a tied estimator value.

| theta | legacy ordering | stable | censored ordering | stable |
|---:|---|:---:|---|:---:|
| 0.60 | timing < presence < material | True | presence < material < timing | False |
| 0.65 | timing < presence < material | True | material = presence < timing | False |
| 0.70 | timing < presence < material | True | material < presence < timing | True |
| 0.75 | timing < presence < material | True | presence < material < timing | False |
| 0.80 | timing < presence < material | True | presence < material < timing | False |

## Sweep values

| theta | axis | crossing fraction | legacy mean | censored median |
|---:|---|---:|---:|---:|
| 0.60 | timing | 0.385000 | 0.111688 | 1.000000 |
| 0.60 | presence | 0.575000 | 0.190000 | 0.450000 |
| 0.60 | material | 1.000000 | 0.520750 | 0.600000 |
| 0.65 | timing | 0.385000 | 0.111688 | 1.000000 |
| 0.65 | presence | 0.570000 | 0.192105 | 0.600000 |
| 0.65 | material | 1.000000 | 0.577000 | 0.600000 |
| 0.70 | timing | 0.385000 | 0.113636 | 1.000000 |
| 0.70 | presence | 0.565000 | 0.214159 | 0.675000 |
| 0.70 | material | 1.000000 | 0.638250 | 0.600000 |
| 0.75 | timing | 0.385000 | 0.114935 | 1.000000 |
| 0.75 | presence | 0.565000 | 0.214159 | 0.675000 |
| 0.75 | material | 0.995000 | 0.696734 | 0.750000 |
| 0.80 | timing | 0.385000 | 0.114935 | 1.000000 |
| 0.80 | presence | 0.565000 | 0.214159 | 0.675000 |
| 0.80 | material | 0.985000 | 0.767005 | 0.750000 |
