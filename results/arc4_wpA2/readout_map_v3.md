# Phase-2 Readout Map v3 (p2cfg1)

theta_read = 0.7; ODE-target; audio-tagger probe on cached x0(s) previews. 200/200 clips. Values and 95% CIs use per-clip means with a 1000-draw clip bootstrap (seed 0).

Categorical values use exact-match accuracy. Material values are mean embedding cosines, not accuracies. The categorical majority baseline is the modal ODE-target-label frequency among evaluated rows in each cell. Balanced accuracy joins labels through the deterministic Phase-1 subject ID and is bootstrapped by clip; any missing join is a hard error.

| axis | probe | metric | legacy s_read (absolute theta) | s_read_margin |
|---|---|---|---:|---:|
| class | audio_tagger | exact_match | 0.75 | 0.6 |
| material | audio_tagger | cosine | 0.6 | not applicable |
| presence | audio_tagger | exact_match | 0.35 | never |
| timing | audio_tagger | exact_match | 0.05 | never |

## Baseline lens at s=0.05

| axis | metric value | majority baseline | margin |
|---|---:|---:|---:|
| presence | 0.543750 | 0.893750 | -0.350000 |
| timing | 0.946250 | 0.941250 | 0.005000 |
| class | 0.250000 | 0.338750 | -0.088750 |
| material | 0.267555 | n/a | n/a |

**FLAGGED - Track P:** the persisted Track-P JSON contains only aggregate best-layer scores and layer IDs, not per-example predictions or clip IDs. Applying a clip bootstrap would require retraining and an unregistered choice about layer selection inside versus outside resampling.
