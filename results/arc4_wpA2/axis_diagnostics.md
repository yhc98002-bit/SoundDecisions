# Arc-4 WP-A2 axis diagnostics

This table combines Tier-0 axis validity, the determination-status partition, and the reconstructed Phase-2 readout map. The readout columns in the first table are the frozen earliest grid point, s=0.05; the complete trajectory follows.

## Cross-axis diagnostic

| axis | majority_share | k_eff | a_between_video | a_ind_mean | abstain_rate | video_determined | crossing | censored | readout_metric_s0.05 | readout_margin_s0.05 | balanced_accuracy_s0.05 | Tier-0 verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| presence | 0.877812 | 1.273099 | 0.786962 | 0.813292 | 0.000000 | 70 | 113 | 17 | 0.543750 | -0.350000 | 0.558165 | DEGENERATE |
| timing | 0.940625 | 1.128591 | 0.885779 | 0.902750 | 0.000000 | 121 | 77 | 2 | 0.946250 | 0.005000 | 0.352839 | DEGENERATE |
| class | 0.271006 | 6.233897 | 0.151856 | 0.377680 | 0.323125 | 10 | 172 | 18 | 0.250000 | -0.088750 | 0.108190 | INFORMATIVE |
| material | n/a | n/a | 0.382574 | 0.637024 | n/a | 0 | 200 | 0 | 0.267555 | n/a | n/a | INFORMATIVE |

Categorical majority, k_eff, abstain, readout margin, and balanced accuracy are undefined for the continuous material axis and are shown as n/a. Material's readout metric remains legacy mean cosine here; it is not called accuracy.

## Readout trajectory

| axis | s | metric | pooled value | ci_lo | ci_hi | majority baseline | margin over majority | balanced accuracy | bal_ci_lo | bal_ci_hi |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| presence | 0.05 | exact_match | 0.543750 | 0.497469 | 0.590000 | 0.893750 | -0.350000 | 0.558165 | 0.500976 | 0.615260 |
| presence | 0.15 | exact_match | 0.551250 | 0.508750 | 0.591250 | 0.893750 | -0.342500 | 0.572727 | 0.515492 | 0.625420 |
| presence | 0.25 | exact_match | 0.668750 | 0.626250 | 0.708750 | 0.893750 | -0.225000 | 0.628095 | 0.568552 | 0.687786 |
| presence | 0.35 | exact_match | 0.757500 | 0.722469 | 0.793750 | 0.893750 | -0.136250 | 0.708844 | 0.648334 | 0.763092 |
| presence | 0.45 | exact_match | 0.802500 | 0.768750 | 0.835000 | 0.893750 | -0.091250 | 0.744385 | 0.689777 | 0.793761 |
| presence | 0.60 | exact_match | 0.836250 | 0.807500 | 0.867500 | 0.893750 | -0.057500 | 0.773632 | 0.723708 | 0.823450 |
| presence | 0.75 | exact_match | 0.911250 | 0.887500 | 0.933750 | 0.893750 | 0.017500 | 0.820773 | 0.768445 | 0.868362 |
| presence | 0.90 | exact_match | 0.977500 | 0.967500 | 0.987500 | 0.893750 | 0.083750 | 0.956314 | 0.928593 | 0.982485 |
| timing | 0.05 | exact_match | 0.946250 | 0.927469 | 0.963750 | 0.941250 | 0.005000 | 0.352839 | 0.275988 | 0.427380 |
| timing | 0.15 | exact_match | 0.953750 | 0.935000 | 0.968781 | 0.941250 | 0.012500 | 0.607005 | 0.543090 | 0.680959 |
| timing | 0.25 | exact_match | 0.965000 | 0.951250 | 0.977500 | 0.941250 | 0.023750 | 0.669837 | 0.592780 | 0.754716 |
| timing | 0.35 | exact_match | 0.968750 | 0.955000 | 0.981250 | 0.941250 | 0.027500 | 0.574170 | 0.496878 | 0.651207 |
| timing | 0.45 | exact_match | 0.968750 | 0.956219 | 0.980000 | 0.941250 | 0.027500 | 0.703005 | 0.627896 | 0.777732 |
| timing | 0.60 | exact_match | 0.976250 | 0.965000 | 0.986250 | 0.941250 | 0.035000 | 0.836836 | 0.766770 | 0.914582 |
| timing | 0.75 | exact_match | 0.985000 | 0.976250 | 0.993750 | 0.941250 | 0.043750 | 0.936670 | 0.878862 | 0.989222 |
| timing | 0.90 | exact_match | 0.998750 | 0.996250 | 1.000000 | 0.941250 | 0.057500 | 0.995833 | 0.985116 | 1.000000 |
| class | 0.05 | exact_match | 0.250000 | 0.216250 | 0.286250 | 0.338750 | -0.088750 | 0.108190 | 0.081530 | 0.132503 |
| class | 0.15 | exact_match | 0.290000 | 0.256250 | 0.325031 | 0.338750 | -0.048750 | 0.130436 | 0.104176 | 0.157423 |
| class | 0.25 | exact_match | 0.287500 | 0.252500 | 0.325000 | 0.338750 | -0.051250 | 0.141351 | 0.112619 | 0.170587 |
| class | 0.35 | exact_match | 0.345000 | 0.308750 | 0.381250 | 0.338750 | 0.006250 | 0.212504 | 0.181396 | 0.247962 |
| class | 0.45 | exact_match | 0.436250 | 0.398719 | 0.478781 | 0.338750 | 0.097500 | 0.320443 | 0.256572 | 0.381513 |
| class | 0.60 | exact_match | 0.557500 | 0.517500 | 0.596250 | 0.338750 | 0.218750 | 0.462904 | 0.384244 | 0.543629 |
| class | 0.75 | exact_match | 0.717500 | 0.686250 | 0.748750 | 0.338750 | 0.378750 | 0.783248 | 0.744506 | 0.823545 |
| class | 0.90 | exact_match | 0.940000 | 0.923719 | 0.955000 | 0.338750 | 0.601250 | 0.951687 | 0.923692 | 0.972687 |
| material | 0.05 | cosine | 0.267555 | 0.252179 | 0.282860 | n/a | n/a | n/a | n/a | n/a |
| material | 0.15 | cosine | 0.334460 | 0.315587 | 0.352604 | n/a | n/a | n/a | n/a | n/a |
| material | 0.25 | cosine | 0.455032 | 0.433266 | 0.475788 | n/a | n/a | n/a | n/a | n/a |
| material | 0.35 | cosine | 0.584864 | 0.560075 | 0.607278 | n/a | n/a | n/a | n/a | n/a |
| material | 0.45 | cosine | 0.696152 | 0.673588 | 0.716537 | n/a | n/a | n/a | n/a | n/a |
| material | 0.60 | cosine | 0.850942 | 0.837146 | 0.862711 | n/a | n/a | n/a | n/a | n/a |
| material | 0.75 | cosine | 0.962605 | 0.959220 | 0.965882 | n/a | n/a | n/a | n/a | n/a |
| material | 0.90 | cosine | 0.998658 | 0.998511 | 0.998795 | n/a | n/a | n/a | n/a | n/a |

## Reconstruction checks

- Phase-2 v3 pooled values reproduce WP-A v2 at all 32 cells: max_abs_delta = 0 (tolerance 1e-09).
- Phase-2 v3 pooled values reproduce the committed legacy source at all 32 cells: max_abs_delta = 2.2204460492503131e-16 (tolerance 1e-09).
- Class confident-subset reconstruction reproduces the committed Phase-1 CSV: max_abs_delta = 0; s_commit confident = 0.345930, naive = 0.462593; gap confident = 0.404070, naive = 0.287407.
- Track-P internal probes remain outside this table because persisted per-example predictions are unavailable; retraining is required.
