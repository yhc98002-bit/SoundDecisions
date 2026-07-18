# Class variance decomposition

Scientific status: `SUPPORTED_EXPLORATORILY` for the decomposition only. The historical commitment-window replication remains `NOT_SUPPORTED`.

Progress is treated as a fixed stratum. Components use an unbalanced crossed video/base-seed method of moments; uncertainty is a 5,000-draw video-cluster bootstrap retaining all seeds within sampled videos.

| component | variance | 95% video-bootstrap CI | fraction |
|---|---:|---:|---:|
| Video | 0.079268 | [0.043577, 0.104772] | 42.19% |
| Additive base seed | 0.000000 | [0.000000, 0.002605] | 0.00% |
| Video x seed interaction | 0.078212 | [0.063292, 0.092409] | 41.63% |
| Fork Monte Carlo (non-abstention) | 0.028045 | [0.023118, 0.032968] | 14.93% |
| Identifiable abstention subcomponent | 0.002340 | [0.001820, 0.002913] | 1.25% |

The additive seed component is estimated at zero, but video-by-seed interaction is large; therefore this is not evidence that seed choice is irrelevant for a given video. Measurer repeatability is `UNRESOLVED` because each WAV was measured once. The abstention component is a subcomponent of fork Monte Carlo variance, not an independent term.
