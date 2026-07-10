# Arc-3 Tier-B §B2 — Conditioning-Channel Audit (MMAudio bottleneck)

Ridge + MLP probe of the clip's MEASURED class from the pooled RAW CLIP + Synchformer conditioning (pre-DiT), frozen 60/40 clip split; chance = eval majority-class prior; bootstrap unit = video.

- clips (feature & label): 200  classes: 11  eval majority prior: 0.2702702702702703
- B1 DiT-internal best class acc: 0.4375

| conditioning variant | family | eval acc | chance | Δ over chance | 95% CI (by video) | n_eval |
|---|---|---|---|---|---|---|
| pooled_all | ridge | 0.378 | 0.270 | 0.108 | [0.270, 0.486] | 74 |
| pooled_all | mlp | 0.365 | 0.270 | 0.095 | [0.257, 0.473] | 74 |
| clip_f | ridge | 0.419 | 0.270 | 0.149 | [0.311, 0.527] | 74 |
| clip_f | mlp | 0.311 | 0.270 | 0.041 | [0.256, 0.459] | 74 |
| sync_f | ridge | 0.270 | 0.270 | 0.000 | [0.176, 0.365] | 74 |
| sync_f | mlp | 0.365 | 0.270 | 0.095 | [0.257, 0.473] | 74 |
| clip_f_c | ridge | 0.297 | 0.270 | 0.027 | [0.203, 0.405] | 74 |
| clip_f_c | mlp | 0.338 | 0.270 | 0.068 | [0.230, 0.446] | 74 |

## Decision (frozen pre-reg §B2)
- best cond probe: clip_f/ridge (acc 0.4189189189189189)
- near chance (<= chance+0.15): True
- substantially below B1 (<= B1_best-0.15): False
- **TOKEN: COND_NOT_BOTTLENECK**  (CONTINUE regardless; offline, no pause)
