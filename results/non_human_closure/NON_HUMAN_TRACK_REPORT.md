# SoundDecisions non-human experimental closure

Date: 2026-07-18 (Asia/Shanghai)
Starting commit: `a1e8f3ae324e8886379c19c5bc312d7ebc942946`
Execution branch: `agent/non-human-closure`
Protocol SHA-256: `5c4fc4025995c16e355feb8cc02fbb3627891d47f6df052becde4845eaa7bd09`

## PI decision summary

The frozen multi-seed pooled-sustained Class rule did **not** reproduce the historical commitment window near `s≈0.35`. Its registered all-cell point crossing is `s=0.90`; 24.66% of video-bootstrap draws are noncrossing even at the end of the sampled grid. The frozen result is `NOT_SUPPORTED`, not a confirmation or a stable-across-seed finding. The historical number and the pooled result are different estimands, so the individual early-crossing evidence is reported separately below.

The B-1 same-forward lineage gate passed. Recollected internal features are valid for all 816 base trajectories and eight progress points. For the primary fork-majority target, only the pooled fixed-capacity MLP clears both the majority and conditioning-only information criteria, first at `s=0.45`. External preview clears the same criterion at `s=0.45` and is numerically stronger. No internal family clears the criterion for the ODE-final target, while external preview again does at `s=0.45`. Internal readout earlier than external preview is therefore `NOT_SUPPORTED`. No family reaches the predeclared action-readout accuracy/calibration/coverage criterion.

Valid legacy Material 2AFC could not be constructed. Existing outcome-independent metadata supports only 3 complete videos / 96 cells, below the frozen 20-video / 640-cell floor. That workstream is `INCOMPLETE_ARTIFACTS`; no candidate preview was replayed and no Material margin was measured. A distinct Material window remains `UNRESOLVED`.

| question | decision | status |
|---|---|---|
| Historical Class window under the frozen multi-seed rule | Not reproduced; registered all-cell pooled point crossing `s=0.90` | `NOT_SUPPORTED` |
| Class variance decomposition | Video and video-by-seed interaction dominate | `SUPPORTED_EXPLORATORILY` |
| B-1 same-forward identity | Gate passed; engineering lineage established | `NOT_TESTED` scientifically |
| Fork-majority internal information | Pooled MLP first qualifies at `s=0.45` | `SUPPORTED_EXPLORATORILY` |
| Internal readout earlier than external | Same point for primary target; absent for secondary | `NOT_SUPPORTED` |
| Action-quality Class readout | No representation qualifies | `NOT_SUPPORTED` |
| Valid legacy Material 2AFC | Insufficient metadata-matched coverage | `INCOMPLETE_ARTIFACTS` |
| Material window distinct from Class | No valid Material curve | `UNRESOLVED` |
| Event-centered v2 axes | Not run in this closure | `NOT_TESTED` |

## 1. Did Class commitment replicate across seeds?

Not under the frozen pooled-sustained replication rule. The exact 48-video, 17-seed bank contains 816 base finals and 78,336 fork finals. All 79,152 WAVs were measured once with the pinned PANNs Class measurer; the full 527-way outputs, normalized coarse posteriors, abstention fields, identities, and hashes are retained in the immutable artifact root.

At the registered all-cell pooled sustained threshold `theta=0.70`, the point crossing is `s=0.90`. Across 5,000 video-cluster bootstrap draws, 3,767 cross and 1,233 are noncrossing. Conditional on crossing, the percentile range is `[0.75, 0.90]`; this is not an unconditional confidence interval. The historical estimate is `s=0.34593`, outside that conditional range and more than one sampled progress step from the new point estimate. The frozen classifier therefore assigns `not_reproduced` / `NOT_SUPPORTED`.

The threshold sensitivity is substantial and must remain visible:

| theta | point estimate | bootstrap crossing / noncrossing | conditional crossing range |
|---:|---:|---:|---:|
| 0.50 | 0.45 | 5000 / 0 | [0.25, 0.60] |
| 0.60 | 0.75 | 4957 / 43 | [0.45, 0.90] |
| 0.70 | 0.90 | 3767 / 1233 | [0.75, 0.90] |
| 0.80 | noncrossing | 567 / 4433 | [0.90, 0.90] |
| 0.90 | noncrossing | 2 / 4998 | [0.90, 0.90] |

The historical `s=0.34593` was the mean **unsustained individual first crossing** among historical crossers, not a pooled sustained crossing. The closest B2 individual summary is correspondingly early but heterogeneous: nine of 48 videos are video-determined, and among the remaining 663 video-seed units, 634 are scorable; 582 cross, 52 do not, and 29 are unscorable. Among crossers only, mean first crossing is `0.2997` and median is `0.25`; 526 have a sustained crossing, and noncrossers are right-censored without numeric imputation. This individual result is directionally consistent with an early window but is not the registered replication estimand.

The registered pooled curve includes the nine video-determined videos. All nine have `A_ind=1`, for which normalized commitment gain is fixed to zero. An explicitly post-hoc, read-only sensitivity excluding those videos moves the `theta=0.70` pooled sustained crossing from `s=0.90` to `s=0.60`; it still does not cross by the frozen `s≤0.45` reproduction cutoff. Thus the registered status is unchanged, but `s=0.90` must be understood as the all-cell estimate rather than a characterization of nondetermined videos alone. At `theta=0.70`, 13 registered base-seed curves have sustained crossing at `s=0.90` and four are noncrossing; none has sustained crossing by `s=0.45`.

The appropriate frozen-rule classification is therefore **not reproduced, with substantial video-by-seed heterogeneity**. It is not “stable across seeds.” The early individual first-crossing evidence must remain visible, but it cannot substitute for the registered pooled sustained result.

## 2. Where does Class variation come from?

The progress-stratified crossed decomposition and 5,000-draw video-cluster bootstrap give:

| component | variance | 95% video-bootstrap CI | fraction |
|---|---:|---:|---:|
| Video | 0.079268 | [0.043577, 0.104772] | 42.19% |
| Additive base seed | 0.000000 | [0.000000, 0.002605] | 0.00% |
| Video × seed interaction | 0.078212 | [0.063292, 0.092409] | 41.63% |
| Fork Monte Carlo, excluding abstention increment | 0.028045 | [0.023118, 0.032968] | 14.93% |
| Identifiable abstention subcomponent | 0.002340 | [0.001820, 0.002913] | 1.25% |

The additive seed estimate is boundary-clipped to zero; its raw method-of-moments estimate is negative at every progress stratum. This does not mean seed is irrelevant. Almost as much variation lies in the video×seed interaction as in video itself: a seed does not act as one globally good or bad offset, but its effect depends on the video. Fork Monte Carlo contributes a smaller but non-negligible tail. The abstention term is a subcomponent of fork resampling, not a sixth independent component. The 17-final estimate of each video baseline is held fixed, so its finite-reference uncertainty is not separately decomposed; video and interaction components are conditional on the measured baseline. Measurer repeatability is `UNRESOLVED` because each immutable WAV was measured once.

## 3. Did the B-1 lineage gate pass?

Yes. Four outcome-independent calibration clips (`3780`, `1813`, `3112`, `1048`) defined the tolerance before clip `1002` was evaluated. Two same-device calibration replays covered 64 state evaluations and 7,776 cross-replay rows. The eligible paths were exactly identical, so the frozen `q0.999(higher) × 2` tolerances were zero. This is appropriate for the registered same-device identity question; it is not a cross-device reproducibility claim.

Two fresh held-out replays of clip `1002` then passed all 1,960 eligible comparisons with zero failures and no tolerance change. The prohibited comparison between different fp16/fp32 reduction orders was never used as identity evidence. The engineering gate status is `PASS`; it is not itself a semantic scientific claim.

After the gate, all 816 B2 base trajectories were recollected at eight progress points into 6,528 state units across eight immutable shards. Each unit retains latent, device latent/time, velocity, Tweedie latent, tokens and both pooled paths, token statistics, selected attention/QKV outputs, complete conditioning, external preview representation, and full dtype/shape/hash/hook/device provenance. The old 25,600 feature bundles are not primary evidence.

## 4. Which internal representations predict fork-majority Class?

Only the fixed-capacity pooled MLP satisfies the registered information-readout definition at any progress point. Its earliest qualifying point is `s=0.45`, and it also clears conditioning-only at `s=0.60`; the support is not sustained at `s=0.75` or `s=0.90`. Latent-only, velocity-only, Tweedie latent, pooled linear, token statistics, single-query attention, and selected cross-attention do not jointly clear the majority and conditioning-only paired video-bootstrap criteria at any sampled progress.

At `s=0.45` for fork-majority:

| representation | balanced accuracy (95% video CI) | ECE | selective coverage |
|---|---:|---:|---:|
| Majority/null | 0.039 [0.021, 0.066] | 0.118 | 0.000 |
| Conditioning only | 0.103 [0.052, 0.164] | 0.037 | 0.000 |
| External preview | 0.223 [0.170, 0.278] | 0.191 | 0.000 |
| Pooled MLP | 0.170 [0.125, 0.228] | 0.422 | 0.821 |

The pooled MLP exceeds conditioning by `+0.0669`, paired 95% video-bootstrap CI `[+0.0159, +0.1170]`. It does not exceed external preview: the difference is `-0.0532`, CI `[-0.1116, +0.0097]`. At `s=0.60`, it again exceeds conditioning but remains below external preview. This supports a weak exploratory internal information signal, not a calibrated decision rule and not the statement that “the model knows.”

These intervals are pointwise and unadjusted across the 11 families and eight progress points. The fork-majority and ODE-final targets are model-output-derived continuity proxies, not human semantic ground truth; target coverage is 82.7–89.3% across progress for fork-majority and 70.6% for ODE-final. All fixed MLP fits reached the frozen 120-iteration cap, so their convergence warnings are retained. These limitations make the pooled-MLP result appropriate only as `SUPPORTED_EXPLORATORILY`.

For the secondary ODE-final target, no internal representation clears conditioning-only. At `s=0.45`, external preview reaches balanced accuracy `0.250` `[0.186, 0.347]` and exceeds conditioning by `+0.1268` `[+0.0262, +0.2411]`; pooled MLP is `0.138` `[0.095, 0.199]` and does not exceed conditioning.

## 5. Is internal Class readout earlier or better than conditioning and external preview?

Internal readout is not earlier under the frozen operational rule. For fork-majority, pooled MLP and external preview both first qualify at `s=0.45`; pooled MLP is not significantly better than external at that point. For ODE-final, external preview qualifies at `s=0.45` and no internal family qualifies anywhere. The earlier-than-external claim is `NOT_SUPPORTED` for both targets; this is not proof that earlier internal information is absent.

No representation reaches the predeclared action criterion of balanced accuracy at least `0.70`, ECE at most `0.10`, and coverage at least `0.80`. The information point here is therefore distinct from the historical statement that external preview becomes reliably actionable around `s≈0.75`; the two criteria should not be conflated.

The external baseline here is a trained probe over the same-progress PANNs preview representation, not the historical direct preview-label readout. Its exploratory information point at `s=0.45` therefore cannot be substituted for the historical external action-readout estimate near `s≈0.75`.

Evaluation used six outer and four inner video-group folds. Every seed, progress point, representation, and target from a video remains in one outer fold. Hyperparameters were selected only on inner video folds. The retained output contains all 113,212 outer-fold candidate predictions, probabilities, selected parameters, target coverage, and paired video-bootstrap statistics. Primary fork-majority coverage ranges from 675 to 729 of 816 candidates by progress; secondary ODE-final coverage is 576/816 at every progress. Missing targets are explicit and never imputed.

## 6. Was valid legacy Material 2AFC possible?

No. The exact legacy inventory is intact at 200 videos / 6,400 cells with 800 surviving subject-final embeddings. The outcome-independent matcher used only existing coarse Class, admissible UCS material category, scene metadata, automatic timing, and source-audio loudness. It found 52 matchable subjects across 22 videos, but the required complete four-subject rule retained only videos `2622`, `4122`, and `4855`: 3 videos / 96 cells.

Because this is below the frozen 20-video / 640-cell floor, the workstream stopped as `INCOMPLETE_ARTIFACTS` before candidate replay. No preview embedding, cosine, margin, A/B orientation, 2AFC decision, AUROC, indifference rate, bootstrap interval, or readout curve was measured. Embeddings were not used to invent material truth.

Consequently, a Material readout window distinct from Class is `UNRESOLVED`. The blocker is metadata coverage, not evidence against Material continuity.

## 7. Evidence classes and limits

- **Exploratory scientific continuity evidence:** Class multi-seed commitment, variance decomposition, and grouped-CV readout. These concern clip-level coarse Class only.
- **Engineering evidence:** B-1 same-forward identity and the complete lineage-valid feature recollection.
- **Legacy continuity artifact evidence:** the fail-closed Material reference-coverage audit. No Material performance was measured.
- **Unresolved:** measurer repeatability, valid Material 2AFC, and any distinct Material window. Class action readout was tested and is `NOT_SUPPORTED` under the frozen criterion.
- **Not tested:** Presence, Timing, event-centered Class or Material, Binding, B6, condition swaps, causal intervention, map scheduling, sealed confirmation, second backbone, and paper claims.

No result in this report is an event-centered Axis Specification v2 scientific PASS. No sealed confirmatory cohort was used.

## 8. Bugs, deviations, and protocol visibility

All scientific rules were frozen before the corresponding held-out outcomes. No threshold, taxonomy, probe family, target rule, or Material matching rule was weakened after inspection. Engineering corrections and failed attempts remain in Git or the immutable log root.

Two post-launch reducer changes are especially relevant:

- `b2a8687` parallelized read-only feature-shard validation while preserving input order and hashes; no evidence rule changed.
- `c276668` corrected a conclusion-label bug after readout: when external preview clears conditioning and no internal family does, the permitted conclusion is “internal readout not earlier,” not “conditioning explains most information.” `merged_v1` is preserved; `merged_v2` has byte-identical predictions and metrics and changes only the conclusion label.

ModelScope mirrors were unavailable. The run used already-present, hash-pinned local Hugging Face assets in offline mode; there were no silent downloads. `an29` was used for Class posterior shards when GPUs were available, but all its GPUs were occupied during full feature recollection and readout, so those workloads used every safely free A800 on `an12` (GPUs 4–7, two independent TP1 readout replicas per GPU).

The outcome-blind Material feasibility artifact is bound to protocol SHA-256 `a9f2a84653495045be039b17d1113de21fe1f6e951fffc0e5b65deb925473f39`, while subsequent Class/B-1/readout artifacts use the final SHA-256 `5c4fc4025995c16e355feb8cc02fbb3627891d47f6df052becde4845eaa7bd09`. The sole intervening protocol edit, committed in `86f18c5`, added a pinned B-1 asset/offline-environment contract. It did not alter the Material reference, matching, coverage, or stop rules, so the earlier fail-closed Material result remains rule-identical. Both revisions and the diff are retained in Git.

The complete attempt ledger and unresolved items are in `BUGS_DEVIATIONS_UNRESOLVED.md`; exact commands and environment bindings are in `REPRO.json` and `COMMANDS.md`.

## 9. Single next experiment

Run one fresh-video, multi-seed exploratory replication with the representation family frozen to **pooled MLP versus conditioning-only and external preview**, centered on `s={0.35, 0.45, 0.60}` and using the same grouped nested-CV, target rules, calibration thresholds, and paired video bootstrap. Power it for the observed video×seed interaction and retain enough videos to resolve the small pooled-MLP advantage over conditioning. Do not add architecture search or causal intervention.

This is the most justified next step because the only qualifying internal signal is weak, appears no earlier than external preview, and is not action-calibrated. Replicating that specific signal on fresh videos should precede any intervention or policy experiment.

## 10. Canonical evidence

- `CLASS_POSTERIOR_MEASUREMENT_REPORT.{json,md}` and immutable posterior SHA-256 `a381e0f8e662aae482959f0cd2e7fe46f3cb1e8ef2ef7040b4e65913ff86cd1e`
- `CLASS_MULTISEED_COMMITMENT.{json,md,csv}` plus detailed video/seed crossing CSVs
- post-hoc audit sensitivity `CLASS_VIDEO_DETERMINED_SENSITIVITY.json`, SHA-256 `fd2b02fcffbdfba320d5ee630ddf4d3369a17120b64f44004e1e140e17e8b505`
- `CLASS_VARIANCE_DECOMPOSITION.{json,md}`
- `FEATURE_LINEAGE_REPORT.{json,md}` and `feature_manifests/`
- `CLASS_INTERNAL_READOUT_REPORT.{json,md}` and complete compressed outer predictions SHA-256 `6463c4b23a1df34d92d89efd3161ff8ce43e65e510c7ca2130a57e96de291ca3`
- `MATERIAL_CONTINUITY_2AFC_REPORT.{json,md}` and insufficiency artifact SHA-256 `1bf4adf7eee37766665d421312b59f819d35d4fcf88b1ae14d784ec931b82dd8`

This report ends the authorized non-human closure. No downstream axis, intervention, confirmatory, second-backbone, or paper work was started.
