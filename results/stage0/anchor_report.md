# Event Anchor Validation Report

**Total clips:** 785
**Total events:** 3122
**Single-event clips:** 0
**Multi-event clips:** 785
**Mean uncertainty:** 0.0500 s
**Max uncertainty:** 0.0500 s

## Provenance Breakdown

| source | n_clips |
|---|---|
| foleybench_audio_onset | 785 |

## Anchor Priority (project config)

Priority order (highest first): foleybench_metadata → visual_onset_detector → light_human_marks.

## Notes

- Anchor uncertainty is required for timing and binding axes (anchor_uncertainty_required = true).
- Clips without uncertainty estimates are not used for timing/binding axes.

## Stage-0 σ_anchor Summary (audio-track vs visual onsets)

σ per clip = |primary audio-track onset − nearest visual onset|; stats over clips with BOTH anchors.

| stat | value |
|---|---|
| n_clips | 785 |
| median σ (s) | 0.7573 |
| mean σ (s) | 1.2954 |
| max σ (s) | 8.8203 |
| coverage audio | 1.0000 |
| coverage visual | 1.0000 |
| coverage both | 1.0000 |
| recommended gross-timing bin (s) | 1.5147 |
| clips with processing errors | 0 |

Propagation rule (manual §3.2): gross-timing bins ≥ 2·σ_anchor; the recommended bin width is max(0.5 s, 2·median σ).

## PROPOSED AMENDMENT (pending PI approval)

The `foleybench_audio_onset` anchor source above detects onsets on the clip's OWN audio track. This source is **NOT** part of the approved manual §3.2 anchor chain (foleybench_metadata → visual_onset_detector → light_human_marks); it is recorded here as a **PROPOSED AMENDMENT pending PI approval**. Until approved, audio-track anchors are diagnostic evidence for σ_anchor estimation only; the frozen anchor source for the timing and binding axes remains the approved chain, validated against the 30-clip human check set (data/manifests/anchor_check_30.csv).
