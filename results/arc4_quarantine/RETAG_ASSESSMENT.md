# Cached-final posterior retag assessment

Status: **not feasible from retained artifacts**.

The primary run store contains zero WAV files under `results/stage0/finals/`,
`previews/`, and `audit_wavs/`, and zero WAV files elsewhere under `results/`.
The retained class measurement rows store the `SelfTarget` schema
`{axis_id, embedding, kind, label}`; they do not store the 527-dimensional PANNs
probability vector or a coarse-class posterior. Pooled DiT features and final
class labels are insufficient to reconstruct those probabilities.

Consequently, cached finals cannot be retagged to full posteriors without
regenerating audio. No historical-retag job was launched. This is an
artifact-availability finding, not an evaluation result. The cheapest resolution
is to retain measurement-ready audio whenever a required generation already has
its waveform in memory. The corrected B6 generation queue now writes atomic
IEEE-float source, donor, and swap WAVs without instantiating a tagger; those WAVs
can be retagged later in an isolated evaluation job without regeneration.
Retagging any historical cohort outside B6 still requires regenerating that
cohort's audio.
