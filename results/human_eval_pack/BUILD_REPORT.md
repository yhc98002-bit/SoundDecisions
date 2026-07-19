> **SUPERSEDED (2026-07-19).** This report describes the locked pre-release instrument. The authorized curator delivery is documented in `ROUND1_RELEASE_REPORT.md`.

# Blinded human-evaluation package report

Status: **INCOMPLETE_ARTIFACTS; INSTRUMENT BUILT AND LOCKED; RATING NOT AUTHORIZED.**

## Scope and item inventory

The Goal-1 cohort plan defines a 120-group development design quota but does
not name a completed 120-item cohort. This package therefore uses only the two
concrete candidate pools named by that plan:

- 30 anchor-and-Presence task assignments from the historical anchor candidate
  manifest;
- 60 two-event curation task assignments from the two-event candidate
  manifest; and
- eight overlaps, giving 82 unique blinded videos and 90 task assignments.

All 82 indexed MP4s were present. No item was substituted and no missing-video
flag was needed. The node-local media directory contains 82 hash-verified,
independent blinded copies (81,405,967 bytes). It is a candidate-curation
package, not the missing weak, control, unrelated-background, or
reserved-confirmatory strata.

## Files

- `rate.html`: single offline instrument with the blinded manifest embedded;
  it has no fetch, server, CDN, or unblinding dependency.
- `blinded_items.json` and `blinded_items.schema.json`: public opaque item
  records, task membership, relative media paths, exact frame rates, and
  durations.
- `ratings.schema.json`: schema for complete and lossless partial exports.
- `INSTRUCTIONS.md`: one-page rater instructions and the verbatim Presence
  rubric.
- `score_ac1.py`: schema-validated two-plus-rater Gwet AC1, interval overlap,
  and deterministic task-stratified 20% MLLM-primary audit selection.
- `unblinding_map.sealed.json`: encrypted source map, never read by
  `rate.html`.
- `SHA256SUMS.json`: registers the sealed map and public instrument files.
- `build_pack.py`: curator-only, fail-closed media verification, opaque-ID
  generation, sealing, and inline-manifest assembly.
- `media/*.mp4`: node-local ignored, hash-verified independent copies under
  blinded filenames; large media is not committed to Git.
- `FLAGS.json`: evidence and the cheapest valid resolution for the absent
  admitted calibration slice and stable target-event records.

## Three-screen walkthrough

The current manifest keeps **Start or resume** disabled. Once the blockers in
`FLAGS.json` are resolved, a newly hashed authorized manifest uses these screens:

1. **Anchor marking:** enter a rater ID, open the Anchor tab, and describe the
   target visible event. Audio is forced muted. The player provides -1/+1
   frame stepping using the video's own frame rate and 0.25x/0.5x/1x playback.
   Mark the closed start/end interval or select **Too uncertain**.
2. **Presence rubric:** the same blinded item opens with audio enabled and the
   rater's event description visible. Select Target-present, Absent, or
   Uncertain; separately mark unrelated background as present/not present; add
   an optional note.
3. **Two-event curation:** audio is forced muted. Confirm or reject whether two
   separable visible target events exist. A confirmed item requires two event
   descriptions and two intervals ordered by start time.

The order of the 82 unique items is a deterministic shuffle of the rater ID.
Every change autosaves to a rater-and-manifest-specific localStorage key. The
large Export button writes `ratings_<raterID>.json`; Import restores complete
or partial state without dropping one-sided interval marks.

## Blinding and integrity

Public item IDs are keyed opaque identifiers and media filenames contain only
those IDs. Source IDs, paths, hashes, categories, cohort roles, and source
metadata are confined to the AES-256-CBC/PBKDF2 sealed map. The node-only key
has mode `0600` and key ID `0e5a3a337c85bf32`. The sealed-map SHA256 is
`ebdba8ad3d4d4b266394b183ec4da9b83a134c1d48a1fe8aa41ae0a7d447c4c5`;
the public manifest SHA256 is
`892097cc63aba36b07f8cb2779f741a8b4b48db9f97a0a1fe42700b6cb0f09b3`.
The sealed envelope was decrypted in memory and its plaintext hash and 82-item
cardinality were verified. All 82 media copies match their sealed source hashes,
have link count one, and do not share source inodes. The current generic target
prompts are for candidate curation only; they are not stable specified-event
targets and cannot support Presence agreement. A future authorized manifest
must contain the adjudicated event identities/descriptions and the PI-approved
admitted calibration strata.

## Verification

Focused command:

```bash
export PYTHONHASHSEED=0
/XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions/.venv/bin/python \
  -m pytest tests/test_human_eval_build.py tests/test_human_eval_html.py \
  tests/test_human_eval_schema.py tests/test_human_eval_scoring.py -q
```

Focused result: `22 passed` at both `PYTHONHASHSEED=0` and `1`. Full CPU suite
result: `1092 passed, 3 skipped` with zero failures. The scorer's fixed-scale
known-answer fixture has AC1 `33/49` and mean anchor interval IoU `5/6`.
Static and Node tests verify the offline/no-network contract; a browser-driver
`file://` smoke test remains unrun on this node.

To score completed exports:

```bash
export PYTHONHASHSEED=0
/XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions/.venv/bin/python \
  results/human_eval_pack/score_ac1.py ratings_rater_a.json ratings_rater_b.json \
  --output human_eval_scores.json
```
