# Human-evaluation release report

Status: **ROUND-1 CURATION READY FOR DELIVERY; NO RATINGS OR SCIENTIFIC RESULTS YET.**

The deliverable is `releases/round1_curation_v4.zip`, SHA256
`f2af52d17c916be234acb1e8e83470e8a25d2b3725dc6b98554ebc7826eec71b`, size
68,695,051 bytes. It is authorized for
one assigned lead curator. The
authorization is recorded by the versioned Git history and hashed release
manifest; no electronic signature workflow is required.

## Contents and counts

The archive contains one offline `rate.html`, instructions, public manifest,
manifest and ratings schemas, `SHA256SUMS.txt`, and 82 blinded video files. It
has 90 task assignments:

- 30 anchor-curation assignments;
- 60 two-event-curation assignments; and
- eight overlapping videos, for 82 unique videos.

The public manifest is `round1_v4_blinded_items.json`, SHA256
`f9892840226dee50484a38f6606ad1606ec1b67256dc2a5da3658cdca47bad8a`.
The separate encrypted unblinding map is
`round1_v4_unblinding_map.sealed.json`, SHA256
`dea79eaaa0b31ea891a101e45df5e56509f91bfd4dda20f825f24f314438b533`.
It is never included in or loaded by the rater package. The complete checksum
registry is `ROUND1_V4_SHA256SUMS.json`. The per-video audio registry is
node-only under `private/`, is ignored by Git, and is represented publicly only
by its aggregate SHA256 in the release record.

The prior v1, v2, and v3 archives were retired before delivery. Their original
bytes, hashes, and reasons remain in `RETIRED_RELEASES.json`; neither is a
valid rater entry point.

## Screenshot-free walkthrough

1. **Start or resume.** The curator enters the assigned ID. That ID determines
   a stable shuffled item order. The page restores any matching local draft or
   imports an earlier JSON export.
2. **Anchor curation.** A physically video-only clip and its FoleyBench caption
   are shown. The caption is only a locating hint. The curator writes a
   visual-only event description, marks the closed visual interval with
   one-frame stepping and 0.25/0.5/1x playback, or selects **Too uncertain**.
3. **Two-event curation.** The curator confirms or rejects whether two
   separable visible target events exist. Confirmation requires two visual-only
   descriptions and two chronologically ordered closed intervals.

Every change attempts a localStorage save. Export remains available even when
browser storage is unavailable, and import/export preserves partial drafts.
The large export button writes `ratings_<raterID>.json`.

Presence is intentionally a separate Round-2 screen. It is built only after
the returned Round-1 export has fixed event IDs, visual descriptions, and
anchors. Round 2 restores audio, fixes those event fields read-only, and must be
completed by a different evaluator. One Round-2 export supports descriptive
counts only; AC1 requires two or more independent exports.

## Scientific and blinding controls

- Round-1 delivered MP4s are remuxed to exactly one video stream and zero audio
  or auxiliary streams; UI muting is not the only guard.
- Anchor, P1, and P2 descriptions are curator-authored and explicitly forbid
  sound/audio language. Candidate captions are never substituted as event
  descriptions.
- Public IDs are keyed opaque IDs. No model, condition, cfg, seed, source path,
  source clip ID, or project conclusion appears in the rater manifest.
- Anchor events (`A1`) and pair events (`P1`, `P2`) remain distinct even on an
  overlapping source video.
- The Round-1 reducer requires an exact one-export/item/hash join. The Round-2
  builder additionally verifies event-ID prefixes/suffixes, sources,
  descriptions, interval bounds, subrecords, flattened unions, and counts.
- Round 2 refuses silent media and independently copies hash-verified,
  audio-bearing source files. It sanitizes each copy to exactly one video and
  one audio stream, removes container/stream metadata and chapters, and binds
  the private media registry to the tracked v4 release record.

## Operator workflow

Give the curator only the ZIP. Verify before transfer:

```bash
sha256sum results/human_eval_pack/releases/round1_curation_v4.zip
```

After receiving exactly one completed curator export:

```bash
export PYTHONHASHSEED=0
.venv/bin/python results/human_eval_pack/curate_round1.py \
  --manifest results/human_eval_pack/round1_v4_blinded_items.json \
  --ratings ratings_<curatorID>.json \
  --output event_catalog_v2.json

.venv/bin/python results/human_eval_pack/build_round2.py \
  --event-catalog event_catalog_v2.json \
  --media-root results/human_eval_pack \
  --audio-media-registry results/human_eval_pack/private/round1_v4_audio_media_registry.json \
  --round1-release-record results/human_eval_pack/ROUND1_V4_RELEASE.json \
  --output-dir results/human_eval_pack/releases/round2_presence_v2
```

Score the one-evaluator Round-2 export descriptively:

```bash
export PYTHONHASHSEED=0
.venv/bin/python results/human_eval_pack/score_round2.py \
  --manifest results/human_eval_pack/releases/round2_presence_v2/round2_manifest.json \
  --ratings ratings_<evaluatorID>.json \
  --output round2_presence_summary.json
```

## Verification

Focused artifact-independent command:

```bash
export PYTHONHASHSEED=0
.venv/bin/python -m pytest -q \
  tests/test_human_eval_release_builder.py \
  tests/test_human_eval_round1_ui.py \
  tests/test_human_eval_round_processing.py
```

The real release was rebuilt twice from the same frozen inputs. Both runs were
byte-identical: ZIP, public manifest, encrypted map, private registry, and
release-directory digest all matched. The structural audit validated 88 ZIP
entries, all 87 internal checksums, all 82 source/delivered joins, and 82/82
media files as exactly one video stream with no audio or auxiliary stream.

Focused tests passed under both hash seeds: **83 passed** with
`PYTHONHASHSEED=0` and **83 passed** with `PYTHONHASHSEED=1`. The full
artifact-bearing node suite passed with **1175 passed, 3 skipped, 0 failed**.
An independent artifact-free detached checkout of commit `3eacb23` also passed
with **1175 passed, 3 skipped, 0 failed**. That checkout contained no release
ZIP, media directory, private audio registry, or key.

The package has static and executed Node tests for its offline helpers, state
transitions, deterministic shuffle, strict import, and export round trip. An
actual `file://` browser smoke could not run on this node because its Firefox
command is an uninstalled Snap stub and no Chrome/Chromium binary is present.
This is the only remaining operational preflight for the receiving workstation,
not a missing scientific datum.

## Scope limits

This 82-video set curates the concrete 30-anchor and 60-two-event candidate
pools. It is not the proposed complete 120-group calibration design and cannot
by itself produce a measurement-calibration PASS. Round 1 has one curator and
does not estimate inter-rater agreement. Round 2 has not yet run, so this
release contains no Presence verdict, AC1 value, or project conclusion.
