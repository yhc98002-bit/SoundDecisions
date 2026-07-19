# Round 2: Presence rating

This package contains fixed target events from a separate curation round. You must be a
different evaluator from the Round-1 curator. Enter your assigned evaluator ID exactly.

For each item, watch and listen to the clip. The event ID, description, and visual anchor
are fixed and cannot be edited. Apply this rubric:

> For one specified visible event, determine whether a corresponding audio event occurs
> near its anchor: present, absent, or uncertain. Salient unrelated background audio does
> not count. The unit is the event.

Choose `target-present`, `absent`, or `uncertain`. Set the unrelated-background flag when
salient unrelated audio is present, whether or not the target is present. Notes are
optional. Progress is saved in this browser. Use **Export ratings JSON** before leaving;
the exported file is the rating record. Import restores a previous export for the same
package and evaluator. **Import JSON is available before Start**. If browser storage is
unavailable or contains a damaged save, a validated import continues in memory without
overwriting the damaged record; export again before closing the page.

This is a single-rater semantic audit. Its output supports descriptive counts only; it
does not establish inter-rater agreement or AC1. A future second independent rating can
be evaluated with the dedicated `score_round2_ac1.py` adapter below.

## Operator commands

Build only from the audio-bearing blinded pack root, never the silent Round-1 release:

```bash
export PYTHONHASHSEED=0
.venv/bin/python results/human_eval_pack/build_round2.py \
  --event-catalog event_catalog_v2.json \
  --media-root results/human_eval_pack \
  --audio-media-registry results/human_eval_pack/private/round1_v4_audio_media_registry.json \
  --round1-release-record results/human_eval_pack/ROUND1_V4_RELEASE.json \
  --output-dir results/human_eval_pack/releases/round2_presence_v2
```

The builder verifies both private inputs against the tracked Round-1 v4 release record,
then remuxes only the first video and first audio stream with metadata and chapters
removed. `SHA256SUMS.json` binds every delivered file. Deliver only a directory that
contains `COMPLETE.json`; a directory without that marker is an interrupted build and
must not be repaired or reused.

After receiving two or more independent exports, compute AC1 with the fixed response
scales:

```bash
export PYTHONHASHSEED=0
.venv/bin/python results/human_eval_pack/score_round2_ac1.py \
  --manifest results/human_eval_pack/releases/round2_presence_v2/round2_manifest.json \
  --ratings ratings_rater_1.json ratings_rater_2.json \
  --output round2_presence_ac1.json
```
