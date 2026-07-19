# Human-evaluation delivery

The current reviewer package is the node-local archive
`releases/round1_curation_v4.zip` (SHA256
`f2af52d17c916be234acb1e8e83470e8a25d2b3725dc6b98554ebc7826eec71b`).
It is ready for one assigned lead curator. No electronic signature is required.

Do not distribute v1, v2, or v3. They were retired before delivery and are recorded
in `RETIRED_RELEASES.json`. The older top-level `rate.html` is a disabled,
historical pre-release instrument; it is not the current entry point.

## Give this to the Round-1 curator

1. Transfer `round1_curation_v4.zip` and verify its SHA256.
2. Extract it without changing the directory structure.
3. Open the extracted `rate.html` directly in a browser (`file://`); no server or
   network connection is needed.
4. Use the assigned curator ID, complete all items, and return the exported
   `ratings_<raterID>.json` unchanged.

Round 1 contains 82 blinded videos and 90 assignments: 30 anchor-curation and
60 two-event-curation assignments, with eight videos serving both tasks. Its
media are physically video-only. Captions help locate candidate visible actions;
all event descriptions must use only visible actions and objects.

## After the export returns

Reduce exactly one curator export to a fixed event catalog:

```bash
export PYTHONHASHSEED=0
.venv/bin/python results/human_eval_pack/curate_round1.py \
  --manifest results/human_eval_pack/round1_v4_blinded_items.json \
  --ratings ratings_<curatorID>.json \
  --output event_catalog_v2.json
```

Then build Round 2 from the separate audio-bearing blinded media root. Never
point this command at the silent Round-1 release:

```bash
export PYTHONHASHSEED=0
.venv/bin/python results/human_eval_pack/build_round2.py \
  --event-catalog event_catalog_v2.json \
  --media-root results/human_eval_pack \
  --audio-media-registry results/human_eval_pack/private/round1_v4_audio_media_registry.json \
  --round1-release-record results/human_eval_pack/ROUND1_V4_RELEASE.json \
  --output-dir results/human_eval_pack/releases/round2_presence_v2
```

Round 2 is completed by one evaluator whose ID differs from the curator's. Its
single-rater output is descriptive only. `score_round2_ac1.py` is the separate
agreement path if two or more independent Round-2 exports are later collected.

See `ROUND1_RELEASE_REPORT.md` for the audit, limitations, and verification
commands.
