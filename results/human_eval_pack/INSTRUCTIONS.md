# Blinded rating instructions

**Do not begin rating until the signed freeze envelope has been announced and
the coordinator supplies a newly hashed authorized manifest.** The candidate
manifest committed with this instrument is deliberately locked. Use only the
assigned `rate.html` and `media/` directory. Do not seek source IDs or open an
unblinding file.

## Start and recovery

1. Open `rate.html` directly in a browser. No server or network connection is
   needed.
2. Enter the assigned rater ID. The item order is deterministically shuffled
   for that ID; do not share an ID with another rater.
3. Work through all three tabs. Progress saves in the browser after every
   change.
4. Use **Export ratings JSON** at the end of each session. To resume on another
   browser, use **Import ratings JSON**. Keep the newest exported file.

## Anchor marking

Keep audio muted while marking the visual event. Briefly describe the visible
target event, then mark its closed interval `[start, end]`. Use the frame-step
buttons for the boundary frames; playback speeds are 0.25x, 0.5x, and 1x.

The specification states: "Every event has a stable `event_id`, description,
and visual anchor represented as a closed time interval `[lo_s, hi_s]` with
source and provenance. An exact frame without justified uncertainty is
forbidden." Audio-only onset detection cannot define the visual anchor. Select
**Too uncertain** when a defensible interval cannot be marked; do not guess.

## Presence rubric

Enable audio for this screen and judge the event named on the Anchor tab.

> For one specified visible event, determine whether a corresponding audio
> event occurs near its anchor: `present`, `absent`, or `uncertain`. Salient
> unrelated background audio does not count. The unit is the event.

Choose **Target-present**, **Absent**, or **Uncertain**. Separately mark whether
unrelated background audio is present. The note is optional and should only
explain the rating.

## Two-event curation

Keep audio muted while curating visible events. Confirm only when two separable
target events exist. Describe and mark both closed intervals, ordering Event 1
before Event 2 by interval start. Reject the item when two defensible,
separable events cannot be identified; do not force ambiguous cases.

The exported file is named `ratings_<raterID>.json` and conforms to
`ratings.schema.json`. It contains blinded IDs only.
