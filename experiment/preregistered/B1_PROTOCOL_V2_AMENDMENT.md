# Arc-4 B-1 protocol V2 pre-evaluation amendment

Frozen before any B-1 completeness gate completed and before any B-1
evaluation metric or prediction was produced or read.

This amendment fixes two implementation details in `B1_PROTOCOL_V2.md` before
evaluation:

1. Balanced-accuracy bootstrap intervals use the full true-class universe of
   the corresponding unresampled confident split. A video-bootstrap draw that
   omits any class in that universe is discarded and replaced using the same
   seed-0 RNG stream until 1,000 valid draws have been collected. Outputs record
   the attempted, valid, and discarded draw counts. Raw accuracy is recomputed
   over all trajectory rows carried by the sampled videos; it is not an
   unweighted mean of per-video accuracies.
2. The completeness gate must validate an immutable collection-completion
   manifest and its SHA256 sidecar. That manifest records and hashes the launch
   ledger and worker logs, and pins cfg 1.0, `sqrt_down`, seed 0,
   `small_16k`, 20 steps, 8.0 seconds, 16 independents, the eight-point progress
   grid, 200 journals, 25,600 per-token bundles, and 25,600 matching pooled
   bundles. Per-clip journals alone do not establish all of those fields.

V2's use of outer balanced accuracy >= 0.70 plus raw margin over majority >=
0.15 is an explicit pre-evaluation correction to the older raw-accuracy screen,
required by the GPU-track instruction to use confident labels and balanced
accuracy. Raw accuracy, balanced accuracy, majority baseline, and their margin
remain published together. No threshold or rule may change after evaluation.
