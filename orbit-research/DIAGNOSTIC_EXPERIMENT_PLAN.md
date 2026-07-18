# Non-human closure diagnostic and scale plan

## Frozen inputs

- Git start: `a1e8f3ae324e8886379c19c5bc312d7ebc942946`
- B2: four immutable quarantine roots totaling 79,152 WAV files.
- Class measurer: pinned local PANNs Cnn14 checkpoint and committed legacy
  coarse taxonomy.
- B-1: deterministic four-clip calibration set plus held-out clip `1002`.
- Material: exact 6,400 legacy cells and retained final generations; material
  truth may come only from exact unambiguous existing metadata categories.

## Diagnostic gates

1. Validate inventory cardinality, unique IDs, WAV hashes, and checkpoint/map
   hashes without regenerating audio.
2. Run schema, deterministic-reproduction, corruption, partial-shard, and
   leakage tests for each implementation.
3. Class: measure a deterministic smoke shard and validate full- and coarse-
   posterior normalization, row identity, and immutable shard completion.
4. B-1: collect all compared paths in one forward evaluation. Estimate the
   numerical tolerance exclusively on the four calibration clips, freeze it,
   then apply it once to held-out clip `1002`.
5. Material: freeze candidate/reference identities before computing any cosine
   margin. If strict metadata-only matching misses the frozen coverage floor,
   stop as `INCOMPLETE_ARTIFACTS` without replay.

## Scale decision

Full Class measurement may launch only after the smoke shard and merge checks
pass. Full B-1 recollection and probe fitting may launch only if the identity
pilot passes. Material replay may launch only if the frozen metadata reference
manifest meets the registered coverage floor. Every worker writes an immutable
node/GPU shard; no worker appends to a shared scientific result file.

## Failure semantics

A process exit is not a scientific result. Failed attempts remain under an
immutable run root. Engineering failure cannot be interpreted as a negative
readout finding, and missing Material labels cannot be replaced with embedding-
defined ground truth.

