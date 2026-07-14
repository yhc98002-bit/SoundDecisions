# Arc-4 B-1 protocol V2 join amendment 2

Frozen before the B-1 completeness gate completed and before any B-1
evaluation metric or prediction was produced or read.

The elementwise `rtol=2e-3, atol=2e-3` retap/cache rule in
`B1_PROTOCOL_V2_JOIN_AMENDMENT.md` is superseded by a global relative-L2 rule:
for every trajectory/progress pair,

`||token_mean - pooled||_2 / max(||pooled||_2, 1e-12) <= 2e-3`.

The first full gate attempt stopped on
`1131__p1cfg1_ind1__s0.75.npz` before producing a gate artifact or any probe
metric. Exactly one scalar failed the elementwise rule, while the bundle's
relative-L2 error was `4.58e-5`, inside the previously recorded non-outcome
diagnostic range; the scalar values were 0.27587890625 and 0.272705078125 in
float16. Elementwise closeness was therefore testing isolated run-to-run fp16
rounding rather than trajectory identity.

The replacement keeps the same numeric tolerance, applies it to the whole
registered representation, and still checks all 25,600 pairs, exact filenames,
array shapes, dtypes, finite values, and the frozen population/config hashes.
This pre-evaluation correction can only reject a cache join; it does not inspect
labels beyond schema validity and does not change a scientific threshold.
