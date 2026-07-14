# Arc-4 B-1 protocol V2 join amendment 3

Frozen before rerunning the full-corpus join gate and before any B-1 evaluation
metric or prediction was produced or read.

The global relative-L2 threshold in join amendment 2 is tightened from `2e-3`
to `2e-4`, applied both to each complete `(12, 448)` bundle and separately to
each of its 12 layers. This value is approximately twice the maximum `9.5e-5`
relative-L2 error in the previously disclosed first-10 non-outcome diagnostic
and was fixed before scanning the remaining corpus.

Code audit identified the source of the nonzero join error: the original pooled
tap computes the token mean on GPU and then stores float16, while the retap
stores tokens in float16 before computing the mean in float32 and storing the
result in float16. These rounding orders are not elementwise identical near
cancellation, but a matching trajectory remains close in representation norm.
For the one bundle that stopped the first gate, global relative-L2 was
`4.58e-5`, maximum layer relative-L2 was `5.10e-5`, and float64 cosine was
`0.999999998953`.

Join amendments 1 and 2 remain audit history. Amendment 3 is the binding
pre-evaluation identity rule. Exact file names, schemas, dtypes, finite values,
input hashes, and all 25,600 pair checks remain mandatory.
