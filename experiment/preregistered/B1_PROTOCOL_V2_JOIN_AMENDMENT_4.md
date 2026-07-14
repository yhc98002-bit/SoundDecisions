# Arc-4 B-1 protocol V2 join amendment 4

Frozen before scanning beyond the previously disclosed first-10 join diagnostic
and before any B-1 evaluation metric or prediction was produced or read.

The per-layer `2e-4` condition added in join amendment 3 is withdrawn. The
binding identity rule is the whole-bundle relative-L2 condition fixed there:

`||token_mean - pooled||_F / max(||pooled||_F, 1e-12) <= 2e-4`.

The second full-gate attempt stopped on the already disclosed first-10 bundle
`1002__p1cfg1_ind0__s0.15.npz`. Its whole-bundle relative-L2 error was
`9.523e-5`, while one cancellation-small layer had relative-L2 `6.24e-4`.
Thus the added per-layer denominator reintroduced the same small-value
rounding sensitivity that the normwise correction was meant to remove. No gate
artifact, class prediction, probe metric, or undisclosed corpus-wide join
distribution was produced or inspected.

Exact names, schemas, dtypes, finite values, content hashes, frozen input
hashes, and the 25,600 whole-bundle checks remain mandatory. Amendments 1–3
remain audit history; amendment 4 is the binding pre-evaluation join rule.
