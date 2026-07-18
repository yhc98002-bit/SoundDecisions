# Class posterior measurement

Artifact status: `COMPLETE`. Scientific status: `NOT_TESTED` (this file validates measurement, not a Class claim).

The canonical B2 bank contains 79,152 measured WAVs: 816 base finals and 78,336 fork finals. The persisted archive includes the full 527-way output, normalized 15-way coarse posterior, coarse sums, top/confident label, confidence, margin, entropy, abstention, all IDs, hashes, and model/measurer revisions.

- Overall abstention: 0.3056
- Base-final abstention: 0.2941
- Fork-final abstention: 0.3057
- Coarse map: `v3-2026-06-12` / `55b5a1d4116caa4503a6b4b17192425da487a9c4385a287e343d850795be4fe7`
- Posterior data SHA-256: `a381e0f8e662aae482959f0cd2e7fe46f3cb1e8ef2ef7040b4e65913ff86cd1e`

| s | records | abstention | mean confidence | mean margin | mean entropy |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 9792 | 0.3146 | 0.3183 | 0.2026 | 3.3637 |
| 0.15 | 9792 | 0.3143 | 0.3207 | 0.2044 | 3.3535 |
| 0.25 | 9792 | 0.3065 | 0.3237 | 0.2063 | 3.3446 |
| 0.35 | 9792 | 0.3037 | 0.3237 | 0.2055 | 3.3447 |
| 0.45 | 9792 | 0.3026 | 0.3231 | 0.2053 | 3.3439 |
| 0.60 | 9792 | 0.2982 | 0.3242 | 0.2066 | 3.3370 |
| 0.75 | 9792 | 0.3056 | 0.3254 | 0.2074 | 3.3313 |
| 0.90 | 9792 | 0.3002 | 0.3281 | 0.2102 | 3.3229 |

No abstention threshold or coarse taxonomy was tuned on B2 outcomes.
