# Stage-0 Gate — Self-Target Split (§3.3)

Self-target gate (determinism ≥ 0.95 AND robustness ≥ 0.85); validity is correctness-layer (NOT a precondition).

| axis | det | rob | self-target | note |
|---|---|---|---|---|
| presence | 1.000 | 0.983 | PASS |  |
| timing | 1.000 | 0.955 | PASS |  |
| class | 1.000 | 0.833 | fail | kept DIAGNOSTIC (in maps, not gating) |
| material | 1.000 | 0.976 | PASS |  |

**3/4 axes pass the self-target gate (need ≥ 3).** validation='OK', manifest_ok=True.

**Tokens: `GO_MAPS_PHASE`**

trajectory access OK; SDE validation token='OK'; manifest OK; 3/4 axes passed reliability gate (threshold=3).
