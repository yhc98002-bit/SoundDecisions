# Stage-0 Validity Diagnostic — Cohen's κ vs Gwet's AC1 (DIAGNOSTIC ONLY)

Frozen θ_cal = 0.6 is **unchanged**; this only characterises *why* the κ gate reads low. κ collapses under skewed marginals (the κ paradox); AC1 is the skew-robust counterpart. Both are reported; the PI arbitrates the statistic and the pending human sidecar.

| axis | n | raw agree p_o | Cohen κ (gate) | Gwet AC1 |
|---|---|---|---|---|
| presence | 100 | 0.750 | -0.043 | 0.674 |
| timing | 100 | 0.800 | 0.051 | 0.793 |
| class | 100 | 0.230 | 0.191 | 0.181 |

**Marginals (show the skew that breaks κ):**
- presence: measured={'present': 92, 'absent': 8}  gold={'present': 81, 'absent': 19}
- timing: measured={'0': 92, '4': 1, '1': 6, '2': 1}  gold={'0': 85, '1': 12, '6': 1, '9': 1, '7': 1}
- class: measured={'impact_friction': 5, 'vehicles': 24, 'electronics_ui': 10, 'animals': 13, 'doors_furniture': 8, 'abstain': 29, 'water_liquid': 2, 'tools_hand': 1, 'other': 3, 'guns_explosions': 5}  gold={'impact_friction': 8, 'speech_vocal': 34, 'food_cooking': 1, 'tools_hand': 7, 'animals': 6, 'music': 10, 'doors_furniture': 8, 'footsteps_walk': 1, 'vehicles': 10, 'guns_explosions': 6, 'machines_motors': 2, 'water_liquid': 4, 'electronics_ui': 2, 'ambient_nature': 1}

## Class axis — confound decomposition

- **full** (n=100): p_o=0.230, κ=0.191, AC1=0.181
- **gold-event-restricted** (drop 45 clips whose gold ∈ ['ambient_nature', 'music', 'speech_vocal'] — the 15-vs-12 label-space mismatch the measurer can never satisfy) (n=55): p_o=0.418, κ=0.344, AC1=0.373
- **both-confident** — *favorable upper bound*, conditions on measurer non-abstention (additionally drop 9 measurer abstentions) (n=46): p_o=0.500, κ=0.413, AC1=0.459 — class fails even here.

_AC1 q = number of categories observed in the rater union (a diagnostic choice; a future GATE on AC1 should preregister the full rating-scale q or report q-sensitivity — for timing the two agree to ~0.003)._

## Cohen-κ cross-check vs the frozen gate (faithfulness of this join)
Reproduces the gate's seed-0 50-clip determinism subsample and matches its `validity` to establish statistic-level faithfulness (the per-clip (clip, measured, gold) triples are in `reliability_diagnostic.json` for a literal audit).
- presence (n=50): diagnostic κ=-0.124567 vs gate validity=-0.124567 → MATCH
- timing (n=50): diagnostic κ=-0.028278 vs gate validity=-0.028278 → MATCH
- class (n=50): diagnostic κ=0.145121 vs gate validity=0.145121 → MATCH
