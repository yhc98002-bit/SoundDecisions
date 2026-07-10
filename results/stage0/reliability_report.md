# Axis Reliability Report — Stage 0 (manual section 3.3; feeds Tab. 1)

Thresholds (frozen): determinism >= 0.95, robustness >= 0.85, validity kappa >= 0.6 (vs MLLM sidecar; HUMAN sidecar pending at the PI checkpoint).

| axis | determinism | robustness | validity (kappa vs MLLM) | passed | demoted | reason |
|---|---|---|---|---|---|---|
| presence | 1.000 | 0.983 | -0.125 | False | True | AXIS_DEMOTED:presence — validity -0.125 < theta_cal 0.600 |
| timing | 1.000 | 0.955 | -0.028 | False | True | AXIS_DEMOTED:timing — validity -0.028 < theta_cal 0.600 |
| class | 1.000 | 0.833 | 0.145 | False | True | AXIS_DEMOTED:class — robustness 0.833 < theta_robust 0.850; validity 0.145 < theta_cal 0.600 |
| material | 1.000 | 0.976 | nan | False | True | AXIS_DEMOTED:material — validity nan < theta_cal 0.600; no MLLM-judgeable gold for an embedding axis — second-embedder validation is a PI-checkpoint question |

**Surviving axes: [] (0; need >= 3 for GO_MAPS_PHASE) — INSUFFICIENT**

**Tokens:** AXIS_DEMOTED:presence, AXIS_DEMOTED:timing, AXIS_DEMOTED:class, AXIS_DEMOTED:material
