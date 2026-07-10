# Validity Full Suite — §3.3 (DIAGNOSTIC; confusion matrix is the truth-teller)

Human clips: 50; timing bin = 1.5147s. The human-vs-measurer κ ≥ 0.6 is the correctness-layer gate (NOT a GO_MAPS_PHASE precondition under the self-target split). Scalars are on the confident subset (class abstain dropped); the confusion matrix keeps abstain.

## human_vs_measurer
| axis | n_conf | raw | Cohen κ | Gwet AC1 | PABAK | abstain a/b |
|---|---|---|---|---|---|---|
| presence | 50 | 0.720 | 0.112 | 0.603 | 0.440 | 0/0 |
| class | 27 | 0.296 | 0.227 | 0.235 | 0.232 | 18/12 |
| timing | 19 | 0.526 | 0.166 | 0.458 | 0.408 | 0/0 |

_timing ±1-bin agreement: 0.632 (exact-bin is harsh — measurer onset fires at t≈0)._

_class confusion (human_vs_measurer; rows=human, cols=measurer):_

| r\c | abstain | animals | doors_furniture | electronics_ui | food_cooking | footsteps_walk | guns_explosions | impact_friction | machines_motors | other | tools_hand | vehicles | water_liquid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abstain | 7 | 2 | 2 | 3 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 3 | 0 |
| animals | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| doors_furniture | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| electronics_ui | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| food_cooking | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| footsteps_walk | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| guns_explosions | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 | 0 | 0 |
| impact_friction | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 |
| machines_motors | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 |
| other | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| tools_hand | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| vehicles | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| water_liquid | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 |

## human_vs_qwen
| axis | n_conf | raw | Cohen κ | Gwet AC1 | PABAK | abstain a/b |
|---|---|---|---|---|---|---|
| presence | 50 | 0.880 | 0.658 | 0.817 | 0.760 | 0/0 |
| class | 32 | 0.656 | 0.622 | 0.628 | 0.628 | 18/0 |
| timing | 19 | 0.211 | -0.149 | 0.115 | 0.079 | 0/0 |

_timing ±1-bin agreement: 0.526 (exact-bin is harsh — measurer onset fires at t≈0)._

_class confusion (human_vs_qwen; rows=human, cols=qwen):_

| r\c | abstain | animals | doors_furniture | electronics_ui | food_cooking | footsteps_walk | guns_explosions | impact_friction | machines_motors | music | other | speech_vocal | tools_hand | vehicles | water_liquid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abstain | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 14 | 0 | 0 | 0 |
| animals | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| doors_furniture | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| electronics_ui | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| food_cooking | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| footsteps_walk | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| guns_explosions | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| impact_friction | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| machines_motors | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 1 | 1 | 0 |
| music | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| other | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| speech_vocal | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| tools_hand | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| vehicles | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| water_liquid | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |

## qwen_vs_measurer
| axis | n_conf | raw | Cohen κ | Gwet AC1 | PABAK | abstain a/b |
|---|---|---|---|---|---|---|
| presence | 100 | 0.750 | -0.043 | 0.674 | 0.500 | 0/0 |
| class | 70 | 0.329 | 0.264 | 0.284 | 0.281 | 0/30 |
| timing | 100 | 0.800 | 0.051 | 0.793 | 0.767 | 0/0 |

_timing ±1-bin agreement: 0.960 (exact-bin is harsh — measurer onset fires at t≈0)._

_class confusion (qwen_vs_measurer; rows=qwen, cols=measurer):_

| r\c | abstain | ambient_nature | animals | doors_furniture | electronics_ui | food_cooking | footsteps_walk | guns_explosions | impact_friction | machines_motors | music | other | speech_vocal | tools_hand | vehicles | water_liquid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abstain | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ambient_nature | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| animals | 2 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| doors_furniture | 2 | 0 | 1 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| electronics_ui | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| food_cooking | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| footsteps_walk | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| guns_explosions | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| impact_friction | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 3 | 0 |
| machines_motors | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| music | 6 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 2 | 0 |
| other | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| speech_vocal | 14 | 0 | 6 | 3 | 5 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 0 | 3 | 0 |
| tools_hand | 2 | 0 | 1 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| vehicles | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 0 |
| water_liquid | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 |

