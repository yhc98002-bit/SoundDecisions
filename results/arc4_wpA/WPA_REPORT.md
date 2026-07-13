# WP-A Execution Report

Branch: `arc4-wpA`

## T0 - Preflight

Status: DONE

Files touched: `results/arc4_wpA/WPA_REPORT.md`

- Starting commit: `60efafdadff7c20577d88dc65201fe2a89b46b8c` (`main`).
- Environment: project `.venv`, Python 3.10.12, NumPy/SciPy/pytest with optional librosa and soundfile installed. The unactivated user Python lacks SciPy, librosa, and soundfile and stopped collection at `tests/test_labeling_tool.py`; it is not the project runtime used below.
- Baseline: `export PYTHONHASHSEED=0; export PATH="$PWD/.venv/bin:$PATH"; python -m pytest tests/ -q` -> 1023 passed, 16 warnings in 343.47 s.
- Gate-A dial caches: all 24 files matching `results/stage0/gate_a/dial_noise__dial_cfg1__*.npz` are present.
- Phase-1 journal evidence: `results/stage0/measurements/measurements.jsonl` and 200 `p1cfg1__*.json` completion journals are present.
- Phase-2 journal evidence: 200 `results/stage0/journal/p2cfg1__*.json` files are present. Their per-evaluation rows contain `clip` and `j` but no `gen_id`; T4 balanced accuracy must therefore be omitted and flagged.
- Stage-R condition-swap evidence: 40 `cswap__*.json` journals are present with per-pair `source_val`, `donor_val`, and `swap_val` fields for all four axes.
- Phase-1 per-video commitment curves: recoverable directly from `commitment_map_p1cfg1.csv` (200 clips x 8 s points for the shared axes).
- Frozen-file preflight: all eight existing SHA256 values recorded in `experiment/preregistered/SHA256SUMS.json` match.
- Tests added: none (preflight only).
- Deviations: none.

## T1 - Gap-aware R-class labels

Status: DONE

Files touched: `scripts/phase3_decision.py`, `tests/test_phase3_labels.py`, `results/arc4_wpA/phase3_decision_corrected.json`, `results/arc4_wpA/phase3_decision_corrected.md`, and this report.

- Class row before -> after: `early-action (committed & readable)` -> `R2-in-window (committed at 0.35, readable from 0.75)` for class (`s_commit=0.345930`, `s_read=0.750000`, gap `0.404070`).
- GO booleans: `GO_MAP=True` -> `True`; `GO_READOUT=True` -> `True`, with byte-identical JSON boolean encodings.
- Tests added: all five ordered branches, the negative-gap suffix, the corrected class row, and unchanged Arc-3 GO booleans. `tests/test_phase3_labels.py`: 3 passed.
- Determinism: two fresh processes produced identical corrected outputs (JSON SHA256 `297322ed8093713a6ef79d05f4d526cc43ccf86da0a9784ffec4e10f8742a167`; Markdown SHA256 `8266a6d110bcd1dca3a44f18b52d302fb8f854c09d641d8a9ec745008606a377`).
- Deviations: none.

## T2 - Matched-compute headroom

Status: DONE

Files touched: `foley_cw/policy_offline.py`, `scripts/phase4_policy.py`, `tests/test_policy_offline.py`, `tests/test_phase4_policy.py`, `results/arc4_wpA/policy_pareto_corrected.csv`, `results/arc4_wpA/policy_report_corrected.md`, and this report.

- Per-clip same-compute allocation: floor -> ceiling. Aggregate same-compute BoN NFE `46925` -> `51525`, now above gated NFE `49151`; final correctness `0.350` -> `0.365` because the rounded-up baseline evaluates more cached candidates.
- Headroom sentence before -> after: `NO` with gated `0.785/49151` and same-compute `0.350/46925` -> `YES` with gated `0.785/49151` and same-compute `0.365/51525` under the corrected 0.01 quality-gap and 2% compute-tolerance rule.
- Tests added/updated: ceiling allocation and aggregate NFE invariant; predicate pinned to the corrected 200-clip replay values and both rejection boundaries. Targeted result: 16 passed.
- Determinism: two fresh full cached replays produced identical outputs (CSV SHA256 `2a7b29d3ae5327a40c5a461d7bc74a965c48fd1fc353eff049cba33f2bdd12ab`; Markdown SHA256 `1d1f2aa140b367b7ffa4e6ecc25f64eb6b1c63cea69492691e6660dd2ff0e368`).
- Deviations: the corrected Pareto CSV is retained beside the required Markdown so every sentence value has a machine-readable source; no Arc-3 output was overwritten.
