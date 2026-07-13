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

## T3 - Bridge statistics and calibration

Status: DONE

Files touched: `scripts/b4_bridge.py`, `tests/test_b4_bridge_stats.py`, `results/arc4_wpA/b4_bridge_corrected.json`, `results/arc4_wpA/b4_bridge_corrected.md`, and this report. `foley_cw/bridge.py` was not changed.

- Axis bootstrap seeding: process-randomized `hash(axis)` -> stable `zlib.crc32(axis.encode("utf-8")) % 1000`.
- Bootstrap scalar floor: fixed full-data floor -> best scalar-policy mean recomputed inside every clip resample. Presence CI changed from Arc-3 `[0.351514, 0.740283]` to `[0.128809, 0.670316]`; the full-data point remains governed by the same best-on-full-data definition (T2's allocator correction separately changes the scalar roster and raises presence recovery from `0.554688` to `0.604167`).
- Mean recovery before -> after: Arc-3 material-inclusive provisional `0.513997 [0.354651, 0.647799]` -> corrected material-inclusive diagnostic `0.526367 [0.311186, 0.636955]`; corrected material-excluded **citable** value `0.515191 [0.360404, 0.537848]`.
- Material is tagged `UNCALIBRATED_COSINE`; overall joint recovery remains `0.000000` (95% CI `[0.000000, 0.127712]`).
- Policy display labels now identify `diffrs_scalar` as `final_window_scalar_reject` and `smc_scalar` as `final_window_scalar_resample` while preserving JSON keys. The Markdown contains the requested s=0.90 scalar caveat and exact scoring-call ledger; no matched-scoring wording remains.
- Seed 0/1/2/3 citable means: `0.515191`, `0.483796`, `0.502525`, `0.464286`; each tier token is `BRIDGE_PARTIAL`, so the tier token is seed-stable. Separate `--seed 1`, `--seed 2`, and `--seed 3` runs exactly matched the seed-0 robustness block.
- Tests added: CRC32 seed pin and synthetic bootstrap comparison against a fixed-floor control; targeted result: 13 passed including the existing bridge tests.
- Determinism: two fresh seed-0 processes produced byte-identical outputs (JSON SHA256 `ce48f44351c572d4f386ec3ac17acbb684dd94a60b7f3860e2d70c889900e014`; Markdown SHA256 `9d6a2750a27c16206d26c09a007a847da294d4b84f51ff6ad13e42679e2d4095`).
- Deviations: none.

## T4 - Phase-2 aggregation upgrade

Status: FLAGGED (core v2 aggregation completed; three requested lenses are artifact-limited or acceptance-inconsistent)

Files touched: `scripts/phase2_readout.py`, `tests/test_phase2_readout_aggregate.py`, `results/arc4_wpA/readout_map_p2cfg1_v2.csv`, `results/arc4_wpA/readout_map_p2cfg1_v2.md`, and this report.

- Aggregation before -> after: pooled row means without uncertainty -> per-clip means followed by a 1000-draw clip bootstrap (seed 0), with `metric`, 95% CI, `n_clips`, categorical majority baseline, and margin columns.
- Legacy reproduction: all 32 `(axis, s)` point values match to tolerance; maximum absolute difference is `2.22e-16` (required `<=1e-9`). Material is labeled `cosine` and is not called accuracy in the Markdown.
- Timing at `s=0.05`: exact match `0.946250 [0.927469, 0.963750]`; literal mode frequency of the 800 evaluated ODE targets is `753/800 = 0.941250`; margin `0.005000`.
- `s_read_margin`: class `0.60`; presence and timing never reach `+0.15`; material is not applicable. Legacy absolute-threshold readout remains presence `0.35`, timing `0.05`, class `0.75`, material `0.60`.
- FLAGGED - acceptance mismatch: the requested timing margin of approximately `0.02-0.05` is inconsistent with the cached target-label mode frequency; the specified majority semantics yield exactly `0.005`. No alternate baseline was substituted.
- FLAGGED - balanced accuracy: omitted because the Phase-2 journal rows do not carry `gen_id`, as established in T0. Only this column is omitted.
- FLAGGED - Track P: persisted output contains aggregate best-layer scores and layer IDs but no per-example predictions or clip IDs. A clip bootstrap is not recoverable without retraining and choosing unregistered layer-selection semantics.
- Tests added: clip bootstrap, majority/margin semantics, optional balanced-accuracy emission, legacy equality guard, and material reporting; targeted result: 5 passed.
- Determinism: two fresh processes produced byte-identical outputs (CSV SHA256 `1996ab42ec2f6d286c7d760ea90729a9a6af3d504405213b1769eb0fe6e79c7b`; Markdown SHA256 `2b823c1ce932be20ef5b5b56a152862906fe6ad9067784b4b37955014419cbcf`).
- Deviations: limited to the three explicit flags above; Arc-3 outputs remain untouched.
