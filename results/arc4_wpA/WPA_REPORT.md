# WP-A Execution Report
Branch: `arc4-wpA`. Starting commit: `60efafdadff7c20577d88dc65201fe2a89b46b8c` (`main`). CPU-only; project `.venv`; every Python shell used `PYTHONHASHSEED=0`.

## T0 - Preflight
Status: DONE. Files: this report.
Baseline `1023 passed, 16 warnings`; 24/24 cfg1 dial caches; Phase-1 measurement and 200 completion journals; 200 Phase-2 journals whose rows lack `gen_id`; 40 swaps with source/donor/swapped values; complete per-clip Phase-1 curves; all eight frozen hashes matched. No test added; no deviation. The unactivated user Python dependency error is listed under out-of-scope findings.

## T1 - Gap-aware R-class labels
Status: DONE. Files: `scripts/phase3_decision.py`, `tests/test_phase3_labels.py`, `phase3_decision_corrected.{json,md}`.
Class changed `early-action` -> `R2-in-window (committed at 0.35, readable from 0.75)` for gap `0.404070`; GO_MAP/GO_READOUT stayed byte-identical `True/True`. All five branches plus suffix and GO invariance: 3 passed. Two-process hashes: `297322ed...`, `8266a6d1...`. No deviation.

## T2 - Matched-compute headroom
Status: DONE. Files: `foley_cw/policy_offline.py`, `scripts/phase4_policy.py`, policy tests, `policy_{pareto,report}_corrected.*`.
Candidate allocation floor -> ceiling: same-compute `0.350/46925` -> `0.365/51525`, versus gated `0.785/49151`; sentence `NO` -> `YES` under +0.01 quality and 1.02 compute rules. Targeted 16 passed. Two-process hashes: `2a7b29d3...`, `1d1f2aa1...`. Auxiliary corrected CSV retained; Arc-3 untouched.

## T3 - Bridge statistics
Status: DONE. Files: `scripts/b4_bridge.py`, `tests/test_b4_bridge_stats.py`, `b4_bridge_corrected.{json,md}`; `foley_cw/bridge.py` untouched.
`hash(axis)` -> CRC32; scalar floor moved inside each bootstrap. Arc-3 provisional inclusive `0.513997 [0.354651,0.647799]` -> corrected inclusive diagnostic `0.526367 [0.311186,0.636955]`; citable material-excluded `0.515191 [0.360404,0.537848]`; joint recovery `0.000000`. Material tagged `UNCALIBRATED_COSINE`; scalar labels/caveat and exact scoring ledger added. Seeds 0..3 citable means `0.515191/0.483796/0.502525/0.464286`, all `BRIDGE_PARTIAL` (stable). CRC32/floor tests plus bridge tests: 13 passed. Two-process hashes: `ce48f443...`, `9d6a2750...`. No deviation.

## T4 - Phase-2 aggregation
Status: FLAGGED. Files: `scripts/phase2_readout.py`, `tests/test_phase2_readout_aggregate.py`, `readout_map_p2cfg1_v2.{csv,md}`.
Per-clip means plus 1000 clip bootstraps replace pooled means; metric, CI, majority, margin, and dual s_read fields added. All 32 points reproduce legacy (max error `2.22e-16`); timing s=.05 is `0.946250 [0.927469,0.963750]`, baseline `0.941250`, margin `0.005000`, not requested ~0.02-0.05. Balanced accuracy omitted because rows lack `gen_id`; Track-P omitted because per-example predictions were not persisted. Five tests passed. Two-process hashes: `1996ab42...`, `2b823c1c...`.

## T5 - Collision-corrected swaps
Status: FLAGGED. Files: this report only; no script/report/token emitted after semantic stop.
All 40 journals reproduce 32 committed cells with exact `n` and max rate error `0.0`; all per-donor ceilings join. Conflict: donor!=source class s=.05 is `8/17=0.470588`, while AMD-18 fixes `8/20=0.40`. Material existing-match semantics give 39/40 collisions and floor `0.9878125` above ~0.78 ceiling, while strict cosine ties give zero. No test added because implementation stopped rather than improvising.

## T6 - Abstain-filtered entropy lens
Status: DONE. Files: `scripts/c_two_budgets.py`, its tests, `entropy_lens_v2.{json,md}`.
Default output remains byte-identical. Inclusive distinct series `[4.8333,4.1250,3.7500,3.7083,3.7500,3.6250]`; excluding abstain `[3.8333,3.1250,2.7917,2.7917,2.8333,2.7083]`; abstains `[135,142,142,129,128,120]/384` with Wilson CIs. Exact 24-cache guard and missing-cache skip added; 12 passed. Two-process hashes: `f6e4b842...`, `c36ad2ff...`. No deviation.

## T7 - Window robustness
Status: DONE. Files: `scripts/arc4_window_robustness.py`, `window_robustness.{csv,md}`.
At theta .70: timing `77/200`, legacy `0.113636`, censored median `1.0`; presence `113/200`, `0.214159`, `0.675`; material `200/200`, `0.638250`, `0.600`. Legacy ordering is stable across all five thresholds; censored ordering is not. No unit test (new diagnostic), but strict grid/provenance/legacy guards and py_compile passed. Two-process hashes: `848d4e2e...`, `ca9bd83f...`. No deviation.

## T8 - Test/release hygiene
Status: DONE. Files: `tests/test_real_measurer.py`, `RELEASE_NOTES.md`.
Librosa timing path now skips when absent; notes distinguish full-artifact `1023 passed` from trimmed-copy ~931 collected and document librosa/soundfile. T8-targeted real-measurer run: 18 passed; no-librosa timing: skipped. Deviation: out-of-scope labeling-tool soundfile collection import was documented, not changed.

## T9 - Operations
Status: DONE. Files: `real_measurer.py` and tests, `run_on_node.sh`, current node references in scripts/CLI/plan/ORBIT state.
Current `an17` -> `an12`; wrapper exports hash seed. CLAP pinned `8fa0f1c...`, AST `f826b80d...`; all four Transformers loads use `local_files_only=True`. `FOLEY_CW_WEIGHTS_SOURCE=modelscope|hf` defaults to a required local ModelScope mirror; neither mode downloads. Real-measurer 22 passed; bash/py_compile passed. Deviation: mirror absent, so existing HF cache needs explicit `FOLEY_CW_WEIGHTS_SOURCE=hf`.

## T10 - Status documentation
Status: DONE. Files: `results/CURRENT_STATUS.md` and top-only banners in the three requested historical documents.
Status records `1041 passed` and code-completion commit `b3e1974b0a0a1f64d650eec0604c662a8b0d841a`; bodies unchanged. No test. Deviation: Git field uses T9 because a file cannot contain its own commit hash; T10/T11 are governance-only.

## T11 - Amendment ledger
Status: DONE. Files: new `amendments_arc4.md`, append-only `SHA256SUMS.json`.
AMD-13..18 added verbatim; appended hash `729e8e6984d18c76420c46e0a94054c47fdcc50396988528bc33a58ab02eabdf`; existing entries unchanged. No test or deviation.

## Final verification gate
1. PASS - all eight pre-existing frozen files match recorded SHA256 values.
2. PASS - full suite: **1041 passed, 16 warnings, 0 failures** in 235.27 s.
3. PASS - T1..T4 outputs are byte-identical across two fresh processes (hashes above).
4. PASS - T5 raw join: 40 journals, 32 cells, exact `n`, max error `0.0`.
5. PASS - `an17` grep is empty excluding `.git` and historical result/snapshot documents.
6. PASS - exactly 12 commits, one each for T0..T11; branch not pushed or merged.

## Out-of-scope findings
- User Python lacks SciPy/librosa/soundfile; authoritative runs used the complete project `.venv`.
- `tests/test_labeling_tool.py` imports soundfile at collection; documented, not changed.
- No CLAP/AST ModelScope mirror is currently populated under `weights/modelscope`.
- Pre-existing user edits in `AGENTS.md` and `CLAUDE.md` remain uncommitted and untouched.
