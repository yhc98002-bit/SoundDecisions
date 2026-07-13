# Release notes

## Arc-4 WP-A test accounting

The Arc-3 report's **"1023 tests green"** is a historical result from the primary
working tree: `1023 passed` in the project `.venv`, with the full run-storage
artifacts available. It remains valid as the test result recorded at Arc-3 close;
it is not a fixed collection count for every distribution of the repository.

The trimmed GitHub copy collects approximately **931 tests**, depending on the
installed optional packages. That copy intentionally omits local environments,
model/data checkouts, raw measurement journals, feature tensors, preview/final
audio, and large cached arrays. The two numbers are therefore not directly
comparable: `1023` is a passed-test count from the full checkout, while `~931` is
an approximate collected-test count from a trimmed environment. Module-level
optional-dependency skips can reduce collection in the trimmed environment, and
artifact-dependent tests that are still collected must skip when their cached
inputs are absent. Neither case should be reported as a reproduced Arc-3 pass.

For reproducible reporting, record the checkout type, Python environment, and the
complete pytest summary (`passed`, `skipped`, and `errors`) with each run.

## Optional audio dependencies

| Package | Used for | Behavior when unavailable | Needed for full local suite |
|---|---|---|---|
| `librosa` | Real-measurer timing/onset detection; visual-anchor audio onset extraction and resampling | Production imports are lazy. Tests that exercise these paths use `pytest.importorskip`; the real-measurer timing test skips. | Yes, to execute rather than skip librosa-backed tests. |
| `soundfile` | RunStore preview/final WAV serialization and the audio labeling tool | RunStore imports are lazy and its WAV tests skip. The labeling-tool test module imports `soundfile` during collection, so a soundfile-free environment cannot collect the complete suite. | Yes, for complete collection and WAV-path coverage. |

Both packages are intentionally listed as optional/heavy comments in
`requirements.txt`; the NumPy-only analytic and synthetic core does not require
them at import time.
