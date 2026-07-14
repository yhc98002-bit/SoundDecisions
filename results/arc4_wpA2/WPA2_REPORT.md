# WP-A2 execution report

Branch `arc4-wpA2`, based on canonical WP-A head `d261b0d`. Primary-tree work was CPU-only in the project `.venv`; every Python process used an explicit `PYTHONHASHSEED`.

## Corrective patch

- **CO-1 - DONE.** Added the supplied verifier unchanged as `scripts/arc4_wpA2_verify.py`; its SHA256 is the required `b6f0b3624864dce5345060e657d26d1bbeeb254a4bd7b5a3e3bdb599af017975`, and only the first WP-A2 commit touches it.
- **P1 - DONE.** Updated `results/PI_REPORT_arc3.md`, `PI_review_guidance.markdown`, and the two experiment plans. Review routing now starts at `results/CURRENT_STATUS.md`; current claims changed from fully resolved / F-1 refuted / DIAGNOSTIC-strong to diagnostic-exploratory, mechanism UNRESOLVED, with pending Arc-4 tokens. Historical report bodies remain audit history.
- **P2 - DONE.** Updated `scripts/b4_bridge.py`, `results/arc4_wpA/b4_bridge_corrected.{json,md}`, and added `tests/test_b4_joint_floor.py`. The joint scalar floor is now recomputed within each video-bootstrap draw: fixed-floor CI `[0.000000, 0.127712]` -> resampled-floor CI `[0.000000, 0.190920]`. **Joint recovery is zero under the simulated symmetric keep-flip error model.** Every per-axis mean is non-citable sensitivity output.
- **P3 - DONE.** Updated `scripts/phase4_policy.py`, `results/arc4_wpA/policy_report_corrected.md`, and appended AMD-22. False matched-scoring wording was removed. The comparator is conservatively rounded up: baseline NFE `51,525` versus gated `49,151`, or 4.8% more in the baseline's favour; the screen remains YES.
- **P4 - DONE.** Updated `scripts/c_two_budgets.py`, its tests, and emitted `entropy_lens_v3.{json,md}`. Default output changed from a withdrawn mode-collapse claim with label-level Wilson intervals to descriptive counts with `clip_bootstrap`; the Arc-3 narrative is available only under `--legacy-arc3` at a historical path.
- **P5 - DONE.** Updated canonical provenance in `results/arc4_wpA/WPA_REPORT.md` and `results/CURRENT_STATUS.md`; added `.github/workflows/cpu-tests.yml`, `tests_full.log`, and its checksum. Computed float assertions across the affected tests now use tolerant comparisons. Public history is base `c3d2f2f`, code-completion `71ca39e`, WP-A head `d261b0d`, fast-forwarded to `main`. Full-artifact suite: **1061 passed**.

## Axis validity

- **A2-1 - DONE.** Added `scripts/arc4_axis_validity.py`, its regression tests, and `axis_validity.json`. Presence is DEGENERATE (`majority_share=0.8778125`, `k_eff=1.273099`); timing is DEGENERATE (`0.940625`, `1.128591`); class is INFORMATIVE (`k_eff=6.233897`); material is INFORMATIVE at Tier-0 (`relative_agreement=0.412115`). All four within-minus-between video-bootstrap CIs have positive lower bounds.
- **A2-2 - DONE.** Added `scripts/arc4_class_reconstruction.py`, repaired `foley_cw/commitment.py`, added `tests/test_commitment_abstain.py`, and emitted `class_reconstruction.{json,md}`. Raw journals reproduce the committed class CSV with maximum delta `0.0`. Confident vs naive: `s_commit 0.345930 -> 0.462593`, gap `0.404070 -> 0.287407`, and crossing clips `172 -> 135`; the lenses differ, so the sensitivity check has power. The R2-in-window gap survives on the production confident-subset curve.
- **A2-3 - DONE.** Added `scripts/arc4_determination_partition.py`, tests, `determination_partition.csv`, and `window_partitioned.json`. Video-determined clips, previously liable to be treated as censored, are now separate: presence `70`, timing `121`, class `10`, material `0`. Crossing/censored-only Kaplan-Meier medians are reported beside the exactly reproduced legacy means.
- **A2-4 - FLAGGED.** `FLAGS.json` records that Phase-2 material rows retain only scalar own-target cosine, with no candidate/reference embeddings or preview WAVs. Legacy absolute cosine remains visible, but no relative 2AFC commitment/readout number was synthesized. Cheapest resolution: rerun only the cached Phase-2 material preview/measurer path after freezing matched negatives and persist the three 512-D embeddings.
- **A2-5 - DONE.** Updated `scripts/phase2_readout.py`, aggregation tests, and emitted `readout_map_v3.{csv,md}` plus `axis_diagnostics.md`. Every cached row joined through deterministic subject IDs; pooled values reproduce the legacy table within `2.22e-16`. Timing at `s=0.05` is `0.005` over majority, not the earlier expected 0.02-0.05. Balanced-accuracy bootstraps keep the full true-class universe; timing's CI is `[0.275988, 0.427380]`. Track-P remains unavailable because per-example predictions were not persisted; retraining is required.
- **A2-6 - DONE.** Added `scripts/arc4_swap_final.py`, tests, and `swap_final.json`. All 32 committed swap cells reproduce with maximum delta `0.0`. Class follow-only is published as two estimands: unconditional `8/20=0.400` and primary donor-not-source `8/17=0.471`, each with Clopper-Pearson CIs. Material uses nearest-reference; no cross-scale floor/ceiling comparison or mechanism token is emitted.
- **A2-7 - DONE.** Appended AMD-19 verbatim and re-registered `_amendment_2026-07_arc4` as `31d643159e4af14bd0921c6df91133a262a41adada6f4ae3c538bdd0e9838996`. Existing preregistration entries were not changed.

## Which axes survive

Only **class** currently supports a commitment/readout claim: it is Tier-0 INFORMATIVE, the raw-journal reconstruction is exact, and the confident R2 gap remains `0.404070`. **Presence** and **timing** do not survive Tier-0 because their cohort marginals cross the frozen degeneracy thresholds. **Material** passes Tier-0 informativeness but does not yet survive the scale correction: A2-4 is flagged, so relative commitment/readout remains unresolved rather than negative.

## Reproduction and deviations

`REPRO.json` records 14 final artifacts reproduced byte-identically in fresh processes at hash seeds `0, 0, 1`. `numbers_index.json` traces every headline number above to a committed artifact and key. The only task-level deviation is A2-4's evidence-backed missing-cache flag. No out-of-scope scientific finding was modified; unavailable Track-P examples are documented but not reconstructed.

## Verification gate

- PASS: eight pre-existing frozen files retain their registered hashes; the append-only amendment digest covers AMD-19 and AMD-22.
- PASS: full primary-tree suite has **1061 passed**, zero failures; the committed log checksum matches.
- PASS: artifact-free cloned checkout suite is green.
- PASS: all 14 registered analysis artifacts are byte-identical across the required three processes.
- PASS: branch is `arc4-wpA2`, task commits are separate, verifier has one untouched commit, and the final worktree is clean.

Final verifier output: `GOAL REACHED — all checks pass (1 flagged with evidence).`
