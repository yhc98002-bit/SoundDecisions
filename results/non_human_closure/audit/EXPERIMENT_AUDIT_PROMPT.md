# Independent experiment-integrity audit request

Audit the SoundDecisions non-human closure using only the paths below. Read the
artifacts and code yourself; do not rely on an external narrative summary.

Repository root:
`/XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions-non-human-closure`

Immutable artifact root:
`/XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions-non-human-artifacts/nhc_20260717T151345p0800_a052920`

Protocol and implementation:

- `experiment/non_human_closure/PROTOCOL.json`
- `experiment/non_human_closure/PROTOCOL.md`
- `experiment/non_human_closure/CLASS_READOUT_IMPLEMENTATION.json`
- `experiment/non_human_closure/material_reference_manifest.schema.json`

Code and tests:

- `foley_cw/b2_class_closure.py`
- `foley_cw/b1_lineage.py`
- `foley_cw/b2_feature_recollection.py`
- `foley_cw/class_internal_readout.py`
- `foley_cw/material_reference_feasibility.py`
- `scripts/b2_class_closure.py`
- `scripts/b1_lineage_pilot.py`
- `scripts/b2_feature_recollection.py`
- `scripts/class_internal_readout.py`
- `scripts/material_reference_feasibility.py`
- `scripts/class_video_determined_sensitivity.py`
- `scripts/materialize_non_human_closure.py`
- `scripts/validate_non_human_closure_bundle.py`
- `tests/test_b1_lineage.py`
- `tests/test_b2_class_closure.py`
- `tests/test_b2_feature_recollection.py`
- `tests/test_b2_inventory_merge.py`
- `tests/test_class_internal_readout.py`
- `tests/test_class_video_determined_sensitivity.py`
- `tests/test_material_reference_feasibility.py`
- `tests/test_materialize_non_human_closure.py`
- `tests/test_validate_non_human_closure_bundle.py`

Committed result bundle:

- `results/non_human_closure/NON_HUMAN_TRACK_REPORT.md`
- `results/non_human_closure/EXECUTION_STATUS.json`
- `results/non_human_closure/CLASS_POSTERIOR_MEASUREMENT_REPORT.json`
- `results/non_human_closure/CLASS_MULTISEED_COMMITMENT.json`
- `results/non_human_closure/CLASS_MULTISEED_COMMITMENT.csv`
- `results/non_human_closure/CLASS_VARIANCE_DECOMPOSITION.json`
- `results/non_human_closure/CLASS_VIDEO_DETERMINED_SENSITIVITY.json`
- `results/non_human_closure/FEATURE_LINEAGE_REPORT.json`
- `results/non_human_closure/feature_manifests/FEATURE_RECOLLECTION_COMPLETION.json`
- `results/non_human_closure/feature_manifests/FEATURE_RECOLLECTION_MANIFEST.jsonl`
- `results/non_human_closure/CLASS_INTERNAL_READOUT_REPORT.json`
- `results/non_human_closure/CLASS_INTERNAL_READOUT_OUTER_PREDICTIONS.jsonl.gz`
- `results/non_human_closure/MATERIAL_CONTINUITY_2AFC_REPORT.json`
- `results/non_human_closure/MATERIAL_REFERENCE_INSUFFICIENCY.json`
- `results/non_human_closure/MATERIAL_REFERENCE_INSUFFICIENCY_SUMMARY.json`
- `results/non_human_closure/NUMBERS_INDEX.json`
- `results/non_human_closure/REPRO.json`
- `results/non_human_closure/COMMANDS.md`
- `results/non_human_closure/BUGS_DEVIATIONS_UNRESOLVED.md`
- `results/non_human_closure/PROGRESS_LOG.md`
- `orbit-research/RUN_LEDGER.jsonl`

Canonical immutable completions:

- `class/merged_v2/CLASS_POSTERIORS_MERGED.completion.json`
- `class/analysis_v1/CLASS_MULTISEED_ANALYSIS.completion.json`
- `b1/calibration/calibration_v1/TOLERANCE.json`
- `b1/heldout/heldout_v1/HELDOUT_REPORT.json`
- `b1/heldout/heldout_v1/COMPLETED.json`
- `b1_full/merged_v2/FEATURE_RECOLLECTION_COMPLETION.json`
- `class_readout/targets_v1/TARGETS_COMPLETION.json`
- `class_readout/merged_v1/CLASS_INTERNAL_READOUT_COMPLETION.json`
- `class_readout/merged_v2/CLASS_INTERNAL_READOUT_COMPLETION.json`
- `material/feasibility/MATERIAL_REFERENCE_INSUFFICIENCY.json`

Perform these checks:

1. Fake or circular ground truth, including embeddings or model outputs used as
   semantic truth without the reported continuity limitation.
2. Score and target integrity, abstention/missingness handling, denominators,
   right-censoring, conditional versus unconditional intervals, and any
   comparison of non-equivalent estimands.
3. Split leakage across video groups, inner/outer selection leakage, duplicate
   candidate predictions, and tuning on held-out/B2 outcomes.
4. Same-forward identity validity, held-out isolation, forbidden quantization
   comparisons, and feature lineage from gate through probe views.
5. Variance-decomposition interpretation, bootstrap clustering, boundary
   estimates, and unidentified sources of uncertainty.
6. Material reference validity, coverage gate, outcome-blind matching, and any
   fabricated 2AFC metric.
7. Result-to-claim alignment for every headline conclusion and scientific
   status in the integrated report.
8. Reproducibility, asset/protocol hashes, immutable shard/reducer integrity,
   silent downloads, failed-attempt preservation, and missing provenance.
9. Scope compliance: no human-blocked axis PASS, sealed cohort, B6, causal
   intervention, map scheduling, second backbone, or paper claim.
10. Any required correction that would change a result, status, conclusion, or
    PI decision.

Return a single JSON object with keys:

- `verdict`: `PASS`, `WARN`, or `FAIL`;
- `blocking_findings`: array;
- `nonblocking_findings`: array;
- `verified_claims`: array;
- `required_corrections`: array;
- `residual_risks`: array;
- `files_examined`: array.

Each finding must name the supporting path and exact field, row, function, or
line whenever possible. Do not propose new out-of-scope experiments as a way to
repair an existing claim.
