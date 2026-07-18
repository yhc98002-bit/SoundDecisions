#!/usr/bin/env python3
"""Materialize the canonical non-human closure evidence bundle.

Large posterior arrays and feature tensors remain in the immutable artifact
root.  This command copies only reports, manifests, checksums, detailed CSVs,
and compressed candidate-level predictions into Git.  All writes are
create-only and every source binding is checked before the first output write.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PROTOCOL_SHA256 = "5c4fc4025995c16e355feb8cc02fbb3627891d47f6df052becde4845eaa7bd09"
EXPECTED_POSTERIORS = 79_152
EXPECTED_FEATURE_UNITS = 6_528
EXPECTED_TRAJECTORIES = 816
EXPECTED_VIDEOS = 48
EXPECTED_READOUT_PREDICTIONS = 113_212
PROGRESSES = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
EXPECTED_MATERIALIZED_OUTPUTS = frozenset({
    "CLASS_BASE_SEED_CROSSINGS.csv",
    "CLASS_INTERNAL_READOUT_OUTER_PREDICTIONS.jsonl.gz",
    "CLASS_INTERNAL_READOUT_REPORT.json",
    "CLASS_INTERNAL_READOUT_REPORT.md",
    "CLASS_MULTISEED_COMMITMENT.csv",
    "CLASS_MULTISEED_COMMITMENT.json",
    "CLASS_MULTISEED_COMMITMENT.md",
    "CLASS_POOLED_CROSSINGS.csv",
    "CLASS_POSTERIOR_MEASUREMENT_REPORT.json",
    "CLASS_POSTERIOR_MEASUREMENT_REPORT.md",
    "CLASS_VARIANCE_DECOMPOSITION.json",
    "CLASS_VARIANCE_DECOMPOSITION.md",
    "CLASS_VIDEO_BASELINES.csv",
    "CLASS_VIDEO_CROSSING_DISTRIBUTIONS.csv",
    "CLASS_VIDEO_SEED_CROSSINGS.csv",
    "FEATURE_LINEAGE_REPORT.json",
    "FEATURE_LINEAGE_REPORT.md",
    "MATERIAL_CONTINUITY_2AFC_REPORT.json",
    "MATERIAL_CONTINUITY_2AFC_REPORT.md",
    "MATERIAL_REFERENCE_INSUFFICIENCY.json",
    "MATERIAL_REFERENCE_INSUFFICIENCY_SUMMARY.json",
    "MATERIAL_SOURCE_AUDIO_LOUDNESS.json",
    "feature_manifests/B1_HELDOUT_COMPLETION.json",
    "feature_manifests/B1_HELDOUT_REPORT.json",
    "feature_manifests/B1_TOLERANCE.json",
    "feature_manifests/FEATURE_CHECKSUMS.sha256",
    "feature_manifests/FEATURE_MANIFEST_INDEX.json",
    "feature_manifests/FEATURE_RECOLLECTION_COMPLETION.json",
    "feature_manifests/FEATURE_RECOLLECTION_MANIFEST.jsonl",
    *(f"feature_manifests/FEATURE_SHARD_{index:02d}_COMPLETION.json" for index in range(8)),
})


class MaterializationError(RuntimeError):
    """Raised before or during fail-closed evidence materialization."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MaterializationError(f"expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterializationError(message)


def require_canonical_output_paths(paths: Iterable[str]) -> None:
    """Require the exact, duplicate-free 37-file materialization contract."""
    rows = list(paths)
    require(len(rows) == len(set(rows)), "duplicate materialized output path")
    require(
        set(rows) == EXPECTED_MATERIALIZED_OUTPUTS,
        "materialized output path set differs from the canonical 37-file contract",
    )


def write_json_create(path: Path, value: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return path


def write_text_create(path: Path, value: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value.rstrip() + "\n")
    return path


def copy_create(source: Path, destination: Path) -> Path:
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    require(
        sha256_file(source) == sha256_file(destination),
        f"copy hash mismatch: {source} -> {destination}",
    )
    return destination


def _counts(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _numeric_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def posterior_report(artifact_root: Path) -> tuple[dict[str, Any], str]:
    root = artifact_root / "class" / "merged_v2"
    completion_path = root / "CLASS_POSTERIORS_MERGED.completion.json"
    completion = load_json(completion_path)
    require(completion.get("status") == "COMPLETE", "Class posterior merge is incomplete")
    require(completion.get("canonical_b2") is True, "Class posterior merge is noncanonical")
    require(
        int(completion.get("record_count", -1)) == EXPECTED_POSTERIORS,
        "Class posterior cardinality mismatch",
    )
    require(
        completion.get("protocol_sha256") == PROTOCOL_SHA256,
        "Class posterior protocol mismatch",
    )
    data_path = root / str(completion.get("data_file", ""))
    require(sha256_file(data_path) == completion.get("data_sha256"), "posterior data hash mismatch")
    require(data_path.stat().st_size == int(completion.get("data_bytes", -1)), "posterior data size mismatch")
    with np.load(data_path, allow_pickle=False) as archive:
        required = {
            "clipwise_output_527", "coarse_posterior", "coarse_score_sums",
            "top_class", "confidence", "margin", "entropy", "abstain",
            "confident_label", "video_id", "base_seed", "fork_index",
            "progress", "audio_sha256", "role",
        }
        require(required.issubset(archive.files), "posterior archive missing required arrays")
        require(archive["clipwise_output_527"].shape == (EXPECTED_POSTERIORS, 527), "527-way posterior shape mismatch")
        require(archive["coarse_posterior"].shape == (EXPECTED_POSTERIORS, 15), "coarse posterior shape mismatch")
        require(
            np.allclose(archive["coarse_posterior"].sum(axis=1), 1.0, atol=1e-6, rtol=0.0),
            "coarse posterior normalization mismatch",
        )
        roles = np.asarray(archive["role"]).astype(str)
        abstain = np.asarray(archive["abstain"], dtype=bool)
        top = np.asarray(archive["top_class"]).astype(str)
        confident = np.asarray(archive["confident_label"]).astype(str)
        require(np.array_equal(top[~abstain], confident[~abstain]), "confident-label lineage mismatch")
        fork_mask = roles == "fork"
        base_mask = roles == "base"
        require(int(base_mask.sum()) == 816 and int(fork_mask.sum()) == 78_336, "posterior role cardinality mismatch")
        by_progress: list[dict[str, Any]] = []
        progress_array = np.asarray(archive["progress"], dtype=np.float64)
        for progress in PROGRESSES:
            mask = fork_mask & np.isclose(progress_array, progress, atol=1e-6, rtol=0.0)
            require(int(mask.sum()) == 9_792, f"fork posterior count mismatch at {progress}")
            by_progress.append(
                {
                    "progress": progress,
                    "records": int(mask.sum()),
                    "abstention_rate": float(abstain[mask].mean()),
                    "mean_confidence": float(np.asarray(archive["confidence"])[mask].mean()),
                    "mean_margin": float(np.asarray(archive["margin"])[mask].mean()),
                    "mean_entropy": float(np.asarray(archive["entropy"])[mask].mean()),
                }
            )
        report: dict[str, Any] = {
            "schema": "sounddecisions.class_posterior_measurement_report.v1",
            "artifact_status": "COMPLETE",
            "scientific_status": "NOT_TESTED",
            "scope": "measurement artifact for exploratory Class continuity",
            "canonical_b2": True,
            "record_count": EXPECTED_POSTERIORS,
            "role_counts": {"base": int(base_mask.sum()), "fork": int(fork_mask.sum())},
            "design": {
                "videos": len(set(np.asarray(archive["video_id"]).astype(str))),
                "base_seeds": len(set(int(value) for value in np.asarray(archive["base_seed"]))),
                "progress_points": list(PROGRESSES),
                "forks_per_state": 12,
            },
            "abstention": {
                "overall_rate": float(abstain.mean()),
                "base_rate": float(abstain[base_mask].mean()),
                "fork_rate": float(abstain[fork_mask].mean()),
                "by_progress": by_progress,
            },
            "distributions": {
                "top_class_counts": _counts(top),
                "confident_class_counts": _counts(confident[~abstain]),
                "confidence": _numeric_summary(archive["confidence"]),
                "margin": _numeric_summary(archive["margin"]),
                "entropy": _numeric_summary(archive["entropy"]),
            },
            "persisted_arrays": completion["array_schema"],
            "coarse_map_revision": completion["coarse_map_revision"],
            "coarse_map_sha256": completion["coarse_map_sha256"],
            "coarse_posterior_rule_id": completion["coarse_posterior_rule_id"],
            "abstention_rule_id": completion["abstention_rule_id"],
            "abstain_delta": completion["abstain_delta"],
            "tagger_revision": completion["tagger_revision"],
            "tagger_checkpoint_sha256": completion["tagger_checkpoint_sha256"],
            "measurer_revision": completion["measurer_revision"],
            "protocol_sha256": completion["protocol_sha256"],
            "inventory_manifest_sha256": completion["inventory_manifest_sha256"],
            "inventory_records_sha256": completion["inventory_records_sha256"],
            "merged_completion": str(completion_path.resolve()),
            "merged_completion_sha256": sha256_file(completion_path),
            "posterior_data": str(data_path.resolve()),
            "posterior_data_sha256": sha256_file(data_path),
            "posterior_data_bytes": data_path.stat().st_size,
            "input_shards": completion["input_shards"],
            "network_downloads": "none; pinned local checkpoint",
        }
    lines = [
        "# Class posterior measurement",
        "",
        "Artifact status: `COMPLETE`. Scientific status: `NOT_TESTED` (this file validates measurement, not a Class claim).",
        "",
        "The canonical B2 bank contains 79,152 measured WAVs: 816 base finals and 78,336 fork finals. The persisted archive includes the full 527-way output, normalized 15-way coarse posterior, coarse sums, top/confident label, confidence, margin, entropy, abstention, all IDs, hashes, and model/measurer revisions.",
        "",
        f"- Overall abstention: {report['abstention']['overall_rate']:.4f}",
        f"- Base-final abstention: {report['abstention']['base_rate']:.4f}",
        f"- Fork-final abstention: {report['abstention']['fork_rate']:.4f}",
        f"- Coarse map: `{report['coarse_map_revision']}` / `{report['coarse_map_sha256']}`",
        f"- Posterior data SHA-256: `{report['posterior_data_sha256']}`",
        "",
        "| s | records | abstention | mean confidence | mean margin | mean entropy |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_progress:
        lines.append(
            f"| {row['progress']:.2f} | {row['records']} | {row['abstention_rate']:.4f} | "
            f"{row['mean_confidence']:.4f} | {row['mean_margin']:.4f} | {row['mean_entropy']:.4f} |"
        )
    lines.extend(["", "No abstention threshold or coarse taxonomy was tuned on B2 outcomes."])
    return report, "\n".join(lines)


def commitment_markdown(commitment: Mapping[str, Any]) -> str:
    primary = commitment["theta_0.70_summary"]
    replication = commitment["replication_classification"]
    lines = [
        "# Class multi-seed commitment",
        "",
        "Scientific status: `NOT_SUPPORTED`. This is an exploratory multi-seed continuity replication, not event-centered v2 confirmation.",
        "",
        f"At the registered all-cell pooled sustained threshold theta=0.70, the point crossing is `s={replication['pooled_sustained_crossing_theta_0.70']:.2f}`. In the 5,000-draw video bootstrap, 3,767 draws cross and 1,233 are noncrossing; among crossing draws, the conditional percentile range is `[0.75, 0.90]`. This range is not an unconditional confidence interval. The frozen classification is `not_reproduced`.",
        "",
        f"The historical estimate `s={replication['historical_s_commit']:.3f}` is a crossers-only mean of unsustained individual first crossings, whereas the frozen B2 decision is an all-cell pooled sustained crossing. They are different estimands, so the conditional pooled range is not a compatibility interval for the historical value. The individual B2 evidence must be read separately below.",
        "",
        f"Individual video-seed units are heterogeneous: {primary['n_crossing']}/{primary['n_scorable_nondetermined_video_seed_units']} scorable nondetermined units cross, {primary['n_noncrossing']} do not, and {primary['n_unscorable']} remain unscorable. Among crossers only, the mean first crossing is {primary['mean_first_crossing_crossers']:.3f} and median is {primary['median_first_crossing_crossers']:.2f}; noncrossers are right-censored and never imputed.",
        "",
        "## Pooled curve",
        "",
        "| s | mean commitment gain (95% video CI) | confident fork agreement | fork abstention | scorable cells |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in commitment["curves_by_progress"]:
        lines.append(
            f"| {row['progress']:.2f} | {row['mean_commitment_gain']:.3f} "
            f"[{row['commitment_gain_ci_low']:.3f}, {row['commitment_gain_ci_high']:.3f}] | "
            f"{row['mean_a_fork_confident']:.3f} | {row['mean_fork_abstention_rate']:.3f} | "
            f"{row['n_scorable_cells']}/{row['n_video_seed_cells']} |"
        )
    lines.extend(
        [
            "",
            "## Registered threshold sensitivity",
            "",
            "| theta | point estimate | bootstrap crossing / noncrossing draws | conditional crossing percentile range |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in commitment["pooled_crossings"]:
        crossing = "noncrossing" if row["sustained_crossing"] is None else f"{row['sustained_crossing']:.2f}"
        conditional = (
            "none"
            if int(row["bootstrap_sustained_crossing_draws"]) == 0
            else f"[{row['sustained_crossing_bootstrap_ci_low']:.2f}, {row['sustained_crossing_bootstrap_ci_high']:.2f}]"
        )
        draws = (
            f"{row['bootstrap_sustained_crossing_draws']} / "
            f"{row['bootstrap_noncrossing_draws']}"
        )
        lines.append(
            f"| {row['theta_commit']:.2f} | {crossing} | {draws} | {conditional} |"
        )
    lines.extend(
        [
            "",
            "Nine of 48 videos are video-determined under the registered A_ind >= 0.90 rule. The registered pooled curve includes these cases, whose A_ind=1 commitment gain is fixed to zero; `CLASS_VIDEO_DETERMINED_SENSITIVITY.json` reports the separately labeled post-hoc exclusion sensitivity. Detailed per-video, per-seed, crossing, noncrossing, and baseline records are included as CSV files.",
        ]
    )
    return "\n".join(lines)


def variance_markdown(variance: Mapping[str, Any]) -> str:
    components = variance["overall_mean_components"]
    fractions = variance["overall_mean_component_fractions"]
    ci = variance["overall_mean_component_video_bootstrap_ci"]
    labels = (
        ("video", "Video"),
        ("base_seed", "Additive base seed"),
        ("video_by_seed_interaction", "Video x seed interaction"),
        ("fork_monte_carlo_nonabstention", "Fork Monte Carlo (non-abstention)"),
        ("abstention_within_fork_monte_carlo", "Identifiable abstention subcomponent"),
    )
    lines = [
        "# Class variance decomposition",
        "",
        "Scientific status: `SUPPORTED_EXPLORATORILY` for the decomposition only. The historical commitment-window replication remains `NOT_SUPPORTED`.",
        "",
        "Progress is treated as a fixed stratum. Components use an unbalanced crossed video/base-seed method of moments; uncertainty is a 5,000-draw video-cluster bootstrap retaining all seeds within sampled videos.",
        "",
        "| component | variance | 95% video-bootstrap CI | fraction |",
        "|---|---:|---:|---:|",
    ]
    for key, label in labels:
        lines.append(
            f"| {label} | {components[key]:.6f} | [{ci[key]['ci_low']:.6f}, {ci[key]['ci_high']:.6f}] | {fractions[key]:.2%} |"
        )
    lines.extend(
        [
            "",
            "The additive seed estimate is boundary-clipped to zero; its raw method-of-moments estimate is negative at every progress stratum. Video-by-seed interaction is large, so this is not evidence that seed choice is irrelevant for a given video. The 17-final video baselines are held fixed and their finite-reference uncertainty is not separately decomposed. Measurer repeatability is `UNRESOLVED` because each WAV was measured once. The abstention component is a subcomponent of fork Monte Carlo variance, not an independent term.",
        ]
    )
    return "\n".join(lines)


def lineage_report(artifact_root: Path) -> tuple[dict[str, Any], str, dict[str, Path]]:
    calibration_root = artifact_root / "b1" / "calibration" / "calibration_v1"
    heldout_root = artifact_root / "b1" / "heldout" / "heldout_v1"
    feature_root = artifact_root / "b1_full" / "merged_v2"
    tolerance_path = calibration_root / "TOLERANCE.json"
    calibration_completion_path = calibration_root / "COMPLETED.json"
    heldout_report_path = heldout_root / "HELDOUT_REPORT.json"
    heldout_completion_path = heldout_root / "COMPLETED.json"
    feature_completion_path = feature_root / "FEATURE_RECOLLECTION_COMPLETION.json"
    tolerance = load_json(tolerance_path)
    heldout = load_json(heldout_report_path)
    feature = load_json(feature_completion_path)
    require(tolerance.get("status") == "FROZEN", "B-1 tolerance is not frozen")
    require(tolerance.get("heldout_rejected") is True, "held-out clip contaminated calibration")
    require(tolerance.get("heldout_clip") == "1002", "unexpected B-1 held-out clip")
    require(heldout.get("status") == "PASS" and int(heldout.get("failure_count", -1)) == 0, "B-1 held-out gate failed")
    require(heldout.get("tolerance_unchanged") is True, "B-1 held-out tolerance changed")
    require(int(heldout.get("evaluated_metrics", -1)) == 1960, "B-1 metric cardinality mismatch")
    require(feature.get("status") == "COMPLETE" and feature.get("canonical_b2") is True, "feature recollection incomplete")
    require(int(feature.get("unit_count", -1)) == EXPECTED_FEATURE_UNITS, "feature unit cardinality mismatch")
    require(int(feature.get("base_trajectory_count", -1)) == EXPECTED_TRAJECTORIES, "feature trajectory cardinality mismatch")
    require(int(feature.get("video_count", -1)) == EXPECTED_VIDEOS, "feature video cardinality mismatch")
    require(feature.get("protocol_sha256") == PROTOCOL_SHA256, "feature protocol mismatch")
    report = {
        "schema": "sounddecisions.feature_lineage_report.v1",
        "artifact_status": "COMPLETE",
        "scientific_status": "NOT_TESTED",
        "scope": "engineering identity gate and lineage-valid B2 feature recollection",
        "same_forward_gate": {
            "status": "PASS",
            "calibration_clips": tolerance["calibration_clips"],
            "heldout_clip": heldout["heldout_clip"],
            "progress_points": heldout["progress_points"],
            "calibration_replay_units": tolerance["source_replay_units"],
            "heldout_replay_units": heldout["source_replay_units"],
            "evaluated_heldout_metrics": heldout["evaluated_metrics"],
            "failure_count": heldout["failure_count"],
            "tolerance_unchanged": heldout["tolerance_unchanged"],
            "forbidden_comparison": tolerance["forbidden_comparison"],
            "tolerance_sha256": sha256_file(tolerance_path),
            "calibration_completion_sha256": sha256_file(calibration_completion_path),
            "heldout_report_sha256": sha256_file(heldout_report_path),
            "heldout_completion_sha256": sha256_file(heldout_completion_path),
        },
        "feature_recollection": {
            "status": "COMPLETE",
            "primary_evidence_uses_legacy_25600_bundles": False,
            "base_trajectories": feature["base_trajectory_count"],
            "state_units": feature["unit_count"],
            "videos": feature["video_count"],
            "progress_points": feature["progress_points"],
            "shard_count": len(feature["input_shards"]),
            "feature_collector_sha256": feature["feature_collector_sha256"],
            "collector_project_git_commit": feature["project_git_commit"],
            "manifest_sha256": feature["manifest_sha256"],
            "completion_sha256": sha256_file(feature_completion_path),
            "lineage_gate": feature["lineage_gate"],
        },
        "protocol_sha256": PROTOCOL_SHA256,
        "large_tensors_location": str((artifact_root / "b1_full" / "feature_shards").resolve()),
        "large_tensors_in_git": False,
        "engineering_failures_preserved": [
            "two interrupted calibration attempts after source guard detection",
            "two held-out launches stopped before model load when HF_HUB_DISABLE_XET was omitted",
            "one serial feature merge terminated before output creation and replaced by deterministic parallel validation",
        ],
    }
    lines = [
        "# Feature lineage",
        "",
        "The B-1 same-forward identity gate **passed**. This is an engineering lineage result (`scientific_status: NOT_TESTED`), not a semantic Class finding.",
        "",
        f"Four deterministic calibration clips defined the tolerance before held-out access. Two fresh replays of clip `1002` then passed all {heldout['evaluated_metrics']} eligible comparisons with zero failures and no tolerance change. The forbidden non-equivalent reduction/quantization comparison was never used.",
        "",
        f"After the gate, 816 B2 base trajectories were recollected at all eight progress points into {feature['unit_count']} lineage-valid state units across {len(feature['input_shards'])} immutable shards. The old 25,600 bundles are not primary evidence.",
        "",
        f"- Tolerance SHA-256: `{sha256_file(tolerance_path)}`",
        f"- Held-out report SHA-256: `{sha256_file(heldout_report_path)}`",
        f"- Feature completion SHA-256: `{sha256_file(feature_completion_path)}`",
        f"- Feature manifest SHA-256: `{feature['manifest_sha256']}`",
        "",
        "Large tensors remain outside Git. Complete state and shard manifests, hashes, hook metadata, dtypes, shapes, devices, and environment provenance are retained in the immutable artifact root.",
    ]
    sources = {
        "tolerance": tolerance_path,
        "heldout_report": heldout_report_path,
        "heldout_completion": heldout_completion_path,
        "feature_completion": feature_completion_path,
        "feature_manifest": feature_root / str(feature["manifest"]),
    }
    return report, "\n".join(lines), sources


def validate_readout(artifact_root: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path, Path]:
    root = artifact_root / "class_readout" / "merged_v2"
    report_path = root / "CLASS_INTERNAL_READOUT_REPORT.json"
    markdown_path = root / "CLASS_INTERNAL_READOUT_REPORT.md"
    completion_path = root / "CLASS_INTERNAL_READOUT_COMPLETION.json"
    report = load_json(report_path)
    completion = load_json(completion_path)
    require(report.get("status") == "COMPLETE", "Class readout report incomplete")
    require(report.get("targets_separate") is True, "Class readout targets were merged")
    require(int(report.get("prediction_count", -1)) == EXPECTED_READOUT_PREDICTIONS, "readout prediction cardinality mismatch")
    require(report.get("protocol_sha256") == PROTOCOL_SHA256, "readout protocol mismatch")
    require(
        all(report["decisions"][target]["scientific_status"] == "NOT_SUPPORTED" for target in ("fork_majority", "ode_final")),
        "unexpected readout conclusion status",
    )
    predictions_path = root / str(report.get("predictions_file", ""))
    require(sha256_file(predictions_path) == report.get("predictions_sha256"), "readout prediction hash mismatch")
    require(sha256_file(report_path) == completion.get("report_sha256"), "readout report completion mismatch")
    require(sha256_file(markdown_path) == completion.get("markdown_sha256"), "readout markdown completion mismatch")
    return report, completion, report_path, markdown_path, predictions_path


def materialize(
    artifact_root: Path, result_dir: Path, support_dir: Path
) -> list[Path]:
    artifact_root = Path(artifact_root).resolve()
    result_dir = Path(result_dir).resolve()
    support_dir = Path(support_dir).resolve()
    require(artifact_root.is_dir(), f"artifact root missing: {artifact_root}")
    require(result_dir.is_dir(), f"result directory missing: {result_dir}")
    require(support_dir.is_dir(), f"support directory missing: {support_dir}")

    posterior, posterior_md = posterior_report(artifact_root)
    analysis_root = artifact_root / "class" / "analysis_v1"
    commitment_source = analysis_root / "CLASS_MULTISEED_COMMITMENT.json"
    variance_source = analysis_root / "CLASS_VARIANCE_DECOMPOSITION.json"
    commitment = load_json(commitment_source)
    variance = load_json(variance_source)
    analysis_completion = load_json(analysis_root / "CLASS_MULTISEED_ANALYSIS.completion.json")
    require(analysis_completion.get("status") == "COMPLETE", "Class analysis incomplete")
    require(analysis_completion.get("scientific_status") == "NOT_SUPPORTED", "unexpected Class analysis status")
    require(commitment.get("replication_label") == "not_reproduced", "unexpected Class replication label")
    require(commitment.get("scientific_status") == "NOT_SUPPORTED", "unexpected Class commitment status")

    lineage, lineage_md, lineage_sources = lineage_report(artifact_root)
    readout, _, readout_source, readout_md_source, prediction_source = validate_readout(artifact_root)

    material_source = artifact_root / "material" / "feasibility" / "MATERIAL_REFERENCE_INSUFFICIENCY.json"
    loudness_source = artifact_root / "material" / "feasibility" / "SOURCE_AUDIO_LOUDNESS.json"
    material_summary = load_json(support_dir / "MATERIAL_REFERENCE_INSUFFICIENCY_SUMMARY.json")
    legacy_inventory = material_summary.get("legacy_inventory", {})
    require(
        legacy_inventory.get("legacy_journal_videos") == 200
        and legacy_inventory.get("candidate_indices") == [0, 1, 2, 3]
        and legacy_inventory.get("legacy_cells_inventoried") == 6_400
        and legacy_inventory.get("surviving_subject_final_embeddings") == 800,
        "Material legacy inventory binding mismatch",
    )
    require(
        legacy_inventory.get("measurements_sha256")
        == material_summary.get("canonical_evidence", {}).get("measurements_sha256"),
        "Material legacy inventory measurement hash mismatch",
    )
    coverage_cross_check = legacy_inventory.get("coverage_cross_check", {})
    require(
        coverage_cross_check
        == {
            "subjects_with_strict_match": 52,
            "excluded_subjects": 748,
            "candidate_subjects_total": 800,
        },
        "Material legacy inventory coverage cross-check mismatch",
    )
    require(
        sha256_file(material_source) == material_summary["canonical_evidence"]["sha256"],
        "Material insufficiency hash mismatch",
    )
    require(
        sha256_file(loudness_source) == material_summary["canonical_evidence"]["source_audio_loudness_sha256"],
        "Material loudness hash mismatch",
    )

    outputs: list[Path] = []
    for name in (
        "MATERIAL_CONTINUITY_2AFC_REPORT.json",
        "MATERIAL_CONTINUITY_2AFC_REPORT.md",
        "MATERIAL_REFERENCE_INSUFFICIENCY_SUMMARY.json",
    ):
        source = support_dir / name
        destination = result_dir / name
        require(source.is_file(), f"supporting Material report missing: {source}")
        require(not source.is_symlink(), f"supporting Material report may not be a symlink: {source}")
        if source.resolve() == destination.resolve():
            outputs.append(destination)
            continue
        outputs.append(copy_create(source, destination))
    outputs.append(write_json_create(result_dir / "CLASS_POSTERIOR_MEASUREMENT_REPORT.json", posterior))
    outputs.append(write_text_create(result_dir / "CLASS_POSTERIOR_MEASUREMENT_REPORT.md", posterior_md))
    outputs.append(copy_create(commitment_source, result_dir / "CLASS_MULTISEED_COMMITMENT.json"))
    outputs.append(copy_create(analysis_root / "CLASS_MULTISEED_COMMITMENT.csv", result_dir / "CLASS_MULTISEED_COMMITMENT.csv"))
    outputs.append(write_text_create(result_dir / "CLASS_MULTISEED_COMMITMENT.md", commitment_markdown(commitment)))
    outputs.append(copy_create(variance_source, result_dir / "CLASS_VARIANCE_DECOMPOSITION.json"))
    outputs.append(write_text_create(result_dir / "CLASS_VARIANCE_DECOMPOSITION.md", variance_markdown(variance)))
    for name in (
        "CLASS_VIDEO_CROSSING_DISTRIBUTIONS.csv",
        "CLASS_VIDEO_SEED_CROSSINGS.csv",
        "CLASS_POOLED_CROSSINGS.csv",
        "CLASS_BASE_SEED_CROSSINGS.csv",
        "CLASS_VIDEO_BASELINES.csv",
    ):
        outputs.append(copy_create(analysis_root / name, result_dir / name))
    outputs.append(write_json_create(result_dir / "FEATURE_LINEAGE_REPORT.json", lineage))
    outputs.append(write_text_create(result_dir / "FEATURE_LINEAGE_REPORT.md", lineage_md))
    outputs.append(copy_create(readout_source, result_dir / "CLASS_INTERNAL_READOUT_REPORT.json"))
    outputs.append(copy_create(readout_md_source, result_dir / "CLASS_INTERNAL_READOUT_REPORT.md"))
    outputs.append(copy_create(prediction_source, result_dir / "CLASS_INTERNAL_READOUT_OUTER_PREDICTIONS.jsonl.gz"))
    outputs.append(copy_create(material_source, result_dir / "MATERIAL_REFERENCE_INSUFFICIENCY.json"))
    outputs.append(copy_create(loudness_source, result_dir / "MATERIAL_SOURCE_AUDIO_LOUDNESS.json"))

    feature_dir = result_dir / "feature_manifests"
    outputs.append(copy_create(lineage_sources["tolerance"], feature_dir / "B1_TOLERANCE.json"))
    outputs.append(copy_create(lineage_sources["heldout_report"], feature_dir / "B1_HELDOUT_REPORT.json"))
    outputs.append(copy_create(lineage_sources["heldout_completion"], feature_dir / "B1_HELDOUT_COMPLETION.json"))
    outputs.append(copy_create(lineage_sources["feature_completion"], feature_dir / "FEATURE_RECOLLECTION_COMPLETION.json"))
    outputs.append(copy_create(lineage_sources["feature_manifest"], feature_dir / "FEATURE_RECOLLECTION_MANIFEST.jsonl"))
    feature_completion = load_json(lineage_sources["feature_completion"])
    shard_index: list[dict[str, Any]] = []
    for item in sorted(feature_completion["input_shards"], key=lambda row: int(row["shard_index"])):
        index = int(item["shard_index"])
        source = Path(item["completion"])
        require(sha256_file(source) == item["completion_sha256"], f"feature shard {index} hash mismatch")
        destination = feature_dir / f"FEATURE_SHARD_{index:02d}_COMPLETION.json"
        outputs.append(copy_create(source, destination))
        shard_index.append(
            {
                "shard_index": index,
                "unit_count": int(item["unit_count"]),
                "source": str(source.resolve()),
                "source_sha256": sha256_file(source),
                "git_manifest": str(destination.relative_to(result_dir)),
                "git_manifest_sha256": sha256_file(destination),
            }
        )
    feature_index = {
        "schema": "sounddecisions.feature_manifest_index.v1",
        "artifact_status": "COMPLETE",
        "large_tensors_in_git": False,
        "large_tensor_root": str((artifact_root / "b1_full" / "feature_shards").resolve()),
        "unit_count": EXPECTED_FEATURE_UNITS,
        "trajectory_count": EXPECTED_TRAJECTORIES,
        "video_count": EXPECTED_VIDEOS,
        "progress_points": list(PROGRESSES),
        "recollection_completion_sha256": sha256_file(lineage_sources["feature_completion"]),
        "recollection_manifest_sha256": sha256_file(lineage_sources["feature_manifest"]),
        "shards": shard_index,
    }
    outputs.append(write_json_create(feature_dir / "FEATURE_MANIFEST_INDEX.json", feature_index))
    checksum_targets = sorted(path for path in outputs if feature_dir in path.parents)
    checksum_text = "\n".join(
        f"{sha256_file(path)}  {path.relative_to(feature_dir)}" for path in checksum_targets
    )
    outputs.append(write_text_create(feature_dir / "FEATURE_CHECKSUMS.sha256", checksum_text))

    relative_outputs = [str(path.relative_to(result_dir)) for path in outputs]
    require_canonical_output_paths(relative_outputs)

    materialization = {
        "schema": "sounddecisions.non_human_materialization.v1",
        "artifact_status": "COMPLETE",
        "artifact_root": str(artifact_root),
        "result_dir": str(result_dir),
        "protocol_sha256": PROTOCOL_SHA256,
        "class_status": commitment["scientific_status"],
        "readout_status": readout["decisions"]["fork_majority"]["scientific_status"],
        "material_status": material_summary["scientific_status"],
        "outputs": [
            {
                "path": str(path.relative_to(result_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(outputs)
        ],
    }
    outputs.append(write_json_create(result_dir / "MATERIALIZATION_MANIFEST.json", materialization))
    return outputs


def validate_materialization(artifact_root: Path, result_dir: Path) -> dict[str, Any]:
    artifact_root, result_dir = Path(artifact_root).resolve(), Path(result_dir).resolve()
    manifest = load_json(result_dir / "MATERIALIZATION_MANIFEST.json")
    require(manifest.get("artifact_status") == "COMPLETE", "materialization manifest incomplete")
    require(manifest.get("artifact_root") == str(artifact_root), "materialization artifact-root mismatch")
    require(manifest.get("protocol_sha256") == PROTOCOL_SHA256, "materialization protocol mismatch")
    rows = manifest.get("outputs")
    require(isinstance(rows, list), "materialization outputs are not a list")
    observed_paths: list[str] = []
    for item in rows:
        require(isinstance(item, dict), "materialization output row is not an object")
        relative = Path(str(item["path"]))
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe materialized path")
        relative_text = relative.as_posix()
        observed_paths.append(relative_text)
        path = result_dir / relative
        require(path.is_file() and not path.is_symlink(), f"materialized file missing/unsafe: {path}")
        require(path.stat().st_size == int(item["bytes"]), f"materialized size mismatch: {path}")
        require(sha256_file(path) == item["sha256"], f"materialized hash mismatch: {path}")
    require_canonical_output_paths(observed_paths)
    posterior_report(artifact_root)
    validate_readout(artifact_root)
    lineage_report(artifact_root)
    return {
        "status": "PASS",
        "validated_outputs": len(rows),
        "manifest_sha256": sha256_file(result_dir / "MATERIALIZATION_MANIFEST.json"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("materialize", "validate"))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--support-dir",
        type=Path,
        help="directory containing the already-audited Material stop reports; defaults to result-dir",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "materialize":
        outputs = materialize(
            args.artifact_root,
            args.result_dir,
            args.result_dir if args.support_dir is None else args.support_dir,
        )
        payload = {"status": "COMPLETE", "created_outputs": len(outputs)}
    else:
        payload = validate_materialization(args.artifact_root, args.result_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
