"""Dataset manifest schema + synthetic dataset builder for foley-cw.

This module implements:
  - DatasetManifest: the FoleyBench subset specification (subset configs, axis
    sample counts, clip constraints, anchor provenance).
  - EventAnchor: visible-event timestamps with uncertainty and source tag.
  - build_manifest: construct a DatasetManifest from configs/dataset.json content.
  - build_synthetic_dataset: a CPU/CI-only dataset of n_videos synthetic entries,
    each with a SyntheticVideoCond and an EventAnchor, for use in dry-run tests.
  - load_foleybench: build a DatasetManifest from a real FoleyBench root directory;
    raises FileNotFoundError when the root is absent (NO download).
  - Markdown renderers: manifest_to_markdown, anchor_report_markdown.

All imports are stdlib + numpy only at module level, faithful to the hard rule.
FoleyBench reference: Dixit et al., 2025, arXiv:2511.13219.

Conventions (from refine-logs/EXPERIMENT_PLAN.md §2 / Phase 0.3-0.4):
  - Single-event subset is the primary analysis dataset; two-event subset is
    the optional Tier-3 binding subset.
  - Event-anchor priority: foleybench_metadata > visual_onset_detector >
    light_human_marks.
  - anchor_uncertainty_required = True: clips without timing uncertainty
    estimates are not used for the timing/binding axes.
  - Axes below min_usable_n_per_axis are reported as underpowered, not as
    results.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .synthetic_backend import SyntheticGaussianFlow, SyntheticVideoCond


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass
class EventAnchor:
    """Visible-event timestamps for a single clip, with uncertainty and provenance.

    timestamps:  list of event-onset times in seconds (one per sound event).
    uncertainty: per-timestamp uncertainty estimate (seconds, 1-sigma or half-range);
                 required for timing and binding axes (anchor_uncertainty_required).
    source:      provenance tag from the priority list:
                 "foleybench_metadata" | "visual_onset_detector" | "light_human_marks"
                 | "unknown".
    """
    timestamps: list[float]
    uncertainty: list[float]            # per-event, same length as timestamps
    source: str = "unknown"             # provenance

    def __post_init__(self) -> None:
        if len(self.timestamps) != len(self.uncertainty):
            raise ValueError(
                "EventAnchor: timestamps and uncertainty must have the same length; "
                f"got {len(self.timestamps)} vs {len(self.uncertainty)}"
            )

    @property
    def n_events(self) -> int:
        return len(self.timestamps)

    @property
    def is_single_event(self) -> bool:
        return self.n_events == 1

    @property
    def max_uncertainty(self) -> float:
        """Max per-event uncertainty in seconds; float('inf') if no events."""
        if not self.uncertainty:
            return float("inf")
        return float(np.max(self.uncertainty))


@dataclass
class SubsetSpec:
    """Specifies one subset (single-event or two-event) within the manifest."""
    name: str
    enabled: bool
    purpose: str
    min_n_per_axis: int                        # minimum usable n for this subset
    clip_duration_range_s: tuple[float, float] = (1.0, 10.0)


@dataclass
class DatasetManifest:
    """FoleyBench (or synthetic) subset specification.

    This is the output of build_manifest / load_foleybench; it records WHAT
    we intend to use and the per-axis sample-count constraints, but does NOT
    hold the actual clip objects (those live in the dataset loader).

    Fields
    ------
    source              : dataset name (e.g. "FoleyBench").
    foleybench_root     : path to the local FoleyBench root, or None (not yet
                          available / synthetic).
    subsets             : ordered list of SubsetSpec objects.
    clip_duration_range_s : (min_s, max_s) clip duration filter.
    class_balance_policy: description of class-balance strategy.
    event_anchor_priority: ordered list of anchor provenance sources.
    anchor_uncertainty_required : whether uncertainty estimate is mandatory.
    min_usable_n_per_axis: dict mapping axis_id -> minimum usable n; axes
                          below this are reported as underpowered.
    extra               : any additional config keys passed through verbatim.
    """
    source: str
    foleybench_root: Optional[str]
    subsets: list[SubsetSpec]
    clip_duration_range_s: tuple[float, float]
    class_balance_policy: str
    event_anchor_priority: list[str]
    anchor_uncertainty_required: bool
    min_usable_n_per_axis: dict[str, int]
    extra: dict[str, Any] = field(default_factory=dict)

    def usable_n(self, axis_id: str) -> int:
        """Return declared minimum usable n for axis_id, defaulting to 0 if absent."""
        return self.min_usable_n_per_axis.get(axis_id, 0)


# ---------------------------------------------------------------------------
# build_manifest: from configs/dataset.json content
# ---------------------------------------------------------------------------

def build_manifest(dataset_cfg: dict) -> DatasetManifest:
    """Construct a DatasetManifest from the parsed configs/dataset.json dict.

    Faithfully reads the fields defined in configs/dataset.json:
      source, foleybench_root, single_event_subset, two_event_subset,
      clip_duration_s, class_balance, event_anchor_priority,
      anchor_uncertainty_required, min_usable_n_per_axis.

    Any unrecognised top-level keys are captured in `extra` for forward
    compatibility.
    """
    known_keys = {
        "source", "foleybench_root", "single_event_subset", "two_event_subset",
        "clip_duration_s", "class_balance", "event_anchor_priority",
        "anchor_uncertainty_required", "min_usable_n_per_axis", "_doc",
    }
    extra = {k: v for k, v in dataset_cfg.items() if k not in known_keys}

    # Clip duration
    dur = dataset_cfg.get("clip_duration_s", [1.0, 10.0])
    clip_duration_range_s: tuple[float, float] = (float(dur[0]), float(dur[1]))

    # Subsets
    subsets: list[SubsetSpec] = []
    se = dataset_cfg.get("single_event_subset", {})
    if se:
        subsets.append(SubsetSpec(
            name="single_event",
            enabled=bool(se.get("enabled", True)),
            purpose="primary analysis (Tier 1-2 axes)",
            min_n_per_axis=int(se.get("min_n_per_axis", 40)),
            clip_duration_range_s=clip_duration_range_s,
        ))
    te = dataset_cfg.get("two_event_subset", {})
    if te:
        subsets.append(SubsetSpec(
            name="two_event",
            enabled=bool(te.get("enabled", False)),
            purpose=str(te.get("purpose", "binding (Tier 3)")),
            min_n_per_axis=int(te.get("min_n_per_axis", 30)),
            clip_duration_range_s=clip_duration_range_s,
        ))

    return DatasetManifest(
        source=str(dataset_cfg.get("source", "FoleyBench")),
        foleybench_root=dataset_cfg.get("foleybench_root"),
        subsets=subsets,
        clip_duration_range_s=clip_duration_range_s,
        class_balance_policy=str(
            dataset_cfg.get("class_balance", "balance coarse event classes; report per-class n")
        ),
        event_anchor_priority=list(
            dataset_cfg.get("event_anchor_priority",
                            ["foleybench_metadata", "visual_onset_detector", "light_human_marks"])
        ),
        anchor_uncertainty_required=bool(
            dataset_cfg.get("anchor_uncertainty_required", True)
        ),
        min_usable_n_per_axis=dict(
            dataset_cfg.get("min_usable_n_per_axis",
                            {"presence": 40, "timing": 40, "class": 40,
                             "material": 40, "binding": 30})
        ),
        extra=extra,
    )


# ---------------------------------------------------------------------------
# build_synthetic_dataset
# ---------------------------------------------------------------------------

def build_synthetic_dataset(
    n_videos: int,
    dim: int = 4,
    seed: int = 0,
) -> list[dict]:
    """Build a tiny synthetic dataset for CI / dry-run (NO real audio, NO GPU).

    Each element is a dict with keys:
      video_id  : str, e.g. "vid000"
      cond      : SyntheticVideoCond (distinct mu per video, well-separated)
      anchors   : EventAnchor (one synthetic onset per video, with uncertainty)

    The synthetic EventAnchor assigns a single event onset near the centre of
    a 3-second virtual clip, with a small fixed uncertainty.  The source tag is
    "synthetic" to distinguish it from real FoleyBench provenance.

    Distinct video means distinct mu: backed by SyntheticGaussianFlow.make_video_bank
    (seed-deterministic, mu_scale=2.0).
    """
    conds = SyntheticGaussianFlow.make_video_bank(n_videos, dim=dim, seed=seed)

    # Deterministic synthetic timestamps: spread onsets across [0.5, 2.5] seconds
    rng = np.random.default_rng(seed + 1)
    results: list[dict] = []
    for i, cond in enumerate(conds):
        t_onset = float(rng.uniform(0.5, 2.5))
        uncertainty = float(rng.uniform(0.01, 0.05))   # tight synthetic uncertainty
        anchors = EventAnchor(
            timestamps=[t_onset],
            uncertainty=[uncertainty],
            source="synthetic",
        )
        results.append({
            "video_id": cond.video_id,
            "cond": cond,
            "anchors": anchors,
        })
    return results


# ---------------------------------------------------------------------------
# load_foleybench  (NO download; raise if root missing)
# ---------------------------------------------------------------------------

def load_foleybench(root: str | Path) -> DatasetManifest:
    """Build a DatasetManifest from a local FoleyBench root directory.

    This function reads FoleyBench metadata (metadata.json / annotations at
    root level if present) and constructs the manifest.  It does NOT download
    any data; if the root directory does not exist, FileNotFoundError is raised.

    The FoleyBench paper: Dixit et al., 2025, arXiv:2511.13219.

    Current implementation:
      - Validates that root exists.
      - Reads optional metadata.json / dataset_info.json if present.
      - Falls back to default manifest values from configs/dataset.json
        when the metadata file is absent (the audit-only path).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"FoleyBench root not found: {root!r}. "
            "This project does NOT download datasets. "
            "Provide the path to a local FoleyBench copy "
            "(arXiv:2511.13219)."
        )

    # Attempt to read a metadata.json or dataset_info.json at root level.
    meta: dict = {}
    for candidate in ["metadata.json", "dataset_info.json"]:
        mpath = root / candidate
        if mpath.exists():
            with open(mpath, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            break

    # Build from the default config augmented by whatever the metadata holds.
    # Load default dataset config from the package configs directory.
    configs_dir = Path(__file__).resolve().parent.parent / "configs"
    with open(configs_dir / "dataset.json", "r", encoding="utf-8") as fh:
        default_cfg: dict = json.load(fh)

    # Override defaults with whatever the on-disk metadata provides.
    merged = {**default_cfg, **meta}
    merged["foleybench_root"] = str(root)
    manifest = build_manifest(merged)
    return manifest


# ---------------------------------------------------------------------------
# Markdown renderers
# ---------------------------------------------------------------------------

def manifest_to_markdown(manifest: DatasetManifest) -> str:
    """Render a DatasetManifest as a plain-markdown string (dataset_subset_manifest.md).

    Covers: source, root, subsets, clip duration, class balance, anchor priority,
    uncertainty requirement, and the declared min_usable_n_per_axis table.
    """
    lines: list[str] = [
        "# Dataset Subset Manifest",
        "",
        f"**Source:** {manifest.source}",
        f"**FoleyBench root:** {manifest.foleybench_root or '*(not set — synthetic / pending)*'}",
        f"**Clip duration:** {manifest.clip_duration_range_s[0]:.1f} – "
        f"{manifest.clip_duration_range_s[1]:.1f} s",
        f"**Class balance:** {manifest.class_balance_policy}",
        f"**Anchor uncertainty required:** {'yes' if manifest.anchor_uncertainty_required else 'no'}",
        "",
        "## Event-Anchor Priority",
        "",
    ]
    for rank, src in enumerate(manifest.event_anchor_priority, start=1):
        lines.append(f"{rank}. {src}")
    lines += [
        "",
        "## Subsets",
        "",
    ]
    for ss in manifest.subsets:
        status = "enabled" if ss.enabled else "disabled"
        lines.append(
            f"- **{ss.name}** ({status}): {ss.purpose}; "
            f"min n/axis = {ss.min_n_per_axis}; "
            f"clip {ss.clip_duration_range_s[0]:.1f}–{ss.clip_duration_range_s[1]:.1f} s"
        )
    lines += [
        "",
        "## Minimum Usable n per Axis",
        "",
        "Axes below their minimum are reported as **underpowered**, not as results "
        "(refine-logs/EXPERIMENT_PLAN.md §3).",
        "",
        "| axis_id | min_usable_n |",
        "|---|---|",
    ]
    for axis_id, n in sorted(manifest.min_usable_n_per_axis.items()):
        lines.append(f"| {axis_id} | {n} |")
    lines += [
        "",
        "## Notes",
        "",
        "- FoleyBench reference: Dixit et al., 2025, arXiv:2511.13219.",
        "- Do NOT rebuild the benchmark; use FoleyBench or equivalent.",
        "- Thresholds are pre-registered from pilot/anchor data; see `go_no_go_decision.md`.",
        "",
    ]
    if manifest.extra:
        lines += [
            "## Additional Config",
            "",
            "```json",
            json.dumps(manifest.extra, indent=2),
            "```",
            "",
        ]
    return "\n".join(lines)


def anchor_report_markdown(anchors: list[EventAnchor] | dict[str, EventAnchor]) -> str:
    """Render an event-anchor validation report as plain markdown.

    Accepts either a list of EventAnchor objects or a dict mapping video_id ->
    EventAnchor.  Reports: total clips, events per source, mean/max uncertainty,
    single-event vs multi-event split.
    """
    # Normalise input
    if isinstance(anchors, dict):
        anchor_list: list[EventAnchor] = list(anchors.values())
        id_list: list[str] = list(anchors.keys())
    else:
        anchor_list = anchors
        id_list = [f"clip_{i:04d}" for i in range(len(anchors))]

    n_clips = len(anchor_list)
    if n_clips == 0:
        return "# Event Anchor Validation Report\n\nNo anchors provided.\n"

    # Tally sources
    source_counts: dict[str, int] = {}
    total_events = 0
    all_uncertainties: list[float] = []
    n_single = 0
    n_multi = 0
    for a in anchor_list:
        src = a.source
        source_counts[src] = source_counts.get(src, 0) + 1
        total_events += a.n_events
        all_uncertainties.extend(a.uncertainty)
        if a.is_single_event:
            n_single += 1
        else:
            n_multi += 1

    unc_arr = np.array(all_uncertainties, dtype=float) if all_uncertainties else np.array([])
    mean_unc = float(np.mean(unc_arr)) if unc_arr.size > 0 else float("nan")
    max_unc = float(np.max(unc_arr)) if unc_arr.size > 0 else float("nan")

    lines = [
        "# Event Anchor Validation Report",
        "",
        f"**Total clips:** {n_clips}",
        f"**Total events:** {total_events}",
        f"**Single-event clips:** {n_single}",
        f"**Multi-event clips:** {n_multi}",
        f"**Mean uncertainty:** {mean_unc:.4f} s",
        f"**Max uncertainty:** {max_unc:.4f} s",
        "",
        "## Provenance Breakdown",
        "",
        "| source | n_clips |",
        "|---|---|",
    ]
    for src, cnt in sorted(source_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {src} | {cnt} |")

    lines += [
        "",
        "## Anchor Priority (project config)",
        "",
        "Priority order (highest first): foleybench_metadata → visual_onset_detector "
        "→ light_human_marks.",
        "",
        "## Notes",
        "",
        "- Anchor uncertainty is required for timing and binding axes "
          "(anchor_uncertainty_required = true).",
        "- Clips without uncertainty estimates are not used for timing/binding axes.",
        "",
    ]
    return "\n".join(lines)
