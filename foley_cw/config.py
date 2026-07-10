"""Load typed configuration from configs/*.json (stdlib json; no extra deps).

JSON (not YAML) is used so the whole config path runs on a bare numpy+stdlib environment.
Threshold values ship UNFROZEN (frozen=false): they are non-binding code/CI placeholders
and MUST be frozen from pilot/anchor data and recorded in go_no_go_decision.md before any
headline map is inspected (refine-logs/EXPERIMENT_PLAN.md §3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import (
    AgreementMetric,
    AlphaGridSpec,
    Axis,
    AxisKind,
    AxisTier,
    ScheduleSpec,
    Thresholds,
)

# Repo-root-relative default config directory.
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@dataclass
class Config:
    thresholds: Thresholds
    schedule: ScheduleSpec
    alpha_grid: AlphaGridSpec
    axes: list[Axis]
    dataset: dict[str, Any]


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_axes(path: Path) -> list[Axis]:
    raw = _load_json(path)
    items = raw["axes"] if isinstance(raw, dict) else raw
    axes: list[Axis] = []
    for a in items:
        axes.append(
            Axis(
                id=a["id"],
                name=a["name"],
                tier=AxisTier(a["tier"]),
                kind=AxisKind(a["kind"]),
                agreement=AgreementMetric(a["agreement"]),
                measure=a["measure"],
                requires=a.get("requires"),
                note=a.get("note"),
            )
        )
    return axes


def load_thresholds(path: Path) -> Thresholds:
    raw = _load_json(path)
    return Thresholds(
        theta_commit=float(raw["theta_commit"]),
        theta_read=float(raw["theta_read"]),
        theta_rel=float(raw["theta_rel"]),
        theta_robust=float(raw["theta_robust"]),
        theta_cal=float(raw["theta_cal"]),
        frozen=bool(raw.get("frozen", False)),
        frozen_from=raw.get("frozen_from"),
    )


def load_schedule(path: Path) -> ScheduleSpec:
    raw = _load_json(path)
    sp = raw.get("scan_points")
    kwargs: dict[str, Any] = {
        "n_steps": int(raw.get("n_steps", 32)),
        "K_forks": int(raw.get("K_forks", 16)),
        "N_independent": int(raw.get("N_independent", 16)),
        "g_kind": raw.get("g_kind", "constant"),
        "g_value": float(raw.get("g_value", 1.0)),
    }
    if sp is not None:
        kwargs["scan_points"] = tuple(float(x) for x in sp)
    return ScheduleSpec(**kwargs)


def load_alpha_grid(path: Path) -> AlphaGridSpec:
    raw = _load_json(path)
    pa = raw.get("primary_alpha")
    return AlphaGridSpec(
        pilot_grid=tuple(float(x) for x in raw["pilot_grid"]),
        diversity_min=float(raw.get("diversity_min", 0.02)),
        audio_validity_min=float(raw.get("audio_validity_min", 0.5)),
        primary_alpha=None if pa is None else float(pa),
    )


def load_config(config_dir: Path | str | None = None) -> Config:
    d = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
    return Config(
        thresholds=load_thresholds(d / "thresholds.json"),
        schedule=load_schedule(d / "schedule.json"),
        alpha_grid=load_alpha_grid(d / "alpha_grid.json"),
        axes=load_axes(d / "axes.json"),
        dataset=_load_json(d / "dataset.json"),
    )
