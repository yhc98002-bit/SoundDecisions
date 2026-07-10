"""Core data contract for foley_cw.

These dataclasses / enums are the fixed interface every module codes against. The
mathematically load-bearing seam (time convention, model backend, score conversion) is
in `time_map.py`, `model_adapter.py`, and `score_sde.py`; this module only holds the
plain-data types they exchange.

Conventions (frozen project-wide; see refine-logs/EXPERIMENT_PLAN.md §2):
  * Generation PROGRESS  s in [0, 1]  is the ONLY public time axis: s=0 is pure noise,
    s=1 is final audio. Raw model time `t` is never exposed without a stated convention
    (see time_map.SToT).
  * x_s = intermediate state at progress s. x0(s) = Tweedie best-guess of final audio.
  * alpha = stochasticity knob of the marginal-preserving SDE fork kernel; alpha=0 is
    the deterministic ODE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


# --------------------------------------------------------------------------------------
# Axis taxonomy (refine-logs/EXPERIMENT_PLAN.md §7 Phase 0, "Axis tiers for v1")
# --------------------------------------------------------------------------------------
class AxisTier(str, Enum):
    """Run priority for an axis."""

    TIER1 = "TIER1"          # always run: presence, gross timing, coarse class
    TIER2 = "TIER2"          # run iff reliability strong: material / fine class
    TIER3 = "TIER3"          # stretch, clean two-event clips only: multi-event binding
    SEPARATE = "SEPARATE"    # analysed separately, NOT as a window (e.g. offscreen
    #                          hallucination -> seed predictability)
    EXCLUDED = "EXCLUDED"    # explicitly out of scope for v1 (e.g. delayed callback)


class AxisKind(str, Enum):
    """How the self-target for an axis is represented, which fixes the agreement metric."""

    CATEGORICAL = "categorical"  # label -> exact-match rate or Krippendorff's alpha
    EMBEDDING = "embedding"      # vector -> mean pairwise cosine


class AgreementMetric(str, Enum):
    EXACT_MATCH = "exact_match"
    KRIPPENDORFF_ALPHA = "krippendorff_alpha"
    MEAN_PAIRWISE_COSINE = "mean_pairwise_cosine"


@dataclass(frozen=True)
class Axis:
    """A Foley correctness axis whose commitment/readout windows we measure."""

    id: str
    name: str
    tier: AxisTier
    kind: AxisKind
    agreement: AgreementMetric
    measure: str                     # name of the per-axis measurement (see axes.measures)
    requires: Optional[str] = None   # e.g. "two_event_clips"
    note: Optional[str] = None

    def __post_init__(self) -> None:
        # Guard the kind<->metric coupling so a categorical axis can't be scored with a
        # cosine and vice versa (a silent measurement bug the plan warns about).
        if self.kind is AxisKind.EMBEDDING and self.agreement is not AgreementMetric.MEAN_PAIRWISE_COSINE:
            raise ValueError(f"embedding axis {self.id!r} must use mean_pairwise_cosine")
        if self.kind is AxisKind.CATEGORICAL and self.agreement is AgreementMetric.MEAN_PAIRWISE_COSINE:
            raise ValueError(f"categorical axis {self.id!r} cannot use mean_pairwise_cosine")


@dataclass
class SelfTarget:
    """The model's OWN final value for an axis on one piece of audio.

    NOTE: this is the self-target (the tagger/onset measurement of the COMPLETED audio),
    NOT human/MLLM correctness-vs-video. Correctness only enters in the conditional
    policy phase (Phase 4) via a separate calibration sidecar. (Assumption A1.)
    """

    axis_id: str
    kind: AxisKind
    label: Optional[Any] = None              # for CATEGORICAL
    embedding: Optional[np.ndarray] = None   # for EMBEDDING

    def __post_init__(self) -> None:
        if self.kind is AxisKind.CATEGORICAL and self.label is None:
            raise ValueError("categorical SelfTarget needs a label")
        if self.kind is AxisKind.EMBEDDING and self.embedding is None:
            raise ValueError("embedding SelfTarget needs an embedding")


# --------------------------------------------------------------------------------------
# Pre-registered thresholds (refine-logs/EXPERIMENT_PLAN.md §3)
# --------------------------------------------------------------------------------------
@dataclass
class Thresholds:
    """Pre-registered decision thresholds.

    These MUST be frozen from pilot / anchor data and recorded in go_no_go_decision.md
    BEFORE the headline maps are inspected; a threshold sweep (sensitivity) is reported
    afterward. `frozen=False` means the values are non-binding code/CI placeholders.
    """

    theta_commit: float      # commitment gain threshold for s_commit
    theta_read: float        # accuracy / AUROC threshold for s_read
    theta_rel: float         # determinism (test-retest) threshold
    theta_robust: float      # robustness threshold
    theta_cal: float         # validity (calibration) threshold
    frozen: bool = False
    frozen_from: Optional[str] = None


# --------------------------------------------------------------------------------------
# Fork / schedule configuration
# --------------------------------------------------------------------------------------
@dataclass
class ScheduleSpec:
    """Progress schedule for integration and for the commitment/readout scan.

    `n_steps` integration sub-steps span s in [0, 1]. `scan_points` are the progress
    values s at which we fork (commitment) and probe (readout). K forks per scan point;
    N_independent full generations for the video-prior A_independent.
    """

    n_steps: int = 32
    scan_points: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    K_forks: int = 16
    N_independent: int = 16
    g_kind: str = "constant"   # diffusion-scale g(s) family; see score_sde.make_g
    g_value: float = 1.0

    def integration_s_grid(self) -> np.ndarray:
        """Uniform progress grid s_0=0 .. s_{n}=1 used to discretize the SDE/ODE."""
        return np.linspace(0.0, 1.0, self.n_steps + 1)


@dataclass
class AlphaGridSpec:
    """Pilot grid for the SDE stochasticity knob alpha and the smallest-valid-alpha rule.

    Primary operating alpha = the SMALLEST alpha that produces measurable tail diversity
    while preserving valid generated audio (audio-validity guard). `primary_alpha` is
    selected at runtime; None means not yet selected.
    """

    pilot_grid: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
    diversity_min: float = 0.02          # min tail diversity to count alpha as "probing"
    audio_validity_min: float = 0.5      # min presence/quality on forks to count as valid
    primary_alpha: Optional[float] = None


# --------------------------------------------------------------------------------------
# Result cells (rows of the output CSVs)
# --------------------------------------------------------------------------------------
@dataclass
class CommitmentCell:
    """One row of commitment_map.csv: A_fork / A_independent / normalized gain at (axis, s, alpha)."""

    axis_id: str
    s: float
    alpha: float
    a_fork: float
    a_independent: float
    commit_gain: float
    n_videos: int


@dataclass
class ReadoutCell:
    """One row of readout_map.csv: probe accuracy/AUROC at (axis, probe, s, target)."""

    axis_id: str
    probe: str
    s: float
    target: str          # "ode" | "fork_majority"
    score: float         # accuracy or AUROC in [0, 1]
    n_videos: int


@dataclass
class WindowEstimate:
    """A window s_commit / s_read with a bootstrap-over-videos CI."""

    axis_id: str
    kind: str            # "commit" | "read"
    s_hat: float         # min s crossing the threshold (NaN if never crosses)
    ci_low: float
    ci_high: float
    n_videos: int
    underpowered: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReliabilityResult:
    """Per-axis three-part reliability gate outcome (Phase 0.5)."""

    axis_id: str
    determinism: float
    robustness: float
    validity: float
    passed: bool
    demoted: bool = False
    reason: str = ""


@dataclass
class ValidationResult:
    """A single Phase-0.2 SDE-validation check with a pass/fail and evidence value."""

    name: str            # e.g. "alpha0_reproduces_ode"
    passed: bool
    value: float
    threshold: float
    detail: str = ""


@dataclass
class GoNoGoDecision:
    """Emitted decision token(s) plus justification (Phase 0 / Phase 3)."""

    tokens: list[str]
    justification: str
    thresholds: Optional[Thresholds] = None
    extra: dict[str, Any] = field(default_factory=dict)
