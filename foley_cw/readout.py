"""Phase 2 — Readout Map (foley_cw/readout.py).

Measures s_read(axis, probe, target): the minimum progress s at which a probe can predict
the axis self-target from the blurry Tweedie x0(s), for two target flavors:

  * "ode"           — self-target of the deterministic (alpha=0) ODE completion of x_s.
  * "fork_majority" — majority self-target across K stochastic fork completions from x_s.

Scientific contract (refine-logs/EXPERIMENT_PLAN.md Phase 2):
  - Readout uses the DECODED x0(s) = tweedie_x0(v, x_s, t) as the probe input (the
    blurry best-guess preview; NOT the final audio).
  - Accuracy is measured against the model's OWN self-target (self-target of ODE
    completion, or majority self-target of forks), NOT vs human/MLLM correctness-vs-video.
  - Bootstrap unit = video; CI computed by bootstrap_over_videos.
  - probe_ladder(include_stubs=False) returns only CPU-runnable probes for synthetic runs.
  - Both target kinds are always run and reported; interpretation alongside commitment
    status (commit.py) is the caller's responsibility.

Accuracy metric:
  - Categorical axes: exact-match accuracy (fraction where probe label == target label).
  - Embedding axes: mean pairwise cosine similarity (probe embedding vs target embedding).

Interpretation note from the plan:
  Predicting an uncommitted deterministic path (low commit(s,axis)) is NOT the same as
  reading a decided axis — the probe is reading one path among many. Only readout where
  commit(s,axis) is high licenses early action.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .axes import Measurer, measure_self_target
from .probes import Probe, probe_ladder
from .score_sde import fork_tail, generate_trajectory, ode_complete, x0_at
from .stats import window_with_ci
from .types import (
    Axis,
    AxisKind,
    ReadoutCell,
    ScheduleSpec,
    SelfTarget,
    Thresholds,
    WindowEstimate,
)

# ---------------------------------------------------------------------------
# Target helpers
# ---------------------------------------------------------------------------


def ode_target(
    backend: Any,
    x_s: np.ndarray,
    s: float,
    cond: Any,
    axis: Axis,
    measurer: Any,
    schedule: ScheduleSpec,
) -> SelfTarget:
    """ODE-target: self-target of the deterministic (alpha=0) completion of x_s.

    This is the 'original path this candidate would realize' — a single deterministic
    output computed via ode_complete, then measured with the axis measurer.

    Parameters
    ----------
    backend   : FlowModelBackend
    x_s       : intermediate state at progress s
    s         : progress value in [0, 1]
    cond      : video conditioning
    axis      : which axis to measure
    measurer  : Measurer (SyntheticMeasurer or RealMeasurer)
    schedule  : ScheduleSpec

    Returns
    -------
    SelfTarget — the axis self-target of the ODE-completed audio
    """
    audio = ode_complete(backend, x_s, s, cond, schedule)
    return measure_self_target(audio, axis, measurer)


def fork_majority_target(
    backend: Any,
    x_s: np.ndarray,
    s: float,
    cond: Any,
    axis: Axis,
    measurer: Any,
    alpha: float,
    schedule: ScheduleSpec,
    rng: np.random.Generator,
) -> SelfTarget:
    """Fork-majority target: majority self-target across K stochastic fork completions.

    Runs fork_tail (K forks from x_s to s=1 with stochasticity alpha), measures each
    fork's self-target, then returns the majority (mode for categorical, mean for
    embedding).

    Parameters
    ----------
    backend   : FlowModelBackend
    x_s       : intermediate state at progress s
    s         : progress value in [0, 1]
    cond      : video conditioning
    axis      : which axis to measure
    measurer  : Measurer
    alpha     : SDE stochasticity knob
    schedule  : ScheduleSpec (K_forks used)
    rng       : numpy random generator

    Returns
    -------
    SelfTarget — the majority/mean self-target of K fork completions
    """
    K = schedule.K_forks
    fork_audios = fork_tail(backend, x_s, s, cond, alpha, K, schedule, rng)
    targets = [measure_self_target(audio, axis, measurer) for audio in fork_audios]

    if axis.kind is AxisKind.CATEGORICAL:
        # Majority vote (mode of labels)
        labels = [t.label for t in targets]
        # Use numpy to find the mode
        unique_labels, counts = _label_mode(labels)
        majority_label = unique_labels[int(np.argmax(counts))]
        return SelfTarget(axis_id=axis.id, kind=AxisKind.CATEGORICAL, label=majority_label)

    else:  # EMBEDDING
        # Mean of unit-normed embeddings, re-normalized
        embs = np.stack([np.asarray(t.embedding, dtype=float) for t in targets], axis=0)
        mean_emb = np.mean(embs, axis=0)
        norm = float(np.linalg.norm(mean_emb))
        if norm > 1e-12:
            mean_emb = mean_emb / norm
        else:
            mean_emb = np.zeros_like(mean_emb)
        return SelfTarget(axis_id=axis.id, kind=AxisKind.EMBEDDING, embedding=mean_emb)


def _label_mode(labels: list) -> tuple[list, np.ndarray]:
    """Return (unique_labels, counts) for a list of hashable labels."""
    unique_labels = []
    seen: dict = {}
    for lbl in labels:
        key = repr(lbl)  # make hashable for any label type
        if key not in seen:
            seen[key] = (len(unique_labels), lbl)
            unique_labels.append(lbl)
        seen[key] = (seen[key][0], lbl)

    counts = np.zeros(len(unique_labels), dtype=int)
    for lbl in labels:
        key = repr(lbl)
        counts[seen[key][0]] += 1
    return unique_labels, counts


# ---------------------------------------------------------------------------
# Per-video readout accuracy helpers
# ---------------------------------------------------------------------------


def _probe_accuracy(
    probe: Probe,
    x0_audio: np.ndarray,
    axis: Axis,
    target: SelfTarget,
) -> float:
    """Accuracy of probe.predict(x0_audio) vs a target SelfTarget.

    For categorical axes: 1.0 if labels match, 0.0 otherwise.
    For embedding axes: cosine similarity between probe embedding and target embedding,
    if they share the same dimension; 0.0 if dimensions differ (e.g. when a heuristic
    probe returns audio-space vectors while the measurer uses a learned projection).
    """
    predicted = probe.predict(x0_audio, axis)

    if axis.kind is AxisKind.CATEGORICAL:
        return 1.0 if predicted.label == target.label else 0.0
    else:  # EMBEDDING
        pred_emb = np.asarray(predicted.embedding, dtype=float).ravel()
        tgt_emb = np.asarray(target.embedding, dtype=float).ravel()
        # Guard: if dimensions differ (e.g. heuristic probe vs projected measurer target),
        # cosine cannot be computed; return 0.0 (no readout signal for this probe/axis pair).
        if pred_emb.shape[0] != tgt_emb.shape[0]:
            return 0.0
        p_norm = float(np.linalg.norm(pred_emb))
        t_norm = float(np.linalg.norm(tgt_emb))
        if p_norm < 1e-12 or t_norm < 1e-12:
            return 0.0
        cos = float(np.dot(pred_emb / p_norm, tgt_emb / t_norm))
        return float(np.clip(cos, -1.0, 1.0))


def readout_curve_for_video(
    backend: Any,
    video: Any,
    axis: Axis,
    probe: Probe,
    target_kind: str,
    alpha: float,
    schedule: ScheduleSpec,
    measurer: Any,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute the readout accuracy curve for one video over all scan points.

    For each scan point s in schedule.scan_points:
      1. Get x_s from a trajectory generated for this video.
      2. Compute x0(s) = Tweedie best-guess decoded audio at progress s.
      3. Compute the target self-target (ODE or fork_majority).
      4. Score probe.predict(x0(s)) accuracy vs that target.

    Parameters
    ----------
    backend     : FlowModelBackend
    video       : an object with .cond attribute (SyntheticVideoCond or similar) or
                  a dict with 'cond' key (as returned by build_synthetic_dataset)
    axis        : which axis to score
    probe       : Probe instance
    target_kind : "ode" or "fork_majority"
    alpha       : SDE stochasticity knob (used for fork_majority target; ignored for ode)
    schedule    : ScheduleSpec
    measurer    : Measurer (e.g. SyntheticMeasurer)
    rng         : numpy random generator

    Returns
    -------
    ndarray of shape (n_scan_points,) — accuracy at each scan point s
    """
    if target_kind not in ("ode", "fork_majority"):
        raise ValueError(f"target_kind must be 'ode' or 'fork_majority', got {target_kind!r}")

    # Resolve cond from video:
    #   - dict with "cond" key  (from dataset.build_synthetic_dataset)
    #   - object with .cond attribute  (wrapper around a SyntheticVideoCond)
    #   - direct conditioning object   (SyntheticVideoCond from make_video_bank)
    if isinstance(video, dict):
        cond = video["cond"]
    elif hasattr(video, "cond"):
        cond = video.cond
    else:
        # video IS the cond (e.g. SyntheticVideoCond passed directly)
        cond = video

    scan_points = list(schedule.scan_points)
    n_s = len(scan_points)

    # Generate a single trajectory for this video, recording all scan points.
    # We use alpha=0.0 for the base trajectory (ODE), then fork from each x_s if needed.
    traj = generate_trajectory(
        backend, cond, schedule, rng,
        alpha=0.0,
        record_points=tuple(scan_points),
    )
    states = traj["states"]

    accuracies = np.empty(n_s, dtype=float)
    for idx, s in enumerate(scan_points):
        x_s = states[float(s)]

        # Compute x0(s): Tweedie decoded audio
        x0_audio = x0_at(backend, x_s, s, cond)

        # Compute target
        if target_kind == "ode":
            target = ode_target(backend, x_s, s, cond, axis, measurer, schedule)
        else:  # fork_majority
            target = fork_majority_target(
                backend, x_s, s, cond, axis, measurer, alpha, schedule, rng
            )

        # Score the probe
        accuracies[idx] = _probe_accuracy(probe, x0_audio, axis, target)

    return accuracies


# ---------------------------------------------------------------------------
# build_readout_map
# ---------------------------------------------------------------------------


def build_readout_map(
    backend: Any,
    videos: list,
    axes: list[Axis],
    probes: list[Probe],
    alpha: float,
    schedule: ScheduleSpec,
    thresholds: Thresholds,
    measurer: Any,
    rng: np.random.Generator,
    min_n_per_axis: "dict[str, int] | None" = None,
) -> tuple[list[ReadoutCell], dict[tuple, WindowEstimate]]:
    """Build the full readout map: probe accuracy vs s for all axes × probes × targets.

    For each (axis, probe, target_kind) combination:
      1. Compute per-video readout accuracy curves over schedule.scan_points.
      2. Aggregate into ReadoutCell rows (one per (axis, probe, s, target) combination).
      3. Compute WindowEstimate (s_read with bootstrap CI) by finding the first crossing
         of thresholds.theta_read in the mean accuracy curve.

    Parameters
    ----------
    backend    : FlowModelBackend
    videos     : list of video objects (each with .cond or dict with 'cond' key)
    axes       : list of Axis descriptors to measure
    probes     : list of Probe instances (use probe_ladder(include_stubs=False) for CPU)
    alpha      : SDE stochasticity knob for fork_majority target
    schedule   : ScheduleSpec
    thresholds : Thresholds (theta_read used as the crossing threshold)
    measurer   : Measurer (SyntheticMeasurer or RealMeasurer)
    rng        : numpy random generator

    Returns
    -------
    cells   : list[ReadoutCell] — one row per (axis, probe, s, target_kind)
    windows : dict[(axis_id, probe_name, target_kind) -> WindowEstimate] — s_read with CI
    """
    scan_points = np.asarray(schedule.scan_points, dtype=float)
    n_s = len(scan_points)
    n_videos = len(videos)
    target_kinds = ("ode", "fork_majority")

    cells: list[ReadoutCell] = []
    windows: dict[tuple, WindowEstimate] = {}

    for axis in axes:
        for probe in probes:
            for target_kind in target_kinds:
                # Collect per-video accuracy curves: shape (n_videos, n_s)
                per_video_curves = np.empty((n_videos, n_s), dtype=float)
                for v_idx, video in enumerate(videos):
                    per_video_curves[v_idx] = readout_curve_for_video(
                        backend=backend,
                        video=video,
                        axis=axis,
                        probe=probe,
                        target_kind=target_kind,
                        alpha=alpha,
                        schedule=schedule,
                        measurer=measurer,
                        rng=rng,
                    )

                # Compute mean curve for ReadoutCell rows
                mean_curve = np.mean(per_video_curves, axis=0)

                # Build ReadoutCell rows (one per scan point s)
                for s_idx, s in enumerate(scan_points):
                    cells.append(ReadoutCell(
                        axis_id=axis.id,
                        probe=probe.name,
                        s=float(s),
                        target=target_kind,
                        score=float(mean_curve[s_idx]),
                        n_videos=n_videos,
                    ))

                # Compute WindowEstimate using bootstrap over videos
                win = window_with_ci(
                    per_video_curves=per_video_curves,
                    s_grid=scan_points,
                    theta=thresholds.theta_read,
                    kind="read",
                    axis_id=axis.id,
                    n_boot=1000,
                    ci=0.95,
                    seed=0,
                    min_n=int((min_n_per_axis or {}).get(axis.id, 1)),
                )
                # Expose per-video curves so the gap CI can bootstrap s_read jointly with
                # s_commit over the same videos (plan §3: CIs on gaps).
                win.extra["per_video_curves"] = per_video_curves
                win.extra["s_grid"] = scan_points
                windows[(axis.id, probe.name, target_kind)] = win

    return cells, windows
