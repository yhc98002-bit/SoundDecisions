"""Phases 1-2-3 maps entrypoint (commitment map + readout map + gap/decision).

Usage (synthetic dry-run, no MMAudio, no GPU):
    python -m foley_cw.cli.phases123_maps --synthetic --out /tmp/fcw_p123 --n-videos 6

On --synthetic the script:
  1. Builds a synthetic dataset.
  2. Selects the primary alpha via commitment.select_primary_alpha.
  3. Builds the commitment map (Phase 1).
  4. Builds the readout map (Phase 2).
  5. Computes gap / R1-R2 / separation metrics (Phase 3).
  6. Writes:
       results/commitment_map.csv
       results/readout_map.csv
       results/commitment_readout_gap_report.md
       results/go_no_go_decision.md  (records pre-registered thresholds BEFORE maps)
  7. Calls gap.decide_phase3 and prints the emitted Phase-3 tokens.

On the NON-synthetic path (--no-synthetic), MMAudio is required but not
installed here.  The script prints a clear message and exits non-zero WITHOUT
fabricating any results.

NOTE: pre-registered thresholds are written to go_no_go_decision.md BEFORE the
maps are inspected (plan §3 requirement).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _build_fast_schedule(n_steps: int = 8, K_forks: int = 4, N_independent: int = 4):
    """Return a lightweight ScheduleSpec for fast synthetic dry-runs."""
    from foley_cw.types import ScheduleSpec
    return ScheduleSpec(
        n_steps=n_steps,
        scan_points=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        K_forks=K_forks,
        N_independent=N_independent,
        g_kind="constant",
        g_value=1.0,
    )


def run_phases123_synthetic(
    out_dir: Path,
    n_videos: int,
    seed: int,
    config_dir: Path | None,
    fast: bool,
    min_n_override: int | None = None,
) -> list[str]:
    """Run Phases 1-2-3 on the synthetic backend.  Returns emitted Phase-3 tokens.

    min_n_override: if given, use this minimum usable n for EVERY axis instead of the
    per-axis production minimums in configs/dataset.json. The production defaults far exceed
    a small CI dry-run, so with the default (None) a small-n run correctly reports all axes
    underpowered and emits STOP_PROJECT; pass a small value (e.g. 2) to exercise the full
    GO/STOP decision logic end-to-end.
    """
    from foley_cw.config import load_config
    from foley_cw.dataset import build_synthetic_dataset
    from foley_cw.synthetic_backend import SyntheticGaussianFlow
    from foley_cw.axes import SyntheticMeasurer
    from foley_cw.types import AxisTier, GoNoGoDecision
    from foley_cw.commitment import build_commitment_map
    from foley_cw.readout import build_readout_map
    from foley_cw.probes import probe_ladder
    from foley_cw.stats import separation_score, ordered_non_overlapping
    from foley_cw.gap import decide_phase3, r1_r2_crosstab
    import foley_cw.reporting as rpt

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Config ----------
    cfg = load_config(config_dir)
    schedule = _build_fast_schedule() if fast else cfg.schedule
    thresholds = cfg.thresholds
    alpha_grid = cfg.alpha_grid

    # ---------- Backend + dataset ----------
    backend = SyntheticGaussianFlow(dim=4, sigma2=0.25)
    rng = np.random.default_rng(seed)
    measurer = SyntheticMeasurer()
    dataset = build_synthetic_dataset(n_videos=n_videos, dim=4, seed=seed)

    from foley_cw.reliability import reliability_gate
    from foley_cw.score_sde import generate_trajectory as _gen_traj

    video_conds = [item["cond"] for item in dataset]
    if min_n_override is not None:
        min_n_per_axis = {ax.id: int(min_n_override) for ax in cfg.axes}
    elif isinstance(cfg.dataset, dict):
        min_n_per_axis = cfg.dataset.get("min_usable_n_per_axis")
    else:
        min_n_per_axis = None

    # Pre-registration guard (plan §3): thresholds must be frozen before headline maps.
    if not thresholds.frozen:
        print(
            "[phases123_maps] WARNING: thresholds are UNFROZEN placeholders "
            "(configs/thresholds.json frozen=false). Per plan §3 they must be frozen from "
            "pilot/anchor data BEFORE the headline maps. This synthetic dry-run validates "
            "CODE ONLY; its windows are non-binding.",
            file=sys.stderr,
        )

    # Phase-0.5 reliability gate consumed here (plan §3: a window is valid ONLY for axes
    # that pass determinism + robustness + validity). Phases 1-3 build maps ONLY for
    # reliable axes; demoted axes are dropped here, not silently mapped. (EXCLUDED/SEPARATE
    # tiers are not window axes.)
    # The synthetic dataset (build_synthetic_dataset) creates SINGLE-event anchors, so axes
    # that require clean two-event clips (Tier-3 binding) are not measurable here and are
    # excluded — running them on single-event data would be uninterpretable (plan §7).
    candidate_axes = [
        ax for ax in cfg.axes
        if ax.tier not in (AxisTier.EXCLUDED, AxisTier.SEPARATE)
        and ax.requires != "two_event_clips"
    ]
    gate_audios = [
        _gen_traj(backend, cnd, schedule, np.random.default_rng(seed + 1),
                  alpha=0.0, record_points=(1.0,))["audio"]
        for cnd in video_conds[:min(3, len(video_conds))]
    ]
    reliability_results = [
        reliability_gate(ax, gate_audios, measurer, thresholds,
                         np.random.default_rng(seed + 2), sidecar=None)
        for ax in candidate_axes
    ]
    active_axes = [ax for ax, rr in zip(candidate_axes, reliability_results) if rr.passed]
    demoted_axis_ids = [rr.axis_id for rr in reliability_results if not rr.passed]

    # ---------- STEP 0: Record pre-registered thresholds BEFORE any maps are built ----------
    # Write go_no_go_decision.md as a PLACEHOLDER first (pre-registration requirement).
    # The final tokens will be written after the maps.  Per plan §3: thresholds must be
    # written BEFORE the headline curves are inspected.
    placeholder_decision = GoNoGoDecision(
        tokens=["THRESHOLDS_REGISTERED_PRE_MAP"],
        justification=(
            "Thresholds pre-registered below BEFORE map inspection, as required by "
            "EXPERIMENT_PLAN.md §3.  Final tokens will be overwritten after Phase 3."
        ),
        thresholds=thresholds,
        extra={"note": "placeholder — overwritten after Phase 3 map construction"},
    )
    rpt.go_no_go_decision(
        decision=placeholder_decision,
        path=out_dir / "go_no_go_decision.md",
    )

    # ---------- Phase 1: Commitment map ----------
    cells_commit, commit_windows, primary_alpha = build_commitment_map(
        backend=backend,
        videos=video_conds,
        axes=active_axes,
        alpha_grid=alpha_grid,
        schedule=schedule,
        thresholds=thresholds,
        measurer=measurer,
        rng=rng,
        min_n_per_axis=min_n_per_axis,
    )
    alpha_ok = primary_alpha is not None
    effective_alpha = primary_alpha if primary_alpha is not None else 0.05

    rpt.write_commitment_map_csv(cells_commit, out_dir / "commitment_map.csv",
                                 windows=commit_windows)

    # ---------- Phase 2: Readout map ----------
    probes = probe_ladder(include_stubs=False)  # CPU-runnable only
    cells_read, read_windows = build_readout_map(
        backend=backend,
        videos=video_conds,
        axes=active_axes,
        probes=probes,
        alpha=effective_alpha,
        schedule=schedule,
        thresholds=thresholds,
        measurer=measurer,
        rng=rng,
        min_n_per_axis=min_n_per_axis,
    )
    rpt.write_readout_map_csv(cells_read, out_dir / "readout_map.csv",
                              windows=read_windows)

    # ---------- Phase 3: Gap / separation / decision ----------
    # Separation is reported over RESULT windows only (non-NaN AND not underpowered), to match
    # decide_phase3 (which recomputes it internally) — underpowered axes are not results (§3).
    result_commit_windows = {
        k: w for k, w in commit_windows.items()
        if not np.isnan(w.s_hat) and not w.underpowered
    }
    sep = separation_score(result_commit_windows)
    non_overlapping = ordered_non_overlapping(result_commit_windows)

    # Build mean commitment curves per axis for the R1/R2 crosstab.
    # The cells_commit rows hold the mean commit_gain per (axis, s, alpha).
    # We extract the primary_alpha curves for each axis.
    s_grid = np.array(list(schedule.scan_points), dtype=float)

    commit_curves_mean: dict[str, np.ndarray] = {}
    for ax in active_axes:
        # Find cells for this axis at the primary (effective) alpha
        axis_cells = [
            c for c in cells_commit
            if c.axis_id == ax.id and abs(c.alpha - effective_alpha) < 1e-9
        ]
        if axis_cells:
            # Sort by s to align with s_grid
            axis_cells_sorted = sorted(axis_cells, key=lambda c: c.s)
            commit_curves_mean[ax.id] = np.array(
                [c.commit_gain for c in axis_cells_sorted], dtype=float
            )
        else:
            commit_curves_mean[ax.id] = np.zeros(len(s_grid), dtype=float)

    # Mean readout curves per axis (use the energy_onset probe, "ode" target)
    read_curves_mean: dict[str, np.ndarray] = {}
    for ax in active_axes:
        read_cells = [
            c for c in cells_read
            if c.axis_id == ax.id and c.probe == "energy_onset" and c.target == "ode"
        ]
        if read_cells:
            read_cells_sorted = sorted(read_cells, key=lambda c: c.s)
            read_curves_mean[ax.id] = np.array(
                [c.score for c in read_cells_sorted], dtype=float
            )
        else:
            read_curves_mean[ax.id] = np.zeros(len(s_grid), dtype=float)

    crosstab = r1_r2_crosstab(
        commit_curves=commit_curves_mean,
        readout_curves=read_curves_mean,
        s_grid=s_grid,
        thresholds=thresholds,
    )

    # Threshold sweep values for the (result-filtered) separation sensitivity below. The
    # plan's "threshold sensitivity" is re-reported separation under the theta_commit sweep
    # (see separation_under_thresholds), which is result-window filtered; the older
    # single-axis s_commit sweep was redundant and is intentionally not produced.
    sweep_thetas = [0.3, 0.5, 0.7, 0.9]

    # Bootstrap-over-video gap CIs (plan §3: CIs on gaps, not only point estimates). Reuse
    # the per-video curves stashed on the windows so s_read and s_commit are resampled jointly
    # over the same videos.
    from foley_cw.stats import bootstrap_gap_ci, separation_under_thresholds
    gap_cis: dict = {}
    for key, rw in read_windows.items():
        cw = commit_windows.get(rw.axis_id)
        if cw is None:
            continue
        # Only result windows produce a gap; underpowered / no-crossing windows are not
        # results (plan §3) and are excluded from the gap evidence.
        if rw.underpowered or cw.underpowered or np.isnan(rw.s_hat) or np.isnan(cw.s_hat):
            continue
        c_pv = cw.extra.get("per_video_curves")
        r_pv = rw.extra.get("per_video_curves")
        if c_pv is None or r_pv is None:
            continue
        gap_cis[key] = bootstrap_gap_ci(
            c_pv, r_pv, s_grid,
            theta_commit=thresholds.theta_commit, theta_read=thresholds.theta_read,
            n_boot=200, seed=seed,
        )

    # Threshold sensitivity: re-report axis SEPARATION under a theta_commit sweep (plan §3),
    # not just one axis's s_hat.
    per_axis_commit_curves = {
        ax_id: cw.extra["per_video_curves"]
        for ax_id, cw in commit_windows.items()
        if "per_video_curves" in cw.extra
    }
    separation_sensitivity = (
        separation_under_thresholds(per_axis_commit_curves, s_grid, sweep_thetas,
                                    min_n_per_axis=min_n_per_axis, n_boot=50, seed=seed)
        if per_axis_commit_curves else None
    )

    # Write the gap report
    rpt.commitment_readout_gap_report(
        commit_windows=commit_windows,
        read_windows=read_windows,
        crosstab={str(k): v for k, v in crosstab.items()},
        separation_score_val=sep,
        ordered_non_overlapping=non_overlapping,
        gap_cis=gap_cis,
        separation_sensitivity=separation_sensitivity,
        notes=(
            f"Synthetic dry-run; n_videos={n_videos}; primary_alpha={primary_alpha}; "
            f"fast={fast}. Reliability gate consumed: mapped axes={[ax.id for ax in active_axes]}; "
            f"demoted/dropped axes={demoted_axis_ids}. "
            f"thresholds.frozen={thresholds.frozen} "
            f"({'NON-BINDING placeholders' if not thresholds.frozen else 'frozen'})."
        ),
        path=out_dir / "commitment_readout_gap_report.md",
    )

    # ---------- Phase-3 decision ----------
    decision = decide_phase3(
        commit_windows=commit_windows,
        read_windows=read_windows,
        separation=sep,
        thresholds=thresholds,
        alpha_ok=alpha_ok,
    )
    decision.extra["thresholds_frozen"] = thresholds.frozen
    decision.extra["mapped_axes"] = [ax.id for ax in active_axes]
    decision.extra["reliability_demoted_axes"] = demoted_axis_ids
    if not thresholds.frozen:
        decision.extra["caveat"] = (
            "Thresholds UNFROZEN (placeholders); windows are non-binding code validation only."
        )

    # Overwrite go_no_go_decision.md with the FINAL decision (which still records thresholds).
    rpt.go_no_go_decision(
        decision=decision,
        path=out_dir / "go_no_go_decision.md",
    )

    return decision.tokens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m foley_cw.cli.phases123_maps",
        description=(
            "Phases 1-2-3 maps: commitment map, readout map, gap analysis, and Phase-3 "
            "go/no-go decision.  Writes CSV + Markdown result files."
        ),
    )
    parser.add_argument(
        "--synthetic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the synthetic CPU backend (default: True). "
             "--no-synthetic requires MMAudio wired by a human engineer with GPU.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Path to the configs/ directory. Defaults to the package-bundled configs/.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results"),
        help="Output directory for result files (default: results/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducibility (default: 0).",
    )
    parser.add_argument(
        "--n-videos",
        type=int,
        default=6,
        help="Number of synthetic videos for the dry-run (default: 6).",
    )
    parser.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a reduced schedule (fewer steps/forks) for a faster dry-run (default: True).",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=None,
        help="Override the minimum usable n for EVERY axis (default: per-axis production "
             "values from configs/dataset.json). With the default, a small dry-run correctly "
             "reports axes underpowered (STOP_PROJECT); pass e.g. 2 to exercise the full "
             "decision logic.",
    )

    args = parser.parse_args(argv)

    if not args.synthetic:
        print(
            "[phases123_maps] ERROR: --no-synthetic was requested.\n"
            "Phases 1-2-3 with MMAudio require:\n"
            "  * MMAudio source + weights installed (not vendored in this repo)\n"
            "  * GPU access (owner=human, on SSH nodes an12/an22 with A800s)\n"
            "  * Phase-0 feasibility diagnostic completed (GO_MAPS_PHASE token)\n"
            "\n"
            "This is an AUDIT-ONLY build (STOP-B); no GPU or MMAudio results are fabricated.\n"
            "Run with --synthetic (the default) to validate the math on CPU.\n",
            file=sys.stderr,
        )
        return 1

    out_dir: Path = args.out
    print(f"[phases123_maps] Running synthetic Phases 1-2-3 with {args.n_videos} videos ...")
    print(f"[phases123_maps] Output directory: {out_dir}")
    print(f"[phases123_maps] Pre-registered thresholds will be written to go_no_go_decision.md "
          "BEFORE the maps are inspected (plan §3).")

    tokens = run_phases123_synthetic(
        out_dir=out_dir,
        n_videos=args.n_videos,
        seed=args.seed,
        config_dir=args.config_dir,
        fast=args.fast,
        min_n_override=args.min_n,
    )

    print(f"\n[phases123_maps] Phase-3 tokens emitted: {tokens}")

    # List written files
    all_written = sorted(out_dir.glob("*"))
    print(f"[phases123_maps] Files written to {out_dir}:")
    for f in all_written:
        print(f"  {f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
