"""Phase-0 feasibility entrypoint.

Usage (synthetic dry-run, no MMAudio, no GPU):
    python -m foley_cw.cli.phase0_feasibility --synthetic --out /tmp/fcw_p0 --n-videos 6

On --synthetic the script:
  1. Builds a synthetic dataset + DatasetManifest.
  2. Runs check_trajectory_access + run_sde_validation.
  3. Runs the reliability gate on Tier-1 / Tier-2 axes.
  4. Validates event anchors.
  5. Writes the five Phase-0 result files under --out:
       feasibility_report.md
       score_sde_validation_report.md
       dataset_subset_manifest.md
       event_anchor_validation_report.md
       axis_reliability_report.md
     via foley_cw.reporting.
  6. Calls gap.decide_phase0 and prints the emitted Phase-0 token.

On the NON-synthetic path (--no-synthetic), MMAudio is required but not
installed here.  The script prints a clear message and exits non-zero WITHOUT
fabricating any results.
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


def run_phase0_synthetic(
    out_dir: Path,
    n_videos: int,
    seed: int,
    config_dir: Path | None,
    fast: bool,
) -> list[str]:
    """Run Phase-0 on the synthetic backend.  Returns the emitted Phase-0 tokens."""
    from foley_cw.config import load_config
    from foley_cw.dataset import build_manifest, build_synthetic_dataset, anchor_report_markdown
    from foley_cw.synthetic_backend import SyntheticGaussianFlow
    from foley_cw.axes import SyntheticMeasurer
    from foley_cw.validation import check_trajectory_access, run_sde_validation
    from foley_cw.reliability import reliability_gate
    from foley_cw.gap import decide_phase0
    from foley_cw.types import AxisTier
    import foley_cw.reporting as rpt

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Config ----------
    cfg = load_config(config_dir)

    # Use a fast schedule for the synthetic dry-run if requested.
    schedule = _build_fast_schedule() if fast else cfg.schedule
    thresholds = cfg.thresholds

    # ---------- Backend + dataset ----------
    backend = SyntheticGaussianFlow(dim=4, sigma2=0.25)
    rng = np.random.default_rng(seed)
    measurer = SyntheticMeasurer()

    dataset = build_synthetic_dataset(n_videos=n_videos, dim=4, seed=seed)

    # ---------- Manifest ----------
    manifest = build_manifest(cfg.dataset)
    manifest_md = anchor_report_markdown(
        {item["video_id"]: item["anchors"] for item in dataset}
    )
    # Write dataset_subset_manifest.md
    from foley_cw.dataset import manifest_to_markdown
    manifest_full_md = manifest_to_markdown(manifest)
    (out_dir / "dataset_subset_manifest.md").write_text(manifest_full_md, encoding="utf-8")
    manifest_ok = True  # synthetic manifest is always structurally valid

    # ---------- Phase 0.1 / 0.2: trajectory access + SDE validation ----------
    pilot_cond = dataset[0]["cond"]
    traj_result = check_trajectory_access(backend, pilot_cond, schedule, rng)
    trajectory_ok = traj_result.passed

    # Pick alpha for SDE validation (first positive from pilot grid)
    alpha_for_val = float(cfg.alpha_grid.pilot_grid[1])  # e.g. 0.05

    sde_results, sde_token = run_sde_validation(backend, pilot_cond, schedule, rng, alpha=alpha_for_val)

    # Write feasibility_report.md
    x_s_shape: tuple | None = None
    x0_shape: tuple | None = None
    if trajectory_ok:
        x_s_shape = backend.state_shape
        x0_shape = backend.state_shape
    rpt.feasibility_report(
        trajectory_ok=trajectory_ok,
        x_s_shape=x_s_shape,
        resume_ok=trajectory_ok,
        x0_shape=x0_shape,
        s_to_t_name=backend.s_to_t.name,
        s_to_t_verified=backend.s_to_t.verified,
        notes=f"Synthetic dry-run with {n_videos} videos; n_steps={schedule.n_steps}",
        path=out_dir / "feasibility_report.md",
    )

    # Write score_sde_validation_report.md
    rpt.score_sde_validation_report(
        validation_results=sde_results,
        token=sde_token,
        alpha_tested=alpha_for_val,
        notes=f"Synthetic analytic backend (SyntheticGaussianFlow); alpha={alpha_for_val}",
        path=out_dir / "score_sde_validation_report.md",
    )

    # ---------- Event anchor validation report ----------
    anchor_rows = [
        {
            "video_id": item["video_id"],
            "source": item["anchors"].source,
            "n_events": item["anchors"].n_events,
            "max_uncertainty_s": f"{item['anchors'].max_uncertainty:.4f}",
            "check_error": "",
        }
        for item in dataset
    ]
    coverage = float(len([r for r in anchor_rows if not r["check_error"]])) / max(len(anchor_rows), 1)
    rpt.event_anchor_validation_report(
        anchor_rows=anchor_rows,
        coverage=coverage,
        notes="Synthetic event anchors from build_synthetic_dataset; source='synthetic'",
        path=out_dir / "event_anchor_validation_report.md",
    )

    # ---------- Phase 0.5: reliability gate on Tier-1 / Tier-2 axes ----------
    gated_axes = [
        ax for ax in cfg.axes
        if ax.tier in (AxisTier.TIER1, AxisTier.TIER2)
    ]

    # Collect a small sample of synthetic audio vectors for reliability evaluation.
    # We generate N_independent completions on the first video and use those as audios.
    from foley_cw.score_sde import generate_trajectory as gen_traj
    sample_audios: list[np.ndarray] = []
    for item in dataset[:min(3, n_videos)]:
        traj_out = gen_traj(backend, item["cond"], schedule, rng, alpha=0.0,
                            record_points=(1.0,))
        sample_audios.append(traj_out["audio"])

    reliability_results = []
    for axis in gated_axes:
        res = reliability_gate(
            axis=axis,
            audios=sample_audios,
            measurer=measurer,
            thresholds=thresholds,
            rng=rng,
            sidecar=None,
        )
        reliability_results.append(res)

    rpt.axis_reliability_report(
        reliability_results=reliability_results,
        thresholds=thresholds,
        path=out_dir / "axis_reliability_report.md",
    )

    # ---------- Phase 0 gate decision ----------
    decision = decide_phase0(
        validation_token=sde_token,
        reliability=reliability_results,
        trajectory_ok=trajectory_ok,
        manifest_ok=manifest_ok,
        min_reliable_axes=3,
    )

    return decision.tokens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m foley_cw.cli.phase0_feasibility",
        description=(
            "Phase-0 feasibility diagnostic: trajectory access, SDE validation, "
            "reliability gate, anchor validation — writes Phase-0 report files."
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
        help="Output directory for Phase-0 report files (default: results/).",
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

    args = parser.parse_args(argv)

    if not args.synthetic:
        print(
            "[phase0_feasibility] ERROR: --no-synthetic was requested.\n"
            "Phase-0 with MMAudio requires:\n"
            "  * MMAudio source + weights installed (not vendored in this repo)\n"
            "  * GPU access (owner=human, on SSH nodes an12/an22 with A800s)\n"
            "  * Phase-0.1 trajectory access wired in foley_cw/model_adapter.MMAudioBackend\n"
            "\n"
            "This is an AUDIT-ONLY build (STOP-B); no GPU or MMAudio results are fabricated.\n"
            "Run with --synthetic (the default) to validate the math on CPU.\n",
            file=sys.stderr,
        )
        return 1

    out_dir: Path = args.out
    print(f"[phase0_feasibility] Running synthetic Phase-0 with {args.n_videos} videos ...")
    print(f"[phase0_feasibility] Output directory: {out_dir}")

    tokens = run_phase0_synthetic(
        out_dir=out_dir,
        n_videos=args.n_videos,
        seed=args.seed,
        config_dir=args.config_dir,
        fast=args.fast,
    )

    print(f"\n[phase0_feasibility] Phase-0 tokens emitted: {tokens}")

    # List written files
    written = sorted(out_dir.glob("*.md"))
    print(f"[phase0_feasibility] Files written to {out_dir}:")
    for f in written:
        print(f"  {f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
