#!/usr/bin/env python
"""Phase-0.1/0.2 crux on the REAL MMAudio backend (GPU).

Runs the LOAD-BEARING part of diag_phase0_feasibility — trajectory access + the
velocity->score SDE validation — against MMAudio via foley_cw.mmaudio_backend.MMAudioBackend,
using foley_cw's backend-agnostic checks (foley_cw.validation). Text conditioning is used so
NO FoleyBench video is needed for this crux; the dataset/anchor manifest and reliability gate
(Phase 0.3/0.4/0.5) come later and need real data.

Emits the Phase-0.2 token (OK / FIX_SCORE_CONVERSION / FORK_ALPHA_NO_VALID_OPERATING_POINT)
and a trajectory-access pass/fail. This is honest feasibility evidence, NOT a map and NOT a
claim.

Usage (on an17):
    .venv/bin/python scripts/phase0_mmaudio_validate.py --prompt "footsteps on gravel" \
        --variant small_16k --duration 4 --num-steps 25 --alpha 0.2 --k 8 --out results/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Make foley_cw importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="small_16k")
    ap.add_argument("--prompt", default="footsteps on gravel")
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--num-steps", type=int, default=25)
    ap.add_argument("--alpha", type=float, default=0.2, help="test alpha for fork validity/diversity")
    ap.add_argument("--k", type=int, default=8, help="K forks")
    ap.add_argument("--cfg", type=float, default=1.0, help="cfg_strength for the fork SDE (1.0 = pure conditional)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    from foley_cw.mmaudio_backend import MMAudioBackend
    from foley_cw.types import ScheduleSpec
    from foley_cw import validation as V

    t0 = time.time()
    print(f"[phase0] loading MMAudio {args.variant} (float32, cfg={args.cfg}) on {args.device} ...", flush=True)
    backend = MMAudioBackend(
        variant=args.variant, device=args.device, full_precision=True,
        cfg_strength=args.cfg, num_steps=args.num_steps, duration_sec=args.duration,
        enable_conditions=False,  # crux uses an unconditional cond; no CLIP/synchformer needed
    )
    print(f"[phase0] loaded in {time.time()-t0:.1f}s; latent state_shape={backend.state_shape}", flush=True)

    # Unconditional cond: the trajectory + velocity->score crux tests the flow-network mechanics,
    # not the conditioning content, so empty conditioning is sufficient (and needs no CLIP).
    cond = backend.make_empty_cond(video_id="empty")
    rng = np.random.default_rng(args.seed)
    schedule = ScheduleSpec(
        n_steps=args.num_steps,
        scan_points=(0.0, 0.25, 0.5, 0.75, 1.0),
        K_forks=args.k, N_independent=args.k,
        g_kind="constant", g_value=1.0,
    )

    results = {}

    # Phase 0.1: trajectory access (extract x_s, resume, compute x0(s)).
    print("[phase0] checking trajectory access ...", flush=True)
    ta = V.check_trajectory_access(backend, cond, schedule, rng)
    results["trajectory_access"] = _vr(ta)
    print(f"  trajectory_access: passed={ta.passed} ({ta.detail})", flush=True)

    # Phase 0.2: velocity->score SDE validation (alpha=0 ODE, continuity, fork validity, diversity).
    print(f"[phase0] running SDE validation (alpha={args.alpha}, K={args.k}) ...", flush=True)
    t1 = time.time()
    checks, token = V.run_sde_validation(backend, cond, schedule, rng, alpha=args.alpha)
    print(f"[phase0] SDE validation done in {time.time()-t1:.1f}s", flush=True)
    for c in checks:
        results.setdefault("sde_checks", []).append(_vr(c))
        print(f"  {c.name}: passed={c.passed} value={c.value:.4g} thr={c.threshold:.4g} | {c.detail}", flush=True)

    results["sde_token"] = token
    results["meta"] = {
        "variant": args.variant, "prompt": args.prompt, "duration": args.duration,
        "num_steps": args.num_steps, "alpha": args.alpha, "k": args.k, "cfg": args.cfg,
        "device": args.device, "seed": args.seed, "state_shape": list(backend.state_shape),
        "elapsed_total_s": round(time.time() - t0, 1),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "phase0_mmaudio_validation.json").write_text(json.dumps(results, indent=2))
    print(f"\n[phase0] SDE token: {token}")
    print(f"[phase0] trajectory_access: {'PASS' if ta.passed else 'FAIL'}")
    print(f"[phase0] wrote {args.out/'phase0_mmaudio_validation.json'}")
    # Exit nonzero if the crux failed, so callers can gate.
    ok = ta.passed and token == "OK"
    return 0 if ok else 2


def _vr(r) -> dict:
    return {"name": r.name, "passed": bool(r.passed), "value": float(r.value),
            "threshold": float(r.threshold), "detail": r.detail}


if __name__ == "__main__":
    raise SystemExit(main())
