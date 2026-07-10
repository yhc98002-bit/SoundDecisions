#!/usr/bin/env python
"""Video-conditioning de-risk: one real video -> audio, end-to-end, on the real MMAudio.

Confirms MMAudioBackend(enable_conditions=True) can (a) read a video, (b) encode CLIP+sync
conditioning, (c) generate video-conditioned audio through foley_cw's integrator + MMAudio's
VAE/vocoder. Also re-checks the velocity->score SDE crux WITH real video conditioning (cfg=1),
to confirm the conversion holds in the conditioned regime too.

Run on an17 (HF cache pre-populated, offline):
    HF_HOME=.../.hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 \
    python scripts/video_conditioning_test.py --video <mp4> --out results/
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--variant", default="small_16k")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--num-steps", type=int, default=25)
    ap.add_argument("--gen-cfg", type=float, default=4.5, help="deployed cfg for the headline generation")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from foley_cw.mmaudio_backend import MMAudioBackend
    from foley_cw.types import ScheduleSpec
    from foley_cw import score_sde as K
    from foley_cw import validation as V
    import soundfile as sf

    out = {}
    t0 = time.time()
    sr = 16000 if "16k" in args.variant else 44100

    # ---- (1) deployed-cfg video-conditioned generation (the de-risk headline) ----
    print(f"[vid] loading {args.variant} enable_conditions=True (gen cfg={args.gen_cfg}) ...", flush=True)
    be = MMAudioBackend(variant=args.variant, device="cuda", full_precision=True,
                        cfg_strength=args.gen_cfg, num_steps=args.num_steps,
                        duration_sec=args.duration, enable_conditions=True)
    print(f"[vid] loaded in {time.time()-t0:.1f}s; state_shape={be.state_shape}", flush=True)

    print(f"[vid] encoding video conditioning: {args.video}", flush=True)
    tc = time.time()
    cond = be.make_video_cond(args.video, prompt="", video_id=os.path.basename(args.video))
    print(f"[vid] conditioning encoded in {time.time()-tc:.1f}s", flush=True)

    sch = ScheduleSpec(n_steps=args.num_steps, scan_points=(0.0, 0.5, 1.0), K_forks=4, N_independent=4)
    tg = time.time()
    traj = K.generate_trajectory(be, cond, sch, np.random.default_rng(args.seed), alpha=0.0,
                                 record_points=(0.0, 0.5, 1.0))
    audio = traj["audio"]
    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"[vid] generated audio in {time.time()-tg:.1f}s: shape={audio.shape} "
          f"dur={audio.shape[-1]/sr:.2f}s finite={np.isfinite(audio).all()} rms={rms:.4f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    wav_path = args.out / f"vidcond_{os.path.splitext(os.path.basename(args.video))[0]}.wav"
    sf.write(str(wav_path), audio.astype(np.float32), sr)
    print(f"[vid] wrote {wav_path}", flush=True)
    out["generation"] = {"video": args.video, "audio_path": str(wav_path), "sr": sr,
                         "samples": int(audio.shape[-1]), "rms": rms,
                         "finite": bool(np.isfinite(audio).all()), "gen_cfg": args.gen_cfg}

    # ---- (2) re-check the velocity->score SDE crux WITH video conditioning (cfg=1) ----
    print("[vid] re-checking SDE crux under video conditioning (cfg=1) ...", flush=True)
    be.cfg_strength = 1.0  # pure conditional velocity for the marginal-preserving fork SDE
    rng = np.random.default_rng(args.seed + 1)
    checks, token = V.run_sde_validation(be, cond, sch, rng, alpha=0.2)
    out["sde_crux_video_cond"] = {"token": token,
                                  "checks": [{"name": c.name, "passed": bool(c.passed),
                                              "value": float(c.value), "detail": c.detail}
                                             for c in checks]}
    for c in checks:
        print(f"  {c.name}: passed={c.passed} value={c.value:.4g} | {c.detail}", flush=True)
    print(f"[vid] SDE token (video-conditioned): {token}", flush=True)

    out["meta"] = {"variant": args.variant, "duration": args.duration, "num_steps": args.num_steps,
                   "elapsed_s": round(time.time() - t0, 1)}
    (args.out / "video_conditioning_test.json").write_text(json.dumps(out, indent=2))
    print(f"[vid] wrote {args.out/'video_conditioning_test.json'}", flush=True)
    ok = out["generation"]["finite"] and out["generation"]["rms"] > 1e-4 and token == "OK"
    print(f"[vid] DE-RISK {'PASS' if ok else 'CHECK'}: video-conditioned gen valid + SDE token={token}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
