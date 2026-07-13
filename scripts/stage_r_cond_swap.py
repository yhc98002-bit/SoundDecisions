#!/usr/bin/env python
"""Stage R — Condition-swap intervention (manual §8.1, Fig. 5). The REAL GPU runner.

The clean causal test that disambiguates seed vs. conditioning vs. entropy-reduction
(PI Decision 1). Per SOURCE clip (journal unit), paired with a DONOR clip, under the
deployed (cfg=4.5, sqrt_down) certified kernel:

  * SOURCE base ODE trajectory (alpha=0), states recorded at the swap s-points; its
    own final audio is measured -> per-axis SOURCE value.
  * DONOR base ODE trajectory (alpha=0); its own final audio measured -> per-axis
    DONOR value.
  * at each swap s: continue the SOURCE state x_s to s=1 with the DONOR's cond
    (mode=swap) — or with an interpolated cond (mode=interp, --w sweep) — under the
    deterministic ODE at the deployed cfg; measure the swapped final -> per-axis
    SWAPPED value. (~10% wav audit on swapped finals.)

s-points bracket each source clip's s_commit (read from the cfg=4.5 commitment map);
fall back to a fixed bracket if a clip has no mapped s_commit. The --sanity mode runs
the pre-registered controls (swap at s~=0 -> full follow; s~=1 -> full retention) on a
small clip subset. The CPU --aggregate step flattens journals into follow/retention
curves + s_cond per axis (foley_cw.condition_swap) and writes cond_swap_map.csv.

Mirrors scripts/phase1_commitment.py (InstrumentedBackend + RealFoleyMeasurer +
RunStore + StorageBudget, --shard i/n over an12/an29, rng_for, journaled/resumable).
Exploratory; outputs NEVER feed decision tokens (§8.1). DO NOT run without approval.

Run sharded (deployed cfg):
  scripts/run_on_node.sh an12 'for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i \
    python scripts/stage_r_cond_swap.py --shard $i/8 > logs/condswap_$i.log 2>&1 & done; wait'
Aggregate (CPU): python scripts/stage_r_cond_swap.py --aggregate
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw import condition_swap as CS  # noqa: E402
from foley_cw import score_sde as K  # noqa: E402
from foley_cw.config import load_config  # noqa: E402
from foley_cw.kernel_provenance import assert_certified_kernel  # noqa: E402
from foley_cw.run_store import RunStore  # noqa: E402
from foley_cw.storage_budget import StorageBudget  # noqa: E402
from foley_cw.types import AxisKind, ScheduleSpec  # noqa: E402

# §8.1: class + gross timing are the headline swap axes; presence/material carried
# as context (cheap — same forward pass) so the 2-D picture has all four axes.
SWAP_AXIS_IDS = ("class", "timing", "presence", "material")
PHASE1_S_GRID = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
# Fixed fallback bracket when a clip has no mapped s_commit (3 s-points, §8.1 scale).
DEFAULT_BRACKET = (0.25, 0.45, 0.60)
# Sanity controls: swap at the extreme ends of the schedule (§8.1).
SANITY_S = (0.05, 0.90)


def rng_for(seed: int, *parts) -> np.random.Generator:
    entropy = [seed] + [zlib.crc32(str(p).encode()) for p in parts]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _value_for_axis(target, kind: AxisKind):
    """Extract the comparable axis value from a measured SelfTarget."""
    if kind is AxisKind.EMBEDDING:
        return np.asarray(target.embedding, dtype=float)
    return target.label


def load_s_commit(commit_csv: Path) -> dict:
    """Per-(clip, axis) s_commit from a Phase-1 commitment map: earliest s whose
    commit_gain >= 0.5 (a simple bracket anchor; the swap brackets around it).
    Returns {clip: {axis_id: s_commit_or_None}}. Missing file -> empty (fallback)."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    if not commit_csv.exists():
        return {}
    rows = list(csv.DictReader(commit_csv.open()))
    by_ca: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        cg = r.get("commit_gain")
        if cg in (None, "", "None"):
            continue
        try:
            by_ca[(r["clip"], r["axis_id"])].append((float(r["s"]), float(cg)))
        except (TypeError, ValueError):
            continue
    for (clip, ax), pts in by_ca.items():
        pts.sort()
        sc = next((s for s, g in pts if g >= 0.5), None)
        out[clip][ax] = sc
    return out


def bracket_for(clip: str, s_commit_map: dict, grid=PHASE1_S_GRID,
                axis: str = "class") -> tuple[float, ...]:
    """3 s-points bracketing the clip's s_commit on the integration grid (§8.1)."""
    sc = (s_commit_map.get(clip, {}) or {}).get(axis)
    if sc is None:
        return DEFAULT_BRACKET
    g = list(grid)
    i = int(np.argmin([abs(s - sc) for s in g]))
    lo = g[max(i - 1, 0)]
    hi = g[min(i + 1, len(g) - 1)]
    return tuple(sorted({lo, g[i], hi}))


def make_donor_map(clips: list[str], seed: int) -> dict[str, str]:
    """Deterministic source->donor pairing: a fixed derangement of the clip list so
    every source has a distinct donor (and no clip is its own donor)."""
    order = sorted(clips)
    perm = list(order)
    rng = np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(b"donor")]))
    rng.shuffle(perm)
    for i, c in enumerate(order):
        if perm[i] == c:  # break self-pairings by rotating one step
            perm[i], perm[(i + 1) % len(perm)] = perm[(i + 1) % len(perm)], perm[i]
    return {order[i]: perm[i] for i in range(len(order))}


def _base_traj(ib, cond, schedule, rng, record_points):
    return K.generate_trajectory(ib, cond, schedule, rng, alpha=0.0,
                                 record_points=tuple(record_points))


def run_pair(ib, measurer, axes, store: RunStore, source: str, donor: str,
             clips_root: Path, schedule: ScheduleSpec, seed: int, cfg: float,
             tag: str, s_points: tuple[float, ...], mode: str, w: float,
             audit_frac: float = 0.10) -> dict:
    """One source/donor swap unit: base SOURCE + base DONOR + swapped completions."""
    t0 = time.time(); nfe0 = ib.nfe
    ib.cfg_strength = cfg
    src_cond = ib.make_video_cond(str(clips_root / f"{source}.mp4"), video_id=source)
    don_cond = ib.make_video_cond(str(clips_root / f"{donor}.mp4"), video_id=donor)
    rec = tuple(s_points)

    # --- SOURCE base ODE trajectory (fork source for the swap) + own final ---
    src_gid = f"{source}__{tag}_src"
    src = _base_traj(ib, src_cond, schedule, rng_for(seed, source, "src"), rec)
    store.put_final_wav(src_gid, src["audio"])
    src_val = {}
    for a in axes:
        tgt = measurer.measure(src["audio"], a)
        src_val[a.id] = _value_for_axis(tgt, a.kind)
        store.record_measurement(src_gid, a.id, tgt,
                                 extra={"role": f"{tag}_source_final", "clip": source,
                                        "donor": donor, "cfg": cfg})

    # --- DONOR base ODE trajectory + own final (the follow target) ---
    don_gid = f"{donor}__{tag}_don_for_{source}"
    don = _base_traj(ib, don_cond, schedule, rng_for(seed, donor, "don"), (1.0,))
    store.put_final_wav(don_gid, don["audio"])
    don_val = {}
    for a in axes:
        tgt = measurer.measure(don["audio"], a)
        don_val[a.id] = _value_for_axis(tgt, a.kind)
        store.record_measurement(don_gid, a.id, tgt,
                                 extra={"role": f"{tag}_donor_final", "clip": donor,
                                        "source": source, "cfg": cfg})

    # --- swap (or interp) completion of the SOURCE state at each s with DONOR cond ---
    swap_val = {a.id: {} for a in axes}  # axis -> {s_key: value}
    audit_rng = rng_for(seed, source, "audit")
    for s in s_points:
        x_s = src["states"][s]
        if mode == "interp":
            mixed = CS.mix_cond(ib, src_cond, don_cond, w)
            audio = CS.cond_interp_complete(ib, x_s, s, mixed, schedule)
            srole = f"{tag}_interp_w{w:g}"
        else:
            audio = CS.cond_swap_complete(ib, x_s, s, don_cond, schedule)
            srole = f"{tag}_swap"
        sgid = f"{source}__to__{donor}__{tag}_{mode}_s{s:.2f}"
        if audit_rng.random() < audit_frac:
            store.put_final_wav(sgid, audio)
        for a in axes:
            tgt = measurer.measure(audio, a)
            swap_val[a.id][f"{s:.2f}"] = (
                _value_for_axis(tgt, a.kind).tolist() if a.kind is AxisKind.EMBEDDING
                else _value_for_axis(tgt, a.kind))
            store.record_measurement(sgid, a.id, tgt,
                                     extra={"role": srole, "s": float(s), "mode": mode,
                                            "w": w, "source": source, "donor": donor,
                                            "cfg": cfg})

    nfe = ib.nfe - nfe0
    elapsed = time.time() - t0
    src_json = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in src_val.items()}
    don_json = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in don_val.items()}
    print(f"[condswap {source}->{donor}] {elapsed:.0f}s nfe={nfe} s={list(s_points)}", flush=True)
    return {"source": source, "donor": donor, "cfg": cfg, "mode": mode, "w": w,
            "schedule": schedule.g_kind, "s_points": list(s_points),
            "source_val": src_json, "donor_val": don_json, "swap_val": swap_val,
            "elapsed_s": round(elapsed, 1), "nfe_velocity_calls": int(nfe)}


def aggregate(out: Path, tag: str, axes) -> int:
    """Flatten swap journals -> per-axis follow/retention curves + s_cond + sanity.

    Pools swap pairs across all source clips at each scanned s (per axis), so the
    rates are over the whole source population (manual §8.1: 40 source clips)."""
    store = RunStore(out)
    kind_by_axis = {a.id: a.kind for a in axes}
    # axis -> s_key -> {"swapped":[], "donor":[], "source":[]}
    pooled: dict[str, dict[str, dict[str, list]]] = {
        a.id: defaultdict(lambda: {"swapped": [], "donor": [], "source": []}) for a in axes
    }
    n_units = 0
    for jpath in sorted((out / "journal").glob(f"{tag}__*.json")):
        d = json.loads(jpath.read_text())
        if "swap_val" not in d:
            continue
        n_units += 1
        for a in axes:
            sv = d["swap_val"].get(a.id, {})
            dval = d["donor_val"].get(a.id)
            srcval = d["source_val"].get(a.id)
            if dval is None or srcval is None:
                continue
            for sk, val in sv.items():
                pooled[a.id][sk]["swapped"].append(val)
                pooled[a.id][sk]["donor"].append(dval)
                pooled[a.id][sk]["source"].append(srcval)

    rows = []
    summary = {}
    for a in axes:
        swapped_by_s = {float(sk): v["swapped"] for sk, v in pooled[a.id].items()}
        donor_by_s = {float(sk): v["donor"] for sk, v in pooled[a.id].items()}
        source_by_s = {float(sk): v["source"] for sk, v in pooled[a.id].items()}
        if not swapped_by_s:
            summary[a.id] = {"s_cond": None, "sanity": None}
            continue
        res = CS.summarize_axis(swapped_by_s, donor_by_s, source_by_s, a.kind)
        summary[a.id] = {"s_cond": res["s_cond"], "sanity": res["sanity"]}
        for s in sorted(res["rates"], key=float):
            r = res["rates"][s]
            rows.append({"axis_id": a.id, "kind": a.kind.value, "s": s,
                         "follow_rate": r["follow"], "retention_rate": r["retention"],
                         "neither_rate": r["neither"], "n": r["n"]})

    out_dir = out / "stage_r"; out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"cond_swap_map_{tag}.csv"
    with csv_path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["axis_id", "kind", "s", "follow_rate",
                                           "retention_rate", "neither_rate", "n"])
        wr.writeheader(); wr.writerows(rows)
    (out_dir / f"cond_swap_summary_{tag}.json").write_text(
        json.dumps({"tag": tag, "n_units": n_units, "axes": summary}, indent=2,
                   default=lambda o: float(o) if isinstance(o, np.floating) else o))
    print(f"[aggregate] {n_units} units, {len(rows)} rows -> {csv_path}")
    for ax, s in summary.items():
        print(f"  {ax}: s_cond={s['s_cond']} sanity={s.get('sanity')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/manifests/phase1_manifest_frozen.json"))
    ap.add_argument("--clips-root", type=Path, default=Path("data/FoleyBench/clips"))
    ap.add_argument("--out", type=Path, default=Path("results/stage0"))
    ap.add_argument("--certified", type=Path,
                    default=Path("results/stage_m_rerun/certified_kernels.json"))
    ap.add_argument("--commit-map", type=Path,
                    default=Path("results/stage0/phase1/commitment_map_p1cfg45.csv"),
                    help="cfg=4.5 commitment map; s_commit anchors the swap bracket")
    ap.add_argument("--cfg", type=float, default=4.5, help="deployed cfg (§8.1)")
    ap.add_argument("--schedule", default="sqrt_down")
    ap.add_argument("--variant", default="small_16k")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-sources", type=int, default=40, help="§8.1 scale: 40 source clips")
    ap.add_argument("--mode", choices=["swap", "interp"], default="swap",
                    help="full conditioning swap, or the pre-registered interp fallback")
    ap.add_argument("--w", type=float, default=0.5,
                    help="interp mix weight (1-w)*source + w*donor (mode=interp)")
    ap.add_argument("--bracket-axis", default="class",
                    help="axis whose s_commit anchors the 3-point bracket")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clip-set", default="single_event",
                    choices=["single_event", "two_event", "both"])
    ap.add_argument("--tag", default=None, help="default condswap_cfg<cfg>_<mode>")
    ap.add_argument("--sanity", action="store_true",
                    help="run the s~=0 / s~=1 sanity controls on the first 10 sources")
    # §8.1 condition-swap is EXPLORATORY and "never a gate" — it completes the
    # deterministic ODE at the deployed cfg as a causal probe, not a headline commitment
    # grid, so a candidate (un-ratified) kernel is acceptable here (unlike a commitment
    # grid, which must run under a ratified tuple). Default False; pass --require-ratified
    # to enforce ratification once cfg=4.5 is certified.
    ap.add_argument("--require-ratified", action=argparse.BooleanOptionalAction,
                    default=False)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    tag = args.tag or f"condswap_cfg{args.cfg:g}_{args.mode}" + (
        f"_w{args.w:g}" if args.mode == "interp" else "") + ("_sanity" if args.sanity else "")
    man = json.loads(args.manifest.read_text())
    if args.clip_set == "both":
        clips = sorted(set(man["clips"]["single_event"]) | set(man["clips"]["two_event"]))
    else:
        clips = sorted(str(c) for c in man["clips"][args.clip_set])

    axes = [a for a in load_config().axes if a.id in SWAP_AXIS_IDS]

    if args.aggregate:
        return aggregate(args.out, tag, axes)

    # Provenance guard (§15.8): deployed cfg must carry a certified/ratified kernel.
    cert = assert_certified_kernel(args.cfg, args.schedule, args.certified,
                                   require_ratified=args.require_ratified)
    print(f"[condswap] kernel OK: {cert['token']} (ratified={cert['ratified']})", flush=True)

    donor_map = make_donor_map(clips, args.seed)
    s_commit_map = load_s_commit(args.commit_map)
    sources = clips[:10] if args.sanity else clips[: args.n_sources]

    schedule = ScheduleSpec(n_steps=args.num_steps, scan_points=PHASE1_S_GRID,
                            g_kind=args.schedule, g_value=1.0)
    grid = schedule.integration_s_grid()

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    budget = StorageBudget(cap_gb=100.0)
    store = RunStore(args.out, budget=budget)
    store.account_preexisting_tree()

    todo = [c for i, c in enumerate(sources) if i % shard_n == shard_i]
    if args.limit:
        todo = todo[: args.limit]
    todo = [c for c in todo if not store.is_done(f"{tag}__{c}")]
    print(f"[condswap] shard {args.shard}: {len(todo)} sources (tag={tag}, cfg={args.cfg}, "
          f"mode={args.mode}, sanity={args.sanity})", flush=True)
    if not todo:
        return 0

    from foley_cw.feature_tap import InstrumentedBackend
    from foley_cw.mmaudio_backend import MMAudioBackend
    from foley_cw.real_measurer import RealFoleyMeasurer

    backend = MMAudioBackend(variant=args.variant, device=args.device, full_precision=True,
                             cfg_strength=args.cfg, num_steps=args.num_steps,
                             duration_sec=args.duration, enable_conditions=True)
    ib = InstrumentedBackend(backend)
    measurer = RealFoleyMeasurer(device=args.device)

    for source in todo:
        donor = donor_map[source]
        if args.sanity:
            s_points = tuple(s for s in SANITY_S
                             if np.any(np.isclose(grid, s, atol=1e-9)) or True)
        else:
            s_points = bracket_for(source, s_commit_map, axis=args.bracket_axis)
        payload = run_pair(ib, measurer, axes, store, source, donor, args.clips_root,
                           schedule, args.seed, cfg=args.cfg, tag=tag,
                           s_points=s_points, mode=args.mode, w=args.w)
        payload["budget"] = budget.summary()
        store.journal_done(f"{tag}__{source}", payload)
    print(f"[condswap] shard {args.shard} complete; budget: {budget.summary()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
