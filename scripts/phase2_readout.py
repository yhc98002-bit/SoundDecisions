#!/usr/bin/env python
"""Phase 2 - readout maps (manual section 5) from cached x0(s) previews.

Does an external probe reading the blurry Tweedie preview x̂0(s) recover the trajectory's
final self-target, and at what earliest s (s_read)? Subjects = the Phase-1 independents
(each an ODE trajectory; ODE-target = its own final self-target). The audio-tagger probe
is the only implemented probe path (RealFoleyMeasurer on the preview wav).

Per (clip, subject j, s): load previews/<gid>__s{s}.wav, run the probe → predicted label
(or embedding), compare to the final self-target recorded for that independent. Accuracy
for categorical axes and cosine for material are reduced to per-clip means, then
bootstrapped by clip. s_read(axis, probe, ode) is reported against both the absolute
theta_read threshold and a 0.15 margin over the categorical majority baseline.

Collection is sharded like the Phase-1 runner. --aggregate is CPU-only and writes a v3
re-analysis without changing the committed Arc-3 or WP-A files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw.run_store import RunStore, sanitize_unit_id  # noqa: E402
from foley_cw.storage_budget import StorageBudget  # noqa: E402

SUBJECT_TAG = "p1cfg1"   # readout subjects = the Phase-1 cfg=1.0 independents

PHASE1_S_GRID = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
LABEL_AXES = ("presence", "timing", "class")
EMB_AXES = ("material",)


def load_final_targets(measurements: Path, tag: str) -> dict:
    """label[gid][axis] and emb[gid]['material'] for the independents (final self-target)."""
    lab: dict = defaultdict(dict); emb: dict = defaultdict(dict)
    role = f"{tag}_independent"
    with measurements.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            if (d.get("extra") or {}).get("role") != role:
                continue
            ax = d["axis_id"]; tgt = d.get("target") or {}
            if ax in LABEL_AXES and tgt.get("label") is not None:
                lab[d["gen_id"]][ax] = str(tgt["label"])
            elif ax in EMB_AXES and tgt.get("embedding") is not None:
                emb[d["gen_id"]][ax] = np.asarray(tgt["embedding"], dtype=np.float32)
    return lab, emb


def _cos(a, b) -> float:
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def run_clip(measurer, axes, store, clip, finals_lab, finals_emb, n_subj, tag, sf):
    import numpy as np
    rows = []
    prev_dir = store._dir("previews")
    for j in range(n_subj):
        gid = f"{clip}__{SUBJECT_TAG}_ind{j}"   # subject = p1 independent
        lab = finals_lab.get(gid, {}); emb = finals_emb.get(gid, {})
        if not lab:
            continue
        for s in PHASE1_S_GRID:
            prev = prev_dir / f"{sanitize_unit_id(gid)}__s{s:.2f}.wav"
            if not prev.exists():
                continue
            wav, _sr = sf.read(prev, dtype="float32")
            audio = np.asarray(wav, dtype=np.float32)
            for a in axes:
                tgt = measurer.measure(audio, a)
                if a.id in LABEL_AXES:
                    correct = 1.0 if str(tgt.label) == lab.get(a.id) else 0.0
                else:
                    correct = _cos(tgt.embedding, emb.get(a.id)) if a.id in emb else float("nan")
                rows.append({"clip": clip, "j": j, "axis_id": a.id, "s": s,
                             "probe": "audio_tagger", "target": "ode", "correct": correct})
    return rows


def bootstrap_clip_mean(
    values_by_clip: dict[str, list[float]], n_boot: int = 1000, seed: int = 0
) -> tuple[float, float, float, int]:
    """Mean of per-clip means and a percentile CI from resampled clips."""
    clip_means = np.asarray(
        [
            np.mean([v for v in values_by_clip[clip] if np.isfinite(v)])
            for clip in sorted(values_by_clip)
            if any(np.isfinite(v) for v in values_by_clip[clip])
        ],
        dtype=float,
    )
    if clip_means.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    point = float(np.mean(clip_means))
    if clip_means.size == 1:
        return point, point, point, 1
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, clip_means.size, size=(n_boot, clip_means.size))
    boot = np.mean(clip_means[indices], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi), int(clip_means.size)


def majority_frequency(labels: list[str]) -> float:
    """Frequency of the modal ODE-target label among evaluated rows."""
    if not labels:
        return float("nan")
    counts = Counter(labels)
    return float(max(counts.values()) / len(labels))


def balanced_accuracy_from_correct(
    labels: list[str],
    correct: list[float],
    classes: tuple[str, ...] | None = None,
) -> float:
    """Mean recall over a fixed true-class universe.

    A score is undefined when any registered class is absent. This prevents
    bootstrap draws from silently changing the balanced-accuracy estimand.
    """
    by_class: dict[str, list[float]] = defaultdict(list)
    for label, value in zip(labels, correct):
        if np.isfinite(value):
            by_class[label].append(float(value))
    class_universe = classes if classes is not None else tuple(sorted(by_class))
    if not class_universe or any(not by_class.get(label) for label in class_universe):
        return float("nan")
    return float(np.mean([np.mean(by_class[label]) for label in class_universe]))


def bootstrap_balanced_accuracy(
    rows_by_clip: dict[str, list[tuple[str, float]]],
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Balanced accuracy with clips, not individual generations, as resampling units."""
    clips = sorted(rows_by_clip)
    if not clips:
        return float("nan"), float("nan"), float("nan")

    class_universe = tuple(
        sorted({label for pairs in rows_by_clip.values() for label, _ in pairs})
    )

    def score(selected: list[str]) -> float:
        pairs = [pair for clip in selected for pair in rows_by_clip[clip]]
        return balanced_accuracy_from_correct(
            [label for label, _ in pairs],
            [value for _, value in pairs],
            classes=class_universe,
        )

    point = score(clips)
    if len(clips) == 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    attempts = 0
    max_attempts = max(n_boot * 100, n_boot)
    while len(draws) < n_boot and attempts < max_attempts:
        indices = rng.integers(0, len(clips), size=len(clips))
        value = score([clips[int(index)] for index in indices])
        attempts += 1
        if np.isfinite(value):
            draws.append(value)
    if len(draws) != n_boot:
        raise ValueError(
            "balanced-accuracy clip bootstrap could not retain every true class: "
            f"{len(draws)}/{n_boot} valid draws after {attempts} attempts"
        )
    finite = np.asarray(draws, dtype=float)
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return point, float(lo), float(hi)


def _gid_for_row(row: dict) -> str:
    """Recover the Phase-1 subject ID used by run_clip for baseline labels."""
    if row.get("gen_id"):
        return str(row["gen_id"])
    return f"{row['clip']}__{SUBJECT_TAG}_ind{int(row['j'])}"


def summarize_rows(
    journal_rows: list[dict],
    final_labels: dict,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[list[dict], bool]:
    """Aggregate journal rows by cell after reducing repeated subjects within clip."""
    grouped: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    labels_by_cell: dict[tuple, list[str]] = defaultdict(list)
    balanced_by_cell: dict[tuple, dict[str, list[tuple[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in journal_rows:
        axis = row["axis_id"]
        cell = (axis, row["probe"], row["target"], float(row["s"]))
        value = float(row["correct"])
        grouped[cell][str(row["clip"])].append(value)
        if axis in LABEL_AXES and row["target"] == "ode" and np.isfinite(value):
            gid = _gid_for_row(row)
            label = final_labels.get(gid, {}).get(axis)
            if label is None:
                raise ValueError(
                    f"missing ODE-target label for gen_id={gid}, axis={axis}"
                )
            labels_by_cell[cell].append(str(label))
            balanced_by_cell[cell][str(row["clip"])].append((str(label), value))

    axis_order = {axis: i for i, axis in enumerate(LABEL_AXES + EMB_AXES)}
    rows = []
    for cell in sorted(
        grouped,
        key=lambda item: (axis_order.get(item[0], len(axis_order)), item[1], item[2], item[3]),
    ):
        axis, probe, target, s = cell
        value, ci_lo, ci_hi, n_clips = bootstrap_clip_mean(
            grouped[cell], n_boot=n_boot, seed=seed
        )
        metric = "cosine" if axis in EMB_AXES else "exact_match"
        majority = (
            majority_frequency(labels_by_cell[cell])
            if metric == "exact_match" and target == "ode"
            else float("nan")
        )
        row = {
            "axis_id": axis,
            "probe": probe,
            "target": target,
            "s": s,
            "metric": metric,
            "accuracy": value,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n_clips": n_clips,
            "majority_baseline": majority,
            "margin_over_majority": (
                value - majority
                if np.isfinite(value) and np.isfinite(majority)
                else float("nan")
            ),
        }
        if metric == "exact_match" and target == "ode":
            balanced, bal_lo, bal_hi = bootstrap_balanced_accuracy(
                balanced_by_cell[cell], n_boot=n_boot, seed=seed
            )
        else:
            balanced = bal_lo = bal_hi = float("nan")
        row["balanced_accuracy"] = balanced
        row["bal_ci_lo"] = bal_lo
        row["bal_ci_hi"] = bal_hi
        rows.append(row)
    return rows, bool(rows)


def _legacy_values(path: Path) -> dict[tuple, float]:
    if not path.exists():
        raise FileNotFoundError(f"legacy readout map is missing: {path}")
    return {
        (row["axis_id"], row["probe"], row["target"], float(row["s"])): float(
            row["accuracy"]
        )
        for row in csv.DictReader(path.open(newline="", encoding="utf-8"))
    }


def _first_crossing(rows: list[dict], field: str, threshold: float) -> dict[tuple, float]:
    by_key: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        by_key[(row["axis_id"], row["probe"], row["target"])].append(
            (float(row["s"]), float(row[field]))
        )
    out = {}
    for key, values in by_key.items():
        crossings = [s for s, value in values if np.isfinite(value) and value >= threshold]
        out[key] = min(crossings) if crossings else float("nan")
    return out


def aggregate(
    out: Path,
    clips,
    tag,
    theta_read,
    output_dir: Path = Path("results/arc4_wpA2"),
    legacy_csv: Path | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    artifact_stem: str = "readout_map_v3",
):
    store = RunStore(out)
    journal_rows = []
    missing = []
    for clip in clips:
        unit = f"{tag}__{clip}"
        if not store.is_done(unit):
            missing.append(clip); continue
        for r in store.load_journal(unit).get("rows", []):
            row = dict(r)
            row.setdefault("clip", str(clip))
            if str(row["clip"]) != str(clip):
                raise ValueError(
                    f"journal row clip mismatch: unit={unit}, row={row['clip']}"
                )
            journal_rows.append(row)

    final_labels, _ = load_final_targets(
        out / "measurements" / "measurements.jsonl", SUBJECT_TAG
    )
    rows, _ = summarize_rows(
        journal_rows, final_labels, n_boot=n_boot, seed=seed
    )

    legacy_path = legacy_csv or (out / "phase1" / f"readout_map_{tag}.csv")
    legacy = _legacy_values(legacy_path)
    actual_keys = {
        (row["axis_id"], row["probe"], row["target"], float(row["s"]))
        for row in rows
    }
    if actual_keys != set(legacy):
        raise ValueError(
            f"v3/legacy cell mismatch: missing={sorted(set(legacy) - actual_keys)}, "
            f"extra={sorted(actual_keys - set(legacy))}"
        )
    for row in rows:
        key = (row["axis_id"], row["probe"], row["target"], float(row["s"]))
        if not math.isclose(
            float(row["accuracy"]), legacy[key], rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                f"legacy pooled value not reproduced for {key}: "
                f"v3={row['accuracy']}, legacy={legacy[key]}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{artifact_stem}.csv"
    fieldnames = [
        "axis_id", "probe", "target", "s", "metric", "accuracy", "ci_lo", "ci_hi",
        "n_clips", "majority_baseline", "margin_over_majority", "balanced_accuracy",
        "bal_ci_lo", "bal_ci_hi",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

    complete = not missing
    sread_absolute = _first_crossing(rows, "accuracy", theta_read)
    sread_margin = _first_crossing(rows, "margin_over_majority", 0.15)

    def fmt(value: float) -> str:
        return "never" if not np.isfinite(value) else f"{value:g}"

    L = [f"# Phase-2 Readout Map v3 ({tag})", "",
         f"theta_read = {theta_read}; ODE-target; audio-tagger probe on cached x0(s) "
         f"previews. {len(clips)-len(missing)}/{len(clips)} clips. Values and 95% CIs "
         f"use per-clip means with a {n_boot}-draw clip bootstrap (seed {seed}).", "",
         "Categorical values use exact-match accuracy. Material values are mean embedding "
         "cosines, not accuracies. The categorical majority baseline is the modal "
         "ODE-target-label frequency among evaluated rows in each cell. Balanced accuracy "
         "joins labels through the deterministic Phase-1 subject ID and is bootstrapped by "
         "clip; any missing join is a hard error.", "",
         "| axis | probe | metric | legacy s_read (absolute theta) | s_read_margin |",
         "|---|---|---|---:|---:|"]
    for key in sorted(sread_absolute):
        metric = "cosine" if key[0] in EMB_AXES else "exact_match"
        margin_text = "not applicable" if metric == "cosine" else fmt(sread_margin[key])
        L.append(
            f"| {key[0]} | {key[1]} | {metric} | {fmt(sread_absolute[key])} | "
            f"{margin_text} |"
        )

    early = [row for row in rows if math.isclose(float(row["s"]), 0.05)]
    L += ["", "## Baseline lens at s=0.05", "",
          "| axis | metric value | majority baseline | margin |",
          "|---|---:|---:|---:|"]
    for row in early:
        if row["metric"] == "cosine":
            L.append(f"| {row['axis_id']} | {float(row['accuracy']):.6f} | n/a | n/a |")
        else:
            L.append(
                f"| {row['axis_id']} | {float(row['accuracy']):.6f} | "
                f"{float(row['majority_baseline']):.6f} | "
                f"{float(row['margin_over_majority']):.6f} |"
            )

    L += ["", "**FLAGGED - Track P:** the persisted Track-P JSON contains only aggregate "
          "best-layer scores and layer IDs, not per-example predictions or clip IDs. "
          "Applying a clip bootstrap would require retraining and an unregistered choice "
          "about layer selection inside versus outside resampling."]
    md_path = output_dir / f"{artifact_stem}.md"
    md_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"[readout] wrote {csv_path} and {md_path}; missing={len(missing)}")
    return 0 if complete else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifests/phase1_manifest_frozen.json"))
    ap.add_argument("--out", type=Path, default=Path("results/stage0"))
    ap.add_argument("--thresholds", type=Path, default=Path("configs/thresholds.json"))
    ap.add_argument("--tag", default="p2cfg1")
    ap.add_argument("--n-subjects", type=int, default=4)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--analysis-out", type=Path, default=Path("results/arc4_wpA2"))
    ap.add_argument("--artifact-stem", default="readout_map_v3")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    args = ap.parse_args()

    theta_read = json.loads(args.thresholds.read_text())["theta_read"]
    clips = sorted(str(c) for c in json.loads(args.manifest.read_text())["clips"]["single_event"])
    if args.aggregate:
        return aggregate(
            args.out, clips, args.tag, theta_read, output_dir=args.analysis_out,
            n_boot=args.n_boot, seed=args.bootstrap_seed, artifact_stem=args.artifact_stem,
        )

    import soundfile as sf
    from foley_cw.config import load_config
    from foley_cw.real_measurer import RealFoleyMeasurer

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    store = RunStore(args.out, budget=StorageBudget(cap_gb=100.0))
    store.account_preexisting_tree()
    todo = [c for i, c in enumerate(clips) if i % shard_n == shard_i]
    if args.limit:
        todo = todo[: args.limit]
    todo = [c for c in todo if not store.is_done(f"{args.tag}__{c}")]
    print(f"[readout] shard {args.shard}: {len(todo)} clips", flush=True)
    if not todo:
        return 0
    axes = [a for a in load_config().axes if a.id in (LABEL_AXES + EMB_AXES)]
    measurer = RealFoleyMeasurer(device=args.device)
    finals_lab, finals_emb = load_final_targets(args.out / "measurements" / "measurements.jsonl",
                                                "p1cfg1")
    for clip in todo:
        rows = run_clip(measurer, axes, store, clip, finals_lab, finals_emb,
                        args.n_subjects, args.tag, sf)
        store.journal_done(f"{args.tag}__{clip}", {"clip": clip, "rows": rows})
        print(f"[readout {clip}] {len(rows)} rows", flush=True)
    print(f"[readout] shard {args.shard} complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
