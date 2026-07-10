#!/usr/bin/env python
"""Stage-0 reliability VALIDITY diagnostic — why the kappa gate reads low, decomposed.

ZERO-BUDGET and READ-ONLY w.r.t. the MLLM: it makes NO qwen call and touches NO frozen
threshold. It re-measures the on-disk screening finals with the deterministic
RealFoleyMeasurer (the gate's authoritative input; determinism=1.0 so this reproduces
the gate's labels exactly) and pairs them with the qwen gold
(results/stage0/mllm_sidecar/sidecar.csv).

Faithfulness is ESTABLISHED (statistic-level), not assumed: the frozen gate computes
validity over only a seed-0 50-clip subsample of the gold (run_real_reliability receives
the determinism subset, not all gold), so this script reproduces that exact selection and
asserts its Cohen's kappa matches the gate's `validity` before trusting the fuller 100-clip
view. A scalar-κ match is statistic-level (not per-clip) faithfulness; for a literal audit
the gate-subset (clip, measured, gold) triples are emitted to the .json.

It exists because the frozen gate uses Cohen's kappa for validity, and Cohen's kappa
COLLAPSES (even goes negative) under skewed marginals — the "kappa paradox" — which is
exactly the presence/timing regime here (~90% one category). For every categorical axis
it reports, over the SAME (measured, gold) pairs the gate uses:

  * n, raw observed agreement p_o
  * Cohen's kappa            (reproduces the gate's `validity`; cross-checked below)
  * Gwet's AC1               (skew-robust chance correction)
  * each rater's marginal label distribution (to show the skew)

For the CLASS axis it additionally decomposes the two confounds that are NOT
measurement invalidity:
  (a) label-space mismatch: the gold was elicited under the full 15-class qwen prompt,
      so some gold labels are speech/music/ambient — classes the EVENT-RESTRICTED
      12-class measurer can never emit, hence guaranteed disagreements.  We report the
      gold-event-restricted subset (drop those gold clips).
  (b) measurer abstention: the measurer abstains below the cross-group margin; the qwen
      gold never abstains.  We additionally report the both-confident subset.

This is a DIAGNOSTIC, never scientific evidence and never a gate. It does not re-tune
theta_cal=0.6; whether to adopt AC1 (or run the human sidecar) is a PI-checkpoint
question. Output: results/stage0/reliability_diagnostic.{json,md}.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw.sidecar import cohens_kappa, gwet_ac1  # noqa: E402

CATEGORICAL_AXES = ("presence", "timing", "class")


def _coerce_gold(axis_id: str, label):
    """Mirror the gate's gold typing: timing -> int bin, others -> str."""
    if axis_id == "timing":
        return int(round(float(label)))
    return str(label)


def _load_gold(sidecar_csv: Path) -> dict[str, dict[str, object]]:
    gold: dict[str, dict[str, object]] = {a: {} for a in CATEGORICAL_AXES}
    for r in csv.DictReader(sidecar_csv.open()):
        ax = r["axis_id"]
        if ax in gold:
            gold[ax][r["clip"]] = _coerce_gold(ax, r["label"])
    return gold


def _measure_finals(clips, finals_dir: Path, device: str) -> dict[str, dict[str, object]]:
    """Re-measure the on-disk screening finals — the gate's authoritative input.

    Deterministic (determinism=1.0), so this reproduces the labels the gate used.
    Imports are local so the rest of the module stays importable on the numpy core.
    """
    import soundfile as sf

    from foley_cw.config import load_config
    from foley_cw.real_measurer import RealFoleyMeasurer

    axes = {a.id: a for a in load_config().axes if a.id in CATEGORICAL_AXES}
    measurer = RealFoleyMeasurer(device=device)
    measured: dict[str, dict[str, object]] = {a: {} for a in CATEGORICAL_AXES}
    for i, clip in enumerate(clips):
        p = finals_dir / f"{clip}__screen_ind0.wav"
        if not p.exists():
            continue
        wav, _sr = sf.read(p, dtype="float32")
        audio = np.asarray(wav, dtype=np.float32)
        for ax in CATEGORICAL_AXES:
            measured[ax][clip] = measurer.measure(audio, axes[ax]).label
        if (i + 1) % 25 == 0:
            print(f"[diagnostic] measured {i + 1}/{len(clips)} finals", flush=True)
    return measured


def _stats(measured_list: list, gold_list: list) -> dict:
    n = len(measured_list)
    if n == 0:
        return {"n": 0, "p_o": float("nan"), "cohens_kappa": float("nan"),
                "gwet_ac1": float("nan")}
    p_o = sum(1 for x, y in zip(measured_list, gold_list) if x == y) / n
    return {
        "n": n,
        "p_o": p_o,
        "cohens_kappa": cohens_kappa(measured_list, gold_list),
        "gwet_ac1": gwet_ac1(measured_list, gold_list),
        "measured_marginal": dict(Counter(str(x) for x in measured_list)),
        "gold_marginal": dict(Counter(str(x) for x in gold_list)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar-csv", type=Path,
                    default=Path("results/stage0/mllm_sidecar/sidecar.csv"))
    ap.add_argument("--finals-dir", type=Path, default=Path("results/stage0/finals"))
    ap.add_argument("--coarse-map", type=Path,
                    default=Path("configs/coarse_class_map.json"))
    ap.add_argument("--gate-report", type=Path,
                    default=Path("results/stage0/reliability_report.json"),
                    help="for the Cohen-kappa cross-check (faithfulness of this join)")
    ap.add_argument("--gate-seed", type=int, default=0,
                    help="must equal stage0_reliability --seed (the det-subset RNG seed)")
    ap.add_argument("--gate-n-clips", type=int, default=50,
                    help="must equal stage0_reliability --n-clips (det-subset size)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("results/stage0"))
    args = ap.parse_args()

    gold = _load_gold(args.sidecar_csv)
    all_gold_clips = sorted({c for by in gold.values() for c in by})
    measured = _measure_finals(all_gold_clips, args.finals_dir, args.device)
    excluded = set(json.loads(args.coarse_map.read_text()).get("class_excluded_coarse", []))

    # Reproduce the gate's determinism-subset selection (stage0_reliability.py): the
    # validity kappa is computed over this 50-clip seed-0 sample, NOT all 100 gold.
    gate_subset = set(list(np.random.default_rng(args.gate_seed)
                           .permutation(all_gold_clips))[: args.gate_n_clips])

    out: dict[str, object] = {
        "_doc": "Stage-0 validity diagnostic — Cohen kappa vs Gwet AC1, with the class "
                "label-space + abstain confounds decomposed. DIAGNOSTIC ONLY, not a gate; "
                "theta_cal=0.6 unchanged.",
        "excluded_classes": sorted(excluded),
        "axes": {},
    }
    for ax in CATEGORICAL_AXES:
        clips = sorted(set(measured[ax]) & set(gold[ax]))
        m = [measured[ax][c] for c in clips]
        g = [gold[ax][c] for c in clips]
        # gate-faithful subset (the seed-0 50-clip determinism sample the gate scored)
        gclips = [c for c in clips if c in gate_subset]
        block = {"full": _stats(m, g),
                 "gate_subset": _stats([measured[ax][c] for c in gclips],
                                       [gold[ax][c] for c in gclips]),
                 # literal per-clip audit trail (Codex SHOULD): the exact triples whose
                 # Cohen kappa must equal the gate's validity.
                 "gate_subset_pairs": [{"clip": c, "measured": str(measured[ax][c]),
                                        "gold": str(gold[ax][c])} for c in gclips]}
        if ax == "class":
            # (a) gold-event-restricted: drop clips whose GOLD is an excluded class.
            keep = [c for c in clips if gold[ax][c] not in excluded]
            mr = [measured[ax][c] for c in keep]
            gr = [gold[ax][c] for c in keep]
            block["gold_event_restricted"] = _stats(mr, gr)
            block["n_gold_excluded_dropped"] = len(clips) - len(keep)
            # (b) both-confident: additionally drop measurer abstentions.
            keep2 = [c for c in keep if measured[ax][c] != "abstain"]
            block["both_confident"] = _stats([measured[ax][c] for c in keep2],
                                             [gold[ax][c] for c in keep2])
            block["n_measured_abstain_in_restricted"] = sum(
                1 for c in keep if measured[ax][c] == "abstain")
        out["axes"][ax] = block

    # --- Faithfulness cross-check: our Cohen kappa must reproduce the gate's validity.
    xcheck = {}
    if args.gate_report.exists():
        rep = json.loads(args.gate_report.read_text())
        gate_val = {r["axis_id"]: r["validity"] for r in rep.get("results", [])}
        for ax in CATEGORICAL_AXES:
            ours = out["axes"][ax]["gate_subset"]["cohens_kappa"]
            theirs = gate_val.get(ax)
            ok = (theirs is not None and math.isfinite(ours) and math.isfinite(theirs)
                  and abs(ours - theirs) < 1e-6)
            xcheck[ax] = {"diagnostic_kappa_gate_subset": ours, "gate_validity": theirs,
                          "n_gate_subset": out["axes"][ax]["gate_subset"]["n"],
                          "match": ok}
    out["gate_kappa_crosscheck"] = xcheck

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "reliability_diagnostic.json").write_text(json.dumps(out, indent=2))

    # --- Markdown ---
    L = ["# Stage-0 Validity Diagnostic — Cohen's κ vs Gwet's AC1 (DIAGNOSTIC ONLY)", "",
         "Frozen θ_cal = 0.6 is **unchanged**; this only characterises *why* the κ gate "
         "reads low. κ collapses under skewed marginals (the κ paradox); AC1 is the "
         "skew-robust counterpart. Both are reported; the PI arbitrates the statistic and "
         "the pending human sidecar.", "",
         "| axis | n | raw agree p_o | Cohen κ (gate) | Gwet AC1 |",
         "|---|---|---|---|---|"]
    for ax in CATEGORICAL_AXES:
        s = out["axes"][ax]["full"]
        L.append(f"| {ax} | {s['n']} | {s['p_o']:.3f} | {s['cohens_kappa']:.3f} | "
                 f"{s['gwet_ac1']:.3f} |")
    L += ["", "**Marginals (show the skew that breaks κ):**"]
    for ax in CATEGORICAL_AXES:
        s = out["axes"][ax]["full"]
        L.append(f"- {ax}: measured={s['measured_marginal']}  gold={s['gold_marginal']}")

    c = out["axes"]["class"]
    L += ["", "## Class axis — confound decomposition", "",
          f"- **full** (n={c['full']['n']}): p_o={c['full']['p_o']:.3f}, "
          f"κ={c['full']['cohens_kappa']:.3f}, AC1={c['full']['gwet_ac1']:.3f}",
          f"- **gold-event-restricted** (drop {c['n_gold_excluded_dropped']} clips whose "
          f"gold ∈ {sorted(excluded)} — the 15-vs-12 label-space mismatch the measurer "
          f"can never satisfy) (n={c['gold_event_restricted']['n']}): "
          f"p_o={c['gold_event_restricted']['p_o']:.3f}, "
          f"κ={c['gold_event_restricted']['cohens_kappa']:.3f}, "
          f"AC1={c['gold_event_restricted']['gwet_ac1']:.3f}",
          f"- **both-confident** — *favorable upper bound*, conditions on measurer "
          f"non-abstention (additionally drop "
          f"{c['n_measured_abstain_in_restricted']} measurer abstentions) "
          f"(n={c['both_confident']['n']}): p_o={c['both_confident']['p_o']:.3f}, "
          f"κ={c['both_confident']['cohens_kappa']:.3f}, "
          f"AC1={c['both_confident']['gwet_ac1']:.3f} — class fails even here.",
          "", "_AC1 q = number of categories observed in the rater union (a diagnostic "
          "choice; a future GATE on AC1 should preregister the full rating-scale q or "
          "report q-sensitivity — for timing the two agree to ~0.003)._"]

    L += ["", "## Cohen-κ cross-check vs the frozen gate (faithfulness of this join)",
          "Reproduces the gate's seed-0 50-clip determinism subsample and matches its "
          "`validity` to establish statistic-level faithfulness (the per-clip "
          "(clip, measured, gold) triples are in `reliability_diagnostic.json` for a "
          "literal audit)."]
    if xcheck:
        for ax, x in xcheck.items():
            tv = "—" if x["gate_validity"] is None else f"{x['gate_validity']:.6f}"
            L.append(f"- {ax} (n={x['n_gate_subset']}): diagnostic κ="
                     f"{x['diagnostic_kappa_gate_subset']:.6f} vs gate validity={tv} → "
                     f"{'MATCH' if x['match'] else 'MISMATCH'}")
    else:
        L.append("- (gate report not found; cross-check skipped)")
    (args.out / "reliability_diagnostic.md").write_text("\n".join(L) + "\n")

    print("\n".join(L))
    all_match = bool(xcheck) and all(x["match"] for x in xcheck.values())
    print(f"\n[diagnostic] cross-check all match: {all_match}")
    return 0 if (not xcheck or all_match) else 2


if __name__ == "__main__":
    raise SystemExit(main())
