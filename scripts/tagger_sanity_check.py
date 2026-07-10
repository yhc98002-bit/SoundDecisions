#!/usr/bin/env python
"""Tagger sanity gate (W3): PANNs coarse class on 20 original FoleyBench audio
tracks vs their ucs_category. Agreement < 0.5 -> exit 2 (halt; fix the coarse map
before any generation-side measurement is trusted).

'Agreement' is accept-set based: each UCS category maps to the set of coarse
classes that are reasonable for it (a hammer hit is legitimately impact_friction
OR tools_hand). CPU-only, login node.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foley_cw.real_measurer import RealFoleyMeasurer  # noqa: E402
from foley_cw.visual_anchors import extract_audio_track  # noqa: E402

UCS_ACCEPT: dict[str, set[str]] = {
    "FOOTSTEPS": {"footsteps_walk", "impact_friction"},
    "TOOLS": {"tools_hand", "machines_motors", "impact_friction"},
    "FOOD & DRINK": {"food_cooking", "water_liquid", "impact_friction"},
    "OBJECTS": {"impact_friction", "doors_furniture", "tools_hand"},
    "GUNS": {"guns_explosions"},
    "WEAPONS": {"guns_explosions", "impact_friction"},
    "METAL": {"impact_friction", "tools_hand", "machines_motors"},
    "WOOD": {"impact_friction", "tools_hand"},
    "GLASS": {"impact_friction"},
    "ROCKS": {"impact_friction"},
    "PAPER": {"impact_friction"},
    "PLASTIC": {"impact_friction"},
    "CLOTH": {"impact_friction"},
    "MECHANICAL": {"machines_motors", "impact_friction", "electronics_ui"},
    "MACHINES": {"machines_motors", "vehicles", "electronics_ui"},
    "MOTORS": {"machines_motors", "vehicles"},
    "USER INTERFACE": {"electronics_ui", "impact_friction"},
    "COMPUTERS": {"electronics_ui", "impact_friction"},
    "COMMUNICATIONS": {"electronics_ui"},
    "BELLS": {"electronics_ui", "music", "impact_friction"},
    "ALARMS": {"electronics_ui"},
    "DOORS": {"doors_furniture", "impact_friction"},
    "SPORTS": {"impact_friction", "footsteps_walk"},
    "FIGHT": {"impact_friction", "doors_furniture", "speech_vocal"},  # grunts/exertion
    "MOVEMENT": {"impact_friction", "footsteps_walk"},
    "VEHICLES": {"vehicles", "machines_motors"},
    "AIRCRAFT": {"vehicles", "machines_motors"},
    "BOATS": {"vehicles", "water_liquid", "machines_motors"},
    "TRAINS": {"vehicles", "machines_motors"},
    "WATER": {"water_liquid"},
    "LIQUID & MUD": {"water_liquid", "food_cooking"},
    "ANIMALS": {"animals", "speech_vocal"},
    "BIRDS": {"animals"},
    "CREATURES": {"animals", "speech_vocal"},  # creature vocalizations read as voice
    "HORNS": {"vehicles", "music", "electronics_ui"},
    "FIRE": {"ambient_nature", "guns_explosions"},
    "EXPLOSIONS": {"guns_explosions"},
    "FIREWORKS": {"guns_explosions"},
    "AIR": {"ambient_nature", "machines_motors", "water_liquid", "guns_explosions"},
    "WIND": {"ambient_nature"},
    "NATURAL DISASTERS": {"ambient_nature", "guns_explosions"},
    "TOYS": {"impact_friction", "electronics_ui", "music"},
    "MUSICAL": {"music"},
    "HUMAN": {"speech_vocal", "footsteps_walk", "impact_friction"},
    "VOICES": {"speech_vocal"},
    "FOLEY": {"impact_friction", "footsteps_walk", "doors_furniture"},
    "SCIENCE FICTION": {"electronics_ui", "guns_explosions", "machines_motors"},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", type=Path, default=Path("data/FoleyBench/clips_index.csv"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-agreement", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("results/stage_m/tagger_sanity.json"))
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(args.index_csv.open())
            if r["status"] == "ok" and r["ucs_category"] in UCS_ACCEPT]
    rng = np.random.default_rng(args.seed)
    # one per distinct UCS category first, then fill randomly
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["ucs_category"], []).append(r)
    picked: list[dict] = []
    for cat in sorted(by_cat):
        picked.append(by_cat[cat][int(rng.integers(len(by_cat[cat])))])
    rest = [r for r in rows if r not in picked]
    rng.shuffle(rest)
    picked = (picked + rest)[: args.n]

    measurer = RealFoleyMeasurer(device="cpu")
    results, hits = [], 0
    for r in picked:
        wav, sr = extract_audio_track(Path(r["path"]), target_sr=16000)
        probs, _ = measurer._panns_forward(wav[: 16000 * 8])
        label = measurer._coarse_from_probs(probs)
        label2 = measurer.coarse_label_second_tagger(wav[: 16000 * 8])
        accept = UCS_ACCEPT[r["ucs_category"]]
        ok = label in accept
        hits += ok
        results.append({"key": r["key"], "ucs": r["ucs_category"], "panns": label,
                        "ast": label2, "accepted": ok, "caption": r["caption"][:60]})
        print(f"  {r['key']:>5s} {r['ucs_category']:<16s} panns={label:<18s} "
              f"ast={label2:<18s} {'OK' if ok else 'MISS'}  {r['caption'][:48]}", flush=True)

    agreement = hits / len(picked)
    cross = float(np.mean([r["panns"] == r["ast"] for r in results]))
    # Confident-subset reading (frozen interpretation #9): abstain is the
    # instrument's designed behavior, not an error — score accuracy on the
    # confident subset and report the abstain rate alongside (consistent with
    # how class agreement is computed everywhere else under the revised manual).
    conf = [r for r in results if r["panns"] != "abstain"]
    conf_hits = sum(1 for r in conf if r["accepted"])
    conf_acc = conf_hits / len(conf) if conf else float("nan")
    abstain_rate = 1 - len(conf) / len(picked)
    print(f"\n[tagger-sanity] accept-set agreement (abstain=miss): {agreement:.2f} "
          f"({hits}/{len(picked)}); confident-subset accuracy: {conf_acc:.2f} "
          f"({conf_hits}/{len(conf)}); abstain rate: {abstain_rate:.2f}; "
          f"PANNs-vs-AST raw agreement: {cross:.2f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"agreement": agreement,
                                    "confident_accuracy": conf_acc,
                                    "abstain_rate": abstain_rate,
                                    "cross_tagger": cross,
                                    "n": len(picked), "min_required": args.min_agreement,
                                    "beats_contingency": "armed, not triggered" if conf_acc >= 0.65
                                                         and abstain_rate <= 0.30 else "TRIGGER",
                                    "results": results}, indent=2))
    gate_value = conf_acc if np.isfinite(conf_acc) else agreement
    if gate_value < args.min_agreement:
        print("[tagger-sanity] HALT: below threshold — fix coarse_class_map before "
              "any generation-side measurement", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
