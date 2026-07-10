#!/usr/bin/env python
"""Build the FROZEN AudioSet-527 -> coarse Foley class map (manual section 3.1).

Maps each AudioSet display name to one of ~12 coarse Foley classes via ordered
keyword rules (first match wins), informed by the FoleyBench ucs_category
distribution (TOOLS, FOOTSTEPS, FOOD & DRINK, OBJECTS, GUNS, METAL, MECHANICAL,
USER INTERFACE, SPORTS, COMMUNICATIONS, DOORS, COMPUTERS, WOOD, WEAPONS, FIGHT).
Unmatched labels fall into 'other'. Silence/noise/hum labels are listed as
non_event_indices (excluded from the presence eventness gate).

Output: configs/coarse_class_map.json — versioned; once Stage M consumes it,
the file is frozen (changing it invalidates Stage-M/Stage-0 measurements).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Ordered: first matching rule wins. Keywords are lowercase substring matches.
RULES: list[tuple[str, list[str]]] = [
    ("speech_vocal", ["speech", "shout", "yell", "whisper", "conversation", "narration",
                      "babbling", "laughter", "giggle", "crying", "sobbing", "screaming",
                      "singing", "choir", "rapping", "humming", "male", "female", "child",
                      "snoring", "gasp", "cough", "sneeze", "burp", "hiccup", "chatter",
                      "whoop", "battle cry", "grunt", "groan", "whistling", "beatboxing",
                      "yodeling", "chant", "mantra", "crowd", "cheering", "applause",
                      "chuckle", "snicker", "belly laugh", "wail", "moan", "sigh",
                      "throat clearing", "breathing", "pant", "snort", "sniff"]),
    ("music", ["music", "musical", "instrument", "guitar", "piano", "drum", "violin",
               "cello", "flute", "saxophone", "trumpet", "organ", "synthesizer", "banjo",
               "sitar", "mandolin", "ukulele", "keyboard (musical)", "orchestra", "bass",
               "harmonica", "accordion", "bagpipes", "didgeridoo", "shofar", "theremin",
               "singing bowl", "tubular bells", "marimba", "xylophone", "vibraphone",
               "steelpan", "harp", "bell", "chime", "harpsichord", "mellotron", "scratching",
               "song", "soundtrack", "lullaby", "jingle", "tender", "exciting", "angry",
               "sad", "scary", "wedding", "christmas", "dance", "electronic", "ambient",
               "chant", "a capella", "beat", "theme", "video game", "background"]),
    ("animals", ["animal", "dog", "bark", "howl", "cat", "meow", "purr", "bird", "chirp",
                 "tweet", "crow", "owl", "duck", "goose", "chicken", "rooster", "turkey",
                 "pig", "oink", "cow", "moo", "horse", "neigh", "sheep", "goat", "bleat",
                 "frog", "croak", "snake", "rattle", "insect", "cricket", "mosquito", "fly",
                 "bee", "wasp", "whale", "roar", "growl", "livestock", "pets",
                 "canidae", "rodents", "mouse", "patter", "wild animals"]),
    ("water_liquid", ["water", "rain", "stream", "waterfall", "ocean", "waves", "splash",
                      "drip", "pour", "fill", "gush", "trickle", "squish", "liquid",
                      "bathtub", "sink (filling", "toilet flush", "gargling", "boiling"]),
    ("guns_explosions", ["gunshot", "gunfire", "machine gun", "fusillade", "artillery",
                         "cap gun", "firecracker", "fireworks", "explosion", "eruption",
                         "boom", "bang", "burst", "pop"]),
    ("footsteps_walk", ["walk", "footsteps", "run", "shuffle"]),
    ("doors_furniture", ["door", "doorbell", "knock", "cupboard", "drawer", "squeak",
                         "creak", "slam", "sliding"]),
    ("electronics_ui", ["beep", "bleep", "computer keyboard", "typing", "typewriter",
                        "telephone", "ringtone", "dial tone", "busy signal", "alarm",
                        "siren", "buzzer", "smoke detector", "fire alarm", "foghorn",
                        "whistle", "kettle whistle", "steam whistle", "clock", "tick",
                        "tick-tock", "electronic", "camera", "single-lens reflex",
                        "printer", "scanner", "static", "radio", "television",
                        "white noise", "pink noise", "sine wave", "chirp tone",
                        "sound effect", "electric shaver", "electric toothbrush"]),
    ("vehicles", ["car", "vehicle", "truck", "bus", "motorcycle", "train", "rail",
                  "subway", "aircraft", "airplane", "helicopter", "jet", "boat", "ship",
                  "engine", "motor vehicle", "traffic", "horn", "honk", "skidding",
                  "tire squeal", "race car", "propeller", "lawn mower", "chainsaw",
                  "ice cream truck", "bicycle", "skateboard", "accelerating", "revving"]),
    ("machines_motors", ["machine", "motor", "mechanical fan", "air conditioning", "pump",
                         "compressor", "generator", "drill", "jackhammer", "power tool",
                         "sewing machine", "blender", "food processor", "mixer",
                         "vacuum cleaner", "hair dryer", "washing machine", "dishwasher",
                         "mechanisms", "ratchet", "pulleys", "gears", "clockwork",
                         "air brake", "hydraulic", "pneumatic", "idling"]),
    ("tools_hand", ["tool", "hammer", "saw", "sawing", "filing", "rasp", "sanding",
                    "chisel", "wrench", "screwdriver", "axe", "chopping (wood)",
                    "wood block", "carving"]),
    ("food_cooking", ["chopping (food)", "frying", "sizzle", "cutlery", "silverware",
                      "dishes", "pots", "pans", "chewing", "mastication", "biting",
                      "crunch", "slurp", "stir", "microwave oven", "kettle", "cooking",
                      "food", "drink", "eating", "swallowing"]),
    ("impact_friction", ["thump", "thud", "bang", "smash", "crash", "breaking", "shatter",
                         "glass", "clang", "clank", "clatter", "clink", "jingle (metal)",
                         "tap", "knock (impact)", "slap", "smack", "whack", "thwack",
                         "basketball bounce", "bouncing", "drop", "scrape", "scratch",
                         "rub", "friction", "grind", "crushing", "crumpling", "crinkling",
                         "tearing", "rip", "zipper", "velcro", "snap", "crackle",
                         "rustle", "rustling", "flap", "whip", "whoosh", "swoosh",
                         "thunk", "plop", "splat", "squeal", "screech", "chink", "clack",
                         "coins", "keys jangling", "shuffling cards", "writing",
                         "eraser", "page turn", "paper", "plastic", "metal", "wood",
                         "rolling", "wobble", "vibration", "rumble", "roll"]),
    ("ambient_nature", ["wind", "thunder", "thunderstorm", "fire", "crackling fire",
                        "environment", "outside", "inside", "field recording", "silence",
                        "room tone", "echo", "reverberation", "hum", "mains hum", "buzz",
                        "noise", "throbbing", "heartbeat", "heart murmur", "hiss",
                        "steam"]),
]

NON_EVENT_KEYWORDS = ["silence", "white noise", "pink noise", "static", "hum", "mains hum",
                      "noise", "room tone", "echo", "reverberation", "inside, small room",
                      "inside, large room", "inside, public space", "outside, urban",
                      "outside, rural", "field recording"]

COARSE_CLASSES = [r[0] for r in RULES] + ["other"]


def classify(name: str) -> str:
    low = name.lower()
    for coarse, keys in RULES:
        if any(k in low for k in keys):
            return coarse
    return "other"


def main() -> int:
    labels_csv = REPO / "weights" / "measurers" / "class_labels_indices.csv"
    out_path = REPO / "configs" / "coarse_class_map.json"
    rows = list(csv.DictReader(labels_csv.open()))
    assert len(rows) == 527, f"expected 527 AudioSet classes, got {len(rows)}"

    index_to_coarse, non_event = {}, []
    for r in rows:
        idx, name = int(r["index"]), r["display_name"]
        index_to_coarse[idx] = classify(name)
        if any(k in name.lower() for k in NON_EVENT_KEYWORDS):
            non_event.append(idx)

    counts = Counter(index_to_coarse.values())
    print("coarse-class sizes:")
    for c in COARSE_CLASSES:
        print(f"  {c:18s} {counts.get(c, 0):4d}")
    print(f"non_event indices: {len(non_event)}")
    print("\nsample of 'other':",
          [r["display_name"] for r in rows if classify(r["display_name"]) == "other"][:15])

    payload = {
        "version": "v2-2026-06-11",
        "source": "AudioSet class_labels_indices.csv (527) -> ordered keyword rules; "
                  "first match wins; informed by FoleyBench ucs_category distribution",
        "frozen": True,
        "coarse_classes": COARSE_CLASSES,
        "index_to_coarse": {str(k): v for k, v in sorted(index_to_coarse.items())},
        "non_event_indices": sorted(non_event),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
