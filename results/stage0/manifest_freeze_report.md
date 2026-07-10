# Phase-1 Manifest Freeze Report — §3.1 (FROZEN)

Selected **200 single-event + 60 two-event = 260 clips**; 60/40 split = 161 train / 99 eval. timing_bin_s = 1.5147 (anchor: approved_chain(metadata>visual>light_human)). class = DIAGNOSTIC (kept in maps, not gating).

## Per-axis usable / non-pinned n (headroom stratification, all clips)
| axis | usable n | non-pinned n | pinned n | unscorable | mean A_ind |
|---|---|---|---|---|---|
| presence | 94 | 94 | 106 | 60 | 0.811 |
| timing | 56 | 56 | 144 | 60 | 0.899 |
| class | 167 | 167 | 32 | 61 | 0.428 |
| material | 197 | 197 | 3 | 60 | 0.647 |

Headroom note (§3.1): material & class carry the seed/trajectory-share story and have abundant non-pinned headroom; timing is the most conditioning-pinned axis (expected). cfg-specific video-pinned exclusions recorded per axis in the manifest.

Pinned exclusions intersected with manifest: cfg=1.0 {'presence': 106, 'timing': 144, 'class': 32, 'material': 3}.
