# Gate-A Report — seed-marginalized exchangeability (revised manual 1.2)

Feature space: sqrt 527-dim tagger-prob vectors + extended-alphabet label TV.
Null: 32 cfg=1 cells; per-s theta_mmd={'0.05': 0.0723, '0.9': 0.0556}, theta_tv={'0.05': 0.5938, '0.9': 0.5312}

## Internal null @ cfg=1 (HARD): **CFG_KERNEL_OK(cfg=1, schedule=sqrt_down)**
Guards: {"power_reject_frac": 0.95, "cross_clip_mmd_median": 0.3216, "cross_clip_mmd_p95": 0.5679, "null_ks_p": 0.1313}
Per-s: {"0.05": {"n_cells": 16, "n_low_p": 1, "cap": 2, "ok": true, "median_mmd2": 0.013181051205795491, "median_tv": 0.25}, "0.9": {"n_cells": 16, "n_low_p": 1, "cap": 2, "ok": true, "median_mmd2": -0.009796137142934258, "median_tv": 0.3125}}
internal null consistent with an exact kernel

## Adjudicated @ cfg=4.5 (non-gating): **CFG_KERNEL_OK(cfg=4.5, schedule=sqrt_down)**
Per-s: {"0.05": {"checks": {"mmd:mw": true, "mmd:exceedance": true, "mmd:gross": true, "tv:mw": true, "tv:exceedance": true, "tv:gross": true}, "stats": {"mmd_mw_p": 0.5075174065359023, "mmd_n_exceed": 2, "mmd_worst": 0.0799732481888642, "tv_mw_p": 0.6380709425776626, "tv_n_exceed": 2, "tv_worst": 0.625}, "n_cells": 16, "ok": true}, "0.9": {"checks": {"mmd:mw": true, "mmd:exceedance": true, "mmd:gross": true, "tv:mw": true, "tv:exceedance": true, "tv:gross": true}, "stats": {"mmd_mw_p": 0.332353907751532, "mmd_n_exceed": 1, "mmd_worst": 0.0852848983468879, "tv_mw_p": 0.8323051327532374, "tv_n_exceed": 0, "tv_worst": 0.375}, "n_cells": 16, "ok": true}}
exchangeable with the cfg=1.0 null at all test points

