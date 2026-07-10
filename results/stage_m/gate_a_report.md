# Gate-A Report — fork-kernel distributional validity (manual section 1.2)

Reference (cfg=1.0): 64 cells -> theta_mmd=0.472, fp_ref=1.000, per-axis theta_marginal={'class': 0.8562, 'presence': 0.3562}

Guards: power_reject_frac=1.00 (need >= 0.80), cross-clip MMD median=0.5081, null KS p=0.977 (need >= 0.01)

## Verdict at cfg=4.5: **GATE_A_UNDERPOWERED**

- median MMD^2 = nan (threshold 0.472)
- fraction significant = nan (cap 2.000)
- worst per-axis marginal mismatch = {}

theta_mmd 0.472 not separated from cross-clip MMD median 0.508 — fix bandwidth/embedding, never pass silently

