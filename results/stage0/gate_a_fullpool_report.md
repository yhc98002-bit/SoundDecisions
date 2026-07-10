# Full-pool Gate-A — cfg=4.5 ratification (§1.2/§15.8) — RULING-1 scaled caps

cfg=1.0 cells 200, cfg=4.5 cells 200; n_perm=1000; Gate-A s=(0.05, 0.9). **Scaled cap = Binomial(n,0.05) 95th pct = 15** (recovers the frozen 2 at n=16; RULING-1 bug-fix correcting the exposure n, not a re-tune).

- **cfg=1.0 internal null: `CFG_KERNEL_OK(cfg=1, schedule=sqrt_down)` passed=True** (low-p cap 15)  internal null consistent with an exact kernel
- **cfg=4.5 calibrated: `CFG_KERNEL_OK(cfg=4.5, schedule=sqrt_down)` passed=True** (MMD exceedance worst 15 vs cap 15 → with CAVEAT)  exchangeable with the cfg=1.0 null at all test points

guards: power_reject_frac=0.983, cross_clip_mmd_median=0.3105, null_ks_p=0.483

**Verdict: RATIFIED** (ledger updated: True). cfg=4.5 carries the caveat: *near-exchangeable on tagger-probs, not provably exact* on every cfg=4.5 claim.

