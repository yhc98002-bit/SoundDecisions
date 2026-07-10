# Stage-M Micro-Map Report (revised manual section 2)

Stage-M outputs are engineering diagnostics, never scientific evidence.
Pass criteria evaluated at the headline cfg=1; the cfg=4.5 arm is adjudicated, not gating.

| # | Criterion | Pass | Detail |
|---|---|---|---|
| 1 | endpoints | PASS | label endpoints (confident subset) + embedding seed floor/growth OK |
| 2 | monotonicity | PASS | commit(s) non-decreasing within CI tolerance (headline cfg) |
| 3 | kernel_headline | PASS | Gate-A internal null @ cfg=1: CFG_KERNEL_OK(cfg=1, schedule=sqrt_down) | adjudicated (non-gating) @ cfg=4.5: CFG_KERNEL_OK(cfg=4.5, schedule=sqrt_down) |
| 4 | measurer_determinism | PASS | determinism == 1.0 on identical wavs (extended alphabet) |
| 5 | informativeness | PASS | 2/16 video-pinned on 'class' @ cfg=1; abstain@s=0.9 = 0.21 (cap 0.3) |

**Tokens:** MICROMAP_PASS, CFG_KERNEL_OK(cfg=1, schedule=sqrt_down), CFG_KERNEL_OK(cfg=4.5, schedule=sqrt_down)

