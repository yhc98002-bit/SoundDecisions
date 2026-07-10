# Stage-M Micro-Map Report (manual section 2)

Stage-M outputs are engineering diagnostics, never scientific evidence.

| # | Criterion | Pass | Detail |
|---|---|---|---|
| 1 | endpoints | FAIL | cfg=1 class: A_fork(0.9)=0.625 < 0.9; cfg=4.5 class: A_fork(0.9)=0.500 < 0.9 |
| 2 | monotonicity | PASS | commit(s) non-decreasing within CI tolerance |
| 3 | kernel_cfg4.5 | FAIL | SDE re-validation at cfg=4.5: OK; Gate-A: GATE_A_UNDERPOWERED (median MMD2=nan, frac_sig=nan) — theta_mmd 0.472 not separated from cross-clip MMD median 0.508 — fix bandwidth/embedding, never pass silently |
| 4 | measurer_determinism | PASS | determinism == 1.0 on identical wavs for all axes |
| 5 | informativeness | PASS | 7/16 clips with A_independent > 0.9 on 'class' at cfg=4.5 |

**Tokens:** MICROMAP_FAIL(endpoints,kernel_cfg4.5), GATE_A_UNDERPOWERED


**Failure routing:** late-endpoint failure -> suspect terminal-time SDE numerics (check substep density between s=0.90 and s=1) | kernel failure -> manual 1.2 fallback: headline at cfg=1.0 + readout-transfer check at deployed cfg + scope note
