# Algorithmic Formalization

> Generated view of `experiment/experiment_pack.json` (`algorithmic_formalization`),
> faithfully transcribed from the frozen `refine-logs/EXPERIMENT_PLAN.md` (§2, Phase 0.2,
> Phase 1, Phase 2). The pack is the source of truth.

## Time convention

Generation progress `s ∈ [0,1]` is the canonical axis for **all** reported windows (`s=0`
noise, `s=1` audio). Map `s ↔ t` to MMAudio's actual integration time **once**, in audited
code (some flow models integrate `t: 1→0`). `x_s` = intermediate state; `x0(s)` = Tweedie
best-guess of the final audio (the readout probe input).

## Commitment fork kernel (marginal-preserving SDE; α is the only knob)

```text
fork_tail(x_s, s, alpha, K):              # integrate progress s -> 1 (final audio)
  comps = []
  for k in 1..K:
    x = x_s
    for (s_i -> s_next) in progress_schedule(s -> 1):
      t_i   = s_to_t(s_i)                  # audited mapping (Phase 0.1)
      v     = v_theta(x, t_i, video_cond)
      score = score_from_velocity(v, x, t_i)   # VALIDATED in Phase 0.2
      sigma = alpha * g(s_i)               # alpha=0 => deterministic ODE
      x     = step_euler_maruyama(x, v, score, sigma, ds)
    comps.append(decode(x))               # final audio
  return comps
```

A common rectified-flow score form is `score(x,t) ≈ (t·v − x)/(1−t)` (equivalently
`∇log p_t = −(x_t + (1−t)v)/t`) — **audit sign and direction against MMAudio's code; do not
copy blindly.** Reserve Restart re-noising for Phase 6 rollback; do **not** use it for
commitment.

## α selection

Predefined pilot grid. Primary operating `α` = the **smallest** α producing measurable tail
diversity while preserving valid generated audio (audio-validity guard = presence/quality
measure on forks). Report the full `A(axis, s, α)` surface as secondary; use the single
primary α for the headline `s_commit`. If no α works → `FORK_ALPHA_NO_VALID_OPERATING_POINT`
(route to a different kernel or `GO_DIAGNOSTIC`; do not push α up and call broken audio
"uncommitted").

## Normalized commitment gain

```text
commit(s, axis) = clip( (A_fork(x_s,axis,α) − A_independent(video,axis))
                        / (1 − A_independent(video,axis)), 0, 1 )
s_commit(axis)  = min s with commit(s,axis) ≥ θ_commit          # bootstrapped over videos
```

Agreement metric per axis: categorical (presence / timing-bin / class / binding) →
exact-match rate or Krippendorff's α across forks; embedding (material) → mean pairwise
cosine.

## Readout

```text
s_read(axis, probe, target) = min s with accuracy/AUROC ≥ θ_read   # bootstrapped over videos
```
for both **ODE-target** and **fork-majority** targets. Commitment uses **clean** final
completions; readout uses **blurry** `x0(s)` — the blur penalty is part of the early gap,
disentangled by R1/R2.

## Highest silent-bug risk

The velocity→score conversion. `α=0` reproducing the ODE is **necessary but does NOT test
the score term** (it is multiplied by 0). The real test is nonzero-α: small-α continuity +
fork audio validity + nontrivial diversity. Emit `FIX_SCORE_CONVERSION` and halt if α=0
fails or small-α continuity is violated.
