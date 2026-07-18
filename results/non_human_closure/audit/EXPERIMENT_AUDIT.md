# Non-human closure experiment-integrity audit

Review verdict: `WARN`. There is no blocking data-integrity or split-leakage
finding in the completed read-only audits. The warning is material: the
requested independent Claude/Opus review could not authenticate, and several
exploratory inference/provenance limitations remain.

## Cross-model audit process

The exact path-only checklist was launched with:

```text
claude -p --dangerously-skip-permissions --output-format json --model opus --effort max
```

The CLI returned `Not logged in · Please run /login`, with zero input tokens,
zero output tokens, and zero cost. Its status is `ENGINEERING_FAILURE`. No
external-model review verdict exists, and this file does not relabel the
failure as a PASS. The prompt, raw JSON, and debug trace are retained beside
this report.

## Completed independent checks

Three read-only audits separately reconstructed Class commitment/variance,
readout predictions/metrics/splits, and deliverable cardinalities/hashes.

- Class: the registered all-cell pooled sustained crossing at theta 0.70 is
  `s=0.90`; 3,767/5,000 bootstrap draws cross and 1,233 do not. All variance
  components and intervals match the canonical JSON.
- Readout: all 113,212 prediction keys are unique; all 48 videos remain in one
  outer fold; 1,056 inner selections have no group overlap; all 176 metric
  cells were independently reconstructed. Pooled MLP clears conditioning only
  at `s=0.45` and `s=0.60`, never external preview; no action cell qualifies.
- Deliverables: core reports, 6,528 feature-manifest units, candidate
  predictions, materialization hashes, and protocol-transition disclosure are
  present and internally consistent.

## Corrections driven by audit

- The report now separates the historical crossers-only unsustained individual
  mean from the frozen all-cell pooled sustained replication estimand.
- A create-only post-hoc sensitivity documents `s=0.60` after excluding the
  nine video-determined cases; it does not change the registered result.
- Bootstrap crossing percentiles are explicitly conditional on draws that
  cross, not an unconditional CI.
- The seed boundary estimate, fixed finite-baseline uncertainty, proxy targets,
  incomplete target coverage, pointwise multiple comparisons, nonsustained
  MLP support, convergence warnings, and external-baseline mismatch are
  explicit.
- Determinism tests now repeat both stochastic probe families and compare exact
  probability hashes.

## Residual risks

Cross-model review independence is unavailable without CLI authentication.
Measurer repeatability is unidentified; the video baseline's finite-reference
uncertainty is held fixed; readout comparisons are exploratory and pointwise;
and readout shard completions lack executable source/git hashes. Material lacks
sufficient metadata-matched coverage, so there is no 2AFC performance result.

These limitations do not convert an engineering gap into a scientific
negative. They constrain the evidence exactly as stated in the integrated PI
report.
