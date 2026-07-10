# Null-Result Contract

> Generated view of `experiment/experiment_pack.json` (`null_result_contract`), faithfully
> transcribed from the frozen `refine-logs/EXPERIMENT_PLAN.md` (§3, §7) and
> `refine-logs/FINAL_PROPOSAL_SHORT.md` (§7). The pack is the source of truth.

The maps target the model's own **self-target** (NOT correctness-vs-video) and are
correctness-/human-label-free but **measurement-dependent**. The scientific make-or-break is
**window separation + early readability** (`GO_MAP` + `GO_READOUT`), both measurable
**without correctness labels**.

## Outcome framing (pre-committed)

- **METHOD** — separated commitment windows + early readability + axis-gated pruning (or
  rollback) improves fixed-budget Foley correctness over the strongest baseline beyond CIs.
- **DIAGNOSTIC (strong)** — separated commitment windows, but cheap probes lag far behind
  commitment (R2-dominated). Publish the commitment–readout gap + probe-limitation analysis;
  motivate internal probes. **Genuinely publishable.**
- **NEGATIVE (publishable) = `STOP_ADSR`** — all `s_commit` coincide or only near `s=1`:
  cross-modal Foley correctness has no useful window separation in the tested model and
  method novelty collapses to scalar rejection. Honest diagnostic, route to diagnostic
  framing. **Do not force a method claim.**
- **`STOP_PROJECT`** — no usable trajectory access, no reliable axis measurement, or
  tail-forking is not meaningful.

## Standing rules

- **Novelty boundary:** if axes do not separate, route to DIAGNOSTIC / NEGATIVE; never force
  a method claim. Prefer early stop. Do not infer missing results. Do not upgrade claim tier
  without gate evidence.
- The floor (DIAGNOSTIC / NEGATIVE) is genuinely good; a failed first policy (Phase 4) does
  not kill a good map.
- **Internal probes are non-blocking.** Until they run, report "gap under available external
  probes", never "irreducible uncommitted information".
