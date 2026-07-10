# Proposal Stability

After STOP A, the approved proposal and its operational experiment plan are frozen by
default. For project `foley-cw` (SoundDecisions), the frozen artifacts are the two
human-authored files in `refine-logs/`.

## STOP A Record

- STOP A date: 2026-06-09
- Frozen proposal path: `refine-logs/FINAL_PROPOSAL_SHORT.md` (the Final Proposal)
- Frozen experiment-plan path: `refine-logs/EXPERIMENT_PLAN.md` (the operational contract)
- Proposal status at freeze: PROPOSAL_READY
- Human decision: Approved. Freeze both files and backfill the canonical ORBIT packs from
  them so STOP B / experiment execution can proceed. The proposal layer was corrected by
  the user on this date (the short proposal had been an accidental duplicate of the plan;
  it is now a distinct Final Proposal).

## Canonical Bindings (generated from the frozen sources — not the source of truth)

- `proposal/proposal_pack.json`  ← faithful backfill of `refine-logs/FINAL_PROPOSAL_SHORT.md`
- `experiment/experiment_pack.json` ← faithful backfill of `refine-logs/EXPERIMENT_PLAN.md`

These packs are a structured index of the frozen Markdown; the frozen Markdown remains the
authoritative text. No content was invented and no proposal/plan skill was rerun.

## Allowed Reasons To Reopen

- `/novelty-check` classifies new prior work as `STRONG_BLOCKER`.
- A formal diagnostic recorded in `RESEARCH_DECISION_LOG.md` shows that a central
  paper-breaking hypothesis is false (e.g. `STOP_ADSR` / `STOP_PROJECT`).
- The user explicitly asks to revise the proposal or the plan.

## Related / Concurrent Work Handling

- Ordinary related work goes to `orbit-research/CONCURRENT_WORK_WATCHLIST.md`.
- Recent concurrent work goes to the watchlist by default.
- Positioning updates may be recorded in related-work notes or downstream paper artifacts.
- Do not rewrite the frozen proposal/plan for normal related work.

## Human Decisions

| Date       | Trigger                              | Decision                                              | Reopened? | Notes |
|------------|--------------------------------------|-------------------------------------------------------|-----------|-------|
| 2026-06-09 | STOP A approval + canonical backfill | Freeze proposal + experiment plan; backfill packs     | NO        | Soft (record-only) freeze; files left at 0644, byte-unchanged. |
