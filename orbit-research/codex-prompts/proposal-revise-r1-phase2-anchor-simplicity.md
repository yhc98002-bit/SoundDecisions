You are an independent reviewer checking a research-proposal revision. Do NOT assume the revision is correct; your job is to catch anchor drift, complexity creep, and cross-file inconsistency. You have file access — read these four files before answering:

1. PRE-REVISION proposal (the immutable problem anchor lives in its sections 1–2: one-sentence thesis + core scientific question):
   /XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions/refine-logs/FINAL_PROPOSAL_SHORT.md.prerevise.1
2. The PI-review critique that drove the revision (binding feedback):
   /XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions/critic.md
3. REVISED proposal:
   /XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions/refine-logs/FINAL_PROPOSAL_SHORT.md
4. REVISED experiment plan:
   /XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions/refine-logs/EXPERIMENT_PLAN.md
   (pre-revision plan for diff context: /XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions/refine-logs/EXPERIMENT_PLAN.md.prerevise.1)

## Check 1 — Anchor check

Problem anchor (immutable): "For a video-conditioned audio flow model, when does the generator commit to each Foley output axis, and when can that commitment be read out by available probes?" — with the inference policy strictly downstream of that map.

Question: does the revised pair of documents, taken together, still solve this anchored problem? Specifically:
1. Is the dominant contribution still aimed at the anchored problem, or did the revisions (e.g., internal readout C3, causal intervention C4, coupling ablation, SMC-ITA positioning) drift to a different problem?
2. Did any "factual" assumption become a "working" assumption without a corresponding plan to test it?
3. Did the mechanism family change in a way that no longer addresses the named bottleneck (when are decisions made / when are they readable)?

Return on its own line exactly one of: ANCHOR_PASS | ANCHOR_DRIFT | ANCHOR_AMBIGUOUS
Then a one-paragraph rationale per critique group (A1–A12 proposal items; B2–B16 plan items). If DRIFT, name the causing critique ID(s).

## Check 2 — Simplicity check

Constants: MAX_NEW_TRAINABLE_COMPONENTS = 2; MAX_PRIMARY_CLAIMS = 2; "smallest adequate mechanism wins."

1. Does the revised method exceed 2 new trainable components? (Count: linear probes on cached internal features; the conditional Phase-5 cheap process verifier; anything else you find.)
2. Does the claim count exceed 2 primary claims? (Contributions C1–C5 exist — judge whether the document still has ≤2 PRIMARY claims with the rest clearly subordinate, or whether it now asserts 5 co-equal headline claims.)
3. Did revisions add a mechanism that could be removed without losing the anchored claim?

Return on its own line exactly one of: SIMPLICITY_PASS | SIMPLICITY_VIOLATION
Then list specific violations, each with the critique ID that caused it.

## Check 3 — Consistency audit (critique Part C)

Verify across the two REVISED files and report any failures concretely (quote the offending line):
- terminology: `s`, `s_commit`, `s_read`, `x_s`, `x0(s)` used consistently; no raw `t_commit`/`t_read` as reported variables; "label-free" never used without "measurement-dependent" qualification;
- claim discipline: no "will show / demonstrates / proves / beats / is superior" for untested claims;
- file lists: every report named in plan phase text appears in the Files-to-Produce table and vice versa;
- decision tokens: every token used in phase text / launch commands / proposal Next Gate is defined somewhere, and no old token was silently dropped;
- the two files agree on: Phase 0A semantics (sanity only), internal logging mandatory vs analysis non-blocking, A_independent cap, α-robust ordering in GO_MAP, SMC-ITA baseline requirement, correctness factorization, policy demoted to downstream scheduler.

End with a short prioritized list of concrete fixes (if any). Plain text only.
