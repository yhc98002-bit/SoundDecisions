ROLE: Semantic plan-code audit reviewer (STOP B). You are an independent cross-model
reviewer. Be skeptical and specific. Do NOT improve the code; JUDGE whether it faithfully
implements the planned method.

REPO: /XYFS02/HDD_POOL/paratera_xy/pxy1289/HaocunYe/Research/SoundDecisions

CONTEXT: This is the `foley-cw` project ("When Are Foley Decisions Made? Commitment and
Readout Windows in V2A Flow Generation"). This is the STOP-B `audit-only` implementation:
planned code only, validated on CPU against an analytic synthetic backend. MMAudio (the real
V2A flow model) and GPU are intentionally NOT available; the MMAudio adapter is a documented
stub to be wired in a later Phase-0 GPU diagnostic.

AUTHORITATIVE SPEC (read these first):
- refine-logs/EXPERIMENT_PLAN.md   (the frozen operational plan — the contract)
- experiment/experiment_pack.json  (structured pack: algorithmic_formalization, controls,
  null_result_contract, decision_tree, formal_diagnostics)
- experiment/EXPERIMENT_PLAN_EXEC.md

IMPLEMENTATION TO AUDIT (foley_cw/):
- types.py, time_map.py, model_adapter.py, synthetic_backend.py, score_sde.py, config.py
- agreement.py, axes.py, stats.py, dataset.py, probes.py, validation.py, reporting.py
- commitment.py, readout.py, reliability.py, gap.py
- cli/phase0_feasibility.py, cli/phases123_maps.py
- configs/*.json, tests/ (skim for what is actually asserted)

JUDGE these plan-fidelity questions specifically (cite file:line for every finding):

1. velocity->score conversion (the plan's stated HIGHEST silent-bug risk). The plan gives
   score = (t*v - x)/(1-t) for the rectified-linear convention and says to AUDIT sign/
   direction. Is score_sde.score_from_velocity correct and consistent with
   synthetic_backend's analytic score? Is the alpha=0 path the deterministic ODE, and is the
   marginal-preserving SDE drift the correct v + 0.5*sigma^2*score (not a different
   coefficient)? Is the MMAudio convention left UNVERIFIED rather than silently assumed?

2. Time convention: is s in [0,1] the only public axis, with s<->t isolated in time_map and
   MMAudio's direction marked unverified? Any place that uses raw t without the seam?

3. Commitment (commitment.py): is commitment ALWAYS normalized as gain over A_independent
   (never raw fork agreement)? Is A_independent computed from N independent full alpha=0
   generations (independent noise)? Is the smallest-valid-alpha rule (diversity AND
   audio-validity guard) implemented, returning FORK_ALPHA_NO_VALID_OPERATING_POINT when no
   alpha qualifies? Is Restart re-noising correctly ABSENT from the commitment kernel?

4. Readout (readout.py): both ODE-target and fork-majority targets? Readout uses the blurry
   x0(s) (Tweedie), commitment uses clean final completions? Probe ladder present with
   CLAP/SyncNet/ImageBind marked LEGACY and MLLM-on-preview as headline; heavy probes are
   stubs, not fabricated?

5. Self-target vs correctness: do the maps target the model's OWN final axis value (taggers/
   onset on generated audio), NOT human/MLLM correctness-vs-video? Correctness must only
   enter later (Phase 4), not in the maps.

6. Statistics (stats.py): bootstrap UNIT = video (resample videos, not measurements)? CIs on
   s_commit/s_read/gap over videos? Pre-registered thresholds frozen BEFORE headline maps
   (check cli/phases123_maps writes thresholds to go_no_go_decision.md before the maps)?
   Underpowered axes flagged, not reported as results?

7. Reliability gate (reliability.py): three parts (determinism + robustness + validity), with
   demotion (material/fine demoted unless strong on all three)?

8. Decision tokens (gap.py): decide_phase0 emits GO_MAPS_PHASE only when trajectory OK + SDE
   validated + manifest OK + >=3 reliable axes. decide_phase3 emits the right tokens and they
   are NOT self-contradictory (e.g. STOP_ADSR and GO_MAP must be mutually exclusive). Verify
   this mutual exclusivity actually holds in code.

9. Honesty of the MMAudio seam: model_adapter.MMAudioBackend must RAISE (not fabricate) and
   the synthetic dry-run must not be presented as a real result. No correctness-vs-another-
   model-as-ground-truth anywhere.

10. Any place where the code silently diverges from the plan, or a planned artifact/column is
    missing from the CSV/MD outputs.

OUTPUT FORMAT (exactly):
VERDICT: <MATCHES_PLAN | PARTIAL_MISMATCH | CRITICAL_MISMATCH | ERROR>
Then a numbered list of findings (each: file:line, what the plan requires, what the code does,
severity critical/major/minor). Then a short "what is correct" section so the verdict is
balanced. MATCHES_PLAN = faithful with at most minor notes; PARTIAL_MISMATCH = scoped issues
that don't block the synthetic-validated path; CRITICAL_MISMATCH = a real method/control/metric
divergence that would invalidate the planned diagnostic; ERROR = could not audit.
