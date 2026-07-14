# Experiment Plan

> **STATUS NOTICE** — Historical plan text below includes pending phase labels that have since completed. See [`results/CURRENT_STATUS.md`](../results/CURRENT_STATUS.md) for current status.

> Generated view of `experiment/experiment_pack.json`. Do not treat this Markdown file as the source of truth.

## Status

- Pack status: `ready`
- Updated at: `2026-06-09T00:00:00Z`
- Proposal ref: `proposal/proposal_pack.json`
- Safe next command: `/diagnostic-to-review "experiment/experiment_pack.json"`

## Decision Tree

- phase0-gate: Phase 0 feasibility + reliability gate (STRICT, do first): trajectory access; velocity->score SDE validated at alpha=0 AND nonzero-alpha; dataset+anchor manifest; >=3 axes pass determinism+robustness+validity. Tokens: GO_MAPS_PHASE -> Phase 1; FIX_SCORE_CONVERSION -> halt/fix; NO_TRAJECTORY_ACCESS -> STOP_PROJECT; STOP_PROJECT (<3 reliable axes or no usable anchors).
- phase1-commitment: Phase 1 commitment map (C1a): per-axis s_commit normalized vs A_independent video-prior. Tokens: COMMITMENT_MAP_DONE; FORK_ALPHA_NO_VALID_OPERATING_POINT (route to different kernel or GO_DIAGNOSTIC, not auto-kill).
- phase2-readout: Phase 2 readout map (C1b): per-axis per-probe s_read for ODE-target AND fork-majority targets. Token: READOUT_MAP_DONE.
- phase3-makeorbreak: Phase 3 gap + separation + GO/NO-GO (make-or-break ends here). Emit all that apply: GO_MAP (separated commitment windows beyond CIs); GO_READOUT (a feasible probe reads early axes well before the end); GO_RESTRICTED (only presence/gross-timing actionable); GO_DIAGNOSTIC (commitment exists but cheap readout lags, R2-dominated); STOP_ADSR (windows coincide / only near s=1 -> publishable NEGATIVE); STOP_PROJECT. GO_MAP + GO_READOUT is the correctness-label-free scientific make-or-break and the only gate to Phase 4.
- phase4-policy: Phase 4 axis-gated population pruning (CONDITIONAL; needs correctness sidecar + policy_preregistration.md; only after GO_MAP + GO_READOUT). Tokens: GO_POLICY / GO_RESTRICTED / DIAGNOSTIC_ONLY.
- phases567-conditional: Phase 5 cheap process-aware verifier (only if maps show headroom); Phase 6 axis-gated rollback (only if forking stable); Phase 7 internal-feature probes (only if external-probe gap large; non-blocking).

## Controls

- video-prior-normalization: MANDATORY: normalize commitment against A_independent(video,axis) = self-target agreement across N independent full generations of the same video. Report commitment as gain over this video-conditioned prior; never report raw fork agreement as commitment.
- preregistered-thresholds: Freeze theta_commit, theta_read, theta_rel, theta_robust, theta_cal from pilot/anchor data and record in go_no_go_decision.md BEFORE computing headline maps; report threshold sensitivity (sweep) afterward.
- bootstrap-over-videos: Bootstrap unit = video (resample videos, not individual measurements) for all CIs on s_commit, s_read, and gaps. Declare minimum usable n per axis; underpowered axes reported as such, not as results.
- reliability-gate: Per-axis three-part gate on generated audio before any map: determinism (test-retest) + robustness (event-window shift / loudness-norm / resample / compression / small noise) + validity (vs small human/MLLM calibration sidecar). Demote or drop on failure; material/fine-class is the prime demotion risk.
- same-candidate-pools: All baselines score the SAME candidate pools. Readout-side ladder: energy/onset; CLAP; SyncNet/AV-sync; ImageBind; audio tagger; MLLM-on-preview; (Phase 7) internal-feature probe. Policy-side: full BoN; same-compute BoN; random pruning; scalar DiffRS-style rejection; final-score reranking; seed restart; oracle axis-gated pruning (upper bound). Rollback-side: Restart without axis gating; scalar rejection + rollback; full seed restart.
- legacy-vs-headline-probes: CLAP / SyncNet / ImageBind are LEGACY baselines, not the headline probe; the main probe-side comparison is a strong MLLM-on-preview.

## Null Result Contract

Maps target the model's own self-target (NOT correctness-vs-video) and are correctness-/human-label-free but measurement-dependent. Make-or-break = window separation + early readability (GO_MAP + GO_READOUT), both without correctness labels. Outcomes and how they are framed: METHOD (separated windows + early readability + axis-gated pruning/rollback beats the strongest baseline beyond CIs); DIAGNOSTIC-strong (separated commitment but cheap probes lag far behind, R2-dominated -> publish the commitment-readout gap + probe-limitation analysis, motivate internal probes); NEGATIVE-publishable = STOP_ADSR (all s_commit coincide or only near s=1 -> cross-modal Foley correctness has no useful window separation in the tested model and method novelty collapses to scalar rejection; honest diagnostic); STOP_PROJECT (no usable trajectory access, no reliable axis measurement, or tail-forking not meaningful). Novelty boundary: if axes do not separate, route to DIAGNOSTIC / NEGATIVE; do NOT force a method claim. The floor (DIAGNOSTIC / NEGATIVE) is genuinely good. Internal probes are non-blocking; until they run, report 'gap under available external probes', never 'irreducible uncommitted information'.

## Component Ladder

- rung0-reliability: Phase 0: trajectory access + validated velocity->score SDE + dataset/anchor manifest + per-axis reliability gate. Prerequisite for every map.
- rung1-commitment: C1a commitment map: marginal-preserving stochastic tail-forks -> s_commit(axis), normalized over A_independent.
- rung2-readout: C1b readout map: probe ladder on x0(s) for ODE-target and fork-majority targets -> s_read(axis, probe).
- rung3-gap-separation: C1c gap + axis separation + R1/R2 cross-tab + GO/NO-GO (make-or-break ends here).
- rung4-pruning: C3 axis-gated population pruning vs baselines incl. oracle upper bound (conditional, needs correctness sidecar + pre-registration).
- rung5-verifier: C4 cheap process-aware verifier (only after maps show headroom).
- rung6-rollback: C5 axis-gated rollback via SDE/Restart re-noising (only if forking stable).
- rung7-internal-probes: Phase 7 internal-feature probes (latent / AV cross-attention), only if external-probe gap is large; non-blocking.

## Algorithmic Formalization

Canonical time axis: generation progress s in [0,1] (s=0 noise, s=1 audio); map s<->t to MMAudio's actual integration time ONCE in audited code (some flow models integrate t:1->0). x_s = intermediate state; x0(s) = Tweedie best-guess of final audio (readout input). Fork kernel (marginal-preserving SDE, alpha the only knob): for k in 1..K, x=x_s; for each progress step s_i->s_next in schedule(s->1): t_i=s_to_t(s_i); v=v_theta(x,t_i,video_cond); score=score_from_velocity(v,x,t_i); sigma=alpha*g(s_i) (alpha=0 => deterministic ODE); x=step_euler_maruyama(x,v,score,sigma,ds); comps.append(decode(x)). A common rectified-flow score form is score(x,t) ~ (t*v - x)/(1-t) (equivalently grad-log p_t = -(x_t+(1-t)v)/t) — AUDIT sign/direction against MMAudio code; do not copy blindly. Reserve Restart re-noising for Phase 6 rollback; do NOT use it for commitment. alpha selection: predefined pilot grid; primary operating alpha = smallest alpha producing measurable tail diversity while preserving valid audio (audio-validity guard); report full A(axis,s,alpha) surface as secondary, single primary alpha for headline s_commit. Normalized commitment gain: commit(s,axis)=clip((A_fork(x_s,axis,alpha)-A_independent(video,axis))/(1-A_independent(video,axis)),0,1); s_commit(axis)=min s with commit>=theta_commit, bootstrapped over videos. Agreement metric per axis: categorical (presence/timing-bin/class/binding) -> exact-match rate or Krippendorff alpha across forks; embedding (material) -> mean pairwise cosine. Readout: s_read(axis,probe,target)=min s with accuracy/AUROC>=theta_read, bootstrapped over videos; commitment uses clean final completions, readout uses blurry x0(s) (the blur penalty is part of the early gap, disentangled by R1/R2). Highest silent-bug risk = the velocity->score conversion: alpha=0 reproducing the ODE is necessary but does NOT test the score term (multiplied by 0); the real test is nonzero-alpha (small-alpha continuity + fork audio validity + nontrivial diversity).

## Plan-Code Audit

_Not recorded yet._

## Probes

_None recorded._

## Formal Diagnostics

- diag_phase0_feasibility: {'id': 'diag_phase0_feasibility', 'kind': 'implementation_smoke', 'claim_relevance': 'paper_scope_affecting', 'command': '/run-experiment "foley-cw Phase 0 (STRICT): confirm MMAudio trajectory access (extract x_s, resume from x_s, compute x0(s)) and audit the s<->t mapping; derive and VALIDATE the velocity->score SDE (alpha=0 reproduces ODE AND nonzero-alpha checks: small-alpha continuity, fork audio validity, nontrivial diversity); build dataset_subset_manifest from FoleyBench (single-event + optional 2-event, class balance, duration, anchor source, usable n per axis); build+validate event-anchor protocol; run the three-part reliability gate (determinism + robustness + validity) on Tier-1 axes plus Tier-2 if strong. Do NOT build maps, train a verifier, or use correctness-vs-video. Output feasibility_report.md, score_sde_validation_report.md, dataset_subset_manifest.md, event_anchor_validation_report.md, axis_reliability_report.md. Emit GO_MAPS_PHASE / FIX_SCORE_CONVERSION / NO_TRAJECTORY_ACCESS / STOP_PROJECT."', 'status': 'pending', 'owner': 'human', 'expected_result_paths': ['results/feasibility_report.md', 'results/score_sde_validation_report.md', 'results/dataset_subset_manifest.md', 'results/event_anchor_validation_report.md', 'results/axis_reliability_report.md'], 'success_signal': 'Emit GO_MAPS_PHASE: trajectory access OK; SDE validated at alpha=0 AND nonzero-alpha; dataset+anchor manifest ready; >=3 axes pass determinism+robustness+validity.', 'null_result_interpretation': 'FIX_SCORE_CONVERSION and halt if alpha=0 fails or small-alpha continuity is violated; NO_TRAJECTORY_ACCESS -> STOP_PROJECT; STOP_PROJECT if <3 reliable axes or no usable anchors. This gate is load-bearing; do not proceed to maps without GO_MAPS_PHASE.'}
- diag_maps_phases123: {'id': 'diag_maps_phases123', 'kind': 'paper_bearing_main', 'claim_relevance': 'primary_evidence', 'command': '/run-experiment "foley-cw Phases 1-3 (make-or-break, no correctness labels): COMMITMENT map via marginal-preserving stochastic tail-forks (predefined alpha grid, smallest-valid-alpha rule + audio-validity guard, full A(axis,s,alpha) surface), normalized as commitment gain over A_independent(video,axis) per the video-prior baseline; READOUT map via the probe ladder on x0(s) for BOTH ODE-target and fork-majority targets; report s_commit, s_read, gap, R1/R2 cross-tab, axis-separation with bootstrap-over-videos CIs and pre-registered thresholds + sensitivity. Output commitment_map.csv, readout_map.csv, commitment_readout_gap_report.md, go_no_go_decision.md. Emit GO_MAP / GO_READOUT / GO_RESTRICTED / GO_DIAGNOSTIC / STOP_ADSR / STOP_PROJECT / FORK_ALPHA_NO_VALID_OPERATING_POINT."', 'status': 'pending', 'owner': 'human', 'expected_result_paths': ['results/commitment_map.csv', 'results/readout_map.csv', 'results/commitment_readout_gap_report.md', 'results/go_no_go_decision.md'], 'success_signal': 'GO_MAP + GO_READOUT: axes show separated commitment windows beyond CIs AND at least one feasible probe reads the early axes well before the end. This is the correctness-label-free scientific make-or-break and the only gate to Phase 4.', 'null_result_interpretation': 'STOP_ADSR if all s_commit coincide or only near s=1 (publishable NEGATIVE; collapses to scalar DiffRS); GO_DIAGNOSTIC if commitment exists but cheap readout lags far behind (R2-dominated; publish gap + probe-limitation); FORK_ALPHA_NO_VALID_OPERATING_POINT routes to a different kernel or GO_DIAGNOSTIC (not auto-kill). Requires diag_phase0_feasibility = GO_MAPS_PHASE first.'}
