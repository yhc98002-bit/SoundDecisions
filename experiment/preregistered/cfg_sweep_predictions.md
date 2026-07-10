# Pre-registered cfg-sweep predictions (June-13 manual §1.2)

Frozen 2026-06-13 from "LONG_RANGE_EXPERIMENT_PLAN _June_13th.md", before any Phase-1+ curve was inspected. Supersedes the 2026-06-12 freeze (the June-13 manual promoted the three-share decomposition + F-1 to the paper headline; pre-registered text changed).

**Science, not gate — cross-cfg behavior.** Pre-registered predictions: `s_commit` shifts earlier as cfg increases; inter-axis separation may compress; ordering changes carry no directional bet. The F-1 predictions (§8.3) extend these to the limit where commitment collapses onto the initial noise.

### 1.3 α and noise schedule
- Primary α = smallest α with measurable tail diversity and valid fork audio; set value `PRIMARY_ALPHA = 1.6` at `cfg = 1.0`. Full `A(axis, s, α)` surface reported as secondary evidence.
- **Early-heavy g(s) schedules** (`linear_down`, `sqrt_down`) are the sanctioned kernel-redesign route — explore early where decisions are open, quiet the tail. Pilot at `cfg ∈ {2.5, 4.5}`; each tuple needs its own Gate-A pass.
- **Discretization arm:** `n_steps = 40` on a micro-map-scale sub-grid bounds integrator error vs. dynamics in the high-cfg locking. Production grids use `n_steps = 20` so every scan point lies on the integration grid.
- `FORK_ALPHA_NO_VALID_OPERATING_POINT` routes to the schedule pilots or diagnostic framing — never to silently raising α.

### 1.4 Logging & storage contract (HARD)
Pooled per-layer hidden states at grid s-points for **all** generations; every-step pooled features for base trajectories only; token-level activations for ≤ 8 debug clips; `x̂0(s)` previews stored for base + independents; fork finals measured on the fly with a 10% wav audit sample; all per-axis measurements stored. **Hard cap 100 GB** — halt and report if exceeded; no silent expansion or degradation.

### 1.5 Evidence & statistics
Stage-M and synthetic outputs are never citable as evidence; evidence comes only from Phase 1–3+ runs on the frozen manifest. All thresholds (`θ_commit`, `θ_read`, `θ_rel`, `θ_robust`, `θ_cal`, δ) are frozen before inspecting headline curves; sensitivity sweeps after. Bootstrap unit = video. Power rule: CI width is dominated by across-video variance — **add clips, not forks** (K = 12 is past diminishing returns). Per-axis minimum usable n is declared in the manifest; underpowered axes are reported as underpowered, never as results.

### 1.6 Parallelism
Non-gating analyses (Track P) start as soon as their inputs are cached but never feed decision tokens. Track L items launch immediately; nothing human-time waits on GPUs.

### 1.7 Novelty boundary & anti-overclaim
Not a final-audio verifier paper, not a DiffRS clone, not an SMC-for-V2A paper: the contribution is the per-axis commitment/readout map for cross-modally conditioned Foley decisions, its causal validation, and what it licenses. Scalar intermediate-reward search for V2A and multi-verifier ITS for joint AV generation exist — cite both; claim nothing without head-to-head evidence (Phase 4). Do not present headline-cfg results as deployment-relevant without the readout-transfer evidence and scope note. Do not cite music-ADSR as proof for V2A. Until Track P reports, the gap is "gap under available external probes," not "irreducible." No generator fine-tuning, no delayed callback, no new foundation model. If axes do not separate, route to DIAGNOSTIC/NEGATIVE; never force a method claim.
