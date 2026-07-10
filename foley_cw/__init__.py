"""foley_cw — Foley Commitment / Readout Windows in V2A flow generation.

Implementation of the `foley-cw` experiment plan (see refine-logs/EXPERIMENT_PLAN.md
and experiment/experiment_pack.json). This package measures, for a video-conditioned
audio FLOW model (MMAudio v1), two per-axis quantities:

  * commitment  s_commit(axis) — when the model's own self-target stabilizes under
    stochastic re-completion, ABOVE the video-conditioned prior;
  * readout     s_read(axis, probe) — when a probe can predict that self-target from
    the running x0(s).

The make-or-break is whether commitment windows separate across axes and whether early
axes are readable early (GO_MAP + GO_READOUT), both WITHOUT correctness labels.

Design boundary (read foley_cw/README.md):
  * Everything model-specific to MMAudio (s<->t integration direction, the velocity
    field parameterization, the velocity->score conversion sign, latent decode) is
    isolated in `model_adapter.MMAudioBackend` and `time_map`, and is marked
    UNVERIFIED — it MUST be audited/validated against MMAudio source in Phase 0.
  * `synthetic_backend.SyntheticGaussianFlow` provides a closed-form analytic flow so
    the highest-silent-bug-risk math (velocity->score, marginal-preserving SDE fork)
    is unit-testable on CPU with no GPU and no MMAudio.

This package is numpy-only at import time. Heavy / model dependencies (torch, MMAudio,
CLAP, taggers, librosa) are imported lazily inside the methods that need them so the
analytic / planning paths run on a bare numpy environment.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
