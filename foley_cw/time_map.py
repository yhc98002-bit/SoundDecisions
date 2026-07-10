"""The s <-> t seam: progress axis <-> model integration time.

This is a deliberate single chokepoint for one of the two highest silent-bug risks the
plan names (refine-logs/EXPERIMENT_PLAN.md §2, watch-list #5): MMAudio's internal time
direction. The whole codebase scans in PROGRESS s (s=0 noise, s=1 audio) and converts to
the model's native time t ONLY here, exactly once.

Some flow models integrate t: 0->1 (t aligned with progress); others integrate t: 1->0
(t reversed). We do NOT know MMAudio's convention from inside this repo (MMAudio source
is not vendored here). `MMAUDIO_S_TO_T` is therefore UNVERIFIED and MUST be pinned in
Phase 0.1 by reading MMAudio's sampler and asserting the smoke generation matches.

The synthetic backend uses `IdentitySToT` (t = s), under which the rectified-flow math
in score_sde.py is exact and fully unit-tested. That is the validated path; the MMAudio
path inherits the same kernel but its mapping must be audited before use.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SToT:
    """Affine progress->time map  t(s) = t0 + (t1 - t0) * s.

    `t0` is the model time at progress s=0 (noise); `t1` at progress s=1 (audio).
    Ascending models: t0=0, t1=1. Descending models: t0=1, t1=0.

    The map is exposed only through `s_to_t`; `dt_ds` carries the (signed) Jacobian used
    by the integrator so the drift is applied as v*dt, never v*ds, which is what makes a
    reversed-time model integrate correctly instead of silently backwards.
    """

    t0: float
    t1: float
    name: str = "affine"
    verified: bool = False  # True only once pinned against the real model in Phase 0.1

    def s_to_t(self, s: float) -> float:
        return self.t0 + (self.t1 - self.t0) * s

    @property
    def dt_ds(self) -> float:
        """d t / d s — constant for an affine map, with the sign of the time direction."""
        return self.t1 - self.t0

    @property
    def descending(self) -> bool:
        return self.t1 < self.t0


# Validated convention for the synthetic / rectified-flow oracle: t == s, ascending.
IdentitySToT = SToT(t0=0.0, t1=1.0, name="identity_ascending", verified=True)

# Reversed-time convention (t: 1 -> 0) for models that integrate noise->data as t decreasing.
DescendingSToT = SToT(t0=1.0, t1=0.0, name="descending", verified=False)

# UNVERIFIED placeholder for MMAudio. Phase 0.1 MUST replace this (and set verified=True)
# after auditing MMAudio's sampler. We default to the ascending convention but flag it as
# unverified so any use before Phase 0.1 is loud, not silent.
MMAUDIO_S_TO_T = SToT(t0=0.0, t1=1.0, name="mmaudio_UNVERIFIED", verified=False)
