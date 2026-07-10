"""Kernel-provenance assertion (June-13 manual sections 1.2 / 15.8).

A `CFG_KERNEL_OK` token is invalid without a `schedule=` suffix, and a commitment
grid may only run under the exact (cfg, schedule) tuple its kernel certification
used. This module is the code-side guard: Phase-1 commitment runners call
`assert_certified_kernel(cfg, schedule)` at startup, which refuses to proceed
unless `results/.../certified_kernels.json` records a ratified OK certification
for that exact tuple.

The cfg=4.5 deployed arm is CANDIDATE-only out of Stage M (pilot cells); its
ledger entry has `ratified=false` until a full-Phase-1-pool Gate-A re-certifies
it. `require_ratified=True` (the default for a headline commitment grid) therefore
rejects cfg=4.5 until that ratification writes a ratified entry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class KernelNotCertifiedError(RuntimeError):
    """No ratified (cfg, schedule) kernel certification — refuse to run the grid."""


def load_certifications(path: Path) -> list[dict]:
    """Flatten a certified_kernels.json into a list of {cfg, schedule, ok, ratified}."""
    d = json.loads(Path(path).read_text())
    return [v for k, v in d.items() if not k.startswith("_") and isinstance(v, dict)]


def find_certification(cfg: float, schedule: str, path: Path) -> Optional[dict]:
    for c in load_certifications(path):
        if (abs(float(c.get("cfg", -1)) - float(cfg)) < 1e-9
                and str(c.get("schedule", "constant")) == str(schedule)):
            return c
    return None


def assert_certified_kernel(cfg: float, schedule: str, path: Path,
                            require_ratified: bool = True) -> dict:
    """Return the matching certification or raise KernelNotCertifiedError.

    require_ratified=True (headline commitment grid): the entry must have
    ok=true AND ratified=true. Set False only for explicitly non-gating,
    diagnostic runs that may use a candidate kernel (and must say so in their
    report).
    """
    if not Path(path).exists():
        raise KernelNotCertifiedError(
            f"no certification ledger at {path}; run Stage-M evaluation (and, for "
            f"cfg=4.5, the full-Phase-1-pool Gate-A) before a (cfg={cfg:g}, "
            f"schedule={schedule}) commitment grid")
    c = find_certification(cfg, schedule, path)
    if c is None:
        raise KernelNotCertifiedError(
            f"no kernel certification for (cfg={cfg:g}, schedule={schedule}) in {path}; "
            f"a commitment grid may only run under a certified (cfg, schedule) tuple "
            "(manual 15.8)")
    # Validate the token TEXT, not just the booleans (Codex T1 finding): an
    # inconsistent ledger (ok=true with a FAIL token, or a token missing the
    # mandatory schedule= suffix) must be rejected, never trusted.
    token = str(c.get("token", ""))
    suffix = f"schedule={schedule}" if schedule != "constant" else ""
    if suffix and suffix not in token:
        raise KernelNotCertifiedError(
            f"ledger token {token!r} for (cfg={cfg:g}, schedule={schedule}) is missing "
            f"the mandatory '{suffix}' suffix (manual 15.8: a CFG_KERNEL token is invalid "
            "without it)")
    if not c.get("ok", False) or not token.startswith("CFG_KERNEL_OK"):
        raise KernelNotCertifiedError(
            f"kernel for (cfg={cfg:g}, schedule={schedule}) is not a certified OK "
            f"(ok={c.get('ok')}, token={token!r}); cannot run a commitment grid there")
    if require_ratified and not c.get("ratified", False):
        raise KernelNotCertifiedError(
            f"kernel for (cfg={cfg:g}, schedule={schedule}) is CANDIDATE-only "
            f"({c.get('scope')}); ratify via the full-Phase-1-pool Gate-A before "
            "running the headline commitment grid (manual 1.2/15.8)")
    return c
