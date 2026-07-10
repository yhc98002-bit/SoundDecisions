"""Hard storage budget for run outputs (manual §1.4, experiment/LONG_RANGE_EXPERIMENT_PLAN.md).

Contract (§1.4 Logging & storage contract — HARD):
  * 100 GB hard cap on cached run outputs.
  * If actual or PROJECTED usage exceeds the cap: halt and report via
    StorageCapExceeded.  No silent expansion, downsampling, or format
    degradation — the only permitted reaction to an exceeded cap is to stop
    and surface the numbers.

This module is stdlib-only and thread-simple by design: runners are
single-process, so no locking is needed.  All sizes are plain byte counts;
1 GB = 10^9 bytes (decimal — the conservative choice: the cap trips earlier
than with GiB accounting, never later).
"""

from __future__ import annotations

import os
from pathlib import Path

#: Decimal gigabyte.  Conservative vs. GiB: a 100 GB cap in decimal bytes is
#: SMALLER than 100 GiB, so the halt fires earlier, never later.
BYTES_PER_GB: int = 1_000_000_000


class StorageCapExceeded(Exception):
    """Raised when actual or projected storage exceeds the hard cap.

    Per §1.4 this is a halt-and-report signal: callers must NOT catch it to
    degrade output formats or skip logging; they may only stop the run and
    surface the report.
    """

    def __init__(self, spent_bytes: int, cap_bytes: int, context: str = "") -> None:
        self.spent_bytes = int(spent_bytes)
        self.cap_bytes = int(cap_bytes)
        self.context = context
        msg = (
            f"storage cap exceeded: {self.spent_bytes} bytes "
            f"({self.spent_bytes / BYTES_PER_GB:.3f} GB) vs cap {self.cap_bytes} bytes "
            f"({self.cap_bytes / BYTES_PER_GB:.3f} GB)"
        )
        if context:
            msg += f" — {context}"
        msg += ". HALT AND REPORT (manual §1.4): no silent expansion, downsampling, or format degradation."
        super().__init__(msg)


class StorageBudget:
    """Byte accounting against the §1.4 hard cap (default 100 GB).

    Usage pattern:
      * ``account(nbytes, context)`` after every write, then ``check_or_halt()``.
      * ``check_projection_or_halt(n_units, bytes_per_unit)`` BEFORE launching a
        run whose per-unit cost is known, so the halt happens before any compute
        is spent.

    Single-process by contract — no locks.
    """

    def __init__(self, cap_gb: float = 100.0) -> None:
        if cap_gb <= 0:
            raise ValueError(f"StorageBudget: cap_gb must be positive, got {cap_gb}")
        self.cap_gb = float(cap_gb)
        self.cap_bytes = float(cap_gb) * BYTES_PER_GB
        self._spent_bytes: int = 0
        self._n_accounts: int = 0
        self._by_context: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Accounting
    # ------------------------------------------------------------------

    @property
    def spent_bytes(self) -> int:
        return self._spent_bytes

    def account(self, nbytes: int, context: str = "") -> None:
        """Record *nbytes* written under *context* (free-form category string)."""
        nbytes = int(nbytes)
        if nbytes < 0:
            raise ValueError(f"StorageBudget.account: nbytes must be >= 0, got {nbytes}")
        self._spent_bytes += nbytes
        self._n_accounts += 1
        self._by_context[context] = self._by_context.get(context, 0) + nbytes

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project(self, n_units: int, bytes_per_unit: float) -> float:
        """Projected TOTAL bytes (already spent + n_units * bytes_per_unit)."""
        return float(self._spent_bytes) + float(n_units) * float(bytes_per_unit)

    # ------------------------------------------------------------------
    # Halt checks
    # ------------------------------------------------------------------

    def check_or_halt(self, context: str = "") -> None:
        """Raise StorageCapExceeded when spent > cap; otherwise no-op."""
        if self._spent_bytes > self.cap_bytes:
            raise StorageCapExceeded(self._spent_bytes, int(self.cap_bytes), context=context)

    def check_projection_or_halt(
        self,
        n_units: int,
        bytes_per_unit: float,
        context: str = "",
    ) -> None:
        """Raise StorageCapExceeded BEFORE a run whose projection exceeds the cap.

        Does not mutate the spent counter; a failed projection costs nothing.
        """
        projected = self.project(n_units, bytes_per_unit)
        if projected > self.cap_bytes:
            detail = (
                f"projection: {n_units} units x {float(bytes_per_unit):.1f} B/unit "
                f"on top of {self._spent_bytes} B already spent"
            )
            full_context = f"{context} [{detail}]" if context else detail
            raise StorageCapExceeded(int(projected), int(self.cap_bytes), context=full_context)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Plain-dict report of the budget state (for logs / halt reports)."""
        return {
            "cap_gb": self.cap_gb,
            "cap_bytes": self.cap_bytes,
            "spent_bytes": self._spent_bytes,
            "spent_gb": self._spent_bytes / BYTES_PER_GB,
            "remaining_bytes": self.cap_bytes - self._spent_bytes,
            "utilization": self._spent_bytes / self.cap_bytes,
            "n_accounts": self._n_accounts,
            "by_context": dict(self._by_context),
        }


def measure_tree(path: Path) -> int:
    """Total on-disk bytes (sum of file sizes) under *path*, recursively.

    Returns 0 for a missing path; for a regular file, its size.  Symlinks are
    NOT followed (lstat), so a link into a large external tree cannot inflate
    or hide budget usage.
    """
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return int(os.lstat(path).st_size)
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for fname in filenames:
            try:
                total += os.lstat(os.path.join(dirpath, fname)).st_size
            except OSError:
                continue  # raced deletion; skip rather than crash the report
    return int(total)
