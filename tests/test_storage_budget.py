"""Tests for foley_cw/storage_budget.py — §1.4 hard storage cap.

CPU-only, no network, no GPU, stdlib + numpy.  Key contracts checked:
  * account() accumulates spent_bytes; negative byte counts are rejected.
  * check_or_halt raises StorageCapExceeded strictly when spent > cap, with
    spent_bytes / cap_bytes / context carried on the exception.
  * check_projection_or_halt raises BEFORE a run whose projection exceeds the
    cap and does not mutate spent state.
  * summary() reports the fields a halt report needs.
  * measure_tree sums on-disk bytes recursively and returns 0 for missing paths.
"""

from __future__ import annotations

import pytest

from foley_cw.storage_budget import (
    BYTES_PER_GB,
    StorageBudget,
    StorageCapExceeded,
    measure_tree,
)


# --------------------------------------------------------------------------------------
# account / spent_bytes
# --------------------------------------------------------------------------------------

class TestAccount:
    def test_starts_at_zero(self):
        budget = StorageBudget(cap_gb=1.0)
        assert budget.spent_bytes == 0

    def test_account_accumulates(self):
        budget = StorageBudget(cap_gb=1.0)
        budget.account(100, context="features")
        budget.account(250, context="previews")
        assert budget.spent_bytes == 350

    def test_account_zero_is_allowed(self):
        budget = StorageBudget(cap_gb=1.0)
        budget.account(0)
        assert budget.spent_bytes == 0

    def test_negative_bytes_rejected(self):
        budget = StorageBudget(cap_gb=1.0)
        with pytest.raises(ValueError):
            budget.account(-1)

    def test_nonpositive_cap_rejected(self):
        with pytest.raises(ValueError):
            StorageBudget(cap_gb=0.0)

    def test_default_cap_is_100_gb(self):
        budget = StorageBudget()
        assert budget.cap_gb == pytest.approx(100.0)
        assert budget.cap_bytes == pytest.approx(100.0 * BYTES_PER_GB)


# --------------------------------------------------------------------------------------
# check_or_halt
# --------------------------------------------------------------------------------------

class TestCheckOrHalt:
    def _tiny_budget(self, cap_bytes: float) -> StorageBudget:
        return StorageBudget(cap_gb=cap_bytes / BYTES_PER_GB)

    def test_under_cap_does_not_raise(self):
        budget = self._tiny_budget(1000)
        budget.account(999)
        budget.check_or_halt()  # no raise

    def test_exactly_at_cap_does_not_raise(self):
        """Contract: raises when spent > cap (strict), not at spent == cap."""
        budget = self._tiny_budget(1000)
        budget.account(1000)
        budget.check_or_halt()  # no raise

    def test_over_cap_raises(self):
        budget = self._tiny_budget(1000)
        budget.account(1001)
        with pytest.raises(StorageCapExceeded):
            budget.check_or_halt()

    def test_exception_carries_fields(self):
        budget = self._tiny_budget(1000)
        budget.account(5000, context="features")
        with pytest.raises(StorageCapExceeded) as excinfo:
            budget.check_or_halt(context="features/gen001__s0.30.npz")
        exc = excinfo.value
        assert exc.spent_bytes == 5000
        assert exc.cap_bytes == 1000
        assert exc.context == "features/gen001__s0.30.npz"

    def test_exception_message_mentions_halt(self):
        """§1.4: the failure mode is halt-and-report, never silent degradation."""
        budget = self._tiny_budget(10)
        budget.account(100)
        with pytest.raises(StorageCapExceeded) as excinfo:
            budget.check_or_halt()
        assert "HALT" in str(excinfo.value)


# --------------------------------------------------------------------------------------
# project / check_projection_or_halt
# --------------------------------------------------------------------------------------

class TestProjection:
    def test_project_includes_already_spent(self):
        budget = StorageBudget(cap_gb=1.0)
        budget.account(500)
        assert budget.project(n_units=10, bytes_per_unit=100.0) == pytest.approx(1500.0)

    def test_project_with_nothing_spent(self):
        budget = StorageBudget(cap_gb=1.0)
        assert budget.project(n_units=4, bytes_per_unit=2.5) == pytest.approx(10.0)

    def test_projection_over_cap_raises_before_run(self):
        budget = StorageBudget(cap_gb=2000 / BYTES_PER_GB)  # cap = 2000 bytes
        budget.account(500)
        with pytest.raises(StorageCapExceeded):
            budget.check_projection_or_halt(n_units=100, bytes_per_unit=100.0,
                                            context="phase1 grid")

    def test_projection_does_not_mutate_spent(self):
        budget = StorageBudget(cap_gb=2000 / BYTES_PER_GB)
        budget.account(500)
        with pytest.raises(StorageCapExceeded):
            budget.check_projection_or_halt(n_units=100, bytes_per_unit=100.0)
        assert budget.spent_bytes == 500

    def test_projection_within_cap_does_not_raise(self):
        budget = StorageBudget(cap_gb=2000 / BYTES_PER_GB)
        budget.account(500)
        budget.check_projection_or_halt(n_units=10, bytes_per_unit=100.0)  # 1500 <= 2000

    def test_projection_exception_carries_projected_total(self):
        budget = StorageBudget(cap_gb=1000 / BYTES_PER_GB)
        budget.account(200)
        with pytest.raises(StorageCapExceeded) as excinfo:
            budget.check_projection_or_halt(n_units=10, bytes_per_unit=200.0)
        assert excinfo.value.spent_bytes == 2200  # projected total, not current spent


# --------------------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------------------

class TestSummary:
    def test_summary_fields(self):
        budget = StorageBudget(cap_gb=1.0)
        budget.account(100, context="features")
        budget.account(50, context="previews")
        s = budget.summary()
        for key in ("cap_gb", "cap_bytes", "spent_bytes", "spent_gb",
                    "remaining_bytes", "utilization", "n_accounts", "by_context"):
            assert key in s, f"summary() missing field {key!r}"
        assert s["spent_bytes"] == 150
        assert s["n_accounts"] == 2
        assert s["by_context"] == {"features": 100, "previews": 50}
        assert s["cap_bytes"] == pytest.approx(BYTES_PER_GB)
        assert s["remaining_bytes"] == pytest.approx(BYTES_PER_GB - 150)
        assert s["utilization"] == pytest.approx(150 / BYTES_PER_GB)

    def test_summary_is_json_serializable(self):
        import json
        budget = StorageBudget(cap_gb=0.5)
        budget.account(10, context="journal")
        json.dumps(budget.summary())  # must not raise


# --------------------------------------------------------------------------------------
# measure_tree
# --------------------------------------------------------------------------------------

class TestMeasureTree:
    def test_missing_path_is_zero(self, tmp_path):
        assert measure_tree(tmp_path / "does_not_exist") == 0

    def test_single_file(self, tmp_path):
        f = tmp_path / "a.bin"
        f.write_bytes(b"x" * 123)
        assert measure_tree(f) == 123

    def test_nested_tree_sums_all_files(self, tmp_path):
        (tmp_path / "sub" / "deeper").mkdir(parents=True)
        (tmp_path / "top.bin").write_bytes(b"a" * 10)
        (tmp_path / "sub" / "mid.bin").write_bytes(b"b" * 20)
        (tmp_path / "sub" / "deeper" / "leaf.bin").write_bytes(b"c" * 30)
        assert measure_tree(tmp_path) == 60

    def test_empty_dir_is_zero(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        assert measure_tree(d) == 0


# --------------------------------------------------------------------------------------
# Module-level import safety
# --------------------------------------------------------------------------------------

def test_module_importable_without_heavy_deps():
    """foley_cw.storage_budget must import with stdlib only (no torch/soundfile)."""
    import foley_cw.storage_budget  # noqa: F401
