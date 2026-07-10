"""On-disk run-output store implementing the §1.4 logging & storage contract.

Manual: experiment/LONG_RANGE_EXPERIMENT_PLAN.md §1.4 (HARD; implement before
any large run).  Mapping from contract clause to API:

  * Pooled per-layer features at grid s-points for ALL generations
      -> put_features(gen_id, s, feats)
  * Every-step pooled features for BASE trajectories only
      -> put_step_features(gen_id, ts, feats)   (caller restricts to base)
  * x̂0(s) previews for base + independents
      -> put_preview(gen_id, s, wav)
  * Fork finals measured on the fly; wavs retained only for a ~10% audit sample
      -> put_final_wav(gen_id, wav, audit_only=True)  (deterministic selection
         via sha256(gen_id) % 10 == 0; see audit_selected)
  * ALL per-axis measurements stored for every generation
      -> record_measurement(gen_id, axis_id, target, extra)
  * 100 GB hard cap, halt-and-report, no silent degradation
      -> every write is accounted against the StorageBudget and immediately
         checked; an over-cap write raises StorageCapExceeded.

Resumable units: journal_done / is_done / done_units / load_journal give an
atomic (tmp + os.replace) completion journal so an interrupted run never
re-executes finished units and a partially-written journal entry is never
counted as done.

Layout under root/:
  features/<gen_id>__s{s:.2f}.npz     key 'pooled' (fp16, compressed)
  features/<gen_id>__steps.npz        keys 'ts' (f32), 'pooled' (fp16)
  previews/<gen_id>__s{s:.2f}.wav     PCM_16
  finals/<gen_id>.wav                 PCM_16 (base + independents)
  audit_wavs/<gen_id>.wav             PCM_16 (~10% of fork finals)
  measurements/measurements.jsonl     one JSON object per measurement
  journal/<unit_id>.json              completion journal (unit_id sanitized)

numpy + stdlib at import time; soundfile is imported lazily inside the wav
writers so the numpy core stays importable without it.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from .storage_budget import StorageBudget
from .types import AxisKind, SelfTarget

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Subdirectories created lazily under the store root.
_SUBDIRS: tuple[str, ...] = (
    "features", "previews", "finals", "audit_wavs", "measurements", "journal", "gate_a",
    "arc3/pertoken", "arc3/cond_feats",   # Arc-3 confirmatory feature dumps (B1 per-token, B2)
)
#: Fork-final audit sampling modulus: sha256(gen_id) % 10 == 0 -> ~10% retained.
_AUDIT_MODULUS: int = 10
#: Default preview / final sample rate (Hz).
_DEFAULT_SR: int = 16000


# ---------------------------------------------------------------------------
# SelfTarget JSON (de)serialization
# ---------------------------------------------------------------------------

def to_jsonable_target(target: SelfTarget) -> dict:
    """Serialize a SelfTarget to a plain JSON-able dict.

    Fields mirror foley_cw.types.SelfTarget: axis_id, kind (enum value),
    label, embedding (list of floats or None).
    """
    label = target.label
    if isinstance(label, np.generic):
        label = label.item()
    emb = target.embedding
    return {
        "axis_id": target.axis_id,
        "kind": target.kind.value,
        "label": label,
        "embedding": None if emb is None else [float(v) for v in np.asarray(emb).ravel()],
    }


def target_from_jsonable(d: dict) -> SelfTarget:
    """Inverse of to_jsonable_target.  Embedding comes back as a float64 array."""
    emb = d.get("embedding")
    return SelfTarget(
        axis_id=d["axis_id"],
        kind=AxisKind(d["kind"]),
        label=d.get("label"),
        embedding=None if emb is None else np.asarray(emb, dtype=float),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_unit_id(unit_id: str) -> str:
    """Make a unit/gen id safe for a single filename ('/' and ':' -> '_')."""
    return unit_id.replace("/", "_").replace(":", "_")


def audit_selected(gen_id: str) -> bool:
    """Deterministic ~10% audit selection for fork-final wav retention.

    Hash-based (sha256 of the RAW gen_id, before filename sanitization) so the
    selection is reproducible across runs and machines and independent of
    insertion order.
    """
    digest = hashlib.sha256(gen_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % _AUDIT_MODULUS == 0


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def _require_soundfile():
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - exercised only without soundfile
        raise ImportError(
            "foley_cw.run_store: writing previews/finals requires the 'soundfile' "
            "package (pip install soundfile)."
        ) from exc
    return sf


# ---------------------------------------------------------------------------
# RunStore
# ---------------------------------------------------------------------------

class RunStore:
    """Single-process store for one run's cached outputs, budget-accounted.

    Every byte written through this class is reported to *budget* (when given)
    and the cap is checked immediately after each write — an over-cap write
    raises StorageCapExceeded AFTER landing on disk, so the halt report can
    point at real files instead of phantom state.
    """

    def __init__(self, root: Path, budget: Optional[StorageBudget] = None) -> None:
        self._root = Path(root)
        self._budget = budget

    @property
    def root(self) -> Path:
        return self._root

    @property
    def budget(self) -> Optional[StorageBudget]:
        return self._budget

    # ------------------------------------------------------------------
    # Internal plumbing
    # ------------------------------------------------------------------

    def _dir(self, name: str) -> Path:
        if name not in _SUBDIRS:
            raise ValueError(f"RunStore: unknown subdir {name!r}; expected one of {_SUBDIRS}")
        d = self._root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _account(self, nbytes: int, category: str, detail: str) -> None:
        if self._budget is None:
            return
        self._budget.account(nbytes, context=category)
        self._budget.check_or_halt(context=detail)

    def _account_file(self, path: Path, category: str) -> None:
        self._account(path.stat().st_size, category, str(path.relative_to(self._root)))

    def _write_wav(self, path: Path, wav: np.ndarray, sr: int, category: str) -> Path:
        sf = _require_soundfile()
        data = np.asarray(wav, dtype=np.float32)
        sf.write(str(path), data, int(sr), subtype="PCM_16")
        self._account_file(path, category)
        return path

    # ------------------------------------------------------------------
    # Features (§1.4: pooled at grid s for all gens; every-step for base only)
    # ------------------------------------------------------------------

    def put_features(self, gen_id: str, s: float, feats: np.ndarray) -> Path:
        """Store pooled per-layer features at one grid s-point (fp16, compressed)."""
        path = self._dir("features") / f"{sanitize_unit_id(gen_id)}__s{s:.2f}.npz"
        np.savez_compressed(path, pooled=np.asarray(feats).astype(np.float16))
        self._account_file(path, "features")
        return path

    def put_npz(self, subdir: str, name: str, **arrays: np.ndarray) -> Path:
        """Store an arbitrary compressed npz under a known subdir, budget-accounted.

        Used e.g. for the Gate-A embedding bundles so they never bypass the
        section-1.4 storage accounting.
        """
        path = self._dir(subdir) / f"{sanitize_unit_id(name)}.npz"
        np.savez_compressed(path, **arrays)
        self._account_file(path, subdir)
        return path

    def account_preexisting_tree(self) -> int:
        """Charge the budget with everything already under root (multi-shard /
        resumed runs share one 100 GB cap; each process starts from the on-disk
        truth instead of zero)."""
        from .storage_budget import measure_tree

        nbytes = measure_tree(self._root)
        if nbytes and self._budget is not None:
            self._budget.account(nbytes, context="preexisting")
            self._budget.check_or_halt(context="preexisting tree at startup")
        return nbytes

    def put_step_features(self, gen_id: str, ts: np.ndarray, feats: np.ndarray) -> Path:
        """Store every-step pooled features for a BASE trajectory.

        'ts' is kept at float32 (progress values must stay exact enough to
        match the scan grid); the pooled features — the storage-heavy part —
        are fp16 per the contract.
        """
        ts_arr = np.asarray(ts)
        feats_arr = np.asarray(feats)
        if ts_arr.shape[0] != feats_arr.shape[0]:
            raise ValueError(
                f"put_step_features: ts has {ts_arr.shape[0]} steps but feats has "
                f"{feats_arr.shape[0]} leading entries; they must align per step."
            )
        path = self._dir("features") / f"{sanitize_unit_id(gen_id)}__steps.npz"
        np.savez_compressed(
            path,
            ts=ts_arr.astype(np.float32),
            pooled=feats_arr.astype(np.float16),
        )
        self._account_file(path, "features")
        return path

    # ------------------------------------------------------------------
    # Previews and finals (§1.4: previews for base + independents; fork
    # finals retained only for the ~10% audit sample)
    # ------------------------------------------------------------------

    def put_preview(self, gen_id: str, s: float, wav: np.ndarray, sr: int = _DEFAULT_SR) -> Path:
        """Store a decoded x̂0(s) preview at one grid s-point as PCM_16 wav."""
        path = self._dir("previews") / f"{sanitize_unit_id(gen_id)}__s{s:.2f}.wav"
        return self._write_wav(path, wav, sr, "previews")

    def put_final_wav(
        self,
        gen_id: str,
        wav: np.ndarray,
        sr: int = _DEFAULT_SR,
        audit_only: bool = False,
    ) -> Optional[Path]:
        """Store a final wav.

        audit_only=False (base / independents): always stored under finals/.
        audit_only=True (fork finals): stored under audit_wavs/ ONLY when the
        deterministic ~10% audit selection picks this gen_id; otherwise the wav
        is dropped (the caller has already measured it on the fly) and None is
        returned.
        """
        if audit_only:
            if not self.audit_selected(gen_id):
                return None
            path = self._dir("audit_wavs") / f"{sanitize_unit_id(gen_id)}.wav"
            return self._write_wav(path, wav, sr, "audit_wavs")
        path = self._dir("finals") / f"{sanitize_unit_id(gen_id)}.wav"
        return self._write_wav(path, wav, sr, "finals")

    @staticmethod
    def audit_selected(gen_id: str) -> bool:
        """Deterministic ~10% audit selection (exposed for tests)."""
        return audit_selected(gen_id)

    # ------------------------------------------------------------------
    # Measurements (§1.4: ALL per-axis measurements stored)
    # ------------------------------------------------------------------

    @property
    def measurements_path(self) -> Path:
        return self._root / "measurements" / "measurements.jsonl"

    def record_measurement(
        self,
        gen_id: str,
        axis_id: str,
        target: Any,
        extra: Optional[dict] = None,
    ) -> None:
        """Append one measurement record as a JSON line.

        *target* may be a SelfTarget (serialized via to_jsonable_target) or an
        already-JSON-able dict (passed through).
        """
        if isinstance(target, SelfTarget):
            target_json: Any = to_jsonable_target(target)
        else:
            target_json = target
        record = {
            "gen_id": gen_id,
            "axis_id": axis_id,
            "target": target_json,
            "extra": dict(extra) if extra else {},
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        line = json.dumps(record, default=_json_default) + "\n"
        path = self._dir("measurements") / "measurements.jsonl"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
        self._account(
            len(line.encode("utf-8")), "measurements",
            str(path.relative_to(self._root)),
        )

    def iter_measurements(self) -> Iterator[dict]:
        """Yield measurement records in append order; empty if none recorded."""
        path = self.measurements_path
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if raw:
                    yield json.loads(raw)

    # ------------------------------------------------------------------
    # Completion journal (resumable units; atomic tmp + os.replace)
    # ------------------------------------------------------------------

    def _journal_path(self, unit_id: str) -> Path:
        return self._root / "journal" / f"{sanitize_unit_id(unit_id)}.json"

    def journal_done(self, unit_id: str, payload: dict) -> Path:
        """Atomically mark *unit_id* done with *payload*.

        Write-to-tmp + os.replace so a crash mid-write leaves only a *.tmp
        file, which is_done / done_units ignore — a unit is never half-done.
        """
        jdir = self._dir("journal")
        final = jdir / f"{sanitize_unit_id(unit_id)}.json"
        tmp = final.with_suffix(".json.tmp")
        data = json.dumps(payload, default=_json_default, sort_keys=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp, final)
        self._account_file(final, "journal")
        return final

    def is_done(self, unit_id: str) -> bool:
        return self._journal_path(unit_id).exists()

    def done_units(self) -> set[str]:
        """Set of completed unit ids (in SANITIZED form, as stored on disk).

        Callers holding raw ids should use is_done(unit_id), which sanitizes
        before checking; '/' and ':' are not recoverable from filenames.
        """
        jdir = self._root / "journal"
        if not jdir.is_dir():
            return set()
        return {p.stem for p in jdir.glob("*.json") if p.is_file()}

    def load_journal(self, unit_id: str) -> dict:
        path = self._journal_path(unit_id)
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
