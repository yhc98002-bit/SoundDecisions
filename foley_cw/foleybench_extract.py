"""FoleyBench clip extraction, validation, and clip selection.

Serves experiment/LONG_RANGE_EXPERIMENT_PLAN.md §2 (Stage-M clip source) and
§3.1 (Phase 0.3 candidate pool for `A_independent` screening).

Contract
--------
The FoleyBench HF snapshot (arXiv:2511.13219) ships clips as base64-encoded MP4
payloads embedded in parquet shards (`data/train-*.parquet`, ~1.4 GB each), plus
a metadata CSV (`foleybench.csv`).  This module:

  * streams parquet rows WITHOUT loading whole shards into memory
    (``iter_foleybench_rows``);
  * decodes a payload to raw MP4 bytes with an `ftyp`-magic check
    (``decode_video_bytes`` / ``VideoDecodeError``);
  * extracts filtered clips to ``<key>.mp4`` files atomically and keeps an
    idempotent index CSV (``extract_clips`` / ``ExtractionReport``);
  * second-pass validates extracted files with PyAV (``validate_clips_av``);
  * selects deterministic, ucs_category-stratified clip lists for Stage M and
    the Phase-0.3 screening pool from the single-source discrete subset
    (``select_stage_m_clips`` / ``select_screening_pool``);
  * writes manifests atomically (``write_manifest_json``).

Per the manual, Stage-M clips are diagnostics only and are NOT carried forward
into the Phase-1 manifest; the screening selection therefore takes an explicit
``exclude`` set.

Heavy deps (pyarrow, av) are imported lazily inside the functions that need
them; the module itself stays importable on a numpy-only environment.
"""

from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Glob pattern for FoleyBench parquet shards inside the dataset's data/ dir.
_SHARD_GLOB: str = "train-*.parquet"

#: Columns of the extraction index CSV (order is the on-disk column order).
INDEX_COLUMNS: tuple[str, ...] = (
    "key", "path", "sha256", "duration", "caption", "ucs_category",
    "audioset_category", "source_type", "discrete_vs_rest", "status",
)

#: Provenance tag written into manifests.
_GENERATED_BY: str = "foley_cw.foleybench_extract"

#: Minimum payload length for the MP4 magic check (size box + 'ftyp' + brand).
_MIN_MP4_LEN: int = 12


class VideoDecodeError(Exception):
    """Raised when a FoleyBench video payload cannot be decoded to valid MP4 bytes."""


# ---------------------------------------------------------------------------
# Parquet streaming
# ---------------------------------------------------------------------------

def _require_pyarrow_parquet():
    """Lazy-import pyarrow.parquet with a clear error message."""
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "foley_cw.foleybench_extract requires pyarrow to stream FoleyBench "
            "parquet shards (pip install pyarrow). The numpy core does not "
            "depend on it."
        ) from exc
    return pq


def iter_foleybench_rows(
    parquet_dir: Path,
    columns: list[str],
    batch_size: int = 8,
) -> Iterator[dict]:
    """Stream rows from all ``train-*.parquet`` shards under *parquet_dir*.

    Uses ``pyarrow.parquet.ParquetFile.iter_batches`` so at most *batch_size*
    rows are materialised at a time — shards are ~1.4 GB and must NEVER be read
    whole.  Shards are visited in sorted filename order; rows are yielded as
    plain dicts restricted to *columns*.

    Raises FileNotFoundError if the directory holds no matching shards.
    """
    pq = _require_pyarrow_parquet()
    parquet_dir = Path(parquet_dir)
    shards = sorted(parquet_dir.glob(_SHARD_GLOB))
    if not shards:
        raise FileNotFoundError(
            f"no {_SHARD_GLOB!r} shards found under {parquet_dir} "
            "(expected the FoleyBench data/ directory; NO download is performed)"
        )
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=batch_size, columns=list(columns)):
            for row in batch.to_pylist():
                yield row


# ---------------------------------------------------------------------------
# Payload decoding
# ---------------------------------------------------------------------------

def decode_video_bytes(payload: Any) -> bytes:
    """Decode a FoleyBench video payload to raw MP4 bytes.

    Accepts raw ``bytes``/``bytearray`` or a base64-encoded ``str`` (the HF
    snapshot stores ``video_data`` as a base64 string).  Validates the MP4
    magic — ``b'ftyp'`` at byte offset 4 — after decoding; anything else raises
    ``VideoDecodeError`` so callers can record a decode-error row instead of
    writing a corrupt file.
    """
    if isinstance(payload, (bytes, bytearray)):
        data = bytes(payload)
    elif isinstance(payload, str):
        try:
            data = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise VideoDecodeError(f"invalid base64 video payload: {exc}") from exc
    else:
        raise VideoDecodeError(
            f"unsupported video payload type {type(payload).__name__!r}; "
            "expected bytes, bytearray, or base64 str"
        )

    if len(data) < _MIN_MP4_LEN or data[4:8] != b"ftyp":
        raise VideoDecodeError(
            f"payload is not an MP4 (no 'ftyp' at offset 4; got "
            f"{data[4:8]!r} in {len(data)}-byte payload)"
        )
    return data


# ---------------------------------------------------------------------------
# Metadata CSV
# ---------------------------------------------------------------------------

def load_metadata_csv(csv_path: Path) -> dict[str, dict]:
    """Load ``foleybench.csv`` keyed by ``str(key)``.

    Columns: key, duration, dataset, width, height, caption, discrete_vs_rest,
    source_type, sound_type, ucs_category, audioset_category, metadata.
    Values are kept as raw strings; duplicate keys keep the LAST row.
    """
    csv_path = Path(csv_path)
    out: dict[str, dict] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            key = str(row.get("key", "")).strip()
            if key:
                out[key] = dict(row)
    return out


# ---------------------------------------------------------------------------
# Index CSV helpers (idempotent rewrite)
# ---------------------------------------------------------------------------

def _natural_key(key: str) -> tuple:
    """Sort numeric keys numerically, everything else lexicographically."""
    return (0, int(key), "") if key.isdigit() else (1, 0, key)


def _load_index_rows(index_csv: Path) -> dict[str, dict]:
    """Read the index CSV into {key: row}; empty dict if the file is absent."""
    index_csv = Path(index_csv)
    rows: dict[str, dict] = {}
    if not index_csv.exists():
        return rows
    with open(index_csv, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            key = str(row.get("key", "")).strip()
            if key:
                rows[key] = {col: row.get(col, "") for col in INDEX_COLUMNS}
    return rows


def _write_index_rows(index_csv: Path, rows: dict[str, dict]) -> None:
    """Atomically rewrite the WHOLE index CSV (collect-then-write idempotence)."""
    index_csv = Path(index_csv)
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(index_csv) + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(INDEX_COLUMNS))
        writer.writeheader()
        for key in sorted(rows, key=_natural_key):
            writer.writerow({col: rows[key].get(col, "") for col in INDEX_COLUMNS})
    os.replace(tmp, index_csv)


def _index_row(key: str, meta: dict, path: str, sha256: str, status: str) -> dict:
    return {
        "key": key,
        "path": path,
        "sha256": sha256,
        "duration": meta.get("duration", ""),
        "caption": meta.get("caption", ""),
        "ucs_category": meta.get("ucs_category", ""),
        "audioset_category": meta.get("audioset_category", ""),
        "source_type": meta.get("source_type", ""),
        "discrete_vs_rest": meta.get("discrete_vs_rest", ""),
        "status": status,
    }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

@dataclass
class ExtractionReport:
    """Counters from one ``extract_clips`` pass over the parquet shards."""

    n_seen: int = 0
    n_extracted: int = 0
    n_skipped_existing: int = 0
    n_decode_error: int = 0
    n_filtered_out: int = 0


def extract_clips(
    parquet_dir: Path,
    csv_path: Path,
    out_dir: Path,
    index_csv: Path,
    only_discrete: bool = True,
    keys: Optional[set[str]] = None,
) -> ExtractionReport:
    """Extract filtered FoleyBench clips to ``out_dir/<key>.mp4`` and index them.

    Filter (a row must pass ALL that apply):
      * ``keys``: if given, the row key must be in this set;
      * metadata row must exist in *csv_path* for the key (otherwise the clip's
        Discrete/source labels are unknowable — counted as filtered out);
      * ``only_discrete``: metadata ``discrete_vs_rest == 'Discrete'``.

    Writes are ATOMIC: ``<key>.mp4.tmp`` then ``os.replace`` — a crash never
    leaves a partial final file.  If the final file already exists the row is
    skipped (``n_skipped_existing``); its index row is recomputed from disk
    only when missing.  Decode failures still get an index row with status
    ``decode_error`` and empty path.  The index CSV is rewritten WHOLE on every
    call (collect-then-write), so reruns never duplicate rows.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata_csv(csv_path)
    index_rows = _load_index_rows(index_csv)
    report = ExtractionReport()

    for row in iter_foleybench_rows(Path(parquet_dir), columns=["key", "video_data"]):
        report.n_seen += 1
        key = str(row.get("key", "")).strip()

        if keys is not None and key not in keys:
            report.n_filtered_out += 1
            continue
        meta = metadata.get(key)
        if meta is None:
            report.n_filtered_out += 1
            continue
        if only_discrete and meta.get("discrete_vs_rest") != "Discrete":
            report.n_filtered_out += 1
            continue

        final = out_dir / f"{key}.mp4"
        if final.exists():
            report.n_skipped_existing += 1
            if key not in index_rows:
                digest = hashlib.sha256(final.read_bytes()).hexdigest()
                index_rows[key] = _index_row(key, meta, str(final), digest, "ok")
            continue

        try:
            data = decode_video_bytes(row.get("video_data"))
        except VideoDecodeError:
            report.n_decode_error += 1
            index_rows[key] = _index_row(key, meta, "", "", "decode_error")
            continue

        tmp = out_dir / f"{key}.mp4.tmp"
        tmp.write_bytes(data)
        os.replace(tmp, final)
        digest = hashlib.sha256(data).hexdigest()
        index_rows[key] = _index_row(key, meta, str(final), digest, "ok")
        report.n_extracted += 1

    _write_index_rows(Path(index_csv), index_rows)
    return report


# ---------------------------------------------------------------------------
# PyAV second-pass validation
# ---------------------------------------------------------------------------

def _require_av():
    """Lazy-import PyAV with a clear error message."""
    try:
        import av  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "foley_cw.foleybench_extract.validate_clips_av requires PyAV "
            "(pip install av). The numpy core does not depend on it."
        ) from exc
    return av


def _av_clip_is_valid(av_mod: Any, path: str) -> bool:
    """True iff *path* opens with >=1 decodable video frame and duration > 0."""
    try:
        with av_mod.open(path) as container:
            vstreams = list(container.streams.video)
            if not vstreams:
                return False
            vs = vstreams[0]
            n_frames = int(vs.frames or 0)
            if n_frames <= 0:
                # Some muxers omit the frame count; decoding one frame proves > 0.
                for _frame in container.decode(vs):
                    n_frames = 1
                    break
            duration = container.duration or vs.duration or 0
            return n_frames > 0 and duration > 0
    except Exception:
        return False


def validate_clips_av(index_csv: Path) -> dict:
    """Second pass: open every status=ok clip with PyAV and demote failures.

    A clip passes iff it opens, has a video stream with > 0 frames, and a
    positive duration.  Failing rows have status flipped to ``decode_error``
    in the index (path kept for inspection).  Returns
    ``{"n_ok": int, "n_failed": int, "ok_rate": float}`` over the checked rows
    (``ok_rate = 0.0`` when there is nothing to check, which the CLI gate
    treats as a halt).
    """
    av_mod = _require_av()
    index_csv = Path(index_csv)
    rows = _load_index_rows(index_csv)

    n_ok = 0
    n_failed = 0
    for key, row in rows.items():
        if row.get("status") != "ok":
            continue
        if _av_clip_is_valid(av_mod, row.get("path", "")):
            n_ok += 1
        else:
            n_failed += 1
            rows[key] = {**row, "status": "decode_error"}

    _write_index_rows(index_csv, rows)
    n_checked = n_ok + n_failed
    ok_rate = (n_ok / n_checked) if n_checked > 0 else 0.0
    return {"n_ok": n_ok, "n_failed": n_failed, "ok_rate": float(ok_rate)}


# ---------------------------------------------------------------------------
# Stratified selection (Stage M + screening pool)
# ---------------------------------------------------------------------------

def _eligible_pool(index_csv: Path, exclude: frozenset[str] | set[str]) -> dict[str, list[str]]:
    """Selection pool keys grouped by ucs_category.

    Pool = status ok AND source_type 'Single-source' AND discrete_vs_rest
    'Discrete' (manual §3.1: single-event candidate pool), minus *exclude*.
    Within-category lists are in natural key order (pre-shuffle determinism).
    """
    rows = _load_index_rows(Path(index_csv))
    by_cat: dict[str, list[str]] = {}
    for key in sorted(rows, key=_natural_key):
        row = rows[key]
        if (
            row.get("status") == "ok"
            and row.get("source_type") == "Single-source"
            and row.get("discrete_vs_rest") == "Discrete"
            and key not in exclude
        ):
            by_cat.setdefault(row.get("ucs_category", ""), []).append(key)
    return by_cat


def _stratified_select(
    by_cat: dict[str, list[str]],
    n: int,
    seed: int,
) -> list[str]:
    """Deterministic round-robin over seed-shuffled categories and members.

    Categories are shuffled once, members within each category are shuffled
    once, then one key is drawn per category per pass until *n* keys are taken
    or the pool is exhausted.  Same seed -> same list (np RNG only).
    """
    rng = np.random.default_rng(seed)
    cats = sorted(by_cat)
    cat_order = [cats[i] for i in rng.permutation(len(cats))]
    queues: dict[str, list[str]] = {}
    for cat in cat_order:
        members = by_cat[cat]
        queues[cat] = [members[i] for i in rng.permutation(len(members))]

    selected: list[str] = []
    while len(selected) < n and any(queues[c] for c in cat_order):
        for cat in cat_order:
            if len(selected) >= n:
                break
            if queues[cat]:
                selected.append(queues[cat].pop(0))
    return selected


def select_stage_m_clips(index_csv: Path, n: int = 16, seed: int = 0) -> list[str]:
    """Stage-M clip selection (manual §2): n keys, stratified by ucs_category.

    Pool: status ok, Single-source, Discrete.  Deterministic for a given seed;
    returns fewer than *n* keys when the pool is smaller.
    """
    return _stratified_select(_eligible_pool(index_csv, frozenset()), n, seed)


def select_screening_pool(
    index_csv: Path,
    n: int = 400,
    exclude: set[str] = frozenset(),
    seed: int = 0,
) -> list[str]:
    """Phase-0.3 screening candidate pool (manual §3.1): same stratified rule.

    Same pool as Stage M minus *exclude* (Stage-M clips are not carried
    forward).  Deterministic for a given seed.
    """
    return _stratified_select(_eligible_pool(index_csv, frozenset(exclude)), n, seed)


# ---------------------------------------------------------------------------
# Manifest writing
# ---------------------------------------------------------------------------

def write_manifest_json(path: Path, payload: dict) -> None:
    """Atomically write *payload* as sorted-keys JSON with a generated_by field.

    The payload is not mutated; ``generated_by`` is added only when absent so
    callers can override provenance.  Write is tmp-then-``os.replace``.
    """
    record = dict(payload)
    record.setdefault("generated_by", _GENERATED_BY)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)
