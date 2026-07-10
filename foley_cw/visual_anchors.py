"""Event-anchor extraction pipeline for Stage 0 (manual §3.2 — Phase 0.4 event anchors).

Serves experiment/LONG_RANGE_EXPERIMENT_PLAN.md §3.2: visible-event anchors with
per-event uncertainty, the audio-vs-visual disagreement sigma (σ_anchor), and the
propagation rule "gross-timing bins ≥ 2·σ_anchor" (floored at 0.5 s for the gross
timing axis).

Contract
--------
  * audio_onsets(wav, sr)        — librosa spectral-flux onsets on a decoded waveform,
                                   FIXED hop/peak-pick parameters (module constants),
                                   returned salience-descending.
  * extract_audio_track(path)    — PyAV decode of an mp4's audio stream to mono
                                   float32 at target_sr (librosa resample if needed).
  * visual_onsets(path)          — cv2 frame-diff + Farneback dense-flow motion energy;
                                   onsets = positive peaks of the temporal derivative
                                   of motion energy (scipy find_peaks, documented
                                   prominence), returned salience-descending.
  * anchors_for_clip(path)       — both EventAnchors plus the per-clip
                                   sigma_s = |primary audio onset − nearest visual onset|.
  * summarize_sigma(records)     — σ_anchor stats over clips, coverage, and the
                                   recommended gross-timing bin width max(0.5, 2·median σ).

Anchor-source provenance (manual §3.2 source chain is: foleybench_metadata →
visual_onset_detector → light_human_marks):
  * 'visual_onset_detector' anchors produced here ARE in the approved chain.
  * 'foleybench_audio_onset' anchors (onsets of the clip's OWN audio track) are NOT in
    the approved chain; they are a PROPOSED AMENDMENT pending PI approval, used here to
    estimate σ_anchor cheaply and to seed the 30-clip human check set. They must not be
    silently promoted to the frozen anchor source.

Ordering convention: all onset lists and the EventAnchor.timestamps built from them are
SALIENCE-DESCENDING; index 0 is the primary event. (dataset.EventAnchor itself imposes
no ordering.)

numpy + foley_cw.dataset only at import time; librosa / av / cv2 / scipy are imported
lazily inside the functions that need them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from .dataset import EventAnchor

# ---------------------------------------------------------------------------
# Fixed, documented parameters (frozen for Stage 0; do not tune per clip)
# ---------------------------------------------------------------------------

#: STFT hop for the librosa onset envelope (frames are hop/sr seconds apart;
#: 512 @ 16 kHz = 32 ms — well inside the ±50 ms audio-anchor tolerance).
_HOP_LENGTH: int = 512
#: peak_pick window parameters, in onset-envelope FRAMES (librosa.util.peak_pick).
_PRE_MAX: int = 3
_POST_MAX: int = 3
_PRE_AVG: int = 3
_POST_AVG: int = 5
#: peak_pick threshold above the local moving average, on the max-normalised envelope.
_DELTA: float = 0.07
#: peak_pick minimum gap between consecutive onsets, in frames.
_WAIT: int = 4

#: Per-event 1-sigma uncertainty assigned to audio-track onsets (seconds).
#: ~1.5 envelope frames at 16 kHz / hop 512; fixed by the task spec.
_AUDIO_ONSET_UNCERTAINTY_S: float = 0.05

#: find_peaks prominence for visual onsets, as a fraction of the peak-to-peak range of
#: the motion-energy temporal derivative (relative so it is invariant to frame size
#: and absolute pixel scale).
_PEAK_PROMINENCE_FRAC: float = 0.10

#: Floor for the recommended gross-timing bin width (seconds). The propagation rule is
#: bins ≥ 2·σ_anchor; 0.5 s is the minimum meaningful "gross timing" bin.
_BIN_FLOOR_S: float = 0.5


# ---------------------------------------------------------------------------
# audio_onsets
# ---------------------------------------------------------------------------

def audio_onsets(
    wav: np.ndarray,
    sr: int,
    max_events: int = 4,
) -> list[tuple[float, float]]:
    """Detect audio onsets on a decoded waveform with FIXED parameters.

    Pipeline: librosa.onset.onset_strength (hop=_HOP_LENGTH) → max-normalise the
    envelope → librosa.onset.onset_detect / peak_pick with the module-constant
    parameters (_PRE_MAX/_POST_MAX/_PRE_AVG/_POST_AVG/_DELTA/_WAIT, backtrack=False).

    Parameters
    ----------
    wav:
        Mono waveform (any float dtype; flattened).
    sr:
        Sample rate of *wav* in Hz.
    max_events:
        Maximum number of onsets returned.

    Returns
    -------
    list of (time_s, salience) tuples, SALIENCE-DESCENDING, at most *max_events* long.
    salience = max-normalised onset-envelope value at the detected frame, in [0, 1].
    Empty list for silent / too-short input.
    """
    try:
        import librosa
    except ImportError as e:  # pragma: no cover - env-dependent
        raise ImportError(
            "audio_onsets requires librosa (pip install librosa)."
        ) from e

    y = np.asarray(wav, dtype=np.float32).ravel()
    if y.size < 2 * _HOP_LENGTH:
        return []

    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_HOP_LENGTH)
    env_max = float(np.max(env)) if env.size else 0.0
    if not np.isfinite(env_max) or env_max <= 0.0:
        return []  # silent clip: no spectral flux anywhere
    env_norm = env / env_max

    frames = librosa.onset.onset_detect(
        onset_envelope=env_norm,
        sr=sr,
        hop_length=_HOP_LENGTH,
        backtrack=False,
        units="frames",
        normalize=False,  # already max-normalised; salience read off env_norm
        pre_max=_PRE_MAX,
        post_max=_POST_MAX,
        pre_avg=_PRE_AVG,
        post_avg=_POST_AVG,
        delta=_DELTA,
        wait=_WAIT,
    )
    if frames.size == 0:
        return []

    times = librosa.frames_to_time(frames, sr=sr, hop_length=_HOP_LENGTH)
    events = [
        (float(t), float(env_norm[int(f)]))
        for t, f in zip(times, frames)
    ]
    events.sort(key=lambda e: -e[1])
    return events[:max_events]


# ---------------------------------------------------------------------------
# extract_audio_track
# ---------------------------------------------------------------------------

def extract_audio_track(
    video_path: Path,
    target_sr: int = 16000,
) -> tuple[np.ndarray, int]:
    """Decode the first audio stream of an mp4 to mono float32 at *target_sr*.

    PyAV decode at the native sample rate; channels are down-mixed by mean;
    resampling (when native rate != target_sr) uses librosa.resample.

    Raises
    ------
    ImportError
        If PyAV (or librosa, when resampling is needed) is unavailable.
    ValueError
        If the container has no audio stream or no decodable audio frames.

    Returns
    -------
    (wav, sr): mono float32 waveform in [-1, 1] and the (target) sample rate.
    """
    try:
        import av
    except ImportError as e:  # pragma: no cover - env-dependent
        raise ImportError(
            "extract_audio_track requires PyAV (pip install av)."
        ) from e

    chunks: list[np.ndarray] = []
    native_sr: Optional[int] = None
    with av.open(str(video_path)) as container:
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
            raise ValueError(f"no audio stream in {video_path}")
        stream = audio_streams[0]
        for frame in container.decode(stream):
            native_sr = int(frame.sample_rate)
            arr = frame.to_ndarray()
            if arr.ndim == 1:
                arr = arr[None, :]
            # Integer PCM → float32 in [-1, 1].
            if arr.dtype.kind == "f":
                arr = arr.astype(np.float32)
            elif arr.dtype == np.int16:
                arr = arr.astype(np.float32) / 32768.0
            elif arr.dtype == np.int32:
                arr = arr.astype(np.float32) / 2147483648.0
            elif arr.dtype == np.uint8:
                arr = (arr.astype(np.float32) - 128.0) / 128.0
            else:
                arr = arr.astype(np.float32)
            # Packed (interleaved) multi-channel comes back as (1, n*ch).
            layout = frame.layout
            nch = getattr(layout, "nb_channels", None)
            if nch is None:
                nch = len(getattr(layout, "channels", ())) or arr.shape[0]
            if not frame.format.is_planar and nch > 1 and arr.shape[0] == 1:
                arr = arr.reshape(-1, nch).T
            chunks.append(arr)

    if not chunks or native_sr is None:
        raise ValueError(f"no decodable audio frames in {video_path}")

    mono = np.concatenate(chunks, axis=1).mean(axis=0)
    if native_sr != target_sr:
        try:
            import librosa
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError(
                "extract_audio_track requires librosa to resample "
                f"{native_sr} Hz → {target_sr} Hz."
            ) from e
        mono = librosa.resample(mono, orig_sr=native_sr, target_sr=target_sr)
    return np.ascontiguousarray(mono, dtype=np.float32), int(target_sr)


# ---------------------------------------------------------------------------
# visual_onsets
# ---------------------------------------------------------------------------

def visual_onsets(
    video_path: Path,
    max_events: int = 4,
    downscale: int = 4,
    max_fps: float = 12.0,
) -> list[tuple[float, float, float]]:
    """Detect visual motion onsets via frame-diff + Farneback flow energy.

    Pipeline (all parameters fixed / documented):
      1. cv2.VideoCapture; grayscale frames downscaled by *downscale* (INTER_AREA),
         sampled at ≤ *max_fps* (integer frame stride).
      2. Per sampled frame, motion energy = mean |Δgray| / 255 + mean Farneback flow
         magnitude / frame diagonal. Both terms are dimensionless and O(1), so the
         sum is scale-invariant across resolutions; flow IS computed at every
         sampled frame.
      3. Onset candidates = positive peaks of the temporal derivative of motion
         energy: scipy.signal.find_peaks with prominence =
         _PEAK_PROMINENCE_FRAC × peak-to-peak range of the derivative.
      4. half_width_s = half the FWHM of the derivative peak (scipy peak_widths,
         rel_height=0.5), floored at half the sampling interval — used downstream
         as the per-event anchor uncertainty.

    Returns
    -------
    list of (time_s, salience, half_width_s), SALIENCE-DESCENDING, at most
    *max_events* long. salience = peak prominence of the energy derivative.
    Empty list when the video is unreadable or has < 4 sampled frames.
    """
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - env-dependent
        raise ImportError(
            "visual_onsets requires OpenCV (pip install opencv-python-headless)."
        ) from e
    try:
        from scipy.signal import find_peaks, peak_widths
    except ImportError as e:  # pragma: no cover - env-dependent
        raise ImportError("visual_onsets requires scipy (pip install scipy).") from e

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 25.0  # container without fps metadata: assume PAL default
    stride = max(1, int(round(fps / max(max_fps, 1e-6))))
    sample_dt = stride / fps

    # energies[m] = (frame_index_of_current_frame, energy between previous and
    # current SAMPLED frames); the onset time is the CURRENT frame's timestamp.
    energies: list[tuple[int, float]] = []
    prev: Optional[np.ndarray] = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            small = cv2.resize(
                gray,
                (max(1, w // downscale), max(1, h // downscale)),
                interpolation=cv2.INTER_AREA,
            )
            if prev is not None:
                diff_term = float(
                    np.mean(np.abs(small.astype(np.float32) - prev.astype(np.float32)))
                ) / 255.0
                sh, sw = small.shape
                # Farneback parameters adapted to small frames: winsize and pyramid
                # depth must not exceed the (downscaled) image size.
                win = int(max(5, min(15, min(sh, sw) // 2)))
                levels = 1 if min(sh, sw) < 32 else 3
                flow = cv2.calcOpticalFlowFarneback(
                    prev, small, None, 0.5, levels, win, 3, 5, 1.1, 0
                )
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                flow_term = float(np.mean(mag)) / float(np.hypot(sh, sw))
                energies.append((idx, diff_term + flow_term))
            prev = small
        idx += 1
    cap.release()

    if len(energies) < 3:
        return []

    energy = np.array([e for _, e in energies], dtype=float)
    d = np.diff(energy)  # d[j] = energy[j+1] - energy[j]; onset at energy sample j+1
    ptp = float(d.max() - d.min())
    if not np.isfinite(ptp) or ptp <= 0.0:
        return []  # static clip: no motion change anywhere
    prominence = max(_PEAK_PROMINENCE_FRAC * ptp, 1e-9)

    # Pad with the derivative minimum so an abrupt onset adjacent to the clip
    # boundary still registers as a local maximum.
    pad = float(d.min())
    dp = np.concatenate(([pad], d, [pad]))
    peaks, props = find_peaks(dp, prominence=prominence)
    if peaks.size == 0:
        return []
    widths = peak_widths(dp, peaks, rel_height=0.5)[0]

    events: list[tuple[float, float, float]] = []
    for p, prom, width in zip(peaks, props["prominences"], widths):
        energy_idx = int(p)  # padded index p ↔ d index p-1 ↔ energy sample p
        if energy_idx >= len(energies):
            continue
        frame_idx = energies[energy_idx][0]
        time_s = frame_idx / fps
        half_width_s = max(0.5 * sample_dt, 0.5 * float(width) * sample_dt)
        events.append((float(time_s), float(prom), float(half_width_s)))

    events.sort(key=lambda e: -e[1])
    return events[:max_events]


# ---------------------------------------------------------------------------
# anchors_for_clip
# ---------------------------------------------------------------------------

def anchors_for_clip(video_path: Path, sr: int = 16000) -> dict[str, Any]:
    """Compute audio-track and visual EventAnchors for one clip, plus sigma_s.

    Returns a dict with keys:
      'audio'  : EventAnchor(source='foleybench_audio_onset',
                 uncertainty=_AUDIO_ONSET_UNCERTAINTY_S per event) or None when the
                 clip has no audio stream / no detectable onsets. NOTE: this source
                 is a PROPOSED AMENDMENT to the §3.2 chain (see module docstring).
      'visual' : EventAnchor(source='visual_onset_detector',
                 uncertainty=per-event half_width_s) or None.
      'sigma_s': |primary (highest-salience) audio onset − nearest visual onset| in
                 seconds; NaN when either anchor is missing. This per-clip sigma feeds
                 σ_anchor and the gross-timing bin rule (manual §3.2).

    EventAnchor timestamps are SALIENCE-DESCENDING (timestamps[0] = primary event).
    Audio decode/detection failures degrade to 'audio': None (the visual chain is the
    approved fallback); ImportError for a missing dependency is re-raised.
    """
    video_path = Path(video_path)

    audio_anchor: Optional[EventAnchor] = None
    try:
        wav, wav_sr = extract_audio_track(video_path, target_sr=sr)
        a_onsets = audio_onsets(wav, wav_sr)
        if a_onsets:
            audio_anchor = EventAnchor(
                timestamps=[t for t, _ in a_onsets],
                uncertainty=[_AUDIO_ONSET_UNCERTAINTY_S] * len(a_onsets),
                source="foleybench_audio_onset",
            )
    except ImportError:
        raise
    except Exception:
        audio_anchor = None  # no/undecodable audio stream: visual chain still applies

    v_onsets = visual_onsets(video_path)
    visual_anchor: Optional[EventAnchor] = None
    if v_onsets:
        visual_anchor = EventAnchor(
            timestamps=[t for t, _, _ in v_onsets],
            uncertainty=[hw for _, _, hw in v_onsets],
            source="visual_onset_detector",
        )

    sigma_s = float("nan")
    if audio_anchor is not None and visual_anchor is not None:
        t_primary = audio_anchor.timestamps[0]
        sigma_s = float(min(abs(t_primary - tv) for tv in visual_anchor.timestamps))

    return {"audio": audio_anchor, "visual": visual_anchor, "sigma_s": sigma_s}


# ---------------------------------------------------------------------------
# summarize_sigma
# ---------------------------------------------------------------------------

def _has_events(anchor: Any) -> bool:
    """True if *anchor* (EventAnchor or JSON-style dict) holds ≥ 1 timestamp."""
    if anchor is None:
        return False
    if isinstance(anchor, EventAnchor):
        return anchor.n_events > 0
    if isinstance(anchor, dict):
        return len(anchor.get("timestamps", [])) > 0
    return False


def summarize_sigma(anchor_records: list[dict]) -> dict[str, float]:
    """Aggregate σ_anchor statistics over per-clip anchor records.

    Parameters
    ----------
    anchor_records:
        List of dicts shaped like anchors_for_clip output ('audio', 'visual',
        'sigma_s'); the anchors may be EventAnchor objects or their JSON dicts.

    Returns
    -------
    dict with keys:
      n_clips            : number of records.
      median_sigma_s     : median sigma over clips with BOTH anchors (finite sigma).
      mean_sigma_s       : mean of the same.
      max_sigma_s        : max of the same.
      coverage_audio     : fraction of clips with a non-empty audio anchor.
      coverage_visual    : fraction of clips with a non-empty visual anchor.
      coverage_both      : fraction with both.
      recommended_bin_s  : max(_BIN_FLOOR_S, 2·median_sigma_s) per the §3.2 rule
                           "gross-timing bins ≥ 2·σ_anchor"; NaN when no clip has
                           both anchors (no recommendation without evidence).
    All sigma stats are NaN when no clip has a finite sigma; coverages are NaN for
    an empty record list.
    """
    n = len(anchor_records)
    nan = float("nan")
    if n == 0:
        return {
            "n_clips": 0,
            "median_sigma_s": nan,
            "mean_sigma_s": nan,
            "max_sigma_s": nan,
            "coverage_audio": nan,
            "coverage_visual": nan,
            "coverage_both": nan,
            "recommended_bin_s": nan,
        }

    n_audio = sum(_has_events(r.get("audio")) for r in anchor_records)
    n_visual = sum(_has_events(r.get("visual")) for r in anchor_records)
    n_both = sum(
        _has_events(r.get("audio")) and _has_events(r.get("visual"))
        for r in anchor_records
    )

    sigmas = np.array(
        [float(r.get("sigma_s", nan)) for r in anchor_records], dtype=float
    )
    finite = sigmas[np.isfinite(sigmas)]

    if finite.size > 0:
        median_sigma = float(np.median(finite))
        mean_sigma = float(np.mean(finite))
        max_sigma = float(np.max(finite))
        recommended_bin = max(_BIN_FLOOR_S, 2.0 * median_sigma)
    else:
        median_sigma = mean_sigma = max_sigma = nan
        recommended_bin = nan

    return {
        "n_clips": n,
        "median_sigma_s": median_sigma,
        "mean_sigma_s": mean_sigma,
        "max_sigma_s": max_sigma,
        "coverage_audio": n_audio / n,
        "coverage_visual": n_visual / n,
        "coverage_both": n_both / n,
        "recommended_bin_s": recommended_bin,
    }
