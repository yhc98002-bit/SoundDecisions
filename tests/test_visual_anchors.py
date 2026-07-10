"""Tests for foley_cw/visual_anchors.py — Stage-0 event anchors (manual §3.2).

All tests run on CPU, no network, no GPU, no real FoleyBench data. Optional deps
(librosa, cv2, scipy, av) are gated with pytest.importorskip; the synthetic video
tests additionally guard cv2.VideoWriter codec availability (skip if mp4v cannot
encode in this environment).

Key contracts checked here:
  * audio_onsets recovers a known two-click train within ±50 ms, salience-descending,
    capped at max_events; silent audio gives no onsets.
  * visual_onsets localises an abrupt appearance/move of a white square to within
    ±2 frames at 12 fps and returns positive half-widths.
  * anchors_for_clip on a silent (no-audio-stream) mp4 degrades to audio=None with
    sigma_s = NaN while still producing the visual anchor.
  * summarize_sigma: median/mean/max over clips with both anchors, NaN handling,
    coverage fractions, and the recommended-bin rule max(0.5, 2·median σ).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from foley_cw.dataset import EventAnchor
from foley_cw.visual_anchors import summarize_sigma


# --------------------------------------------------------------------------------------
# Synthetic signal helpers
# --------------------------------------------------------------------------------------

def _click_train(sr: int, duration_s: float, click_times: list[float], seed: int = 0) -> np.ndarray:
    """Mono waveform with short broadband noise bursts (clicks) at the given times."""
    rng = np.random.default_rng(seed)
    wav = np.zeros(int(sr * duration_s), dtype=np.float32)
    burst_len = int(0.004 * sr)  # 4 ms burst
    envelope = np.exp(-np.arange(burst_len) / (burst_len / 4.0)).astype(np.float32)
    for t in click_times:
        n0 = int(t * sr)
        burst = 0.9 * envelope * rng.standard_normal(burst_len).astype(np.float32)
        wav[n0:n0 + burst_len] += burst[: max(0, wav.size - n0)]
    return wav


def _write_motion_video(
    path: Path,
    cv2,
    fps: float = 12.0,
    size: int = 64,
    n_frames: int = 36,
    onset_frame: int = 18,
) -> bool:
    """Write an mp4: black frames, then a white square appears at onset_frame and
    moves 2 px/frame. Returns False when the codec/container is unusable here."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (size, size))
    if not writer.isOpened():
        return False
    for i in range(n_frames):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        if i >= onset_frame:
            offset = 8 + 2 * (i - onset_frame)
            x0 = min(offset, size - 16)
            frame[24:40, x0:x0 + 16, :] = 255
        writer.write(frame)
    writer.release()
    # Readback guard: some builds write an unreadable/empty container.
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    return bool(ok)


# --------------------------------------------------------------------------------------
# audio_onsets
# --------------------------------------------------------------------------------------

class TestAudioOnsets:
    SR = 16000
    CLICKS = [0.3, 0.7]

    def test_two_clicks_recovered_within_50ms(self):
        pytest.importorskip("librosa")
        from foley_cw.visual_anchors import audio_onsets

        wav = _click_train(self.SR, 1.2, self.CLICKS, seed=0)
        onsets = audio_onsets(wav, self.SR, max_events=4)
        assert len(onsets) >= 2
        times = [t for t, _ in onsets]
        for true_t in self.CLICKS:
            err = min(abs(t - true_t) for t in times)
            assert err <= 0.05, f"click at {true_t}s recovered with error {err * 1e3:.1f}ms"

    def test_salience_descending_and_in_range(self):
        pytest.importorskip("librosa")
        from foley_cw.visual_anchors import audio_onsets

        wav = _click_train(self.SR, 1.2, self.CLICKS, seed=1)
        onsets = audio_onsets(wav, self.SR)
        saliences = [s for _, s in onsets]
        assert saliences == sorted(saliences, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in saliences)

    def test_max_events_caps_output(self):
        pytest.importorskip("librosa")
        from foley_cw.visual_anchors import audio_onsets

        wav = _click_train(self.SR, 1.2, self.CLICKS, seed=2)
        onsets = audio_onsets(wav, self.SR, max_events=1)
        assert len(onsets) == 1

    def test_silence_gives_no_onsets(self):
        pytest.importorskip("librosa")
        from foley_cw.visual_anchors import audio_onsets

        assert audio_onsets(np.zeros(self.SR, dtype=np.float32), self.SR) == []

    def test_too_short_input_gives_no_onsets(self):
        pytest.importorskip("librosa")
        from foley_cw.visual_anchors import audio_onsets

        assert audio_onsets(np.zeros(64, dtype=np.float32), self.SR) == []


# --------------------------------------------------------------------------------------
# visual_onsets
# --------------------------------------------------------------------------------------

class TestVisualOnsets:
    FPS = 12.0
    ONSET_FRAME = 18

    @pytest.fixture
    def video_path(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        pytest.importorskip("scipy")
        path = tmp_path / "motion.mp4"
        if not _write_motion_video(path, cv2, fps=self.FPS, onset_frame=self.ONSET_FRAME):
            pytest.skip("cv2 VideoWriter mp4v codec unavailable in this environment")
        return path

    def test_onset_within_two_frames(self, video_path):
        from foley_cw.visual_anchors import visual_onsets

        onsets = visual_onsets(video_path, max_events=4)
        assert len(onsets) >= 1
        t_top = onsets[0][0]  # highest-salience event
        expected = self.ONSET_FRAME / self.FPS
        tol = 2.0 / self.FPS + 1e-9
        assert abs(t_top - expected) <= tol, (
            f"visual onset at {t_top:.3f}s, expected {expected:.3f}s ± {tol:.3f}s"
        )

    def test_tuple_shape_and_half_width_positive(self, video_path):
        from foley_cw.visual_anchors import visual_onsets

        onsets = visual_onsets(video_path)
        for tup in onsets:
            assert len(tup) == 3
            t, salience, half_width = tup
            assert t >= 0.0
            assert salience > 0.0
            assert half_width > 0.0

    def test_salience_descending_and_capped(self, video_path):
        from foley_cw.visual_anchors import visual_onsets

        onsets = visual_onsets(video_path, max_events=2)
        assert len(onsets) <= 2
        saliences = [s for _, s, _ in onsets]
        assert saliences == sorted(saliences, reverse=True)

    def test_missing_file_returns_empty(self, tmp_path):
        pytest.importorskip("cv2")
        pytest.importorskip("scipy")
        from foley_cw.visual_anchors import visual_onsets

        assert visual_onsets(tmp_path / "nope.mp4") == []


# --------------------------------------------------------------------------------------
# anchors_for_clip (degrade path: cv2-authored mp4 has no audio stream)
# --------------------------------------------------------------------------------------

class TestAnchorsForClip:
    def test_no_audio_stream_degrades_to_visual_only(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        pytest.importorskip("scipy")
        pytest.importorskip("av")
        from foley_cw.visual_anchors import anchors_for_clip

        path = tmp_path / "silent.mp4"
        if not _write_motion_video(path, cv2):
            pytest.skip("cv2 VideoWriter mp4v codec unavailable in this environment")

        rec = anchors_for_clip(path)
        assert set(rec) == {"audio", "visual", "sigma_s"}
        assert rec["audio"] is None
        assert isinstance(rec["visual"], EventAnchor)
        assert rec["visual"].source == "visual_onset_detector"
        assert rec["visual"].n_events >= 1
        # Visual uncertainty is the per-event half-width (positive seconds).
        assert all(u > 0.0 for u in rec["visual"].uncertainty)
        assert math.isnan(rec["sigma_s"])


# --------------------------------------------------------------------------------------
# summarize_sigma (pure numpy; no optional deps)
# --------------------------------------------------------------------------------------

def _rec(audio_ts=None, visual_ts=None, sigma=float("nan")):
    audio = None
    if audio_ts is not None:
        audio = EventAnchor(
            timestamps=list(audio_ts),
            uncertainty=[0.05] * len(audio_ts),
            source="foleybench_audio_onset",
        )
    visual = None
    if visual_ts is not None:
        visual = EventAnchor(
            timestamps=list(visual_ts),
            uncertainty=[0.1] * len(visual_ts),
            source="visual_onset_detector",
        )
    return {"audio": audio, "visual": visual, "sigma_s": sigma}


class TestSummarizeSigma:
    def test_stats_and_floor_rule(self):
        """median σ = 0.2 → 2σ = 0.4 < 0.5 floor → recommended bin = 0.5."""
        records = [
            _rec(audio_ts=[1.0], visual_ts=[1.1], sigma=0.1),
            _rec(audio_ts=[2.0], visual_ts=[2.3], sigma=0.3),
            _rec(audio_ts=[3.0], visual_ts=None, sigma=float("nan")),
        ]
        s = summarize_sigma(records)
        assert s["n_clips"] == 3
        assert s["median_sigma_s"] == pytest.approx(0.2)
        assert s["mean_sigma_s"] == pytest.approx(0.2)
        assert s["max_sigma_s"] == pytest.approx(0.3)
        assert s["coverage_audio"] == pytest.approx(1.0)
        assert s["coverage_visual"] == pytest.approx(2.0 / 3.0)
        assert s["coverage_both"] == pytest.approx(2.0 / 3.0)
        assert s["recommended_bin_s"] == pytest.approx(0.5)

    def test_two_sigma_rule_above_floor(self):
        """median σ = 0.5 → 2σ = 1.0 > 0.5 floor → recommended bin = 1.0."""
        records = [
            _rec(audio_ts=[1.0], visual_ts=[1.4], sigma=0.4),
            _rec(audio_ts=[2.0], visual_ts=[2.6], sigma=0.6),
        ]
        s = summarize_sigma(records)
        assert s["median_sigma_s"] == pytest.approx(0.5)
        assert s["recommended_bin_s"] == pytest.approx(1.0)

    def test_all_missing_gives_nan_stats_and_zero_coverage(self):
        records = [_rec(), _rec(audio_ts=[1.0]), _rec(visual_ts=[2.0])]
        s = summarize_sigma(records)
        assert math.isnan(s["median_sigma_s"])
        assert math.isnan(s["mean_sigma_s"])
        assert math.isnan(s["max_sigma_s"])
        # No clip has both anchors -> no σ evidence -> no bin recommendation.
        assert math.isnan(s["recommended_bin_s"])
        assert s["coverage_audio"] == pytest.approx(1.0 / 3.0)
        assert s["coverage_visual"] == pytest.approx(1.0 / 3.0)
        assert s["coverage_both"] == pytest.approx(0.0)

    def test_empty_records(self):
        s = summarize_sigma([])
        assert s["n_clips"] == 0
        for k in ("median_sigma_s", "mean_sigma_s", "max_sigma_s",
                  "coverage_audio", "coverage_visual", "coverage_both",
                  "recommended_bin_s"):
            assert math.isnan(s[k])

    def test_accepts_json_style_anchor_dicts(self):
        """Records round-tripped through anchors.json (dicts, not EventAnchor)."""
        records = [
            {
                "audio": {"timestamps": [1.0], "uncertainty": [0.05],
                          "source": "foleybench_audio_onset"},
                "visual": {"timestamps": [1.2], "uncertainty": [0.1],
                           "source": "visual_onset_detector"},
                "sigma_s": 0.2,
            },
            {"audio": None, "visual": None, "sigma_s": float("nan")},
        ]
        s = summarize_sigma(records)
        assert s["coverage_both"] == pytest.approx(0.5)
        assert s["median_sigma_s"] == pytest.approx(0.2)


# --------------------------------------------------------------------------------------
# Import safety
# --------------------------------------------------------------------------------------

def test_module_importable_without_heavy_deps():
    """foley_cw.visual_anchors must import with numpy only (librosa/cv2/av/scipy
    are lazy inside the functions)."""
    import foley_cw.visual_anchors  # noqa: F401
