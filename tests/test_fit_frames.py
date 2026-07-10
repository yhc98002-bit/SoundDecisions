"""Test the encoded-feature seq-length normalization that keeps off-by-one clips
from crashing MMAudio's preprocess_conditions."""
import pytest
torch = pytest.importorskip("torch")
from foley_cw.mmaudio_backend import _fit_seq


def test_exact_passthrough():
    f = torch.arange(64 * 1024).reshape(1, 64, 1024).float()
    assert _fit_seq(f, 64) is f


def test_pad_clip_63_to_64():
    f = torch.arange(63 * 4).reshape(1, 63, 4).float()
    out = _fit_seq(f, 64)
    assert out.shape == (1, 64, 4)
    assert torch.equal(out[:, :63], f)
    assert torch.equal(out[:, 63], f[:, 62])  # last feature repeated


def test_pad_sync_184_to_192():
    f = torch.zeros(1, 184, 768)
    assert _fit_seq(f, 192).shape == (1, 192, 768)


def test_truncate():
    f = torch.arange(1 * 66 * 4).reshape(1, 66, 4).float()
    out = _fit_seq(f, 64)
    assert out.shape == (1, 64, 4)
    assert torch.equal(out, f[:, :64])
