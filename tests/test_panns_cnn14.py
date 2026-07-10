"""Tests for foley_cw/measurers_panns_cnn14.py — vendored PANNs Cnn14_16k.

CPU-only, NO network, NO checkpoint, NO GPU: random-init forward contract, upstream
checkpoint-key compatibility (names only; weights need the real .pth), and the
AudioSet label-csv parser.  torch is required for the module import; torchlibrosa is
required only for model construction (both skipped via importorskip when absent).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from foley_cw.measurers_panns_cnn14 import (  # noqa: E402
    CLASSES_NUM,
    EMBEDDING_DIM,
    Cnn14_16k,
    load_audioset_labels,
    load_cnn14_16k,
)


@pytest.fixture(scope="module")
def model():
    """Random-init eval-mode Cnn14_16k (construction needs torchlibrosa)."""
    pytest.importorskip("torchlibrosa")
    torch.manual_seed(0)
    m = Cnn14_16k()
    m.eval()
    return m


# --------------------------------------------------------------------------------------
# Forward contract (random weights; checks shapes/ranges, not semantics)
# --------------------------------------------------------------------------------------

class TestForward:
    def test_output_shapes_and_ranges(self, model):
        rng = np.random.default_rng(0)
        wav = (rng.standard_normal((2, 16000)) * 0.1).astype(np.float32)
        with torch.no_grad():
            out = model(torch.from_numpy(wav))
        assert set(out.keys()) == {"clipwise_output", "embedding"}
        clipwise = out["clipwise_output"]
        embedding = out["embedding"]
        assert tuple(clipwise.shape) == (2, CLASSES_NUM) == (2, 527)
        assert tuple(embedding.shape) == (2, EMBEDDING_DIM) == (2, 2048)
        assert torch.isfinite(clipwise).all()
        assert torch.isfinite(embedding).all()
        assert bool((clipwise >= 0.0).all()) and bool((clipwise <= 1.0).all())

    def test_eval_mode_is_deterministic(self, model):
        """eval() bypasses SpecAugmentation and dropout: repeated forward is identical."""
        assert not model.training
        rng = np.random.default_rng(1)
        wav = torch.from_numpy((rng.standard_normal((1, 16000)) * 0.1).astype(np.float32))
        with torch.no_grad():
            a = model(wav)
            b = model(wav)
        assert torch.equal(a["clipwise_output"], b["clipwise_output"])
        assert torch.equal(a["embedding"], b["embedding"])

    def test_batch_size_one(self, model):
        rng = np.random.default_rng(2)
        wav = torch.from_numpy((rng.standard_normal((1, 16000)) * 0.1).astype(np.float32))
        with torch.no_grad():
            out = model(wav)
        assert tuple(out["clipwise_output"].shape) == (1, 527)
        assert tuple(out["embedding"].shape) == (1, 2048)


# --------------------------------------------------------------------------------------
# Architecture fidelity (checkpoint strict=True compatibility, names + shapes)
# --------------------------------------------------------------------------------------

class TestArchitecture:
    def test_conv_chain_channel_progression(self, model):
        expected = [(1, 64), (64, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048)]
        for i, (cin, cout) in enumerate(expected, start=1):
            block = getattr(model, f"conv_block{i}")
            assert block.conv1.in_channels == cin
            assert block.conv1.out_channels == cout
            assert block.conv2.in_channels == cout
            assert block.conv2.out_channels == cout

    def test_head_shapes(self, model):
        assert model.fc1.in_features == 2048 and model.fc1.out_features == 2048
        assert model.fc_audioset.in_features == 2048
        assert model.fc_audioset.out_features == 527
        assert model.bn0.num_features == 64

    def test_state_dict_has_upstream_checkpoint_keys(self, model):
        """Key names must match the released checkpoint for strict=True loading."""
        keys = set(model.state_dict().keys())
        expected = {
            "bn0.weight",
            "bn0.bias",
            "fc1.weight",
            "fc1.bias",
            "fc_audioset.weight",
            "fc_audioset.bias",
            "spectrogram_extractor.stft.conv_real.weight",
            "spectrogram_extractor.stft.conv_imag.weight",
            "logmel_extractor.melW",
        }
        for i in range(1, 7):
            for sub in (
                "conv1.weight",
                "conv2.weight",
                "bn1.weight",
                "bn1.bias",
                "bn2.weight",
                "bn2.bias",
            ):
                expected.add(f"conv_block{i}.{sub}")
        missing = expected - keys
        assert not missing, f"missing upstream checkpoint keys: {sorted(missing)}"

    def test_wrong_front_end_config_rejected(self):
        pytest.importorskip("torchlibrosa")
        with pytest.raises(ValueError):
            Cnn14_16k(sample_rate=32000)
        with pytest.raises(ValueError):
            Cnn14_16k(hop_size=320)


# --------------------------------------------------------------------------------------
# Checkpoint loader (no checkpoint available in CI: only the no-download guard)
# --------------------------------------------------------------------------------------

class TestLoader:
    def test_missing_checkpoint_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_cnn14_16k(tmp_path / "Cnn14_16k_mAP=0.438.pth")


# --------------------------------------------------------------------------------------
# AudioSet label csv parser
# --------------------------------------------------------------------------------------

class TestLoadAudiosetLabels:
    def _write(self, tmp_path, text):
        p = tmp_path / "class_labels_indices.csv"
        p.write_text(text, encoding="utf-8")
        return p

    def test_parses_three_row_fake_csv(self, tmp_path):
        p = self._write(
            tmp_path,
            "index,mid,display_name\n"
            '0,/m/09x0r,"Speech"\n'
            '2,/m/05zppz,"Male speech, man speaking"\n'
            '1,/m/0k4j,"Car"\n',
        )
        labels = load_audioset_labels(p)
        # Ordered by the index column, quoted commas preserved.
        assert labels == ["Speech", "Car", "Male speech, man speaking"]

    def test_returns_list_of_str(self, tmp_path):
        p = self._write(tmp_path, "index,mid,display_name\n0,/m/0,A\n1,/m/1,B\n2,/m/2,C\n")
        labels = load_audioset_labels(p)
        assert isinstance(labels, list)
        assert all(isinstance(s, str) for s in labels)
        assert labels == ["A", "B", "C"]

    def test_non_contiguous_indices_rejected(self, tmp_path):
        p = self._write(tmp_path, "index,mid,display_name\n0,/m/0,A\n2,/m/2,C\n")
        with pytest.raises(ValueError):
            load_audioset_labels(p)

    def test_wrong_columns_rejected(self, tmp_path):
        p = self._write(tmp_path, "idx,name\n0,A\n")
        with pytest.raises(ValueError):
            load_audioset_labels(p)
