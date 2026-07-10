"""Vendored PANNs Cnn14_16k audio tagger (GPU/torch seam; NOT part of the numpy core).

Serves manual §3.3 (experiment/LONG_RANGE_EXPERIMENT_PLAN.md, Phase 0.5 measurer
choices): "coarse class = 16 kHz-native tagger (BEATs or PANNs-16k) as primary".
This module is the PANNs-16k half of that pre-registered choice; it is imported
explicitly by the real-measurer wiring, never by the numpy-only core.

Faithful single-file port of ``Cnn14_16k`` from:
    qiuqiangkong/audioset_tagging_cnn
    https://github.com/qiuqiangkong/audioset_tagging_cnn
    MIT License, Copyright (c) 2019 Qiuqiang Kong.
    Paper: Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for
    Audio Pattern Recognition", IEEE/ACM TASLP 2020 (arXiv:1912.10211).

Port fidelity contract:
  * Module/parameter names match upstream exactly (spectrogram_extractor,
    logmel_extractor, spec_augmenter, bn0, conv_block1..6, fc1, fc_audioset) so the
    released checkpoint (Cnn14_16k_mAP=0.438.pth) loads with strict=True.
  * Front-end: torchlibrosa Spectrogram + LogmelFilterBank with the EXACT Cnn14_16k
    config (sample_rate=16000, window_size=512, hop_size=160, mel_bins=64, fmin=50,
    fmax=8000, classes_num=527) and SpecAugmentation that is bypassed in eval mode.
  * forward(input: (B, samples) float32 waveform) -> {'clipwise_output': (B, 527)
    sigmoid, 'embedding': (B, 2048)} — same contract as upstream.

This project does NOT download checkpoints; ``load_cnn14_16k`` only reads a local
file.  torchlibrosa is an optional dependency (guarded import below).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - torch present in the seam env
    raise ImportError(
        "foley_cw.measurers_panns_cnn14 requires torch (GPU-seam module; the numpy "
        "core must not import it). Install torch in the seam environment."
    ) from exc

# torchlibrosa is not installed in the base env yet; defer the failure to model
# construction so the module (and the csv label helper) stays importable.
try:
    from torchlibrosa.augmentation import SpecAugmentation
    from torchlibrosa.stft import LogmelFilterBank, Spectrogram

    _TORCHLIBROSA_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as exc:
    SpecAugmentation = None  # type: ignore[assignment]
    LogmelFilterBank = None  # type: ignore[assignment]
    Spectrogram = None  # type: ignore[assignment]
    _TORCHLIBROSA_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Exact Cnn14_16k configuration (upstream asserts these values)
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16000
WINDOW_SIZE: int = 512
HOP_SIZE: int = 160
MEL_BINS: int = 64
FMIN: int = 50
FMAX: int = 8000
CLASSES_NUM: int = 527
EMBEDDING_DIM: int = 2048


# ---------------------------------------------------------------------------
# Upstream init helpers (pytorch/models.py)
# ---------------------------------------------------------------------------

def init_layer(layer: nn.Module) -> None:
    """Initialize a Linear or Convolutional layer (upstream-identical)."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn: nn.Module) -> None:
    """Initialize a Batchnorm layer (upstream-identical)."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def do_mixup(x: torch.Tensor, mixup_lambda: torch.Tensor) -> torch.Tensor:
    """Mixup x of even indexes (0, 2, ...) with x of odd indexes (1, 3, ...).

    Upstream-identical; training-only path, unused in eval-mode measurement.
    """
    out = (
        x[0::2].transpose(0, -1) * mixup_lambda[0::2]
        + x[1::2].transpose(0, -1) * mixup_lambda[1::2]
    ).transpose(0, -1)
    return out


# ---------------------------------------------------------------------------
# ConvBlock (upstream-identical)
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Two 3x3 conv + BN + ReLU layers followed by pooling (upstream ConvBlock)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.init_weight()

    def init_weight(self) -> None:
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(
        self,
        input: torch.Tensor,
        pool_size: tuple[int, int] = (2, 2),
        pool_type: str = "avg",
    ) -> torch.Tensor:
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise ValueError(f"ConvBlock: incorrect pool_type {pool_type!r}")
        return x


# ---------------------------------------------------------------------------
# Cnn14_16k (upstream-identical architecture; checkpoint-compatible strict=True)
# ---------------------------------------------------------------------------

class Cnn14_16k(nn.Module):
    """PANNs Cnn14 trained on 16 kHz AudioSet (upstream ``Cnn14_16k``).

    forward(input) with input a (batch, samples) float32 waveform at 16 kHz returns
    {'clipwise_output': (batch, 527) sigmoid probabilities,
     'embedding': (batch, 2048) penultimate features}.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        window_size: int = WINDOW_SIZE,
        hop_size: int = HOP_SIZE,
        mel_bins: int = MEL_BINS,
        fmin: int = FMIN,
        fmax: int = FMAX,
        classes_num: int = CLASSES_NUM,
    ) -> None:
        super().__init__()

        if _TORCHLIBROSA_IMPORT_ERROR is not None:
            raise ImportError(
                "Cnn14_16k requires torchlibrosa for the Spectrogram/LogmelFilterBank/"
                "SpecAugmentation front-end: pip install torchlibrosa"
            ) from _TORCHLIBROSA_IMPORT_ERROR

        # Upstream asserts the exact 16k front-end config; use ValueError so the
        # guard survives `python -O` (asserts are stripped).
        expected = {
            "sample_rate": SAMPLE_RATE,
            "window_size": WINDOW_SIZE,
            "hop_size": HOP_SIZE,
            "mel_bins": MEL_BINS,
            "fmin": FMIN,
            "fmax": FMAX,
        }
        got = {
            "sample_rate": sample_rate,
            "window_size": window_size,
            "hop_size": hop_size,
            "mel_bins": mel_bins,
            "fmin": fmin,
            "fmax": fmax,
        }
        if got != expected:
            raise ValueError(
                f"Cnn14_16k front-end config is fixed by the released checkpoint; "
                f"expected {expected}, got {got}"
            )

        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

        # Spec augmenter (training only; eval mode bypasses it in forward)
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )

        # bn0 normalises over the MEL axis: applied on the (1,3)-transposed tensor.
        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weight()

    def init_weight(self) -> None:
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(
        self,
        input: torch.Tensor,
        mixup_lambda: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Input: (batch_size, data_length) waveform.  Upstream-identical path."""
        x = self.spectrogram_extractor(input)  # (batch, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)           # (batch, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram (training only)
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        # Upstream uses pool_size=(1, 1) on the last block (no spatial reduction).
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {"clipwise_output": clipwise_output, "embedding": embedding}

        return output_dict


# ---------------------------------------------------------------------------
# Checkpoint loader (local file only; NO download)
# ---------------------------------------------------------------------------

def load_cnn14_16k(checkpoint_path: Path, device: str = "cpu") -> Cnn14_16k:
    """Load the released Cnn14_16k checkpoint into an eval-mode, frozen model.

    The checkpoint format is the upstream training dict: ``torch.load(...)['model']``
    is the state dict, loaded with strict=True so any naming drift in this port
    fails loudly instead of silently mis-measuring.

    Parameters
    ----------
    checkpoint_path:
        Local path to e.g. ``Cnn14_16k_mAP=0.438.pth``.  This project does NOT
        download checkpoints; FileNotFoundError is raised if the path is absent.
    device:
        torch device string for map_location and the returned model.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Cnn14_16k checkpoint not found: {checkpoint_path}. This project does "
            "NOT download checkpoints; provide a local copy of "
            "Cnn14_16k_mAP=0.438.pth (qiuqiangkong/audioset_tagging_cnn release)."
        )

    # The released checkpoint is a plain tensor state dict; prefer the safe
    # weights_only path and fall back for older serialization formats.
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = Cnn14_16k()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# AudioSet label helper (class_labels_indices.csv: index,mid,display_name)
# ---------------------------------------------------------------------------

def load_audioset_labels(csv_path: Path) -> list[str]:
    """Load AudioSet display names ordered by class index.

    Parses ``class_labels_indices.csv`` (columns: index, mid, display_name; display
    names may contain commas and are quoted).  Returns labels[i] = display name of
    class i; for the real AudioSet file this is 527 entries aligned with the
    ``clipwise_output`` axis of Cnn14_16k.
    """
    csv_path = Path(csv_path)
    rows: list[tuple[int, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        if not {"index", "display_name"}.issubset(fieldnames):
            raise ValueError(
                f"{csv_path}: expected columns 'index,mid,display_name', "
                f"got {fieldnames!r}"
            )
        for row in reader:
            rows.append((int(row["index"]), row["display_name"]))

    rows.sort(key=lambda r: r[0])
    indices = [i for i, _ in rows]
    # Labels index the clipwise_output axis: require a contiguous 0..n-1 mapping.
    if indices != list(range(len(rows))):
        raise ValueError(
            f"{csv_path}: class indices must be contiguous 0..n-1; got {indices[:10]}..."
        )
    return [name for _, name in rows]
