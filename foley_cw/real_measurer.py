"""Real per-axis self-target measurer (manual section 3.3 pre-registered choices).

Implements the axes.Measurer protocol on real 16 kHz waveforms:

  presence_detector  -> energy gate (RMS dBFS > -45) AND tagger eventness
                        (max PANNs prob over event classes >= 0.2); categorical.
  audio_tagger_top1  -> PANNs Cnn14_16k (16 kHz-native) 527 AudioSet probs summed
                        into the FROZEN coarse-class groups (configs/coarse_class_map.json),
                        argmax group label; categorical.
  onset_timing_bin   -> librosa spectral-flux onset, first detected onset time
                        binned at timing_bin_s (placeholder 0.5 s until Phase 0.4
                        freezes >= 2*sigma_anchor); categorical ("none" if no onset).
  audio_embedding    -> CLAP (laion/clap-htsat-unfused) audio embedding, 16k->48k
                        resample (documented), unit-normed; embedding axis.
  binding_label / seed_predictability -> NotImplementedError (not in this slice).

Cross-tagger agreement (Phase 0.5) uses AST (MIT/ast-finetuned-audioset-10-10-0.4593,
16 kHz-native) through the same coarse map: coarse_label_second_tagger().

Gate-A embedding space: panns_embedding() exposes the Cnn14_16k 2048-d penultimate
embedding (same forward pass as the class axis; one embedder, one fewer moving part).

Determinism contract: all models run in eval mode with no sampling; measure() on
identical audio returns identical SelfTargets (Stage-M criterion 4). A one-entry
forward cache keyed by the audio bytes hash lets presence/class/embedding share a
single PANNs forward on the same waveform.

GPU seam module — NOT imported by the numpy core. Heavy deps (torch, torchaudio,
librosa, transformers, torchlibrosa) are imported lazily. Tests inject tagger_fn /
embed_fn / tagger2_fn to run CPU-only without checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .types import Axis, AxisKind, SelfTarget

_REPO_ROOT = Path(__file__).resolve().parent.parent

PRESENCE_RMS_DBFS_MIN = -45.0
PRESENCE_EVENT_PROB_MIN = 0.2
PRESENT, ABSENT = "present", "absent"
NO_ONSET_LABEL = "none"
#: Class-axis abstain label (revised manual 3.3). Frozen interpretation #2
#: (stage_m_rerun_interpretations.md): abstain when the CROSS-GROUP top-2 margin
#: among event classes is < CLASS_ABSTAIN_DELTA. delta = 0.05 is ~17x the observed
#: knife-edge flip scale (0.003) and upper-bounds the prob jitter of the
#: pre-registered robustness perturbations (O(0.01-0.03)).
ABSTAIN = "abstain"
CLASS_ABSTAIN_DELTA = 0.05

CLAP_MODEL_ID = "laion/clap-htsat-unfused"
CLAP_REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_REVISION = "f826b80d28226b62986cc218e5cec390b1096902"
_MODEL_REVISIONS = {
    CLAP_MODEL_ID: CLAP_REVISION,
    AST_MODEL_ID: AST_REVISION,
}


def load_coarse_map(path: Path) -> dict:
    """Frozen AudioSet-527 -> coarse-class mapping (versioned; v3 adds the
    event-restriction key class_excluded_coarse per revised manual 3.3)."""
    d = json.loads(Path(path).read_text())
    for key in ("version", "coarse_classes", "index_to_coarse"):
        if key not in d:
            raise ValueError(f"coarse_class_map missing key {key!r}")
    d["index_to_coarse"] = {int(k): v for k, v in d["index_to_coarse"].items()}
    d.setdefault("class_excluded_coarse", [])
    d.setdefault("non_event_indices", [])
    return d


class RealFoleyMeasurer:
    """axes.Measurer implementation for real MMAudio-generated audio.

    Parameters mirror the pre-registered measurer choices; tagger_fn / embed_fn /
    tagger2_fn are CPU-test injection points:
      tagger_fn(audio: np.ndarray) -> (probs527: np.ndarray, embedding2048: np.ndarray)
      embed_fn(audio: np.ndarray) -> np.ndarray  (material embedding, any dim)
      tagger2_fn(audio: np.ndarray) -> probs527: np.ndarray
    """

    def __init__(self, sr: int = 16000, device: str = "cpu",
                 weights_dir: Path = _REPO_ROOT / "weights" / "measurers",
                 coarse_map_path: Path = _REPO_ROOT / "configs" / "coarse_class_map.json",
                 timing_bin_s: float = 0.5, n_timing_bins: int = 16,
                 presence_rms_dbfs: float = PRESENCE_RMS_DBFS_MIN,
                 presence_event_prob: float = PRESENCE_EVENT_PROB_MIN,
                 class_abstain_delta: float = CLASS_ABSTAIN_DELTA,
                 tagger_fn: Optional[Callable] = None,
                 embed_fn: Optional[Callable] = None,
                 tagger2_fn: Optional[Callable] = None) -> None:
        if sr != 16000:
            raise ValueError(f"RealFoleyMeasurer is 16 kHz-native (Cnn14_16k); got sr={sr}")
        self.sr = sr
        self.device = device
        self.weights_dir = Path(weights_dir)
        self.timing_bin_s = float(timing_bin_s)
        self.n_timing_bins = int(n_timing_bins)
        self.presence_rms_dbfs = float(presence_rms_dbfs)
        self.presence_event_prob = float(presence_event_prob)
        self.class_abstain_delta = float(class_abstain_delta)
        self._tagger_fn = tagger_fn
        self._embed_fn = embed_fn
        self._tagger2_fn = tagger2_fn
        self._coarse_map_path = Path(coarse_map_path)
        self._coarse: Optional[dict] = None
        self._panns = None
        self._clap = None
        self._ast = None
        self._fwd_cache: tuple[Optional[str], Optional[tuple]] = (None, None)

    # ------------------------------------------------------------------
    # lazy model loading
    # ------------------------------------------------------------------
    @property
    def coarse(self) -> dict:
        if self._coarse is None:
            self._coarse = load_coarse_map(self._coarse_map_path)
        return self._coarse

    def _ensure_panns(self):
        if self._panns is None:
            from .measurers_panns_cnn14 import load_cnn14_16k
            ckpt = self.weights_dir / "Cnn14_16k_mAP=0.438.pth"
            self._panns = load_cnn14_16k(ckpt, device=self.device)
        return self._panns

    def _pretrained_spec(self, model_id: str) -> tuple[str, dict[str, object]]:
        """Resolve a pinned, local-only Transformers model source."""
        if model_id not in _MODEL_REVISIONS:
            raise KeyError(f"unregistered pretrained model {model_id!r}")
        source = os.environ.get("FOLEY_CW_WEIGHTS_SOURCE", "modelscope").strip().lower()
        if source not in {"modelscope", "hf"}:
            raise ValueError(
                "FOLEY_CW_WEIGHTS_SOURCE must be 'modelscope' or 'hf'; "
                f"got {source!r}"
            )
        revision = _MODEL_REVISIONS[model_id]
        if source == "hf":
            return model_id, {"revision": revision, "local_files_only": True}

        mirror_root = Path(os.environ.get(
            "FOLEY_CW_MODELSCOPE_ROOT", self.weights_dir.parent / "modelscope"
        ))
        candidates = (
            mirror_root / model_id,
            mirror_root / model_id.replace("/", "--"),
        )
        local_path = next((path for path in candidates if path.is_dir()), None)
        if local_path is None:
            raise FileNotFoundError(
                f"ModelScope mirror for {model_id!r} not found under {mirror_root}. "
                "Populate the mirror or set FOLEY_CW_WEIGHTS_SOURCE=hf to use the "
                "pinned local Hugging Face cache. Downloads are disabled."
            )
        return str(local_path), {"revision": revision, "local_files_only": True}

    def _panns_forward(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(probs527, embedding2048); one-entry cache keyed by audio bytes hash."""
        a = np.ascontiguousarray(audio, dtype=np.float32)
        key = hashlib.sha1(a.tobytes()).hexdigest()
        ck, cv = self._fwd_cache
        if ck == key:
            return cv  # type: ignore[return-value]
        if self._tagger_fn is not None:
            probs, emb = self._tagger_fn(a)
        else:
            import torch
            model = self._ensure_panns()
            with torch.no_grad():
                out = model(torch.from_numpy(a[None, :]).to(self.device))
            probs = out["clipwise_output"][0].float().cpu().numpy()
            emb = out["embedding"][0].float().cpu().numpy()
        result = (np.asarray(probs, dtype=float), np.asarray(emb, dtype=float))
        self._fwd_cache = (key, result)
        return result

    # ------------------------------------------------------------------
    # per-axis measures
    # ------------------------------------------------------------------
    def _rms_dbfs(self, audio: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        return 20.0 * np.log10(max(rms, 1e-12))

    def _presence_detector(self, audio: np.ndarray) -> str:
        energy_ok = self._rms_dbfs(audio) > self.presence_rms_dbfs
        probs, _ = self._panns_forward(audio)
        event_idx = [i for i in range(probs.shape[0])
                     if i not in set(self.coarse.get("non_event_indices", []))]
        event_ok = float(probs[event_idx].max()) >= self.presence_event_prob if event_idx else False
        return PRESENT if (energy_ok and event_ok) else ABSENT

    def _audio_tagger_top1(self, audio: np.ndarray) -> str:
        probs, _ = self._panns_forward(audio)
        return self._coarse_from_probs(probs)

    def _event_indices(self, n: int) -> np.ndarray:
        """Cached boolean mask of event-class indices (revised manual 3.3):
        excludes non_event_indices and every index mapping to a coarse group in
        class_excluded_coarse (speech/music/ambient)."""
        cached = getattr(self, "_event_mask", None)
        if cached is not None and cached.shape[0] == n:
            return cached
        excluded_groups = set(self.coarse["class_excluded_coarse"])
        mask = np.ones(n, dtype=bool)
        for idx in self.coarse["non_event_indices"]:
            if idx < n:
                mask[idx] = False
        for idx, grp in self.coarse["index_to_coarse"].items():
            if grp in excluded_groups and idx < n:
                mask[idx] = False
        self._event_mask = mask
        return mask

    def class_diagnostics(self, probs: np.ndarray) -> dict:
        """Event-restricted class decision + the non-gating instruments
        (revised manual 3.3): label (or 'abstain'), cross-group top-2 margin,
        event-prob entropy, top-1 concentration, top1/top2 identities.

        Decision rule (frozen interpretation #2): top-1 over event classes; the
        runner-up is the best event class whose COARSE GROUP differs from the
        top-1's (within-group flips do not change the label and must not trigger
        abstain). Abstain iff p(top1) - p(cross-group runner-up) < delta.
        """
        probs = np.asarray(probs, dtype=float)
        mask = self._event_indices(probs.shape[0])
        ev_idx = np.flatnonzero(mask)
        ev_probs = probs[ev_idx]
        order = np.argsort(ev_probs)[::-1]
        top1_i = int(ev_idx[order[0]])
        top1_p = float(ev_probs[order[0]])
        top1_grp = self.coarse["index_to_coarse"].get(top1_i, "other")
        runner_p, runner_i = 0.0, -1
        for o in order[1:]:
            gi = int(ev_idx[o])
            if self.coarse["index_to_coarse"].get(gi, "other") != top1_grp:
                runner_p, runner_i = float(ev_probs[o]), gi
                break
        margin = top1_p - runner_p
        p_norm = ev_probs / max(ev_probs.sum(), 1e-12)
        entropy = float(-(p_norm * np.log(np.maximum(p_norm, 1e-12))).sum())
        label = top1_grp if margin >= self.class_abstain_delta else ABSTAIN
        return {"label": label, "margin": float(margin), "entropy": entropy,
                "top1_index": top1_i, "top1_prob": top1_p,
                "runner_index": runner_i, "runner_prob": runner_p,
                "top1_group": top1_grp,
                "concentration": top1_p / max(float(ev_probs.sum()), 1e-12)}

    def _coarse_from_probs(self, probs: np.ndarray) -> str:
        # Revised manual 3.3: event-restricted top-1-then-map with a cross-group
        # abstain margin. (History: group-SUM was rejected at the W3 sanity gate
        # for large-group bias; unrestricted top-1 was rejected at the first
        # Stage-M run for speech/music tagger confusion + knife-edge flips.)
        return self.class_diagnostics(probs)["label"]

    def _onset_timing_bin(self, audio: np.ndarray) -> Any:
        import librosa
        env = librosa.onset.onset_strength(y=np.asarray(audio, dtype=np.float32), sr=self.sr)
        onsets = librosa.onset.onset_detect(onset_envelope=env, sr=self.sr, units="time",
                                            backtrack=False)
        if len(onsets) == 0:
            return NO_ONSET_LABEL
        return int(min(float(onsets[0]) // self.timing_bin_s, self.n_timing_bins - 1))

    def _audio_embedding(self, audio: np.ndarray) -> np.ndarray:
        if self._embed_fn is not None:
            emb = np.asarray(self._embed_fn(np.asarray(audio, dtype=np.float32)), dtype=float)
        else:
            import torch
            import torchaudio.functional as AF
            if self._clap is None:
                from transformers import ClapModel, ClapProcessor
                name, load_kwargs = self._pretrained_spec(CLAP_MODEL_ID)
                self._clap = (
                    ClapModel.from_pretrained(name, **load_kwargs).to(self.device).eval(),
                    ClapProcessor.from_pretrained(name, **load_kwargs),
                )
            model, proc = self._clap
            wav48 = AF.resample(torch.from_numpy(np.asarray(audio, dtype=np.float32)),
                                self.sr, 48000)
            inputs = proc(audios=wav48.numpy(), sampling_rate=48000, return_tensors="pt")
            with torch.no_grad():
                emb_t = model.get_audio_features(**{k: v.to(self.device) for k, v in inputs.items()})
            emb = emb_t[0].float().cpu().numpy().astype(float)
        n = np.linalg.norm(emb)
        return emb / n if n > 0 else emb

    # ------------------------------------------------------------------
    # public extras
    # ------------------------------------------------------------------
    def panns_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Cnn14_16k 2048-d penultimate embedding (Gate-A distributional space)."""
        _, emb = self._panns_forward(audio)
        return emb

    def panns_posterior(self, audio: np.ndarray) -> np.ndarray:
        """Cnn14_16k 527-way probability vector for raw posterior retention.

        This reuses the one-audio forward cache, so a caller that has just
        measured presence/class does not incur another model forward pass.
        """
        probs, _ = self._panns_forward(audio)
        return np.asarray(probs, dtype=float).copy()

    def coarse_label_second_tagger(self, audio: np.ndarray) -> str:
        """AST coarse label for cross-tagger agreement (Phase 0.5)."""
        if self._tagger2_fn is not None:
            probs = np.asarray(self._tagger2_fn(np.asarray(audio, dtype=np.float32)), dtype=float)
            return self._coarse_from_probs(probs)
        import torch
        if self._ast is None:
            from transformers import ASTFeatureExtractor, ASTForAudioClassification
            name, load_kwargs = self._pretrained_spec(AST_MODEL_ID)
            model = ASTForAudioClassification.from_pretrained(
                name, **load_kwargs
            ).to(self.device).eval()
            self._assert_ast_label_alignment(model)
            self._ast = (model, ASTFeatureExtractor.from_pretrained(name, **load_kwargs))
        model, fe = self._ast
        inputs = fe(np.asarray(audio, dtype=np.float32), sampling_rate=self.sr,
                    return_tensors="pt")
        with torch.no_grad():
            logits = model(**{k: v.to(self.device) for k, v in inputs.items()}).logits
        probs = torch.sigmoid(logits)[0].float().cpu().numpy()
        return self._coarse_from_probs(np.asarray(probs, dtype=float))

    def _assert_ast_label_alignment(self, model) -> None:
        """Fail loudly unless AST's id2label order matches the local AudioSet CSV
        (the index-based coarse map silently mislabels otherwise). Verified true
        for snapshot f826b80d... in the Codex review; asserted at every load."""
        import csv as _csv
        csv_path = self.weights_dir / "class_labels_indices.csv"
        rows = list(_csv.DictReader(csv_path.open()))
        id2label = model.config.id2label
        mismatches = [(int(r["index"]), r["display_name"], id2label.get(int(r["index"])))
                      for r in rows
                      if id2label.get(int(r["index"]), "").strip().lower()
                      != r["display_name"].strip().lower()]
        if mismatches:
            raise RuntimeError(
                f"AST id2label does not match {csv_path} ({len(mismatches)} mismatches, "
                f"first: {mismatches[0]}); the index-based coarse map cannot be shared")

    # ------------------------------------------------------------------
    # Measurer protocol
    # ------------------------------------------------------------------
    def measure(self, audio: np.ndarray, axis: Axis) -> SelfTarget:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        dispatch: dict[str, Callable[[np.ndarray], Any]] = {
            "presence_detector": self._presence_detector,
            "onset_timing_bin": self._onset_timing_bin,
            "audio_tagger_top1": self._audio_tagger_top1,
            "audio_embedding": self._audio_embedding,
        }
        if axis.measure in ("binding_label", "seed_predictability"):
            raise NotImplementedError(f"{axis.measure} is out of scope for this slice "
                                      "(manual section 2: presence + coarse class first)")
        if axis.measure not in dispatch:
            raise KeyError(f"unknown measure {axis.measure!r}")
        value = dispatch[axis.measure](audio)
        if axis.kind is AxisKind.EMBEDDING:
            return SelfTarget(axis_id=axis.id, kind=axis.kind, embedding=np.asarray(value))
        return SelfTarget(axis_id=axis.id, kind=axis.kind, label=value)
