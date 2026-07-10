"""B4 — Oracle->Non-oracle bridge (pre-reg §B4; METHOD make-or-break; OFFLINE).

Replays the Phase-4 axis-gated pruning simulator (foley_cw.policy_offline) on the cached
cfg=4.5 pool, but REPLACES the oracle's per-candidate axis knowledge with a REALISTIC
NON-ORACLE per-axis scorer whose quality is bounded by the Phase-2 EXTERNAL readout accuracy
(readout_map_p2cfg1.csv) at each axis's commit window. Pure numpy. No GPU. No new generation.

Non-oracle scorer (the floor model; pre-reg §B4 "bounded by readout quality"):
  The oracle keeps candidate i on axis a iff its TRUE final self-target matches the per-clip
  consensus target. A realistic readout reads the in-window preview and is right with prob p_a =
  the external readout accuracy at that axis's commit window. We model the non-oracle KEEP
  DECISION as the oracle keep decision flipped SYMMETRICALLY with prob (1 - p_a), for every axis
  (categorical and embedding alike). This makes the keep-accuracy exactly p_a, so survivor purity
  is bounded by readout quality and does NOT inflate with the label cardinality K.

  [CORRECTED 2026-06-21 after adversarial review: the earlier model read each candidate's label
  with accuracy p_a and drew wrong reads uniform over OTHER labels — for a K-class axis a truly
  wrong candidate is then misread-as-consensus only with prob (1-p)/(K-1), so survivor purity
  climbed with K and EXCEEDED p_a, inflating recovery (it produced a spurious BRIDGE_METHOD). The
  symmetric keep-flip is the honest, K-invariant floor the pre-reg requires.]

  class uses the EXTERNAL audio-tagger readout (B1 emitted R2_CLASS_CONFIRMED — no internal class
  head exists); this is the documented FLOOR for class.

We average over N_NOISE seeded noise realizations (variance reduction), then bootstrap recovery
by VIDEO (the pre-registered bootstrap unit). The simulator itself is the FROZEN
foley_cw.policy_offline; we feed the noisy keep mask through its `axis_score` channel (the gated
policy keeps survivors with axis_score>0.5), so the non-oracle replay reuses the exact same
accounting (matched generator-NFE AND scoring-calls) with no change to the frozen simulator.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from foley_cw import policy_offline as P

# Number of independent noise realizations averaged per (clip, scorer) before bootstrap.
N_NOISE_DEFAULT: int = 64

# Phase-1 s-grid (mirrors scripts/phase4_policy.py / phase1_commitment.py).
PHASE1_S_GRID: tuple[float, ...] = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)


def rng_for(seed: int, *parts) -> np.random.Generator:
    """Deterministic Generator from a seed + arbitrary string parts (mirrors phase1)."""
    entropy = [seed] + [zlib.crc32(str(p).encode()) for p in parts]
    return np.random.default_rng(np.random.SeedSequence(entropy))


# ---------------------------------------------------------------------------
# Per-clip axis read state: the data the non-oracle scorer needs to draw a noisy read.
# ---------------------------------------------------------------------------
@dataclass
class AxisReadState:
    """Per-clip, per-axis state for drawing a noisy non-oracle keep mask.

    is_embedding : material-style axis (binary above-median membership) vs categorical.
    labels       : length-N candidate labels (categorical) OR length-N above-median bool
                   (embedding), index-aligned with the pool.
    consensus    : the per-clip consensus target the readout is matching against
                   (the majority label, or True for "in-consensus" membership on embeddings).
    label_set    : sorted unique labels observed in this clip pool (categorical only).
    p_acc        : the external readout accuracy at this axis's commit window in [0,1].
    """

    is_embedding: bool
    labels: np.ndarray
    consensus: object
    label_set: list
    p_acc: float


@dataclass
class BridgePool:
    """A clip's replay pool plus the per-axis non-oracle read state.

    Wraps the frozen P.ClipPool (labels / final_score / true axis_score) and adds the
    read_state needed to synthesize a noisy non-oracle keep mask per axis.
    """

    pool: P.ClipPool
    read_state: dict[str, AxisReadState] = field(default_factory=dict)
    video: Optional[str] = None

    def __post_init__(self) -> None:
        if self.video is None:
            self.video = self.pool.clip


# ---------------------------------------------------------------------------
# Non-oracle noisy read: scalar readout accuracy -> per-candidate keep mask
# ---------------------------------------------------------------------------
def noisy_keep_mask(state: AxisReadState, rng: np.random.Generator) -> np.ndarray:
    """Draw a length-N float keep mask (1.0 keep / 0.0 prune) from the readout floor model.

    HONEST FLOOR (pre-reg §B4 "recovery is bounded by readout quality"): the non-oracle
    keep DECISION agrees with the oracle keep decision with probability p_acc (the external
    readout accuracy at this axis window), via a SYMMETRIC flip. This makes the keep-accuracy
    exactly p_acc for every axis — categorical and embedding alike — so survivor purity is
    bounded by readout quality and does NOT inflate with the label cardinality K (the defect
    of the earlier uniform-over-other-labels model, which let multiclass axes "self-purify"
    far above p_acc). At p_acc=1 the non-oracle recovers the oracle; at p_acc=0.5 it is
    uninformative; below 0.5 it is worse than chance — the correct calibration of "the scorer
    is right with probability p_acc". The mask is fed to the FROZEN simulator via axis_score
    (kept iff >0.5).
    """
    n = int(len(state.labels))
    p = float(np.clip(state.p_acc, 0.0, 1.0))

    # Oracle keep decision per candidate: does its TRUE in-window axis value match consensus?
    if state.is_embedding:
        oracle_keep = (np.asarray(state.labels, dtype=bool) == bool(state.consensus))
    else:
        oracle_keep = np.array([lab == state.consensus for lab in state.labels], dtype=bool)

    # Symmetric flip of the keep DECISION with prob (1 - p): keep-accuracy == p_acc exactly.
    flip = rng.random(n) >= p
    nonoracle_keep = oracle_keep.copy()
    nonoracle_keep[flip] = ~nonoracle_keep[flip]
    return nonoracle_keep.astype(float)


def make_nonoracle_pool(
    bp: BridgePool, rng: np.random.Generator
) -> P.ClipPool:
    """Clone the frozen pool but overwrite axis_score with a fresh noisy non-oracle keep mask.

    Feeding the noisy mask through axis_score makes the FROZEN `oracle_axis_gated` policy prune
    on the NON-oracle read (it keeps survivors with axis_score>0.5), with identical accounting.
    """
    new_axis_score: dict[str, np.ndarray] = {}
    for a, st in bp.read_state.items():
        new_axis_score[a] = noisy_keep_mask(st, rng)
    # axes present in labels but without read_state keep their (true) behavior off — but every
    # in-scope axis is given a read_state by the builder, so this is exhaustive in practice.
    return P.ClipPool(
        clip=bp.pool.clip,
        labels={a: v.copy() for a, v in bp.pool.labels.items()},
        final_score=bp.pool.final_score.copy(),
        axis_score=new_axis_score,
    )


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------
def headroom_recovery(non_oracle: float, scalar: float, oracle: float) -> float:
    """(non_oracle - scalar)/(oracle - scalar), clipped to [0,1]. 0 if denom<=0."""
    denom = oracle - scalar
    if denom <= 1e-12:
        return 0.0
    return float(np.clip((non_oracle - scalar) / denom, 0.0, 1.0))


def decision_token(mean_recovery: float) -> str:
    """Pre-reg §B4 decision rule on the MEAN per-axis recovery."""
    if mean_recovery >= 0.5:
        return "BRIDGE_METHOD"
    if mean_recovery >= 0.2:
        return "BRIDGE_PARTIAL"
    return "BRIDGE_WEAK"
