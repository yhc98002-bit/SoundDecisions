"""Tests for foley_cw.bridge — B4 oracle->non-oracle bridge (pre-reg §B4)."""
from __future__ import annotations

import numpy as np
import pytest

from foley_cw import bridge as B
from foley_cw import policy_offline as P


# ---------------------------------------------------------------------------
# noisy_keep_mask: floor model behavior
# ---------------------------------------------------------------------------
def _cat_state(labels, consensus, p_acc):
    return B.AxisReadState(
        is_embedding=False, labels=np.array(labels, dtype=object),
        consensus=consensus, label_set=sorted(set(labels), key=str), p_acc=p_acc,
    )


def test_perfect_readout_categorical_equals_oracle():
    # p=1.0 -> the noisy read is always correct -> keep iff true label == consensus.
    labels = ["a", "b", "a", "c", "a"]
    st = _cat_state(labels, consensus="a", p_acc=1.0)
    rng = np.random.default_rng(0)
    keep = B.noisy_keep_mask(st, rng)
    true_keep = np.array([1.0 if l == "a" else 0.0 for l in labels])
    assert np.allclose(keep, true_keep)


def test_zero_readout_categorical_never_keeps_correct_by_truth():
    # p=0.0 -> every read is flipped to a DIFFERENT label, so a candidate whose true label is
    # the consensus is NEVER read as consensus (its read is forced off-consensus); a candidate
    # whose true label != consensus MAY be (mis)read as consensus.
    labels = ["a", "a", "b", "c"]
    st = _cat_state(labels, consensus="a", p_acc=0.0)
    rng = np.random.default_rng(1)
    keep = B.noisy_keep_mask(st, rng)
    # the two true-"a" candidates can never be kept (their read is forced != "a")
    assert keep[:2] == pytest.approx([0.0, 0.0])


def test_embedding_perfect_readout_keeps_in_consensus():
    above = np.array([True, False, True, False])
    st = B.AxisReadState(is_embedding=True, labels=above, consensus=True, label_set=[], p_acc=1.0)
    keep = B.noisy_keep_mask(st, np.random.default_rng(0))
    assert np.allclose(keep, above.astype(float))


def test_embedding_zero_readout_inverts():
    above = np.array([True, False, True, False])
    st = B.AxisReadState(is_embedding=True, labels=above, consensus=True, label_set=[], p_acc=0.0)
    keep = B.noisy_keep_mask(st, np.random.default_rng(0))
    assert np.allclose(keep, (~above).astype(float))


def test_noisy_mask_deterministic_given_rng():
    st = _cat_state(["a", "b", "a", "c", "b"], consensus="a", p_acc=0.6)
    k1 = B.noisy_keep_mask(st, np.random.default_rng(123))
    k2 = B.noisy_keep_mask(st, np.random.default_rng(123))
    assert np.allclose(k1, k2)


def test_higher_accuracy_increases_consensus_keep_rate():
    # Averaged over many draws, higher p_acc -> the readout keeps the true-consensus candidates
    # more often and the off-consensus candidates less often (more oracle-like).
    labels = ["a", "a", "a", "b", "c", "b", "c", "a"]
    consensus = "a"
    true_keep = np.array([1.0 if l == consensus else 0.0 for l in labels])

    def agree_rate(p):
        st = _cat_state(labels, consensus, p_acc=p)
        accs = []
        for t in range(400):
            k = B.noisy_keep_mask(st, np.random.default_rng(t))
            accs.append(np.mean(k == true_keep))
        return float(np.mean(accs))

    assert agree_rate(0.9) > agree_rate(0.5) > agree_rate(0.1)


# ---------------------------------------------------------------------------
# recovery + decision token
# ---------------------------------------------------------------------------
def test_headroom_recovery_clipping_and_bounds():
    assert B.headroom_recovery(0.6, 0.4, 0.8) == pytest.approx(0.5)
    assert B.headroom_recovery(0.9, 0.4, 0.8) == pytest.approx(1.0)  # clipped at 1
    assert B.headroom_recovery(0.3, 0.4, 0.8) == pytest.approx(0.0)  # clipped at 0
    assert B.headroom_recovery(0.5, 0.5, 0.5) == pytest.approx(0.0)  # zero headroom -> 0


def test_decision_token_boundaries():
    assert B.decision_token(0.5) == "BRIDGE_METHOD"
    assert B.decision_token(0.49) == "BRIDGE_PARTIAL"
    assert B.decision_token(0.2) == "BRIDGE_PARTIAL"
    assert B.decision_token(0.19) == "BRIDGE_WEAK"


# ---------------------------------------------------------------------------
# end-to-end on a synthetic pool through the FROZEN simulator
# ---------------------------------------------------------------------------
def _synthetic_bridge_pool(clip="c0", n=12, p_acc=0.9, seed=0):
    """A pool where one axis ('class') separates correct/incorrect candidates; final_score
    correlates weakly so the oracle (axis-gated) beats a scalar BoN."""
    rng = np.random.default_rng(seed)
    # consensus label 'good' for 6 candidates, 'bad'/'ugly' for the rest.
    labels_class = ["good"] * 6 + ["bad"] * 3 + ["ugly"] * 3
    rng.shuffle(labels_class)
    consensus = "good"
    ok = np.array([l == consensus for l in labels_class], dtype=bool)
    # final_score nearly flat (scalar BoN can't separate good from bad).
    final_score = 0.5 + 0.001 * rng.standard_normal(n)
    pool = P.ClipPool(
        clip=clip,
        labels={"class": ok},
        final_score=final_score,
        axis_score={"class": ok.astype(float)},   # TRUE oracle keep
    )
    rs = {
        "class": B.AxisReadState(
            is_embedding=False, labels=np.array(labels_class, dtype=object),
            consensus=consensus, label_set=sorted(set(labels_class), key=str), p_acc=p_acc,
        )
    }
    return B.BridgePool(pool=pool, read_state=rs, video=clip)


def test_make_nonoracle_pool_is_valid_clippool():
    bp = _synthetic_bridge_pool()
    npool = B.make_nonoracle_pool(bp, np.random.default_rng(0))
    assert isinstance(npool, P.ClipPool)
    assert npool.n == bp.pool.n
    assert "class" in npool.axis_score
    assert set(np.unique(npool.axis_score["class"])).issubset({0.0, 1.0})
    # frozen pool untouched
    assert np.allclose(bp.pool.axis_score["class"], bp.pool.labels["class"].astype(float))


def test_end_to_end_recovery_monotone_in_readout_accuracy():
    axes = ("class",)
    gates = [P.GateSpec(s=0.35, axes=("class",))]

    def mean_nonoracle_correct(p_acc, n_clips=24, n_noise=48):
        vals = []
        for c in range(n_clips):
            bp = _synthetic_bridge_pool(clip=f"c{c}", p_acc=p_acc, seed=c)
            acc = 0.0
            for t in range(n_noise):
                npool = B.make_nonoracle_pool(bp, B.rng_for(0, "noise", bp.pool.clip, t))
                r = P.simulate_policy(
                    npool, "oracle_axis_gated", gates=gates, axes=axes,
                    rng=B.rng_for(0, "no", bp.pool.clip, t),
                )
                acc += float(r.axis_correct.get("class", 0.0))
            vals.append(acc / n_noise)
        return float(np.mean(vals))

    # oracle ceiling (p=1.0) vs low-readout floor (p=0.34, the class floor analogue).
    hi = mean_nonoracle_correct(0.95)
    lo = mean_nonoracle_correct(0.34)
    assert hi > lo                        # better readout -> more oracle-like correctness
    assert hi > 0.9                       # near-perfect readout recovers nearly all of the gate


def test_end_to_end_deterministic():
    axes = ("class",)
    gates = [P.GateSpec(s=0.35, axes=("class",))]
    bp = _synthetic_bridge_pool(clip="cX", p_acc=0.7, seed=3)

    def run():
        npool = B.make_nonoracle_pool(bp, B.rng_for(0, "noise", "cX", 5))
        return P.simulate_policy(
            npool, "oracle_axis_gated", gates=gates, axes=axes,
            rng=B.rng_for(0, "no", "cX", 5),
        ).axis_correct["class"]

    assert run() == run()
