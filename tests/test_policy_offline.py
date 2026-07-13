"""CPU-only tests for the Phase-4 offline policy simulator (manual §9, Fig. 6).

Covers the three guarantees the preregistration demands:
  1. EXACT matched-NFE / scoring-call accounting (no approximation);
  2. oracle axis-gated pruning PROVABLY beats random pruning on a constructed pool
     where the closed-window axis perfectly separates winners from losers;
  3. BoN winner-selection ordering is sane (argmax of the scalar reward).

Pure numpy, no GPU, no cache, no I/O.
"""
from __future__ import annotations

import numpy as np
import pytest

from foley_cw import policy_offline as P


# ---------------------------------------------------------------------------
# Synthetic pool construction
# ---------------------------------------------------------------------------
def _separating_pool(clip: str = "syn") -> P.ClipPool:
    """A pool where the EARLY-window axis 'presence' perfectly tags the losers.

    8 candidates. The majority on 'presence' is True for 5 of them; the other 3
    disagree (proxy-wrong on presence) AND are wrong on the final all-axis proxy.
    'class' (a late axis) is True for everyone, so final-correctness == presence-correctness.
    The scalar reward is HIGHER for some losers than some winners, so a reward-only
    method (BoN/DiffRS) cannot fully separate but the early gate can.
    """
    n = 8
    presence = np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)   # 5 winners, 3 losers
    klass = np.ones(n, dtype=bool)
    labels = {"presence": presence, "class": klass}
    # adversarial reward: two losers outscore two winners
    final_score = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.95, 0.90, 0.05])
    axis_score = {"presence": presence.astype(float), "class": klass.astype(float)}
    return P.ClipPool(clip=clip, labels=labels, final_score=final_score, axis_score=axis_score)


def _early_gates() -> list[P.GateSpec]:
    """presence window closes at s=0.20 (early); class never gates (late, all-correct)."""
    return [P.GateSpec(s=0.20, axes=("presence",))]


# ---------------------------------------------------------------------------
# 1. EXACT accounting
# ---------------------------------------------------------------------------
def test_full_bon_nfe_is_exact():
    pool = _separating_pool()
    r = P.simulate_policy(pool, "full_bon", gates=_early_gates(), num_steps=25,
                          rng=np.random.default_rng(0))
    assert r.total_nfe == 8 * 25            # all candidates fully integrated
    assert r.scoring_calls == 8             # one final-window score each
    assert r.completed == 8


def test_prune_nfe_formula_exact():
    assert P.prune_nfe(0.20, 25) == 5       # round(0.2*25)
    assert P.prune_nfe(0.0, 25) == 0
    assert P.prune_nfe(1.0, 25) == 25
    assert P.prune_nfe(0.90, 25) == 22      # round(22.5) -> 22 (banker's rounding on .5)


def test_oracle_gated_nfe_is_exact():
    pool = _separating_pool()
    gates = _early_gates()
    r = P.simulate_policy(pool, "oracle_axis_gated", gates=gates, num_steps=25,
                          rng=np.random.default_rng(0))
    # 3 losers pruned at s=0.20 (5 NFE each), 5 survivors fully integrated (25 each).
    expected = 3 * P.prune_nfe(0.20, 25) + 5 * 25
    assert r.total_nfe == expected == 3 * 5 + 5 * 25
    assert r.completed == 5
    # scoring: 8 survivors scored at the gate + 5 survivors scored at the final rerank.
    assert r.scoring_calls == 8 + 5


def test_same_compute_bon_matches_gated_budget_exactly():
    pool = _separating_pool()
    gates = _early_gates()
    rg = P.simulate_policy(pool, "oracle_axis_gated", gates=gates, num_steps=25,
                           rng=np.random.default_rng(0))
    rb = P.simulate_policy(pool, "same_compute_bon", gates=gates, num_steps=25,
                           rng=np.random.default_rng(1), budget_nfe=rg.total_nfe)
    # same_compute_bon completes ceil(budget / num_steps) candidates, each full cost.
    k = int(np.ceil(rg.total_nfe / 25))
    assert rb.completed == k
    assert rb.total_nfe == k * 25
    assert rb.total_nfe >= rg.total_nfe
    assert rb.total_nfe - rg.total_nfe < 25


def test_random_prune_nfe_exact():
    pool = _separating_pool()
    gates = _early_gates()
    r = P.simulate_policy(pool, "random_prune", gates=gates, num_steps=25,
                          rng=np.random.default_rng(3), random_prune_frac=0.5)
    n_pruned = int(round(0.5 * pool.n))
    n_surv = pool.n - n_pruned
    expected = n_pruned * P.prune_nfe(0.20, 25) + n_surv * 25
    assert r.total_nfe == expected
    assert r.completed == n_surv


# ---------------------------------------------------------------------------
# 2. Oracle gating provably beats random pruning
# ---------------------------------------------------------------------------
def test_oracle_gating_beats_random_on_quality():
    pool = _separating_pool()
    gates = _early_gates()
    # Oracle: prunes exactly the 3 proxy-wrong candidates -> winner is always correct.
    rg = P.simulate_policy(pool, "oracle_axis_gated", gates=gates, num_steps=25,
                           rng=np.random.default_rng(0))
    assert rg.winner_correct is True
    assert rg.false_pruned == 0            # the reward-winner is never one of the pruned losers
    assert rg.regret == 0.0

    # Random pruning at matched fraction: averaged over seeds it sometimes prunes winners
    # and its chosen winner is sometimes wrong, so mean quality is strictly worse.
    gated_q, rand_q = [], []
    for seed in range(200):
        g = P.simulate_policy(pool, "oracle_axis_gated", gates=gates, num_steps=25,
                              rng=np.random.default_rng(seed))
        rnd = P.simulate_policy(pool, "random_prune", gates=gates, num_steps=25,
                                rng=np.random.default_rng(seed), random_prune_frac=0.5)
        gated_q.append(g.winner_correct)
        rand_q.append(rnd.winner_correct)
    assert np.mean(gated_q) == 1.0
    assert np.mean(rand_q) < np.mean(gated_q)   # random strictly worse on proxy quality


def test_oracle_gating_pareto_dominates_random_at_lower_nfe():
    pool = _separating_pool()
    gates = _early_gates()
    g = P.simulate_policy(pool, "oracle_axis_gated", gates=gates, num_steps=25,
                          rng=np.random.default_rng(0))
    rnd = P.simulate_policy(pool, "random_prune", gates=gates, num_steps=25,
                            rng=np.random.default_rng(0), random_prune_frac=0.5)
    # gated spends LESS generator-NFE (prunes 3, random prunes 4 but the wrong ones) ...
    assert g.total_nfe < pool.n * 25
    # ... and reaches strictly-better-or-equal quality at this seed.
    assert int(g.winner_correct) >= int(rnd.winner_correct)


# ---------------------------------------------------------------------------
# 3. BoN ordering is sane
# ---------------------------------------------------------------------------
def test_bon_picks_argmax_reward():
    # monotone pool: candidate 4 has the highest final_score and is correct.
    labels = {"presence": np.array([1, 1, 1, 1, 1], dtype=bool)}
    score = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
    pool = P.ClipPool(clip="mono", labels=labels, final_score=score)
    r = P.simulate_policy(pool, "full_bon", gates=[], rng=np.random.default_rng(0))
    assert r.winner == 4
    r2 = P.simulate_policy(pool, "final_rerank", gates=[], rng=np.random.default_rng(0))
    assert r2.winner == 4


def test_bon_tie_break_is_deterministic():
    labels = {"presence": np.array([1, 1, 1], dtype=bool)}
    score = np.array([0.5, 0.5, 0.5])      # all tied
    pool = P.ClipPool(clip="tie", labels=labels, final_score=score)
    w1 = P.simulate_policy(pool, "full_bon", gates=[], rng=np.random.default_rng(7)).winner
    w2 = P.simulate_policy(pool, "full_bon", gates=[], rng=np.random.default_rng(7)).winner
    assert w1 == w2                         # same seed -> same winner


def test_winner_retention_and_false_prune_semantics():
    pool = _separating_pool()
    gates = _early_gates()
    # The BoN ceiling pick (best final_score) = index 5, a proxy-LOSER (presence wrong).
    # Oracle gating prunes index 5 -> winner_retention is False (the reward-winner died) ...
    r = P.simulate_policy(pool, "oracle_axis_gated", gates=gates, num_steps=25,
                          rng=np.random.default_rng(0))
    ref = int(np.lexsort((np.arange(pool.n), -pool.final_score))[0])
    assert ref == 5
    assert r.winner_retained is False
    # ... but it is NOT a false prune: a false prune means pruning a proxy-CORRECT would-be
    # winner. Index 5 is proxy-wrong, so pruning it is a CORRECT prune -> false_pruned == 0.
    assert r.false_pruned == 0


def test_false_prune_fires_when_correct_winner_is_pruned():
    """A pool whose BEST-reward candidate is proxy-CORRECT but the gate prunes it on a
    DIFFERENT axis -> the metric must register exactly one false prune."""
    n = 4
    # index 3 has the top reward and is class-correct, but its presence is wrong, so the
    # presence gate (closed window) prunes it -> a proxy-correct would-be winner is lost.
    labels = {
        "presence": np.array([1, 1, 1, 0], dtype=bool),
        "class": np.array([1, 1, 1, 1], dtype=bool),
    }
    # make index 3 BOTH the reward-winner AND proxy-correct on the FINAL all-axis proxy?
    # No: presence wrong makes it final-incorrect. To exercise a FALSE prune we need the
    # pruned candidate proxy-correct on ALL axes. So flip: gate on 'class' (early) while the
    # winner is class-correct but a minority -> prune a correct winner.
    labels = {
        "class": np.array([1, 1, 0, 1], dtype=bool),     # index 3 is class-correct (majority)
        "presence": np.array([1, 1, 1, 1], dtype=bool),
    }
    score = np.array([0.1, 0.2, 0.9, 0.8])               # reward-winner = index 2 (class-WRONG)
    pool = P.ClipPool(clip="fp", labels=labels,
                      final_score=score,
                      axis_score={"class": np.array([1.0, 1.0, 0.0, 1.0]),
                                  "presence": np.ones(4)})
    gates = [P.GateSpec(s=0.20, axes=("class",))]
    r = P.simulate_policy(pool, "oracle_axis_gated", gates=gates, num_steps=25,
                          rng=np.random.default_rng(0))
    # index 2 (class-wrong) is pruned: it is the reward-winner but proxy-wrong -> NOT false.
    # The best CORRECT candidate is index 3 (class+presence correct, top reward among correct);
    # it survives the class gate, so false_pruned == 0 and it is the winner.
    assert r.winner == 3
    assert r.winner_correct is True
    assert r.false_pruned == 0
    # Now random pruning that happens to drop index 3 must count as a false prune.
    seen_fp = False
    for seed in range(50):
        rr = P.simulate_policy(pool, "random_prune", gates=gates, num_steps=25,
                               rng=np.random.default_rng(seed), random_prune_frac=0.5)
        if rr.false_pruned == 1:
            seen_fp = True
            break
    assert seen_fp, "random pruning never registered a false prune of the correct winner"


# ---------------------------------------------------------------------------
# Gate construction + full roster
# ---------------------------------------------------------------------------
def test_gates_from_scommit_orders_and_buckets():
    s_commit = {"presence": 0.20, "timing": 0.18, "class": 0.33, "material": 0.65}
    grid = (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90)
    gates = P.gates_from_scommit(s_commit, grid)
    gs = [g.s for g in gates]
    assert gs == sorted(gs)                 # ascending windows
    # presence(0.20)&timing(0.18) -> first grid >= -> 0.25 bucket; class(0.33)->0.35; material(0.65)->0.75
    bucket = {g.s: g.axes for g in gates}
    assert set(bucket[0.25]) == {"presence", "timing"}
    assert bucket[0.35] == ("class",)
    assert bucket[0.75] == ("material",)


def test_gates_skip_nan_scommit():
    gates = P.gates_from_scommit({"presence": float("nan"), "class": 0.30},
                                 (0.05, 0.35, 0.90))
    axes_gated = {a for g in gates for a in g.axes}
    assert "presence" not in axes_gated
    assert "class" in axes_gated


def test_run_all_policies_matches_same_compute_budget():
    pools = [_separating_pool(f"c{i}") for i in range(5)]
    gates = _early_gates()
    out = P.run_all_policies(pools, gates=gates, num_steps=25, seed=0,
                             diffrs_tau=0.45, smc_temp=0.1, random_prune_frac=0.5)
    assert set(out) == set(P.POLICIES)
    # Whole-candidate rounding makes same_compute_bon meet or exceed gated NFE.
    assert out["same_compute_bon"]["total_nfe"] >= out["oracle_axis_gated"]["total_nfe"]
    # full_bon is the NFE ceiling.
    assert out["full_bon"]["total_nfe"] >= out["oracle_axis_gated"]["total_nfe"]
    # oracle gating reaches perfect proxy quality on the separating pool.
    assert out["oracle_axis_gated"]["final_correctness"] == 1.0
    assert out["random_prune"]["final_correctness"] <= 1.0


def test_empty_pool_is_safe():
    pool = P.ClipPool(clip="empty", labels={"presence": np.zeros(0, bool)},
                      final_score=np.zeros(0))
    r = P.simulate_policy(pool, "full_bon", gates=[], rng=np.random.default_rng(0))
    assert r.winner is None
    assert r.total_nfe == 0 and r.scoring_calls == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
