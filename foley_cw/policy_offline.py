"""Phase-4 OFFLINE policy simulator (manual §9, Fig. 6). Pure numpy, deterministic.

REPLAY gating on cached deployed-cfg (cfg=4.5) Phase-1 independents — NO new generation.
A clip's pool is its up-to-16 cached independents, each with per-axis final self-target
labels and a per-(candidate,axis) running-consensus score the script extracts from the
cache. The simulator is policy-agnostic: the scalar reward and per-axis scores are INPUTS
(see ClipPool), identical across policies, so Fig. 6's two axes (generator-NFE and
scoring-calls) are matched by construction.

CORRECTNESS here is the ORACLE PROXY of experiment/preregistered/policy_preregistration.md:
agreement with the per-clip MAJORITY self-target across that clip's independents — a
self-consistency proxy, NOT human/MLLM correctness-vs-video. Every metric inherits that caveat.

Policies (preregistered): full_bon, same_compute_bon, random_prune, diffrs_scalar (scalar
DiffRS rejection), smc_scalar (SMC-style scalar resampling), final_rerank, oracle_axis_gated
(prune on axes whose commitment window has closed, gate s = s_commit from
determination_budget_p1cfg45.csv). Accounting (both matched, EXACT): a candidate pruned at
progress s costs round(s*num_steps) generator-NFE, a survivor costs num_steps; one scoring
call per (candidate, window) the scorer is invoked at.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# Default deployed-cfg integration budget (mmaudio_backend num_steps at cfg=4.5).
NFE_FULL_DEFAULT: int = 25

# In-scope axes for the offline pool (categorical + the material embedding axis).
DEFAULT_AXES: tuple[str, ...] = ("presence", "timing", "class", "material")

# Frozen policy roster (preregistration §"Policies / baselines").
POLICIES: tuple[str, ...] = (
    "full_bon",
    "same_compute_bon",
    "random_prune",
    "diffrs_scalar",
    "smc_scalar",
    "final_rerank",
    "oracle_axis_gated",
)


# ---------------------------------------------------------------------------
# Pool / candidate data contract (built by the script from cached measurements)
# ---------------------------------------------------------------------------
@dataclass
class ClipPool:
    """One clip's replay pool of N cached deployed-cfg candidates.

    All arrays are length-N and index-aligned by candidate. The simulator NEVER
    invents these; the script fills them from the cache, the test from synthetic data.

    labels:        axis_id -> length-N proxy-correctness bool array (candidate's final
                   self-target AGREES with the per-clip majority self-target on that axis).
                   This is the ORACLE PROXY, precomputed by the caller (so the simulator
                   stays pool-agnostic and the proxy definition lives in one place).
    final_score:   length-N scalar reward at the FINAL window (used by BoN / rerank /
                   DiffRS / SMC). Higher = preferred. Deterministic, cache-derived.
    axis_score:    axis_id -> length-N scalar in [0,1], the running-consensus agreement of
                   the candidate's in-window self-target with the plurality (used by
                   oracle_axis_gated to keep window-consistent candidates). Optional; if
                   absent the gated policy keeps candidates whose labels[axis] is True.
    clip:          clip id (for reporting / seeding).
    """

    clip: str
    labels: dict[str, np.ndarray]
    final_score: np.ndarray
    axis_score: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(len(self.final_score))

    def __post_init__(self) -> None:
        self.final_score = np.asarray(self.final_score, dtype=float)
        n = self.n
        for a, v in self.labels.items():
            arr = np.asarray(v, dtype=bool)
            if arr.shape[0] != n:
                raise ValueError(f"pool {self.clip}: labels[{a}] len {arr.shape[0]} != n {n}")
            self.labels[a] = arr
        for a, v in self.axis_score.items():
            arr = np.asarray(v, dtype=float)
            if arr.shape[0] != n:
                raise ValueError(f"pool {self.clip}: axis_score[{a}] len {arr.shape[0]} != n {n}")
            self.axis_score[a] = arr

    def final_correct(self, axes: tuple[str, ...]) -> np.ndarray:
        """Length-N bool: candidate proxy-correct on ALL in-scope axes present in the pool."""
        present = [a for a in axes if a in self.labels]
        if not present:
            return np.zeros(self.n, dtype=bool)
        out = np.ones(self.n, dtype=bool)
        for a in present:
            out &= self.labels[a]
        return out


@dataclass
class GateSpec:
    """A pruning window. axes = axis ids whose commitment window has closed by progress s."""

    s: float
    axes: tuple[str, ...]


@dataclass
class PolicyResult:
    """Per-clip outcome of one policy on one pool (all proxy-correctness)."""

    policy: str
    clip: str
    winner: Optional[int]            # chosen candidate index, or None if pool empty
    winner_correct: bool             # final (all-axis) proxy-correctness of the winner
    axis_correct: dict[str, bool]    # per-axis proxy-correctness of the winner
    completed: int                   # candidates run to completion
    total_nfe: int                   # EXACT generator-NFE (preregistration accounting)
    scoring_calls: int               # EXACT scoring-call count
    winner_retained: bool            # did the pool's best-final candidate survive to the end
    false_pruned: int                # # pruned candidates that were the pool's best-final winner
    regret: float                    # best-achievable all-axis proxy-correctness - achieved


# ---------------------------------------------------------------------------
# Gate construction from s_commit
# ---------------------------------------------------------------------------
def gates_from_scommit(
    s_commit: dict[str, float],
    s_grid: tuple[float, ...],
    axes: tuple[str, ...] = DEFAULT_AXES,
) -> list[GateSpec]:
    """Build ordered pruning windows from per-axis s_commit.

    An axis is actionable at the EARLIEST grid point s >= its s_commit (its commitment
    window has closed there). Axes are grouped by that grid point; one GateSpec per grid
    point that owns >=1 axis, in ascending s. Axes with NaN/absent s_commit (never
    committed) are never gated. The final window (rerank) is appended separately by the
    simulator, not here.
    """
    grid = sorted(s_grid)
    bucket: dict[float, list[str]] = {}
    for a in axes:
        sc = s_commit.get(a)
        if sc is None or not np.isfinite(sc):
            continue
        # earliest grid point at or after s_commit
        g = next((s for s in grid if s >= sc), None)
        if g is None:
            continue
        bucket.setdefault(g, []).append(a)
    return [GateSpec(s=g, axes=tuple(sorted(bucket[g]))) for g in sorted(bucket)]


# ---------------------------------------------------------------------------
# Accounting primitives (EXACT)
# ---------------------------------------------------------------------------
def prune_nfe(s: float, num_steps: int) -> int:
    """Generator-NFE charged for a candidate pruned at progress s: round(s*num_steps)."""
    return int(round(float(s) * int(num_steps)))


def _argmax_by_score(idx: np.ndarray, score: np.ndarray, rng: np.random.Generator) -> int:
    """Argmax of score over candidate indices `idx`, ties broken by a seeded shuffle.

    Deterministic given rng; never returns a non-member of idx.
    """
    sub = score[idx]
    best = float(np.max(sub))
    winners = idx[np.flatnonzero(sub >= best - 1e-12)]
    if winners.shape[0] == 1:
        return int(winners[0])
    return int(rng.choice(winners))


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------
def simulate_policy(
    pool: ClipPool,
    policy: str,
    *,
    gates: list[GateSpec],
    axes: tuple[str, ...] = DEFAULT_AXES,
    num_steps: int = NFE_FULL_DEFAULT,
    rng: Optional[np.random.Generator] = None,
    budget_nfe: Optional[int] = None,
    diffrs_tau: float = 0.0,
    smc_temp: float = 1.0,
    random_prune_frac: float = 0.5,
) -> PolicyResult:
    """Run one policy on one pool. Deterministic given `rng`. Pure numpy.

    `budget_nfe` caps generator-NFE for same_compute_bon (set by the script to the gated
    policy's realized NFE). The pool's "winner" reference for retention/false-prune is the
    candidate with the highest final_score (the BoN ceiling pick).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = pool.n
    present_axes = tuple(a for a in axes if a in pool.labels)
    final_correct = pool.final_correct(axes)
    # winner_retention reference = the BoN ceiling pick = best final_score
    # (deterministic tie-break on lowest index).
    ref_winner = int(np.lexsort((np.arange(n), -pool.final_score))[0]) if n else None
    # false_prune reference = the best correct candidate the pool COULD yield =
    # best final_score AMONG the proxy-correct candidates (preregistration: a false prune
    # is pruning a candidate that was proxy-correct AND would have been the winner).
    correct_idx = np.flatnonzero(final_correct)
    if correct_idx.size:
        ref_correct = int(correct_idx[np.lexsort(
            (correct_idx, -pool.final_score[correct_idx]))[0]])
    else:
        ref_correct = None
    best_possible = float(np.max(final_correct.astype(float))) if n else 0.0

    alive = np.arange(n)
    nfe = 0
    scoring = 0
    pruned_idx: list[int] = []

    def _finish(winner: Optional[int], completed_idx: np.ndarray) -> PolicyResult:
        nonlocal nfe, scoring
        # survivors run to completion: charge full NFE for the remaining fraction.
        # (We accumulate prune-NFE eagerly; survivors get num_steps total here.)
        completed = int(completed_idx.shape[0])
        if winner is None:
            wcorr = False
            acorr = {a: False for a in present_axes}
            achieved = 0.0
        else:
            wcorr = bool(final_correct[winner])
            acorr = {a: bool(pool.labels[a][winner]) for a in present_axes}
            achieved = float(final_correct[winner])
        retained = (ref_winner is not None) and (ref_winner in set(completed_idx.tolist()))
        # false prune: the best correct candidate was pruned (proxy-correct AND the pool's
        # would-be correct winner). Pruning the reward-winner when it is proxy-WRONG is a
        # CORRECT prune and never counts here.
        fp = sum(1 for j in pruned_idx if ref_correct is not None and j == ref_correct)
        return PolicyResult(
            policy=policy, clip=pool.clip, winner=winner, winner_correct=wcorr,
            axis_correct=acorr, completed=completed, total_nfe=int(nfe),
            scoring_calls=int(scoring), winner_retained=bool(retained),
            false_pruned=int(fp), regret=float(best_possible - achieved),
        )

    if n == 0:
        return _finish(None, np.array([], dtype=int))

    # ---- full_bon / final_rerank: complete all, score at final window only ----
    if policy in ("full_bon", "final_rerank"):
        nfe = n * num_steps
        scoring = n                      # one final-window score per candidate
        winner = _argmax_by_score(alive, pool.final_score, rng)
        return _finish(winner, alive)

    # ---- same_compute_bon: BoN under a generator-NFE budget (fewer completed) ----
    if policy == "same_compute_bon":
        cap = budget_nfe if budget_nfe is not None else n * num_steps
        k = max(1, min(n, int(cap // num_steps)))
        # deterministic candidate subset: first k by a seeded permutation of indices
        order = rng.permutation(n)
        chosen = np.sort(order[:k])
        nfe = k * num_steps
        scoring = k
        winner = _argmax_by_score(chosen, pool.final_score, rng)
        pruned_idx.extend(int(j) for j in alive if j not in set(chosen.tolist()))
        return _finish(winner, chosen)

    # ---- random_prune: prune a fraction at the first window, complete survivors ----
    if policy == "random_prune":
        s0 = gates[0].s if gates else 0.5
        n_prune = int(round(random_prune_frac * n))
        n_prune = min(n_prune, n - 1)            # never prune the whole pool
        order = rng.permutation(n)
        pruned = np.sort(order[:n_prune])
        survivors = np.sort(order[n_prune:])
        scoring += n                              # scored once at the prune window
        for j in pruned:
            nfe += prune_nfe(s0, num_steps)
            pruned_idx.append(int(j))
        nfe += survivors.shape[0] * num_steps
        scoring += survivors.shape[0]             # final-window rerank score
        winner = _argmax_by_score(survivors, pool.final_score, rng)
        return _finish(winner, survivors)

    # ---- diffrs_scalar: reject below threshold at first window, complete survivors ----
    if policy == "diffrs_scalar":
        s0 = gates[0].s if gates else 0.5
        scoring += n
        keep_mask = pool.final_score >= diffrs_tau
        if not keep_mask.any():
            keep_mask[int(np.argmax(pool.final_score))] = True   # keep best, never empty
        survivors = np.flatnonzero(keep_mask)
        pruned = np.flatnonzero(~keep_mask)
        for j in pruned:
            nfe += prune_nfe(s0, num_steps)
            pruned_idx.append(int(j))
        nfe += survivors.shape[0] * num_steps
        scoring += survivors.shape[0]
        winner = _argmax_by_score(survivors, pool.final_score, rng)
        return _finish(winner, survivors)

    # ---- smc_scalar: resample population ~ softmax(score/T) at first window ----
    if policy == "smc_scalar":
        s0 = gates[0].s if gates else 0.5
        scoring += n
        logits = pool.final_score / max(smc_temp, 1e-6)
        logits = logits - np.max(logits)
        w = np.exp(logits)
        w = w / w.sum()
        resampled = rng.choice(n, size=n, replace=True, p=w)
        kept = np.unique(resampled)               # distinct survivors actually completed
        pruned = np.array([j for j in range(n) if j not in set(kept.tolist())], dtype=int)
        for j in pruned:
            nfe += prune_nfe(s0, num_steps)
            pruned_idx.append(int(j))
        nfe += kept.shape[0] * num_steps
        scoring += kept.shape[0]
        winner = _argmax_by_score(kept, pool.final_score, rng)
        return _finish(winner, kept)

    # ---- oracle_axis_gated: stagewise prune on closed-window axes ----
    if policy == "oracle_axis_gated":
        survivors = alive.copy()
        for gate in gates:
            if survivors.shape[0] <= 1:
                break
            scoring += survivors.shape[0]         # score every survivor at this window
            keep = np.ones(survivors.shape[0], dtype=bool)
            for a in gate.axes:
                if a not in pool.labels:
                    continue
                # keep candidates whose in-window axis self-target matches consensus.
                if a in pool.axis_score:
                    a_ok = pool.axis_score[a][survivors] > 0.5
                else:
                    a_ok = pool.labels[a][survivors]
                keep &= a_ok
            if not keep.any():
                # Empty-pool fallback: keep the best-final_score survivor (the BoN pick).
                # NEVER re-consult true labels here — under the B4 non-oracle replay that would
                # leak ground truth into the scorer; final_score is the honest, non-leaking
                # fallback for both the genuine oracle and the non-oracle gating.
                keep[int(np.argmax(pool.final_score[survivors]))] = True
            newly_pruned = survivors[~keep]
            for j in newly_pruned:
                nfe += prune_nfe(gate.s, num_steps)
                pruned_idx.append(int(j))
            survivors = survivors[keep]
        nfe += survivors.shape[0] * num_steps
        scoring += survivors.shape[0]             # final rerank score
        winner = _argmax_by_score(survivors, pool.final_score, rng)
        return _finish(winner, survivors)

    raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")


# ---------------------------------------------------------------------------
# Aggregation across clips
# ---------------------------------------------------------------------------
def aggregate(results: list[PolicyResult], axes: tuple[str, ...] = DEFAULT_AXES) -> dict:
    """Mean metrics over clips for one policy (preregistration metric set). Empty -> zeros."""
    present_axes = tuple(axes)
    if not results:
        base = {
            "policy": None, "n_clips": 0, "final_correctness": 0.0,
            "completed_candidates": 0.0, "total_nfe": 0, "scoring_calls": 0,
            "winner_retention": 0.0, "false_prune_rate": 0.0, "regret": 0.0,
        }
        for a in present_axes:
            base[f"correct_{a}"] = 0.0
        return base
    n = len(results)
    out = {
        "policy": results[0].policy,
        "n_clips": n,
        "final_correctness": float(np.mean([r.winner_correct for r in results])),
        "completed_candidates": float(np.mean([r.completed for r in results])),
        "total_nfe": int(sum(r.total_nfe for r in results)),
        "scoring_calls": int(sum(r.scoring_calls for r in results)),
        "winner_retention": float(np.mean([r.winner_retained for r in results])),
        "false_prune_rate": float(np.mean([r.false_pruned for r in results])),
        "regret": float(np.mean([r.regret for r in results])),
    }
    for a in present_axes:
        vals = [r.axis_correct[a] for r in results if a in r.axis_correct]
        out[f"correct_{a}"] = float(np.mean(vals)) if vals else 0.0
    return out


def run_all_policies(
    pools: list[ClipPool],
    *,
    gates: list[GateSpec],
    axes: tuple[str, ...] = DEFAULT_AXES,
    num_steps: int = NFE_FULL_DEFAULT,
    seed: int = 0,
    diffrs_tau: float = 0.0,
    smc_temp: float = 1.0,
    random_prune_frac: float = 0.5,
) -> dict[str, dict]:
    """Run every preregistered policy on all pools; return policy -> aggregate metrics.

    same_compute_bon's NFE budget is matched, PER CLIP, to oracle_axis_gated's realized
    generator-NFE (the matched-compute x-coordinate of Fig. 6). Deterministic given `seed`:
    each (policy, clip) gets an independent sub-stream via SeedSequence.
    """
    import zlib

    def _rng(*parts) -> np.random.Generator:
        ent = [seed] + [zlib.crc32(str(p).encode()) for p in parts]
        return np.random.default_rng(np.random.SeedSequence(ent))

    # First pass: gated NFE per clip, to set same_compute_bon's matched budget.
    gated_nfe: dict[str, int] = {}
    per_policy: dict[str, list[PolicyResult]] = {p: [] for p in POLICIES}
    for pool in pools:
        r = simulate_policy(
            pool, "oracle_axis_gated", gates=gates, axes=axes, num_steps=num_steps,
            rng=_rng("oracle_axis_gated", pool.clip),
            diffrs_tau=diffrs_tau, smc_temp=smc_temp, random_prune_frac=random_prune_frac,
        )
        gated_nfe[pool.clip] = r.total_nfe
        per_policy["oracle_axis_gated"].append(r)

    for pool in pools:
        for policy in POLICIES:
            if policy == "oracle_axis_gated":
                continue
            r = simulate_policy(
                pool, policy, gates=gates, axes=axes, num_steps=num_steps,
                rng=_rng(policy, pool.clip),
                budget_nfe=gated_nfe.get(pool.clip),
                diffrs_tau=diffrs_tau, smc_temp=smc_temp, random_prune_frac=random_prune_frac,
            )
            per_policy[policy].append(r)

    return {p: aggregate(per_policy[p], axes) for p in POLICIES}
