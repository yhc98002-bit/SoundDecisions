"""Agreement metrics for foley_cw self-target distributions.

Three metrics are implemented, all in numpy (no scipy, no torch):

  * categorical_agreement  — mean pairwise exact-match probability in [0,1];
                             1.0 when all labels are identical.
  * krippendorff_alpha_nominal — Krippendorff's alpha for nominal (categorical)
                                 data, numpy-only.  Handles the all-agree and
                                 single-category degenerate cases without NaN.
  * mean_pairwise_cosine    — mean of cosine similarities over all i<j pairs;
                              result in [-1, 1].

Dispatch via `agreement(targets, metric)` which takes a list[SelfTarget] and
the axis's AgreementMetric enum member.

Scientific contract (refine-logs/EXPERIMENT_PLAN.md §1 / Phase 1):
  * Agreement is measured on the model's OWN self-targets (self-agreement),
    NOT correctness-vs-video.
  * Categorical axes use exact-match or Krippendorff's alpha depending on the
    axis's registered `agreement` field.
  * Embedding axes use mean pairwise cosine.
  * n < 2 targets → return 1.0 (trivially unanimous).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .types import AgreementMetric, SelfTarget


# ---------------------------------------------------------------------------
# Categorical agreement helpers
# ---------------------------------------------------------------------------

def categorical_agreement(labels: list) -> float:
    """Mean pairwise exact-match probability in [0, 1].

    Agreement(labels) = fraction of all unordered pairs (i, j) with i != j
    where labels[i] == labels[j].  Returns 1.0 for n < 2 or all identical.

    This metric is used for the presence and timing axes (exact_match).
    """
    n = len(labels)
    if n < 2:
        return 1.0
    n_pairs = n * (n - 1) // 2
    if n_pairs == 0:
        return 1.0
    matches = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                matches += 1
    return float(matches) / float(n_pairs)


def krippendorff_alpha_nominal(labels: list) -> float:
    """Krippendorff's alpha for nominal (categorical) data, numpy-only.

    Treats `labels` as a SINGLE rater / coder providing one observation per
    unit — equivalently, treats all K items as ratings on K units by a single
    coder.  In the fork-agreement setting the "units" are generation slots and
    the "values" are the measured self-targets.

    Formula (nominal MTYPE):
        alpha = 1 - D_o / D_e

    where:
        D_o = observed disagreement = (1 / n*(n-1)) * sum_{i!=j} delta(v_i, v_j)
        D_e = expected disagreement = sum_{k,l} n_k * n_l * delta(k, l)
                                      divided by n*(n-1)
        delta(v_i, v_j) = 0 if equal, 1 otherwise (nominal metric)

    Degenerate cases (all labels identical; only one distinct value; n < 2)
    return 1.0.  Numeric instability near D_e=0 is handled by clamping alpha
    to 1.0 rather than dividing by zero.

    Reference: Krippendorff (2004), Content Analysis. The nominal case gives
    a range of [-(n-1), 1] in theory; in practice with positive D_e it is at
    most 1.
    """
    n = len(labels)
    if n < 2:
        return 1.0

    # Count observed disagreements: pairs where labels differ
    n_pairs = n * (n - 1)          # ordered pairs (i, j), i != j
    obs_disagree = 0
    for i in range(n):
        for j in range(n):
            if i != j and labels[i] != labels[j]:
                obs_disagree += 1

    # Observed disagreement rate
    D_o = obs_disagree / n_pairs

    # All agree → D_o = 0 → alpha = 1.0
    if D_o == 0.0:
        return 1.0

    # Expected disagreement: sum_{k != l} n_k * n_l / (n*(n-1))
    # where n_k = count of label k
    unique, counts = np.unique([str(x) for x in labels], return_counts=True)

    if len(unique) == 1:
        # Only one category; D_e = 0; all observations agree → alpha = 1.0
        return 1.0

    # Sum n_k*(n-n_k) over all k == sum_{k!=l} n_k * n_l
    # More directly: n*(n-1) - sum_k n_k*(n_k-1)
    counts_i = counts.astype(float)
    sum_k_nk_nk_minus1 = float(np.sum(counts_i * (counts_i - 1.0)))
    exp_disagree_count = float(n * (n - 1)) - sum_k_nk_nk_minus1
    D_e = exp_disagree_count / float(n * (n - 1))

    if D_e <= 0.0:
        # All units have the same value in the marginal → alpha = 1.0 (guard)
        return 1.0

    alpha = 1.0 - D_o / D_e
    return float(alpha)


# ---------------------------------------------------------------------------
# Embedding agreement
# ---------------------------------------------------------------------------

def mean_pairwise_cosine(embeddings: np.ndarray) -> float:
    """Mean cosine similarity over all i < j pairs in [-1, 1].

    `embeddings` has shape (n, d).  Zero-norm vectors are handled by clipping
    cosines to [-1, 1].  Returns 1.0 for n < 2.
    """
    embeddings = np.asarray(embeddings, dtype=float)
    n = embeddings.shape[0]
    if n < 2:
        return 1.0
    # Unit-normalize rows; guard against zero-norm vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normed = embeddings / norms

    # Gram matrix of cosines
    gram = normed @ normed.T   # (n, n)

    # Extract upper-triangle (i < j) and average
    idx_i, idx_j = np.triu_indices(n, k=1)
    cosines = gram[idx_i, idx_j]
    cosines = np.clip(cosines, -1.0, 1.0)
    return float(np.mean(cosines))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def agreement(targets: list[SelfTarget], metric: AgreementMetric) -> float:
    """Compute agreement across a list of SelfTargets using the stated metric.

    Parameters
    ----------
    targets:
        A list of SelfTarget objects (all for the same axis, same generation
        batch — either K fork completions or N independent generations).
    metric:
        The AgreementMetric registered on the axis.

    Returns
    -------
    float in [0, 1] for EXACT_MATCH and MEAN_PAIRWISE_COSINE; up to 1.0 for
    KRIPPENDORFF_ALPHA (can be negative but is clipped to 0.0 floor when used
    as a probability-like quantity in commit_gain; raw value returned here).
    Returns 1.0 for n < 2 regardless of metric.

    Scientific note: This function measures SELF-agreement (model's own value
    repeatability), NOT correctness-vs-video.  The agreement value feeds the
    normalized commitment gain formula
        commit_gain = clip((A_fork - A_ind) / (1 - A_ind), 0, 1)
    (refine-logs/EXPERIMENT_PLAN.md Phase 1 §"Video-prior normalization").
    """
    if len(targets) < 2:
        return 1.0

    if metric is AgreementMetric.EXACT_MATCH:
        labels = [t.label for t in targets]
        return categorical_agreement(labels)

    if metric is AgreementMetric.KRIPPENDORFF_ALPHA:
        labels = [t.label for t in targets]
        return krippendorff_alpha_nominal(labels)

    if metric is AgreementMetric.MEAN_PAIRWISE_COSINE:
        embs = np.stack([np.asarray(t.embedding, dtype=float) for t in targets], axis=0)
        return mean_pairwise_cosine(embs)

    raise ValueError(f"Unknown AgreementMetric: {metric!r}")


def confident_agreement(labels: list, metric: AgreementMetric,
                        abstain="abstain") -> tuple[float, int]:
    """Agreement on the CONFIDENT subset (revised manual 3.3; frozen
    interpretation #3 in stage_m_rerun_interpretations.md).

    Returns (agreement_or_NaN, n_confident).

    Semantics that deliberately differ from ``agreement()``:
      * abstain labels are dropped before scoring — an abstain-abstain pair NEVER
        counts as agreement (an always-abstaining instrument must not pass an
        endpoint criterion);
      * n_confident < 2 -> NaN (the cell is unscorable), NOT the n<2 -> 1.0
        convention of ``agreement()`` (which is wrong for this purpose);
      * EXACT_MATCH: pairwise exact-match over confident labels;
      * KRIPPENDORFF_ALPHA: abstains are missing ratings; computed on the
        confident labels (all-identical -> 1.0 via the existing degenerate
        guard). Reported-only for the class axis per frozen interpretation #5.

    Callers report abstain_rate = 1 - n_confident/len(labels) alongside.
    """
    confident = [l for l in labels if l != abstain]
    n_conf = len(confident)
    if n_conf < 2:
        return float("nan"), n_conf
    if metric is AgreementMetric.EXACT_MATCH:
        return categorical_agreement(confident), n_conf
    if metric is AgreementMetric.KRIPPENDORFF_ALPHA:
        return krippendorff_alpha_nominal(confident), n_conf
    raise ValueError(f"confident_agreement is for categorical metrics, got {metric!r}")
