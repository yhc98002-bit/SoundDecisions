"""Arc-3 Tier-B §B2 — conditioning-channel audit helpers (CPU, numpy-first).

Pre-registration (experiment/preregistered/arc3_tierB_preregistration.md §B2):

  Question: do the RAW video-conditioning features (CLIP + Synchformer, pre-DiT)
  carry the final class at all? Tests whether class non-readability is a
  *conditioning bottleneck* vs a DiT-internal property.

  Probe: linear ridge + MLP (as B1) predicting the clip's MEASURED class from the
  pooled raw CLIP and Synchformer conditioning tensors, on the FROZEN 60/40 clip
  split. Metric: held-out eval accuracy vs chance (= eval majority-class prior),
  bootstrap by video for CIs.

  Decision rule: cond-feature class acc <= chance + 0.15 AND substantially below
  the DiT-internal best (B1) -> emit COND_BOTTLENECK. CONTINUE regardless.

This module is the CPU/numpy core (pooling, label derivation, probe + bootstrap
math). The GPU-only conditioning extraction lives in scripts/b2_cond_audit.py.
The ridge probe reuses foley_cw.internal_probes.probe_accuracy (same standardize +
closed-form ridge one-vs-all as B1). The MLP mirrors B1's family (1 hidden layer,
width 256, ReLU, weight-decay 1e-3, early-stopped); sklearn is used when available,
else a small numpy fallback so the math is testable CPU-only with no sklearn.

DATA-LEAKAGE DISCIPLINE: probes train ONLY on frozen probe_train clips and are
evaluated ONLY on eval clips; a clip's features never appear in both. The per-clip
label is derived from that clip's own independents' measured class (a self-target
proxy, documented), never from human gold.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from .internal_probes import probe_accuracy

# Frozen majority-vote label proxy (mirrors the B4 per-clip majority self-target).
ABSTAIN = "abstain"

# MLP family (pre-reg §B1, reused in §B2): 1 hidden layer, width 256, ReLU, wd 1e-3.
MLP_HIDDEN = 256
MLP_WEIGHT_DECAY = 1e-3
MLP_MAX_ITER = 300
MLP_SEED = 0


# ---------------------------------------------------------------------------
# Pooling of the raw conditioning tensors
# ---------------------------------------------------------------------------
def pool_cond_tensor(arr: np.ndarray) -> np.ndarray:
    """Mean+max pool a (T, D) conditioning tensor along the sequence axis -> (2D,).

    Accepts (T, D) or (1, T, D) (a leading batch axis is squeezed). A 1-D vector
    (already a per-clip embedding, e.g. clip_f_c) is returned unchanged.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 1:
        return a.astype(np.float32)
    if a.ndim != 2:
        raise ValueError(f"pool_cond_tensor expects (T,D) or (1,T,D); got {a.shape}")
    return np.concatenate([a.mean(axis=0), a.max(axis=0)]).astype(np.float32)


def build_cond_feature(parts: dict[str, np.ndarray], keys: list[str]) -> np.ndarray:
    """Concatenate pooled conditioning parts in a fixed key order -> 1-D feature."""
    return np.concatenate([pool_cond_tensor(parts[k]) for k in keys]).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-clip label derivation (measured class majority self-target proxy)
# ---------------------------------------------------------------------------
def clip_class_label(labels: list[str], drop_abstain: bool = True) -> Optional[str]:
    """Majority measured-class label over a clip's independents (self-target proxy).

    Mirrors the confident-subset spirit: abstain votes are dropped when any
    non-abstain vote exists, so the proxy is the clip's dominant confident class.
    Returns None if the clip has no usable (non-abstain) class vote.
    """
    votes = [str(x) for x in labels if x is not None]
    if drop_abstain:
        confident = [v for v in votes if v != ABSTAIN]
        if confident:
            votes = confident
        else:
            return None
    if not votes:
        return None
    # Deterministic tie-break: most common, then lexicographic.
    counts = Counter(votes)
    top = max(counts.items(), key=lambda kv: (kv[1], _neg_key(kv[0])))
    return top[0]


def _neg_key(s: str):
    # max() with this as secondary key picks the lexicographically-SMALLEST label
    # among ties (stable, deterministic) without sorting the whole counter.
    return tuple(-ord(c) for c in s)


def majority_class_accuracy(y_train: list[str], y_eval: list[str]) -> float:
    """Chance baseline = eval accuracy of predicting the eval majority-class prior.

    Per pre-reg §B1/§B2 'chance = eval majority-class prior (reported alongside)'.
    """
    if not y_eval:
        return float("nan")
    maj = Counter(y_eval).most_common(1)[0][0]
    return float(np.mean([1.0 if t == maj else 0.0 for t in y_eval]))


# ---------------------------------------------------------------------------
# MLP probe (sklearn when present; numpy fallback otherwise)
# ---------------------------------------------------------------------------
def _mlp_accuracy_sklearn(X_tr, y_tr, X_te, y_te) -> float:
    from sklearn.neural_network import MLPClassifier
    mu = X_tr.mean(axis=0); sd = X_tr.std(axis=0); sd = np.where(sd == 0, 1.0, sd)
    Xtr = (X_tr - mu) / sd; Xte = (X_te - mu) / sd
    clf = MLPClassifier(
        hidden_layer_sizes=(MLP_HIDDEN,), activation="relu", alpha=MLP_WEIGHT_DECAY,
        solver="adam", max_iter=MLP_MAX_ITER, early_stopping=True, n_iter_no_change=15,
        validation_fraction=0.15, random_state=MLP_SEED,
    )
    clf.fit(Xtr, y_tr)
    pred = clf.predict(Xte)
    return float(np.mean([1.0 if p == t else 0.0 for p, t in zip(pred, y_te)]))


def _mlp_accuracy_numpy(X_tr, y_tr, X_te, y_te) -> float:
    """One-hidden-layer ReLU MLP, full-batch gradient descent with weight decay and
    an early-stop on a held-out slice of TRAIN (never touches eval). numpy-only
    fallback so the probe math runs/tests with no sklearn."""
    rng = np.random.default_rng(MLP_SEED)
    classes = sorted(set(y_tr), key=str)
    cidx = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    mu = X_tr.mean(axis=0); sd = X_tr.std(axis=0); sd = np.where(sd == 0, 1.0, sd)
    Xtr_all = (X_tr - mu) / sd
    Ytr_all = np.array([cidx[c] for c in y_tr])
    # internal early-stop split (15% of train), deterministic
    n = Xtr_all.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(round(0.15 * n))) if n >= 8 else 0
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xt, Yt = Xtr_all[tr_idx], Ytr_all[tr_idx]
    Xv, Yv = Xtr_all[val_idx], Ytr_all[val_idx]
    d = Xt.shape[1]
    W1 = rng.normal(0, np.sqrt(2.0 / d), (d, MLP_HIDDEN))
    b1 = np.zeros(MLP_HIDDEN)
    W2 = rng.normal(0, np.sqrt(2.0 / MLP_HIDDEN), (MLP_HIDDEN, k))
    b2 = np.zeros(k)
    Yoh = np.zeros((Xt.shape[0], k)); Yoh[np.arange(Xt.shape[0]), Yt] = 1.0
    lr = 0.05
    best_W = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
    best_val = -1.0; stall = 0

    def forward(X, W1, b1, W2, b2):
        h = np.maximum(0.0, X @ W1 + b1)
        z = h @ W2 + b2
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
        return h, p

    for _ in range(MLP_MAX_ITER):
        h, p = forward(Xt, W1, b1, W2, b2)
        m = Xt.shape[0]
        dz = (p - Yoh) / m
        dW2 = h.T @ dz + MLP_WEIGHT_DECAY * W2; db2 = dz.sum(axis=0)
        dh = dz @ W2.T; dh[h <= 0] = 0.0
        dW1 = Xt.T @ dh + MLP_WEIGHT_DECAY * W1; db1 = dh.sum(axis=0)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
        if n_val:
            _, pv = forward(Xv, W1, b1, W2, b2)
            acc = float(np.mean(pv.argmax(axis=1) == Yv))
            if acc > best_val + 1e-4:
                best_val = acc; best_W = (W1.copy(), b1.copy(), W2.copy(), b2.copy()); stall = 0
            else:
                stall += 1
                if stall >= 15:
                    break
    W1, b1, W2, b2 = best_W if n_val else (W1, b1, W2, b2)
    Xte = (X_te - mu) / sd
    _, pte = forward(Xte, W1, b1, W2, b2)
    pred = [classes[i] for i in pte.argmax(axis=1)]
    return float(np.mean([1.0 if pp == tt else 0.0 for pp, tt in zip(pred, y_te)]))


def mlp_accuracy(X_tr, y_tr, X_te, y_te) -> float:
    """MLP-probe eval accuracy (B1 family). NaN on degenerate train/eval."""
    if len(set(y_tr)) < 2 or X_tr.shape[0] < 2 or X_te.shape[0] == 0:
        if X_te.shape[0] == 0:
            return float("nan")
        maj = Counter(y_tr).most_common(1)[0][0] if y_tr else None
        return float(np.mean([1.0 if t == maj else 0.0 for t in y_te]))
    try:
        return _mlp_accuracy_sklearn(X_tr, y_tr, X_te, y_te)
    except Exception:
        return _mlp_accuracy_numpy(X_tr, y_tr, X_te, y_te)


# ---------------------------------------------------------------------------
# Bootstrap-by-video CI on the eval-accuracy of a fitted prediction
# ---------------------------------------------------------------------------
def bootstrap_acc_ci(correct: np.ndarray, eval_clips: list[str], n_boot: int = 1000,
                     seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI for eval accuracy, RESAMPLING UNIT = video (clip).

    correct: 0/1 per eval row; eval_clips: parallel clip id per eval row.
    Resamples clips with replacement; a clip contributes all its eval rows.
    """
    correct = np.asarray(correct, dtype=float)
    clips = np.asarray(eval_clips, dtype=object)
    uniq = sorted(set(eval_clips))
    by_clip = {c: correct[clips == c] for c in uniq}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    nclip = len(uniq)
    for b in range(n_boot):
        pick = rng.integers(0, nclip, nclip)
        vals = np.concatenate([by_clip[uniq[i]] for i in pick])
        boots[b] = vals.mean() if vals.size else np.nan
    lo = float(np.nanpercentile(boots, 100 * alpha / 2))
    hi = float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# Top-level probe runner over a feature matrix + frozen split
# ---------------------------------------------------------------------------
def run_cond_probe(
    X: np.ndarray,
    y: list[str],
    clips: list[str],
    train_clips: set,
    eval_clips_set: set,
    family: str = "ridge",
    lam: float = 1.0,
    n_boot: int = 1000,
    boot_seed: int = 0,
) -> dict:
    """Fit `family` probe on train clips, evaluate on eval clips; return acc, chance,
    n, and a bootstrap-by-video CI. y rows with None label are dropped on both sides.

    family in {"ridge", "mlp"}. Reuses internal_probes.probe_accuracy for ridge.
    """
    X = np.asarray(X, dtype=np.float64)
    y = list(y)
    clips = list(clips)
    keep = np.array([lab is not None for lab in y])
    is_tr = np.array([c in train_clips for c in clips]) & keep
    is_te = np.array([c in eval_clips_set for c in clips]) & keep
    Xtr, ytr = X[is_tr], [y[i] for i in np.where(is_tr)[0]]
    Xte, yte = X[is_te], [y[i] for i in np.where(is_te)[0]]
    eval_clip_ids = [clips[i] for i in np.where(is_te)[0]]

    chance = majority_class_accuracy(ytr, yte)
    if family == "ridge":
        acc = probe_accuracy(Xtr, ytr, Xte, yte, lam=lam)
        correct = _ridge_correct(Xtr, ytr, Xte, yte, lam)
    elif family == "mlp":
        acc = mlp_accuracy(Xtr, ytr, Xte, yte)
        correct = _mlp_correct(Xtr, ytr, Xte, yte)
    else:
        raise ValueError(f"unknown family {family!r}")
    if correct is not None and len(correct) == len(eval_clip_ids) and len(correct):
        lo, hi = bootstrap_acc_ci(correct, eval_clip_ids, n_boot=n_boot, seed=boot_seed)
    else:
        lo = hi = float("nan")
    return {
        "family": family, "accuracy": float(acc), "chance": float(chance),
        "delta_over_chance": float(acc - chance) if np.isfinite(acc) and np.isfinite(chance) else float("nan"),
        "ci95": [lo, hi], "n_train": int(len(ytr)), "n_eval": int(len(yte)),
        "n_classes": int(len(set(ytr))),
    }


def _ridge_correct(Xtr, ytr, Xte, yte, lam):
    from .internal_probes import predict, ridge_linear_classifier
    if len(set(ytr)) < 2 or Xtr.shape[0] < 2 or Xte.shape[0] == 0:
        if Xte.shape[0] == 0:
            return None
        maj = max(set(ytr), key=ytr.count) if ytr else None
        return np.array([1.0 if t == maj else 0.0 for t in yte])
    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd = np.where(sd == 0, 1.0, sd)
    W, classes = ridge_linear_classifier((Xtr - mu) / sd, ytr, lam)
    pred = predict(W, classes, (Xte - mu) / sd)
    return np.array([1.0 if p == t else 0.0 for p, t in zip(pred, yte)])


def _mlp_correct(Xtr, ytr, Xte, yte):
    # Re-fit once and capture per-row correctness for the bootstrap. Falls back to a
    # constant-correct vector consistent with the reported accuracy if a probe path
    # is degenerate (mirrors mlp_accuracy's degenerate handling).
    if len(set(ytr)) < 2 or Xtr.shape[0] < 2 or Xte.shape[0] == 0:
        if Xte.shape[0] == 0:
            return None
        maj = Counter(ytr).most_common(1)[0][0] if ytr else None
        return np.array([1.0 if t == maj else 0.0 for t in yte])
    try:
        from sklearn.neural_network import MLPClassifier
        mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd = np.where(sd == 0, 1.0, sd)
        clf = MLPClassifier(
            hidden_layer_sizes=(MLP_HIDDEN,), activation="relu", alpha=MLP_WEIGHT_DECAY,
            solver="adam", max_iter=MLP_MAX_ITER, early_stopping=True, n_iter_no_change=15,
            validation_fraction=0.15, random_state=MLP_SEED,
        )
        clf.fit((Xtr - mu) / sd, ytr)
        pred = clf.predict((Xte - mu) / sd)
        return np.array([1.0 if p == t else 0.0 for p, t in zip(pred, yte)])
    except Exception:
        # numpy-fallback correctness via a thin wrapper around _mlp_accuracy_numpy is
        # not exposed per-row; recompute predictions inline.
        return _mlp_correct_numpy(Xtr, ytr, Xte, yte)


def _mlp_correct_numpy(Xtr, ytr, Xte, yte):
    # Mirror _mlp_accuracy_numpy but return per-row correctness.
    rng = np.random.default_rng(MLP_SEED)
    classes = sorted(set(ytr), key=str)
    cidx = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd = np.where(sd == 0, 1.0, sd)
    X = (Xtr - mu) / sd
    Y = np.array([cidx[c] for c in ytr])
    n, d = X.shape
    perm = rng.permutation(n)
    n_val = max(1, int(round(0.15 * n))) if n >= 8 else 0
    tr_idx = perm[n_val:]
    Xt, Yt = X[tr_idx], Y[tr_idx]
    W1 = rng.normal(0, np.sqrt(2.0 / d), (d, MLP_HIDDEN)); b1 = np.zeros(MLP_HIDDEN)
    W2 = rng.normal(0, np.sqrt(2.0 / MLP_HIDDEN), (MLP_HIDDEN, k)); b2 = np.zeros(k)
    Yoh = np.zeros((Xt.shape[0], k)); Yoh[np.arange(Xt.shape[0]), Yt] = 1.0
    lr = 0.05
    for _ in range(MLP_MAX_ITER):
        h = np.maximum(0.0, Xt @ W1 + b1)
        z = h @ W2 + b2; z -= z.max(axis=1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
        m = Xt.shape[0]; dz = (p - Yoh) / m
        dW2 = h.T @ dz + MLP_WEIGHT_DECAY * W2; db2 = dz.sum(axis=0)
        dh = dz @ W2.T; dh[h <= 0] = 0.0
        dW1 = Xt.T @ dh + MLP_WEIGHT_DECAY * W1; db1 = dh.sum(axis=0)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
    Xe = (Xte - mu) / sd
    he = np.maximum(0.0, Xe @ W1 + b1); ze = he @ W2 + b2
    pred = [classes[i] for i in ze.argmax(axis=1)]
    return np.array([1.0 if pp == tt else 0.0 for pp, tt in zip(pred, yte)])


def decide_cond_bottleneck(cond_best_acc: float, chance: float, b1_best_acc: float,
                           margin: float = 0.15, b1_gap: float = 0.15) -> dict:
    """Pre-reg §B2 decision: COND_BOTTLENECK iff cond class acc <= chance+margin AND
    substantially below B1's DiT-internal best. We operationalize 'substantially below'
    as cond_best_acc <= b1_best_acc - b1_gap (documented; b1_gap default 0.15).

    Returns {token, near_chance, below_b1, ...}. CONTINUE regardless (no pause)."""
    near_chance = bool(np.isfinite(cond_best_acc) and np.isfinite(chance)
                       and cond_best_acc <= chance + margin)
    below_b1 = bool(np.isfinite(b1_best_acc) and cond_best_acc <= b1_best_acc - b1_gap)
    token = "COND_BOTTLENECK" if (near_chance and below_b1) else "COND_NOT_BOTTLENECK"
    return {
        "token": token, "near_chance": near_chance, "below_b1": below_b1,
        "cond_best_acc": float(cond_best_acc),
        "chance": float(chance) if np.isfinite(chance) else None,
        "b1_best_acc": float(b1_best_acc) if np.isfinite(b1_best_acc) else None,
        "margin": margin, "b1_gap": b1_gap, "continue": True,
    }
