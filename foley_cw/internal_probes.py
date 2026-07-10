"""Track P — internal-feature probes (manual §7). numpy-only, non-gating.

Trains a linear probe per (axis, layer, s) on the cached pooled per-layer features of the
independents pool, predicting each trajectory's OWN final self-target (label-free w.r.t.
human gold). Reports the best-layer internal readout curve s_read_internal(axis) — the
"generator knows before the audio shows" evidence (Fig 4). Outputs never feed decision
tokens (§1.6/§15.4).

The probe is a ridge (regularized least-squares) linear classifier, closed-form and
numpy-only (no sklearn): one-hot the labels, solve W = (XᵀX + λI)⁻¹ XᵀY on the frozen
train split, predict argmax on eval. This is the standard linear-probe readout; accuracy
on the held-out split is the internal-readout score, compared to θ_read for s_read_internal.
"""
from __future__ import annotations

import numpy as np


def ridge_linear_classifier(X: np.ndarray, y: list, lam: float = 1.0):
    """Fit a ridge one-vs-all linear classifier. Returns (W, b, classes).

    X: (n, d) features; y: length-n labels (any hashable). Adds a bias column.
    """
    classes = sorted(set(y), key=str)
    cidx = {c: i for i, c in enumerate(classes)}
    n, d = X.shape
    Y = np.zeros((n, len(classes)), dtype=float)
    for i, lab in enumerate(y):
        Y[i, cidx[lab]] = 1.0
    Xb = np.concatenate([X, np.ones((n, 1))], axis=1)            # bias
    A = Xb.T @ Xb + lam * np.eye(d + 1)
    W = np.linalg.solve(A, Xb.T @ Y)                            # (d+1, k)
    return W, classes


def predict(W, classes, X: np.ndarray) -> list:
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    scores = Xb @ W
    idx = np.argmax(scores, axis=1)
    return [classes[i] for i in idx]


def probe_accuracy(X_tr: np.ndarray, y_tr: list, X_te: np.ndarray, y_te: list,
                   lam: float = 1.0) -> float:
    """Train on (X_tr,y_tr), return eval accuracy on (X_te,y_te). NaN if degenerate."""
    if len(set(y_tr)) < 2 or X_tr.shape[0] < 2 or X_te.shape[0] == 0:
        # single-class train: predict the majority; accuracy = majority share on eval
        if X_te.shape[0] == 0:
            return float("nan")
        maj = max(set(y_tr), key=y_tr.count) if y_tr else None
        return float(np.mean([1.0 if t == maj else 0.0 for t in y_te]))
    # standardize features (per-dim) using train stats
    mu = X_tr.mean(axis=0); sd = X_tr.std(axis=0); sd = np.where(sd == 0, 1.0, sd)
    W, classes = ridge_linear_classifier((X_tr - mu) / sd, y_tr, lam)
    pred = predict(W, classes, (X_te - mu) / sd)
    return float(np.mean([1.0 if p == t else 0.0 for p, t in zip(pred, y_te)]))


def best_layer_curve(
    feats_by_layer_s: dict,          # (layer, s) -> {"train": (X,y), "eval": (X,y)}
    layers: list[int],
    s_grid: list[float],
    lam: float = 1.0,
) -> dict:
    """Per s: accuracy at each layer; best-layer accuracy + argmax layer."""
    out = {"per_s": {}, "best_layer_acc": {}, "best_layer": {}}
    for s in s_grid:
        accs = {}
        for L in layers:
            d = feats_by_layer_s.get((L, s))
            if d is None:
                continue
            Xtr, ytr = d["train"]; Xte, yte = d["eval"]
            accs[L] = probe_accuracy(Xtr, ytr, Xte, yte, lam)
        out["per_s"][s] = accs
        finite = {L: a for L, a in accs.items() if np.isfinite(a)}
        if finite:
            bestL = max(finite, key=finite.get)
            out["best_layer_acc"][s] = float(finite[bestL])
            out["best_layer"][s] = int(bestL)
        else:
            out["best_layer_acc"][s] = float("nan")
            out["best_layer"][s] = None
    return out


def s_read_internal(best_acc_by_s: dict[float, float], theta_read: float) -> float:
    """Earliest s where the best-layer internal probe reaches θ_read; NaN if never."""
    for s in sorted(best_acc_by_s):
        a = best_acc_by_s[s]
        if np.isfinite(a) and a >= theta_read:
            return float(s)
    return float("nan")
