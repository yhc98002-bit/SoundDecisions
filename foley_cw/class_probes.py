"""Arc-3 Tier-B B1 — class readability, fairly tested (pre-reg §B1). NON-GATING.

Tests whether each cfg=1.0 independent's FINAL class is readable EARLY from the
generator's internal POOLED per-layer features, with two probe families fairly compared:

  1. Linear ridge on the pooled per-layer features (the existing Track-P baseline;
     reused via foley_cw.internal_probes.probe_accuracy).
  2. An MLP (1 hidden layer, width 256, ReLU, weight-decay 1e-3, early-stopped on a
     train-internal split) on the same pooled per-layer features.

Both are swept over all 12 joint blocks x the 8-point Phase-1 s-grid, trained ONLY on the
frozen probe_train clips' independents and evaluated ONLY on the eval clips' independents
(no clip in both — data-leakage discipline). Metric = held-out class accuracy; chance =
eval majority-class prior.

Decision (pre-reg §B1, applied by the driver script, NOT here):
  s_read_internal_class = min s with the best (probe x layer) eval acc >= theta_read (0.70)
  AND >= chance + 0.15. s_read_internal_class <= 0.45 -> CLASS_INTERNAL_READOUT_FOUND
  (record the winning (probe, layer) as the B4 class head); else R2_CLASS_CONFIRMED.

The MLP is implemented with sklearn.neural_network.MLPClassifier when sklearn is present
(it is, in this venv), else a numpy-hand-rolled fallback with the SAME architecture and
early-stopping contract. Labels are each trajectory's own measured final class self-target,
kept verbatim including the measurer's 'abstain' outcome, matching the frozen Track-P
convention (scripts/track_p.py.load_labels). Outputs never feed decision tokens.
"""
from __future__ import annotations

import numpy as np

from foley_cw.internal_probes import probe_accuracy  # linear ridge baseline (reused)

try:  # prefer sklearn's MLP (present in this venv); fall back to a numpy twin otherwise
    from sklearn.neural_network import MLPClassifier  # type: ignore
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - exercised only when sklearn missing
    _HAVE_SKLEARN = False


# ---------------------------------------------------------------------------
# chance baseline
# ---------------------------------------------------------------------------
def eval_majority_prior(y_eval: list) -> float:
    """Chance = fraction of the eval set in its single most-common class (majority prior)."""
    if not y_eval:
        return float("nan")
    vals, counts = np.unique(np.asarray(y_eval, dtype=object), return_counts=True)
    return float(counts.max() / len(y_eval))


# ---------------------------------------------------------------------------
# MLP probe (sklearn primary, numpy hand-rolled fallback). Same architecture +
# early-stopping contract either way.
# ---------------------------------------------------------------------------
def _standardize(X_tr, X_te):
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X_tr - mu) / sd, (X_te - mu) / sd


def mlp_probe_accuracy(
    X_tr: np.ndarray, y_tr: list, X_te: np.ndarray, y_te: list,
    *, hidden: int = 256, weight_decay: float = 1e-3, max_epochs: int = 300,
    val_frac: float = 0.2, patience: int = 15, lr: float = 1e-3, seed: int = 0,
) -> float:
    """Train a 1-hidden-layer (width `hidden`) ReLU MLP with L2 weight-decay
    `weight_decay`, early-stopped on a train-internal val split; return eval accuracy.

    Degenerate cases mirror probe_accuracy: empty eval -> NaN; <2 train classes ->
    predict the train-majority on eval (so the MLP can never beat a constant when the
    train signal is degenerate).
    """
    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_te = np.asarray(X_te, dtype=np.float64)
    if X_te.shape[0] == 0:
        return float("nan")
    if len(set(y_tr)) < 2 or X_tr.shape[0] < 4:
        if X_tr.shape[0] == 0:
            return float("nan")
        maj = max(set(y_tr), key=list(y_tr).count)
        return float(np.mean([1.0 if t == maj else 0.0 for t in y_te]))

    Xtr_s, Xte_s = _standardize(X_tr, X_te)

    if _HAVE_SKLEARN:
        # Encode labels to integer codes: sklearn 1.7.2's early-stopping scorer calls
        # np.isnan on the predicted labels, which raises on string classes. Integer
        # codes avoid that path; decode is unnecessary since we only compare to truth.
        classes = sorted(set(y_tr), key=str)
        cidx = {c: i for i, c in enumerate(classes)}
        ytr_codes = np.array([cidx[c] for c in y_tr])
        yte_codes = np.array([cidx.get(c, -1) for c in y_te])  # unseen eval class -> -1 (never matches)
        clf = MLPClassifier(
            hidden_layer_sizes=(hidden,), activation="relu", solver="adam",
            alpha=weight_decay, learning_rate_init=lr, max_iter=max_epochs,
            early_stopping=True, validation_fraction=val_frac, n_iter_no_change=patience,
            random_state=seed,
        )
        clf.fit(Xtr_s, ytr_codes)
        pred = clf.predict(Xte_s)
        return float(np.mean(pred == yte_codes))

    return _mlp_numpy(Xtr_s, list(y_tr), Xte_s, list(y_te), hidden, weight_decay,
                      max_epochs, val_frac, patience, lr, seed)


def _mlp_numpy(Xtr, ytr, Xte, yte, hidden, weight_decay, max_epochs, val_frac,
               patience, lr, seed) -> float:
    """Hand-rolled twin: 1 hidden ReLU layer, softmax, L2 decay, Adam, early-stop on a
    held-out fraction of the TRAIN clips' samples (val split is train-internal only)."""
    rng = np.random.default_rng(seed)
    classes = sorted(set(ytr), key=str)
    cidx = {c: i for i, c in enumerate(classes)}
    y = np.array([cidx[c] for c in ytr])
    n, d = Xtr.shape
    k = len(classes)

    perm = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    va, tr = perm[:n_val], perm[n_val:]
    Xt, yt = Xtr[tr], y[tr]
    Xv, yv = Xtr[va], y[va]

    # He init for ReLU layer, small for the head.
    W1 = rng.normal(0, np.sqrt(2.0 / d), (d, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, np.sqrt(1.0 / hidden), (hidden, k))
    b2 = np.zeros(k)
    mW1 = vW1 = np.zeros_like(W1); mb1 = vb1 = np.zeros_like(b1)
    mW2 = vW2 = np.zeros_like(W2); mb2 = vb2 = np.zeros_like(b2)
    b1_, b2_, eps = 0.9, 0.999, 1e-8

    def forward(X):
        h = np.maximum(0.0, X @ W1 + b1)
        logits = h @ W2 + b2
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
        return h, p

    def val_acc():
        _, p = forward(Xv)
        return float(np.mean(p.argmax(1) == yv))

    best_acc, best, bad, t = -1.0, None, 0, 0
    bs = min(64, len(Xt))
    for _ in range(max_epochs):
        order = rng.permutation(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = order[i:i + bs]
            xb, yb = Xt[idx], yt[idx]
            h, p = forward(xb)
            m = len(idx)
            dlogits = p.copy(); dlogits[np.arange(m), yb] -= 1.0; dlogits /= m
            gW2 = h.T @ dlogits + weight_decay * W2
            gb2 = dlogits.sum(0)
            dh = (dlogits @ W2.T) * (h > 0)
            gW1 = xb.T @ dh + weight_decay * W1
            gb1 = dh.sum(0)
            t += 1
            for prm, g, mv, vv in (
                (W1, gW1, [mW1], [vW1]), (b1, gb1, [mb1], [vb1]),
                (W2, gW2, [mW2], [vW2]), (b2, gb2, [mb2], [vb2])):
                mv[0][:] = b1_ * mv[0] + (1 - b1_) * g
                vv[0][:] = b2_ * vv[0] + (1 - b2_) * (g * g)
                mhat = mv[0] / (1 - b1_ ** t)
                vhat = vv[0] / (1 - b2_ ** t)
                prm -= lr * mhat / (np.sqrt(vhat) + eps)
        acc = val_acc()
        if acc > best_acc + 1e-4:
            best_acc, best, bad = acc, (W1.copy(), b1.copy(), W2.copy(), b2.copy()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best is not None:
        W1, b1, W2, b2 = best
    h = np.maximum(0.0, Xte @ W1 + b1)
    pred = (h @ W2 + b2).argmax(1)
    return float(np.mean([classes[pi] == t for pi, t in zip(pred, yte)]))


# ---------------------------------------------------------------------------
# Combined sweep: per s, best over {ridge, mlp} x layers; tracks the winning spec.
# ---------------------------------------------------------------------------
def class_readability_curve(
    feats_by_layer_s: dict,   # (layer, s) -> {"train": (X,y), "eval": (X,y)}
    layers: list[int],
    s_grid: list[float],
    *, lam: float = 1.0, mlp_seed: int = 0,
) -> dict:
    """For each s: linear-ridge and MLP eval accuracy at every layer, plus the best
    (probe x layer) accuracy and the winning spec. eval labels at each s are identical
    across layers (same gids), so chance is computed once per s.
    """
    out = {
        "per_s": {},          # s -> {"ridge": {L:acc}, "mlp": {L:acc}}
        "best_acc": {},       # s -> best over probe x layer
        "best_spec": {},      # s -> {"probe","layer","acc"}
        "chance": {},         # s -> eval majority prior
        "ridge_best": {},     # s -> best ridge over layers (for reference)
        "mlp_best": {},       # s -> best mlp over layers (for reference)
    }
    for s in s_grid:
        ridge_accs, mlp_accs = {}, {}
        chance_s = float("nan")
        for L in layers:
            d = feats_by_layer_s.get((L, s))
            if d is None:
                continue
            Xtr, ytr = d["train"]
            Xte, yte = d["eval"]
            if not np.isfinite(chance_s):
                chance_s = eval_majority_prior(yte)
            ridge_accs[L] = probe_accuracy(Xtr, ytr, Xte, yte, lam)
            mlp_accs[L] = mlp_probe_accuracy(Xtr, ytr, Xte, yte, seed=mlp_seed)
        out["per_s"][s] = {"ridge": ridge_accs, "mlp": mlp_accs}
        out["chance"][s] = chance_s

        def _best(accs):
            fin = {L: a for L, a in accs.items() if np.isfinite(a)}
            return (max(fin, key=fin.get), fin[max(fin, key=fin.get)]) if fin else (None, float("nan"))

        rL, rA = _best(ridge_accs)
        mL, mA = _best(mlp_accs)
        out["ridge_best"][s] = rA
        out["mlp_best"][s] = mA
        cands = [("ridge", rL, rA), ("mlp", mL, mA)]
        cands = [c for c in cands if np.isfinite(c[2])]
        if cands:
            probe, L, a = max(cands, key=lambda c: c[2])
            out["best_acc"][s] = float(a)
            out["best_spec"][s] = {"probe": probe, "layer": (None if L is None else int(L)),
                                   "acc": float(a)}
        else:
            out["best_acc"][s] = float("nan")
            out["best_spec"][s] = {"probe": None, "layer": None, "acc": float("nan")}
    return out


def s_read_internal_class(best_acc_by_s: dict, chance_by_s: dict,
                          theta_read: float = 0.70, margin: float = 0.15) -> float:
    """Earliest s with best (probe x layer) eval acc >= theta_read AND >= chance+margin.

    Pre-reg §B1 decision quantity. NaN if never reached. Returns the float s value so the
    driver can compare against 0.45.
    """
    for s in sorted(best_acc_by_s):
        a = best_acc_by_s[s]
        c = chance_by_s.get(s, float("nan"))
        if np.isfinite(a) and np.isfinite(c) and a >= theta_read and a >= c + margin:
            return float(s)
    return float("nan")
