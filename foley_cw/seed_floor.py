"""Arc-3 Tier-B §B3 — seed-floor direct test (well-powered, reduced-dim).

Pre-registered question (frozen `arc3_tierB_preregistration.md` §B3): does the INITIAL
noise seed predict the final class above chance (a "seed floor"), and does that grip grow
with cfg? The Arc-2 dial was underpowered (5000-dim noise, ~230 samples on one closed-form
linear probe). This module fixes that without deviating from the frozen plan:

  * Reduce dim: project the s=0 noise latent to **d=256 via a FIXED Gaussian random
    projection** (seed 0, frozen; Johnson–Lindenstrauss). The projection is regenerated
    deterministically here so the same matrix is used for every cfg and for the full-pool
    GPU collect — JL preserves the linear separability the probe reads while removing the
    sample/dim imbalance.
  * Probe families: **ridge** (the project's closed-form numpy linear probe, reused from
    `internal_probes.probe_accuracy`) and an **MLP** (1 hidden layer, sklearn). Both trained
    ONLY on the frozen `probe_train` clips and evaluated ONLY on the frozen `eval` clips
    (data-leakage discipline: a clip's own independents never appear in both).
  * Metric: held-out class accuracy vs chance (= eval majority-class prior, reported
    alongside). `abstain` labels are dropped from the confident class subset, matching the
    project's categorical convention (`confident_agreement` / ABSTAIN).
  * Trend: OLS slope of (best-probe eval accuracy − chance) vs cfg, with a clip-level
    bootstrap CI (bootstrap unit = video, per the pre-registration).

Decision rule (frozen §B3 — NEVER pauses, emits a SUGGESTED token only; the PI holds the
binding call):
  - best-probe acc > chance + 0.10 at cfg=1.0  → SEED_FLOOR_CONFIRMED (seed floor exists;
    does NOT by itself resurrect F-1).
  - slope across cfg ≈ 0 (bootstrap CI includes 0) or < 0 → F-1 stays refuted (grip flat /
    shrinking).
  - slope > 0 beyond the CI (CI strictly above 0) → SEED_AMPLIFICATION (revise narrative to
    "mixed entropy-reduction + seed-amplification").

numpy + sklearn only; no GPU. This module is non-gating diagnostic code: it produces the
suggested token and the numbers, it does not write any frozen quantity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .internal_probes import probe_accuracy

ABSTAIN = "abstain"

# Frozen §B3 constants.
RP_DIM = 256                 # JL target dimension
RP_SEED = 0                  # frozen Gaussian random-projection seed
DIAL_CFGS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.5)
SEED_FLOOR_MARGIN = 0.10     # acc > chance + 0.10 at cfg=1.0 → SEED_FLOOR_CONFIRMED
CFG1 = 1.0


# ---------------------------------------------------------------------------
# Fixed Gaussian random projection (JL); regenerated deterministically.
# ---------------------------------------------------------------------------
def gaussian_projection(d_in: int, d_out: int = RP_DIM, seed: int = RP_SEED) -> np.ndarray:
    """Frozen Gaussian random-projection matrix R of shape (d_in, d_out).

    Entries ~ N(0, 1/d_out) so that E[‖R x‖²] = ‖x‖² (JL-style norm preservation). Seed is
    frozen; the SAME matrix must be used for every cfg AND for the full-pool GPU collect, so
    the projection is a deterministic function of (d_in, d_out, seed) only.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, d_in, d_out]))
    return rng.normal(0.0, 1.0 / np.sqrt(d_out), size=(d_in, d_out)).astype(np.float64)


def project(noise: np.ndarray, R: np.ndarray | None = None,
            d_out: int = RP_DIM, seed: int = RP_SEED) -> np.ndarray:
    """Project (n, d_in) noise to (n, d_out) with the frozen JL matrix."""
    X = np.asarray(noise, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if R is None:
        R = gaussian_projection(X.shape[1], d_out, seed)
    return X @ R


# ---------------------------------------------------------------------------
# Confident-subset masking (drop abstains, matching the project's class convention).
# ---------------------------------------------------------------------------
def _confident_mask(labels) -> np.ndarray:
    return np.array([str(l) != ABSTAIN for l in labels], dtype=bool)


def chance_accuracy(y_eval) -> float:
    """Eval majority-class prior over the (confident) eval labels."""
    ys = [str(y) for y in y_eval]
    if not ys:
        return float("nan")
    vals, counts = np.unique(np.array(ys, dtype=object), return_counts=True)
    return float(counts.max() / counts.sum())


# ---------------------------------------------------------------------------
# MLP probe (sklearn, 1 hidden layer) — the nonlinear family for §B3.
# ---------------------------------------------------------------------------
def mlp_probe_accuracy(X_tr, y_tr, X_te, y_te,
                       hidden: int = 256, alpha: float = 1e-3,
                       max_iter: int = 300, seed: int = 0) -> float:
    """Train a 1-hidden-layer MLP on (X_tr,y_tr), return eval accuracy on (X_te,y_te).

    Mirrors B1's MLP family (width 256, ReLU, weight-decay 1e-3, early-stopped on a
    train-internal split). Standardizes features with TRAIN statistics only (no eval
    leakage). Falls back to the train-majority prior if the train set is single-class.
    """
    y_tr = [str(y) for y in y_tr]
    y_te = [str(y) for y in y_te]
    if X_te.shape[0] == 0:
        return float("nan")
    if len(set(y_tr)) < 2 or X_tr.shape[0] < 4:
        maj = max(set(y_tr), key=y_tr.count) if y_tr else None
        return float(np.mean([1.0 if t == maj else 0.0 for t in y_te]))
    mu = X_tr.mean(axis=0); sd = X_tr.std(axis=0); sd = np.where(sd == 0, 1.0, sd)
    Xtr = (X_tr - mu) / sd
    Xte = (X_te - mu) / sd
    pred = _fit_predict_mlp(Xtr, y_tr, Xte, hidden, alpha, max_iter, seed)
    if pred is None:
        return float("nan")
    return float(np.mean([1.0 if str(p) == str(t) else 0.0 for p, t in zip(pred, y_te)]))


def _fit_predict_mlp(Xtr, y_tr, Xte, hidden, alpha, max_iter, seed):
    """Fit a 1-hidden-layer MLP on standardized features and return string predictions.

    Labels are mapped to INTEGER codes before fitting: sklearn 1.7.x's early-stopping
    internal scoring calls np.isnan on the predicted labels, which raises on string/object
    targets — so we encode here and decode predictions back to the original label strings.
    """
    try:
        from sklearn.neural_network import MLPClassifier
    except Exception:                                       # pragma: no cover - sklearn present
        return None
    classes = sorted(set(y_tr), key=str)
    cidx = {c: i for i, c in enumerate(classes)}
    y_int = np.array([cidx[str(y)] for y in y_tr])
    # early_stopping needs ≥2 per class in the internal validation split; guard small n.
    es = bool(np.bincount(y_int).min() >= 2 and len(y_int) >= 12)
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), activation="relu", alpha=alpha,
                        max_iter=max_iter, early_stopping=es, n_iter_no_change=15,
                        random_state=seed)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(Xtr, y_int)
        out = clf.predict(Xte)
    return [classes[int(i)] for i in out]


# ---------------------------------------------------------------------------
# Per-cfg probing on the frozen split.
# ---------------------------------------------------------------------------
@dataclass
class CfgResult:
    cfg: float
    n_train: int
    n_eval: int
    chance: float
    ridge_acc: float
    mlp_acc: float
    best_acc: float
    best_probe: str
    # bookkeeping for the clip-level bootstrap (eval clip id per eval row, + correctness)
    eval_clip_ids: list = field(default_factory=list)
    ridge_correct: list = field(default_factory=list)
    mlp_correct: list = field(default_factory=list)
    _eval_true: list = field(default_factory=list)


def _ridge_predictions(X_tr, y_tr, X_te, lam: float = 1.0):
    """Ridge per-row predictions on eval (reuses internal_probes' ridge via a thin call)."""
    from .internal_probes import predict, ridge_linear_classifier
    y_tr = [str(y) for y in y_tr]
    if len(set(y_tr)) < 2 or X_tr.shape[0] < 2 or X_te.shape[0] == 0:
        maj = max(set(y_tr), key=y_tr.count) if y_tr else None
        return [maj] * X_te.shape[0]
    mu = X_tr.mean(axis=0); sd = X_tr.std(axis=0); sd = np.where(sd == 0, 1.0, sd)
    W, classes = ridge_linear_classifier((X_tr - mu) / sd, y_tr, lam)
    return predict(W, classes, (X_te - mu) / sd)


def _mlp_predictions(X_tr, y_tr, X_te, hidden=256, alpha=1e-3, max_iter=300, seed=0):
    y_tr = [str(y) for y in y_tr]
    if X_te.shape[0] == 0:
        return []
    if len(set(y_tr)) < 2 or X_tr.shape[0] < 4:
        maj = max(set(y_tr), key=y_tr.count) if y_tr else None
        return [maj] * X_te.shape[0]
    mu = X_tr.mean(axis=0); sd = X_tr.std(axis=0); sd = np.where(sd == 0, 1.0, sd)
    pred = _fit_predict_mlp((X_tr - mu) / sd, y_tr, (X_te - mu) / sd,
                            hidden, alpha, max_iter, seed)
    return pred if pred is not None else [None] * X_te.shape[0]


def probe_cfg(noise_by_clip: dict, labels_by_clip: dict, train_clips, eval_clips,
              R: np.ndarray | None = None, lam: float = 1.0, mlp_seed: int = 0,
              cfg: float = float("nan")) -> CfgResult:
    """Fit ridge + MLP class-from-noise probes for ONE cfg on the frozen clip split.

    noise_by_clip[clip] : (n_indep, d_in) s=0 noise rows; labels_by_clip[clip] : (n_indep,)
    final-class labels. Trains on `train_clips`, evaluates on `eval_clips`; abstain rows are
    dropped from both (confident class subset). Per-eval-row clip id + correctness are kept
    so the §B3 bootstrap can resample by video.
    """
    def _stack(clips):
        Xs, ys, cids = [], [], []
        for c in clips:
            if c not in noise_by_clip:
                continue
            noise = np.asarray(noise_by_clip[c]); labs = list(labels_by_clip[c])
            m = _confident_mask(labs)
            if not m.any():
                continue
            Xp = project(noise[m], R)
            Xs.append(Xp); ys += [str(l) for l, keep in zip(labs, m) if keep]
            cids += [c] * int(m.sum())
        if not Xs:
            return np.zeros((0, RP_DIM)), [], []
        return np.concatenate(Xs, axis=0), ys, cids

    Xtr, ytr, _ = _stack(train_clips)
    Xte, yte, cte = _stack(eval_clips)

    chance = chance_accuracy(yte)
    ridge_acc = probe_accuracy(Xtr, ytr, Xte, yte, lam=lam)
    mlp_acc = mlp_probe_accuracy(Xtr, ytr, Xte, yte, seed=mlp_seed)

    ridge_pred = _ridge_predictions(Xtr, ytr, Xte, lam=lam)
    mlp_pred = _mlp_predictions(Xtr, ytr, Xte, seed=mlp_seed)
    ridge_correct = [1 if str(p) == str(t) else 0 for p, t in zip(ridge_pred, yte)]
    mlp_correct = [1 if str(p) == str(t) else 0 for p, t in zip(mlp_pred, yte)]

    cand = [("ridge", ridge_acc), ("mlp", mlp_acc)]
    finite = [(n, a) for n, a in cand if np.isfinite(a)]
    if finite:
        best_probe, best_acc = max(finite, key=lambda t: t[1])
    else:
        best_probe, best_acc = "ridge", float("nan")
    res = CfgResult(cfg=cfg, n_train=len(ytr), n_eval=len(yte), chance=chance,
                    ridge_acc=ridge_acc, mlp_acc=mlp_acc, best_acc=best_acc,
                    best_probe=best_probe, eval_clip_ids=cte,
                    ridge_correct=ridge_correct, mlp_correct=mlp_correct)
    # Stash the per-eval-row TRUE labels so the clip-level bootstrap can recompute chance
    # on each resampled draw (bootstrap unit = video).
    res._eval_true = [str(y) for y in yte]
    return res


# ---------------------------------------------------------------------------
# Trend: OLS slope of (best-probe acc − chance) vs cfg, clip-level bootstrap CI.
# ---------------------------------------------------------------------------
def ols_slope(x, y) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2 or np.allclose(x, x[0]):
        return float("nan")
    xc = x - x.mean()
    denom = float(xc @ xc)
    return float(xc @ (y - y.mean()) / denom) if denom > 0 else float("nan")


def _acc_minus_chance_per_cfg(results: dict, probe: str = "best") -> dict:
    out = {}
    for cfg, r in results.items():
        a = {"best": r.best_acc, "ridge": r.ridge_acc, "mlp": r.mlp_acc}[probe]
        if np.isfinite(a) and np.isfinite(r.chance):
            out[float(cfg)] = float(a - r.chance)
    return out


def bootstrap_slope_ci(results: dict, n_boot: int = 2000, seed: int = 0,
                       probe: str = "best", ci: float = 0.95) -> dict:
    """Bootstrap the OLS slope of (acc − chance) vs cfg by RESAMPLING EVAL CLIPS (video).

    For each bootstrap draw, eval clips are resampled with replacement (one shared draw
    applied at every cfg, since the dial reuses the same clip pool across cfg). The per-cfg
    accuracy is recomputed on the resampled eval rows; chance is recomputed on the same
    resampled eval labels; the slope is the OLS fit of (acc − chance) vs cfg. Bootstrap
    unit = video, per the frozen pre-registration.
    """
    cfgs = sorted(results)
    # Build per-cfg, per-clip eval correctness + labels, keyed by clip id.
    eval_clip_set = set()
    for cfg in cfgs:
        eval_clip_set.update(results[cfg].eval_clip_ids)
    eval_clips = sorted(eval_clip_set)
    if len(eval_clips) < 2:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_boot": 0, "eval_clips": len(eval_clips)}

    # index: per cfg → clip → (list of correct[0/1] for chosen probe, list of labels)
    probe_for = {cfg: (results[cfg].best_probe if probe == "best" else probe) for cfg in cfgs}
    per = {}
    for cfg in cfgs:
        r = results[cfg]
        correct = r.ridge_correct if probe_for[cfg] == "ridge" else r.mlp_correct
        d = {}
        for cid, cor, lab in zip(r.eval_clip_ids, correct, _eval_labels(r)):
            d.setdefault(cid, {"cor": [], "lab": []})
            d[cid]["cor"].append(cor); d[cid]["lab"].append(lab)
        per[cfg] = d

    point = ols_slope(cfgs, [_acc_minus_chance_per_cfg(results, probe).get(c, np.nan)
                             for c in cfgs])
    rng = np.random.default_rng(seed)
    slopes = []
    n = len(eval_clips)
    for _ in range(n_boot):
        draw = [eval_clips[i] for i in rng.integers(0, n, n)]
        xs, ys = [], []
        for cfg in cfgs:
            cors, labs = [], []
            for cid in draw:
                blk = per[cfg].get(cid)
                if blk:
                    cors += blk["cor"]; labs += blk["lab"]
            if not labs:
                continue
            acc = float(np.mean(cors))
            ch = chance_accuracy(labs)
            if np.isfinite(acc) and np.isfinite(ch):
                xs.append(cfg); ys.append(acc - ch)
        s = ols_slope(xs, ys)
        if np.isfinite(s):
            slopes.append(s)
    if not slopes:
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "n_boot": 0, "eval_clips": n}
    lo = float(np.percentile(slopes, 100 * (1 - ci) / 2))
    hi = float(np.percentile(slopes, 100 * (1 + ci) / 2))
    return {"point": float(point), "lo": lo, "hi": hi, "n_boot": len(slopes),
            "eval_clips": n}


def _eval_labels(r: CfgResult) -> list:
    """Per-eval-row TRUE labels, stashed on the result by `probe_cfg`, so the clip-level
    bootstrap can recompute the majority-class chance baseline on each resampled draw."""
    return getattr(r, "_eval_true", [None] * len(r.eval_clip_ids))


# ---------------------------------------------------------------------------
# Decision rule (frozen §B3): suggested token + rationale.
# ---------------------------------------------------------------------------
def decide(results: dict, slope_ci: dict, margin: float = SEED_FLOOR_MARGIN) -> dict:
    r1 = results.get(CFG1)
    seed_floor = bool(r1 is not None and np.isfinite(r1.best_acc) and np.isfinite(r1.chance)
                      and (r1.best_acc - r1.chance) > margin)
    lo, hi = slope_ci.get("lo"), slope_ci.get("hi")
    slope_positive = bool(lo is not None and np.isfinite(lo) and lo > 0.0)      # CI strictly > 0
    slope_flat_or_neg = bool(hi is not None and np.isfinite(hi)
                             and (lo is None or not np.isfinite(lo) or lo <= 0.0))

    tokens = []
    if seed_floor:
        tokens.append("SEED_FLOOR_CONFIRMED")
    if slope_positive:
        tokens.append("SEED_AMPLIFICATION")
        f1_status = "F-1 stays refuted (seed floor exists; amplification suggested, not F-1)"
    elif slope_flat_or_neg:
        f1_status = "F-1 stays refuted (slope flat or shrinking; CI includes/below 0)"
    else:
        f1_status = "F-1 stays refuted (slope inconclusive)"

    suggested = "+".join(tokens) if tokens else "NO_SEED_FLOOR"
    rationale = []
    if r1 is not None:
        rationale.append(
            f"cfg=1.0 best={r1.best_acc:.3f} ({r1.best_probe}) chance={r1.chance:.3f} "
            f"margin={r1.best_acc - r1.chance:+.3f} vs +{margin:.2f} → "
            f"{'seed floor' if seed_floor else 'no seed floor'}")
    rationale.append(
        f"slope(acc−chance vs cfg)={slope_ci.get('point', float('nan')):+.4f} "
        f"CI=[{lo if lo is None else round(lo,4)},{hi if hi is None else round(hi,4)}] → "
        f"{'CI>0 amplification' if slope_positive else 'CI includes/below 0'}")
    return {"suggested_token": suggested, "seed_floor_confirmed": seed_floor,
            "slope_positive": slope_positive, "f1_status": f1_status,
            "rationale": "; ".join(rationale)}
