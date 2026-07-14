"""Numerically stable, CPU-only probe math for the frozen Arc-4 B1 protocol."""

from __future__ import annotations

import hashlib
from copy import deepcopy

import numpy as np
from scipy.linalg import solve


def inner_clip_split(train_clips: set[str]) -> tuple[set[str], set[str]]:
    """Frozen 100/26 fit/validation split from B1_PROTOCOL.md."""
    ordered = sorted(
        train_clips,
        key=lambda clip: hashlib.sha256(
            f"arc4-b1-inner-v1:{clip}".encode("utf-8")).hexdigest(),
    )
    if len(ordered) != 126:
        raise ValueError(f"expected 126 B1 outer-training clips, got {len(ordered)}")
    validation = set(ordered[:26])
    return set(ordered[26:]), validation


def _standardize_fit(X_fit: np.ndarray, *others: np.ndarray):
    X_fit = np.asarray(X_fit, dtype=np.float64)
    mean = X_fit.mean(axis=0)
    scale = X_fit.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return ((X_fit - mean) / scale,
            *[(np.asarray(array, dtype=np.float64) - mean) / scale for array in others])


def ridge_predict(
    X_train: np.ndarray,
    y_train: list[str],
    X_eval: np.ndarray,
    *,
    lam: float = 1.0,
) -> np.ndarray:
    """Frozen one-vs-all ridge, solved in float64 as a positive-definite system."""
    X_train, X_eval = _standardize_fit(X_train, X_eval)
    classes = sorted(set(y_train))
    if not classes:
        raise ValueError("ridge received no training labels")
    class_index = {label: index for index, label in enumerate(classes)}
    targets = np.zeros((len(y_train), len(classes)), dtype=np.float64)
    targets[np.arange(len(y_train)), [class_index[label] for label in y_train]] = 1.0
    train_aug = np.concatenate([X_train, np.ones((X_train.shape[0], 1))], axis=1)
    eval_aug = np.concatenate([X_eval, np.ones((X_eval.shape[0], 1))], axis=1)
    gram = train_aug.T @ train_aug
    gram.flat[::gram.shape[0] + 1] += float(lam)
    weights = solve(gram, train_aug.T @ targets, assume_a="pos", check_finite=True)
    return np.asarray([classes[index] for index in np.argmax(eval_aug @ weights, axis=1)])


def mlp_predict(
    X_fit: np.ndarray,
    y_fit: list[str],
    X_validation: np.ndarray,
    y_validation: list[str],
    X_eval: np.ndarray,
    *,
    outer_classes: list[str],
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Frozen one-hidden-layer MLP with explicit clip-grouped early stopping."""
    from sklearn.neural_network import MLPClassifier

    X_fit, X_validation, X_eval = _standardize_fit(X_fit, X_validation, X_eval)
    classes = sorted(set(outer_classes))
    class_index = {label: index for index, label in enumerate(classes)}
    y_fit_codes = np.asarray([class_index[label] for label in y_fit], dtype=int)
    y_val_codes = np.asarray([class_index.get(label, -1) for label in y_validation], dtype=int)

    model = MLPClassifier(
        hidden_layer_sizes=(256,),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=1,
        shuffle=True,
        random_state=seed,
        warm_start=True,
    )
    best_accuracy = -1.0
    best_epoch = 0
    best_parameters = None
    stale = 0
    for epoch in range(1, 301):
        if epoch == 1:
            model.partial_fit(X_fit, y_fit_codes, classes=np.arange(len(classes)))
        else:
            model.partial_fit(X_fit, y_fit_codes)
        validation_accuracy = float(np.mean(model.predict(X_validation) == y_val_codes))
        if validation_accuracy > best_accuracy + 1e-4:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            best_parameters = (deepcopy(model.coefs_), deepcopy(model.intercepts_))
            stale = 0
        else:
            stale += 1
            if stale >= 15:
                break
    if best_parameters is None:
        raise RuntimeError("MLP early stopping did not retain a checkpoint")
    model.coefs_, model.intercepts_ = best_parameters
    predicted_codes = model.predict(X_eval)
    predictions = np.asarray([classes[int(index)] for index in predicted_codes])
    return predictions, {
        "best_epoch": best_epoch,
        "inner_validation_accuracy": best_accuracy,
        "epochs_run": epoch,
    }


def accuracy_and_video_ci(
    predictions: np.ndarray,
    targets: list[str],
    eval_clip_ids: list[str],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    correct = np.asarray(predictions) == np.asarray(targets)
    unique_clips = sorted(set(eval_clip_ids))
    clip_ids = np.asarray(eval_clip_ids, dtype=object)
    clip_accuracy = np.asarray([correct[clip_ids == clip].mean() for clip in unique_clips])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique_clips), size=(n_boot, len(unique_clips)))
    bootstrap = clip_accuracy[draws].mean(axis=1)
    return {
        "accuracy": float(correct.mean()),
        "ci_lo": float(np.percentile(bootstrap, 2.5)),
        "ci_hi": float(np.percentile(bootstrap, 97.5)),
        "n_eval_trajectories": int(len(targets)),
        "n_eval_clips": int(len(unique_clips)),
    }


def select_spec(specs: list[dict]) -> dict:
    """Protocol tie order: accuracy, family, ridge before MLP, lower layer."""
    family_rank = {"pooled": 0, "token_mean_max": 1, "xattn_clip": 2}
    probe_rank = {"ridge": 0, "mlp": 1}
    return min(
        specs,
        key=lambda spec: (
            -float(spec["accuracy"]),
            family_rank[spec["family"]],
            probe_rank[spec["probe"]],
            int(spec["layer"]),
        ),
    )
