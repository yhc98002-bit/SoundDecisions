"""Float32 probe math for the frozen Arc-4 B-1 V2 protocol."""

from __future__ import annotations

import hashlib
from copy import deepcopy

import numpy as np
from scipy.linalg import solve


def inner_clip_split(train_clips: set[str]) -> tuple[set[str], set[str]]:
    """Frozen 100/26 fit/validation split from B1_PROTOCOL_V2.md."""
    ordered = sorted(
        train_clips,
        key=lambda clip: hashlib.sha256(
            f"arc4-b1-inner-v1:{clip}".encode("utf-8")
        ).hexdigest(),
    )
    if len(ordered) != 126:
        raise ValueError(f"expected 126 B1 outer-training clips, got {len(ordered)}")
    validation = set(ordered[:26])
    return set(ordered[26:]), validation


def _standardize_fit(X_fit: np.ndarray, *others: np.ndarray):
    """Fit a standardizer without promoting cached float32 features."""
    X_fit = np.asarray(X_fit, dtype=np.float32)
    mean = X_fit.mean(axis=0, dtype=np.float32)
    scale = X_fit.std(axis=0, dtype=np.float32)
    scale = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float32)
    return (
        (X_fit - mean) / scale,
        *[
            (np.asarray(array, dtype=np.float32) - mean) / scale
            for array in others
        ],
    )


def balanced_accuracy(
    predictions: np.ndarray,
    targets: list[str] | np.ndarray,
    *,
    class_universe: list[str] | tuple[str, ...] | None = None,
) -> float:
    """Unweighted recall over an explicit true-class universe."""
    predicted = np.asarray(predictions, dtype=object)
    true = np.asarray(targets, dtype=object)
    classes = (
        list(class_universe)
        if class_universe is not None
        else sorted(set(true.tolist()))
    )
    if not classes:
        raise ValueError("balanced accuracy received no classes")
    recalls = []
    for label in classes:
        mask = true == label
        if not np.any(mask):
            return float("nan")
        recalls.append(float(np.mean(predicted[mask] == true[mask])))
    return float(np.mean(recalls))


def classification_metrics(
    predictions: np.ndarray,
    targets: list[str] | np.ndarray,
    *,
    class_universe: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Raw/balanced accuracy and the explicit majority margin."""
    predicted = np.asarray(predictions, dtype=object)
    true = np.asarray(targets, dtype=object)
    if len(true) == 0 or predicted.shape != true.shape:
        raise ValueError("classification metrics require aligned non-empty rows")
    counts = {
        label: int(np.sum(true == label)) for label in sorted(set(true.tolist()))
    }
    accuracy = float(np.mean(predicted == true))
    majority = max(counts.values()) / len(true)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy(
            predicted, true, class_universe=class_universe
        ),
        "majority_baseline": float(majority),
        "margin_over_majority": float(accuracy - majority),
        "class_counts": counts,
    }


def ridge_predict(
    X_train: np.ndarray,
    y_train: list[str],
    X_eval: np.ndarray,
    *,
    lam: float = 1.0,
) -> np.ndarray:
    """Frozen one-vs-all ridge with float32 standardization and solve."""
    X_train, X_eval = _standardize_fit(X_train, X_eval)
    classes = sorted(set(y_train))
    if not classes:
        raise ValueError("ridge received no training labels")
    class_index = {label: index for index, label in enumerate(classes)}
    targets = np.zeros((len(y_train), len(classes)), dtype=np.float32)
    targets[np.arange(len(y_train)), [class_index[label] for label in y_train]] = 1.0
    train_aug = np.concatenate(
        [X_train, np.ones((X_train.shape[0], 1), dtype=np.float32)], axis=1
    )
    eval_aug = np.concatenate(
        [X_eval, np.ones((X_eval.shape[0], 1), dtype=np.float32)], axis=1
    )
    gram = train_aug.T @ train_aug
    gram.flat[:: gram.shape[0] + 1] += np.float32(lam)
    rhs = train_aug.T @ targets
    weights = solve(gram, rhs, assume_a="pos", check_finite=True)
    return np.asarray(
        [classes[index] for index in np.argmax(eval_aug @ weights, axis=1)]
    )


def _new_mlp(seed: int):
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(
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


def _class_codes(labels: list[str], classes: list[str]) -> np.ndarray:
    class_index = {label: index for index, label in enumerate(classes)}
    try:
        return np.asarray(
            [class_index[label] for label in labels], dtype=np.int64
        )
    except KeyError as exc:
        raise ValueError(
            f"label outside frozen outer-training classes: {exc.args[0]}"
        ) from exc


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
    """Fit a candidate MLP with balanced-accuracy early stopping."""
    X_fit, X_validation, X_eval = _standardize_fit(
        X_fit, X_validation, X_eval
    )
    classes = sorted(set(outer_classes))
    y_fit_codes = _class_codes(y_fit, classes)
    validation_targets = np.asarray(y_validation, dtype=object)
    validation_classes = sorted(set(validation_targets.tolist()))

    model = _new_mlp(seed)
    best_balanced = -1.0
    best_accuracy = -1.0
    best_epoch = 0
    best_parameters = None
    stale = 0
    for epoch in range(1, 301):
        if epoch == 1:
            model.partial_fit(X_fit, y_fit_codes, classes=np.arange(len(classes)))
        else:
            model.partial_fit(X_fit, y_fit_codes)
        val_codes = model.predict(X_validation)
        val_predictions = np.asarray([classes[int(index)] for index in val_codes])
        validation_accuracy = float(np.mean(val_predictions == validation_targets))
        validation_balanced = balanced_accuracy(
            val_predictions,
            validation_targets,
            class_universe=validation_classes,
        )
        if validation_balanced > best_balanced + 1e-4:
            best_balanced = validation_balanced
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
        "inner_validation_balanced_accuracy": best_balanced,
        "epochs_run": epoch,
    }


def mlp_predict_fixed_epochs(
    X_train: np.ndarray,
    y_train: list[str],
    X_eval: np.ndarray,
    *,
    outer_classes: list[str],
    epochs: int,
    seed: int = 0,
) -> np.ndarray:
    """Refit the inner-selected MLP on all outer-training rows."""
    if not 1 <= int(epochs) <= 300:
        raise ValueError(f"invalid frozen MLP epoch count: {epochs}")
    X_train, X_eval = _standardize_fit(X_train, X_eval)
    classes = sorted(set(outer_classes))
    y_codes = _class_codes(y_train, classes)
    model = _new_mlp(seed)
    for epoch in range(1, int(epochs) + 1):
        if epoch == 1:
            model.partial_fit(X_train, y_codes, classes=np.arange(len(classes)))
        else:
            model.partial_fit(X_train, y_codes)
    predicted_codes = model.predict(X_eval)
    return np.asarray([classes[int(index)] for index in predicted_codes])


def accuracy_and_video_ci(
    predictions: np.ndarray,
    targets: list[str],
    eval_clip_ids: list[str],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Trajectory metrics with a clip bootstrap carrying every sampled row."""
    predicted = np.asarray(predictions, dtype=object)
    true = np.asarray(targets, dtype=object)
    if (
        len(true) == 0
        or predicted.shape != true.shape
        or len(eval_clip_ids) != len(true)
    ):
        raise ValueError("video bootstrap requires aligned non-empty rows")
    unique_clips = sorted(set(eval_clip_ids))
    clip_ids = np.asarray(eval_clip_ids, dtype=object)
    class_universe = sorted(set(true.tolist()))
    by_clip = {clip: np.flatnonzero(clip_ids == clip) for clip in unique_clips}
    rng = np.random.default_rng(seed)
    raw_draws = []
    balanced_draws = []
    attempted = 0
    max_attempts = max(1000, int(n_boot) * 1000)
    while len(raw_draws) < int(n_boot) and attempted < max_attempts:
        attempted += 1
        sampled = rng.integers(0, len(unique_clips), size=len(unique_clips))
        row_indices = np.concatenate(
            [by_clip[unique_clips[index]] for index in sampled]
        )
        draw_true = true[row_indices]
        if any(not np.any(draw_true == label) for label in class_universe):
            continue
        draw = classification_metrics(
            predicted[row_indices],
            draw_true,
            class_universe=class_universe,
        )
        raw_draws.append(draw["accuracy"])
        balanced_draws.append(draw["balanced_accuracy"])
    if len(raw_draws) != int(n_boot):
        raise RuntimeError(
            f"only {len(raw_draws)}/{n_boot} valid bootstrap draws "
            f"after {attempted} attempts"
        )
    point = classification_metrics(
        predicted, true, class_universe=class_universe
    )
    raw = np.asarray(raw_draws, dtype=np.float64)
    balanced = np.asarray(balanced_draws, dtype=np.float64)
    return {
        **point,
        "ci_lo": float(np.percentile(raw, 2.5)),
        "ci_hi": float(np.percentile(raw, 97.5)),
        "bal_ci_lo": float(np.percentile(balanced, 2.5)),
        "bal_ci_hi": float(np.percentile(balanced, 97.5)),
        "n_eval_trajectories": int(len(targets)),
        "n_eval_clips": int(len(unique_clips)),
        "bootstrap": {
            "seed": int(seed),
            "requested": int(n_boot),
            "attempted": attempted,
            "valid": len(raw_draws),
            "discarded_missing_class": attempted - len(raw_draws),
            "class_universe": class_universe,
        },
    }


def select_spec(specs: list[dict]) -> dict:
    """V2 inner-only order: balanced, raw, family, probe, layer."""
    family_rank = {"pooled": 0, "token_mean_max": 1, "xattn_clip": 2}
    probe_rank = {"ridge": 0, "mlp": 1}
    return min(
        specs,
        key=lambda spec: (
            -float(spec["inner_validation_balanced_accuracy"]),
            -float(spec["inner_validation_accuracy"]),
            family_rank[spec["family"]],
            probe_rank[spec["probe"]],
            int(spec["layer"]),
        ),
    )
