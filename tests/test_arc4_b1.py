import numpy as np

from foley_cw.arc4_b1 import (
    accuracy_and_video_ci,
    inner_clip_split,
    mlp_predict,
    ridge_predict,
    select_spec,
)


def test_inner_clip_split_is_grouped_and_deterministic():
    clips = {f"clip-{index:03d}" for index in range(126)}
    fit, validation = inner_clip_split(clips)
    assert len(fit) == 100
    assert len(validation) == 26
    assert not fit & validation
    assert fit | validation == clips
    assert (fit, validation) == inner_clip_split(clips)


def test_ridge_predict_and_video_bootstrap():
    X_train = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y_train = ["left", "left", "right", "right"]
    X_eval = np.array([[-3.0], [-0.5], [0.5], [3.0]])
    predictions = ridge_predict(X_train, y_train, X_eval)
    assert predictions.tolist() == ["left", "left", "right", "right"]
    stats = accuracy_and_video_ci(
        predictions,
        ["left", "left", "right", "right"],
        ["a", "a", "b", "b"],
        n_boot=100,
        seed=0,
    )
    assert stats["accuracy"] == 1.0
    assert stats["ci_lo"] == stats["ci_hi"] == 1.0


def test_mlp_uses_explicit_validation_checkpoint():
    rng = np.random.default_rng(0)
    X_fit = np.r_[rng.normal(-1, 0.1, (20, 3)), rng.normal(1, 0.1, (20, 3))]
    y_fit = ["a"] * 20 + ["b"] * 20
    X_val = np.r_[rng.normal(-1, 0.1, (6, 3)), rng.normal(1, 0.1, (6, 3))]
    y_val = ["a"] * 6 + ["b"] * 6
    predictions, info = mlp_predict(
        X_fit, y_fit, X_val, y_val, X_val,
        outer_classes=["a", "b"],
        seed=0,
    )
    assert predictions.shape == (12,)
    assert 1 <= info["best_epoch"] <= info["epochs_run"] <= 300
    assert 0.0 <= info["inner_validation_accuracy"] <= 1.0


def test_select_spec_uses_frozen_tie_order():
    specs = [
        {"accuracy": 0.7, "family": "token_mean_max", "probe": "ridge", "layer": 0},
        {"accuracy": 0.7, "family": "pooled", "probe": "mlp", "layer": 0},
        {"accuracy": 0.7, "family": "pooled", "probe": "ridge", "layer": 2},
        {"accuracy": 0.7, "family": "pooled", "probe": "ridge", "layer": 1},
    ]
    assert select_spec(specs) is specs[-1]
