import json

import numpy as np
import pytest

import foley_cw.arc4_b1 as b1
from foley_cw.arc4_b1 import (
    accuracy_and_video_ci,
    classification_metrics,
    inner_clip_split,
    mlp_predict,
    mlp_predict_fixed_epochs,
    ridge_predict,
    select_spec,
)
from scripts.arc4_b1_probe import (
    AMENDMENT_SHA256,
    JOIN_AMENDMENT_2_SHA256,
    JOIN_AMENDMENT_3_SHA256,
    JOIN_AMENDMENT_4_SHA256,
    JOIN_AMENDMENT_SHA256,
    PROTOCOL_SHA256,
    _filter_confident_rows,
    _join_failure_record,
    _load_inner_journal,
    _load_outer_journal,
    _join_relative_l2,
    _prediction_records,
    _validate_manifest_hash,
    _valid_pertoken_v2,
    _valid_pooled_v2,
)


def test_inner_clip_split_is_grouped_and_deterministic():
    clips = {f"clip-{index:03d}" for index in range(126)}
    fit, validation = inner_clip_split(clips)
    assert len(fit) == 100
    assert len(validation) == 26
    assert not fit & validation
    assert fit | validation == clips
    assert (fit, validation) == inner_clip_split(clips)


def test_v2_bundle_gate_pins_shapes_and_dtypes(tmp_path):
    pertoken = tmp_path / "pertoken.npz"
    np.savez(
        pertoken,
        token_mean=np.zeros((12, 448), dtype=np.float16),
        token_mean_max=np.zeros((12, 896), dtype=np.float16),
        tokens_sub=np.zeros((12, 64, 448), dtype=np.float16),
        xattn_clip=np.zeros((4, 64), dtype=np.float16),
        xattn_frac=np.zeros(4, dtype=np.float32),
    )
    assert _valid_pertoken_v2(pertoken)

    wrong_xattn = tmp_path / "wrong_xattn.npz"
    np.savez(
        wrong_xattn,
        token_mean=np.zeros((12, 448), dtype=np.float16),
        token_mean_max=np.zeros((12, 896), dtype=np.float16),
        tokens_sub=np.zeros((12, 64, 448), dtype=np.float16),
        xattn_clip=np.zeros((4, 63), dtype=np.float16),
        xattn_frac=np.zeros(4, dtype=np.float32),
    )
    assert not _valid_pertoken_v2(wrong_xattn)

    pooled = tmp_path / "pooled.npz"
    np.savez(pooled, pooled=np.zeros((12, 448), dtype=np.float16))
    assert _valid_pooled_v2(pooled)
    wrong_dtype = tmp_path / "wrong_dtype.npz"
    np.savez(wrong_dtype, pooled=np.zeros((12, 448), dtype=np.float32))
    assert not _valid_pooled_v2(wrong_dtype)


def test_frozen_population_manifest_hash_is_mandatory(tmp_path):
    substituted = tmp_path / "manifest.json"
    substituted.write_text("{}\n")
    with pytest.raises(ValueError, match="population manifest hash"):
        _validate_manifest_hash(substituted)


def test_normwise_join_tolerates_isolated_fp16_rounding_not_large_drift():
    pooled = np.ones(1000, dtype=np.float32)
    pooled[0] = np.float32(0.272705078125)
    token_mean = pooled.copy()
    token_mean[0] = np.float32(0.27587890625)
    assert not np.allclose(token_mean, pooled, rtol=2e-3, atol=2e-3)
    assert _join_relative_l2(token_mean, pooled) < 2e-4
    token_mean[:10] += np.float32(0.1)
    assert _join_relative_l2(token_mean, pooled) > 2e-4


def test_join_failure_record_is_explicitly_non_scientific():
    pooled = np.full((2, 2), 2.0, dtype=np.float32)
    token_mean = pooled.copy()
    token_mean[0, 0] += 0.25

    row = _join_failure_record(
        name="clip__p1cfg1_ind0__s0.05.npz",
        token_mean=token_mean,
        pooled=pooled,
        checked_before_failure=17,
    )

    assert row["status"] == "B1_INCOMPLETE"
    assert row["scientific_token"] is None
    assert row["evaluation_started"] is False
    assert row["probe_metrics_computed"] is False
    assert row["prediction_ledgers_produced"] is False
    assert row["labels_loaded_for_set_and_universe_validation"] is True
    assert row["first_failing_bundle_in_sorted_scan"].endswith("s0.05.npz")
    assert row["bundles_checked_before_failure"] == 17
    assert row["observed"]["relative_l2"] == pytest.approx(
        np.linalg.norm(token_mean.astype(np.float64) - pooled.astype(np.float64))
        / np.linalg.norm(pooled.astype(np.float64))
    )


def test_ridge_solve_stays_float32(monkeypatch):
    seen = []
    scipy_solve = b1.solve

    def checked_solve(gram, rhs, **kwargs):
        seen.append((gram.dtype, rhs.dtype))
        return scipy_solve(gram, rhs, **kwargs)

    monkeypatch.setattr(b1, "solve", checked_solve)
    X_train = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float16)
    y_train = ["left", "left", "right", "right"]
    X_eval = np.array([[-3.0], [-0.5], [0.5], [3.0]], dtype=np.float16)
    predictions = ridge_predict(X_train, y_train, X_eval)
    assert predictions.tolist() == ["left", "left", "right", "right"]
    assert seen == [(np.dtype("float32"), np.dtype("float32"))]


def test_metrics_publish_balanced_accuracy_and_majority_margin():
    targets = ["major"] * 8 + ["minor"] * 2
    predictions = ["major"] * 10
    metrics = classification_metrics(np.asarray(predictions), targets)
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["majority_baseline"] == pytest.approx(0.8)
    assert metrics["margin_over_majority"] == pytest.approx(0.0)


def test_video_bootstrap_carries_all_unequal_clip_rows():
    targets = ["x"] * 10
    predictions = np.asarray(["x"] * 9 + ["wrong"])
    clips = ["many"] * 9 + ["one"]
    seed = next(
        seed
        for seed in range(100)
        if set(np.random.default_rng(seed).integers(0, 2, size=2).tolist())
        == {0, 1}
    )
    stats = accuracy_and_video_ci(
        predictions, targets, clips, n_boot=1, seed=seed
    )
    assert stats["accuracy"] == pytest.approx(0.9)
    assert stats["ci_lo"] == pytest.approx(0.9)
    assert stats["ci_hi"] == pytest.approx(0.9)
    assert stats["ci_lo"] != pytest.approx(0.5)


def test_balanced_bootstrap_replaces_missing_class_draws():
    predictions = np.asarray(["a", "b"])
    stats = accuracy_and_video_ci(
        predictions,
        ["a", "b"],
        ["clip-a", "clip-b"],
        n_boot=20,
        seed=0,
    )
    assert stats["bootstrap"]["valid"] == 20
    assert stats["bootstrap"]["attempted"] > 20
    assert stats["bootstrap"]["discarded_missing_class"] > 0
    assert stats["balanced_accuracy"] == 1.0


class _DtypeCheckingMLP:
    def __init__(self):
        self.coefs_ = [np.zeros((1, 1), dtype=np.float32)]
        self.intercepts_ = [np.zeros(1, dtype=np.float32)]

    def partial_fit(self, X, y, classes=None):
        assert X.dtype == np.float32
        return self

    def predict(self, X):
        assert X.dtype == np.float32
        return np.zeros(len(X), dtype=np.int64)


def test_mlp_paths_stay_float32(monkeypatch):
    monkeypatch.setattr(b1, "_new_mlp", lambda _seed: _DtypeCheckingMLP())
    X_fit = np.arange(24, dtype=np.float16).reshape(8, 3)
    y_fit = ["a"] * 4 + ["b"] * 4
    X_val = np.arange(12, dtype=np.float16).reshape(4, 3)
    y_val = ["a", "a", "b", "b"]
    predictions, info = mlp_predict(
        X_fit,
        y_fit,
        X_val,
        y_val,
        X_val,
        outer_classes=["a", "b"],
        seed=0,
    )
    assert predictions.shape == (4,)
    assert info["best_epoch"] == 1
    fixed = mlp_predict_fixed_epochs(
        X_fit,
        y_fit,
        X_val,
        outer_classes=["a", "b"],
        epochs=2,
        seed=0,
    )
    assert fixed.shape == (4,)


def test_inner_mlp_does_not_require_validation_only_class(monkeypatch):
    monkeypatch.setattr(b1, "_new_mlp", lambda _seed: _DtypeCheckingMLP())
    X_fit = np.arange(12, dtype=np.float32).reshape(4, 3)
    X_val = np.arange(6, dtype=np.float32).reshape(2, 3)
    predictions, info = mlp_predict(
        X_fit,
        ["a", "a", "b", "b"],
        X_val,
        ["a", "validation-only"],
        X_val,
        outer_classes=["a", "b"],
        seed=0,
    )
    assert predictions.tolist() == ["a", "a"]
    assert info["inner_validation_balanced_accuracy"] == pytest.approx(0.5)


def test_mlp_uses_explicit_balanced_validation_checkpoint():
    rng = np.random.default_rng(0)
    X_fit = np.r_[rng.normal(-1, 0.1, (20, 3)), rng.normal(1, 0.1, (20, 3))]
    y_fit = ["a"] * 20 + ["b"] * 20
    X_val = np.r_[rng.normal(-1, 0.1, (6, 3)), rng.normal(1, 0.1, (6, 3))]
    y_val = ["a"] * 6 + ["b"] * 6
    predictions, info = mlp_predict(
        X_fit,
        y_fit,
        X_val,
        y_val,
        X_val,
        outer_classes=["a", "b"],
        seed=0,
    )
    assert predictions.shape == (12,)
    assert 1 <= info["best_epoch"] <= info["epochs_run"] <= 300
    assert 0.0 <= info["inner_validation_balanced_accuracy"] <= 1.0


def test_select_spec_is_inner_only_and_ignores_deceptive_outer_fields():
    specs = [
        {
            "inner_validation_balanced_accuracy": 0.70,
            "inner_validation_accuracy": 0.80,
            "outer_accuracy": 0.99,
            "family": "token_mean_max",
            "probe": "ridge",
            "layer": 0,
        },
        {
            "inner_validation_balanced_accuracy": 0.71,
            "inner_validation_accuracy": 0.72,
            "outer_accuracy": 0.01,
            "family": "pooled",
            "probe": "mlp",
            "layer": 0,
        },
    ]
    assert select_spec(specs) is specs[1]
    specs[0]["outer_accuracy"], specs[1]["outer_accuracy"] = 0.0, 1.0
    assert select_spec(specs) is specs[1]
    with pytest.raises(KeyError):
        select_spec(
            [{"outer_accuracy": 1.0, "family": "pooled", "probe": "ridge", "layer": 0}]
        )


def test_selected_spec_is_invariant_to_outer_target_flip():
    candidates = [
        {
            "inner_validation_balanced_accuracy": 0.7,
            "inner_validation_accuracy": 0.7,
            "family": "pooled",
            "probe": "ridge",
            "layer": 0,
        },
        {
            "inner_validation_balanced_accuracy": 0.6,
            "inner_validation_accuracy": 0.9,
            "family": "token_mean_max",
            "probe": "mlp",
            "layer": 1,
        },
    ]

    def choose_then_score(outer_targets):
        selected = select_spec(candidates)
        classification_metrics(np.asarray(["a", "a"]), outer_targets)
        return selected["family"], selected["probe"], selected["layer"]

    assert choose_then_score(["a", "b"]) == choose_then_score(["b", "a"])


def test_select_spec_uses_frozen_tie_order():
    specs = [
        {
            "inner_validation_balanced_accuracy": 0.7,
            "inner_validation_accuracy": 0.7,
            "family": "token_mean_max",
            "probe": "ridge",
            "layer": 0,
        },
        {
            "inner_validation_balanced_accuracy": 0.7,
            "inner_validation_accuracy": 0.7,
            "family": "pooled",
            "probe": "mlp",
            "layer": 0,
        },
        {
            "inner_validation_balanced_accuracy": 0.7,
            "inner_validation_accuracy": 0.7,
            "family": "pooled",
            "probe": "ridge",
            "layer": 2,
        },
        {
            "inner_validation_balanced_accuracy": 0.7,
            "inner_validation_accuracy": 0.7,
            "family": "pooled",
            "probe": "ridge",
            "layer": 1,
        },
    ]
    assert select_spec(specs) is specs[-1]


def test_abstain_filter_and_prediction_ledger_are_explicit():
    rows = [("c1", 0, "g1"), ("c1", 1, "g2"), ("c2", 0, "g3")]
    labels = {"g1": "impact", "g2": "abstain", "g3": "scrape"}
    confident = _filter_confident_rows(rows, labels)
    assert confident == [rows[0], rows[2]]
    mask = np.asarray([True, True])
    targets = np.asarray(["impact", "scrape"], dtype=object)
    ledger = _prediction_records(
        confident,
        mask,
        targets,
        np.asarray(["impact", "impact"]),
        split="outer_eval",
        s=0.05,
        family="pooled",
        probe="ridge",
        layer=0,
    )
    assert {row["gen_id"] for row in ledger} == {"g1", "g3"}
    assert all(row["true_label"] != "abstain" for row in ledger)
    assert len({(row["s"], row["gen_id"]) for row in ledger}) == len(ledger)


def _journal_hashes():
    return {
        "protocol": PROTOCOL_SHA256,
        "bootstrap_amendment": AMENDMENT_SHA256,
        "join_amendment": JOIN_AMENDMENT_SHA256,
        "join_amendment_2": JOIN_AMENDMENT_2_SHA256,
        "join_amendment_3": JOIN_AMENDMENT_3_SHA256,
        "join_amendment_4": JOIN_AMENDMENT_4_SHA256,
        "population_manifest": "m",
        "coarse_class_map": "c",
        "measurements": "labels",
        "feature_corpus": "features",
        "collection_completion": "collection",
    }


def test_inner_resume_rejects_summary_corruption(tmp_path):
    implementation = {"library": "lib", "runner": "run"}
    inputs = _journal_hashes()
    spec = {"s": 0.05, "family": "pooled", "probe": "ridge", "layer": 0}
    predictions = [
        {
            **spec,
            "split": "inner_validation",
            "clip": "c1",
            "gen_id": "g1",
            "true_label": "a",
            "predicted_label": "a",
        },
        {
            **spec,
            "split": "inner_validation",
            "clip": "c2",
            "gen_id": "g2",
            "true_label": "b",
            "predicted_label": "b",
        },
    ]
    metrics = classification_metrics(np.asarray(["a", "b"]), ["a", "b"])
    summary = {
        **spec,
        "inner_validation_accuracy": metrics["accuracy"],
        "inner_validation_balanced_accuracy": metrics["balanced_accuracy"],
        "inner_validation_majority_baseline": metrics["majority_baseline"],
        "inner_validation_margin_over_majority": metrics["margin_over_majority"],
        "inner_validation_class_counts": metrics["class_counts"],
    }
    row = {
        "schema": "arc4_b1_inner_candidate_v2",
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_amendment_sha256": AMENDMENT_SHA256,
        "join_amendment_sha256": JOIN_AMENDMENT_SHA256,
        "join_amendment_2_sha256": JOIN_AMENDMENT_2_SHA256,
        "join_amendment_3_sha256": JOIN_AMENDMENT_3_SHA256,
        "join_amendment_4_sha256": JOIN_AMENDMENT_4_SHA256,
        "implementation_sha256": implementation,
        "input_sha256": inputs,
        "spec": spec,
        "summary": summary,
        "predictions": predictions,
    }
    path = tmp_path / "inner.json"
    path.write_text(json.dumps(row))
    assert _load_inner_journal(
        path,
        spec=spec,
        expected_rows=predictions,
        implementation=implementation,
        input_hashes=inputs,
    )["summary"] == summary
    row["summary"]["family"] = "xattn_clip"
    path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="summary identity"):
        _load_inner_journal(
            path,
            spec=spec,
            expected_rows=predictions,
            implementation=implementation,
            input_hashes=inputs,
        )


def test_outer_resume_rejects_refit_corruption(tmp_path):
    implementation = {"library": "lib", "runner": "run"}
    inputs = _journal_hashes()
    selected = {
        "s": 0.05,
        "family": "pooled",
        "probe": "ridge",
        "layer": 0,
        "inner_validation_accuracy": 1.0,
        "inner_validation_balanced_accuracy": 1.0,
    }
    predictions = [
        {
            "s": 0.05,
            "family": "pooled",
            "probe": "ridge",
            "layer": 0,
            "split": "outer_eval",
            "clip": "c1",
            "gen_id": "g1",
            "true_label": "a",
            "predicted_label": "a",
        },
        {
            "s": 0.05,
            "family": "pooled",
            "probe": "ridge",
            "layer": 0,
            "split": "outer_eval",
            "clip": "c2",
            "gen_id": "g2",
            "true_label": "b",
            "predicted_label": "b",
        },
    ]
    outer = accuracy_and_video_ci(
        np.asarray(["a", "b"]), ["a", "b"], ["c1", "c2"], n_boot=1000, seed=0
    )
    row = {
        "schema": "arc4_b1_outer_selected_v2",
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_amendment_sha256": AMENDMENT_SHA256,
        "join_amendment_sha256": JOIN_AMENDMENT_SHA256,
        "join_amendment_2_sha256": JOIN_AMENDMENT_2_SHA256,
        "join_amendment_3_sha256": JOIN_AMENDMENT_3_SHA256,
        "join_amendment_4_sha256": JOIN_AMENDMENT_4_SHA256,
        "implementation_sha256": implementation,
        "input_sha256": inputs,
        "selected_identity": {"s": 0.05, "family": "pooled", "probe": "ridge", "layer": 0},
        "inner_selected_spec": selected,
        "refit": {"epochs": None},
        "outer": outer,
        "predictions": predictions,
    }
    path = tmp_path / "outer.json"
    path.write_text(json.dumps(row))
    _load_outer_journal(
        path,
        selected=selected,
        expected_rows=predictions,
        implementation=implementation,
        input_hashes=inputs,
    )
    row["refit"] = {"epochs": 7}
    path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="refit metadata"):
        _load_outer_journal(
            path,
            selected=selected,
            expected_rows=predictions,
            implementation=implementation,
            input_hashes=inputs,
        )
