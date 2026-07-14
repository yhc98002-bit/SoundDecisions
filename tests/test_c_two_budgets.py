"""Tests for scripts/c_two_budgets.py (Arc-3 Tier-B Part C assembly).

CPU-only, reads cached artifacts. Verifies the assembly is faithful to the cached CSVs/JSON,
that the causal channel is NOT conflated with the observational share, and that the entropy
lens reproduces the documented distinct-class-count sequence. No new generation; no token.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/c_two_budgets.py"

spec = importlib.util.spec_from_file_location("c_two_budgets", SCRIPT)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


@pytest.fixture(scope="module")
def data():
    return C.build()


def require_dial_caches():
    import glob
    missing = []
    for cfg in C.CFG_LIST:
        files = glob.glob(str(ROOT / C.DIAL_GLOB.format(C=cfg)))
        if len(files) != 24:
            missing.append(f"cfg={cfg}: {len(files)}/24")
    if missing:
        pytest.skip("cfg-dial caches unavailable (" + ", ".join(missing) + ")")


def test_inputs_exist():
    for p in (C.BUDGET_CFG1, C.BUDGET_CFG45, C.CSWAP_MAP, C.CSWAP_SUMMARY):
        assert p.exists(), f"missing cached input {p}"


def test_no_token_descriptive(data):
    # Part C is pre-registered as descriptive: nothing in the assembly emits a token.
    blob = json.dumps(data).lower()
    assert "descriptive" in data["pre_registered_as"].lower()
    assert "token" not in {k.lower() for k in data["per_axis"]["class"]}
    # the only mentions of 'token' must be the explicit "no token" note
    assert "no token" in data["pre_registered_as"].lower()


def test_observational_share_matches_csv(data):
    b1 = C.load_budget(C.BUDGET_CFG1)
    b45 = C.load_budget(C.BUDGET_CFG45)
    for a in C.AXES:
        o = data["per_axis"][a]["observational"]
        assert o["conditioning_share_cfg1"] == pytest.approx(b1[a]["conditioning"])
        assert o["conditioning_share_cfg45"] == pytest.approx(b45[a]["conditioning"])
        # delta is consistent
        assert o["conditioning_share_delta_cfg1_to_cfg45"] == pytest.approx(
            b45[a]["conditioning"] - b1[a]["conditioning"])


def test_causal_channel_distinct_from_observational(data):
    # The causal block must be labelled as NOT the conditioning share, and must carry
    # follow/retention/s_cond (cond-swap quantities), not the budget shares.
    for a in C.AXES:
        c = data["per_axis"][a]["causal_conditioning_responsiveness"]
        assert "NOT the observational conditioning share" in c["_label"]
        assert {"follow_rate_low_s", "retention_rate_high_s", "s_cond"} <= set(c)
        # follow-rate must come from the cond-swap map, not equal the conditioning share
        o = data["per_axis"][a]["observational"]
        # they are different quantities; assert not accidentally aliased
        assert c["follow_rate_low_s"] != o["conditioning_share_cfg1"] or a == "__never__"


def test_cswap_follow_matches_map(data):
    cmap = C.load_cswap_map(C.CSWAP_MAP)
    s_low = data["s_low"]
    for a in C.AXES:
        c = data["per_axis"][a]["causal_conditioning_responsiveness"]
        assert c["follow_rate_low_s"] == pytest.approx(cmap[a][s_low]["follow_rate"])


def test_class_is_the_divergence_case(data):
    # Headline: class observational share rises with cfg but cond-swap sanity FAILS.
    cls = data["per_axis"]["class"]
    assert cls["observational"]["conditioning_share_cfg45"] > \
        cls["observational"]["conditioning_share_cfg1"]
    assert cls["causal_conditioning_responsiveness"]["sanity_passed"] is False
    assert cls["divergence_flag"] is True
    # and the well-behaved axes do NOT raise the divergence flag
    for a in ("presence", "timing"):
        assert data["per_axis"][a]["causal_conditioning_responsiveness"]["sanity_passed"] is True
        assert data["per_axis"][a]["divergence_flag"] is False


def test_entropy_lens_sequence(data):
    # Distinct-class-count must reproduce the documented 4.83 -> 3.62 collapse.
    require_dial_caches()
    seq = data["entropy_lens"]["sequence_cfg_1_to_4p5"]
    assert len(seq) == len(C.CFG_LIST)
    assert seq[0] == pytest.approx(4.8333, abs=1e-3)
    assert seq[-1] == pytest.approx(3.625, abs=1e-3)
    # monotone-ish collapse: cfg=4.5 strictly below cfg=1.0 by ~1.2 classes
    assert data["entropy_lens"]["collapse_delta"] == pytest.approx(4.8333 - 3.625, abs=1e-3)
    assert data["entropy_lens"]["collapse_delta"] > 1.0


def test_distinct_count_recompute_from_npz():
    # Independent recompute of cfg=1.0 distinct-class-count straight from the dial npz cache.
    import glob
    require_dial_caches()
    files = sorted(glob.glob(str(ROOT / C.DIAL_GLOB.format(C="1"))))
    dcs = []
    for f in files:
        lab = np.asarray(np.load(f, allow_pickle=True)["labels"]).tolist()
        dcs.append(len(set(lab)))
    assert float(np.mean(dcs)) == pytest.approx(4.8333, abs=1e-3)


def test_outputs_written(tmp_path, monkeypatch):
    require_dial_caches()
    out_json = tmp_path / "entropy_lens_v3.json"
    out_md = tmp_path / "entropy_lens_v3.md"
    monkeypatch.setattr(C, "ARC4_WPA2_OUT_DIR", tmp_path)
    monkeypatch.setattr(C, "ARC4_WPA2_OUT_JSON", out_json)
    monkeypatch.setattr(C, "ARC4_WPA2_OUT_MD", out_md)
    C.main()
    assert out_json.exists() and out_md.exists()
    blob = json.loads(out_json.read_text())
    assert blob["ci_method"] == "clip_bootstrap"
    assert blob["mechanism_claim"] is None
    assert "mode collapse" not in out_json.read_text().lower()
    md = out_md.read_text()
    assert "descriptive" in md.lower()
    assert "causal mechanism" in md.lower()


def test_abstain_filtered_entropy_lens_on_synthetic_caches(tmp_path, monkeypatch):
    np.savez(tmp_path / "dial_cfg1__a.npz", labels=np.array([
        "abstain", "dog", "dog", "cat",
    ]))
    np.savez(tmp_path / "dial_cfg1__b.npz", labels=np.array([
        "abstain", "abstain", "dog", "dog",
    ]))
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setattr(C, "DIAL_GLOB", "dial_cfg{C}__*.npz")
    monkeypatch.setattr(C, "CFG_LIST", ("1",))
    monkeypatch.setattr(C, "EXPECTED_DIAL_CLIPS", 2)

    result = C.build_entropy_lens_v3()
    row = result["by_cfg"]["1"]
    assert row["mean_distinct_including_abstain"] == pytest.approx(2.5)
    assert row["mean_distinct_excluding_abstain"] == pytest.approx(1.5)
    assert row["n_abstain"] == 3
    assert row["n_labels"] == 8
    assert row["abstain_rate"] == pytest.approx(3 / 8)
    assert row["abstain_ci_lo"] < row["abstain_rate"] < row["abstain_ci_hi"]
    assert result["ci_method"] == "clip_bootstrap"
    assert result["decision_token"] is None

    out_dir = tmp_path / "arc4_wpA2"
    monkeypatch.setattr(C, "ARC4_WPA2_OUT_DIR", out_dir)
    monkeypatch.setattr(C, "ARC4_WPA2_OUT_JSON", out_dir / "entropy_lens_v3.json")
    monkeypatch.setattr(C, "ARC4_WPA2_OUT_MD", out_dir / "entropy_lens_v3.md")
    C.main(["--exclude-abstain"])
    assert C.ARC4_WPA2_OUT_JSON.exists()
    assert "distinct excl. abstain" in C.ARC4_WPA2_OUT_MD.read_text()


def test_clip_bootstrap_is_deterministic():
    values = [0.0, 0.25, 0.5, 1.0]
    first = C.bootstrap_clip_mean(values, n_boot=1000, seed=0)
    second = C.bootstrap_clip_mean(values, n_boot=1000, seed=0)
    assert first == pytest.approx(second)
    assert first[0] == pytest.approx(np.mean(values))
    assert first[1] < first[0] < first[2]


def test_missing_dial_caches_are_skipped(monkeypatch):
    monkeypatch.setattr(C, "DIAL_GLOB", "definitely_missing_cfg{C}__*.npz")
    with pytest.raises(pytest.skip.Exception):
        require_dial_caches()
