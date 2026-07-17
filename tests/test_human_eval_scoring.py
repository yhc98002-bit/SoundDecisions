"""Known-answer and contract tests for the offline human-eval scorer."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "results" / "human_eval_pack" / "score_ac1.py"
FIXTURES = ROOT / "tests" / "fixtures" / "human_eval"
SPEC = importlib.util.spec_from_file_location("human_eval_score_ac1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)


def fixture_paths() -> list[Path]:
    return [
        FIXTURES / "ratings_rater_a.json",
        FIXTURES / "ratings_rater_b.json",
    ]


def test_gwet_ac1_known_answer_from_rating_exports() -> None:
    """Fixed 3-choice scale has p_o=3/4, p_e=15/64, hence AC1=33/49."""

    scored = SCORER.score_paths(fixture_paths())
    presence = scored["agreement"]["presence_verdict"]

    assert presence["observed_agreement"] == pytest.approx(0.75)
    assert presence["expected_agreement"] == pytest.approx(15.0 / 64.0)
    assert presence["ac1"] == pytest.approx(33.0 / 49.0)
    assert presence["n_items"] == 4
    assert presence["n_ratings"] == 8


def test_gwet_ac1_item_balances_unequal_rater_counts() -> None:
    ratings = {
        "item-a": ["present", "present"],
        "item-b": ["absent", "uncertain", "uncertain"],
    }

    result = SCORER.gwet_ac1(
        ratings,
        categories=("present", "absent", "uncertain"),
    )

    assert result["observed_agreement"] == pytest.approx(2.0 / 3.0)
    assert result["expected_agreement"] == pytest.approx(11.0 / 36.0)
    assert result["ac1"] == pytest.approx(13.0 / 25.0)


def test_interval_overlap_known_answer() -> None:
    scored = SCORER.score_paths(fixture_paths())
    anchor = scored["interval_overlap"]["anchor"]

    # One pair has IoU 1/3 and three are identical: mean = 5/6.
    assert anchor["n_pairwise_comparisons"] == 4
    assert anchor["mean_iou"] == pytest.approx(5.0 / 6.0)
    assert anchor["median_iou"] == pytest.approx(1.0)
    assert anchor["mean_start_abs_seconds"] == pytest.approx(0.125)
    assert anchor["mean_end_abs_seconds"] == pytest.approx(0.125)


def test_audit_selection_is_task_stratified_and_deterministic() -> None:
    forward = {
        "anchor_presence": [f"A-{index:02d}" for index in range(10)],
        "two_event": [f"P-{index:02d}" for index in range(5)],
    }
    reversed_order = {
        "two_event": list(reversed(forward["two_event"])),
        "anchor_presence": list(reversed(forward["anchor_presence"])),
    }

    first = SCORER.select_audit_items(forward)
    second = SCORER.select_audit_items(reversed_order)

    assert first == second
    assert first["flow"] == "MLLM_PRIMARY_HUMAN_AUDIT"
    assert first["by_task"]["anchor_presence"]["n_selected"] == 2
    assert first["by_task"]["two_event"]["n_selected"] == 1
    assert first["n_selected"] == 3
    assert first["uses_rating_outcomes"] is False
    assert sum(bool(flag["selected_tasks"]) for flag in first["flags"]) == 3


def test_overlap_item_contributes_to_both_task_strata() -> None:
    exports = [SCORER.load_rating_export(path) for path in fixture_paths()]
    exports = copy.deepcopy(exports)
    for payload in exports:
        overlap = payload["ratings"][0]
        overlap["tasks"].append("two_event")
        overlap["pair_curation"] = {
            "verdict": "confirm",
            "event_1": {"start_s": 0.0, "end_s": 0.2},
            "event_2": {"start_s": 0.8, "end_s": 1.0},
            "event_1_description": "first contact",
            "event_2_description": "second contact",
            "note": "",
        }

    scored = SCORER.score_exports(exports)

    assert scored["agreement"]["anchor_status"]["n_items"] == 4
    assert scored["agreement"]["pair_curation_verdict"]["n_items"] == 1
    assert scored["agreement"]["pair_curation_verdict"]["ac1"] == pytest.approx(1.0)
    assert scored["audit_20_percent"]["by_task"]["two_event"]["n_items"] == 1


def test_cli_emits_valid_score_json(tmp_path: Path) -> None:
    output = tmp_path / "score.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *map(str, fixture_paths()),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sounddecisions-human-eval-score-v1.0"
    assert payload["n_raters"] == 2


def test_scorer_rejects_one_export() -> None:
    payload = SCORER.load_rating_export(fixture_paths()[0])
    with pytest.raises(ValueError, match="at least two"):
        SCORER.score_exports([payload])


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda payload: payload.update({"unexpected": True}), "schema validation failed"),
        (lambda payload: payload.update({"started_at": "July 17, 2026"}), "schema validation failed"),
    ],
)
def test_scorer_rejects_exports_outside_committed_schema(
    tmp_path: Path,
    mutate,
    expected: str,
) -> None:
    payload = json.loads(fixture_paths()[0].read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        SCORER.load_rating_export(path)
