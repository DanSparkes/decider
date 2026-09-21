"""The structured rubric grade: dataclasses, IO, aggregation, and CLI."""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from decider.rubric import (
    Rubric,
    RubricDimension,
    RubricRun,
    aggregate_rubric,
)
from decider.rubric_io import load_rubric, save_rubric


def _fake_results(acc: float = 0.85, nll: float = 0.5, brier: float = 0.4,
                  ece: float = 0.03, aurc: float = 0.1, acc_at_80: float = 0.9):
    return {
        "t1": {"acc": acc, "nll": nll, "brier": brier, "ece": ece,
               "aurc": aurc, "acc_at_80": acc_at_80},
        "t2": {"acc": acc, "nll": nll, "brier": brier, "ece": ece,
               "aurc": aurc, "acc_at_80": acc_at_80},
    }


def _fake_dump(sharp: bool = True, n: int = 100, k: int = 4):
    rng = np.random.default_rng(0)
    if sharp:
        probs = np.zeros((n, k))
        for i in range(n):
            probs[i, rng.integers(0, k)] = 1.0
    else:
        probs = np.full((n, k), 1.0 / k)
    golds = rng.integers(0, k, size=n)
    nopts = np.full(n, k)
    qidx = np.zeros(n, dtype=int)
    return {"t1": {"probs": probs, "golds": golds, "nopts": nopts, "qidx": qidx},
            "t2": {"probs": probs.copy(), "golds": golds.copy(),
                   "nopts": nopts.copy(), "qidx": qidx.copy()}}


def test_dimension_between_requires_tuple_and_validates_bounds():
    with pytest.raises(ValueError):
        RubricDimension(metric="acc", operator="between", target=0.5)
    with pytest.raises(ValueError):
        RubricDimension(metric="acc", operator="between", target=(0.9, 0.5))
    d = RubricDimension(metric="ece", operator="between", target=(0.0, 0.1))
    assert d.evaluate(0.05) and not d.evaluate(0.2)


def test_dimension_rejects_tuple_for_scalar_operators_and_negative_weight():
    with pytest.raises(ValueError):
        RubricDimension(metric="acc", operator=">=", target=(0.1, 0.5))
    with pytest.raises(ValueError):
        RubricDimension(metric="acc", operator="<=", target=0.5, weight=-1.0)


def test_dimension_evaluate_all_operators():
    cases = [
        ("acc", ">=", 0.8, 0.85, True),
        ("acc", ">=", 0.8, 0.79, False),
        ("ece", "<=", 0.05, 0.04, True),
        ("ece", "<=", 0.05, 0.06, False),
        ("acc", "==", 1.0, 1.0, True),
        ("acc", "==", 1.0, 0.99, False),
    ]
    for metric, op, target, value, want in cases:
        assert RubricDimension(metric=metric, operator=op, target=target).evaluate(value) is want


def test_rubric_from_dict_requires_name_and_dimensions():
    with pytest.raises(ValueError):
        Rubric.from_dict({"dimensions": {}})
    r = Rubric.from_dict({
        "name": "x",
        "description": "d",
        "dimensions": {"a": {"metric": "acc", "operator": ">=", "target": 0.8,
                             "weight": 2.0, "failure_message": "fail"}},
    })
    assert r.name == "x" and r.description == "d"
    assert isinstance(r.dimensions["a"], RubricDimension)
    assert r.dimensions["a"].weight == 2.0 and r.dimensions["a"].failure_message == "fail"
    assert r.dimensions["a"].weight == 2.0


def test_rubric_default_weight_is_one_and_default_message_is_empty():
    r = Rubric.from_dict({"name": "x", "dimensions": {"a": {"metric": "acc", "operator": ">=", "target": 0.5}}})
    assert r.dimensions["a"].weight == 1.0
    assert r.dimensions["a"].failure_message == ""


def test_run_metric_aggregates_standard_metrics_across_tasks():
    run = RubricRun(results=_fake_results(acc=0.8, nll=0.5, ece=0.03), dump=_fake_dump())
    assert run.metric("acc") == pytest.approx(0.8)
    assert run.metric("nll") == pytest.approx(0.5)
    assert run.metric("ece") == pytest.approx(0.03)


def test_run_metric_synthetic_distinctness_reflects_distribution_shape():
    sharp_run = RubricRun(results=_fake_results(), dump=_fake_dump(sharp=True))
    flat_run = RubricRun(results=_fake_results(), dump=_fake_dump(sharp=False))
    assert sharp_run.metric("option_distinctness") > 0.9
    assert flat_run.metric("option_distinctness") < 0.30       # uniform over k=4 -> 0.25


def test_run_metric_unknown_name_raises():
    run = RubricRun(results=_fake_results(), dump=_fake_dump())
    with pytest.raises(ValueError):
        run.metric("nope")


def test_from_results_loads_pickle(tmp_path):
    pkl = tmp_path / "preds.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(_fake_dump(), f)
    run = RubricRun.from_results(_fake_results(), str(pkl))
    assert set(run.dump.keys()) == {"t1", "t2"}
    assert isinstance(run.dump["t1"]["probs"], np.ndarray)


def test_aggregate_rubric_pass_and_weighted_score():
    rubric = Rubric.from_dict({
        "name": "r",
        "dimensions": {
            "a": {"metric": "acc", "operator": ">=", "target": 0.8, "weight": 2.0},
            "b": {"metric": "ece", "operator": "<=", "target": 0.05, "weight": 1.0},
        },
    })
    run = RubricRun(results=_fake_results(acc=0.85, ece=0.03), dump=_fake_dump())
    report = aggregate_rubric(rubric, run)
    assert report.passed and report.weighted_score == pytest.approx(1.0)
    assert [o.name for o in report.outcomes] == ["a", "b"]
    assert {o.passed for o in report.outcomes} == {True}


def test_aggregate_rubric_fail_renders_failure_message():
    rubric = Rubric.from_dict({
        "name": "r",
        "dimensions": {"acc": {"metric": "acc", "operator": ">=", "target": 0.9,
                               "weight": 1.0, "failure_message": "below 0.9"}},
    })
    report = aggregate_rubric(rubric, RubricRun(results=_fake_results(acc=0.7), dump=_fake_dump()))
    assert not report.passed
    assert report.weighted_score == 0.0
    out = report.render()
    assert "FAIL" in out and "below 0.9" in out


def test_aggregate_rubric_weights_partial_score():
    rubric = Rubric.from_dict({
        "name": "r",
        "dimensions": {
            "a": {"metric": "acc", "operator": ">=", "target": 0.8, "weight": 2.0},
            "b": {"metric": "ece", "operator": "<=", "target": 0.01, "weight": 1.0},
        },
    })
    report = aggregate_rubric(rubric, RubricRun(results=_fake_results(acc=0.9, ece=0.05),
                                                 dump=_fake_dump()))
    assert not report.passed
    assert report.weighted_score == pytest.approx(2.0 / 3.0)


def test_load_save_round_trip_yaml(tmp_path):
    rubric = Rubric.from_dict({
        "name": "round-trip",
        "description": "d",
        "dimensions": {
            "acc": {"metric": "acc", "operator": ">=", "target": 0.8, "weight": 2.0,
                    "failure_message": "below"},
            "between": {"metric": "ece", "operator": "between", "target": [0.0, 0.1]},
        },
    })
    path = tmp_path / "r.yaml"
    save_rubric(rubric, path)
    assert yaml_available()
    loaded = load_rubric(path)
    assert loaded.name == rubric.name
    assert loaded.description == rubric.description
    assert loaded.dimensions["acc"].weight == 2.0
    assert loaded.dimensions["between"].target == (0.0, 0.1)


def test_load_rejects_unknown_extension(tmp_path):
    (tmp_path / "r.txt").write_text("name: x")
    with pytest.raises(ValueError):
        load_rubric(tmp_path / "r.txt")


def test_load_supports_json(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({
        "name": "j",
        "dimensions": {"a": {"metric": "acc", "operator": ">=", "target": 0.5}},
    }))
    assert load_rubric(path).name == "j"


def yaml_available() -> bool:
    try:
        import yaml  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


@pytest.mark.skipif(not yaml_available(), reason="PyYAML not installed")
def test_cli_rubric_mode_passes_on_synthetic_passing_eval(tmp_path):
    """`python -m decider.evaluate --rubric rubrics/routing.yaml --eval-out eval.json`
    on a synthetic eval.json + preds.pkl must return a PASS/FAIL report.
    """
    eval_dir = tmp_path / "out"
    eval_dir.mkdir()
    eval_json = eval_dir / "eval.json"
    preds_pkl = eval_dir / "preds.pkl"

    # A well-calibrated, accurate run.
    eval_json.write_text(json.dumps({"results": _fake_results(acc=0.9, nll=0.4, brier=0.2,
                                                              ece=0.02, aurc=0.05, acc_at_80=0.95)}))
    with open(preds_pkl, "wb") as f:
        pickle.dump(_fake_dump(sharp=True), f)

    rubric_path = tmp_path / "r.yaml"
    rubric_path.write_text("""
name: routing
dimensions:
  acc:
    metric: acc
    operator: ">="
    target: 0.80
    weight: 1.0
  ece:
    metric: ece
    operator: "<="
    target: 0.05
""".strip())

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "decider.evaluate",
         "--rubric", str(rubric_path), "--eval-out", str(eval_json)],
        cwd=repo_root, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    assert "Rubric: routing -> PASS" in result.stdout
    assert "[PASS] acc" in result.stdout and "[PASS] ece" in result.stdout


@pytest.mark.skipif(not yaml_available(), reason="PyYAML not installed")
def test_cli_rubric_mode_fails_on_synthetic_failing_eval(tmp_path):
    eval_dir = tmp_path / "out"
    eval_dir.mkdir()
    eval_json = eval_dir / "eval.json"
    preds_pkl = eval_dir / "preds.pkl"

    eval_json.write_text(json.dumps({"results": _fake_results(acc=0.5, ece=0.2)}))
    with open(preds_pkl, "wb") as f:
        pickle.dump(_fake_dump(sharp=False), f)

    rubric_path = tmp_path / "r.yaml"
    rubric_path.write_text("""
name: routing
dimensions:
  acc:
    metric: acc
    operator: ">="
    target: 0.80
  ece:
    metric: ece
    operator: "<="
    target: 0.05
""".strip())

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "decider.evaluate",
         "--rubric", str(rubric_path), "--eval-out", str(eval_json)],
        cwd=repo_root, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert "Rubric: routing -> FAIL" in result.stdout
    assert "[FAIL] acc" in result.stdout or "[FAIL] ece" in result.stdout


@pytest.mark.skipif(not yaml_available(), reason="PyYAML not installed")
def test_cli_rubric_missing_eval_out_errors(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "decider.evaluate",
         "--rubric", str(tmp_path / "no-such-rubric.yaml")],
        cwd=repo_root, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "--eval-out" in result.stderr or "--rubric requires --eval-out" in result.stderr


def test_checked_in_rubrics_load_and_are_well_formed():
    """The four rubrics in `rubrics/` should load and have at least one dimension each."""
    repo_root = Path(__file__).resolve().parents[1]
    for name in ("routing", "calibration", "abstention", "option_design"):
        path = repo_root / "rubrics" / f"{name}.yaml"
        if not path.exists():
            pytest.skip(f"{path} not present")
        if not yaml_available():
            pytest.skip("PyYAML not installed")
        rubric = load_rubric(path)
        assert rubric.name == name
        assert len(rubric.dimensions) >= 1
        for dim in rubric.dimensions.values():
            assert dim.metric in {
                "acc", "nll", "brier", "ece", "aurc", "acc_at_80",
                "abstain_rate", "gate_pass_rate", "option_distinctness",
            }
            assert dim.operator in {">=", "<=", "==", "between"}
            assert dim.weight > 0


def test_option_design_rubric_references_known_failure_modes():
    """The README's known failure modes (`support` next to `other`, destructive
    actions) must be referenced somewhere in option_design.yaml."""
    repo_root = Path(__file__).resolve().parents[1]
    if not yaml_available():
        pytest.skip("PyYAML not installed")
    path = repo_root / "rubrics" / "option_design.yaml"
    if not path.exists():
        pytest.skip(f"{path} not present")
    rubric = load_rubric(path)
    blob = " ".join([rubric.description] + [d.failure_message for d in rubric.dimensions.values()])
    assert "support" in blob
    assert "other" in blob
    assert "destructive" in blob.lower()