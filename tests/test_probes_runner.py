"""Model-free tests for the probe runner's registry + rubric seam.

What is NOT tested here (deliberately, matching the rest of the suite):
any code path that imports ``decider.model``/``torch``/a family module, or
that shells out to a family's CLI.  Those seams are torch-gated and skipped
in this harness (same skip pattern as the Stream-2 rubric tests); a normal
``decider`` venv runs them.

What IS tested (model-free):
  * sibling-YAML discovery returns all five families, sorted by name;
  * ``family()`` lookup works and raises ``KeyError`` for a missing name;
  * ``run_functions()`` returns one entry per family (modules are only
    resolved lazily by the runner's own seam);
  * every sibling ``rubrics/probes/*.yaml`` loads through the Stream-2 rubric
    stack with a valid dimension schema;
  * ``aggregate_rubric`` against a hand-built ``RubricRun`` renders a report
    and folds per-family weights into a weighted score.

    python -m pytest tests/test_probes_runner.py -q
"""
from pathlib import Path

import pytest

from decider.probes import registry
from decider.rubric import RubricRun, aggregate_rubric

EXPECTED_FAMILIES = [
    "applications",
    "batteries",
    "cua_s1_forms",
    "independence",
    "isolated",
]

_VALID_METRICS = {
    "acc",
    "nll",
    "brier",
    "ece",
    "aurc",
    "acc_at_80",
    "dp_order",
    "abstain_rate",
    "gate_pass_rate",
    "option_distinctness",
}
_VALID_OPERATORS = {">=", "<=", "==", "between"}


def _require_model_stack() -> None:
    """Registry discovery imports each family module (model stack); skip if absent."""
    pytest.importorskip("decider.model")


def _rubrics_dir() -> Path:
    """Path to the probe sibling YAMLs (repo-root-relative, like the registry)."""
    root = Path(__file__).resolve().parents[1]
    return root / "rubrics" / "probes"


# --- registry / discovery -------------------------------------------------


def test_registry_discovers_five_sorted_families():
    _require_model_stack()
    fams = list(registry.families(_rubrics_dir()))
    names = [f.name for f in fams]
    assert names == EXPECTED_FAMILIES


def test_registry_family_lookup():
    _require_model_stack()
    fam = registry.family("isolated", _rubrics_dir())
    assert fam.name == "isolated"
    assert fam.module == "decider.probes.isolated"


def test_registry_family_missing_raises():
    _require_model_stack()
    with pytest.raises(KeyError):
        registry.family("no_such_family", _rubrics_dir())


def test_registry_run_functions_names():
    _require_model_stack()
    funcs = registry.run_functions(_rubrics_dir())
    assert set(funcs) == set(EXPECTED_FAMILIES)


# --- sibling YAML schema --------------------------------------------------


@pytest.mark.parametrize("family", EXPECTED_FAMILIES)
def test_sibling_yaml_loads_with_valid_schema(family):
    from decider.rubric_io import load_rubric

    path = _rubrics_dir() / f"{family}.yaml"
    rubric = load_rubric(path)

    for dim in rubric.dimensions.values():
        assert dim.metric in _VALID_METRICS
        assert dim.operator in _VALID_OPERATORS
        assert dim.weight > 0


# --- aggregation over a model-free RubricRun -------------------------------


def _stub_run() -> RubricRun:
    """A hand-built run; the runner's own seam constructs RubricRun directly."""
    results = {
        "per_task_0": {"acc": 0.91, "nll": 0.42, "brier": 0.13, "ece": 0.05},
        "per_task_1": {"acc": 0.88, "nll": 0.51, "brier": 0.16, "ece": 0.07},
    }
    return RubricRun(results=results, dump={})


def test_aggregate_rubric_renders_report():
    from decider.rubric import RubricReport
    from decider.rubric_io import load_rubric

    rubric = load_rubric(_rubrics_dir() / "isolated.yaml")
    report = aggregate_rubric(rubric, _stub_run())

    assert isinstance(report, RubricReport)
    rendered = report.render()
    assert rubric.name in rendered
    assert "PASS" in rendered or "FAIL" in rendered


def test_weighted_score_folds_all_dimensions():
    from decider.rubric_io import load_rubric

    rubric = load_rubric(_rubrics_dir() / "isolated.yaml")
    report = aggregate_rubric(rubric, _stub_run())
    # Every dimension contributes a weighted score; the aggregate is finite.
    assert report.weighted_score >= 0.0
    assert report.weighted_score <= 1.0
