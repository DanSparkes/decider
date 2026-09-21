"""Structured rubric grades for decider runs.

A rubric is a named collection of dimensions. Each dimension asserts that a
single scalar metric over the run satisfies an operator against a target. The
synthetic metrics (`abstain_rate`, `gate_pass_rate`, `option_distinctness`)
are computed from the per-example dump produced by `decider.evaluate.run_eval`;
the standard ones (`acc`, `nll`, `brier`, `ece`, `aurc`, `acc_at_80`) come
from the aggregated per-task summary.

Public surface (re-exported from `decider`):
- `Rubric`           — the named, weighted set of dimensions
- `RubricDimension`  — one metric / operator / target / weight tuple
- `RubricRun`        — results + dump, used to evaluate dimensions
- `aggregate_rubric` — run a rubric against a run, get a RubricReport
"""
from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Metric = Literal[
    "acc",
    "nll",
    "brier",
    "ece",
    "aurc",
    "acc_at_80",
    "abstain_rate",
    "gate_pass_rate",
    "option_distinctness",
]
Operator = Literal[">=", "<=", "==", "between"]

_ABSTAIN_BELOW = 0.5
_GATE_ABOVE = 0.8


@dataclass(frozen=True)
class RubricDimension:
    """One assertion: `metric operator target`, weighted by `weight`."""

    metric: Metric
    operator: Operator
    target: float | tuple[float, float]
    weight: float = 1.0
    failure_message: str = ""

    def __post_init__(self) -> None:
        if self.operator == "between":
            if not isinstance(self.target, tuple) or len(self.target) != 2:
                raise ValueError(
                    f"operator 'between' requires target as (lo, hi); got {self.target!r}"
                )
            lo, hi = self.target
            if not lo <= hi:
                raise ValueError(f"operator 'between' requires lo <= hi; got ({lo}, {hi})")
        elif isinstance(self.target, tuple):
            raise ValueError(
                f"operator '{self.operator}' requires a scalar target; got tuple {self.target!r}"
            )
        if self.weight <= 0:
            raise ValueError(f"weight must be positive; got {self.weight}")

    def evaluate(self, value: float) -> bool:
        if self.operator == ">=":
            return value >= self.target
        if self.operator == "<=":
            return value <= self.target
        if self.operator == "==":
            return value == self.target
        if self.operator == "between":
            t = self.target
            if not isinstance(t, tuple):
                raise ValueError(
                    f"operator 'between' requires a (lo, hi) target; got {t!r}"
                )
            lo, hi = t
            return lo <= value <= hi
        raise ValueError(f"unknown operator: {self.operator}")


@dataclass
class Rubric:
    """A named collection of dimensions. Each dimension is referenced by key in `.dimensions`."""

    name: str
    description: str = ""
    dimensions: dict[str, RubricDimension] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Rubric:
        if not isinstance(payload, Mapping):
            raise TypeError(f"rubric payload must be a mapping; got {type(payload).__name__}")
        if "name" not in payload:
            raise ValueError("rubric payload missing required key 'name'")
        dims_payload = payload.get("dimensions", {})
        if not isinstance(dims_payload, Mapping):
            raise TypeError("rubric 'dimensions' must be a mapping")
        dims: dict[str, RubricDimension] = {}
        for key, spec in dims_payload.items():
            if not isinstance(spec, Mapping):
                raise TypeError(f"dimension '{key}' must be a mapping")
            target = spec["target"]
            if isinstance(target, list):
                target = tuple(target)
            dims[key] = RubricDimension(
                metric=spec["metric"],
                operator=spec["operator"],
                target=target,
                weight=float(spec.get("weight", 1.0)),
                failure_message=str(spec.get("failure_message", "")),
            )
        return cls(
            name=str(payload["name"]),
            description=str(payload.get("description", "")),
            dimensions=dims,
        )


@dataclass
class RubricRun:
    """A run's results (per-task scalars) plus the per-example dump (probs/golds/nopts/qidx).

    The dump is the pickle that `decider.evaluate.run_eval` saves to `preds.pkl`;
    it lets us compute synthetic metrics (abstain_rate, gate_pass_rate,
    option_distinctness) over the full per-example distribution instead of
    only the aggregated scalars.
    """

    results: Mapping[str, Mapping[str, float]]
    dump: Mapping[str, Mapping[str, np.ndarray]]

    @classmethod
    def from_results(cls, results: Mapping[str, Mapping[str, float]], dump_pkl_path: str) -> RubricRun:
        """Build a RubricRun from a `results` dict and the path to a `preds.pkl` produced by `run_eval`."""
        with open(dump_pkl_path, "rb") as f:
            dump = pickle.load(f)
        return cls(results=results, dump=dump)

    def metric(self, name: str) -> float:
        """Aggregate a metric across all tasks. Standard metrics come from `summarize`; the synthetic ones
        (`abstain_rate`, `gate_pass_rate`, `option_distinctness`) are computed from the dump."""
        if name in {"acc", "nll", "brier", "ece", "aurc", "acc_at_80"}:
            vals = [float(r[name]) for r in self.results.values() if name in r]
            return float(np.mean(vals)) if vals else 0.0
        if name == "abstain_rate":
            return self._abstain_rate()
        if name == "gate_pass_rate":
            return self._gate_pass_rate(_GATE_ABOVE)
        if name == "option_distinctness":
            return self._option_distinctness()
        raise ValueError(f"unknown metric: {name}")

    def _abstain_rate(self) -> float:
        rates: list[float] = []
        for d in self.dump.values():
            probs = np.asarray(d["probs"])
            golds = np.asarray(d["golds"])
            ok = golds >= 0
            if not ok.any():
                continue
            conf = probs[ok].max(axis=1)
            rates.append(float((conf < _ABSTAIN_BELOW).mean()))
        return float(np.mean(rates)) if rates else 0.0

    def _gate_pass_rate(self, threshold: float) -> float:
        rates: list[float] = []
        for d in self.dump.values():
            probs = np.asarray(d["probs"])
            golds = np.asarray(d["golds"])
            ok = golds >= 0
            if not ok.any():
                continue
            conf = probs[ok].max(axis=1)
            rates.append(float((conf >= threshold).mean()))
        return float(np.mean(rates)) if rates else 0.0

    def _option_distinctness(self) -> float:
        """Mean of `max(p_i)` over rows. Higher = the model picks one option decisively.

        For a uniform distribution over `k` options the value is `1/k`; for a
        one-hot prediction it is `1.0`. The dump's `probs` are softmax outputs
        with masked options already zero, so this reflects how decisive the
        model is across the options offered.
        """
        ds: list[float] = []
        for d in self.dump.values():
            probs = np.asarray(d["probs"])
            golds = np.asarray(d["golds"])
            ok = golds >= 0
            if not ok.any():
                continue
            p = probs[ok]
            ds.append(float(p.max(axis=1).mean()))
        return float(np.mean(ds)) if ds else 0.0


@dataclass
class DimensionOutcome:
    name: str
    metric: str
    operator: str
    target: float | tuple[float, float]
    value: float
    weight: float
    passed: bool
    failure_message: str


@dataclass
class RubricReport:
    rubric_name: str
    description: str
    outcomes: list[DimensionOutcome] = field(default_factory=list)
    weighted_score: float = 0.0
    passed: bool = False

    def render(self) -> str:
        head = (
            f"Rubric: {self.rubric_name} -> {'PASS' if self.passed else 'FAIL'} "
            f"(weighted score {self.weighted_score:.3f})"
        )
        lines = [head, ""]
        for o in self.outcomes:
            tag = "PASS" if o.passed else "FAIL"
            target = (
                f"({o.target[0]:.4f}, {o.target[1]:.4f})"
                if isinstance(o.target, tuple)
                else f"{o.target:.4f}"
            )
            lines.append(
                f"  [{tag}] {o.name:32s} {o.metric:22s} {o.value:.4f} {o.operator} {target}"
                f"  weight={o.weight}"
            )
            if not o.passed and o.failure_message:
                for chunk in o.failure_message.splitlines():
                    lines.append(f"           {chunk}")
        return "\n".join(lines)


def aggregate_rubric(rubric: Rubric, run: RubricRun) -> RubricReport:
    """Evaluate every dimension in `rubric` against `run` and return a RubricReport.

    The weighted score is the sum of `weight * 1.0` over passing dimensions divided
    by the sum of `weight * 1.0` over all dimensions; a perfect run scores 1.0,
    a fully-failing run scores 0.0.
    """
    outcomes: list[DimensionOutcome] = []
    weighted_value = 0.0
    weighted_max = 0.0
    all_pass = True
    for name, dim in rubric.dimensions.items():
        value = run.metric(dim.metric)
        passed = dim.evaluate(value)
        outcomes.append(
            DimensionOutcome(
                name=name,
                metric=dim.metric,
                operator=dim.operator,
                target=dim.target,
                value=value,
                weight=dim.weight,
                passed=passed,
                failure_message=dim.failure_message,
            )
        )
        if passed:
            weighted_value += dim.weight
        weighted_max += dim.weight
        if not passed:
            all_pass = False
    score = weighted_value / weighted_max if weighted_max > 0 else 0.0
    return RubricReport(
        rubric_name=rubric.name,
        description=rubric.description,
        outcomes=outcomes,
        weighted_score=score,
        passed=all_pass,
    )