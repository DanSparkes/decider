"""Public re-exports for downstream streams.

Streams 3 and 5 import the rubric surface from `decider`:
    from decider import Rubric, RubricDimension, RubricRun
"""
from decider.rubric import (
    Rubric,
    RubricDimension,
    RubricRun,
    aggregate_rubric,
)

__all__ = [
    "Rubric",
    "RubricDimension",
    "RubricRun",
    "aggregate_rubric",
]
