"""Load and save rubrics in YAML or JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from decider.rubric import Rubric

try:
    import yaml  # PyYAML is the only required third-party dep for the [rubric] extra
except ModuleNotFoundError as e:  # pragma: no cover - exercised at the CLI, not in unit tests
    raise ModuleNotFoundError(
        "decider.rubric_io requires PyYAML; install with `pip install -e .[rubric]`"
    ) from e


def load_rubric(path: str | Path) -> Rubric:
    """Load a rubric from a YAML or JSON file. The format is selected by extension."""
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() in {".yaml", ".yml"}:
        payload: Any = yaml.safe_load(text)
    elif p.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        raise ValueError(f"rubric must be .yaml, .yml or .json; got '{p.suffix}'")
    return Rubric.from_dict(payload)


def save_rubric(rubric: Rubric, path: str | Path) -> None:
    """Save a rubric to a YAML or JSON file. The format is selected by extension."""
    p = Path(path)
    payload = {
        "name": rubric.name,
        "description": rubric.description,
        "dimensions": {
            name: {
                "metric": dim.metric,
                "operator": dim.operator,
                "target": list(dim.target) if isinstance(dim.target, tuple) else dim.target,
                "weight": dim.weight,
                "failure_message": dim.failure_message,
            }
            for name, dim in rubric.dimensions.items()
        },
    }
    if p.suffix.lower() in {".yaml", ".yml"}:
        p.write_text(yaml.safe_dump(payload, sort_keys=False))
    elif p.suffix.lower() == ".json":
        p.write_text(json.dumps(payload, indent=1))
    else:
        raise ValueError(f"rubric must be .yaml, .yml or .json; got '{p.suffix}'")