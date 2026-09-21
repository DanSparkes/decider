"""Discover probe families from `rubrics/probes/*.yaml` and expose each as a runnable unit.

A probe family is a pair of siblings that live side by side in the repo: a
*module* under `decider/probes/` that implements a probe (a decision-level
evaluation recipe du" model) and a *rubric YAML* under `rubrics/probes/` that
declares how that family's results are scored.  Adding a new probe therefore
requires only a sibling YAML plus a family module — the runner and registry
never change.

The rubric YAML uses the same shape as Stream 2's rubric stack
(`Rubric.from_dict`), so a family that reuses the shared dimensions can be
scored by `RubricRun` + `aggregate_rubric` with no adapter code here.

Registry contract
-----------------
A family YAML carries:

    name: <family>             # must equal the file stem
    module: decider.probes.<family>   # the family module to drive
    description: ...          # one line, shown in the report header
    dimensions:               # RubricDimension specs (metric/operator/target/weight)

Helper `families()` scans the directory, builds `ProbeFamily` objects
(name, rubric path, rubric), and resolves each `module` import ahead of time
so the runner can fail fast on a missing module or a YAML with no matching
dimension.  `family()`/`run_functions()` also let callers drive a single
family, or the whole set, without touching the runner.
"""
from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from decider.rubric import Rubric
from decider.rubric_io import load_rubric

#: Directory the family YAMLs live in, relative to the repo root.
RUBRICS_DIR = Path("rubrics/probes")


@dataclass(frozen=True)
class ProbeFamily:
    """A discovered probe family: a module to run + the rubric that scores it."""

    name: str
    module: str
    description: str
    rubric_path: str
    rubric: Rubric = field(repr=False)

    @property
    def dimensions(self) -> Mapping[str, object]:
        return self.rubric.dimensions


def _discover_paths(rubrics_dir: str | Path = RUBRICS_DIR) -> list[Path]:
    root = Path(rubrics_dir)
    return sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml"))


def families(rubrics_dir: str | Path = RUBRICS_DIR) -> Sequence[ProbeFamily]:
    """Discover all probe families, one per sibling YAML under `rubrics_dir`.

    Order is deterministic (sorted by file name) so reports are stable across
    runs.  A YAML whose `name` does not match its file stem, or whose module
    cannot be imported, raises immediately — the runner surfaces these before
    doing any model work.
    """
    out: list[ProbeFamily] = []
    for path in _discover_paths(rubrics_dir):
        rubric = load_rubric(path)
        stem = path.stem
        pkg = path.read_text(encoding="utf-8")
        module = _module_for(stem, rubric, pkg)
        out.append(
            ProbeFamily(
                name=stem,
                module=module,
                description=rubric.description,
                rubric_path=str(path),
                rubric=rubric,
            )
        )
    return out


def _module_for(stem: str, rubric: Rubric, text: str) -> str:
    """Extract the family module name from a YAML, defaulting to `decider.probes.<stem>`."""
    import re

    m = re.search(r"^\s*module:\s*(\S+)\s*$", text, re.MULTILINE)
    module = m.group(1) if m else f"decider.probes.{stem}"
    # Validate eagerly: a family whose module is missing is a hard error, not a skip.
    try:
        importlib.import_module(module)
    except Exception as e:  # mirror rubric_io's fail-fast style
        raise ImportError(f"{stem}: cannot import family module {module!r}: {e}") from e
    return module


def family(name: str, rubrics_dir: str | Path = RUBRICS_DIR) -> ProbeFamily:
    """Return a single family by name, or raise KeyError if absent."""
    for fam in families(rubrics_dir):
        if fam.name == name:
            return fam
    raise KeyError(f"no probe family named {name!r}")


def run_functions(rubrics_dir: str | Path = RUBRICS_DIR) -> Mapping[str, object]:
    """Map family name -> imported module with a `main(args)` / `run(...)` entry point.

    Kept separate from `families()` so callers who only inspect the registry
    (e.g. the report exporter) never need to import the family modules.
    """
    import importlib

    return {
        fam.name: importlib.import_module(fam.module)
        for fam in families(rubrics_dir)
    }
