"""Run every discovered probe family against a model and emit a JSON + Markdown report.

Usage::

    python -m decider.probes.runner --model runs/r12_v7/model [--n 300] \\
        [--rubrics-dir rubrics/probes] [--out probes_report.json]

The runner is deliberately *model-free* at import time: it never imports
``decider.model``, ``torch``, or any family module's run stack.  Discovery is
YAML-only (the same glob + ``load_rubric`` the registry uses, plus each
sibling YAML's ``module:`` key), so register/render all run in a plain Python
venv without torch installed.

Each family is executed out-of-band: the runner shells out to the family
module's own ``main()`` CLI (``python -m decider.probes.<family> <model>
--n N --out <tmp>.json``), which does the model-heavy work and writes a JSON
result file.  The runner then wraps that JSON in a ``RubricRun`` (via
   ``RubricRun.from_results`` when the family emits a ``preds.pkl`` beside its
   JSON; otherwise directly, with an empty dump) and scores it with
   ``aggregate_rubric`` against the family's sibling rubric
(Stream 2's rubric stack).  Results from every family are folded into one
JSON report plus one Markdown page.

Adding a new probe
------------------
Adding a new probe family requires ONLY:

1. a sibling YAML under ``rubrics/probes/`` (same shape as Stream 2's rubric
   YAMLs, plus a ``module: decider.probes.<family>`` key), and
2. a matching module under ``decider/probes/`` exposing ``main()`` that will
   accept a positional ``model`` and a ``--n``/``--out`` pair.

No change to the runner, the registry, or the report renderer.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from decider.rubric import RubricRun, aggregate_rubric
from decider.rubric_io import load_rubric


def _discovery_paths(rubrics_dir: str | Path) -> list[Path]:
    root = Path(rubrics_dir)
    return sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml"))


def _family_module(path: Path) -> str:
    import re

    text = path.read_text(encoding="utf-8")
    m = re.search(r"^\s*module:\s*(\S+)\s*$", text, re.MULTILINE)
    if m:
        return m.group(1)
    return f"decider.probes.{path.stem}"


def _collect_results(family_module: str, model: str, n: int) -> tuple[dict, Path]:
    """Drive one family module to a JSON result file and read it back."""
    out = Path(f"/tmp/decider_s5_{family_module.rsplit('.', 1)[-1]}.json")
    cmd = [
        sys.executable,
        "-m",
        family_module,
        model,
        "--n",
        str(n),
        "--out",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    payload = json.loads(out.read_text(encoding="utf-8"))
    return payload, out


def _default_report_name() -> str:
    return "probes_report"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--rubrics-dir", default="rubrics/probes")
    ap.add_argument("--out", default="probes_report.json")
    a = ap.parse_args()

    rubrics_dir = a.rubrics_dir
    per_family_scores: dict[str, float] = {}
    passed_flags: dict[str, bool] = {}
    failures: dict[str, list[str]] = {}

    for path in _discovery_paths(rubrics_dir):
        stem = path.stem
        rubric = load_rubric(path)
        family_module = _family_module(path)

        payload, tmp = _collect_results(family_module, a.model, a.n)
        results = payload.get("results", payload)
        if tmp.suffix == ".pkl":
            run = RubricRun.from_results(results, tmp)
        else:
            run = RubricRun(results=results, dump={})
        tmp.unlink(missing_ok=True)

        report = aggregate_rubric(rubric, run)  # type: ignore[arg-type]
        per_family_scores[stem] = report.weighted_score
        passed_flags[stem] = report.passed
        failures[stem] = [d.failure_message for d in report.outcomes if not d.passed]

    report_body = _render(
        a.model, a.n, per_family_scores, passed_flags, failures
    )

    out = Path(a.out)
    if out.suffix.lower() in {".md", ".markdown"}:
        out.write_text(report_body, encoding="utf-8")
    else:
        out.write_text(
            json.dumps(
                {
                    "model": a.model,
                    "n": a.n,
                    "passed": all(passed_flags.values()) if passed_flags else True,
                    "families": [
                        {
                            "name": name,
                            "score": per_family_scores[name],
                            "passed": passed_flags[name],
                            "failures": failures[name],
                        }
                        for name in per_family_scores
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"wrote {out}")


def _render(
    model: str,
    n: int,
    scores: dict[str, float],
    flags: dict[str, bool],
    failures: dict[str, list[str]],
) -> str:
    lines = [
        "# Probes report",
        "",
        f"- model: `{model}`",
        f"- n: `{n}`",
        f"- families scored: `{len(scores)}`",
        "",
        "| family | score | passed |",
        "| --- | --- | --- |",
    ]
    for name, score in scores.items():
        lines.append(f"| {name} | {score:.3f} | {'yes' if flags[name] else 'no'} |")
    lines.append("")
    for name, msgs in failures.items():
        if not msgs:
            continue
        lines.append(f"## {name}")
        lines.append("")
        lines.extend(f"- {m.strip()}" for m in msgs)
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
