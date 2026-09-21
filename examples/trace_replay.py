"""Trace replay: record requests against a model, then replay them against a
second model and emit the per-question drift.

Run it end-to-end (requires two decider-2b weight folders or Hub ids):

    python examples/trace_replay.py model.json model_v10.json

The example is self-contained: it generates a small synthetic traffic, writes
both a ``trace.jsonl`` and a sidecar ``trace.states.jsonl``, then runs
``decider.replay.replay`` against the two weight files and prints the report.

If you don't have weights, ``decider.replay`` is also a CLI:

    python -m decider.replay --trace trace.jsonl --old old.json --new new.json

The trace fixture in ``tests/fixtures/example_trace.jsonl`` plus the round-trip
test in ``tests/test_trace.py`` enforce the replay-on-itself contract without
needing a GPU.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import uuid
from pathlib import Path
from typing import Any

from decider.replay import _print_report, replay
from decider.trace import TraceReader, TraceWriter, hash_state, make_output_records

SYNTHETIC_TRAFFIC: list[tuple[str, dict[str, dict[str, Any]]]] = [
    ("My card was charged twice for the same purchase and I want the extra charge refunded.",
     {"team": {"type": "choice", "instructions": "Which team should handle this?",
               "criteria": {"billing": "Charges, invoices",
                            "technical": "Bugs, outages",
                            "returns": "Refunds, exchanges"}},
      "urgent": {"type": "noul", "instructions": "Is the message urgent?"}}),
    ("hey can u turn the lights off in the kitchen",
     {"intent": {"type": "choice", "instructions": "What is the intent?",
                 "criteria": {"smart_home": "Lights, thermostat, locks",
                              "music": "Play, queue, skip",
                              "alarm": "Set, snooze, cancel",
                              "other": "Anything else"}},
      "toxic": {"type": "noul", "instructions": "Is the message toxic?"}}),
    ("The quarterly report shows revenue fell 12% while costs rose sharply.",
     {"sentiment": {"type": "choice", "instructions": "What is the financial sentiment?",
                    "criteria": {"bearish": "Negative outlook",
                                 "neutral": "Reporting only",
                                 "bullish": "Positive outlook"}}}),
]


def _record(d, out_path: Path) -> int:
    """Run synthetic traffic against ``d`` and append to ``out_path``. Returns the
    number of records written. A sidecar ``<out_path>.states.jsonl`` is written so
    ``decider.replay`` can rescore the trace against another model later."""
    w = TraceWriter(out_path)
    side = out_path.with_suffix(".states.jsonl")
    side_fh = side.open("a", encoding="utf-8")
    n = 0
    try:
        for ctx, questions in SYNTHETIC_TRAFFIC:
            rid = uuid.uuid4().hex
            res = d.system_one(ctx, questions)
            answers = res["answers"]
            prompt_token_count = len(ctx.split()) + sum(len(q["instructions"].split()) for q in questions.values())
            w.log(request_id=rid, model=d.name, layout="state_first",
                  state_hash=hash_state(ctx),
                  question_ids=list(questions),
                  n_opts=[len(q["criteria"]) for q in questions.values()],
                  prompt_token_count=prompt_token_count,
                  output=make_output_records(answers))
            side_fh.write(json.dumps({"request_id": rid, "state": ctx, "questions": questions},
                                     ensure_ascii=False) + "\n")
            n += 1
    finally:
        w.close()
        side_fh.close()
    return n


def _load_records_with_states(trace_path: Path) -> list[dict]:
    recs = TraceReader(trace_path).to_list()
    side = trace_path.with_suffix(".states.jsonl")
    if not side.exists():
        return recs
    by_id: dict[str, dict] = {}
    for ln in side.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        obj = json.loads(ln)
        by_id[obj["request_id"]] = obj
    out = []
    for r in recs:
        r2 = dict(r)
        st = by_id.get(r["request_id"])
        if st:
            r2["state"] = st["state"]
            r2["questions"] = st["questions"]
        out.append(r2)
    return out


def _replay(trace_path: Path, old_model: str, new_model: str) -> dict:
    from decider.infer import Decider
    old = Decider(old_model)
    new = Decider(new_model)

    def _score(d, rec):
        state, questions = rec["state"], rec["questions"]
        out = d.system_one(state, questions)["answers"]
        return [out[qid] for qid in questions]

    return replay(_load_records_with_states(trace_path),
                  lambda r: _score(old, r),
                  lambda r: _score(new, r),
                  include_unmatched=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("old", help="old model directory (Hugging Face id or local path)")
    ap.add_argument("new", help="new model directory")
    ap.add_argument("--trace", help="trace file to write / replay (default: a temp file)")
    ap.add_argument("--top", type=int, default=20, help="show only the N highest-KL questions (0 = all)")
    a = ap.parse_args(argv)
    top = a.top or None

    if a.trace:
        trace_path = Path(a.trace)
    else:
        tmp = tempfile.mkdtemp(prefix="decider-trace-")
        trace_path = Path(tmp) / "trace.jsonl"

    print(f"[trace_replay] recording synthetic traffic to {trace_path}")
    from decider.infer import Decider
    n = _record(Decider(a.old), trace_path)
    print(f"[trace_replay] wrote {n} records")

    print(f"[trace_replay] replaying {trace_path} against {a.old} and {a.new}")
    report = _replay(trace_path, a.old, a.new)
    _print_report(report, top=top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())