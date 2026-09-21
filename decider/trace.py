"""JSONL traces for serve-time requests and offline replay.

`TraceWriter` opens a JSONL file in append mode and writes one JSON object per line.
The handle is kept open for the lifetime of the writer so partial crashes keep everything
written so far. `flush()` (and `close()`) fsync the file when durability matters; the
appends themselves are atomic per-line on POSIX because each `write` is under PIPE_BUF
for short lines, and the JSONL parser tolerates a truncated last line.

    from decider.trace import TraceWriter, TraceReader
    w = TraceWriter("/var/log/decider/trace.jsonl")        # honours DECIDER_TRACE in serve.py
    w.log(request_id="r-1", model="decider-2b", layout="state_first",
          state_hash="sha1:...", question_ids=["team", "urgent"],
          n_opts=[3, 2], prompt_token_count=180,
          output=[{"type": "choice", "choice": "billing", "confidence": 0.77,
                   "probabilities": {"billing": 0.77, "technical": 0.19, "sales": 0.04}},
                  {"type": "noul", "noul": 0.95, "confidence": 0.95,
                   "probabilities": {"no": 0.05, "yes": 0.95}}])
    for rec in TraceReader("/var/log/decider/trace.jsonl"): ...
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Self

SCHEMA_VERSION = 1
REQUIRED_TOP_LEVEL = ("ts", "request_id", "model", "layout", "state_hash",
                      "question_ids", "n_opts", "prompt_token_count", "output")


class TraceWriter:
    """Append-only JSONL writer; one JSON object per line.

    The underlying file is opened with mode ``"a"`` (UTF-8) and held open for the life
    of the writer. Every record is written as one ``json.dumps(...) + "\\n"`` so a
    reader that starts mid-file sees well-formed lines up to the truncation point.
    A ``threading.Lock`` serialises writes when many FastAPI handlers share a writer.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._closed = False

    def write(self, record: dict[str, Any]) -> None:
        """Append a record verbatim; ``ts`` is added if missing."""
        if self._closed:
            raise ValueError(f"TraceWriter({self.path}) is closed")
        rec = dict(record)
        rec.setdefault("ts", time.time())
        line = json.dumps(rec, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._fh.write(line + "\n")

    def log(self, *, request_id: str | None = None, model: str, layout: str,
            state_hash: str, question_ids: list[str], n_opts: list[int],
            prompt_token_count: int, output: list[dict[str, Any]],
            ts: float | None = None, **extra: Any) -> str:
        """Convenience wrapper: builds the canonical record and returns the request_id."""
        rid = request_id or uuid.uuid4().hex
        record: dict[str, Any] = {
            "ts": ts if ts is not None else time.time(),
            "request_id": rid,
            "model": model,
            "layout": layout,
            "state_hash": state_hash,
            "question_ids": list(question_ids),
            "n_opts": list(n_opts),
            "prompt_token_count": int(prompt_token_count),
            "output": list(output),
            "schema_version": SCHEMA_VERSION,
        }
        record.update(extra)
        self.write(record)
        return rid

    def flush(self) -> None:
        if self._closed:
            return
        with self._lock:
            self._fh.flush()
            try:
                os.fsync(self._fh.fileno())
            except OSError:
                pass

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._fh.flush()
                os.fsync(self._fh.fileno())
            except OSError:
                pass
            self._fh.close()
            self._closed = True

    def __del__(self) -> None:                # best effort; do not raise from __del__
        try:
            if not self._closed:
                self.close()
        except Exception:  # noqa: S110,BLE001 - __del__ must never raise
            pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class TraceReader:
    """Streaming reader for JSONL trace files.

    Skips blank lines and tolerates a truncated trailing line (a crash mid-``write``
    leaves the file with a partial last record; downstream tools treat it as missing
    rather than raising). Yields ``dict`` records; callers should treat unknown
    fields as forward-compatible additions.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return self.records()

    def records(self) -> Iterator[dict[str, Any]]:
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue                # partial trailing line after a crash

    def to_list(self) -> list[dict[str, Any]]:
        return list(self.records())

    def validate(self) -> list[str]:
        """Return a list of human-readable problems with the file (empty == OK)."""
        problems: list[str] = []
        for i, rec in enumerate(self.records()):
            for key in REQUIRED_TOP_LEVEL:
                if key not in rec:
                    problems.append(f"line {i + 1}: missing {key!r}")
            if "output" in rec and not isinstance(rec["output"], list):
                problems.append(f"line {i + 1}: output must be a list")
            if ("question_ids" in rec and "n_opts" in rec
                    and len(rec["question_ids"]) != len(rec["n_opts"])):
                problems.append(f"line {i + 1}: question_ids / n_opts length mismatch")
        return problems


def hash_state(state: Any) -> str:
    """Stable, short content hash of a render_state output. Used to correlate two traces
    recorded against different model weights; matching hashes mean the prompt is byte-for-byte
    identical (after ``render_state`` + ``render_question``)."""
    if isinstance(state, (dict, list)):
        blob = json.dumps(state, sort_keys=True, ensure_ascii=False)
    else:
        blob = str(state)
    return "sha1:" + hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def make_output_records(answers: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert ``system_one`` / ``decide_json`` style answers into the canonical
    per-question output list. Each entry has a ``type`` and either ``choice`` /
    ``score`` / ``noul`` plus ``confidence`` and ``probabilities``."""
    out: list[dict[str, Any]] = []
    for ans in answers.values():
        rec = {"type": ans.get("type", "choice"),
               "probabilities": dict(ans.get("probabilities", {}))}
        if "choice" in ans:
            rec["choice"] = ans["choice"]
        if "score" in ans:
            rec["score"] = ans["score"]
        if "noul" in ans:
            rec["noul"] = ans["noul"]
        probs = rec["probabilities"]
        if probs:
            conf = max(probs.values())
        else:
            conf = float(ans.get("confidence", ans.get("noul", 0.0)))
        rec["confidence"] = float(ans.get("confidence", conf))
        out.append(rec)
    return out


def records_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Two records describe the same question set if their state_hash, layout and question_ids agree."""
    return (a.get("state_hash") == b.get("state_hash")
            and a.get("layout") == b.get("layout")
            and list(a.get("question_ids", [])) == list(b.get("question_ids", [])))


def group_by_state(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group records by ``state_hash`` (and ``layout``); replay pairs entries that share a key."""
    out: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        key = f"{rec.get('layout', '?')}::{rec.get('state_hash', '?')}"
        out.setdefault(key, []).append(rec)
    return out


__all__ = [
    "SCHEMA_VERSION",
    "TraceReader",
    "TraceWriter",
    "group_by_state",
    "hash_state",
    "make_output_records",
    "records_match",
]