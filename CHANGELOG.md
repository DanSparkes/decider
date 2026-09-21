# Changelog

Newest first. The full per-release notes and figures live in
[docs/CHANGELOG.md](docs/CHANGELOG.md); the entries below are the brief form
that ships in the repository root.

## Unreleased

- JSONL request traces: `decider/trace.py` exposes `TraceWriter` / `TraceReader`
  (append-only, one JSON object per line, partial-crash tolerant). The server
  mirrors every `/decide` and `/v1/systemone` request when `DECIDER_TRACE` is
  set; `DECIDER_TRACE_STATES=1` also writes a `trace.states.jsonl` sidecar so
  the trace can be rescored against another weight file later. The record shape
  is documented in the README (Traces section).
- Offline replay + drift report: `decider/replay.py` is a CLI
  (`python -m decider.replay --trace trace.jsonl --old old.json --new new.json`)
  that emits per-question KL and JS divergence of the served probability
  distributions, the count of top-1 flips, and ΔECE. Replay-on-itself is zero by
  construction and is enforced by `tests/test_trace.py`.
- `examples/trace_replay.py` shows the full loop: record against one model,
  rescore against another, print the drift report.
