# Changelog

Newest first. The full per-release notes and figures live in
[docs/CHANGELOG.md](docs/CHANGELOG.md); the entries below are the brief form
that ships in the repository root.

## Unreleased

### Added — MCP server (stream/mcp)

New Model Context Protocol server exposing decider as typed tools. `decider.mcp_server.mcp_serve(model=...)` builds a
configured server (lazily loading the Decider on first tool call) with two tools — `system_one(state, questions)`,
the Jev/TypeSafe wire format from `decider.systemone`, and `decide(context, questions)` — whose schemas are generated
from `decider.systemone.render_question` (single source of truth). The server runs over stdio via `uv run decider mcp`
(new `[project.scripts] decider` entry point) and ships with an in-memory MCP-client test suite plus the `[mcp]`
optional-dependency group.

### Added — traces and replay (stream/trace)

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

> > > > > > > stream/trace
