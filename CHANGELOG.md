# Changelog

## Unreleased

### Added — MCP server (stream/mcp)

New Model Context Protocol server exposing decider as typed tools. `decider.mcp_server.mcp_serve(model=...)` builds a
configured server (lazily loading the Decider on first tool call) with two tools — `system_one(state, questions)`,
the Jev/TypeSafe wire format from `decider.systemone`, and `decide(context, questions)` — whose schemas are generated
from `decider.systemone.render_question` (single source of truth). The server runs over stdio via `uv run decider mcp`
(new `[project.scripts] decider` entry point) and ships with an in-memory MCP-client test suite plus the `[mcp]`
optional-dependency group.
